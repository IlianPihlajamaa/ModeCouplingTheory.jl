# todo: doesnt work yet
# try
# using Revise
# import ModeCouplingTheory
# equation = ModeCouplingTheory.BetaScalingEquation(0.7, 0.0, 0.0, 1.0)
# sol = ModeCouplingTheory.solve(equation)
#
# then: update a,b with proper values!
mutable struct BetaScalingEquationCoefficients{T}
    λ::T
    σ::T
    δ::T
    t₀::T
    δ_times_t::T
    a::T
    b::T
end

function BetaScalingEquationCoefficients(λ, σ, δ, t₀)
    a = 0.3
    b = 0.0
    return BetaScalingEquationCoefficients(λ, σ, δ, t₀, Float64(0.0), a, b)
end

struct BetaScalingEquation{T,A,B,C,D} <: AbstractMemoryEquation
    coeffs::T
    F₀::A
    K₀::B
    kernel::C
    update_coefficients!::D
end

function BetaScalingEquation(λ::Float64, σ::Float64, δ::Float64, t₀::Float64)
    BetaScalingEquation(BetaScalingEquationCoefficients(λ, σ, δ, t₀), Float64(0.0), nothing, nothing, (coeffs,t) -> coeffs.δ_times_t = coeffs.δ*t)
end

mutable struct FuchsNoMemTempStruct{T,T3,VT}
    F_temp::VT
    F_I::VT
    C1::T3
    C1_temp::T3
    C2::T
    C3::T
    F_old::T
    inplace::Bool
    start_time::Float64
end

function allocate_temporary_arrays(equation::BetaScalingEquation, solver::TimeDoublingSolver)
    start_time = time()
    F_temp = Float64[]
    F_I = Float64[]
    F_old = 0.0
    temp_arrays = FuchsNoMemTempStruct(F_temp, F_I, 1.0, 1.0, 1.0, 1.0, F_old, true, start_time)
    for _ in 1:4*solver.N
        push!(temp_arrays.F_temp, 1.)
        push!(temp_arrays.F_I, 1.)
    end
    return temp_arrays
end

function initialize_F_temp!(equation::BetaScalingEquation, solver::TimeDoublingSolver, temp_arrays::FuchsNoMemTempStruct)
    N = solver.N
    δt = solver.Δt / (4 * N)
    t₀ = equation.coeffs.t₀
    a = equation.coeffs.a

    for it = 1:2N
        temp_arrays.F_temp[it] = (δt*it/t₀)^-a
    end
end

function initialize_integrals!(equation::BetaScalingEquation, solver::TimeDoublingSolver, temp_arrays::FuchsNoMemTempStruct)
    F_I = temp_arrays.F_I
    F_temp = temp_arrays.F_temp
    N = solver.N
    δt = solver.Δt / (4 * N)
    t₀ = equation.coeffs.t₀
    a = equation.coeffs.a

    F_I[1] = (t₀/δt)^a / (1-a)
    for it = 2:2N
        F_I[it] = F_I[1] * (it^(1-a) - (it-1)^(1-a))
    end
end

function initialize_temporary_arrays!(equation::BetaScalingEquation, solver::TimeDoublingSolver, kernel, temp_arrays::FuchsNoMemTempStruct)
    initialize_F_temp!(equation, solver, temp_arrays)
    initialize_integrals!(equation, solver, temp_arrays)
end


function update_Fuchs_parameters!(equation::BetaScalingEquation, solver::TimeDoublingSolver, temp_arrays::FuchsNoMemTempStruct, it::Int)
    N = solver.N
    i2 = 2N
    δt = solver.Δt / (4N)
    F_I = temp_arrays.F_I
    F = temp_arrays.F_temp
    equation.update_coefficients!(equation.coeffs, δt*it)
    λ = equation.coeffs.λ
    σ = equation.coeffs.σ
    δ_times_t = equation.coeffs.δ_times_t

    temp_arrays.C1 = 2*F_I[1]
    temp_arrays.C2 = λ

    c3 = -F[it-i2]*F[i2] + 2*F[it-1]*F_I[1]
    c3 += σ - δ_times_t 
    for j = 2:i2
        c3 += 2 * (F[it-j] - F[it-j+1]) * F_I[j]
    end
    if it-i2 != i2
        c3 += (F[i2] - F[i2+1]) * F_I[it-i2]
    end
    temp_arrays.C3 = c3
end

function update_F!(::TimeDoublingSolver, temp_arrays::FuchsNoMemTempStruct, it::Int)
    c1 = temp_arrays.C1
    c1_temp = temp_arrays.C1_temp
    c2 = temp_arrays.C2
    c3 = temp_arrays.C3
    F_old = temp_arrays.F_temp[it]
    #temp_arrays.F_temp[it] = c1/(2c2)-sqrt((c1/(2c2))^2-c3/c2)
    temp_arrays.F_temp[it] = c1 \ (F_old*F_old*c2 + c3)
end

function update_K_and_F!(solver::TimeDoublingSolver, kernel, temp_arrays::FuchsNoMemTempStruct, it::Int)
    update_F!(solver, temp_arrays, it)
end

function update_integrals!(temp_arrays::FuchsNoMemTempStruct, it::Int)
    F_I = temp_arrays.F_I
    F_temp = temp_arrays.F_temp
    F_I[it] = (F_temp[it] + F_temp[it-1]) / 2
end

function do_time_steps!(equation::BetaScalingEquation, solver::TimeDoublingSolver, kernel, temp_arrays::FuchsNoMemTempStruct)
    N = solver.N
    F_temp = temp_arrays.F_temp
    tolerance = solver.tolerance
    for it = (2N+1):(4N)
        error = typemax(Float64)
        iterations = 1
        F_old = temp_arrays.F_old

        update_Fuchs_parameters!(equation, solver, temp_arrays, it)
        update_F!(solver, temp_arrays, it)

        while error > tolerance
            iterations += 1
            if iterations > solver.max_iterations
                throw(DomainError("Iteration did not converge. Either increase the number of time steps before a time doubling, or choose a different memory kernel."))
            end
            update_K_and_F!(solver, kernel, temp_arrays, it)
            error = find_error(F_temp[it], F_old)
            F_old = F_temp[it]
        end
        update_integrals!(temp_arrays, it)
        solver.kernel_evals += iterations - 1
    end
    return
end

function allocate_results!(t_array, F_array, K_array, solver::TimeDoublingSolver, temp_arrays::FuchsNoMemTempStruct; istart=2(solver.N) + 1, iend=4(solver.N))
    N = solver.N
    δt = solver.Δt / (4N)
    for it = istart:iend
        t = δt * it
        push!(t_array, t)
        push!(F_array, deepcopy(temp_arrays.F_temp[it]))
    end
end

function new_time_mapping!(equation::AbstractMemoryEquation, solver::TimeDoublingSolver, temp_arrays::FuchsNoMemTempStruct)
    F = temp_arrays.F_temp
    F_I = temp_arrays.F_I
    N = solver.N
    for j = 1:N
        F_I[j] = (F_I[2j] + F_I[2j-1]) / 2
        F[j] = F[2j]
    end
    for j = (N+1):2*N
        #F_I[j] = (F_I[2j] + 4 * F_I[2j-1] + F_I[2j-2]) / 6
        F_I[j] = (F_I[2j] + F_I[2j-1]) / 2
        F[j] = F[2j]
    end
    for j = 2N+1:4N
        F_I[j] = equation.F₀ * zero(eltype(eltype(F_I)))
        F[j] = equation.F₀ * zero(eltype(eltype(F)))
    end
    solver.Δt *= 2
end
