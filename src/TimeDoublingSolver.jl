abstract type AbstractFuchsTempStruct end

mutable struct FuchsTempStruct{T,T2,T3,VT,VT2,SC} <: AbstractFuchsTempStruct
    F_temp::VT
    K_temp::VT2
    F_I::VT
    K_I::VT2
    C1::T3
    C1_temp::T3
    C2::T
    C3::T
    temp_vec::T
    F_old::T
    temp_mat::T2
    solve_cache::SC
    inplace::Bool
    start_time::Float64
end

mutable struct TimeDoublingSolver{I,F} <: Solver
    N::I
    Δt::F
    t_max::F
    kernel_evals::I
    max_iterations::I
    tolerance::F
    verbose::Bool
    inplace::Bool
end

"""
    TimeDoublingSolver(N=32, Δt=10^-10, t_max=10.0^10, max_iterations=10^4, tolerance=10^-10, verbose=false, ismutable=true)

Uses the algorithm devised by Fuchs et al.

# Arguments:
* `equation`: an instance of MemoryEquation
* `N`: The number of time points in the interval is equal to `4N`
* `t_max`: when this time value is reached, the integration returns
* `Δt`: starting time step, this will be doubled repeatedly
* `max_iterations`: the maximal number of iterations before convergence is reached for each time doubling step
* `tolerance`: while the error is bigger than this value, convergence is not reached. The error by default is computed as the absolute sum of squares
* `verbose`: if `true`, information will be printed to STDOUT
* `ismutable`: if `true` and if the type of F is mutable, the solver will try to avoid allocating many temporaries
"""
function TimeDoublingSolver(; N=32, Δt=10^-10, t_max=10.0^10, max_iterations=10^4, tolerance=10^-10, verbose=false, ismutable=true)
    return TimeDoublingSolver(N, Δt, t_max, 0, max_iterations, tolerance, verbose, ismutable)
end

"""
    allocate_temporary_arrays(equation::AbstractMemoryEquation, solver::TimeDoublingSolver)

Returns a FuchsTempStruct containing several arrays that are used for intermediate calculations.
"""
function allocate_temporary_arrays(equation::MemoryEquation, solver::TimeDoublingSolver)
    K₀ = equation.K₀
    F₀ = equation.F₀
    C1 = sum([equation.coeffs.α, equation.coeffs.β, equation.coeffs.γ, K₀])
    C2 = K₀ * F₀
    C3 = K₀ * F₀
    temp_vec = K₀ * F₀
    F_old = K₀ * F₀
    temp_mat = sum([equation.coeffs.α, equation.coeffs.β, equation.coeffs.γ, K₀])
    Fmutable = ismutabletype(typeof(F₀))
    inplace = Fmutable & solver.inplace
    start_time = time()
    F_temp = typeof(F₀)[]
    K_temp = typeof(K₀)[]
    F_I = typeof(F₀)[]
    K_I = typeof(K₀)[]
    if inplace && !check_if_diag(temp_mat)
        prob = LinearSolve.LinearProblem(temp_mat, temp_vec)
        cache1 = LinearSolve.init(prob)
        sol = LinearSolve.solve(cache1)
        temp_arrays = FuchsTempStruct(F_temp, K_temp, F_I, K_I, C1, C1 + C1, C2, C3, temp_vec, F_old, temp_mat, sol.cache, inplace, start_time)
    else
        temp_arrays = FuchsTempStruct(F_temp, K_temp, F_I, K_I, C1, C1 + C1, C2, C3, temp_vec, F_old, temp_mat, false, inplace, start_time)
    end
    for _ in 1:4*solver.N
        push!(temp_arrays.F_temp, K₀ * F₀)
        push!(temp_arrays.K_temp, K₀ + K₀)
        push!(temp_arrays.F_I, K₀ * F₀)
        push!(temp_arrays.K_I, K₀ + K₀)
    end
    return temp_arrays
end


"""
    initialize_F_temp!(equation::MemoryEquation, solver::TimeDoublingSolver, temp_arrays::FuchsTempStruct)

Fills the first 2N entries of the temporary arrays of F using forward Euler without a memory kernel in order to kickstart Fuchs' scheme.

"""
function initialize_F_temp!(equation::MemoryEquation, solver::TimeDoublingSolver, temp_arrays::FuchsTempStruct)
    N = solver.N
    δt = solver.Δt / (4 * N)

    F₀ = equation.F₀
    ∂ₜF₀ = equation.∂ₜF₀

    ∂ₜF_old = ∂ₜF₀
    F_old = F₀
    for it = 1:2N
        equation.update_coefficients!(equation.coeffs, δt*it)
        α = equation.coeffs.α
        β = equation.coeffs.β
        γ = equation.coeffs.γ
        δ = equation.coeffs.δ
        second_order = !iszero(α)
        if second_order
            ∂ₜₜF = -α \ (β * ∂ₜF_old + γ * F_old + δ)
            ∂ₜF = ∂ₜF_old + δt * ∂ₜₜF
            F = F_old + δt * ∂ₜF
        else
            ∂ₜF = -β \ (γ * F_old + δ)
            F = F_old + δt * ∂ₜF
        end
        temp_arrays.F_temp[it] = F
        ∂ₜF_old = ∂ₜF
        F_old = F
    end
end

"""
    initialize_K_temp!(solver::TimeDoublingSolver, kernel::MemoryKernel, temp_arrays::FuchsTempStruct)

Evaluates the memory kernel at the first 2N time points.
"""
function initialize_K_temp!(solver::TimeDoublingSolver, kernel::MemoryKernel, temp_arrays::FuchsTempStruct)
    N = solver.N
    δt = solver.Δt / (4 * N)
    for it = 1:2N
        t = it * δt
        if !temp_arrays.inplace
            temp_arrays.K_temp[it] = evaluate_kernel(kernel, temp_arrays.F_temp[it], t)
        else
            evaluate_kernel!(temp_arrays.K_temp[it], kernel, temp_arrays.F_temp[it], t)
        end
    end
end


"""
    initialize_integrals!(equation::AbstractMemoryEquation, solver::TimeDoublingSolver, temp_arrays::FuchsTempStruct)

Initializes the integrals on the first 2N time points as prescribed in the literature.
"""
function initialize_integrals!(equation::AbstractMemoryEquation, solver::TimeDoublingSolver, temp_arrays::FuchsTempStruct)
    F_I = temp_arrays.F_I
    K_I = temp_arrays.K_I
    F_temp = temp_arrays.F_temp
    K_temp = temp_arrays.K_temp
    N = solver.N

    if !temp_arrays.inplace
        # it = 1
        F_I[1] = (F_temp[1] + equation.F₀) / 2
        K_I[1] = (3 * K_temp[1] - K_temp[2]) / 2
        for it = 2:2N
            F_I[it] = (F_temp[it] + F_temp[it-1]) / 2
            K_I[it] = (K_temp[it] + K_temp[it-1]) / 2
        end
    else
        @. F_I[1] = (F_temp[1] + equation.F₀) / 2
        for it = 2:2N
            @. F_I[it] = (F_temp[it] + F_temp[it-1]) / 2
        end
        @. K_I[1] = (3 * K_temp[1] - K_temp[2]) / 2
        for it = 2:2N
            @. K_I[it] = (K_temp[it] + K_temp[it-1]) / 2
        end
    end
end


function initialize_temporary_arrays!(equation::AbstractMemoryEquation, solver::TimeDoublingSolver, kernel::MemoryKernel, temp_arrays::FuchsTempStruct)
    initialize_F_temp!(equation, solver, temp_arrays)
    initialize_K_temp!(solver, kernel, temp_arrays)
    initialize_integrals!(equation, solver, temp_arrays)
end


"""
    update_Fuchs_parameters!(equation::MemoryEquation, solver::TimeDoublingSolver, temp_arrays::FuchsTempStruct, it::Int) 

Updates the parameters c1, c2, c3, according to the appendix of 
"Flenner, Elijah, and Grzegorz Szamel. Physical Review E 72.3 (2005): 031508"
using the naming conventions from that paper. If F is mutable (and therefore also c1,c2,c3), it will
update the variables in place, otherwise it will create new copies. This is controlled by the solver.inplace 
setting.
"""
function update_Fuchs_parameters!(equation::MemoryEquation, solver::TimeDoublingSolver, temp_arrays::FuchsTempStruct, it::Int)
    N = solver.N
    i2 = 2N
    δt = solver.Δt / (4N)
    K_I = temp_arrays.K_I
    F_I = temp_arrays.F_I
    F = temp_arrays.F_temp
    kernel = temp_arrays.K_temp
    equation.update_coefficients!(equation.coeffs, δt*it)
    α = equation.coeffs.α
    β = equation.coeffs.β
    γ = equation.coeffs.γ
    δ = equation.coeffs.δ
    if !temp_arrays.inplace # everything immutable (we are free to allocate)
        c1 = (2 / (δt^2) * α + 3 / (2δt) * β) + K_I[1] + γ

        c2 = F_I[1] - equation.F₀

        c3 = α * (5 * F[it-1] - 4 * F[it-2] + F[it-3]) / δt^2
        c3 += β * (2 / δt * F[it-1] - F[it-2] / (2δt))
        c3 -= δ
        c3 += -kernel[it-i2] * F[i2] + kernel[it-1] * F_I[1] + K_I[1] * F[it-1]
        @inbounds for j = 2:i2
            c3 += (kernel[it-j] - kernel[it-j+1]) * F_I[j]
        end
        @inbounds for j = 2:it-i2
            c3 += K_I[j] * (F[it-j] - F[it-j+1])
        end
        temp_arrays.C1 = c1
        temp_arrays.C2 = c2
        temp_arrays.C3 = c3
    else # perform everything without allocations. The commented code is the corresponding scalar equivalent
        temp_arrays.C1 .= (2 / (δt^2) * α + 3 / (2δt) * β) + K_I[1] + γ
        temp_arrays.C2 .= F_I[1] - equation.F₀
        temp_vec = temp_arrays.temp_vec
        temp_mat = temp_arrays.temp_mat

        c3 = temp_arrays.C3
        # c3 .= α*(5*F[it-1] - 4*F[it-2] + F[it-3])/δt^2
        @. temp_vec = (5 * F[it-1] - 4 * F[it-2] + F[it-3]) / δt^2
        mymul!(c3, α, temp_vec, true, false)

        # c3 .+= β*(2/δt*F[it-1] - F[it-2]/(2δt))
        @. temp_vec = 2 / δt * F[it-1] - F[it-2] / (2δt)
        mymul!(c3, β, temp_vec, true, true)

        # c3 .+= -kernel[it-i2]*F[i2] + kernel[it-1]*F_I[1] + K_I[1]*F[it-1]
        mymul!(c3, kernel[it-i2], F[i2], -true, true)
        mymul!(c3, kernel[it-1], F_I[1], true, true)
        mymul!(c3, K_I[1], F[it-1], true, true)

        for j = 2:i2
            # c3 .+= (kernel[it-j] - kernel[it-j+1])*F_I[j]
            if check_if_diag(temp_mat)
                @. temp_mat.diag = kernel[it-j].diag - kernel[it-j+1].diag
            else
                @. temp_mat = kernel[it-j] - kernel[it-j+1]
            end
            mymul!(c3, temp_mat, F_I[j], true, true)

        end
        for j = 2:it-i2
            # c3 .+= K_I[j]*(F[it-j] - F[it-j+1])
            @. temp_vec = F[it-j] - F[it-j+1]
            mymul!(c3, K_I[j], temp_vec, true, true)
        end
        c3 .-= δ
    end
    return nothing
end

"""
    update_F!(solver::TimeDoublingSolver, temp_arrays::FuchsTempStruct, it::Int) 

updates F using the formula c1*F = -K*C2 + C3.
"""
function update_F!(::MemoryEquation, ::TimeDoublingSolver, temp_arrays::FuchsTempStruct, it::Int)
    c1 = temp_arrays.C1
    c2 = temp_arrays.C2
    c3 = temp_arrays.C3
    if !temp_arrays.inplace
        temp_arrays.F_temp[it] = c1 \ (-temp_arrays.K_temp[it] * c2 + c3)
    else # do the operation above without allocations
        mymul!(temp_arrays.temp_vec, temp_arrays.K_temp[it], c2, true, false)
        @. temp_arrays.temp_vec = -temp_arrays.temp_vec + c3
        if check_if_diag(c1)
            temp_arrays.F_temp[it] .= c1.diag .\ temp_arrays.temp_vec
        else
            c1_temp = temp_arrays.C1_temp
            c1_temp .= c1
            cache = LinearSolve.set_b(temp_arrays.solve_cache, temp_arrays.temp_vec)
            cache = LinearSolve.set_A(cache, c1_temp)
            sol = LinearSolve.solve(cache)
            temp_arrays.F_temp[it] .= sol.u
            # ldiv!(temp_arrays.F_temp[it], qr!(c1_temp, ColumnNorm()), temp_arrays.temp_vec)
        end
    end
end

function update_K_and_F!(equation::AbstractMemoryEquation, solver::TimeDoublingSolver, kernel::MemoryKernel, temp_arrays, it::Int)
    update_K!(solver, kernel, temp_arrays, it)
    update_F!(equation, solver, temp_arrays, it)
end

"""
    update_K!(solver::TimeDoublingSolver, kernel::MemoryKernel, temp_arrays::FuchsTempStruct, it::Int) 

evaluates the memory kernel, updating the value in solver.temp_arrays.K_temp    
"""
function update_K!(solver::TimeDoublingSolver, kernel::MemoryKernel, temp_arrays::FuchsTempStruct, it::Int)
    N = solver.N
    δt = solver.Δt / (4N)
    t = δt * it
    if !temp_arrays.inplace
        temp_arrays.K_temp[it] = evaluate_kernel(kernel, temp_arrays.F_temp[it], t)
    else
        evaluate_kernel!(temp_arrays.K_temp[it], kernel, temp_arrays.F_temp[it], t)
    end
end


"""
    update_integrals!(solver::TimeDoublingSolver, equation::AbstractMemoryEquation, temp_arrays::FuchsTempStruct, it::Int)

Update the discretisation of the integral of F and K, see the literature for details.
"""
function update_integrals!(temp_arrays::FuchsTempStruct, ::AbstractMemoryEquation, it::Int)
    K_I = temp_arrays.K_I
    F_I = temp_arrays.F_I
    F_temp = temp_arrays.F_temp
    K_temp = temp_arrays.K_temp
    F_I[it] = (F_temp[it] + F_temp[it-1]) / 2
    K_I[it] = (K_temp[it] + K_temp[it-1]) / 2
end


"""
    do_time_steps!(equation::MemoryEquation, solver::TimeDoublingSolver, kernel::MemoryKernel, temp_arrays::FuchsTempStruct)

Solves the equation on the time points with index 2N+1 until 4N, for each point doing a recursive iteration
to find the solution to the nonlinear equation C1 F  = -C2 M(F) + C3.
"""
function do_time_steps!(equation::AbstractMemoryEquation, solver::TimeDoublingSolver, kernel, temp_arrays::AbstractFuchsTempStruct)
    N = solver.N
    F_temp = temp_arrays.F_temp
    tolerance = solver.tolerance
    for it = (2N+1):(4N)
        error = typemax(Float64)
        iterations = 1
        F_old = temp_arrays.F_old

        update_Fuchs_parameters!(equation, solver, temp_arrays, it)
        update_F!(equation, solver, temp_arrays, it)

        while error > tolerance
            iterations += 1
            if iterations > solver.max_iterations
                throw(DomainError("Iteration did not converge. Either increase the number of time steps before a time doubling, or choose a different memory kernel."))
            end
            update_K_and_F!(equation, solver, kernel, temp_arrays, it)
            error = find_error(F_temp[it], F_old)
            if !temp_arrays.inplace
                F_old = F_temp[it]
            else
                F_old .= F_temp[it]
            end
        end
        update_integrals!(temp_arrays, equation, it)
        solver.kernel_evals += iterations - 1
    end
    return
end


"""
    allocate_results!(t_array, F_array, K_array, equation::AbstractMemoryEquation, solver::TimeDoublingSolver, temp_arrays::FuchsTempStruct; istart=2(solver.N)+1, iend=4(solver.N))

pushes the found solution, stored in `temp_arrays` with indices `istart` until `istop` to the output arrays.
"""
function allocate_results!(t_array, F_array, K_array, ::AbstractMemoryEquation, solver::TimeDoublingSolver, temp_arrays::FuchsTempStruct; istart=2(solver.N) + 1, iend=4(solver.N))
    N = solver.N
    δt = solver.Δt / (4N)
    for it = istart:iend
        t = δt * it
        push!(t_array, t)
        push!(F_array, deepcopy(temp_arrays.F_temp[it]))
        push!(K_array, deepcopy(temp_arrays.K_temp[it]))
    end
end


"""
new_time_mapping!(equation::AbstractMemoryEquation, solver::TimeDoublingSolver, temp_arrays::FuchsTempStruct)

Performs the time mapping central to Fuchs' algorithm with the conventions prescribed in 
"Flenner, Elijah, and Grzegorz Szamel. Physical Review E 72.3 (2005): 031508". 
Performs them inplace if solver.inplace = true in order to avoid unnecessary allocations.
"""
function new_time_mapping!(equation::AbstractMemoryEquation, solver::TimeDoublingSolver, temp_arrays::FuchsTempStruct)
    F = temp_arrays.F_temp
    K = temp_arrays.K_temp
    F_I = temp_arrays.F_I
    K_I = temp_arrays.K_I
    N = solver.N
    if !temp_arrays.inplace
        for j = 1:N
            F_I[j] = (F_I[2j] + F_I[2j-1]) / 2
            K_I[j] = (K_I[2j] + K_I[2j-1]) / 2
            F[j] = F[2j]
            K[j] = K[2j]
        end
        for j = (N+1):2*N
            F_I[j] = (F_I[2j] + 4 * F_I[2j-1] + F_I[2j-2]) / 6
            K_I[j] = (K_I[2j] + 4 * K_I[2j-1] + K_I[2j-2]) / 6
            F[j] = F[2j]
            K[j] = K[2j]
        end
        for j = 2N+1:4N
            F_I[j] = equation.F₀ * zero(eltype(eltype(F_I)))
            K_I[j] = equation.K₀ * zero(eltype(eltype(K_I)))
            F[j] = equation.F₀ * zero(eltype(eltype(F)))
            K[j] = equation.K₀ * zero(eltype(eltype(K)))
        end
    else
        isdiag = check_if_diag(K_I[1])
        for j = 1:N
            @. F_I[j] = (F_I[2j] + F_I[2j-1]) / 2
            if isdiag
                @. K_I[j].diag = (K_I[2j].diag + K_I[2j-1].diag) / 2
            else
                @. K_I[j] = (K_I[2j] + K_I[2j-1]) / 2
            end
            @. F[j] = F[2j]
            @. K[j] = K[2j]
        end
        for j = (N+1):2*N
            @. F_I[j] = (F_I[2j] + 4 * F_I[2j-1] + F_I[2j-2]) / 6
            if isdiag
                @. K_I[j].diag = (K_I[2j].diag + 4 * K_I[2j-1].diag + K_I[2j-2].diag) / 6
            else
                @. K_I[j] = (K_I[2j] + 4 * K_I[2j-1] + K_I[2j-2]) / 6
            end
            @. F[j] = F[2j]
            @. K[j] = K[2j]
        end
        for j = 2N+1:4N
            Feltype = eltype(F_I[j])
            Keltype = eltype(K_I[j])
            @. F_I[j] .= zero(Feltype)
            @. K_I[j] .= zero(Keltype)
            @. F[j] .= zero(Feltype)
            @. K[j] .= zero(Keltype)
        end
    end
    solver.Δt *= 2
end

function log_results(solver, temp_arrays)
    if solver.verbose
        @printf("elapsed time = %5.2fs, t = %6.2e / %5.2e, kernel evals = %i.\n",
        time()-temp_arrays.start_time,
        solver.Δt,
        solver.t_max, 
        solver.kernel_evals)
    end
end

function convertresults(F_array::Vector{<:Number}, K_array::Vector{<:Number})
    return F_array, K_array
end

"""
    convertresults(F_array::Vector{T}, K_array::Vector{Diagonal{T, T2}}) where {T2, T}

converts the arrays of arrays into multidimensional arrays for ease of use.
"""
function convertresults(F_array::Vector{T}, K_array::Vector{Diagonal{T,T2}}) where {T2,T}
    Nt = length(F_array)
    Nk = length(F_array[1])
    F = zeros(eltype(F_array[1]), Nk, Nt)
    K = zeros(eltype(K_array[1]), Nk, Nt)
    for it in 1:Nt
        for ik = 1:Nk
            F[ik, it] = F_array[it][ik]
            K[ik, it] = K_array[it].diag[ik]
        end
    end
    return F, K
end


"""
    initialize_output_arrays(equation::AbstractMemoryEquation)

initializes arrays that the solver will push results into.
"""
function initialize_output_arrays(equation::AbstractMemoryEquation)
    return [0.0], typeof(equation.F₀)[equation.F₀], typeof(equation.K₀)[equation.K₀]
end


function solve(equation::AbstractMemoryEquation, solver::TimeDoublingSolver)
    # Documented in src/Solvers.jl
    kernel = equation.kernel
    t_array, F_array, K_array = initialize_output_arrays(equation)
    temp_arrays = allocate_temporary_arrays(equation, solver)
    initialize_temporary_arrays!(equation, solver, kernel, temp_arrays)
    allocate_results!(t_array, F_array, K_array, equation, solver, temp_arrays; istart=1, iend=2(solver.N))
    startΔt = solver.Δt
    solver.kernel_evals = 1


    # main loop of the algorithm
    while solver.Δt < solver.t_max * 2
        do_time_steps!(equation, solver, kernel, temp_arrays)
        allocate_results!(t_array, F_array, K_array, equation, solver, temp_arrays)
        log_results(solver, temp_arrays)
        new_time_mapping!(equation, solver, temp_arrays)
    end
    solver.Δt = startΔt
    sol = MemoryEquationSolution(t_array, F_array, K_array, solver)
    return sol
end


# here we have a few modifications that allow for the solver
# to be used in cases where a memory kernel is not separately
# defined, nor stored
#
# these could be moved to a separate file or integrated with
# TimeDoublingSolver.jl, because semantically they belong there

function initialize_temporary_arrays!(equation::AbstractNoKernelEquation, solver::TimeDoublingSolver, _, temp_arrays::FuchsTempStruct)
    initialize_F_temp!(equation, solver, temp_arrays)
    initialize_integrals!(equation, solver, temp_arrays)
end

# version of update_K_and_F! that doesn't update the kernel
# intended for models that don't need to calculate the kernel separately
function update_K_and_F!(equation::AbstractNoKernelEquation, solver::TimeDoublingSolver, ::Nothing, temp_arrays::FuchsTempStruct, it::Int)
    update_F!(equation, solver, temp_arrays, it)
end

function update_integrals!(temp_arrays::FuchsTempStruct, ::AbstractNoKernelEquation, it::Int)
    F_I = temp_arrays.F_I
    F_temp = temp_arrays.F_temp
    F_I[it] = (F_temp[it] + F_temp[it-1]) / 2
end

function allocate_results!(t_array, F_array, K_array,::AbstractNoKernelEquation, solver::TimeDoublingSolver, temp_arrays::FuchsTempStruct; istart=2(solver.N) + 1, iend=4(solver.N))
    N = solver.N
    δt = solver.Δt / (4N)
    for it = istart:iend
        t = δt * it
        push!(t_array, t)
        push!(F_array, deepcopy(temp_arrays.F_temp[it]))
    end
end

function new_time_mapping!(equation::AbstractNoKernelEquation, solver::TimeDoublingSolver, temp_arrays::FuchsTempStruct)
    F = temp_arrays.F_temp
    F_I = temp_arrays.F_I
    N = solver.N
    for j = 1:N
        F_I[j] = (F_I[2j] + F_I[2j-1]) / 2
        F[j] = F[2j]
    end
    for j = (N+1):2*N
        # Flenner/Szamel version:
        #F_I[j] = (F_I[2j] + 4 * F_I[2j-1] + F_I[2j-2]) / 6
        # Hofacker/Fuchs version:
        #F_I[j] = (F_I[2j] + F_I[2j-1]) / 2
        # Hofacker/Fuchs that does not require extra update_integrals!
        F_I[j] = (F[2j] + 2 * F[2j-1] + F[2j-2]) / 4
        F[j] = F[2j]
    end
    for j = 2N+1:4N
        F_I[j] = equation.F₀ * zero(eltype(eltype(F_I)))
        F[j] = equation.F₀ * zero(eltype(eltype(F)))
    end
    solver.Δt *= 2
end

function initialize_output_arrays(equation::AbstractNoKernelEquation)
    return [0.0], typeof(equation.F₀)[equation.F₀], nothing
end