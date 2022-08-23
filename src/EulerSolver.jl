mutable struct EulerSolver{I, F, B} <: Solver
    Ftype::DataType
    N::I
    Δt::F
    t_max::F
    temp_arrays::B
    verbose::Bool
end

"""
    EulerSolver(problem::MCTProblem; t_max=10.0^2, Δt=10^-3, verbose=false)

Constructs a solver object that, when called `solve` upon will solve an `MCTProblem` using a forward Euler method.
It will discretise the integral using a Trapezoidal rule.

arguments:
    `problem` an instance of MCTProblem
    `t_max` when this time value is reached, the integration returns
    `Δt` fixed time step
    `verbosity` if `true`, information will be printed to STDOUT

returns 
    `t` an array of time values
    `F` The solution in an array of which the last dimension corresponds to the time.
    `K` The memory kernel corresponding to each `F`
"""
function EulerSolver(problem::MCTProblem; t_max=10.0^2, Δt=10^-3, verbose=false)
    Ftype = typeof(problem.F₀)
    F_element_type = eltype(problem.F₀)
    N = ceil(Int, t_max/Δt)
    temp_arrays = (∂ₜF_array_reverse = Ftype[], integrand_array = Ftype[])

    for _ in 1:N
        push!(temp_arrays.integrand_array, problem.F₀.*Ref(zero(F_element_type)))
    end
    return EulerSolver(Ftype, N, Δt, t_max, temp_arrays, verbose)
end

function construct_euler_integrand!(integrand, K_array, ∂ₜF_array_reverse, it)
    @assert it <= length(K_array)
    @inbounds @simd for i = 1:it # fill temporary array for the integrand
        integrand[i] = K_array[i]*∂ₜF_array_reverse[i]
    end
end

function trapezoidal_integration(f, δt, it)
    if it == 1
        return f[1]*δt
    end
    result = f[1]/2    
    result += f[it]/2
    itmin1 = it-1
    @inbounds @simd for it2 = 2:itmin1
        result += f[it2]
    end
    result *= δt
    return result
end

function solve(problem::MCTProblem, solver::EulerSolver, kernel::MemoryKernel)
    tstart = time()

    Ftype = problem.Ftype
    N = solver.N
    Δt = solver.Δt
    verbose = solver.verbose
    α = problem.α
    β = problem.β
    γ = problem.γ
    F₀ = problem.F₀
    ∂ₜF₀ = problem.∂ₜF₀
    t_array = Float64[]
    F_array = Ftype[]
    t = 0.0

    K₀ = problem.K₀
    
    kerneltype = problem.Kerneltype
    K_array = kerneltype[]
    push!(t_array, t)
    push!(K_array, K₀)
    push!(F_array, F₀)
    ∂ₜF_array_reverse = solver.temp_arrays.∂ₜF_array_reverse
    pushfirst!(∂ₜF_array_reverse, ∂ₜF₀)
    integrand_array = solver.temp_arrays.integrand_array

    second_order = true
    @assert eltype(F₀) == eltype(K₀)
    if iszero(α)
        second_order = false
    end

    for it = 1:N
        t += Δt
        ∂ₜF_old = ∂ₜF_array_reverse[1]
        F_old = F_array[end]

        construct_euler_integrand!(integrand_array, K_array, ∂ₜF_array_reverse, it)
        time_integral = trapezoidal_integration(integrand_array, Δt, it)
        if second_order
            ∂ₜₜF  = -α\(β*∂ₜF_old + γ*F_old + time_integral)
            ∂ₜF = ∂ₜF_old + Δt* ∂ₜₜF
            F = F_old + Δt * ∂ₜF
        else
            ∂ₜF = -β\(γ * F_old + time_integral)
            F = F_old + Δt * ∂ₜF
        end
        K = kernel(F, t)
        push!(t_array, t)
        push!(K_array, K)
        push!(F_array, F)
        pushfirst!(∂ₜF_array_reverse, ∂ₜF)     
        if verbose && it%1000 == 0
            println("t = ", round(t, digits=4),"/", solver.t_max, 
            ", percentage done = ", round(it/N*100, digits=2), "%",
            ", elapsed time = ", round(time()-tstart, digits=4))
        end
    end
    tstop = time()
    if verbose
        println("Solution found after ", round(tstop-tstart, digits=6) , " seconds.")
    end
    F_array, K_array = convertresults(F_array, K_array)
    return t_array, F_array, K_array
end