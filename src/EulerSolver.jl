mutable struct EulerSolver{I, F} <: Solver
    N::I
    Δt::F
    t_max::F
    verbose::Bool
end

"""
    EulerSolver(equation::LinearMCTEquation; t_max=10.0^2, Δt=10^-3, verbose=false)

Constructs a solver object that, when called `solve` upon will solve an `LinearMCTEquation` using a forward Euler method.
It will discretise the integral using a Trapezoidal rule. Use this solver only for testing purposes. It is a wildy 
inefficient way to solve MCT-like equations.

# Arguments:
* `equation` an instance of LinearMCTEquation
* `t_max` when this time value is reached, the integration returns
* `Δt` fixed time step
* `verbose` if `true`, information will be printed to STDOUT
"""
function EulerSolver(; t_max=10.0^2, Δt=10^-3, verbose=false)
    N = ceil(Int, t_max/Δt)
    return EulerSolver(N, Δt, t_max, verbose)
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


function Euler_step(F_old, ∂ₜF_old, time_integral, equation, solver::EulerSolver, t)
    equation.update_coefficients!(equation.coeffs, t)
    α = equation.coeffs.α
    β = equation.coeffs.β
    γ = equation.coeffs.γ
    δ = equation.coeffs.δ
    Δt = solver.Δt
    if !iszero(α)
        ∂ₜₜF  = -α\(β*∂ₜF_old + γ*F_old + δ + time_integral)
        ∂ₜF = ∂ₜF_old + Δt* ∂ₜₜF
        F = F_old + Δt * ∂ₜF
    else
        ∂ₜF = -β\(γ * F_old + δ + time_integral)
        F = F_old + Δt * ∂ₜF
    end
    return F, ∂ₜF
end

function log_results(solver::EulerSolver, tstart, t, it)
    if solver.verbose && it%(solver.N÷50) == 0
        println("t = ", round(t, digits=4),"/", solver.t_max, 
        ", elapsed time = ", round(time()-tstart, digits=4))
    end
end

function allocate_results!(t_array, F_array, K_array, ∂ₜF_array_reverse, t, K, F, ∂ₜF)
    push!(t_array, t)
    push!(K_array, K)
    push!(F_array, F)
    pushfirst!(∂ₜF_array_reverse, ∂ₜF)
end

function initialize_output_arrays(equation::MCTEquation, solver::EulerSolver)
    F0 = equation.F₀
    dF0 = equation.∂ₜF₀
    ∂ₜF_array_reverse = typeof(dF0)[dF0]
    integrand_array = typeof(F0)[]
    for _ in 1:solver.N
        push!(integrand_array, F0.*Ref(zero(eltype(F0))))
    end
    return typeof(0.0)[0.0], typeof(F0)[F0], typeof(equation.K₀)[equation.K₀], ∂ₜF_array_reverse, integrand_array
end


function solve(equation::LinearMCTEquation, solver::EulerSolver)
    tstart = time()
    kernel = equation.kernel
    N = solver.N
    Δt = solver.Δt
    t_array, F_array, K_array, ∂ₜF_array_reverse, integrand_array = initialize_output_arrays(equation, solver)
    t = 0.0
    for it = 1:N
        t += Δt
        ∂ₜF_old = ∂ₜF_array_reverse[1]
        F_old = F_array[end]
        construct_euler_integrand!(integrand_array, K_array, ∂ₜF_array_reverse, it)
        time_integral = trapezoidal_integration(integrand_array, Δt, it)
        F, ∂ₜF = Euler_step(F_old, ∂ₜF_old, time_integral, equation, solver, t)
        K = evaluate_kernel(kernel, F, t)
        allocate_results!(t_array, F_array, K_array, ∂ₜF_array_reverse, t, K, F, ∂ₜF)
        log_results(solver, tstart, t, it)
    end
    sol = MCTSolution(t_array, F_array, K_array, solver)
    return sol
end