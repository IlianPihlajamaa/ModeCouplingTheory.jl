function solve_steady_state_immutable(γ, F₀, kernel; tolerance=10^-8, max_iterations=10^6, verbose=false)
    t = Inf
    if typeof(γ) <: AbstractVector
        γ = Diagonal(γ)
    end
    K = evaluate_kernel(kernel, F₀, t)
    error = tolerance*2
    Fold = F₀*one(eltype(F₀))
    F = F₀
    iterations = 0
    begintime = time()
    while error > tolerance
        iterations += 1
        if iterations > max_iterations
            error("The recursive iteration did not converge. The error is $error after $iterations iterations.")
        end
        F = (K + γ)\(K*F₀)
        K = evaluate_kernel(kernel, F, t)
        error = find_error(F, Fold)
        Fold = F
        if verbose 
            println("The error is $(error) after $iterations iterations. Elapsed time = $(round((time()-begintime), digits=3)) seconds.")
        end
    end
    if verbose 
        println("converged after $iterations iterations. Elapsed time = $(round((time()-begintime), digits=3)) seconds.")
    end
    return MemoryEquationSolution([Inf64], [F], [K], (tolerance=tolerance, max_iterations=max_iterations, verbose=verbose))    
end

"""
    solve_steady_state(γ, F₀, kernel; tolerance=10^-8, max_iterations=10^6, verbose=false, inplace=true)

Finds the steady-state solution (non-ergodicity parameter) of the generalized Langevin equation by recursive iteration of F = (K + γ)⁻¹ * K(F) * F₀
This function assumes δ = 0, since that would lead to a divergence.

# Arguments:
* `γ`: parameter in front of the linear term in F
* `F₀`: initial condition of F. This is also the initial condition of the rootfinding method.
* `kernel`: callable memory kernel
* `max_iterations`: the maximal number of iterations before convergence is reached
* `tolerance`: while the error is bigger than this value, convergence is not reached. The error by default is computed as the absolute sum of squares between successive iterations
* `verbose`: if `true`, information will be printed to `STDOUT`
* `inplace``: if true and if the type of F is mutable, the solver will try to avoid allocating many temporaries. default = `true``

# Returns:
* The steady state solution
"""
function solve_steady_state(γ, F₀, kernel; tolerance=10^-8, max_iterations=10^6, verbose=false, inplace=true)
    if ismutable(F₀) && inplace
        return solve_steady_state_mutable(γ, F₀, kernel; tolerance=tolerance, max_iterations=max_iterations, verbose=verbose)
    else 
        return solve_steady_state_immutable(γ, F₀, kernel; tolerance=tolerance, max_iterations=max_iterations, verbose=verbose)
    end
end



function solve_steady_state_mutable(γ, F₀, kernel; tolerance=10^-8, max_iterations=10^6, verbose=false)
    t = Inf
    if typeof(γ) <: AbstractVector
        γ = Diagonal(γ)
    elseif typeof(γ) <: UniformScaling
        γ = γ*I(length(F₀))
    end
    K = evaluate_kernel(kernel, F₀, t)
    error = tolerance*2
    F = K*F₀
    Fold = copy(F₀)
    tempvec = copy(F₀)
    tempmat = copy(K+γ)
    iterations = 0
    begintime = time()
    while error > tolerance
        iterations += 1
        if iterations > max_iterations
            error("The recursive iteration did not converge. The error is $error after $iterations iterations.")
        end
        # the following lines compute F = (K + γ)\K*F₀
        # F = (K + γ)\K*F₀
        mymul!(tempvec, K, F₀, true, false)
        if check_if_diag(tempmat)
            tempmat.diag .= K.diag .+ γ.diag
            F .= tempmat.diag.\tempvec
        else
            tempmat .= K .+ γ
            F .= tempmat\tempvec
        end
        evaluate_kernel!(K, kernel, F, t)
        error = find_error(F, Fold)
        Fold .= F
        if verbose 
            println("The error is $(error) after $iterations iterations. Elapsed time = $(round((time()-begintime), digits=3)) seconds.")
        end
    end
    if verbose 
        println("converged after $iterations iterations. Elapsed time = $(round((time()-begintime), digits=3)) seconds.")
    end
    return MemoryEquationSolution([Inf64], [F], [K], (tolerance=tolerance, max_iterations=max_iterations, verbose=verbose))    
end
