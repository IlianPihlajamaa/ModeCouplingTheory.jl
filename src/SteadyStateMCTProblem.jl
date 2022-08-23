function solve_steady_state(γ, F₀::Union{Number,SVector}, kernel; tolerance=10^-8, max_iterations=10^6, verbose=false)
    t = Inf
    K = kernel(F₀, t)
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
        K = kernel(F, t)
        error = find_error(F, Fold)
        Fold = F
        if verbose 
            println("The error is $(error) after $iterations iterations. Elapsed time = $(round((time()-begintime), digits=3)) seconds.")
        end
    end
    if verbose 
        println("converged after $iterations iterations. Elapsed time = $(round((time()-begintime), digits=3)) seconds.")
    end
    return F    
end


function solve_steady_state(γ, F₀::Vector, kernel; tolerance=10^-8, max_iterations=10^6, verbose=false)
    t = Inf
    if typeof(γ) <: AbstractVector
        γ = Diagonal(γ)
    elseif typeof(γ) <: UniformScaling
        γ = γ*I(length(F₀))
    end
    K = kernel(F₀, t)
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
        kernel(K, F, t)
        error = find_error(F, Fold)
        Fold .= F
        if verbose 
            println("The error is $(error)) after $iterations iterations. Elapsed time = $(round((time()-begintime), digits=3)) seconds.")
        end
    end
    if verbose 
        println("converged after $iterations iterations. Elapsed time = $(round((time()-begintime), digits=3)) seconds.")
    end
    return F    
end
