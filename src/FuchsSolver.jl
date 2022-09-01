
mutable struct FuchsTempStruct{T, T2, T3, VT, VT2}
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
end

mutable struct FuchsSolver{I, F, T1, T2, T3, T4, T5} <: Solver
    Ftype::DataType
    Kerneltype::DataType
    N::I
    Δt::F
    t_max::F
    kernel_evals::I
    max_iterations::I
    tolerance::F
    temp_arrays::FuchsTempStruct{T1, T2, T3, T4, T5}
    second_order::Bool
    verbose::Bool
    start_time::Float64
    inplace::Bool
end

check_if_diag(::Diagonal) =  true 
check_if_diag(::Any) = false

"""
    FuchsSolver(problem::LinearMCTProblem; N=32, Δt=10^-10, t_max=10.0^10, max_iterations=10^4, tolerance=10^-10, verbose=false)

Uses the algorithm devised by Fuchs et al. to solve the `LinearMCTProblem`.

# Arguments:
* `problem`: an instance of LinearMCTProblem
* `N`: The number of time points in the interval is equal to `4N`
* `t_max`: when this time value is reached, the integration returns
* `Δt`: starting time step, this will be doubled repeatedly
* `max_iterations`: the maximal number of iterations before convergence is reached for each time doubling step
* `tolerance`: while the error is bigger than this value, convergence is not reached. The error by default is computed as the absolute sum of squares
* `verbose`: if `true`, information will be printed to STDOUT
* `inplace`: if `true` and the type of F is mutable, the solver will try to avoid allocating many temporaries
"""
function FuchsSolver(problem::LinearMCTProblem; N=32, Δt=10^-10, t_max=10.0^10, max_iterations=10^4, tolerance=10^-10, verbose=false, ismutable=true)
    starttime = time()
    Ftype = problem.Ftype
    Kerneltype = problem.Kerneltype
    K₀ = problem.K₀    
    F₀ = problem.F₀
    C1 = sum([problem.α, problem.β, problem.γ, K₀])
    C2 = K₀*F₀
    C3 = K₀*F₀
    temp_vec = K₀*F₀
    F_old = K₀*F₀
    temp_mat = K₀ + K₀
    temp_arrays = FuchsTempStruct(
                                    Ftype[], 
                                    Kerneltype[], 
                                    Ftype[], 
                                    Kerneltype[], 
                                    C1, 
                                    C1 + C1,
                                    C2, 
                                    C3,
                                    temp_vec,
                                    F_old,
                                    temp_mat, 
                                )
    for _ in 1:4N
        push!(temp_arrays.F_temp, K₀*F₀)
        push!(temp_arrays.K_temp, K₀ + K₀)
        push!(temp_arrays.F_I, K₀*F₀)
        push!(temp_arrays.K_I, K₀ + K₀)
    end
    second_order = !iszero(problem.α)
    Fmutable = ismutabletype(Ftype)
    inplace = Fmutable & ismutable
    return FuchsSolver(Ftype, Kerneltype, N, Δt, t_max, 0, max_iterations, tolerance, temp_arrays, second_order, verbose, starttime, inplace)
end


"""
    initialize_F_temp!(problem::MCTProblem, solver::FuchsSolver)

Fills the first 2N entries of the temporary arrays of F using forward Euler without a memory kernel in order to kickstart Fuchs' scheme.

"""
function initialize_F_temp!(problem::LinearMCTProblem, solver::FuchsSolver)
    N = solver.N
    δt = solver.Δt/(4*N)
    α = problem.α
    β = problem.β
    γ = problem.γ
    F₀ = problem.F₀
    ∂ₜF₀ = problem.∂ₜF₀
    second_order = solver.second_order

    ∂ₜF_old = ∂ₜF₀
    F_old = F₀
    for it = 1:2N
        if second_order
            ∂ₜₜF  = -α\(β*∂ₜF_old + γ*F_old)
            ∂ₜF = ∂ₜF_old + δt * ∂ₜₜF
            F = F_old + δt * ∂ₜF
        else
            ∂ₜF = -β\(γ * F_old)
            F = F_old + δt * ∂ₜF
        end
        solver.temp_arrays.F_temp[it] = F
        ∂ₜF_old = ∂ₜF
        F_old = F
    end
end

"""
    initialize_K_temp!(solver::FuchsSolver, kernel::MemoryKernel)

Evaluates the memory kernel at the first 2N time points.
"""
function initialize_K_temp!(solver::FuchsSolver, kernel::MemoryKernel)
    N = solver.N
    δt = solver.Δt/(4*N)
    for it = 1:2N
        t = it*δt
        if !solver.inplace
            solver.temp_arrays.K_temp[it] = evaluate_kernel(kernel, solver.temp_arrays.F_temp[it], t)
        else
            evaluate_kernel!(solver.temp_arrays.K_temp[it], kernel, solver.temp_arrays.F_temp[it], t)
        end
    end
end

isimmutabletype(x) = !ismutabletype(x)

"""
    initialize_integrals!(problem::MCTProblem, solver::FuchsSolver)

Initializes the integrals on the first 2N time points as prescribed in the literature.
"""
function initialize_integrals!(problem::MCTProblem, solver::FuchsSolver)
    F_I = solver.temp_arrays.F_I
    K_I = solver.temp_arrays.K_I
    F_temp = solver.temp_arrays.F_temp
    K_temp = solver.temp_arrays.K_temp
    N = solver.N

    if !solver.inplace
        # it = 1
        F_I[1] =  (F_temp[1] + problem.F₀)/2
        K_I[1] =  (3*K_temp[1] - K_temp[2])/2
        for it = 2:2N
            F_I[it] = (F_temp[it] + F_temp[it-1])/2
            K_I[it] = (K_temp[it] + K_temp[it-1])/2
        end
    else
        @. F_I[1] =  (F_temp[1] + problem.F₀)/2
        for it = 2:2N
            @. F_I[it] = (F_temp[it] + F_temp[it-1])/2
        end
        @. K_I[1] =  (3*K_temp[1] - K_temp[2])/2
        for it = 2:2N
            @. K_I[it] = (K_temp[it] + K_temp[it-1])/2
        end
    end
end


function initialize_temporary_arrays!(problem::MCTProblem, solver::FuchsSolver, kernel::MemoryKernel)
    initialize_F_temp!(problem, solver)
    initialize_K_temp!(solver, kernel)
    initialize_integrals!(problem, solver)
end

""""
    mymul!(c,a,b,α,β)

prescribes how types used in this solver should be multiplied in place. In particular, it performs
C.= β*C .+ α*a*b. defaults to mul!(c,a,b,α,β)
"""
mymul!(c,a,b,α,β) = mul!(c,a,b,α,β)
function mymul!(c::Vector{SMatrix{Ns, Ns, T, Ns2}}, a::Number, b::Vector{SMatrix{Ns, Ns, T, Ns2}}, α::Number, β::Number) where {Ns, Ns2, T}
    α2 = T(α)
    β2 = T(β)
    for ik in eachindex(c)
        c[ik] = β2*c[ik] + α2*a*b[ik]
    end
end

function mymul!(c::Vector{SMatrix{Ns, Ns, T, Ns2}}, a::UniformScaling, b::Vector{SMatrix{Ns, Ns, T, Ns2}}, α::Number, β::Number) where {Ns, Ns2, T}
    α2 = T(α)
    β2 = T(β)
    aλ = a.λ
    for ik in eachindex(c)
        c[ik] = β2*c[ik] + α2*aλ*b[ik]
    end
end

function mymul!(c::Vector{SMatrix{Ns, Ns, T, Ns2}}, a::Diagonal{SMatrix{Ns, Ns, T, Ns2}, Vector{SMatrix{Ns, Ns, T, Ns2}}}, b::Vector{SMatrix{Ns, Ns, T, Ns2}}, α::Number, β::Number) where {Ns, Ns2, T}
    α2 = T(α)
    β2 = T(β)
    adiag = a.diag
    for ik in eachindex(c)
        c[ik] = β2*c[ik] + α2*adiag[ik]*b[ik]
    end
end



"""
    update_Fuchs_parameters!(problem::LinearMCTProblem, solver::FuchsSolver, it::Int) 

Updates the parameters c1, c2, c3, according to the appendix of 
"Flenner, Elijah, and Grzegorz Szamel. Physical Review E 72.3 (2005): 031508"
using the naming conventions from that paper. If F is mutable (and therefore also c1,c2,c3), it will
updata the variables in place, otherwise it will create new copies. This is controlled by the solver.inplace 
setting.
"""
function update_Fuchs_parameters!(problem::LinearMCTProblem, solver::FuchsSolver, it::Int) 
    N = solver.N
    i2 = 2N
    δt = solver.Δt/(4N)
    K_I = solver.temp_arrays.K_I
    F_I = solver.temp_arrays.F_I
    F = solver.temp_arrays.F_temp
    kernel = solver.temp_arrays.K_temp
    α = problem.α
    β = problem.β
    γ = problem.γ
    if !solver.inplace # everything immutable (we are free to allocate)
        c1 = (2/(δt^2)*α + 3/(2δt)*β) + K_I[1] + γ 

        c2 = F_I[1] - problem.F₀

        c3 = α*(5*F[it-1] - 4*F[it-2] + F[it-3])/δt^2 
        c3 += β*(2/δt*F[it-1] - F[it-2]/(2δt))
        c3 += -kernel[it-i2]*F[i2] + kernel[it-1]*F_I[1] + K_I[1]*F[it-1]
        @inbounds for j = 2:i2
            c3 += (kernel[it-j] - kernel[it-j+1])*F_I[j]
        end
        @inbounds for j = 2:it-i2
            c3 += K_I[j]*(F[it-j] - F[it-j+1])
        end
        solver.temp_arrays.C1 = c1
        solver.temp_arrays.C2 = c2
        solver.temp_arrays.C3 = c3
    else # perform everything without allocations. The commented code is the corresponding scalar equivalent
        solver.temp_arrays.C1 .= (2/(δt^2)*α + 3/(2δt)*β) + K_I[1] + γ 
        solver.temp_arrays.C2 .= F_I[1] - problem.F₀
        temp_vec = solver.temp_arrays.temp_vec
        temp_mat = solver.temp_arrays.temp_mat

        c3 = solver.temp_arrays.C3
        # c3 .= α*(5*F[it-1] - 4*F[it-2] + F[it-3])/δt^2
        @. temp_vec = (5*F[it-1] - 4*F[it-2] + F[it-3])/δt^2
        mymul!(c3, α, temp_vec, true, false)

        # c3 .+= β*(2/δt*F[it-1] - F[it-2]/(2δt))
        @. temp_vec = 2/δt*F[it-1] - F[it-2]/(2δt)
        mymul!(c3, β, temp_vec,  true,  true)

        # c3 .+= -kernel[it-i2]*F[i2] + kernel[it-1]*F_I[1] + K_I[1]*F[it-1]
        mymul!(c3, kernel[it-i2], F[i2], -true,  true)
        mymul!(c3, kernel[it-1], F_I[1],  true,  true)
        mymul!(c3, K_I[1], F[it-1],  true,  true)

        for j = 2:i2
            # c3 .+= (kernel[it-j] - kernel[it-j+1])*F_I[j]
            if check_if_diag(temp_mat)
                @. temp_mat.diag = kernel[it-j].diag - kernel[it-j+1].diag
            else
                @. temp_mat = kernel[it-j] - kernel[it-j+1]
            end
            mymul!(c3, temp_mat, F_I[j],  true,  true)

        end
        for j = 2:it-i2
            # c3 .+= K_I[j]*(F[it-j] - F[it-j+1])
            @. temp_vec = F[it-j] - F[it-j+1]
            mymul!(c3, K_I[j], temp_vec, true,  true)
        end
    end
    return nothing
end 

"""
    update_F!(solver::FuchsSolver, it::Int) 

updates F using the formula c1*F = -K*C2 + C3.
"""
function update_F!(solver::FuchsSolver, it::Int) 
    c1 = solver.temp_arrays.C1 
    c1_temp = solver.temp_arrays.C1_temp 
    c2 = solver.temp_arrays.C2 
    c3 = solver.temp_arrays.C3
    if !solver.inplace
        solver.temp_arrays.F_temp[it] = c1 \ (-solver.temp_arrays.K_temp[it]*c2 + c3)
    else # do the operation above without allocations
        mymul!(solver.temp_arrays.temp_vec, solver.temp_arrays.K_temp[it],c2, true, false) 
        @. solver.temp_arrays.temp_vec = -solver.temp_arrays.temp_vec + c3
        if check_if_diag(c1)
            solver.temp_arrays.F_temp[it] .= c1.diag .\ solver.temp_arrays.temp_vec 
        else
            c1_temp .= c1
            ldiv!(solver.temp_arrays.F_temp[it], qr!(c1_temp, ColumnNorm()), solver.temp_arrays.temp_vec)
        end
    end
end 

function update_K_and_F!(solver::FuchsSolver, kernel::MemoryKernel, it::Int)
    update_K!(solver, kernel, it)
    update_F!(solver, it)
end

"""
    update_K!(solver::FuchsSolver, kernel::MemoryKernel, it::Int) 

evaluates the memory kernel, updating the value in solver.temp_arrays.K_temp    
"""
function update_K!(solver::FuchsSolver, kernel::MemoryKernel, it::Int) 
    N = solver.N
    δt = solver.Δt/(4N)
    t = δt*it
    if !solver.inplace
        solver.temp_arrays.K_temp[it] = evaluate_kernel(kernel, solver.temp_arrays.F_temp[it], t)
    else
        evaluate_kernel!(solver.temp_arrays.K_temp[it], kernel, solver.temp_arrays.F_temp[it], t)
    end
end


"""
    update_integrals!(solver::FuchsSolver, it::Int)

Update the discretisation of the integral of F and K, see the literature for details.
"""
function update_integrals!(solver::FuchsSolver, it::Int) 
    K_I = solver.temp_arrays.K_I
    F_I = solver.temp_arrays.F_I
    F_temp = solver.temp_arrays.F_temp
    K_temp = solver.temp_arrays.K_temp
    F_I[it] = (F_temp[it]+F_temp[it-1])/2
    K_I[it] = (K_temp[it]+K_temp[it-1])/2
end


"""
    find_error(F_new::T, F_old::T) where T

Finds the error between a new and old iteration of F. The returned scalar will be compared 
to the tolerance to establish convergence. 
"""
function find_error(F_new::T, F_old::T) where T
    return maximum(abs.(F_new - F_old))
end

function find_error(F_new::T, F_old::T) where T <: Vector
    error = zero(eltype(eltype(F_old)))
    for i in eachindex(F_old)
        new_error = abs(maximum(F_new[i]-F_old[i]))
        if new_error > error
            error = new_error
        end
    end
    return error
end

function find_error(F_new::Number, F_old::Number)
    return abs(F_new - F_old)
end

"""
    do_time_steps!(problem::LinearMCTProblem, solver::FuchsSolver, kernel::MemoryKernel)

Solves the equation on the time points with index 2N+1 until 4N, for each point doing a recursive iteration
to find the solution to the nonlinear equation C1 F  = -C2 M(F) + C3.
"""
function do_time_steps!(problem::MCTProblem, solver::FuchsSolver, kernel::MemoryKernel)
    N = solver.N
    F_temp = solver.temp_arrays.F_temp
    tolerance = solver.tolerance
    for it = (2N+1):(4N)
        error = typemax(Float64)
        iterations = 1
        F_old = solver.temp_arrays.F_old

        update_Fuchs_parameters!(problem, solver, it)
        update_F!(solver, it)

        while error > tolerance
            iterations += 1
            if iterations > solver.max_iterations
                throw(DomainError("Iteration did not converge. Either increase the number of time steps before a time doubling, or choose a different memory kernel."))
            end
            update_K_and_F!(solver, kernel, it)
            error = find_error(F_temp[it], F_old)
            if !solver.inplace
                F_old = F_temp[it]
            else
                F_old .= F_temp[it]
            end
        end
        update_integrals!(solver, it)
        solver.kernel_evals += iterations-1
    end
    return
end


"""
    allocate_results!(t_array, F_array, K_array, solver; istart=2(solver.N)+1, iend=4(solver.N))

pushes the found solution, stored in `solver.temp_arrays` with indices `istart` until `istop` to the output arrays.
"""
function allocate_results!(t_array, F_array, K_array, solver; istart=2(solver.N)+1, iend=4(solver.N))
    N = solver.N
    δt = solver.Δt/(4N)
    for it = istart:iend
        t = δt*it
        push!(t_array, t)
        push!(F_array, deepcopy(solver.temp_arrays.F_temp[it]))
        push!(K_array, deepcopy(solver.temp_arrays.K_temp[it]))
    end
end


"""
new_time_mapping!(problem::MCTProblem, solver::FuchsSolver)

Performs the time mapping central to Fuchs' algorithm with the conventions prescribed in 
"Flenner, Elijah, and Grzegorz Szamel. Physical Review E 72.3 (2005): 031508". 
Performs them inplace if solver.inplace = true in order to avoid unnecessary allocations.
"""
function new_time_mapping!(problem::MCTProblem, solver::FuchsSolver)
    F = solver.temp_arrays.F_temp
    K = solver.temp_arrays.K_temp
    F_I = solver.temp_arrays.F_I
    K_I = solver.temp_arrays.K_I
    N = solver.N
    if !solver.inplace
        for j = 1:N
            F_I[j] = (F_I[2j] + F_I[2j - 1])/2
            K_I[j] = (K_I[2j] + K_I[2j - 1])/2
            F[j] = F[2j]
            K[j] = K[2j]
        end
        for j = (N + 1):2*N
            F_I[j] = (F_I[2j] + 4*F_I[2j - 1] + F_I[2j-2])/6
            K_I[j] = (K_I[2j] + 4*K_I[2j - 1] + K_I[2j-2])/6
            F[j] = F[2j]
            K[j] = K[2j]
        end
        for j = 2N+1:4N
            F_I[j] = problem.F₀*zero(eltype(eltype(F_I)))
            K_I[j] = problem.K₀*zero(eltype(eltype(K_I)))
            F[j] = problem.F₀*zero(eltype(eltype(F)))
            K[j] = problem.K₀*zero(eltype(eltype(K)))
        end
    else
        isdiag = check_if_diag(K_I[1])
        for j = 1:N
            @. F_I[j] = (F_I[2j] + F_I[2j - 1])/2
            if isdiag
                @. K_I[j].diag = (K_I[2j].diag + K_I[2j - 1].diag)/2
            else
                @. K_I[j] = (K_I[2j] + K_I[2j - 1])/2
            end
            @. F[j] = F[2j]
            @. K[j] = K[2j]
        end
        for j = (N + 1):2*N
            @. F_I[j] = (F_I[2j] + 4*F_I[2j - 1] + F_I[2j-2])/6
            if isdiag
                @. K_I[j].diag = (K_I[2j].diag + 4*K_I[2j - 1].diag + K_I[2j-2].diag)/6
            else
                @. K_I[j] = (K_I[2j] + 4*K_I[2j - 1] + K_I[2j-2])/6
            end
            @. F[j] = F[2j]
            @. K[j] = K[2j]
        end
        for j = 2N+1:4N
            @. F_I[j] = zero(problem.FK_elementtype)
            @. K_I[j] = zero(problem.FK_elementtype)
            @. F[j] = zero(problem.FK_elementtype)
            @. K[j] = zero(problem.FK_elementtype)
        end
    end
    solver.Δt *= 2
end

function log_results(solver, p)
    if solver.verbose
        next!(p)
    end
end


function convertresults(F_array::Vector{<:Number}, K_array::Vector{<:Number})
    return F_array, K_array
end


"""
    convertresults(F_array::Vector{T}, K_array::Vector{Diagonal{T, T2}}) where {T2, T}

converts the arrays of arrays into multidimensional arrays for ease of use.
"""
function convertresults(F_array::Vector{T}, K_array::Vector{Diagonal{T, T2}}) where {T2, T}
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

function convertresults(F_array, K_array)
    Nt = length(F_array)
    Nk = length(F_array[1])
    F = zeros(eltype(F_array[1]), Nk, Nt)
    K = zeros(eltype(K_array[1]), Nk, Nk, Nt)
    for it in 1:Nt
        for ik1 = 1:Nk
            F[ik1, it] = F_array[it][ik1]
            for ik2 = 1:Nk
                K[ik1, ik2, it] = K_array[it][ik1, ik2]
            end
        end
    end
    return F, K
end


function solve(problem::MCTProblem, solver::FuchsSolver, kernel::MemoryKernel)
    # Documented in src/Solvers.jl
    t₀ = 0.0
    F₀ = problem.F₀
    K₀ = problem.K₀

    Ftype = problem.Ftype
    kerneltype = problem.Kerneltype

    #initialize the output arrays with the initial conditions
    t_array = Float64[t₀]
    F_array = Ftype[F₀]
    K_array = kerneltype[K₀]

    initialize_temporary_arrays!(problem, solver, kernel)
    allocate_results!(t_array, F_array, K_array, solver; istart=1, iend=2(solver.N))
    startΔt = solver.Δt
    solver.kernel_evals = 1
    # for the progressbar: (turns off in non-interactive environments such as a HPC)
    is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")
    p = Progress(ceil(Int,log2(solver.t_max/solver.Δt)); output = stderr, enabled = !is_logging(stderr))
    # main loop of the algorithm
    while solver.Δt < solver.t_max*2
        do_time_steps!(problem, solver, kernel)
        allocate_results!(t_array, F_array, K_array, solver)
        log_results(solver, p)
        new_time_mapping!(problem, solver)
    end
    solver.Δt = startΔt
    F_array, K_array = convertresults(F_array, K_array)
    return t_array, F_array, K_array
end