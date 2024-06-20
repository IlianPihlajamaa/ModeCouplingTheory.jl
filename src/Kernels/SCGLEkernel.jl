struct SCGLEKernel{T, T2} <: MemoryKernel
    k::Vector{T}
    λ::Vector{T}
    M::Vector{T}
    prefactor::T2
end

"""
    SCGLEKernel(ϕ, k_array, S_array)

Constructor of a SCGLEKernel. It implements the kernel
K(k,t) = λ(k)Δζ(t)
where 
Δζ(t) = kBT / (36πϕ) ∫dq M^2(k,q) Fs(q,t) F(q,t)
in which k and q are vectors. For Fs and F being the self- and collective-intermediate scattering function.

# Arguments:
* `ϕ`: volume fraction
* `k_array`: vector of wavenumbers at which the structure factor is known
* `S_array`: concatenated vector for ones(length(structure factor)) and structure factor

# Returns:

an instance `k` of `SCGLEKernel <: MemoryKernel`, which can be called both in-place and out-of-place:
`evaluate_kernel!(out, kernel, F, t)`
`out = evaluate_kernel(kernel, F, t)`
"""
function SCGLEKernel(ϕ, k_array, S_array)
    Δk = k[2] - k[1]
    kc = 1.302*2π
    prefactor = Δk/(36π*ϕ)
    Nk = length(k_array)
    Nk2 = div(Nk,2)
    λ = zeros(Nk)
    M = zeros(Nk2)
    for (i, k) in enumerate(k_array[1:Nk2])
        λ[i] = 1/(1+(k/kc)^2)
        λ[Nk2+i] = 1/(1+(k/kc)^2)
        S = S_array[Nk2+i]
        M[i] = (k^4)*(1-1/S)^2
    end
    return SCGLEKernel(k_array, λ, M, prefactor)
end

function evaluate_kernel!(out::Diagonal, kernel::SCGLEKernel, F, t)
    out.diag .= zero(eltype(out.diag)) # set the output array to zero
    k_array = kernel.k
    M = kernel.M
    λ = kernel.λ
    Nk = length(k_array)
    Nk2 = div(Nk,2)
    # Δζ(q) Integral 
    Δζ = zero(eltype(M))
    for i in 1:Nk2
        Δζ += M[i]*F[i]*F[Nk2+i]
    end
    Δζ *= kernel.prefactor
    for i = 1:Nk
        out.diag[i] += λ[i]*Δζ
    end
end

function evaluate_kernel(kernel::SCGLEKernel, F, t)
    out = Diagonal(similar(F)) # we need it to produce a diagonal matrix
    evaluate_kernel!(out, kernel, F, t) # call the in-place version
    return out
end

"""
TODO:
MSD
"""
