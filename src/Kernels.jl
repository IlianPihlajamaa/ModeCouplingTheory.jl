abstract type MemoryKernel end

"""
    ExponentiallyDecayingKernel{T<:Number} <: MemoryKernel

Scalar kernel with fields `λ` and `τ` which when called returns `λ exp(-t/τ)`.
"""
struct ExponentiallyDecayingKernel{T1<:Number, T2<:Number} <: MemoryKernel
    λ::T1
    τ::T2
end

function evaluate_kernel(kernel::ExponentiallyDecayingKernel, F::Number, t)
    return kernel.λ * exp(-t/kernel.τ)
end


"""
    SchematicF1Kernel{T<:Number} <: MemoryKernel

Scalar kernel with field `λ` which when called returns `λ F`.
"""
struct SchematicF1Kernel{T<:Number} <: MemoryKernel
    λ::T
end

function evaluate_kernel(kernel::SchematicF1Kernel, F::Number, t)
    λ = kernel.λ
    return λ * F
end

"""
    SchematicF2Kernel{T<:Number} <: MemoryKernel

Scalar kernel with field `λ` which when called returns `λ F^2`.
"""
struct SchematicF2Kernel{T<:Number} <: MemoryKernel
    λ::T
end

function evaluate_kernel(kernel::SchematicF2Kernel, F::Number, t)
    λ = kernel.λ
    return λ * F^2
end

"""
    SchematicF1Kernel{T<:Number} <: MemoryKernel

Scalar kernel with fields `λ1`, `λ2`, and `λ3` which when called returns `λ1 * F^1 + λ2 * F^2 + λ3 * F^3`.
"""
struct SchematicF123Kernel{T<:Number} <: MemoryKernel
    λ1::T
    λ2::T
    λ3::T
end

function evaluate_kernel(kernel::SchematicF123Kernel, F::Number, t)
    return kernel.λ1 * F^1 + kernel.λ2 * F^2 + kernel.λ3 * F^3
end


"""
    SchematicDiagonalKernel{T<:Union{SVector, Vector}} <: MemoryKernel

Matrix kernel with field `λ` which when called returns `Diagonal(λ .* F .^ 2)`, i.e., it implements a non-coupled system of SchematicF2Kernels.
"""
struct SchematicDiagonalKernel{T<:Union{SVector, Vector}} <: MemoryKernel
    λ::T
    SchematicDiagonalKernel(λ::T) where {T<:Union{SVector,Vector}} = eltype(λ) <: Number ? new{T}(λ) : error("element type of this kernel must be a number")
end

function evaluate_kernel(kernel::SchematicDiagonalKernel, F::Union{SVector,Vector}, t)
    λ = kernel.λ
    return Diagonal(λ .* F .^ 2)
end

function evaluate_kernel!(out::Diagonal, kernel::SchematicDiagonalKernel, F::Vector, t)
    λ = kernel.λ
    diag = out.diag
    @. diag = λ * F^2
end

"""
    SchematicMatrixKernel{T<:Union{SVector, Vector}} <: MemoryKernel

Matrix kernel with field `λ` which when called returns `λ * F * Fᵀ`, i.e., it implements Kαβ = λ*Fα*Fβ.
"""
struct SchematicMatrixKernel{T<:Union{SMatrix,Matrix}} <: MemoryKernel
    λ::T
    SchematicMatrixKernel(λ::T) where {T<:Union{SMatrix,Matrix}} = eltype(λ) <: Number ? new{T}(λ) : error("element type of this kernel must be a number")
end

function evaluate_kernel(kernel::SchematicMatrixKernel, F::Union{SVector,Vector}, t)
    λ = kernel.λ
    return λ * F * F'
end

function evaluate_kernel!(out::Matrix, kernel::SchematicMatrixKernel, F::Vector, t)
    λ = kernel.λ
    @tullio out[i, j] = λ[i, k] * F[k] * F[j]
end


struct ModeCouplingKernel{F,V,M, M2} <: MemoryKernel
    ρ::F
    kBT::F
    m::F
    Nk::Int
    k_array::V
    A1::M2
    A2::M2
    A3::M2
    T1::V
    T2::V
    T3::V
    V1::M
    V2::M
    V3::M
end

"""
    ModeCouplingKernel(ρ, kBT, m, k_array, Sₖ)

Constructor of a ModeCouplingKernel. It implements the kernel
K(k,t) = ρ kBT / (16π^3m) ∫dq V^2(k,q) F(q,t) F(k-q,t)
in which k and q are vectors. 

# Arguments:

* `ρ`: number density
* `kBT`: Thermal energy
* `m` : particle mass
* `k_array`: vector of wavenumbers at which the structure factor is known
* `Sₖ`: structure factor

# Returns:

an instance `k` of `ModeCouplingKernel <: MemoryKernel`, which can be called both in-place and out-of-place:
`k`(out, F, t)
out = `k`(F, t)
"""
function ModeCouplingKernel(ρ, kBT, m, k_array, Sₖ)
    Nk = length(k_array)
    T = promote_type(eltype(Sₖ), eltype(k_array), typeof(ρ), typeof(kBT), typeof(m))
    ρ, kBT, m = T(ρ), T(kBT), T(m)
    k_array, Sₖ = T.(k_array), T.(Sₖ)
    Δk = k_array[2] - k_array[1]
    @assert k_array[1] ≈ Δk / 2
    @assert all(diff(k_array) .≈ Δk)
    Cₖ = @. (Sₖ - 1) / (ρ * Sₖ)
    T1 = similar(k_array)
    T2 = similar(k_array)
    T3 = similar(k_array)
    A1 = similar(k_array, (Nk, Nk))
    A2 = similar(k_array, (Nk, Nk))
    A3 = similar(k_array, (Nk, Nk))
    V1 = similar(k_array, (Nk, Nk))
    V2 = similar(k_array, (Nk, Nk))
    V3 = similar(k_array, (Nk, Nk))
    D₀ = kBT / m
    for iq = 1:Nk
        for ip = 1:Nk
            p = k_array[ip]
            q = k_array[iq]
            cp = Cₖ[ip]
            cq = Cₖ[iq]
            V1[iq, ip] = p * q * (cp + cq)^2 / 4 * D₀ * ρ / (8 * π^2) * Δk * Δk
            V2[iq, ip] = p * q * (q^2 - p^2)^2 * (cq - cp)^2 / 4 * D₀ * ρ / (8 * π^2) * Δk * Δk
            V3[iq, ip] = p * q * (q^2 - p^2) * (cq^2 - cp^2) / 2 * D₀ * ρ / (8 * π^2) * Δk * Δk
        end
    end
    kernel = ModeCouplingKernel(ρ, kBT, m, Nk, k_array, A1, A2, A3, T1, T2, T3, V1, V2, V3)
    return kernel
end


function bengtzelius1!(T, A, Nk)
    T0 = zero(eltype(T))
    @inbounds for iq = 1:Nk
        T0 += A[iq, iq]
    end
    T[1] = T0
    @inbounds for ik = 2:Nk
        qmax = Nk - ik + 1
        Tik = T[ik-1]
        @simd for iq = 1:qmax
            ip = iq + ik - 1
            Tik += A[iq, ip] + A[ip, iq]
        end
        @simd for iq = 1:ik-1
            ip = ik - iq
            Tik -= A[iq, ip]
        end
        T[ik] = Tik
    end
end


function bengtzelius3!(T1, T2, T3, A1, A2, A3, Nk)
    T01 = zero(eltype(T1))
    T02 = zero(eltype(T2))
    T03 = zero(eltype(T3))
    @inbounds for iq = 1:Nk
        T01 += A1[iq, iq]
        T02 += A2[iq, iq]
        T03 += A3[iq, iq]
    end
    T1[1] = T01
    T2[1] = T02
    T3[1] = T03

    @inbounds for ik = 2:Nk
        qmax = Nk - ik + 1
        Tik1 = T1[ik-1]
        Tik2 = T2[ik-1]
        Tik3 = T3[ik-1]
        for iq = 1:qmax
            ip = iq + ik - 1
            Tik1 += A1[iq, ip] + A1[ip, iq]
            Tik2 += A2[iq, ip] + A2[ip, iq]
            Tik3 += A3[iq, ip] + A3[ip, iq]
        end
        for iq = 1:ik-1
            ip = ik - iq
            Tik1 -= A1[iq, ip]
            Tik2 -= A2[iq, ip]
            Tik3 -= A3[iq, ip]
        end
        T1[ik] = Tik1
        T2[ik] = Tik2
        T3[ik] = Tik3
    end
end

function fill_A!(kernel::ModeCouplingKernel, F)
    A1 = kernel.A1
    A2 = kernel.A2
    A3 = kernel.A3
    V1 = kernel.V1
    V2 = kernel.V2
    V3 = kernel.V3
    Nk = kernel.Nk
    @turbo for iq = 1:Nk
        for ip = 1:Nk
            fq = F[iq]
            fp = F[ip]
            f4 = fp * fq
            A1[iq, ip] = V1[iq, ip] * f4
            A2[iq, ip] = V2[iq, ip] * f4
            A3[iq, ip] = V3[iq, ip] * f4
        end
    end
end


function evaluate_kernel!(out::Diagonal, kernel::ModeCouplingKernel, F::Vector, t)
    A1 = kernel.A1
    A2 = kernel.A2
    A3 = kernel.A3
    T1 = kernel.T1
    T2 = kernel.T2
    T3 = kernel.T3
    k_array = kernel.k_array

    Nk = kernel.Nk
    fill_A!(kernel, F)
    bengtzelius3!(T1, T2, T3, A1, A2, A3, Nk)

    @inbounds for ik = 1:Nk
        k = k_array[ik]
        out.diag[ik] = k * kernel.T1[ik] + kernel.T2[ik] / k^3 + kernel.T3[ik] / k
    end
end


function evaluate_kernel(kernel::ModeCouplingKernel, F::Vector, t)
    out = Diagonal(similar(kernel.T1))
    evaluate_kernel!(out, kernel, F, t)
    return out
end

function bengtzelius3!(T1, T2, T3, A1, A2, A3, Nk, Ns)
    @assert size(T1) == size(T2) == size(T3) == (Nk,)
    @assert size(T1[1]) == size(T2[1]) == size(T3[1]) == (Ns,Ns)
    @assert size(A1) == size(A2) == size(A3) == (Ns, Ns, Nk, Nk)

    T01 = zero(eltype(T1))
    T02 = zero(eltype(T2))
    T03 = zero(eltype(T3))
    @inbounds @views  for iq = 1:Nk
        T01 += A1[:, :, iq, iq]
        T02 += A2[:, :, iq, iq]
        T03 += A3[:, :, iq, iq]
    end
    T1[1] = T01
    T2[1] = T02
    T3[1] = T03
    
    @inbounds @views for ik = 2:Nk
        Tik1 = T1[ik-1]
        Tik2 = T2[ik-1]
        Tik3 = T3[ik-1]
        for iq = 1:(Nk - ik + 1)
            ip = iq + ik - 1
            Tik1 += A1[:, :, iq, ip] 
            Tik1 += A1[:, :, ip, iq]
            Tik2 += A2[:, :, iq, ip] 
            Tik2 += A2[:, :, ip, iq]
            Tik3 += A3[:, :, iq, ip] 
            Tik3 += A3[:, :, ip, iq]
        end
        for iq = 1:ik-1
            ip = ik - iq
            Tik1 -= A1[:, :, iq, ip]
            Tik2 -= A2[:, :, iq, ip]
            Tik3 -= A3[:, :, iq, ip]
        end
        T1[ik] = Tik1
        T2[ik] = Tik2
        T3[ik] = Tik3
    end

end

struct MultiComponentModeCouplingKernel{F,AF1,AF2,AF3,AF4} <: MemoryKernel
    ρ::AF1
    kBT::F
    m::AF1
    Nk::Int
    Ns::Int
    k_array::AF1
    prefactor::AF2
    C::AF3
    A1::AF4
    A2::AF4
    A3::AF4
    T1::AF3
    T2::AF3
    T3::AF3
end

"""
    MultiComponentModeCouplingKernel(ρ, kBT, m, k_array, Sₖ)

Constructor of a MultiComponentModeCouplingKernel. It implements the kernel
Kαβ(k,t) = ρ  / (2 xα xβ (2π)³) Σμνμ'ν' ∫dq Vμ'ν'(k,q) Fμμ'(q,t) Fνν'(k-q,t) Vμν(k,q)
in which k and q are vectors and α and β species labels. 

# Arguments:

* `ρ`: vector of number densities for each species
* `kBT`: Thermal energy
* `m` : vector of particle masses for each species
* `k_array`: vector of wavenumbers at which the structure factor is known
* `Sₖ`: a `Vector` of Nk `SMatrix`s containing the structure factor of each component at each wave number

# Returns:

an instance `k` of `ModeCouplingKernel <: MemoryKernel`, which can be called both in-place and out-of-place:
k = MultiComponentModeCouplingKernel(ρ, kBT, m, k_array, Sₖ)
out = `k`(F, t)
`k`(out, F, t)
"""
function MultiComponentModeCouplingKernel(ρ, kBT, m, k_array, Sₖ)
    Nk = length(k_array)
    Ns = size(Sₖ[1], 2)
    @assert size(Sₖ) == (Nk,)
    @assert size(Sₖ[1]) == (Ns, Ns)
    @assert size(m) == size(ρ) == (Ns,)
    T = promote_type(eltype(Sₖ[1]), eltype(k_array), eltype(ρ), typeof(kBT), eltype(m))
    ρ, kBT, m = T.(ρ), T(kBT), T.(m)
    k_array = T.(k_array)
    Δk = k_array[2] - k_array[1]
    @assert k_array[1] ≈ Δk / 2
    @assert all(diff(k_array) .≈ Δk)

    ρ_all = sum(ρ)
    x = ρ / sum(ρ)

    S⁻¹ = inv.(Sₖ)

    Cₖ = similar(Sₖ)
    δαβ = Matrix(I(Ns))
    for i = 1:Nk
        Cₖ[i] = (δαβ./x - S⁻¹[i]) /ρ_all
    end

    T1 = similar(Cₖ)
    T2 = similar(Cₖ)
    T3 = similar(Cₖ)
    T1 .= Ref(zero(eltype(T1)))
    T2 .= Ref(zero(eltype(T2)))
    T3 .= Ref(zero(eltype(T3)))
    A1 = zeros(T, (Ns, Ns, Nk, Nk))
    A2 = zeros(T, (Ns, Ns, Nk, Nk))
    A3 = zeros(T, (Ns, Ns, Nk, Nk))
    x = ρ / sum(ρ)

    prefactor = zeros(T, Ns, Ns)
    for α = 1:Ns
        for β = 1:Ns
            prefactor[α, β] = ρ_all * kBT / (2 * m[α] * x[β]) * (Δk / 2 / π)^2
        end
    end

    kernel = MultiComponentModeCouplingKernel(ρ, kBT, m, Nk, Ns, k_array, prefactor, Cₖ, A1, A2, A3, T1, T2, T3)
    return kernel
end


function fill_A!(kernel::MultiComponentModeCouplingKernel, F)

    Nk = kernel.Nk
    Ns = size(F[1],1)
    k_array = kernel.k_array
    C = kernel.C
    prefactor = kernel.prefactor

    A1 = kernel.A1
    A2 = kernel.A2
    A3 = kernel.A3
    @inbounds @fastmath for α = 1:Ns
        for β = 1:Ns
            prefactor_αβ = prefactor[α, β]
            for iq = 1:Nk
                q = k_array[iq]
                Cq = C[iq]
                Fq = F[iq]
                for ip = 1:Nk
                    Cp = C[ip]
                    Fp = F[ip]
                    A1αβqp = zero(eltype(A1))
                    A2αβqp = zero(eltype(A2))
                    A3αβqp = zero(eltype(A3))
                    p = k_array[ip]
                    A_prefactor1 = p * q / 4
                    A_prefactor2 = p * q / 4 * (p^2 - q^2)^2
                    A_prefactor3 = p * q / 2 * (p^2 - q^2)
                    # We sum over 2 dummy indices. dummy1 is always the primed variable (mu', nu') 
                    # and dummy2 is the unprimed counterpart.
                    for dummy1 = 1:Ns 
                        for dummy2 = 1:Ns
                            V_dummy1dummy2αβ_pp = Cp[dummy1, α] * Cp[dummy2, β]
                            V_dummy1dummy2αβ_pq = Cp[dummy1, α] * Cq[dummy2, β]
                            V_dummy1dummy2αβ_qp = Cq[dummy1, α] * Cp[dummy2, β]
                            V_dummy1dummy2αβ_qq = Cq[dummy1, α] * Cq[dummy2, β]
                            term1 = V_dummy1dummy2αβ_pp * Fq[α, β]           * Fp[dummy1, dummy2]
                            term2 = V_dummy1dummy2αβ_pq * Fq[α, dummy2]      * Fp[dummy1, β]
                            term3 = V_dummy1dummy2αβ_qp * Fq[dummy1, β]      * Fp[α, dummy2]
                            term4 = V_dummy1dummy2αβ_qq * Fq[dummy1, dummy2] * Fp[α, β]
                            A1αβqp += term1 + term2 + term3 + term4
                            A2αβqp += term1 - term2 - term3 + term4
                            A3αβqp += term1 - term4
                        end
                    end
                    A1[α, β, iq, ip] = A1αβqp * A_prefactor1 * prefactor_αβ
                    A2[α, β, iq, ip] = A2αβqp * A_prefactor2 * prefactor_αβ
                    A3[α, β, iq, ip] = A3αβqp * A_prefactor3 * prefactor_αβ
                end
            end
        end
    end
end

function evaluate_kernel!(out::Diagonal, kernel::MultiComponentModeCouplingKernel, F::Vector, t)
    A1 = kernel.A1
    A2 = kernel.A2
    A3 = kernel.A3
    T1 = kernel.T1
    T2 = kernel.T2
    T3 = kernel.T3
    k_array = kernel.k_array

    Nk = kernel.Nk
    Ns = kernel.Ns
    fill_A!(kernel, F)

    bengtzelius3!(T1, T2, T3, A1, A2, A3, Nk, Ns)

    for ik = 1:Nk
        k = k_array[ik]
        out.diag[ik] = k * T1[ik] + T2[ik] / k^3 + T3[ik] / k
    end
end

function evaluate_kernel(kernel::MultiComponentModeCouplingKernel, F::Vector, t)
    out = Diagonal(similar(F))
    evaluate_kernel!(out, kernel, F, t)
    return out
end

struct NaiveMultiComponentModeCouplingKernel{F,AF1,AF2,AF3,AF4, AS1} <: MemoryKernel
    ρ::AF1
    kBT::F
    m::AF1
    Nk::Int
    Ns::Int
    k_array::AF1
    prefactor::AF2
    C::AS1
    P::AF3
    V::AF4
end

function NaiveMultiComponentModeCouplingKernel(ρ, kBT, m, k_array, Sₖ)
    Nk = length(k_array)
    Ns = size(Sₖ[1], 2)
    @assert size(Sₖ) == (Nk,)
    @assert size(Sₖ[1]) == (Ns, Ns)
    @assert size(m) == size(ρ) == (Ns,)

    T = promote_type(eltype(Sₖ[1]), eltype(k_array), eltype(ρ), typeof(kBT), eltype(m))
    ρ, kBT, m = T.(ρ), T(kBT), T.(m)
    k_array = T.(k_array)
    Δk = k_array[2] - k_array[1]
    @assert k_array[1] ≈ Δk / 2
    @assert all(diff(k_array) .≈ Δk)

    ρ_all = sum(ρ)
    x = ρ / sum(ρ)

    S⁻¹ = inv.(Sₖ)
    
    Cₖ = similar(Sₖ)
    δαβ = Matrix(I(Ns))
    for i = 1:Nk
        Cₖ[i] = (δαβ./x - S⁻¹[i]) /ρ_all
    end

    prefactor = zeros(T, Ns, Ns)
    for α = 1:Ns
        for β = 1:Ns
            prefactor[α, β] = ρ_all * kBT / (2 * m[α] * x[β]) * (Δk / 2 / π)^2
        end
    end

    P = zeros(T, Ns, Ns, Nk)

    V = zeros(T, Ns, Ns, Ns, Nk, Nk, Nk)
    for α = 1:Ns, β = 1:Ns, γ = 1:Ns, ik = 1:Nk, iq = 1:Nk, ip = 1:Nk
        q = k_array[iq]
        p = k_array[ip]
        k = k_array[ik]
        δβγ = I[β, γ]
        δαγ = I[α, γ]
        cαγ_q = Cₖ[iq][α, γ]
        cβγ_p = Cₖ[ip][β, γ]
        if abs(iq - ik) + 1 <= ip <= min(Nk, ik + iq - 1)
            V[α, β, γ, iq, ip, ik] = 1 / (2k) * ((k^2 + q^2 - p^2) * δβγ * cαγ_q + (k^2 + p^2 - q^2) * δαγ * cβγ_p)
        end
    end
    kernel = NaiveMultiComponentModeCouplingKernel(ρ, kBT, m, Nk, Ns, k_array, prefactor, Cₖ, P, V)

    return kernel
end

function evaluate_kernel!(out::Diagonal, kernel::NaiveMultiComponentModeCouplingKernel, F::Vector, t)
    V = kernel.V
    k_array = kernel.k_array

    Nk = kernel.Nk
    Ns = kernel.Ns

    prefactor = kernel.prefactor

    @inbounds for α = 1:Ns, β = 1:Ns, μ2 = 1:Ns, μ = 1:Ns, ν2 = 1:Ns, ν = 1:Ns, ik = 1:Nk, iq = 1:Nk, ip = 1:Nk
        kernel.P[α, β, ik] += prefactor[α, β] * k_array[ip] * k_array[iq] * V[μ2, ν2, α, iq, ip, ik] * F[iq][μ2, μ] * F[ip][ν2, ν] * V[μ, ν, β, iq, ip, ik] / k_array[ik]
    end
    for ik = 1:Nk
        @views out.diag[ik] = kernel.P[:, :, ik]
    end
end

function evaluate_kernel(kernel::NaiveMultiComponentModeCouplingKernel, F::Vector, t)
    out = Diagonal(similar(F))
    evaluate_kernel!(out, kernel, F, t)
    return out
end


"""
    evaluate_kernel!(out, kernel::MemoryKernel, F, t)

Evaluates the memory kernel in-place, overwriting the elements of the `out` variable. It may mutate the content of `kernel`.
"""
function evaluate_kernel!(out, kernel::MemoryKernel, F, t)
    error("This memory kernel is not defined")
end


"""
    evaluate_kernel(kernel::MemoryKernel, F, t)
    
Evaluates the memory kernel out-place. It may mutate the content of `kernel`.

Returns
* `out` the kernel evaluated at (F, t)
"""
function evaluate_kernel(kernel::MemoryKernel, F, t)
    error("This memory kernel is not defined")
end