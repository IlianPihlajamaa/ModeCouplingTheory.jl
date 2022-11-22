
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

struct TaggedModeCouplingKernel{F, V, M2, M, T5, FF} <: MemoryKernel
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
    tDict::T5
    F::FF
end

"""
TaggedModeCouplingKernel(ρ, kBT, m, k_array, Sₖ, sol)

Constructor of a Tagged ModeCouplingKernel. It implements the kernel
K(k,t) = ρ kBT / (8π^3m) ∫dq V^2(k,q) F(q,t) Fs(k-q,t)
in which k and q are vectors. 

# Arguments:

* `ρ`: number density
* `kBT`: Thermal energy
* `m` : particle mass
* `k_array`: vector of wavenumbers at which the structure factor is known
* `Sₖ`: vector with the elements of the structure factor 
* `sol`: a solution object of an equation with a ModeCouplingKernel.

# Returns:

an instance `k` of `TaggedModeCouplingKernel <: MemoryKernel`, which can be called both in-place and out-of-place:
`k`(out, F, t)
out = `k`(F, t)
"""
function TaggedModeCouplingKernel(ρ, kBT, m, k_array, Sₖ, sol)
    tDict = Dict(zip(sol.t, eachindex(sol.t)))
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
            cq = Cₖ[iq]
            V1[iq, ip] = p * q * (cq)^2 / 4 * D₀ * ρ / (4 * π^2) * Δk ^ 2
            V2[iq, ip] = p * q * (q^2 - p^2)^2 * (cq)^2 / 4 * D₀ * ρ / (4 * π^2) * Δk^ 2
            V3[iq, ip] = p * q * (q^2 - p^2) * (cq^2) / 2 * D₀ * ρ / (4 * π^2) * Δk ^ 2
        end
    end
    kernel = TaggedModeCouplingKernel(ρ, kBT, m, Nk, k_array, A1, A2, A3, T1, T2, T3, V1, V2, V3, tDict, sol.F)

    return kernel
end

function fill_A!(kernel::TaggedModeCouplingKernel, F, t)
    A1 = kernel.A1
    A2 = kernel.A2
    A3 = kernel.A3
    V1 = kernel.V1
    V2 = kernel.V2
    V3 = kernel.V3
    Nk = kernel.Nk
    it = kernel.tDict[t]
    Fc = kernel.F[it]
    @turbo for iq = 1:Nk
        for ip = 1:Nk
            fq = Fc[iq]
            fp = F[ip]
            f4 = fp * fq
            A1[iq, ip] = V1[iq, ip] * f4
            A2[iq, ip] = V2[iq, ip] * f4
            A3[iq, ip] = V3[iq, ip] * f4
        end
    end
end

function evaluate_kernel!(out::Diagonal, kernel::TaggedModeCouplingKernel, Fs, t)
    A1 = kernel.A1
    A2 = kernel.A2
    A3 = kernel.A3
    T1 = kernel.T1
    T2 = kernel.T2
    T3 = kernel.T3
    k_array = kernel.k_array

    Nk = kernel.Nk
    fill_A!(kernel, Fs, t)
    bengtzelius3!(T1, T2, T3, A1, A2, A3, Nk)

    @inbounds for ik = 1:Nk
        k = k_array[ik]
        out.diag[ik] = k * kernel.T1[ik] + kernel.T2[ik] / k^3 + kernel.T3[ik] / k
    end
end

function evaluate_kernel(kernel::TaggedModeCouplingKernel, Fs, t)
    out = Diagonal(similar(Fs)) # we need it to produce a diagonal matrix
    evaluate_kernel!(out, kernel, Fs, t) # call the inplace version
    return out
end