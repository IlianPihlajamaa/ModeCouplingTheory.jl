
struct ModeCouplingKernel3D{F,V,M, M2} <: MemoryKernel
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

struct dDimModeCouplingKernel{I, F, AF1, AF3} <: MemoryKernel
    d::I
    ρ::F
    kBT::F
    m::F
    Nk::I
    k_array::AF1
    prefactor::F
    C::AF1
    Sk::AF1
    V::AF3
    J::AF3
    P::AF1
end

function dDimModeCouplingKernel(ρ, kBT, m, k_array, Sₖ, d)
    Nk = length(k_array)

    Δk = k_array[2] - k_array[1]
    @assert k_array[1] ≈ Δk / 2 "The first grid point must be exactly half the grid spacing."
    @assert all(diff(k_array) .≈ Δk) "The grid must be equidistant"

    S⁻¹ = inv.(Sₖ)
    Cₖ = similar(Sₖ)

    for i = 1:Nk
        Cₖ[i] = (1 - S⁻¹[i]) / ρ
    end

    prefactor = (kBT / m) * ρ * (Δk)^2 * surface_d_dim_unit_sphere(d-1) / (4*pi)^d
    P = zeros(Nk)
    V = zeros(Nk, Nk, Nk)
    J = zeros(Nk, Nk, Nk)

    for iq = 1:Nk, ik = 1:Nk, ip = 1:Nk
        q = k_array[iq]
        p = k_array[ip]
        k = k_array[ik]

        c_p = Cₖ[ip]
        c_k = Cₖ[ik]
        S_q = Sₖ[iq]

        if abs(iq - ik) + 1 <= ip <= min(Nk, ik + iq - 1) 
            J[iq, ik, ip] = k * p * (4*q^2*k^2 - (q^2+k^2-p^2)^2)^((d-3)/2) / q^d
            V[iq, ik, ip] = ((q^2+k^2-p^2)*c_k + (q^2-k^2+p^2)*c_p)^2
        end
    end

    kernel = dDimModeCouplingKernel(d, ρ, kBT, m, Nk, k_array, prefactor, Cₖ, Sₖ, V, J, P)
    
    return kernel
end

"""
    ModeCouplingKernel(ρ, kBT, m, k_array, Sₖ; dims=3)

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
`evaluate_kernel!(out, kernel, F, t)`
`out = evaluate_kernel(kernel, F, t)`
"""
function ModeCouplingKernel(ρ, kBT, m, k_array, Sₖ; dims=3)
    if dims != 3
        return dDimModeCouplingKernel(ρ, kBT, m, k_array, Sₖ, dims)
    end
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
    kernel = ModeCouplingKernel3D(ρ, kBT, m, Nk, k_array, A1, A2, A3, T1, T2, T3, V1, V2, V3)
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

function fill_A!(kernel::ModeCouplingKernel3D, F)
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


function evaluate_kernel!(out::Diagonal, kernel::ModeCouplingKernel3D, F::Vector, t)
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


function evaluate_kernel(kernel::ModeCouplingKernel3D, F::Vector, t)
    out = Diagonal(similar(kernel.T1))
    evaluate_kernel!(out, kernel, F, t)
    return out
end

function evaluate_kernel!(out::Diagonal, kernel::dDimModeCouplingKernel, F::Vector, t)
    V = kernel.V
    J = kernel.J
    Sk = kernel.Sk
    Nk = kernel.Nk
    prefactor = kernel.prefactor
    kernel.P .= zero(eltype(kernel.P)) 
    
    @turbo for iq = 1:Nk 
        for ik = 1:Nk
            for ip = 1:Nk
                kernel.P[iq] += prefactor * J[iq, ik, ip] * V[iq, ik, ip] * F[ik] * F[ip]
            end
        end
    end

    for ik = 1:Nk
        out.diag[ik] = kernel.P[ik]
    end
end

function evaluate_kernel(kernel::dDimModeCouplingKernel, F::Vector, t)
    out = Diagonal(similar(F))
    evaluate_kernel!(out, kernel, F, t)
    return out
end


struct TaggedModeCouplingKernel3D{F, V, M2, M, T5, FF} <: MemoryKernel
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

struct dDimTaggedModeCouplingKernel{I, F, AF1, AF3, T, sol} <: MemoryKernel
    d::I
    ρ::F
    kBT::F
    m::F
    Nk::I
    tDict::T
    k_array::AF1
    prefactor::F
    C::AF1
    Sk::AF1
    sol_col::sol
    V::AF3
    J::AF3
    P::AF1
end

function dDimTaggedModeCouplingKernel(d, ρ, kBT, m, k_array, Sₖ, sol_col)

    tDict = Dict(zip(sol_col.t, eachindex(sol_col.t)))

    Nk = length(k_array)

    T = promote_type(eltype(Sₖ), eltype(k_array), typeof(ρ), typeof(kBT), typeof(m))
    ρ, kBT, m = T(ρ), T(kBT), T(m)

    Δk = k_array[2] - k_array[1]
    @assert k_array[1] ≈ Δk / 2
    @assert all(diff(k_array) .≈ Δk)

    S⁻¹ = inv.(Sₖ)
    Cₖ = similar(Sₖ)

    for i = 1:Nk
        Cₖ[i] = (1 - S⁻¹[i]) / ρ
    end

    prefactor = 2 * (kBT/m) * ρ * (Δk)^2 * surface_d_dim_unit_sphere(d-1) / (4*pi)^d
    P = zeros(T, Nk)
    V = zeros(T, Nk, Nk, Nk)
    J = zeros(T, Nk, Nk, Nk)

    for ik = 1:Nk, iq = 1:Nk, ip = 1:Nk
        q = k_array[iq]
        p = k_array[ip]
        k = k_array[ik]

        c_p = Cₖ[ip]
        c_k = Cₖ[ik]

        if abs(iq - ik) + 1 <= ip <= min(Nk, ik + iq - 1) 
            J[iq, ik, ip] = k * p * (4*q^2*k^2 - (q^2+k^2-p^2)^2)^((d-3)/2) / q^d
            V[iq, ik, ip] = ((q^2+k^2-p^2)*c_k)^2
        end
    end
    kernel = dDimTaggedModeCouplingKernel(d, ρ, kBT, m, Nk, tDict, k_array, prefactor, Cₖ, Sₖ, sol_col, V, J, P)
    return kernel
end

function evaluate_kernel!(out::Diagonal, kernel::dDimTaggedModeCouplingKernel, F::Vector, t)
    V = kernel.V
    J = kernel.J
    Sk = kernel.Sk
    Nk = kernel.Nk
    prefactor = kernel.prefactor
    it = kernel.tDict[t]
    F_col = get_F(kernel.sol_col, it)

    kernel.P .= 0 
    for iq = 1:Nk, ik = 1:Nk, ip = 1:Nk
        kernel.P[iq] += prefactor * J[iq, ik, ip] * V[iq, ik, ip] * F_col[ik] * F[ip]
    end

    for ik = 1:Nk
        @views out.diag[ik] = kernel.P[ik]
    end
end

function evaluate_kernel(kernel::dDimTaggedModeCouplingKernel, F::Vector, t)
    out = Diagonal(similar(F))
    evaluate_kernel!(out, kernel, F, t)
    return out
end

"""
TaggedModeCouplingKernel(ρ, kBT, m, k_array, Sₖ, sol; dims=3)

Constructor of a Tagged ModeCouplingKernel. It implements the kernel
K(k,t) = ρ kBT / (8π^3m) ∫dq V^2(k,q) F(q,t) Fs(k-q,t)
in which k and q are vectors. Here V(k,q) = c(q) (k dot q)/k. 

# Arguments:

* `ρ`: number density
* `kBT`: Thermal energy
* `m` : particle mass
* `k_array`: vector of wavenumbers at which the structure factor is known
* `Sₖ`: vector with the elements of the structure factor 
* `sol`: a solution object of an equation with a ModeCouplingKernel.

# Returns:

an instance `k` of `TaggedModeCouplingKernel <: MemoryKernel`, which can be called both in-place and out-of-place:
`evaluate_kernel!(out, kernel, Fs, t)`
`out = evaluate_kernel(kernel, Fs, t)`
"""
function TaggedModeCouplingKernel(ρ, kBT, m, k_array, Sₖ, sol; dims=3)
    if dims != 3
        return dDimTaggedModeCouplingKernel(dims, ρ, kBT, m, k_array, Sₖ, sol)
    end
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
    kernel = TaggedModeCouplingKernel3D(ρ, kBT, m, Nk, k_array, A1, A2, A3, T1, T2, T3, V1, V2, V3, tDict, sol.F)

    return kernel
end

function fill_A!(kernel::TaggedModeCouplingKernel3D, F, t)
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

function evaluate_kernel!(out::Diagonal, kernel::TaggedModeCouplingKernel3D, Fs, t)
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

function evaluate_kernel(kernel::TaggedModeCouplingKernel3D, Fs, t)
    out = Diagonal(similar(Fs)) # we need it to produce a diagonal matrix
    evaluate_kernel!(out, kernel, Fs, t) # call the inplace version
    return out
end

struct MSDModeCouplingKernel3D{F, V, TDICT, FF, FS} <: MemoryKernel
    ρ::F
    kBT::F
    m::F
    Nk::Int
    k_array::V
    Ck::V
    tDict::TDICT
    F::FF
    Fs::FS
end

struct dDimMSDModeCouplingKernel{I, F, T, AF1, sol1, sol2} <: MemoryKernel
    d::I
    ρ::F
    kBT::F
    m::F
    Nk::I
    tDict::T
    k_array::AF1
    prefactor::F
    C::AF1
    Sk::AF1
    sol_col::sol1
    sol_tagged::sol2
end

function dDimMSDModeCouplingKernel(d, ρ, kBT, m, k_array, Sₖ, sol_col, sol_tagged)
    
    tDict = Dict(zip(sol_col.t, eachindex(sol_col.t)))

    Nk = length(k_array)

    Δk = k_array[2] - k_array[1]
    @assert k_array[1] ≈ Δk / 2
    @assert all(diff(k_array) .≈ Δk)

    S⁻¹ = inv.(Sₖ)
    Cₖ = similar(Sₖ)

    for i = 1:Nk
        Cₖ[i] = (1 - S⁻¹[i]) / ρ
    end

    prefactor = (kBT/m) * ρ * Δk * surface_d_dim_unit_sphere(d) / (d * (2*pi)^d)

    kernel = dDimMSDModeCouplingKernel(d, ρ, kBT, m, Nk, tDict, k_array, prefactor, Cₖ, Sₖ, sol_col, sol_tagged)
    
    return kernel
end



"""
MSDModeCouplingKernel(ρ, kBT, m, k_array, Sₖ, sol, taggedsol; dims=3)

Constructor of a MSDModeCouplingKernel. It implements the kernel
K(k,t) = ρ kBT / (6π^2m) ∫dq q^4 c(q)^2 F(q,t) Fs(q,t)
where the integration runs from 0 to infinity. F and Fs are the coherent
and incoherent intermediate scattering functions, and must be passed in
as solutions of the corresponding equations.

# Arguments:

* `ρ`: number density
* `kBT`: Thermal energy
* `m` : particle mass
* `k_array`: vector of wavenumbers at which the structure factor is known
* `Sₖ`: vector with the elements of the structure factor 
* `sol`: a solution object of an equation with a ModeCouplingKernel.
* `taggedsol`: a solution object of an equation with a TaggedModeCouplingKernel.

# Returns:

an instance `k` of `MSDModeCouplingKernel <: MemoryKernel`, which can be evaluated like:
`k = evaluate_kernel(kernel, F, t)`
"""
function MSDModeCouplingKernel(ρ, kBT, m, k_array, Sₖ, sol, taggedsol; dims=3)
    if dims != 3
        return dDimMSDModeCouplingKernel(dims, ρ, kBT, m, k_array, Sₖ, sol, taggedsol)
    end
    tDict = Dict(zip(sol.t, eachindex(sol.t)))
    Nk = length(k_array)
    T = promote_type(eltype(Sₖ), eltype(k_array), typeof(ρ), typeof(kBT), typeof(m))
    ρ, kBT, m = T(ρ), T(kBT), T(m)
    k_array, Sₖ = T.(k_array), T.(Sₖ)
    Δk = k_array[2] - k_array[1]
    @assert k_array[1] ≈ Δk / 2
    @assert all(diff(k_array) .≈ Δk)
    Cₖ = @. (Sₖ - 1) / (ρ * Sₖ)
    kernel = MSDModeCouplingKernel3D(ρ, kBT, m, Nk, k_array, Cₖ, tDict, sol.F, taggedsol.F)
    return kernel
end

function evaluate_kernel(kernel::MSDModeCouplingKernel3D, MSD, t)
    K = zero(typeof(MSD))
    k_array = kernel.k_array
    Ck = kernel.Ck
    it = kernel.tDict[t]
    Δk = k_array[2] - k_array[1]
    F = kernel.F[it]
    Fs = kernel.Fs[it]
    for iq in eachindex(k_array)
        K += k_array[iq]^4*Ck[iq]^2*F[iq]*Fs[iq]
    end
    K *= Δk*kernel.ρ*kernel.kBT/(6π^2*kernel.m)
    return K
end

function evaluate_kernel(kernel::dDimMSDModeCouplingKernel, F, t)

    K = zero(typeof(F))
    k_array = kernel.k_array
    Ck = kernel.C
    it = kernel.tDict[t]
    d = kernel.d
    
    F = get_F(kernel.sol_col, it)
    Fs = get_F(kernel.sol_tagged, it)
    
    for iq in eachindex(k_array)
        K += k_array[iq]^(d+1) * Ck[iq]^2 * F[iq] * Fs[iq]
    end
    
    K *= kernel.prefactor
    return K
end


