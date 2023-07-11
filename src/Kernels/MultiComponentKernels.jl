
function bengtzelius3!(T1, T2, T3, A1, A2, A3, Nk, Ns)
    @assert size(T1) == size(T2) == size(T3) == (Nk,)
    @assert size(T1[1]) == size(T2[1]) == size(T3[1]) == (Ns, Ns)
    @assert size(A1) == size(A2) == size(A3) == (Ns, Ns, Nk, Nk)

    T01 = zero(eltype(T1))
    T02 = zero(eltype(T2))
    T03 = zero(eltype(T3))
    @inbounds @views for iq = 1:Nk
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
        for iq = 1:(Nk-ik+1)
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

struct MultiComponentModeCouplingKernel3D{F,AF1,AF2,AF3,AF4} <: MemoryKernel
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
    MultiComponentModeCouplingKernel(ρ, kBT, m, k_array, Sₖ; dims=3)

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
`evaluate_kernel!(out, kernel, F, t)`
`out = evaluate_kernel(kernel, F, t)`
"""
function MultiComponentModeCouplingKernel(ρ, kBT, m, k_array, Sₖ; dims=3)
    if dims != 3
        error("MSD is not implemented for dimentions other than 3")
    end
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
        Cₖ[i] = (δαβ ./ x - S⁻¹[i]) / ρ_all
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
            prefactor[α, β] = ρ_all * kBT / (8 * m[α] * x[β]) * (Δk / 2 / π)^2
        end
    end

    kernel = MultiComponentModeCouplingKernel3D(ρ, kBT, m, Nk, Ns, k_array, prefactor, Cₖ, A1, A2, A3, T1, T2, T3)
    return kernel
end


function fill_A!(kernel::MultiComponentModeCouplingKernel3D, F)

    Nk = kernel.Nk
    Ns = size(F[1], 1)
    k_array = kernel.k_array
    C = kernel.C
    prefactor = kernel.prefactor

    A1 = kernel.A1
    A2 = kernel.A2
    A3 = kernel.A3
    x = kernel.ρ/sum(kernel.ρ)
    m = kernel.m
    @fastmath @inbounds for α = 1:Ns
        for β = 1:Ns
            # if β > α 
            #     continue
            # end
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
                    A_prefactor1 = p * q
                    A_prefactor2 = p * q * (p^2 - q^2)^2
                    A_prefactor3 = 2 * p * q * (p^2 - q^2)
                    # We sum over 2 dummy indices. dummy1 is always the primed variable (mu', nu') 
                    # and dummy2 is the unprimed counterpart.
                    fqab = Fq[α, β]
                    fpab = Fp[α, β]
                    for dummy1 = 1:Ns
                        cp1a = Cp[dummy1, α]
                        cq1a = Cq[dummy1, α]
                        fq1b = Fq[dummy1, β]
                        fp1b = Fp[dummy1, β]
                        for dummy2 = 1:Ns
                            V_dummy1dummy2αβ_pp = cp1a * Cp[dummy2, β]
                            V_dummy1dummy2αβ_pq = cp1a * Cq[dummy2, β]
                            V_dummy1dummy2αβ_qp = cq1a * Cp[dummy2, β]
                            V_dummy1dummy2αβ_qq = cq1a * Cq[dummy2, β]
                            term1 = V_dummy1dummy2αβ_pp * fqab * Fp[dummy1, dummy2]
                            term2 = V_dummy1dummy2αβ_pq * Fq[α, dummy2] * fp1b
                            term3 = V_dummy1dummy2αβ_qp * fq1b * Fp[α, dummy2]
                            term4 = V_dummy1dummy2αβ_qq * Fq[dummy1, dummy2] * fpab
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

function evaluate_kernel!(out::Diagonal, kernel::MultiComponentModeCouplingKernel3D, F::Vector, t)
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

function evaluate_kernel(kernel::MultiComponentModeCouplingKernel3D, F::Vector, t)
    out = Diagonal(similar(F))
    evaluate_kernel!(out, kernel, F, t)
    return out
end

struct TaggedMultiComponentModeCouplingKernel3D{F,V,M2,M,T5,FF, V1, FFF} <: MemoryKernel
    s::Int
    ρ::V1
    kBT::F
    m::V1
    Nk::Int
    k_array::V
    Ck::FF
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
    F::FFF
end

"""
TaggedMultiComponentModeCouplingKernel(ρ, kBT, m, k_array, Sₖ, sol; dims=3)

Constructor of a Tagged ModeCouplingKernel. It implements the kernel
K(k,t) = ρ kBT / (8π^3 mₛ) Σαβ ∫dq V^2sαβ(k,q) Fαβ(q,t) Fₛ(k-q,t)
in which k and q are vectors. 

# Arguments:

* `s`: index of the sepcies to tag
* `ρ`: number density
* `kBT`: Thermal energy
* `m` : particle mass
* `k_array`: vector of wavenumbers at which the structure factor is known
* `Sₖ`: vector with the elements of the structure factor 
* `sol`: a solution object of an equation with a MultiComponentModeCouplingKernel.

# Returns:

an instance `k` of `TaggedMultiComponentModeCouplingKernel <: MemoryKernel`, which can be called both in-place and out-of-place:
`evaluate_kernel!(out, kernel, Fs, t)`
`out = evaluate_kernel(kernel, Fs, t)`
"""
function TaggedMultiComponentModeCouplingKernel(s::Int, ρ, kBT, m, k_array, Sₖ, sol; dims=3)
    if dims != 3
        error("MSD is not implemented for dimentions other than 3")
    end
    tDict = Dict(zip(sol.t, eachindex(sol.t)))
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
        Cₖ[i] = (δαβ ./ x - S⁻¹[i]) / ρ_all
    end

    T1 = similar(k_array)
    T2 = similar(k_array)
    T3 = similar(k_array)
    A1 = similar(k_array, (Nk, Nk))
    A2 = similar(k_array, (Nk, Nk))
    A3 = similar(k_array, (Nk, Nk))
    V1 = similar(k_array, (Nk, Nk))
    V2 = similar(k_array, (Nk, Nk))
    V3 = similar(k_array, (Nk, Nk)) 
    prefactor = kBT * Δk^2 * ρ_all / (4 * m[s] * (2π)^2)
    for iq = 1:Nk
        for ip = 1:Nk

            p = k_array[ip]
            q = k_array[iq]

            V1[iq, ip] = prefactor * p * q
            V2[iq, ip] = prefactor * p * q * (q^2 - p^2)^2
            V3[iq, ip] = prefactor * 2 * p * q * (q^2 - p^2)

        end
    end
    # converting c and F to base arrays so that LoopVectorization can use them
    c = zeros(T, Ns, Ns, Nk)
    Fc = [zeros(T, Ns, Ns, Nk) for i in eachindex(sol.F)]
    for a=1:Ns, b = 1:Ns, ik=1:Nk
        c[a,b,ik] = Cₖ[ik][a,b]
        for i = eachindex(sol.F)
            Fc[i][a,b,ik] = sol.F[i][ik][a,b]
        end
    end
    # c = reshape(reinterpret(reshape, eltype(eltype(Cₖ)), Cₖ),Ns,Ns,Nk)
    # Fc = [reshape(reinterpret(reshape, eltype(eltype(F)), F),Ns,Ns,Nk) for F in sol.F]
    kernel = TaggedMultiComponentModeCouplingKernel3D(s, ρ, kBT, m, Nk, k_array, c, A1, A2, A3, T1, T2, T3, V1, V2, V3, tDict, Fc)

    return kernel
end

function fill_A!(kernel::TaggedMultiComponentModeCouplingKernel3D, F, t)
    A1 = kernel.A1
    A2 = kernel.A2
    A3 = kernel.A3
    V1 = kernel.V1
    V2 = kernel.V2
    V3 = kernel.V3
    Nk = kernel.Nk
    it = kernel.tDict[t]
    Fc = kernel.F[it]
    Ns = size(Fc, 1)
    Ck = kernel.Ck
    s = kernel.s
    @turbo for iq = 1:Nk
        for ip = 1:Nk
            fp = F[ip]
            A1new = zero(eltype(A1))
            A2new = zero(eltype(A2))
            A3new = zero(eltype(A3))
            for α = 1:Ns
                for β = 1:Ns
                    csα_q = Ck[s, α, iq]
                    csβ_q = Ck[s, β, iq]
                    fq = Fc[α, β, iq]
                    f4c2 = fp * fq * csα_q * csβ_q
                    A1new += V1[iq, ip] * f4c2
                    A2new += V2[iq, ip] * f4c2
                    A3new += V3[iq, ip] * f4c2
                end
            end
            A1[iq, ip] = A1new
            A2[iq, ip] = A2new
            A3[iq, ip] = A3new
        end
    end
end

function evaluate_kernel!(out::Diagonal, kernel::TaggedMultiComponentModeCouplingKernel3D, Fs, t)
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

function evaluate_kernel(kernel::TaggedMultiComponentModeCouplingKernel3D, Fs, t)
    out = Diagonal(similar(Fs)) # we need it to produce a diagonal matrix
    evaluate_kernel!(out, kernel, Fs, t) # call the inplace version
    return out
end




struct MSDMultiComponentModeCouplingKernel3D{F,V,TDICT,FF, V1, FFF, FS} <: MemoryKernel
    s::Int
    ρ::V1
    kBT::F
    m::V1
    Nk::Int
    k_array::V
    Ck::FF
    tDict::TDICT
    F::FFF
    Fs::FS
end

"""
MSDMultiComponentModeCouplingKernel(s, ρ, kBT, m, k_array, Sₖ, sol, taggedsol; dims=3)

Constructor of a MSDModeCouplingKernel. It implements the kernel
Kₛ(k,t) = ρ kBT / (6π^2mₛ) Σαβ ∫dq q^4 csα(q)csβ(q) Fαβ(q,t) Fₛ(q,t)
where the integration runs from 0 to infinity. F and Fs are the coherent
and incoherent intermediate scattering functions, and must be passed in
as solutions of the corresponding equations.

# Arguments:

* `ρ`: number density
* `kBT`: Thermal energy
* `m` : particle mass of species s
* `k_array`: vector of wavenumbers at which the structure factor is known
* `Sₖ`: vector with the elements of the structure factor 
* `sol`: a solution object of an equation with a MultiComponentModeCouplingKernel.
* `taggedsol`: a solution object of an equation with a TaggedMultiComponentModeCouplingKernel.

# Returns:

an instance `k` of `MSDMultiComponentModeCouplingKernel <: MemoryKernel`, which can be evaluated like:
`k = evaluate_kernel(kernel, F, t)`
"""
function MSDMultiComponentModeCouplingKernel(s::Int, ρ, kBT, m, k_array, Sₖ, sol, taggedsol; dims=3)
    if dims != 3
        error("MSD is not implemented for dimentions other than 3")
    end
    tDict = Dict(zip(sol.t, eachindex(sol.t)))
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
        Cₖ[i] = (δαβ ./ x - S⁻¹[i]) / ρ_all
    end

    kernel = MSDMultiComponentModeCouplingKernel3D(s, ρ, kBT, m, Nk, k_array, Cₖ, tDict, sol.F, taggedsol.F)
    return kernel
end

function evaluate_kernel(kernel::MSDMultiComponentModeCouplingKernel3D, MSD, t)
    s = kernel.s
    K = zero(typeof(MSD))
    it = kernel.tDict[t]
    k_array = kernel.k_array
    Ck = kernel.Ck
    Δk = k_array[2] - k_array[1]
    F = kernel.F[it]
    Fs = kernel.Fs[it]
    Ns = length(Fs[1])
    ρ_all = sum(kernel.ρ)
    for α = 1:Ns, β = 1:Ns
        for iq in eachindex(k_array)
            K += k_array[iq]^4*Ck[iq][α,s]*Ck[iq][s,β]*F[iq][α,β]*Fs[iq]
        end
    end
    K *= Δk*ρ_all*kernel.kBT/(6π^2*kernel.m[s])
    return K
end