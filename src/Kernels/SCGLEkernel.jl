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
Δζ(t) = kBT / (36 π ϕ) ∫dq q⁴ M²(k,q) Fs(q,t) F(q,t)
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
    Δk = k_array[2] - k_array[1]
    kc = 1.305*2π
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

#############
#   MSD     #
#############
struct MSDSCGLEKernel{TDICT,T} <: MemoryKernel
    tDict::TDICT
    Δζ::T
end

"""
MSDSCGLEKernel(ϕ, D⁰, k_array, Sₖ, sol)

Constructor of a MSDSCGLEKernel. It implements the kernel

K(k,t) = D⁰ / (36πϕ) ∫dq q^4 c(q)^2 F(q,t) Fs(q,t)

where the integration runs from 0 to infinity. F and Fs are the coherent
and incoherent intermediate scattering functions, and must be passed in
as solutions of the corresponding equations.

# Arguments:

* `ϕ`: volume fraction
* `D⁰`: Short times diffusion coefficient
* `k_array`: vector of wavenumbers at which the structure factor is known
* `Sₖ`: vector with the elements of the structure factor 
* `sol`: a solution object of an equation with a SCGLEKernel.

# Returns:

an instance `k` of `MSDSCGLEKernel <: MemoryKernel`, which can be evaluated like:
`k = evaluate_kernel(kernel, F, t)`
"""
function MSDSCGLEKernel_constructor(sol, k_array)
    Δζ = []
    t = sol.t
    Nt = length(t)
    kc = 1.305*2π
    λ = 1/(1+(k_array[1]/kc)^2)
    K = get_K(sol)
    for i in 1:Nt
        append!(Δζ, K[i][1,1]/λ)
    end
    tDict = Dict(zip(t, eachindex(t)))
    kernel = MSDSCGLEKernel(tDict, Δζ)
    return kernel
end

function evaluate_kernel(kernel::MSDSCGLEKernel, MSD, t)
    it = kernel.tDict[t]
    return kernel.Δζ[it]
end

function MSD(k::Vector{Float64}, sol, solver)
    MSD0 = 0.0; dMSD0 = 0.0; α = 0.0; β = 1.0; γ = 0.0; δ = -1.0;
    msdkernel = MSDSCGLEKernel_constructor(sol, k)
    msdequation = MemoryEquation(α, β, γ, δ, MSD0, dMSD0, msdkernel)
    msdsol = solve(msdequation, solver)
    return msdsol.t, msdsol.F, msdkernel.Δζ
end

@doc"""
ΔG = (kBT/60π²)∫dk k⁴[(1/S)(∂S/∂k)]²[(F/S)]²
just the G. Naegele formula for shear viscosity relaxation. It must works for the MCT kernel.
"""
function get_ΔG(sol, k_array, Sₖ)
    t = sol.t
    Nk = div(length(k_array),2)
    Δk = k_array[2] - k_array[1]
    k⁴ = k_array[Nk+1:end-1].^4
    ∂S = diff(Sₖ[Nk+1:end])./diff(k_array[Nk+1:end])
    S = Sₖ[Nk+1:end-1]
    F = sol.F
    ΔG = []
    for i in 1:length(t)
        dG = sum(k⁴.*((∂S./S).^2).*((F[i][Nk+1:end-1]./S).^2))
        dG *= Δk/(60*π*π)
        append!(ΔG, dG)
    end
    return ΔG
end

