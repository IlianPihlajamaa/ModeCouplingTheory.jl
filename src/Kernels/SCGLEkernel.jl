struct SCGLEKernel <: MemoryKernel
    k::Vector{Float64}
    λ::Vector{Float64}
    M::Vector{Float64}
    prefactor::Float64
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
    Δζ = 0.0
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

"""
Usage:

using ModeCouplingTheory

# defining the structure factor
function find_analytical_C_k(k, η)
    A = -(1 - η)^-4 *(1 + 2η)^2
    B = (1 - η)^-4*  6η*(1 + η/2)^2
    D = -(1 - η)^-4 * 1/2 * η*(1 + 2η)^2
    Cₖ = @. 4π/k^6 * 
    (
        24*D - 2*B * k^2 - (24*D - 2 * (B + 6*D) * k^2 + (A + B + D) * k^4) * cos(k)
     + k * (-24*D + (A + 2*B + 4*D) * k^2) * sin(k)
     )
    return Cₖ
end

function find_analytical_S_k(k, η)
        Cₖ = find_analytical_C_k(k, η)
        ρ = 6/π * η
        Sₖ = @. 1 + ρ*Cₖ / (1 - ρ*Cₖ)
    return Sₖ
end

# Appling the Verlet-Weiss correction
# [1] Loup Verlet and Jean-Jacques Weis. Phys. Rev. A 5, 939 – Published 1 February 1972
ϕ_VW(ϕ :: Float64) = ϕ*(1.0 - (ϕ / 16.0))
k_VW(ϕ :: Float64, k :: Float64) = k*((ϕ_VW(ϕ)/ϕ)^(1.0/3.0))

# initial setup

∂F0 = zeros(2*Nk); α = 0.0; β = 1.0; γ = @. k^2/S; δ = 0.0

kernel = SCGLEKernel(ϕ, k, S);
equation = MemoryEquation(α, β, γ, δ, S, ∂F0, kernel);
solver = TimeDoublingSolver(Δt=10^-5, t_max=10.0^10, 
    N = 8, tolerance=10^-8, verbose=true);
sol = @time solve(equation, solver);
using Plots
p = plot(xlabel="log10(t)", ylabel="Fs(k,t)", ylims=(0,1))
for ik = [7, 18, 25, 39]
    Fk = get_F(sol, 1:10:800, ik)
    t = get_t(sol)[1:10:800]
    plot!(p, log10.(t), Fk/S[ik], label="k = $(k_array[ik])", lw=3)
end
"""
