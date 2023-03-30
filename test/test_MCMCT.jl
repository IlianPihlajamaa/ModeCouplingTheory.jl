import ModeCouplingTheory.MemoryKernel
import ModeCouplingTheory.evaluate_kernel
import ModeCouplingTheory.evaluate_kernel!


struct NaiveMultiComponentModeCouplingKernel{F,AF1,AF2,AF3,AF4,AS1} <: MemoryKernel
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
        Cₖ[i] = (δαβ ./ x - S⁻¹[i]) / ρ_all
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
    kernel.P .*= 0.0
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
    find_direct_correlation_function_PY(K, diameters, ρ)

returns the exact solution of the Percus Yevick approximation to the direct correlation function for a multicomponent mixture.

ARGS:
    K: wave number (Float64)
    diameters: Vector with diameters of the different species
    ρ: Vector with the number densities of the different species

the vectors `diameters` and `ρ` must have the same length.

ref: Baxter, R.J. Ornstein–Zernike Relation and Percus–Yevick Approximation for Fluid Mixtures, J. Chem. Phys. 52, 4559 (1970)
"""
function find_direct_correlation_function_PY(K, diameters, ρ)
    p = length(diameters)
    @assert p == length(ρ)
    d = (diameters .+ diameters') / 2
    s = (diameters .- diameters') / 2

    ξ = [π / 6 * sum(ρ[j] * d[j, j]^ν for j = 1:p) for ν in 1:3]
    a = [(1 - ξ[3])^(-2) * (1 - ξ[3] + 3 * ξ[2] * d[i, i]) for i = 1:p]
    b = [-3 / 2 * d[i, i]^2 * (1 - ξ[3])^(-2) * ξ[2] for i = 1:p]
    q(r, i, k) = 1 / 2 * a[i] * (r^2 - d[i, k]^2) + b[i] * (r - d[i, k])
    Q̃ = [I[i, k] - 2π * sqrt(ρ[i] * ρ[k]) * quadgk(r -> q(r, i, k) * cis(K * r), s[i, k], d[i, k])[1] for i = 1:p, k = 1:p]

    C = I - real.(Q̃' * Q̃)
    C ./= sqrt.(ρ .* ρ')
    return C
end

function find_structure_factor_PY(K, diameters, ρ)
    x = ρ / sum(ρ)
    c = find_direct_correlation_function_PY(K, diameters, ρ)
    p = length(ρ)
    δ = Matrix{Float64}(I, p, p)
    S = inv(δ ./ x - sum(ρ) * c)
    return S
end

Ns = 2
Nk = 50
ϕ = 0.55
kBT = 1.0
m = ones(Ns)
particle_diameters = [0.8, 1.0]
concentration_ratio = [0.2, 0.8]
concentration_ratio ./= sum(concentration_ratio)
ρ_all = 6ϕ / (π * sum(concentration_ratio .* particle_diameters .^ 3))
ρ = ρ_all * concentration_ratio
dk = 40 / Nk
k_array = dk * (collect(1:Nk) .- 0.5)
x = ρ / sum(ρ)

Sₖ = [SMatrix{Ns,Ns}(find_structure_factor_PY(k_array[i], particle_diameters, ρ)) for i = 1:Nk]
S⁻¹ = inv.(Sₖ)
J = similar(Sₖ) .* 0.0
for ik = 1:Nk
    J[ik] = kBT * k_array[ik]^2 * x ./ m .* I(Ns)
end

F₀ = copy(Sₖ)
∂ₜF₀ = [@SMatrix(zeros(Ns, Ns)) for i = 1:Nk]
α = 1.0
β = 0.0
γ = similar(Sₖ)
γ .*= 0.0
for ik = 1:Nk
    γ .= J .* S⁻¹
end
δ = @SMatrix zeros(Ns, Ns)

kernel = MultiComponentModeCouplingKernel(ρ, kBT, m, k_array, Sₖ)
kernelnaive = NaiveMultiComponentModeCouplingKernel(ρ, kBT, m, k_array, Sₖ);
Ftest = rand(eltype(F₀), Nk)
@test all(evaluate_kernel(kernelnaive, Ftest, 0.0) .≈ evaluate_kernel(kernel, Ftest, 0.0))
system = MemoryEquation(α, β, γ, δ, F₀, ∂ₜF₀, kernel)
solverFuchs = TimeDoublingSolver(N=4, tolerance=10^-12, max_iterations=20000, Δt=10^-4, t_max=10.0^3, verbose=false)
sol =  solve(system, solverFuchs);
tFuchs, FFuchs = sol.t, sol.F

ik = 19

s = 2
α = 1.0
β = 0.0
γ = [kBT * k_array[ik]^2 ./ m[s] for ik = 1:Nk]
F0 = [1.0 for ik = 1:Nk]
dF0 = [0.0 for ik = 1:Nk]
δ = 0.0

taggedkernel = TaggedMultiComponentModeCouplingKernel(s, ρ, kBT, m, k_array, Sₖ, sol);
taggedSystem = MemoryEquation(α, β, γ, δ, F0, dF0, taggedkernel)
taggedsol = solve(taggedSystem, solverFuchs)
@test all(sol[19][end] .≈ [0.13881069002073976 0.04192527228732725; 0.041925272287327266 0.52419581715369]) # regression test
@test taggedsol[19][end] ≈ 0.7678799482793754 # regression tests


s = 2
α = 1.0
β = 0.0
γ = 0.0
δ = -6*kBT / m[s]
msd0 = 0.0
dmsd0 = 0.0

msdkernel = MSDMultiComponentModeCouplingKernel(s, ρ, kBT, m, k_array, Sₖ, sol, taggedsol);
msdequation = MemoryEquation(α, β, γ, δ, msd0, dmsd0, msdkernel);
msdsol = solve(msdequation, solverFuchs)
@test sum(msdsol.F) ≈ 0.6057207573375025