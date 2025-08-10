using ModeCouplingTheory, Test
"""
    find_analytical_C_k_PY(k, η)
Finds the direct correlation function given by the 
analytical Percus-Yevick solution of the Ornstein-Zernike 
equation for hard spheres for a given volume fraction η.

Reference: Wertheim, M. S. "Exact solution of the Percus-Yevick integral equation 
for hard spheres." Physical Review Letters 10.8 (1963): 321.
"""
function find_analytical_C_k_PY(k, η)
    A = -(1 - η)^-4 * (1 + 2η)^2
    B = (1 - η)^-4 * 6η * (1 + η / 2)^2
    D = -(1 - η)^-4 * 1 / 2 * η * (1 + 2η)^2
    Cₖ = @. 4π / k^6 *
            (
        24 * D - 2 * B * k^2 - (24 * D - 2 * (B + 6 * D) * k^2 + (A + B + D) * k^4) * cos(k)
        +
        k * (-24 * D + (A + 2 * B + 4 * D) * k^2) * sin(k)
    )
    return Cₖ
end

"""
    find_analytical_S_k_PY(k, η)
Finds the static structure factor given by the 
analytical Percus-Yevick solution of the Ornstein-Zernike 
equation for hard spheres for a given volume fraction η.
"""
function find_analytical_S_k_PY(k, η)
    Cₖ = find_analytical_C_k_PY(k, η)
    ρ = 6 / π * η
    Sₖ = @. 1 + ρ * Cₖ / (1 - ρ * Cₖ)
    return Sₖ
end

## Here, we test the consistence of the two different implementations for d=3.
## The first one uses straightforward discretisation, while the second one uses Bengtzelius' trick.
## The test for equivalence is performed for the collective ISF, the tagged ISF as well as the MSD. 
## For the sake of testing we use a very coarse wave-vector grid. 

η = 0.50;
ρ = η * 6 / π;
kBT = 1.0;
m = 1.0;
d = 3;
kmax = 40.0
Nk = 20
dk = kmax / Nk
k_array = dk * (collect(1:Nk) .- 0.5)

∂F0 = zeros(Nk)
Sk = find_analytical_S_k_PY.(k_array, η)
α = 0.0;
β = 1.0;
δ = 0;
γ = (kBT / m) .* k_array .^ 2 ./ Sk

solver = TimeDoublingSolver(Δt=10.0^-6, t_max=10.0^-3, N=32, tolerance=10^-12, verbose=false, max_iterations=10^8)

kernel_d = ModeCouplingTheory.dDimModeCouplingKernel(ρ, kBT, m, k_array, Sk, d)
equation_d = MemoryEquation(α, β, γ, δ, Sk, ∂F0, kernel_d)
sol_d = solve(equation_d, solver)

kernelMCT = ModeCouplingKernel(ρ, kBT, m, k_array, Sk)
equation = MemoryEquation(α, β, γ, δ, Sk, ∂F0, kernelMCT)
solMCT = solve(equation_d, solver)

@test sum(abs.(evaluate_kernel(kernel_d, Sk, 0) .- evaluate_kernel(kernelMCT, Sk, 0))) < 10.0^-8

for ik in [5, 10, 15]
    sum_d = sum(get_F(sol_d, ik, :))
    sum_MCT = sum(get_F(solMCT, ik, :))
    @test abs(sum_d - sum_MCT) < 10.0^-8

end


γ_tagged = (kBT / m) .* k_array .^ 2
F_tagged0 = ones(Nk)
kernel_tagged = ModeCouplingTheory.dDimTaggedModeCouplingKernel(d, ρ, kBT, m, k_array, Sk, sol_d)
equation_tagged_d = MemoryEquation(α, β, γ_tagged, δ, F_tagged0, ∂F0, kernel_tagged)
sol_tagged_d = solve(equation_tagged_d, solver)

kernel_taggedMCT = TaggedModeCouplingKernel(ρ, kBT, m, k_array, Sk, solMCT)
equation_taggedMCT = MemoryEquation(α, β, γ_tagged, δ, F_tagged0, ∂F0, kernel_taggedMCT)
sol_taggedMCT = solve(equation_taggedMCT, solver)

@test sum(abs.(evaluate_kernel(kernel_tagged, ones(Nk), 0) .- evaluate_kernel(kernel_taggedMCT, ones(Nk), 0))) < 10.0^-8
for ik in [3, 7, 10]
    sum_d = sum(get_F(sol_tagged_d, :, ik))
    sum_MCT = sum(get_F(sol_taggedMCT, :, ik))
    @test abs(sum_d - sum_MCT) < 10.0^-8
end

α_MSD = 0;
β_MSD = 1.0;
γ_MSD = 0.0;
δ_MSD = -2 * d * kBT / m;
Δ0 = 0.0;
∂Δ0 = 0.0;

kernel_MSD = ModeCouplingTheory.dDimMSDModeCouplingKernel(d, ρ, kBT, m, k_array, Sk, sol_d, sol_tagged_d)
equation_MSD = MemoryEquation(α_MSD, β_MSD, γ_MSD, δ_MSD, Δ0, ∂Δ0, kernel_MSD)

kernel_MSDMCT = MSDModeCouplingKernel(ρ, 1.0, 1.0, k_array, Sk, solMCT, sol_taggedMCT)
equation_MSDMCT = MemoryEquation(α_MSD, β_MSD, γ_MSD, δ_MSD, Δ0, ∂Δ0, kernel_MSDMCT)

@test sum(abs.(evaluate_kernel(kernel_MSD, 0.0, 0) .- evaluate_kernel(kernel_MSDMCT, 0.0, 0))) < 10.0^-8

sol_MSD = solve(equation_MSD, solver)
sol_MSDMCT = solve(equation_MSDMCT, solver)

sum_d_MSD = sum(get_F(sol_MSD, :, 1))
sum_MCT_MSD = sum(get_F(sol_MSDMCT, :, 1))
@test abs(sum_d_MSD - sum_MCT_MSD) < 10.0^-8
