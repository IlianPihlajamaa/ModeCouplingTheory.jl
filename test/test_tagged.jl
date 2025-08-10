F0 = 1.0
∂F0 = 0.0
α = 1.0
β = 0.0
γ = 1.0
δ = 0.0
λ1 = 2.0
λ2 = 1.0
kernel = ModeCouplingTheory.SchematicF2Kernel(λ1)
eq = MemoryEquation(α, β, γ, δ, F0, ∂F0, kernel)
sol = solve(eq)

taggedkernel = ModeCouplingTheory.TaggedSchematicF2Kernel(λ2, sol)
tagged_eq = MemoryEquation(α, β, γ, δ, F0, ∂F0, taggedkernel)
tagged_sol = solve(tagged_eq);

F0 = @SVector [1.0, 1.0]
∂F0 = @SVector [0.0, 0.0]

kernel = SjogrenKernel(λ1, λ2)
eq = MemoryEquation(α, β, γ, δ, F0, ∂F0, kernel)
sol = solve(eq)
@test maximum(abs.(sol[2] .- tagged_sol[1])) < 10^-10


η = 0.514;
ρ = η * 6 / π;
kBT = 1.0;
m = 1.0;

Nk = 100;
kmax = 40.0;
dk = kmax / Nk;
k_array = dk * (collect(1:Nk) .- 0.5);
# We use the Percus-Yevick solution to the structure factor that can be found above.
Sₖ = find_analytical_S_k(k_array, η)

∂F0 = zeros(Nk);
α = 1.0;
β = 0.0;
γ = @. k_array^2 * kBT / (m * Sₖ);

kernel = ModeCouplingKernel(ρ, kBT, m, k_array, Sₖ)
problem = MemoryEquation(α, β, γ, δ, Sₖ, ∂F0, kernel)
solver = TimeDoublingSolver(Δt=10^-5, t_max=10.0^5, N=8, tolerance=10^-8)
sol = solve(problem, solver);

Cₖ = find_analytical_C_k(k_array, η)
F0 = ones(Nk);
∂F0 = zeros(Nk);
α = 1.0;
β = 0.0;
γ = @. k_array^2 * kBT / m;

taggedkernel = ModeCouplingTheory.TaggedModeCouplingKernel(ρ, kBT, m, k_array, Sₖ, sol)
taggedproblem = MemoryEquation(α, β, γ, δ, F0, ∂F0, taggedkernel)
sols = solve(taggedproblem, solver)

@test sum(sum(sol.F)) ≈ 25492.648222645254


MSD0 = 0.0;
dMSD0 = 0.0;
α = 1.0;
β = 0.0;
γ = 0.0;
δ = -6.0 * kBT / m;
msdkernel = MSDModeCouplingKernel(ρ, kBT, m, k_array, Sₖ, sol, sols)
msdequation = MemoryEquation(α, β, γ, δ, MSD0, dMSD0, msdkernel)
msdsol = solve(msdequation, solver)
@test sum(msdsol.F) ≈ 51.75329405084479