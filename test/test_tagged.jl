F0 = 1.0
∂F0 = 0.0
α = 1.0
β = 0.0
γ = 1.0
λ1 = 2.0
λ2 = 1.0
kernel = ModeCouplingTheory.SchematicF2Kernel(λ1)
eq = LinearMCTEquation(α, β, γ, F0, ∂F0, kernel)
sol = solve(eq)

taggedkernel = ModeCouplingTheory.TaggedSchematicF2Kernel(λ2, sol)
tagged_eq = LinearMCTEquation(α, β, γ, F0, ∂F0, taggedkernel)
tagged_sol = solve(tagged_eq);

F0 = @SVector [1.0, 1.0]
∂F0 = @SVector [0.0, 0.0]

kernel = SjogrenKernel(λ1, λ2)
eq = LinearMCTEquation(α, β, γ, F0, ∂F0, kernel)
sol = solve(eq)
@test maximum(abs.(sol[2] .- tagged_sol[1])) < 10^-10


## ModeCouplingTheory

"""
    find_analytical_C_k(k, η)
Finds the direct correlation function given by the 
analytical Percus-Yevick solution of the Ornstein-Zernike 
equation for hard spheres for a given volume fraction η.

Reference: Wertheim, M. S. "Exact solution of the Percus-Yevick integral equation 
for hard spheres." Physical Review Letters 10.8 (1963): 321.
""" 
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

"""
    find_analytical_S_k(k, η)
Finds the static structure factor given by the 
analytical Percus-Yevick solution of the Ornstein-Zernike 
equation for hard spheres for a given volume fraction η.
""" 
function find_analytical_S_k(k, η)
        Cₖ = find_analytical_C_k(k, η)
        ρ = 6/π * η
        Sₖ = @. 1 + ρ*Cₖ / (1 - ρ*Cₖ)
    return Sₖ
end

η = 0.514; ρ = η*6/π; kBT = 1.0; m = 1.0

Nk = 100; kmax = 40.0; dk = kmax/Nk; k_array = dk*(collect(1:Nk) .- 0.5)
# We use the Percus-Yevick solution to the structure factor that can be found above.
Sₖ = find_analytical_S_k(k_array, η)

∂F0 = zeros(Nk); α = 1.0; β = 0.0; γ = @. k_array^2*kBT/(m*Sₖ)

kernel = ModeCouplingKernel(ρ, kBT, m, k_array, Sₖ)
problem = LinearMCTEquation(α, β, γ, Sₖ, ∂F0, kernel)
solver = FuchsSolver(Δt=10^-5, t_max=10.0^5, N = 8, tolerance=10^-8)
sol = @time solve(problem, solver);

Cₖ = find_analytical_C_k(k_array, η)
F0 = ones(Nk); ∂F0 = zeros(Nk); α = 1.0; β = 0.0; γ = @. k_array^2*kBT/m

taggedkernel = ModeCouplingTheory.TaggedModeCouplingKernel(ρ, kBT, m, k_array, Cₖ, sol)
taggedproblem = LinearMCTEquation(α, β, γ, F0, ∂F0, taggedkernel)
sols = solve(taggedproblem, solver)

@test sum(sum(sol.F)) ≈ 25593.438983792006