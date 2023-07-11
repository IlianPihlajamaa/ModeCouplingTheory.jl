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
Nk = 20
ϕ = 0.40
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
system = MemoryEquation(α, β, γ, δ, F₀, ∂ₜF₀, kernel)
solverFuchs = TimeDoublingSolver(N=4, tolerance=10^-12, max_iterations=20000, Δt=10^-4, t_max=10.0^3, verbose=false)
sol =  solve(system, solverFuchs);



@test get_F(sol) == sol.F
@test get_F(sol, 23, 12, (1, 2)) == sol.F[23][12][1,2]
@test get_F(sol, :, 12, (1, 1)) == [sol.F[i][12][1,1] for i in eachindex(sol.F)]
@test get_F(sol, 2:5, 12, (2, 2)) == [sol.F[i][12][2,2] for i in 2:5]
@test get_F(sol, 52, :, (1, 1)) == [sol.F[52][i][1,1] for i in eachindex(sol.F[1])]
@test get_F(sol, 48, 5:17, (2, 2)) == [sol.F[48][i][2,2] for i in 5:17]
@test get_F(sol, 48, 15, (:, :)) == sol.F[48][15][:,:]
@test get_F(sol, 48, 13, (1:1, 1:2)) == sol.F[48][13][1:1,1:2]
@test get_F(sol, :, 13, (1:1, 1:2)) == [sol.F[i][13][1:1,1:2] for i in eachindex(sol.F)]
@test get_F(sol, 14, :, (1:1, 1:2)) == [sol.F[14][i][1:1,1:2] for i in eachindex(sol.F[1])]
@test get_F(sol, 12:14, :, (1:1, 1:2))[1][1][2] == sol.F[12][1][1,2]
@test get_F(sol, 12:14, 2) == get_F(sol, 12:14, 2, (:, :))
@test get_F(sol, 2, 12:14) == get_F(sol, 2, 12:14, (:, :))
@test get_F(sol, 1, 2) == get_F(sol, 1, 2, (:, :))
@test get_F(sol, 12:53) == sol.F[12:53]

@test get_K(sol) == sol.K
@test get_K(sol, 23, 12, (1, 2)) == sol.K[23][12, 12][1,2]
@test get_K(sol, :, 12, (1, 1)) == [sol.K[i][12, 12][1,1] for i in eachindex(sol.K)]
@test get_K(sol, 2:5, 12, (2, 2)) == [sol.K[i][12, 12][2,2] for i in 2:5]
@test get_K(sol, 52, :, (1, 1)) == [sol.K[52][i, i][1,1] for i in axes(sol.K[1], 1)]
@test get_K(sol, 48, 5:17, (2, 2)) == [sol.K[48][i, i][2,2] for i in 5:17]
@test get_K(sol, 48, 15, (:, :)) == sol.K[48][15, 15][:,:]
@test get_K(sol, 48, 13, (1:1, 1:2)) == sol.K[48][13, 13][1:1,1:2]
@test get_K(sol, :, 13, (1:1, 1:2)) == [sol.K[i][13, 13][1:1,1:2] for i in eachindex(sol.K)]
@test get_K(sol, 14, :, (1:1, 1:2)) == [sol.K[14][i, i][1:1,1:2] for i in axes(sol.K[1], 1)]
@test get_K(sol, 12:14, :, (1:1, 1:2))[1][1][2] == sol.K[12][1,1][1,2]
@test get_K(sol, 12:14, 2) == get_K(sol, 12:14, 2, (:, :))
@test get_K(sol, 2, 12:14) == get_K(sol, 2, 12:14, (:, :))
@test get_K(sol, 1, 2) == get_K(sol, 1, 2, (:, :))
@test get_K(sol, 12:53) == sol.K[12:53]

@test get_t(sol) == sol.t