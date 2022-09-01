Ns = 2
Nk = 100
ϕ  = 0.515
kBT = 1.0
m = ones(Ns)
particle_diameters = [0.8,1.0]

concentration_ratio = [0.2,0.8]
concentration_ratio ./= sum(concentration_ratio)

ρ_all = 6ϕ/(π*sum(concentration_ratio .* particle_diameters .^3))
ρ = ρ_all * concentration_ratio

dk = 40/Nk
k_array = dk*(collect(1:Nk) .- 0.5)

x = ρ/sum(ρ)

Sₖdata = reshape(readdlm("Sk_MC.txt"), (2,2,100))
Sₖ = [@SMatrix(zeros(Ns, Ns)) for i = 1:Nk]
for i = 1:Nk
    Sₖ[i] = Sₖdata[:, :, i]
end

S⁻¹ = inv.(Sₖ)

J = similar(Sₖ) .* 0.0

for ik = 1:Nk
    J[ik] = kBT*k_array[ik]^2 * x ./ m .* I(Ns)
end


F₀ = copy(Sₖ)
∂ₜF₀ = [@SMatrix(zeros(Ns, Ns)) for i = 1:Nk]
α = 0.0
β = 1.0
γ = similar(Sₖ)
γ .*= 0.0
for ik = 1:Nk
    γ .= J.*S⁻¹
end


kernel = MultiComponentModeCouplingKernel(ρ, kBT, m, k_array, Sₖ)
kernelnaive = ModeCouplingTheory.NaiveMultiComponentModeCouplingKernel(ρ, kBT, m, k_array, Sₖ)
Ftest = rand(eltype(F₀), Nk)
@test all(evaluate_kernel(kernelnaive, Ftest, 0.0) .≈ evaluate_kernel(kernel, Ftest, 0.0))

# println("Solve MCMCT with Euler")

# system  = LinearMCTProblem(α, β, γ, F₀, ∂ₜF₀, kernel)
# solverEuler = EulerSolver(system, verbose=true, Δt=10^-5, t_max=0.03)
# tEuler, FEuler, KEuler = solve(system, solverEuler, kernel)

# println("Solve MCMCT with Fuchs")
# solverFuchs = FuchsSolver(system, verbose=false, N=2, tolerance=10^-8, max_iterations=10^8)
# solve(system, solverFuchs, kernel)
# tFuchs, FFuchs, KFuchs = solve(system, solverFuchs, kernel)



# ik = 20
# scatter(log10.(tEuler[2:10:end]), getindex.(FEuler[ik,2:10:end], 1,1)/Sₖ[ik][1,1], ls=:dash, lw=2, color=1, label="Faa Euler") 
# scatter!(log10.(tEuler[2:10:end]), getindex.(FEuler[ik,2:10:end], 1,2)/Sₖ[ik][1,2], lw=2, color=2, label="Fab Euler") 
# scatter!(log10.(tEuler[2:10:end]), getindex.(FEuler[ik,2:10:end], 2,1)/Sₖ[ik][2,1], ls=:dash, lw=2, color=3, label="Fba Euler") 
# scatter!(log10.(tEuler[2:10:end]), getindex.(FEuler[ik,2:10:end], 2,2)/Sₖ[ik][2,2], ls=:dash, lw=2, color=4, label="Fbb Euler")

# # tFuchs, FFuchs, KFuchs = solve(system, solverFuchs, kernel)

# plot(log10.(tFuchs), getindex.(FFuchs[ik,:], 1,1)/Sₖ[ik][1,1], ls=:dash, lw=2, color=1, label="Faa Fuchs") 
# plot!(log10.(tFuchs), getindex.(FFuchs[ik,:], 1,2)/Sₖ[ik][1,2], lw=2, color=2, label="Fab Fuchs") 
# plot!(log10.(tFuchs), getindex.(FFuchs[ik,:], 2,1)/Sₖ[ik][2,1], ls=:dash, lw=2, color=3, label="Fba Fuchs") 
# plot!(log10.(tFuchs), getindex.(FFuchs[ik,:], 2,2)/Sₖ[ik][2,2], ls=:dash, lw=2, color=4, label="Fbb Fuchs") |> display
# plot(log10.(tFuchs), getindex.(FFuchs[19, :], 1)/Sₖ[19][1], ls=:dashdot, lw=3, ylims=(0.3,0.4), xlims=(3,5), marker=:o)
# include("RelaxationTime.jl")

# tR = find_relaxation_time(tFuchs, getindex.(FFuchs[19, :]/Sₖ[19][1], 1,1), mode=:log)
# scatter!([log10(tR)], [exp(-1)])