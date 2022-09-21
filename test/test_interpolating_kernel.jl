F0 = 1.0; ∂F0 = 0.0; α = 0.0; β = 1.0; γ = 1.0; λ = 1.0; τ = 1.0;

kernel = ExponentiallyDecayingKernel(λ, τ)
equation = LinearMCTEquation(α, β, γ, F0, ∂F0, kernel)
solver = FuchsSolver(Δt=10^-3, t_max=10.0^2, verbose=false, N = 128, tolerance=10^-10, max_iterations=10^6)
t, F, K =  solve(equation, solver)

t = range(0, 100, length=10^3) |> collect
M = @. λ*exp(-t/τ)

kernel = InterpolatingKernel(t, M, k=3)
equation = LinearMCTEquation(α, β, γ, F0, ∂F0, kernel)
solver = FuchsSolver(Δt=10^-3, t_max=10.0^2, verbose=false, N = 128, tolerance=10^-10, max_iterations=10^6)
t2, F2, K2 =  solve(equation, solver)

@test all(abs.(F2 .- F) .< 0.00001)