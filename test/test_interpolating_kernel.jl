F0 = 1.0; ∂F0 = 0.0; α = 0.0; β = 1.0; γ = 1.0; λ = 1.0; τ = 1.0; δ = 0.0; 

kernel = ExponentiallyDecayingKernel(λ, τ)
equation = MemoryEquation(α, β, γ, δ, F0, ∂F0, kernel)
solver = TimeDoublingSolver(Δt=10^-3, t_max=10.0^2, verbose=false, N = 128, tolerance=10^-10, max_iterations=10^6)
sol =  solve(equation, solver)

t = range(0, 100, length=10^3) |> collect
M = @. λ*exp(-t/τ)

kernel = InterpolatingKernel(t, M, k=3)
equation = MemoryEquation(α, β, γ, δ, F0, ∂F0, kernel)
solver = TimeDoublingSolver(Δt=10^-3, t_max=10.0^2, verbose=false, N = 128, tolerance=10^-10, max_iterations=10^6)
sol2 =  solve(equation, solver)

@test all(abs.(sol2.F .- sol.F) .< 0.00001)