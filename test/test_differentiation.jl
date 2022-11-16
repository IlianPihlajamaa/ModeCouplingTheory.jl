function main_scalar(λ)
    F0 = 1.0
    ∂F0 = 0.0
    α = 0.0
    β = 1.0
    γ = 1.0

    kernel1 = ExponentiallyDecayingKernel(λ, 1.0)
    system1 = LinearMCTEquation(α, β, γ, F0, ∂F0, kernel1)
    solver1 = FuchsSolver(Δt=10^-4, t_max=5*10.0^1, verbose=false, N = 128, tolerance=10^-10, max_iterations=10^6)

    sol =  solve(system1, solver1)
    return [sol.t[2:end], sol.F[2:end], sol.K[2:end]]
end

function exactf(λ, t)
    # t = 10 .^ range(-4,2,length = 100)
    temp = sqrt(λ*(λ+4)) 
    F = @. exp(-0.5* t*(temp+λ+2))  *  (temp*(exp(temp*t)+1)+ λ* (exp(temp*t)-1)) / (2temp) 
    return [t, F]
end

F = main_scalar(5.0)
F_exact = exactf(5.0, F[1])

@test all(abs.(F[2] .- F_exact[2]) .< 10^-3)

# plot(log10.(F_exact[1]), F_exact[2], lw=3, label="Exact") 
# plot!(log10.(F[1]), F[2], ls=:dash, lw=3, label="Fuchs") |> display



dF_exact = ForwardDiff.derivative(y -> exactf(y, F[1]), 5.0)
dF = ForwardDiff.derivative(main_scalar, 5.0)

# plot(log10.(F_exact[1]), dF_exact[2], lw=3, label="Exact") 
# plot!(log10.(F[1]), dF[2], ls=:dash, lw=3, label="Fuchs") |> display

@test all(abs.(dF[2] .- dF_exact[2]) .< 10^-4)

function main_vector(Λ)
    N = 10
    λ = [sin(i*j/π)^4 for i = 1:N, j = 1:N]*Λ
    F0 = ones(N)
    ∂F0 = zeros(N)
    α = 0.0
    β = 1.0
    γ = [sin(i*j/π)^4 for i = 1:N, j = 1:N]/N^2

    kernel = SchematicMatrixKernel(λ)
    system = LinearMCTEquation(α, β, γ, F0, ∂F0, kernel)
    solver = FuchsSolver(Δt=10^-2, t_max=10.0^3, verbose=false, N = 128, tolerance=10^-10, max_iterations=10^6)
    
    sol =  solve(system, solver);
    return sol.F[3000]
end

Λ = 0.1
main_vector(Λ)
dF = ForwardDiff.derivative(main_vector, Λ)
finite_differences = (main_vector(Λ+sqrt(eps(Float64)))-main_vector(Λ))/sqrt(eps(Float64))
@test all(dF .- finite_differences .< 0.001)
