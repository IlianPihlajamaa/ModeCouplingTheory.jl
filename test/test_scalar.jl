F0 = 1.0
∂F0 = 1.0
α = 1.0
β = 1.0
γ = 1.0
δ = 0.0
λ = 2.0
kernel1 = ModeCouplingTheory.SchematicF2Kernel(λ)
system1 = LinearMCTEquation(α, β, γ, δ, F0, ∂F0, kernel1)
solver1 = FuchsSolver(; Δt=10^-3, t_max=10.0^2, verbose=false, N = 128, tolerance=10^-10, max_iterations=10^6)
solver2 = EulerSolver(Δt=10^-3, t_max=10.0^2, verbose=false)

sol1 =  solve(system1, solver1)
sol2 =  solve(system1, solver2)



t_test = 10.0^2/2
a = Spline1D(sol1.t, sol1.F)(t_test)
b = Spline1D(sol2.t, sol2.F)(t_test)
@test(abs(b-a) < 10^-3)

F0 = 1.0
∂F0 = 0.0
α = 0.0
β = 1.0
γ = 1.0
λ = 1.0

kernel1 = SchematicF1Kernel(λ)
system1 = LinearMCTEquation(α, β, γ, δ, F0, ∂F0, kernel1)
solver1 = FuchsSolver(Δt=10^-10, t_max=10.0^2, verbose=false, N = 100, tolerance=10^-14, max_iterations=10^6)


sol1 =  solve(system1, solver1)
t = sol1.t
F_analytic = @. exp(-2*t)*(besseli(0, 2t) + besseli(1, 2t))

# plot(log10.(t), F, label="Fuchs")
# plot!(log10.(t), F_analytic, label="Exact", title="F1 Kernel") |> display
@test(all(abs.(F_analytic - sol1.F) .< 10^-3))

F0 = 1.0
∂F0 = 0.0
α = 0.0
β = 1.0
γ = 1.0
λ = 1.0

kernel1 = ExponentiallyDecayingKernel(λ, 1.0)
system1 = LinearMCTEquation(α, β, γ, δ, F0, ∂F0, kernel1)
solver1 = FuchsSolver(Δt=10^-3, t_max=10.0^2, verbose=false, N = 128, tolerance=10^-10, max_iterations=10^6)
solver2 = EulerSolver(Δt=10^-3, t_max=10.0^2, verbose=false)

sol1 =  solve(system1, solver1)
sol2 =  solve(system1, solver2)

t_analytic = 10 .^ range(-3, 3, length=50)
F_analytic = @. (exp(-0.5*(3+sqrt(5))* t_analytic)*(exp(sqrt(5)*t_analytic) * (1+sqrt(5))-1+sqrt(5)))/(2sqrt(5))
F_analytic[isnan.(F_analytic)] .= 0

# plot(log10.(t1), F1, label="Fuchs", lw=3)
# plot!(log10.(t2), F2, label="Euler", ls=:dash, lw=3) 
# scatter!(log10.(t_analytic), F_analytic, label="Exact") |> display

t_test = 10.0^0
a = Spline1D(sol1.t, sol1.F)(t_test)
b = Spline1D(sol2.t, sol2.F)(t_test)
c = Spline1D(t_analytic, F_analytic)(t_test)
@test(abs(b-a) < 10^-3)
@test(abs(c-a) < 10^-3)


F0 = 1.0
∂F0 = 0.0
α = 0.0
β = 1.0
γ = 1.0
λ = (1.0, 1.0, 1.0)

kernel1 = SchematicF123Kernel(λ...)
system1 = LinearMCTEquation(α, β, γ, δ, F0, ∂F0, kernel1)
solver2 = EulerSolver(Δt=10^-3, t_max=10.0^2, verbose=false)

sol1 =  solve(system1)
sol2 =  solve(system1, solver2)

# plot(log10.(t1), F1, label="Fuchs", lw=3)
# plot!(log10.(t2), F2, label="Euler", ls=:dash, lw=3) |>display

t_test = 10.0^2/2
a = Spline1D(sol1.t, sol1.F)(t_test)
b = Spline1D(sol2.t, sol2.F)(t_test)
@test(abs(b-a) < 10^-3)

import ModeCouplingTheory: MemoryKernel, evaluate_kernel
function myfunc(coeffs, t)
    coeffs.δ = -t
end
struct ZeroKernel <: MemoryKernel end
evaluate_kernel(::ZeroKernel, _, _) = 0.0
α = 1.0; β = 0.0; γ = 1.0; δ = 0.0; F0 = 1.0; dF0 = 0.0; kernel = ZeroKernel()
equation = LinearMCTEquation(α, β, γ, δ, F0, dF0, kernel; update_coefficients! = myfunc)
solver = FuchsSolver(t_max = 10.0^2, N = 600, Δt = 10^-4)
sol = solve(equation, solver)
t = sol.t; F = sol.F
@test maximum(abs.(F - (t .+ cos.(t) .- sin.(t)))) < 10^-1
