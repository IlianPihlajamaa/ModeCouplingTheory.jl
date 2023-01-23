
F0 = 1.0
∂F0 = 1.0
α = 1.0
β = 1.0
γ = 1.0
δ = 0.0
λ = 2.0
λ = 0.7
σ = -0.1
t0 = 0.001

equation = BetaScalingEquation(λ, σ, t0, δ=0.0)

@test equation.coeffs.a≈0.3269591
@test equation.coeffs.b≈0.640741

sol =  solve(equation, TimeDoublingSolver(t_max=10^4.))

# the short-time behavior is the critical power law:
# (t/t0)^a * g(t) ~ 1
@test isapprox(sum((@. (sol.t/t0)^equation.coeffs.a * sol.F)[2:501])/500,1.,rtol=10^-3.)
# the long-time behavior is the von Schweidler law:
# (t/t0)^-b * g(t) ~ -|σ|^-0.5 |σ|^-(b/2a) B(λ)
# B(λ) is tabulated in Götze, J Phys Condens Matter 2, 8485 (1990)
# from there B(0.7)=0.681
@test isapprox(sum((@. (sol.t/t0)^-equation.coeffs.b * sol.F)[3000:lastindex(sol.t)])/(lastindex(sol.t)-2999),-0.681*sqrt(abs(σ))*abs(σ)^(equation.coeffs.b/(2*equation.coeffs.a)),rtol=10^-2.)
