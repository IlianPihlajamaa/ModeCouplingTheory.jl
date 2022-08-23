t = 10 .^ range(-10,10, length=10000)
τ = 100000.0
F = @. exp(-t/τ)*6

τ_α = find_relaxation_time(t, F; threshold=exp(-1), mode=:log)
@test abs(τ-τ_α) < 0.1
τ_α = find_relaxation_time(t, F; threshold=exp(-1), mode=:lin)
@test abs(τ-τ_α) < 1.0