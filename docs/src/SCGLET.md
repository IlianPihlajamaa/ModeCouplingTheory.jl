## Self-consistent Generalized Langevin Equation Theory



```julia
# defining the structure factor
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

function find_analytical_S_k(k, η)
        Cₖ = find_analytical_C_k(k, η)
        ρ = 6/π * η
        Sₖ = @. 1 + ρ*Cₖ / (1 - ρ*Cₖ)
    return Sₖ
end

# Appling the Verlet-Weiss correction
# [1] Loup Verlet and Jean-Jacques Weis. Phys. Rev. A 5, 939 – Published 1 February 1972
ϕ_VW(ϕ :: Float64) = ϕ*(1.0 - (ϕ / 16.0))
k_VW(ϕ :: Float64, k :: Float64) = k*((ϕ_VW(ϕ)/ϕ)^(1.0/3.0))

# initial setup
Nk = 800; dk = 0.4; k = [dk/2 + dk*i for i=0:Nk-1]; η = 0.5
S = find_analytical_S_k(k, η)
∂F0 = zeros(2*Nk); α = 0.0; β = 1.0; γ = @. k^2/S; δ = 0.0

kernel = SCGLEKernel(η, k, S);
equation = MemoryEquation(α, β, γ, δ, S, ∂F0, kernel);
solver = TimeDoublingSolver(Δt=10^-5, t_max=10.0^10, 
    N = 8, tolerance=10^-8, verbose=true);
sol = @time solve(equation, solver);
using Plots
p = plot(xlabel="log10(t)", ylabel="Fs(k,t)", ylims=(0,1))
for ik = [7, 18, 25, 39]
    Fk = get_F(sol, 1:10:800, ik)
    t = get_t(sol)[1:10:800]
    plot!(p, log10.(t), Fk/S[ik], label="k = $(k_array[ik])", lw=3)
end
```