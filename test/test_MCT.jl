"""
Finds the fourier transform of the direct correlation function given by the 
analytical percus yevick solution of the Ornstein Zernike 
equation for hard spheres for a given volume fraction η on the coordinates r
in units of one over the diameter of the particles
"""
function find_analytical_C_k(k, η)
    A = -(1 - η)^-4 * (1 + 2η)^2
    B = (1 - η)^-4 * 6η * (1 + η / 2)^2
    D = -(1 - η)^-4 * 1 / 2 * η * (1 + 2η)^2
    Cₖ = @. 4π / k^6 *
            (
        24 * D - 2 * B * k^2 - (24 * D - 2 * (B + 6 * D) * k^2 + (A + B + D) * k^4) * cos(k)
        +
        k * (-24 * D + (A + 2 * B + 4 * D) * k^2) * sin(k)
    )
    return Cₖ
end

"""
Finds the static structure factor given by the 
analytical percus yevick solution of the Ornstein Zernike 
equation for hard spheres for a given volume fraction η on the coordinates r
in units of one over the diameter of the particles
"""
function find_analytical_S_k(k, η)
    Cₖ = find_analytical_C_k(k, η)
    ρ = 6 / π * η
    Sₖ = @. 1 + ρ * Cₖ / (1 - ρ * Cₖ)
    return Sₖ
end


N = 100
η0 = 0.52
ρ = η0 * 6 / π
kBT = 1.0
m = 1.0

dk = 40 / N
k_array = dk * (collect(1:N) .- 0.5)
Sₖ = find_analytical_S_k(k_array, ρ * π / 6)

F0 = copy(Sₖ)
∂F0 = zeros(N)
α = 0.0
β = 1.0
γ = @. k_array^2 * kBT / (m * Sₖ)
δ = 0.0


kernel = ModeCouplingKernel(ρ, kBT, m, k_array, Sₖ)
system = MemoryEquation(α, β, γ, δ, F0, ∂F0, kernel)
solver = TimeDoublingSolver(Δt=10^-10, t_max=10.0^10, verbose=false, N=2, tolerance=10^-8, max_iterations=10^8)
sol = solve(system, solver);
@test sum(sum(sol.F)) ≈ 13740.912031583604

# plot(log10.(t), F[19, :]/F[19, 1], label="k = $k_array[19]", lw=3, xlabel="log10(t)", ylabel="F(k,t)")
