using Plots, ModeCouplingTheory, Test

"""
    find_analytical_C_k(k, η)
Finds the direct correlation function given by the 
analytical Percus-Yevick solution of the Ornstein-Zernike 
equation for hard spheres for a given volume fraction η.

Reference: Wertheim, M. S. "Exact solution of the Percus-Yevick integral equation 
for hard spheres." Physical Review Letters 10.8 (1963): 321.
""" 
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

"""
    find_analytical_S_k(k, η)
Finds the static structure factor given by the 
analytical Percus-Yevick solution of the Ornstein-Zernike 
equation for hard spheres for a given volume fraction η.
""" 
function find_analytical_S_k(k, η)
        Cₖ = find_analytical_C_k(k, η)
        ρ = 6/π * η
        Sₖ = @. 1 + ρ*Cₖ / (1 - ρ*Cₖ)
    return Sₖ
end

η = 0.523; ρ = η*6/π; kBT = 1.0; m=1.0 ; d = 3

kmax = 40.0
Nk = 100
dk = kmax/Nk
k_array = dk*(collect(1:Nk) .- 0.5)

∂F0 = zeros(Nk)
Sk = find_analytical_S_k.(k_array, η)
α = 0.0 ; β = 1.0 ;  δ = 0
γ = (kBT / m) .* k_array.^2 ./ Sk 

solver = TimeDoublingSolver(Δt=10^-5, t_max=10.0^4, N = 8, tolerance=10^-6, verbose=true, max_iterations = 10^6)

kernel = dDimModeCouplingKernel(ρ, kBT, m, k_array, Sk, d)
