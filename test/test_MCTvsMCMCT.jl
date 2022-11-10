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
Finds the static structure factor given by the 
analytical percus yevick solution of the Ornstein Zernike 
equation for hard spheres for a given volume fraction η on the coordinates r
in units of one over the diameter of the particles
""" 
function find_analytical_S_k(k, η)
        Cₖ = find_analytical_C_k(k, η)
        ρ = 6/π * η
        Sₖ = @. 1 + ρ*Cₖ / (1 - ρ*Cₖ)
    return Sₖ
end


N = 100
η0 = 0.50

ρ = η0*6/π
kBT = 1.0
m = 1.0

dk = 40/N
k_array = dk*(collect(1:N) .- 0.5)
Sₖ = find_analytical_S_k(k_array, ρ*π/6)

F0 = copy(Sₖ)
∂F0 = zeros(N)
α = 0.0
β = 1.0
γ = @. k_array^2*kBT/(m*Sₖ)


kernelnaive = ModeCouplingTheory.NaiveMultiComponentModeCouplingKernel([ρ], kBT, [m], k_array, [@SMatrix([Sₖ[ik]]) for ik = 1:N])
kernelMCMCT = MultiComponentModeCouplingKernel([ρ], kBT, [m], k_array, [@SMatrix([Sₖ[ik]]) for ik = 1:N])
kernelMCT = ModeCouplingKernel(ρ, kBT,m, k_array, Sₖ)

Ftest = rand(SMatrix{1,1,Float64, 1}, N)
@test all(getindex.(evaluate_kernel(kernelnaive, Ftest, 0.0), 1) .≈ getindex.(evaluate_kernel(kernelMCMCT, Ftest, 0.0),1) .≈ evaluate_kernel(kernelMCT, getindex.(Ftest,1), 0.0))


systemMCT = LinearMCTEquation(α, β, γ, F0, ∂F0, kernelMCT)
solverMCT = FuchsSolver(Δt=10^-10, t_max=10.0^10, verbose=false, N = 16, tolerance=10^-10, max_iterations=10^8)
solMCT = solve(systemMCT, solverMCT);


systemMCMCT = LinearMCTEquation(α, β, [@SMatrix([γ[ik]]) for ik = 1:N], [@SMatrix([F0[ik]]) for ik = 1:N], [@SMatrix([∂F0[ik]]) for ik = 1:N], kernelMCMCT)
solverMCMCTEuler = EulerSolver(Δt=10^-5, t_max=2*10.0^-2, verbose=false)
solMCMCTEuler = solve(systemMCMCT, solverMCMCTEuler);


solverMCMCTFuchs = FuchsSolver(Δt=10^-10, t_max=10.0^10, verbose=false, N = 16, tolerance=10^-10, max_iterations=10^8)
solMCMCTFuchs = solve(systemMCMCT, solverMCMCTFuchs);

# plot(log10.(tMCT), FMCT[19,:]/Sₖ[19], lw=3, label="MCT")
# plot!(log10.(tMCMCTEuler), getindex.(FMCMCTEuler[19, :], 1)/Sₖ[19][1], lw=3, label="MCMCT Euler", ls=:dash)
# plot!(log10.(tMCMCTFuchs), getindex.(FMCMCTFuchs[19, :], 1)/Sₖ[19][1], ls=:dashdot, lw=3, ylims=(0.6,1), xlims=(-6,0), label="MCMCT Fuchs")|>display

using Dierckx
t_test = 10.0^-2
a = Spline1D(solMCT.t, solMCT[19]/Sₖ[19])(t_test)
b = Spline1D(solMCMCTEuler.t, getindex.(solMCMCTEuler[19], 1)/Sₖ[19][1])(t_test)
c = Spline1D(solMCMCTFuchs.t, getindex.(solMCMCTFuchs[19], 1)/Sₖ[19][1])(t_test)
@test(a ≈ c)
@test(abs(b-c) < 10^-3)

