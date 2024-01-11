
N = 100
η0 = 0.50

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

kernelnaive = NaiveMultiComponentModeCouplingKernel([ρ], kBT, [m], k_array, [@SMatrix([Sₖ[ik]]) for ik = 1:N])
kernelMCMCT = MultiComponentModeCouplingKernel([ρ], kBT, [m], k_array, [@SMatrix([Sₖ[ik]]) for ik = 1:N])
kernelMCT = ModeCouplingKernel(ρ, kBT, m, k_array, Sₖ)

Ftest = rand(SMatrix{1,1,Float64,1}, N)
@test all(getindex.(evaluate_kernel(kernelnaive, Ftest, 0.0), 1) .≈ getindex.(evaluate_kernel(kernelMCMCT, Ftest, 0.0), 1) .≈ evaluate_kernel(kernelMCT, getindex.(Ftest, 1), 0.0))


systemMCT = MemoryEquation(α, β, γ, δ, F0, ∂F0, kernelMCT)
solverMCT = TimeDoublingSolver(Δt=10^-10, t_max=10.0^10, verbose=false, N=16, tolerance=10^-10, max_iterations=10^8)
solMCT = solve(systemMCT, solverMCT);

systemMCMCT = MemoryEquation(α, β, [@SMatrix([γ[ik]]) for ik = 1:N], @SMatrix(zeros(1, 1)), [@SMatrix([F0[ik]]) for ik = 1:N], [@SMatrix([∂F0[ik]]) for ik = 1:N], kernelMCMCT)
solverMCMCTEuler = EulerSolver(Δt=10^-5, t_max=2 * 10.0^-2, verbose=false)
solMCMCTEuler = solve(systemMCMCT, solverMCMCTEuler);


solverMCMCTFuchs = TimeDoublingSolver(Δt=10^-10, t_max=10.0^10, verbose=false, N=16, tolerance=10^-10, max_iterations=10^8)
solMCMCTFuchs = solve(systemMCMCT, solverMCMCTFuchs);

using Dierckx
t_test = 10.0^-2
a = Spline1D(solMCT.t, solMCT[19] / Sₖ[19])(t_test)
b = Spline1D(solMCMCTEuler.t, getindex.(solMCMCTEuler[19], 1) / Sₖ[19][1])(t_test)
c = Spline1D(solMCMCTFuchs.t, getindex.(solMCMCTFuchs[19], 1) / Sₖ[19][1])(t_test)
@test(a ≈ c)
@test(abs(b - c) < 10^-3)

