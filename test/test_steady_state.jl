## scalar

F0 = 1.0
γ = 1.0
λ = 4.00000001
kernel = SchematicF2Kernel(λ)
sol = solve_steady_state(γ, F0, kernel; tolerance=10^-10, verbose=false)
@test abs(sol.F[1] - 0.5) < 0.001

## SVector
N = 5
F0 = @SVector ones(N)
γ = @SMatrix [sin(i * j / π)^4 / N^2 for i = 1:N, j = 1:N] #some random matrix
λ = @SVector [cos(i)^2 * N^2 for i = 1:N] #some random vector
kernel = SchematicDiagonalKernel(λ)
sol = solve_steady_state(γ, F0, kernel; tolerance=10^-10, verbose=false)
@test sum(sol.F[1]) ≈ 4.892831203320679

# Vector
N = 5
F0 = ones(N)
γ = [sin(i * j / π)^4 / N^2 for i = 1:N, j = 1:N] #some random matrix
λ = [cos(i)^2 * N^2 for i = 1:N] #some random vector
kernel = SchematicDiagonalKernel(λ)
sol = solve_steady_state(γ, F0, kernel; tolerance=10^-10, verbose=false)
@test sum(sol.F[1]) ≈ 4.892831203320679

# Vector out-of-place
N = 5
F0 = ones(N)
γ = [sin(i * j / π)^4 / N^2 for i = 1:N, j = 1:N] #some random matrix
λ = [cos(i)^2 * N^2 for i = 1:N] #some random vector

import ModeCouplingTheory.MemoryKernel
struct MyKernel{T} <: MemoryKernel
    λ::T
end
import ModeCouplingTheory.evaluate_kernel
evaluate_kernel(kernel::MyKernel, F, t) = Diagonal(kernel.ν .* F .^ 2)

sol = solve_steady_state(γ, F0, kernel; tolerance=10^-10, verbose=false, inplace=false)
@test sum(sol.F[1]) ≈ 4.892831203320679

#### MCT ###


N = 100
η0 = 0.51595
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

kernel = ModeCouplingKernel(ρ, kBT, m, k_array, Sₖ)
sol = solve_steady_state(γ, F0, kernel; tolerance=10^-8, verbose=false)
fk = sol.F[1]
@test sum(fk) ≈ 19.285439116785284 # regression test

#### MCMCT ###

Ns = 2
Nk = 100
ϕ = 0.55
kBT = 1.0
m = ones(Ns)
particle_diameters = [0.8, 1.0]
concentration_ratio = [0.2, 0.8]
concentration_ratio ./= sum(concentration_ratio)
ρ_all = 6ϕ / (π * sum(concentration_ratio .* particle_diameters .^ 3))
ρ = ρ_all * concentration_ratio
dk = 40 / Nk
k_array = dk * (collect(1:Nk) .- 0.5)
x = ρ / sum(ρ)
Sₖdata = reshape(readdlm("Sk_MC2.txt"), (2, 2, 100)) .* sqrt.(x .* x')
Sₖ = [@SMatrix(zeros(Ns, Ns)) for i = 1:Nk]
for i = 1:Nk
    Sₖ[i] = Sₖdata[:, :, i]
end
S⁻¹ = inv.(Sₖ)
J = similar(Sₖ) .* 0.0

for ik = 1:Nk
    J[ik] = kBT * k_array[ik]^2 * x ./ m .* I(Ns)
end
F₀ = copy(Sₖ)
∂ₜF₀ = [@SMatrix(zeros(Ns, Ns)) for i = 1:Nk]
α = 0.0
β = 1.0
γ = similar(Sₖ)
for ik = 1:Nk
    γ .= J .* S⁻¹
end

kernel = MultiComponentModeCouplingKernel(ρ, kBT, m, k_array, Sₖ)
sol = solve_steady_state(γ, F₀, kernel; tolerance=10^-12, verbose=false)
fk = sol.F[1]
@test sum(sum(fk)) ≈ 45.50146332163797 # regression test