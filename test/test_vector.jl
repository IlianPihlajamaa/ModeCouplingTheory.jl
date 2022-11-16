N = 5
F0 = @SVector ones(N)
∂F0 = (@SVector zeros(N))
α = 1.0
β = 10.0


γ = @SMatrix [sin(i*j/π)^4/N^2 for i = 1:N, j = 1:N] #some random matrix
λ = @SVector [cos(i)^2*N^2 for i = 1:N] #some random vector

#SMatrix
kernel1 = SchematicDiagonalKernel(λ)
system1 = LinearMCTEquation(α, β, γ, F0, ∂F0, kernel1)
solver1 = FuchsSolver(Δt=10^-2, t_max=10.0^3, verbose=false, N = 128, tolerance=10^-10, max_iterations=10^6)

#Matrix
kernel2 = SchematicDiagonalKernel(Vector(λ))
system2 = LinearMCTEquation(α, β, Matrix(γ), Vector(F0), Vector(∂F0), kernel2)
solver2 = FuchsSolver(Δt=10^-2, t_max=10.0^3, verbose=false, N = 128, tolerance=10^-10, max_iterations=10^6)

inFS = @SVector rand(N)
inF =  Vector(inFS)
out0 = evaluate_kernel(kernel1, inFS, 0.0)
out1 = evaluate_kernel(kernel2, inF, 0.0)
out2 = similar(out1)
evaluate_kernel!(out2, kernel2, inF, 0.0)

@test all(out0 .≈ out1 .≈ out2)

sol1 =  solve(system1, solver1);
sol2 =  solve(system2, solver2);

@test all(sol1.t .≈ sol2.t)
@test all(sol2.F .≈ sol2.F)
@test all(sol1.K .≈ sol2.K)
# solver3 = EulerSolver(Δt=10^-2, t_max=10.0^2, verbose=false)
# sol3 = solve(system1, solver3)

# plot()
# for i = 1:5
#     plot!(log10.(t1), F1[i, :], color=:black, label="Fuchs SVector")
#     scatter!(log10.(t2[2:100:end]), F2[i, 2:100:end], xlims=(-2,2), color=i, label="Fuchs Vector", ls=:dash, lw=3) 
#     plot!(log10.(t3), F3[i, :], color=i, label="Euler", ls=:dashdot, lw=4) 
# end
# plot!() |> display

## SchematicMatrixKernel
Nc = 4
N = 2Nc
α = zeros(2*Nc)
β = Matrix{Float64}(I, 2*Nc, 2*Nc)
γ = zeros(2*Nc, 2*Nc)

for i in 1:Nc
    γ[i,i] = i 
    γ[Nc+i,Nc+i] = i
    γ[Nc+i,i] = i
end

F₀ = ones(2*Nc)
∂ₜF₀ = zeros(2*Nc)

maxiter = 100
thresh = 10^-6
λ = 0.3 * rand(N, N)

#Matrix
kernel1 = SchematicMatrixKernel(λ)
system1 = LinearMCTEquation(α, β, γ, F₀, ∂ₜF₀, kernel1)
solver1 = FuchsSolver(Δt=10^-2, t_max=10.0^1, verbose=false, N = 32, tolerance=10^-10, max_iterations=10^6)

#SMatrix
kernel2 = SchematicMatrixKernel(SMatrix{2*Nc, 2*Nc}(λ))
system2 = LinearMCTEquation(SVector{2*Nc}(α), SMatrix{2*Nc, 2*Nc}(β), SMatrix{2*Nc, 2*Nc}(γ), SVector{2*Nc}(F₀), SVector{2*Nc}(∂ₜF₀), kernel2)
solver2 = FuchsSolver(Δt=10^-2, t_max=10.0^1, verbose=false, N = 32, tolerance=10^-10, max_iterations=10^6)

#SparseMatrix
kernel3 = SchematicMatrixKernel(sparse(λ))
system3 = LinearMCTEquation(α, β, sparse(γ), F₀, ∂ₜF₀, kernel3)
solver3 = FuchsSolver(Δt=10^-2, t_max=10.0^1, verbose=false, N = 32, tolerance=10^-10, max_iterations=10^6)

inFS = @SVector rand(N)
inF =  Vector(inFS)
out0 = evaluate_kernel(kernel1, inFS, 0.0)
out1 = evaluate_kernel(kernel2, inF, 0.0)
out3 = evaluate_kernel(kernel3, inF, 0.0)
out2 = similar(out1)
evaluate_kernel!(out2, kernel2, inF, 0.0)
@test all(out0 .≈ out1 .≈ out2 .≈ out3)

sol1 =  solve(system1, solver1);
sol2 =  solve(system2, solver2);
sol3 =  solve(system3, solver3);

@test all(sol1.t .≈ sol2.t .≈ sol3.t)
@test all(sol1.F .≈ sol2.F .≈ sol3.F)
@test all(sol1.K .≈ sol2.K .≈ sol3.K)
solver3 = EulerSolver(Δt=10^-3, t_max=5.0, verbose=false)
sol3 = solve(system1, solver3)


t_test = 5.0
for i = 1:5
    a2 = Spline1D(sol1.t, sol1[i])(t_test)
    b2 = Spline1D(sol2.t, sol2[i])(t_test)
    c2 = Spline1D(sol3.t, sol3[i])(t_test)
    @test(a2 ≈ b2)
    @test(abs(b2-c2) < 10^-2)
end