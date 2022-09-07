N = 5
F0 = @SVector ones(N)
∂F0 = (@SVector zeros(N))
α = 1.0
β = 10.0


γ = @SMatrix [sin(i*j/π)^4/N^2 for i = 1:N, j = 1:N] #some random matrix
λ = @SVector [cos(i)^2*N^2 for i = 1:N] #some random vector

#SMatrix
kernel1 = SchematicDiagonalKernel(λ)
system1 = LinearMCTProblem(α, β, γ, F0, ∂F0, kernel1)
solver1 = FuchsSolver(Δt=10^-2, t_max=10.0^3, verbose=false, N = 128, tolerance=10^-10, max_iterations=10^6)

#Matrix
kernel2 = SchematicDiagonalKernel(Vector(λ))
system2 = LinearMCTProblem(α, β, Matrix(γ), Vector(F0), Vector(∂F0), kernel2)
solver2 = FuchsSolver(Δt=10^-2, t_max=10.0^3, verbose=false, N = 128, tolerance=10^-10, max_iterations=10^6)

inFS = @SVector rand(N)
inF =  Vector(inFS)
out0 = evaluate_kernel(kernel1, inFS, 0.0)
out1 = evaluate_kernel(kernel2, inF, 0.0)
out2 = similar(out1)
evaluate_kernel!(out2, kernel2, inF, 0.0)

@test all(out0 .≈ out1 .≈ out2)

t1, F1, K1 =  solve(system1, solver1);
t2, F2, K2 =  solve(system2, solver2);

@test all(t1 .≈ t2)
@test all(F1 .≈ F2)
@test all(K1 .≈ K2)
solver3 = EulerSolver(Δt=10^-2, t_max=10.0^2, verbose=false)
t3, F3, K3 = solve(system1, solver3)

# plot()
# for i = 1:5
#     plot!(log10.(t1), F1[i, :], color=:black, label="Fuchs SVector")
#     scatter!(log10.(t2[2:100:end]), F2[i, 2:100:end], xlims=(-2,2), color=i, label="Fuchs Vector", ls=:dash, lw=3) 
#     plot!(log10.(t3), F3[i, :], color=i, label="Euler", ls=:dashdot, lw=4) 
# end
# plot!() |> display

## SchematicMatrixKernel

N = 10
F0 = @SVector ones(N)
∂F0 = @SVector zeros(N)
α = 0.0
β = 1.0
γ = 1.0
λ = @SMatrix [exp(-i*j) for i = 1:N, j = 1:N] # some random matrix

#SMatrix
kernel1 = SchematicMatrixKernel(λ)
system1 = LinearMCTProblem(α, β, γ, F0, ∂F0, kernel1)
solver1 = FuchsSolver(Δt=10^-2, t_max=10.0^3, verbose=false, N = 128, tolerance=10^-10, max_iterations=10^6)

#Matrix
kernel2 = SchematicMatrixKernel(Matrix(λ))
system2 = LinearMCTProblem(α, β, γ, Vector(F0), Vector(∂F0), kernel2)
solver2 = FuchsSolver(Δt=10^-2, t_max=10.0^3, verbose=false, N = 128, tolerance=10^-10, max_iterations=10^6)

inFS = @SVector rand(N)
inF =  Vector(inFS)
out0 = evaluate_kernel(kernel1, inFS, 0.0)
out1 = evaluate_kernel(kernel2, inF, 0.0)
out2 = similar(out1)
evaluate_kernel!(out2, kernel2, inF, 0.0)
@test all(out0 .≈ out1 .≈ out2)

t1, F1, K1 =  solve(system1, solver1);
t2, F2, K2 =  solve(system2, solver2);

@test all(t1 .≈ t2)
@test all(F1 .≈ F2)
@test all(K1 .≈ K2)
solver3 = EulerSolver(Δt=10^-3, t_max=10.0^1, verbose=false)
t3, F3, K3 = solve(system1, solver3)

# plot()
# for i = 1:5
#     plot!(log10.(t1), F1[i,:], color=i, label="Fuchs1") 
#     plot!(log10.(t3), F3[i,:], color=i, label="Euler", ls=:dash, lw=3) 
#     scatter!(log10.(t2[2:100:end]), F2[i,2:100:end], color=i, label="Fuchs2", ls=:dash, markerstrokewidth=0) 
# end
# plot!() |> display

t_test = 10.0^1/2
for i = 1:5
    a2 = Spline1D(t1, F1[i, :])(t_test)
    b2 = Spline1D(t2, F2[i, :])(t_test)
    c2 = Spline1D(t3, F3[i, :])(t_test)
    @test(a2 ≈ b2)
    @test(abs(b2-c2) < 2*10^-3)
end
