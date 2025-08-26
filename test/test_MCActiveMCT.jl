Ns = 2; Nk = 20;
kmax = 40.0/5; dk = kmax/Nk;
k_array = dk*(collect(1:Nk) .- 0.5);

τₚ = 0.001;       # persistence time
x = [0.8, 0.2]    # number fractions
ρ_all = 1.2;      # number density
ρₐ = ρ_all * x    # partial densities

# read sample data
Sk_file = readdlm("dataVincent_Sk_Teff4.0_tau0.001.txt", ';')
wk_file = readdlm("dataVincent_wk_Teff4.0_tau0.001.txt", ';')
w0 = SMatrix{Ns,Ns}(readdlm("dataVincent_w0_Teff4.0_tau0.001.txt",';'));

Sk = [@SMatrix zeros(Ns, Ns) for i=1:Nk];
wk = [@SMatrix zeros(Ns, Ns) for i=1:Nk];

for i=1:Nk  # rewrite to static matrices
    Sk[i] = Sk_file[i,:]
    wk[i] = wk_file[i,:]
end

α = 1.0; β = 1/τₚ; δ = @SMatrix zeros(Ns, Ns);
γ = [@SMatrix zeros(Ns, Ns) for j in 1:length(k_array)];

for i=1:Nk
    γ[i] = k_array[i]^2 .* wk[i] * inv(Sk[i]);
end

solver = TimeDoublingSolver(verbose=false, N=16, Δt = 10^(-6), tolerance=10^-8, max_iterations=10^6, t_max=10.0);

ker = ActiveMultiComponentKernel(ρₐ, k_array, wk, w0, Sk, 3);
prob = MemoryEquation(α, β, γ, δ, Sk, zero(Sk), ker);
sol = solve(prob, solver);

# check interchangeability of indices
@test get_F(sol,1,:,(1,2)) == get_F(sol,1,:,(2,1))           # exactly symmetric at t=0.0
@test get_F(sol,:,:,(1,2)) ≈ get_F(sol,:,:,(2,1)) rtol=1e-5  # approximately symmetric for all times


# equivalence with passive kernel
# NOTE: S(k) should be re-defined for the passive kernel as the data uses different definitions
Sk_file_0 = readdlm("dataVincent_Sk_Teff4.0_tau0.txt", ';')
Sk_0 = [@SMatrix zeros(Ns, Ns) for i=1:Nk];
Sk_2 = [@SMatrix zeros(Ns, Ns) for i=1:Nk];
Na = 1000 * x;

for i=1:Nk
    test = zeros(Ns,Ns)
    Sk_0[i] = Sk_file_0[i,:]
    for α=1:Ns, β=1:Ns
        test[α,β] = Sk_0[i][α,β] * sqrt(Na[α] * Na[β]);
    end
    Sk_2[i] = test ./ sum(Na);
end

wk_pass = [SMatrix{Ns,Ns}(1.0.*I(Ns)) for i=1:Nk];
w0_pass = SMatrix{Ns,Ns}(1.0.*I(Ns));
γ_pass = [@SMatrix zeros(Ns, Ns) for j=1:Nk];
γ_pas2 = [@SMatrix zeros(Ns, Ns) for j=1:Nk];
J = similar(Sk_0) .* 0.0

for i=1:Nk
    γ_pass[i] = k_array[i]^2 .* wk_pass[i] * inv(Sk_0[i]);
    J[i] = k_array[i]^2 * x ./ ones(Ns) .* I(Ns)
end
γ_pas2 = J .* inv.(Sk_2);

ker2 = ActiveMultiComponentKernel(ρₐ, k_array, wk_pass, w0_pass, Sk_0, 3);
prob2 = MemoryEquation(0.0, 1.0, γ_pass, δ, Sk_0, zero(Sk_0), ker2);
sol2 = solve(prob2, solver);

kerP = MultiComponentModeCouplingKernel(ρₐ, 1.0, ones(Ns), k_array, Sk_2);
probP = MemoryEquation(0.0, 1.0, γ_pas2, δ, Sk_2, zero(Sk_2), kerP);
solP = solve(probP, solver);

@test find_relaxation_time(get_t(sol2), get_F(sol2,:,5,1)) ≈ find_relaxation_time(get_t(solP), get_F(solP,:,5,1)) rtol=1e-5
@test sum(sum(sum( get_F(sol2,:,1,1) / Sk_0[1][1] ))) ≈ sum(sum(sum( get_F(solP,:,1,1) / Sk_2[1][1] ))) rtol=1e-7
@test sum(sum(sum( get_F(sol2,:,1,2) / Sk_0[1][2] ))) ≈ sum(sum(sum( get_F(solP,:,1,2) / Sk_2[1][2] ))) rtol=1e-7

# effect of changing the activity (wk and w0)
@test sol2.F != sol.F


# for one species: should be the same as single-component active kernel
Ns = 1; Nk = 20; 
kmax = 20.0; dk = kmax/Nk;
k_array = dk*(collect(1:Nk) .- 0.5);

x = [1.0]; η = 0.5;
ρ_all = 6 * η / (π*x);
ρₐ = ρ_all * x    # partial densities

Sk = [@SMatrix zeros(Ns,Ns) for i=1:Nk];
for i=1:Nk
    Sk[i] = SMatrix{Ns,Ns}(find_analytical_S_k(k_array[i], η))
end

wk = [Matrix(I(Ns)) .* 1.0 for i=1:Nk];
wk_sc = ones(Nk); Sk_sc = find_analytical_S_k(k_array, η);
w0 = Matrix(I(Ns)) .* 1.0

α = 1.0; β = 1/τₚ; δ = @SMatrix zeros(Ns, Ns);
γ = [@SMatrix zeros(Ns, Ns) for j in 1:length(k_array)];
γ_sc = @. k_array^2 * wk_sc / Sk_sc;

for i=1:Nk
    γ[i] = k_array[i]^2 .* wk[i] * inv(Sk[i])
end

ker3 = ActiveMultiComponentKernel(ρₐ, k_array, wk, w0, Sk, 3);
prob3 = MemoryEquation(α, β, γ, δ, copy(Sk), zero(Sk), ker3);
sol3 = solve(prob3, solver);

ker_sc = ActiveMCTKernel(ρ_all[1], k_array, wk_sc, w0[1], Sk_sc, 3);
prob_sc = MemoryEquation(α, β, γ_sc, 0.0, copy(Sk_sc), zero(Sk_sc), ker_sc);
sol_sc = solve(prob_sc, solver);

@test get_F(sol3,:,15,1) ≈ get_F(sol_sc,:,15) rtol=1e-10
@test sum(sum(sum(sol3.F))) ≈ sum(sum(sol_sc.F)) rtol=1e-10