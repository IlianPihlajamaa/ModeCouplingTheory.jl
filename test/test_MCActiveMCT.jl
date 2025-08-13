Ns = 2; Nk = 15;
kmax = 40.0; dk = kmax/Nk;
k_array = dk*(collect(1:Nk) .- 0.5);

η = 0.515;  # volume fraction
τₚ = 1.0;   # persistence time
kbT = 1.0;
m = ones(Ns);

x = [0.2, 0.8]    # number fractions
particle_diameters = [1.0, 1.0]
ρ_all = 6η/(π*sum(x .* particle_diameters .^ 3))  # total density
ρₐ = ρ_all * x    # partial densities

# initial data: all ones to compare with passive kernel
Sk = [SMatrix{Ns,Ns}(Matrix(I(Ns)).*1.0) for i=1:Nk];
wk = [Matrix(I(Ns)) .* 1.0 for i=1:Nk];
w0 = [1.0 0; 0 1.0];

α = 1.0; β = 1/τₚ; δ = @SMatrix zeros(Ns, Ns);
γ = [@SMatrix zeros(Ns, Ns) for j in 1:length(k_array)];

@inbounds for i=1:Nk  # calculating γ
    test = zeros(Ns, Ns);

    for α=1:Ns, β=1:Ns
        for γ=1:Ns
            test[α,β] += k_array[i]^2 * wk[i][α,γ] * inv(Sk[i])[γ,β];
        end
    end
    γ[i] = test;
end

ker = ActiveMultiComponentKernel(ρₐ, k_array, wk, w0, Sk, 3);
prob = MemoryEquation(α, β, γ, δ, copy(Sk), 0.0.*similar(Sk), ker);
sol = solve(prob);

kerP = MultiComponentModeCouplingKernel(ρₐ, 1.0, ones(Ns), k_array, Sk);
probP = MemoryEquation(α, β, γ, δ, copy(Sk), 0.0.*similar(Sk), kerP);
solP = solve(probP);


# compare with passive case
@test sum(sum(sol.F)) ≈ sum(sum(solP.F)) rtol = 1e-5
@test find_relaxation_time(get_t(sol), get_F(sol,:,6,1)) ≈ find_relaxation_time(get_t(solP), get_F(solP,:,6,1))


# check interchangeability of indices
@test get_F(sol,:,5,(1,2)) ≈ get_F(sol,:,5,(2,1)) rtol=1e-5
@test ker.DCF[10][2,1] == ker.DCF[10][1,2]


# change activity
ker2 = ActiveMultiComponentKernel(ρₐ, k_array, wk, 500 .* w0, Sk, 3);
prob2 = MemoryEquation(α, β, γ, δ, copy(Sk), 0.0.*similar(Sk), ker2);
sol2 = solve(prob2);

@test sum(sum(sol2.F)) != sum(sum(sol.F))


# for one species: should be the same as single-component active kernel
Ns = 1; x = [1.0];
ρ_all = 6 * η / (π*x);
ρₐ = ρ_all * x    # partial densities

Sk = [SMatrix{Ns,Ns}(Matrix(I(Ns)).*1.0) for i=1:Nk];
wk = [Matrix(I(Ns)) .* 1.0 for i=1:Nk];
wk_sc = ones(Nk); Sk_sc = ones(Nk);
w0 = Matrix(I(Ns)) .* 1.0

α = 1.0; β = 1/τₚ; δ = @SMatrix zeros(Ns, Ns);
γ = [@SMatrix zeros(Ns, Ns) for j in 1:length(k_array)];
γ_sc = @. k_array^2 * wk_sc / Sk_sc;

@inbounds for i=1:Nk  # calculating γ
    test = zeros(Ns, Ns);

    for α=1:Ns, β=1:Ns
        for γ=1:Ns
            test[α,β] += k_array[i]^2 * wk[i][α,γ] * inv(Sk[i])[γ,β];
        end
    end
    γ[i] = test;

end

ker3 = ActiveMultiComponentKernel(ρₐ, k_array, wk, w0, Sk, 3);
prob3 = MemoryEquation(α, β, γ, δ, copy(Sk), 0.0.*similar(Sk), ker3);
sol3 = solve(prob3);

ker_sc = ActiveMCTKernel(ρ_all[1], k_array, wk_sc, w0[1], Sk_sc, 3);
prob_sc = MemoryEquation(α, β, γ_sc, 0.0, copy(Sk_sc), 0.0.*similar(Sk_sc), ker_sc);
sol_sc = solve(prob_sc);

@test get_F(sol3,:,15,1) ≈ get_F(sol_sc,:,15) rtol=1e-5
@test sum(sum(sum(sol3.F))) ≈ sum(sum(sol_sc.F)) rtol=1e-5
