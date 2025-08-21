using NPZ, Plots, StaticArrays, ModeCouplingTheory, DelimitedFiles, LinearAlgebra

Ns = 2; Nk = 100; kmax = 40;
dk = kmax / Nk; k_array = dk*(collect(1:Nk) .- 0.5);  # k_array is the same as the files provided by Vincent

τₚ = 0.0001; Teff = 4.0; lₚ = sqrt(3 * Teff * τₚ); println("lp = ",lₚ)

# files:
# τₚ = 0.002; lp = 0.154
# τₚ = 0.1;   lp = 1.095
# τₚ = 0.0001; lp = 0.0346
# τₚ = 0.00002; lp = 0.0155  # this file name needs to be written fully (scientific notation)

w0 = npzread("Data_Teunike/Omega_k/omega_inf_KA_ABP_3D_Teff$(Teff)_tau$(τₚ)_p36.npy")
wk_file = npzread("Data_Teunike/Omega_k/omegakuni_KA_ABP_3D_Teff$(Teff)_tau$(τₚ)_p36.npy")
Sk_file = npzread("Data_Teunike/Sk/Skuni_KA_ABP_3D_Teff$(Teff)_tau$(τₚ)_p36.npy")

# k_test = npzread("Data_Teunike/Sk/kuni_KA_ABP_3D_Teff4.0_tau$(τₚ)_p36.npy")
# println(k_test == k_array)

# τₚ = 0.0021333  # tried this for possible rounding errors lₚ (but this didn't change the results)

# w0 = npzread("Data_Teunike/Omega_k/omega_inf_KA_ABP_3D_Teff$(Teff)_tau0.00002_p36.npy")
# wk_file = npzread("Data_Teunike/Omega_k/omegakuni_KA_ABP_3D_Teff$(Teff)_tau0.00002_p36.npy")
# Sk_file = npzread("Data_Teunike/Sk/Skuni_KA_ABP_3D_Teff$(Teff)_tau0.00002_p36.npy")

Sk = [@SMatrix zeros(Ns, Ns) for j=1:Nk];
wk = [@SMatrix zeros(Ns, Ns) for j=1:Nk];

for i=1:Nk
    Sk[i] = Sk_file[i,:,:];
    wk[i] = wk_file[i,:,:];
end

# plot to check
scatter(k_array, [wk[i][1,1] for i=1:Nk])
scatter(k_array, [Sk[i][1,1] for i=1:Nk])

x = [0.8, 0.2]  # species fraction
# ρ_all = 1000 / (9.41^3);
ρ_all = 1.2;
ρₐ = ρ_all * x

# mode-coupling quantities
α = 1.0; β = 1/τₚ; δ = @SMatrix zeros(Ns, Ns);
γ = [@SMatrix zeros(Ns, Ns) for j in 1:length(k_array)];

# wk = [SMatrix{Ns,Ns}(I(Ns)) for i=1:Nk];  # check with passive kernel
# w0 = I(Ns);

for i=1:Nk
    γ[i] = k_array[i]^2 .* wk[i] * inv(Sk[i]);
end

solver = TimeDoublingSolver(verbose=true, N=8, Δt = 32*10^(-6), tolerance=10^-8, max_iterations=10^8);

ker = ModeCouplingTheory.ActiveMultiComponentKernel2(ρₐ, k_array, wk, w0, Sk, 3);
prob = MemoryEquation(α, β, γ, δ, Sk, 0.0.*similar(Sk), ker);
sol = @time solve(prob, solver)

p = 19;  # for all files: Sk_max = 19
plot(log10.(get_t(sol)), get_F(sol,:,p,1)./Sk[p][1], lc=:black, lw=1.8, label="My kernel")
# xlims!((-3.5, 3.5))

# using JLD2
# save_object("Fk_lp_0.16_new.jld2", sol)

file_Fk = readdlm("Fk_lp_0.035.txt", ';')
file_Fk = file_Fk[sortperm(file_Fk[:,1]),:];
t_Fk = file_Fk'[1,:];
Fk_0 = file_Fk'[2,:];
plot!(log10.(t_Fk), Fk_0, lw=1.8, ls=:dash, color=:orange, label = "from paper (lₚ = $(round(lₚ,digits=3)))")
xlims!((-3.5,3.5))
