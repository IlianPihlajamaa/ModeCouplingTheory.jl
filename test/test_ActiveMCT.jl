
Nk = 50; kmax = 20.0; dk = kmax / Nk;
k_array = dk * (collect(1:Nk) .- 0.5);

# same analytical functions as defined in test_MCT.jl
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

function find_analytical_S_k(k, η)
        Cₖ = find_analytical_C_k(k, η)
        ρ = 6/π * η
        Sₖ = @. 1 + ρ*Cₖ / (1 - ρ*Cₖ)
    return Sₖ
end

ρ = 0.51; m = 1.0; kbT = 1.0;
Sk = find_analytical_S_k(k_array, ρ*π/6);
wk = ones(Nk); w0 = 1.0;

α = 1.0; β = 0.0; γ = @. k_array^2 * wk / Sk;

# equivalence to 3D passive kernel
kernel = ModeCouplingKernel(ρ, kbT, m, k_array, Sk);
problem = MemoryEquation(α, β, γ, 0.0, Sk, zeros(Nk), kernel);
sol = solve(problem);

ker_act = ActiveMCTKernel(ρ, k_array, wk, w0, Sk, 3);
prob_act = MemoryEquation(α, β, γ, 0.0, Sk, zeros(Nk), ker_act);
sol_act = solve(prob_act);

sol_steady = solve_steady_state(γ, Sk, kernel);
sol_steady_act = solve_steady_state(γ, Sk, ker_act);

@test sum(sum(sol.F)) ≈ sum(sum(sol_act.F)) rtol=1e-5
@test find_relaxation_time(get_t(sol), get_F(sol,:,6)) ≈ find_relaxation_time(get_t(sol_act), get_F(sol_act,:,6))
@test sum(get_F(sol_steady,1,:)) ≈ sum(get_F(sol_steady_act,1,:)) rtol = 1e-5

w0_2 = 0.8;
ker_act_w0 = ActiveMCTKernel(ρ, k_array, wk, w0_2, Sk, 3);
prob_act_w0 = MemoryEquation(α, β, γ, 0.0, Sk, zeros(Nk), ker_act_w0);
sol_act_w0 = solve(prob_act_w0);

@test sum(sum(sol_act_w0.F)) != sum(sum(sol_act.F))


# equivalence to 2D passive kernel
kernel_2D = ModeCouplingKernel(ρ, kbT, m, k_array, Sk, dims=2);
problem_2D = MemoryEquation(α, β, γ, 0.0, Sk, zeros(Nk), kernel_2D);
sol_2D = solve(problem_2D);

ker_act_2D = ActiveMCTKernel(ρ, k_array, wk, w0, Sk, 2);
prob_act_2D = MemoryEquation(α, β, γ, 0.0, Sk, zeros(Nk), ker_act_2D);
sol_act_2D = solve(prob_act_2D);

sol_steady_2D = solve_steady_state(γ, Sk, kernel_2D);
sol_steady_act_2D = solve_steady_state(γ, Sk, ker_act_2D);

@test sum(sum(sol_2D.F)) ≈ sum(sum(sol_act_2D.F)) rtol=1e-5
@test find_relaxation_time(get_t(sol), get_F(sol,:,6)) ≈ find_relaxation_time(get_t(sol_act), get_F(sol_act,:,6))
@test sum(get_F(sol_steady_2D,1,:)) ≈ sum(get_F(sol_steady_act_2D,1,:))
@test sum(sum(sol_act_2D.F)) != sum(sum(sol_act.F))


# tagged kernel
γ2 = @. k_array^2 * w0;

ker_tag = TaggedModeCouplingKernel(ρ, kbT, m, k_array, Sk, sol, dims=3);
prob_tag = MemoryEquation(α, β, γ2, 0.0, ones(Nk), zeros(Nk), ker_tag);
sol_tag = solve(prob_tag);

ker_tag_act = ModeCouplingTheory.TaggedActiveMCTKernel(ρ, k_array, w0, wk, Sk, sol_act, 3);
prob_tag_act = MemoryEquation(α, β, γ2, 0.0, ones(Nk), zeros(Nk), ker_tag_act);
sol_tag_act = solve(prob_tag_act);

@test sum(sum(sol_tag.F)) ≈ sum(sum(sol_tag_act.F)) rtol = 1e-5