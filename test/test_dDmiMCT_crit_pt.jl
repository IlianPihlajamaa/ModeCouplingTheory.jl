using ModeCouplingTheory, Test, SpecialFunctions

function find_analytical_Ck_2dPY(σ, k_array, ϕ)
    Nk = length(k_array)
    ck = zeros(Nk)

    for ik in 1:Nk 
        k = k_array[ik]

        term1 = -5*(1-ϕ)^2*(k*σ)^2*besselj0(k*σ/2)^2 / 4
        term2 = (4*((ϕ-20)*ϕ+7) + 5*(1-ϕ)^2*(k*σ)^2/4)*besselj1(k*σ/2)^2
        term3 = 2*(ϕ-13)*(1-ϕ)*(k*σ)*besselj1(k*σ/2)*besselj0(k*σ/2)

        ck[ik] = pi * (term1 + term2 + term3) / (6*(1-ϕ)^3*k^2) 
    end

    return ck
end

function find_analytical_Ck_5dPY(η, k)
    T = 1+18η+6η^2 
    Q0 = 1/(120η*(1-η)^3)*(1-33η-87η^2-6η^3-T^(3/2))   
    Q1 = -1/(12*(1-η)^3)*((3+2η)*T^(1/2) + 3+19η+3η^2)   
    Q2 =-T^(1/2)/(24(1-η)^3)*(2+3η+T^(1/2))   

    Q̃0 = -8Q2
    c0 = -(Q̃0)^2
    c1 = 120η*Q0^2
    c3 = 20η*(8Q0*Q2-3Q1^2)
    c5 = -3/8*η*c0
    Ck = @. -(1/(k^10))*
        8*π^2 * (8 * (720c5 - 18c3 * k^2 + c1 * k^4) + (-5760c5 + 
        144*(c3 + 20c5) * k^2 - 8*(c1 + 9c3 + 30c5) * k^4 
        + (3c0 + 4c1 + 6c3 + 8c5) * k^6)*cos(k) + 
        k * (-5760c5 + 48*(3c3 + 20c5) * k^2 - 
        (3c0 + 8*(c1 + 3c3 + 6c5)) * k^4 + 
        (c0 + c1 + c3 + c5) * k^6) * sin(k))

    return Ck
end

# Here we test for the prediction of the critical point of MCT for single component 
# PY hard-spheres. For simplicity, the test consists of verifying that the non-ergodicty parameter 
# is zero (finite) just below (above) the critical point from literature. 

σ = 1.0 
kBT = 1.0
m = 1.0

## Numerical Details of the wave-vector grid
kmax = 40.0
Nk = 100
dk = kmax/Nk
k_array = dk*(collect(1:Nk) .- 0.5)

## 2D test
## We test for the critical packing fraction ϕc = 0.6913
## in d=2 for MCT. Taken form Caraglio et al. [Commun. Comput. Phys., 29 (2021)]

ϕ_low = 0.69 
ϕ_high = 0.70

ρ_low = 4*ϕ_low/(pi*σ^2) 
ρ_high = 4*ϕ_high/(pi*σ^2) 

ck_low = find_analytical_Ck_2dPY(σ, k_array, ϕ_low)
Sk_low = 1 .+ ρ_low .* ck_low ./ (1 .- ρ_low.*ck_low)

ck_high = find_analytical_Ck_2dPY(σ, k_array, ϕ_high)
Sk_high = 1 .+ ρ_high .* ck_high ./ (1 .- ρ_high.*ck_high)

d = 2
∂F0 = zeros(Nk)
α = 0.0 ; β = 1.0 ;  δ = 0
γ_low = (kBT/m) .* k_array.^2 ./ Sk_low 
γ_high = (kBT/m) .* k_array.^2 ./ Sk_high 

# kernel2D_low = ModeCouplingTheory.dDimModeCouplingKernel(ρ_low, kBT, m, k_array, Sk_low, d) 

kernel2D_low = ModeCouplingKernel(ρ_low, kBT, m, k_array, Sk_low ; dims=d)
sol_steady_state_low = solve_steady_state(γ_low, Sk_low, kernel2D_low, verbose=false)
fk_low = get_F(sol_steady_state_low, 1, :)

# kernel2D_high = ModeCouplingTheory.dDimModeCouplingKernel(ρ_high, kBT, m, k_array, Sk_high, d) 

kernel2D_high = ModeCouplingKernel(ρ_high, kBT, m, k_array, Sk_high ; dims=d)
sol_steady_state_high = solve_steady_state(γ_high, Sk_high, kernel2D_high, verbose=false)
fk_high = get_F(sol_steady_state_high, 1, :)

@test sum(fk_high) ≈ 17.987734457716535
@test sum(fk_low) < 10.0^-7


## 5D test 
## We test for the critical packing fraction ϕc = 0.2542
## in d=5 for MCT. Taken form Parisi & Zamponi [Rev. Mod. Phys. 82, 789 (2010)]

d = 5 

ϕ_low = 0.25
ϕ_high = 0.26 

ρ_low = 60*ϕ_low/pi^2 
ρ_high = 60*ϕ_high/pi^2 

ck_low = find_analytical_Ck_5dPY(ϕ_low, k_array)
Sk_low = 1 .+ ρ_low .* ck_low ./ (1 .- ρ_low.*ck_low)

ck_high = find_analytical_Ck_5dPY(ϕ_high, k_array)
Sk_high = 1 .+ ρ_high .* ck_high ./ (1 .- ρ_high.*ck_high)

∂F0 = zeros(Nk)
α = 0.0 ; β = 1.0 ;  δ = 0
γ_low = (kBT/m) .* k_array.^2 ./ Sk_low 
γ_high = (kBT/m) .* k_array.^2 ./ Sk_high 

kernel5D_low = ModeCouplingKernel(ρ_low, kBT, m, k_array, Sk_low ; dims=d)
sol_steady_state_low = solve_steady_state(γ_low, Sk_low, kernel5D_low, verbose=false)
fk_low = get_F(sol_steady_state_low, 1, :)

kernel5D_high = ModeCouplingKernel(ρ_high, kBT, m, k_array, Sk_high; dims=d)
sol_steady_state_high = solve_steady_state(γ_high, Sk_high, kernel5D_high, verbose=false)
fk_high = get_F(sol_steady_state_high, 1, :)

@test sum(fk_high) ≈ 33.463940782589255
@test sum(fk_low) < 10.0^-7
