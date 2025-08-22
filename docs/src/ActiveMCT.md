## Active Mode-Coupling Theory

### Single-component active MCT

Next to standard Mode-Coupling Theory, we also implemented a Mode-Coupling Theory for athermal self-propelled (active) particles as derived in [1,2]. Athermal means that there is no thermal motion. The equation of motion is slightly different than for passive Mode-Coupling Theory:

$$\ddot{F}(k,t) + \frac{1}{\tau_p}\dot{F}(k,t) + \frac{\omega(k) k^2}{S(k)}F(k,t) + \int_0^t \text{d}t'\ M(k,t-t') \dot{F}(k,t') = 0.$$

Here, $\tau_p$ is the persistence time of a single active particle. This version of active MCT requires extra input in the form of spatial velocity correlations $\omega(k)$. Since the effect of active forces is encoded in $S(k)$ and $w(k)$, the theory can be applied to different systems, such as active Brownian particles or active Ornstein-Uhlenbeck particles.

The memory kernel $M(k,t)$ in $d$ dimensions is given by

$$M(k,t) = \frac{\rho\, \omega(k)}{2 (2\pi)^d} \int \text{d}\mathbf{q}\ V(\mathbf{k},\mathbf{q})^2 F(q,t) F(|\mathbf{k}-\mathbf{q}|,t).$$

We also need modified expressions for the vertices $V(\mathbf{k},\mathbf{q})$ and direct correlation function $\mathcal{C}(k)$:

$$V(\mathbf{k},\mathbf{q}) = \frac{\mathbf{k}\cdot\mathbf{q}}{k} \mathcal{C}(q) + \frac{\mathbf{k}\cdot(\mathbf{k}-\mathbf{q})}{k} \mathcal{C}(|\mathbf{k}-\mathbf{q}|)$$

$$\rho \mathcal{C}(k) = 1 - \frac{\omega(k)}{w(\infty) S(k)}$$

The discretization of the kernel is described in `MCT.md`. The kernel is not implemented using Bengtzelius' trick, as we are mostly interested in two-dimensional applications where this is not applicable. The dimensionality of the kernel can be chosen with the parameter `dim` (the default is `dim=3`). Already implemented functionalities from standard MCT were re-used, so any dimension up to $d \approx 20$ should be supported, although here we only explicitly tested `dim=2` and `dim=3`.

### Example code single-component
```julia
using ModeCouplingTheory, Plots

Nk = 50; kmax = 40; dk = kmax / Nk;
k_array = dk*(collect(1:Nk) .- 0.5);

η = 0.514;
ρ = η*6/π;
τₚ = 1.0;   # persistence time

# use analytical functions as example data (as used in MCT.md)
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

# there is no analytical expression for w(k), so we use an example function
# (w(k) is usually obtained from simulation data)
approx_wk(x) = @. 0.8*(1 + cos(1.5*x)*exp(-0.2*x));
wk = approx_wk(k_array); w0=0.85;
Sk = find_analytical_S_k(k_array, η);

# memory equation coefficients
α = 1.0;
β = 1/τₚ;
γ = @. k_array^2 * wk / Sk;
γp = @. k_array^2 * 1.0 / Sk; 
δ = 0.0;

kernelA = ActiveMCTKernel(ρ, k_array, wk, w0, Sk, 3);
problemA = MemoryEquation(α, β, γ, δ, Sk, zeros(Nk), kernelA);
solA = solve(problemA);

# passive system for reference
kernelP = ModeCouplingKernel(ρ, 1.0, 1.0, k_array, Sk);
problemP = MemoryEquation(0.0, 1.0, γp, δ, Sk, zeros(Nk), kernelP);
solP = solve(problemP);

n = 11;
t = get_t(solA);
Fa = get_F(solA,:,n);
Fp = get_F(solP,:,n);

plot(t, Fa, xaxis=(:log10, [10^-4, :auto]), dpi=500, lc=:black, lw=2, labels="Active (τₚ=$(τₚ), w0=$(w0))", framestyle=:box)
plot!(t, Fp, lc=:orange, lw=2, ls=:dash, dpi=500, labels="Passive Brownian system")
xlabel!("time")
ylabel!("F(k,t)")
xlims!((1e-10,1e10))
title!("Active mode-coupling kernel for k = $(k_array[n]), η = $(η)")
```
![image](images/activeMCT_sc_plot.png)


### Tagged-particle kernel

The dynamics of a tagged active particle is governed by the following equation [1]:

$$
\ddot{F}_s(k,t) + \frac{1}{\tau_p} \dot{F}_s(k,t) + w(\infty) k^2 F_s(k,t) + \int_0^t \text{d}t'\ M_s(k,t-t') \dot{F}_s(k,t') = 0,
$$

where $F_s(k,t)$ is the self-intermediate scattering function. The memory kernel is given by

$$
M_s(k,t) = \frac{\rho\, w(\infty)}{(2\pi)^d} \int \text{d}\mathbf{q}\ \left( \frac{\mathbf{k}\cdot(\mathbf{k} - \mathbf{q})}{k} \right)^2 F_s(q,t) F(|\mathbf{k}-\mathbf{q}|,t).
$$

So the tagged-particle memory kernel requires the collective intermediate scattering function $F(k,t)$ as input. An example of the implemented tagged active memory kernel is given below.


- example code

#### Example code tagged-particle kernel




### Multi-component

Active mode-coupling theory can also be solved for mixtures of particles. The multi-component equation reads [2]


$$ \ddot{F}^{\alpha\beta}_k(t) + \frac{1}{\tau_p}\dot{F}^{\alpha\beta}_k(t) + \sum_{\gamma\delta} k^2 \omega^{\alpha\gamma}_k \left( S^{-1}_k \right)^{\gamma\delta} F^{\delta\beta}_k(t) + \sum_\gamma \int_0^t \text{d}t'\ M^{\alpha\gamma}_k(t-t') \dot{F}^{\gamma\beta}_k(t') = 0, $$

where Greek letters denote particle species. Now, the input of the theory consists of partial structure factors $S_k^{\alpha\beta}$ and partial velocity correlations $\omega_k^{\alpha\beta}$. The multicomponent memory kernel can be written as

$$
M^{\alpha\beta}_k(t) = \frac{1}{2 (2\pi)^d} \sum_{\substack{\mu\, \nu \\ \mu'\nu'}}\sum_{\lambda} \int \text{d}\mathbf{q}\ V^{\mu\nu\alpha}_{\mathbf{k},\mathbf{q}} V^{\mu'\nu'\lambda}_{\mathbf{k},\mathbf{q}} F^{\mu\mu'}_q(t) F^{\nu\nu'}_{|\mathbf{k}-\mathbf{q}|}(t) (\omega^{-1}_k)^{\lambda\beta}.
$$

The multicomponent vertices are defined as

$$
V^{\mu\nu\alpha}_{\mathbf{k},\mathbf{q}}= \sum_{\gamma}\frac{\omega_k^{\alpha\gamma}}{\sqrt{\rho_\gamma}} \left( \frac{\mathbf{k}\cdot\mathbf{q}}{k} \delta_{\gamma\nu} \mathcal{C}_q^{\gamma\mu} + \frac{\mathbf{k}\cdot(\mathbf{k}-\mathbf{q})}{k} \delta_{\gamma\mu} \mathcal{C}^{\gamma\nu}_{|\mathbf{k}-\mathbf{q}|} \right),
$$

where $x_\alpha$ is the fraction of particles of species $\alpha$. The modified direct correlation function is defined as 

$$
\mathcal{C}_q^{\alpha\beta} = \delta_{\alpha\beta} - \sum_{\gamma\sigma} (w_\infty^{-1})^{\alpha\gamma} w_q^{\gamma\sigma} (S_q^{-1})^{\sigma\beta}
$$

The multi-component kernel is not implemented using Bengtzelius' trick. If you want to use this kernel in odd dimensions greater than 3, you could consider implementing this trick (see also the passive multi-component MCT kernel) as it would yield a significant speedup of the program.

- mention Tullio for speedup?

#### Example code multi-component kernel

- example code (use Vincent's data)




## References

[1] G. Szamel (2016). “Theory for the dynamics of dense systems of athermal self-propelled particles,” Phys. Rev. E, vol. 93, p. 012603. http://dx.doi.org/10.1103/PhysRevE.93.012603.

[2] V.E. Debets and L.M.C. Janssen (2023). “Mode-coupling theory for mixtures of athermal self-propelled particles,” J. Chem. Phys., vol. 159, p. 014502 https://doi.org/10.1063/5.0155142.
