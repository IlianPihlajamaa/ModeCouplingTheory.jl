# ModeCouplingTheory.jl

This package provides a generic and fast solver of mode-coupling theory-like integrodifferential equations. It uses the algorithm outlined in [Fuchs et al.](https://iopscience.iop.org/article/10.1088/0953-8984/3/26/022/meta) to solve equations of the form

$$\alpha \ddot{F}(t) + \beta \dot{F}(t) + \gamma F(t) + \int_0^t d\tau K(t-\tau)\dot{F}(\tau) = 0$$

in which $\alpha$, $\beta$, and $\gamma$ are coefficients, and $K(t) = K(F(t), t)$. This package exports some commonly used memory kernels, but it is straightforward to define your own. The solver is differentiable and works for scalar- and vector-valued functions $F(t)$. 

# Installation

To install the package run:

```julia
import Pkg
Pkg.add("ModeCouplingTheory")
```

# Example

We can define one of the predefined memory kernels 

```julia
julia> using ModeCouplingTheory
julia> λ = 3.999
3.999
julia> kernel = SchematicF2Kernel(λ)
SchematicF2Kernel{Float64}(3.999)
```
This kernel evaluates $K(t)=\lambda F(t)^2$.

We can now define the equation we want to solve as follows:

```
julia> α = 1.0; β = 0.0; γ = 2.0; F0 = 1.0; ∂F0 = 0.0;
julia> problem = LinearMCTProblem(α, β, γ, F0, ∂F0, kernel)
LinearMCTProblem{Float64, Float64, Float64, Float64, Float64}(1.0, 0.0, 2.0, 1.0, 0.0, 3.999, Float64, Float64, Float64)
```
and a solver:

```julia
julia> solver = FuchsSolver(problem)
```

Now we can solve the equation by calling `solve`:

```julia
julia> using Plots
julia> t, F, K = solve(problem, solver);
julia> plot(log10.(t), F)
```
![image](images/readmefig.png)

Full copy-pastable example:

```
using ModeCouplingTheory, Plots
ν = 3.999
α = 1.0; β = 0.0; γ = 1.0; F0 = 1.0; ∂F0 = 0.0;
kernel = SchematicF2Kernel(ν)
problem = LinearMCTProblem(α, β, γ, F0, ∂F0, kernel)
solver = FuchsSolver(problem)
t, F, K = solve(problem, solver, kernel);
plot(log10.(t), F)
```

