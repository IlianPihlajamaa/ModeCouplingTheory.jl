# ModeCouplingTheory.jl
[![Build status (Github Actions)](https://github.com/IlianPihlajamaa/ModeCouplingTheory.jl/workflows/CI/badge.svg)](https://github.com/IlianPihlajamaa/ModeCouplingTheory.jl/actions)
[![codecov.io](http://codecov.io/github/IlianPihlajamaa/ModeCouplingTheory.jl/coverage.svg?branch=main)](http://codecov.io/github/IlianPihlajamaa/ModeCouplingTheory.jl?branch=main)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://IlianPihlajamaa.github.io/ModeCouplingTheory.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://IlianPihlajamaa.github.io/ModeCouplingTheory.jl/dev)

A generic and fast solver of mode-coupling theory-like integrodifferential equations. It uses the algorithm outlined in [Fuchs et al.](https://iopscience.iop.org/article/10.1088/0953-8984/3/26/022/meta) to solve equations of the form
$$\alpha \ddot{F}(t) + \beta \dot{F}(t) + \gamma F(t) + \delta + \int_0^t d\tau K(t-\tau)\dot{F}(\tau) = 0, $$
in which $\alpha$, $\beta$, $\gamma$, and $\delta$ are coefficients, and $K(t) = K(F(t), t)$ is a memory kernel that may depend on $F$. This package exports some commonly used memory kernels, but it is straightforward to define your own. The solver is differentiable and works for scalar- and vector-valued functions $F(t)$. For more information see the [Documentation](https://IlianPihlajamaa.github.io/ModeCouplingTheory.jl/dev).


# Installation

To install the package run:

```julia
import Pkg
Pkg.add("ModeCouplingTheory")
```
In order to install and use it from Python, see the [From Python](https://ilianpihlajamaa.github.io/ModeCouplingTheory.jl/dev/FromPython.html) page of the documentation.

# Example usage:

We can use one of the predefined memory kernels 

```julia
julia> using ModeCouplingTheory
julia> ν = 3.999
3.999

julia> kernel = SchematicF2Kernel(ν)
SchematicF2Kernel{Float64}(3.999)
```
This kernel evaluates `K(t) = ν F(t)^2` when called.

We can now define the equation we want to solve as follows:

```julia
julia> α = 1.0; β = 0.0; γ = 1.0; δ, = 0.0, F0 = 1.0; ∂F0 = 0.0;
julia> equation = LinearMCTEquation(α, β, γ, δ, F0, ∂F0, kernel);
```
Which we can solve by calling `solve`:

```julia
julia> sol = solve(equation);
julia> using Plots
julia> plot(log10.(sol.t), sol.F)
```

![image](readmefig.png)

Please open an issue if anything is unclear in the documentation, if any unexpected errors arise or for feature requests (such as additional kernels). PRs are of course also welcome.
