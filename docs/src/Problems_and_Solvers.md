
## Equations

The most straightforward workflow for solving MCT-like equations is to first construct a `MemoryKernel`, a `MCTEquation`, and optionally a `Solver`, in that order. The `MemoryKernel` is an object that evaluates the memory kernel when passed to the function `evaluate_kernel`. The `MCTEquation` holds the coefficients, initial conditions, and the just defined `MemoryKernel` belonging to the equation that need to be solved, and the `Solver` stores information related to the numerical integration procedure, such as the time step. Once these three objects have been defined, the function `solve(::MCTEquation, ::Solver)` can be called on them to solve the equation

$$\alpha \ddot{F}(t) + \beta \dot{F}(t) + \gamma F(t) + δ + \int_0^t d\tau K(t-\tau)\dot{F}(\tau) = 0$$

A `LinearMCTEquation` defined by the equation above is constructed by the constructor `LinearMCTEquation(α, β, γ, δ, F₀, ∂ₜF₀, kernel::MemoryKernel)`. Here, `F₀` and `∂ₜF₀` are the initial conditions of $F$ and its time derivative. In the case of vector-valued functions `F`, this package requires that the operation `α*F` is defined and returns the same type as `F`. This means that in general, when `F` is a vector, `α` must either be a matrix with a compatible element type, or a scalar. However, because it is common in practice to find equations in which `α` and `F` are both vectors (such that the multiplication `α*F` is understood to be conducted element-wise), vector-valued `α`s will automatically be promoted to diagonal matrices. `β` and `γ` are treated in the same way. `δ` must in the same way be additively compatible with `F`. `LinearMCTEquation` will also evaluate the `kernel` at $t=0$ to find its initial condition.

### Examples

Scalar problems are the most straightforward:

```julia
kernel = SchematicF1Kernel(0.2); # in the next page of the documentation we will 
                                 # explain how to construct memory kernels
α = 1.0; β = 0.0; γ = 1.0; δ =0.0, F0 = 1.0; ∂F0 = 0.0;
problem = LinearMCTEquation(α, β, γ, δ, F0, ∂F0, kernel)
```

For vector-valued problems, the coefficients can be scalar, vector or matrix-valued. They are automatically promoted to make linear algebra work:

```
julia> N = 5;
julia> kernel = SchematicDiagonalKernel(rand(N));
julia> α = 1.0; β = rand(N); γ = rand(N,N); δ = 0.0; F0 = ones(N); ∂F0 = zeros(N);
julia> problem = LinearMCTEquation(α, β, γ, δ, F0, ∂F0, kernel);

julia> problem.coeffs.α
LinearAlgebra.UniformScaling{Float64}
1.0*I

julia> problem.coeffs.β
5×5 LinearAlgebra.Diagonal{Float64, Vector{Float64}}:
 0.789182   ⋅         ⋅        ⋅         ⋅
  ⋅        0.379832   ⋅        ⋅         ⋅
  ⋅         ⋅        0.50589   ⋅         ⋅
  ⋅         ⋅         ⋅       0.241663   ⋅
  ⋅         ⋅         ⋅        ⋅        0.857202

julia> problem.coeffs.γ
5×5 Matrix{Float64}:
 0.746936  0.963531  0.724356  0.31979   0.600617
 0.731198  0.217209  0.603705  0.373079  0.930195
 0.464137  0.670576  0.973505  0.23666   0.536108
 0.40188   0.797017  0.332496  0.841541  0.434256
 0.401826  0.303485  0.238624  0.239107  0.453554

 julia> problem.coeffs.δ
5-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
 0.0
```

### Limitations

This package is not tested for and is not expected to work when the type of `F` is something other than a `Number`, `Vector` or `SVector`. For example, using types like `OffsetArrays` as initial conditions might lead to unexpected behaviour.

## Solvers

A `Solver` object holds the settings for a specific integration method. This package defines two solvers: `EulerSolver` and `TimeDoublingSolver`. The `EulerSolver` implements a simple forward Euler method (with trapezoidal integration) which is wildly inefficient if the domain of $t$ spans many orders of magnitude (such as it often does in Mode-Coupling Theory). It should therefore mainly be used for testing purposes. The `TimeDoublingSolver` should be used in almost all other cases. The scheme it implements is outlined in [1] and in the appendix of [2]. If no solver is provided to a `solve` call, the default `TimeDoublingSolver` is used.

In short, the equation is discretised and solved on a grid of `4N` time-points, which are equally spaced over an interval `Δt`. It is solved using an implicit method, and thus a fixed point has to be found for each time point. This is done by recursive iteration. When the solution is found, the interval is doubled `Δt => 2Δt` and the solution on the previous grid is mapped onto the first `2N` time points of the new grid. The solution on the other`2N` points is again found by recursive iteration. This is repeated until some final time `t_max` is reached.

A `TimeDoublingSolver` is constructed as follows:

```julia
julia> kernel = SchematicF1Kernel(0.2);
julia> α = 1.0; β = 0.0; γ = 1.0; δ = 0.0; F0 = 1.0; ∂F0 = 0.0;
julia> problem = LinearMCTEquation(α, β, γ, δ, F0, ∂F0, kernel);
julia> solver1 = TimeDoublingSolver() # using all default parameters
julia> solver2 = TimeDoublingSolver(N=128, Δt=10^-5, 
                            t_max=10.0^15, max_iterations=10^8, 
                            tolerance=10^-6, verbose=true)
```
As optional keyword arguments `TimeDoublingSolver` accepts:
* `N`: The number of time points in the interval is equal to `4N`. default = `32`
* `t_max`: when this time value is reached, the integration returns. default = `10.0^10`
* `Δt`: starting time interval, this will be doubled repeatedly. default = `10^-10`
* `max_iterations`: the maximal number of iterations before convergence is reached for each time doubling step. default = `10^4`
* `tolerance`: while the error of the recursive iteration is bigger than this value, convergence is not reached. The error by default is computed as the absolute sum of squared differences. default = `10^-10`
* `verbose`: if `true`, some information will be printed. default = `false`
* `inplace`: if `true` and if the type of `F` is mutable, the solver will try to avoid allocating many temporaries. default = `true`

Having defined a `MemoryKernel`, `MCTEquation` and a `Solver`, one can call `t, F, K = solve(problem, solver)` to solve the problem. It outputs a `Vector` of time points `t` and the solution `F` and memory kernel `K` evaluated at those times points. 

### References
[1] Fuchs, Matthias, et al. "Comments on the alpha-peak shapes for relaxation in supercooled liquids." Journal of Physics: Condensed Matter 3.26 (1991): 5047.

[2] Flenner, Elijah, and Grzegorz Szamel. "Relaxation in a glassy binary mixture: Comparison of the mode-coupling theory to a Brownian dynamics simulation." Physical Review E 72.3 (2005): 031508.