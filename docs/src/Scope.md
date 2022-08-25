## Automatic differentiation

This package is compatible with forward-mode automatic differentiation. This makes it possible to calculate quatities such as $\frac{dF(t)}{d\gamma}$ for example.

### Example

Let's take the derivative of the solution to the generalized Langevin equation with the exponentially decaying kernel with respect to the coupling parameter. First we need to write a function that solves this equation and outputs the solution for a given coupling parameter. Since we know the analytical solution, we can compare with the derivative of that.

```julia
using ModeCouplingTheory, Plots
function my_func(λ)
    F0 = 1.0
    ∂F0 = 0.0
    α = 0.0
    β = 1.0
    γ = 1.0

    kernel1 = ExponentiallyDecayingKernel(λ, 1.0)
    system1 = MCTProblem(α, β, γ, F0, ∂F0, kernel1)
    solver1 = FuchsSolver(system1, Δt=10^-4, t_max=5*10.0^1, verbose=false, N = 128, tolerance=10^-10, max_iterations=10^6)

    t1, F1, K1 =  solve(system1, solver1, kernel1)
    return [t1[2:end], F1[2:end], K1[2:end]]
end

function exact_func(λ, t)
    temp = sqrt(λ*(λ+4)) 
    F = @. exp(-0.5* t*(temp+λ+2)) * (temp*(exp(temp*t)+1)+ λ* (exp(temp*t)-1)) / (2temp) 
    return [t, F]
end

t, F, K = my_func(5.0)
texact, Fexact = exact_func(5.0, t)

p = plot(log10.(texact), Fexact, label="Exact", lw=4) 
scatter!(log10.(t[1:100:end]), F[1:100:end], label="Numerical solution", ls=:dash, lw=4) 
```
![image](images/deriv1.png)

Now we can take the derivative with respect to the argument of the functions we defined:

```julia
using ForwardDiff
_, dF_exact = ForwardDiff.derivative(y -> exact_func(y, t), 5.0)
_, dF, _ = ForwardDiff.derivative(my_func, 5.0)

p = plot(log10.(texact), dF_exact, lw=3, label="Exact", ylabel="dF/dλ(λ=5,t)", xlabel="log10(t)") 
plot!(log10.(t), dF, ls=:dash, lw=3, label="Numerical solution", legend=:topleft)
```
![image](images/deriv2.png)

## Measurement errors

## Steady state (non-ergodicity parameter)

## Relaxation time