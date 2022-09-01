## Overview of the internals of ModeCouplingTheory.jl

In order to be able to solve an "MCT-like" equation, one must construct an instance of a `MemoryKernel`, an `MCTProblem`, and a `Solver`. These three are all abstract types such that extending this package to include functionality for different equations or different types of memory kernels is easy. For example, subtypes of `MemoryKernel` include `SchematicF1Kernel` and `ModeCouplingKernel`. Examples of `Solver`s are `EulerSolver` and `FuchsSolver`. At this point, the only concrete `MCTProblem` is a `LinearMCTProblem`, which implements the equation mentioned in the [Introduction](https://ilianpihlajamaa.github.io/ModeCouplingTheory.jl/dev/index.html). 

When concrete instances of a `MemoryKernel`, an `MCTProblem`, and a `Solver` have been defined by the user, the function `solve(problem::MCTProblem, solver::Solver, kernel::MemoryKernel)` is called to solve the equation defined by `problem`, with the memory kernel `kernel` using the solver `solver`. 

Thus, in order to extend the functionality of this package to solve an equation of a different form, say:

$$\dot{F}(t) + a F(t)^p + \int_0^td\tau K(\tau)\dot{F}(t-\tau) = 0$$

one needs to define 
* a new type of `MCTProblem` (e.g. `AnharmonicMCTProblem <: MCTProblem`) which stores the coefficients $a$, $p$ and initial conditions.
* optionally a new `Solver` (e.g. `AnharmonicSolver <: Solver`) type, which stores some solver settings, such as timesteps and tolerances. In the case that the solution method is very similar to one that is already implemented, (such as it is in this case), it might be possible to use the already defined solvers such as `FuchsSolver`. 
* a new `solve` method that dispatches on the above types to solve this equation with the right method (e.g. one needs to write the method `solve(problem::AnharmonicMCTProblem, solver::AnharmonicSolver, kernel::MemoryKernel)`). However, in the case that e.g. `FuchsSolver` can be reused, instead of a new `solve` method, one can also create new methods for lower-level function that `solve(problem::AnharmonicMCTProblem, solver::FuchsSolver, kernel::MemoryKernel)` calls, specializing for `AnharmonicMCTProblem`, in order to implement the changes necessary to solve this `AnharmonicMCTProblem`.

The newly defined methods for this new equation should then work with any of the defined memory kernels. In summary, the real functionality of this package is implemented by the `solve` function. The memory kernel, solvers and problem types are used in order to specialize solve calls.

## Internals of the Fuchs Solver

