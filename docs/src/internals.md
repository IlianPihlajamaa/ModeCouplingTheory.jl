## Overview of the internals of ModeCouplingTheory.jl

In order to be able to solve an "MCT-like" equation, one must construct an instance of an `MCTProblem`, and optionally a `Solver`. Every `MCTProblem` also needs a `MemoryKernel` in order to define the equation to be solved. These three are all abstract types such that extending this package to include functionality for different equations or different types of memory kernels is easy. For example, subtypes of `MemoryKernel` include `SchematicF1Kernel` and `ModeCouplingKernel`. Examples of `Solver`s are `EulerSolver` and `FuchsSolver`. At this point, the only concrete `MCTProblem` is a `LinearMCTProblem`, which implements the equation mentioned in the [Introduction](https://ilianpihlajamaa.github.io/ModeCouplingTheory.jl/dev/index.html). 

When concrete instances of a an `MCTProblem`, and a `Solver` have been defined by the user, the function `solve(problem::MCTProblem, solver::Solver)` is called to solve the equation defined by `problem`, with the memory kernel `kernel` using the solver `solver`. 

Thus, in order to extend the functionality of this package to solve an equation of a different form, say:

$$\dot{F}(t) + a F(t)^p + \int_0^td\tau K(\tau)\dot{F}(t-\tau) = 0$$

one needs to define 
* a new type of `MCTProblem` (e.g. `AnharmonicMCTProblem <: MCTProblem`) which stores the coefficients $a$, $p$ and initial conditions.
* optionally a new `Solver` (e.g. `AnharmonicSolver <: Solver`) type, which stores some solver settings, such as timesteps and tolerances. In the case that the solution method is very similar to one that is already implemented, (such as it is in this case), it might be possible to use the already defined solvers such as `FuchsSolver`, only extending a few methods, see below. 
* a new `solve` method that dispatches on the above types to solve this equation with the right method (e.g. one needs to write the method `solve(problem::AnharmonicMCTProblem, solver::AnharmonicSolver)`). However, in the case that e.g. `FuchsSolver` can be reused, instead of a new `solve` method, one can also create new methods for lower-level function that `solve(problem::AnharmonicMCTProblem, solver::FuchsSolver)` calls, specializing for `AnharmonicMCTProblem`, in order to implement the changes necessary to solve this `AnharmonicMCTProblem`.

The newly defined methods for this new equation should then work with any of the defined memory kernels. In summary, the real functionality of this package is implemented by the `solve` function. The memory kernel, solvers and problem types are used in order to specialize solve calls.

## Internals of the Fuchs Solver

The basic idea of the algorithm popularizd by Fuchs and coworkers was, that, in order to solve the equations over many orders of magnitude in time, it is helpful to periodically increase the time step of the grid on which the equation is solved. Below we give a more detailed overview of the implementation of the algorithm in this package.

The equations are solved using these steps:
1. `allocate_temporary_arrays(problem::MCTProblem, solver::FuchsSolver)` returns a `FuchsTempStruct` that is used internally to avoid unnecessary allocations.
1. `initialize_temporary_arrays!(problem::MCTProblem, solver::FuchsSolver, kernel::MemoryKernel, temp_arrays::FuchsTempStruct)`: The algorithm is started by initializing temporary variables such as $F$ and $K$ discretised on the time grid of $4N$ points on $t_i = i\Delta t/4N$ where $i = 1,\ldots,4N$. $F(t)$ is solved by a forward Euler method on the first $2N$ points to kickstart the algorithm. The effects of the memory integral is neglected here. 
2. `do_time_steps!(problem::MCTProblem, solver::FuchsSolver, kernel::MemoryKernel, temp_arrays::FuchsTempStruct)`: The full equation is discretised on the time points between $i=2N+1$ and $i=4N$. For each of these time points, the parameters $C_1$, $C_2$, and $C_3$ are calculated by `update_Fuchs_parameters!(problem, solver, temp_arrays, i)` as prescribed in the literature. Now, in order to solve for $F(t_i)$, the fixed point of the mapping $C_1 F  = -C_2 K(F) + C_3$ is found by recurstive iteration. Convergence is established if the maximimal squared error is smaller than a set tolerance.
3. `allocate_results!(t_array, F_array, K_array, solver, temp_arrays::FuchsTempStruct)` the results found by step 2, residing in temporary arrays are pushed to `t_array`, `F_array`, and `K_array`, which are returned when the program exits.
4. `new_time_mapping!(problem, solver, temp_arrays::FuchsTempStruct)`: the results stored in the temporary variables `temp_arrays.F_temp`, `temp_arrays.K_temp`, `temp_arrays.I_F`, `temp_arrays.I_K` are mapped from the time points $i=1\ldots4N$ to the points $i=1\ldots2N$ as prescribed in the literature. The time step $\Delta t$ is also doubled.
5. If `Δt > t_max` the main loop exits and the arrays `t_array`, `F_array`, and `K_array` are returned. 

For information on specific methods, see the next page of the documentation.