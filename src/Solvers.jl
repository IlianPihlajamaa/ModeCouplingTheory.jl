abstract type Solver end
include("EulerSolver.jl")
include("FuchsSolver.jl")

"""
    solve(equation::MCTEquation, solver::Solver)

Solves the `MCTequation` with the provided `kernel` using `solver`. 
Search for a specific solver or kernel object to find more specific information.

If no solver is provided, it uses the default FuchsSolver. 
    
# Returns:
* `t` an array of time values
* `F` The solution in an array of which the last dimension corresponds to the time.
* `K` The memory kernel corresponding to each `F`
"""
function solve(::MCTEquation, ::Solver); error("This solver is not known"); end

solve(p::MCTEquation) = solve(p, FuchsSolver())