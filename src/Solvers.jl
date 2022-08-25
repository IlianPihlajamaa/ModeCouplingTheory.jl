abstract type Solver end
include("EulerSolver.jl")
include("FuchsSolver.jl")

"""
    solve(problem::MCTProblem, solver::Solver, kernel::MemoryKernel)

Solves the `MCTproblem` with the provided `kernel` using `solver`. 
Search for a specific solver or kernel object to find more specific information.
    
# Returns:
* `t` an array of time values
* `F` The solution in an array of which the last dimension corresponds to the time.
* `K` The memory kernel corresponding to each `F`
"""
function solve(::MCTProblem, ::Solver, ::MemoryKernel); error("This solver is not known"); end