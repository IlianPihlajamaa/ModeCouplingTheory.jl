abstract type Solver end
include("EulerSolver.jl")
include("FuchsSolver.jl")

"""
    solve(problem::MCTProblem, solver::Solver, kernel::MemoryKernel)

Solves the `MCTproblem` with the provided `kernel` using `solver`. 
Search for a specific solver or kernel object to find more specific information.
"""
function solve(::MCTProblem, ::Solver, ::MemoryKernel); error("This solver is not known"); end