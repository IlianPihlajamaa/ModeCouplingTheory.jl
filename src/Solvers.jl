abstract type Solver end


"""
    MCTSolution

solution object that holds the solution of an MCTEquation. It has 4 fields
    t: array of t values
    F: array of F for all t
    K: array of K for all T
    solver: solver object that holds the solver settings

an MCT object can be indexed such that 
    sol=MCTSolution
    sol[2]  
gives the F[2] for all t.

"""
struct MCTSolution{T1, T2, T3, T4}
    t::T1
    F::T2
    K::T3
    solver::T4
end

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



import Base.getindex
getindex(sol::MCTSolution, I...) = getindex.(sol.F, I...)
