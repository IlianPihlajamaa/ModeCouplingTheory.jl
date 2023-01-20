abstract type Solver end


"""
    MemoryEquationSolution

solution object that holds the solution of an AbstractMemoryEquation. It has 4 fields
    t: array of t values
    F: array of F for all t
    K: array of K for all T
    solver: solver object that holds the solver settings

an MCT object can be indexed such that 
    sol=MemoryEquationSolution
    sol[2]  
gives the F[2] for all t.

"""
struct MemoryEquationSolution{T1, T2, T3, T4}
    t::T1
    F::T2
    K::T3
    solver::T4
end

include("EulerSolver.jl")
include("TimeDoublingSolver.jl")

"""
    solve(equation::AbstractMemoryEquation, solver::Solver)

Solves the `AbstractMemoryEquation` with the provided `kernel` using `solver`. 
Search for a specific solver or kernel object to find more specific information.

If no solver is provided, it uses the default TimeDoublingSolver. 
    
# Returns:
* `t` an array of time values
* `F` The solution in an array of which the last dimension corresponds to the time.
* `K` The memory kernel corresponding to each `F`
"""
function solve(::AbstractMemoryEquation, ::Solver); error("This solver is not known"); end

solve(p::AbstractMemoryEquation) = solve(p, TimeDoublingSolver())


import Base.getindex
getindex(sol::MemoryEquationSolution, I...) = getindex.(sol.F, I...)
