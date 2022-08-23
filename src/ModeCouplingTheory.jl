

module ModeCouplingTheory
    using StaticArrays, SparseArrays, LinearAlgebra, Random, Tullio, LoopVectorization, ProgressMeter

    for file in ["Kernels.jl", "MCTProblem.jl", "Solvers.jl", "RelaxationTime.jl", "SteadyStateMCTProblem.jl"]
        include(file)
    end

    export solve, FuchsSolver, EulerSolver
    export ModeCouplingKernel, MultiComponentModeCouplingKernel
    export MCTProblem
    export find_relaxation_time, solve_steady_state
end # module
