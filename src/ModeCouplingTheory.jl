
"""
Package to solve mode-coupling theory like equations
"""
module ModeCouplingTheory
    using StaticArrays, SparseArrays, LinearAlgebra, Random, Tullio, LoopVectorization, ProgressMeter
    export solve, FuchsSolver, EulerSolver
    export ModeCouplingKernel, MultiComponentModeCouplingKernel, ExponentiallyDecayingKernel, SchematicDiagonalKernel, SchematicF123Kernel, SchematicF1Kernel, SchematicF2Kernel, SchematicMatrixKernel
    export MCTProblem
    export MemoryKernel
    export find_relaxation_time, solve_steady_state

    for file in ["Kernels.jl", "MCTProblem.jl", "Solvers.jl", "RelaxationTime.jl", "SteadyStateMCTProblem.jl"]
        include(file)
    end


end # module
