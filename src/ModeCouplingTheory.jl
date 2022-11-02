
"""
Package to solve mode-coupling theory like equations
"""
module ModeCouplingTheory
    using StaticArrays, SparseArrays, LinearAlgebra, Random, LoopVectorization, ProgressMeter, Dierckx
    import LinearSolve
    export solve, FuchsSolver, EulerSolver
    export ModeCouplingKernel, MultiComponentModeCouplingKernel, ExponentiallyDecayingKernel, SchematicDiagonalKernel, SchematicF123Kernel, SchematicF1Kernel, SchematicF2Kernel, SchematicMatrixKernel
    export InterpolatingKernel
    export MCTEquation, LinearMCTEquation
    export MemoryKernel, evaluate_kernel, evaluate_kernel!
    export find_relaxation_time, solve_steady_state

    for file in ["Kernels.jl", "MCTEquation.jl", "Solvers.jl", "RelaxationTime.jl", "SteadyStateMCTEquation.jl"]
        include(file)
    end


end # module
