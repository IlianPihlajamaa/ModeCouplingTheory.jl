using StaticArrays, SparseArrays, LinearAlgebra, Random, BenchmarkTools, Tullio, LoopVectorization, ProgressMeter


module ModeCouplingTheory
    for file in ["Kernels.jl", "MCTProblem.jl", "Solvers.jl", "RelaxationTime.jl", "SteadyStateMCTProblem.jl"]
        include(file)
    end
end # module
