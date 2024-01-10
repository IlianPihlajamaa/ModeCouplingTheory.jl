
"""
Package to solve mode-coupling theory like equations
"""
module ModeCouplingTheory
    using StaticArrays, SparseArrays, LinearAlgebra, Random, LoopVectorization, Dierckx, Printf, SpecialFunctions
    import LinearSolve
    
    export solve, TimeDoublingSolver, EulerSolver
    export ModeCouplingKernel, MultiComponentModeCouplingKernel,  ExponentiallyDecayingKernel, SchematicDiagonalKernel, SchematicF123Kernel, SchematicF1Kernel, SchematicF2Kernel, SchematicMatrixKernel
    export SjogrenKernel, TaggedSchematicF2Kernel, TaggedModeCouplingKernel, TaggedMultiComponentModeCouplingKernel
    export InterpolatingKernel
    export AbstractMemoryEquation, MemoryEquation
    export MemoryKernel, evaluate_kernel, evaluate_kernel!
    export find_relaxation_time, solve_steady_state
    export convert_multicomponent_structure_factor
    export MSDModeCouplingKernel, MSDMultiComponentModeCouplingKernel, dDimMSDModeCouplingKernel
    export get_F, get_K, get_t
    abstract type AbstractMemoryEquation end

    for file in ["Kernels.jl", "MemoryEquation.jl", "Solvers.jl", "HelperFunctions.jl", "RelaxationTime.jl", "SteadyStateMemoryEquation.jl"]
        include(file)
    end

    if ccall(:jl_generating_output, Cint, ()) == 1   # if we're precompiling the package, run a simple scalar and MCT code
        let 
            kernel1 = ModeCouplingTheory.SchematicF2Kernel(0.1)
            system1 = MemoryEquation(1.0, 0.0, 1.0, 0.0, 1.0, 0.0, kernel1)
            solver1 = TimeDoublingSolver(; Δt=10^-3, t_max=10.0^-2, verbose=false, N = 2, tolerance=10^-4, max_iterations=10^6)
            sol1 =  solve(system1, solver1)
        end
        let
            η = 0.1; ρ = η*6/π; kBT = 1.0; m = 1.0
            Nk = 5; kmax = 40.0; dk = kmax/Nk; k_array = dk*(collect(1:Nk) .- 0.5)
            F0 = ones(Nk); ∂F0 = zeros(Nk); α = 1.0; β = 0.0; γ = @. k_array^2*kBT/(m); δ = 0.0
            kernel = ModeCouplingKernel(ρ, kBT, m, k_array, F0)
            problem = MemoryEquation(α, β, γ, δ, F0, ∂F0, kernel)
            solver = TimeDoublingSolver(Δt=10^-5, t_max=10.0^-3, N = 2, tolerance=10^-4)
            sol = solve(problem, solver);
            taggedkernel = TaggedModeCouplingKernel(ρ, kBT, m, k_array, F0, sol)
            taggedproblem = MemoryEquation(α, β, γ, δ, F0, ∂F0, taggedkernel)
            sols = solve(taggedproblem, solver)
        end
    end


end # module
