using Test, ModeCouplingTheory
using SpecialFunctions
using ForwardDiff
using Dierckx
using DelimitedFiles
using StaticArrays
using LinearAlgebra


for target in ["scalar", "vector", "MCT", "MCMCT", "MCTvsMCMCT", "relaxationtime", "steady_state", "differentiation"]
    @testset "$target" begin
        include("test_$target.jl")
    end
end
