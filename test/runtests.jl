using Test, ModeCouplingTheory
using SpecialFunctions
using ForwardDiff
using Dierckx
using DelimitedFiles
using StaticArrays
using LinearAlgebra
using SparseArrays


for target in ["scalar", "vector", "MCT", "MCMCT", "MCTvsMCMCT", "relaxationtime", "steady_state", "differentiation", "interpolating_kernel"]
    @testset "$target" begin
        include("test_$target.jl")
    end
end
