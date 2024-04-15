
# using ModeCouplingTheory, LinearAlgebra, StaticArrays
# import ModeCouplingTheory: MemoryKernel, evaluate_kernel

# test 
struct MyWeirdKernel{T} <: MemoryKernel
    α :: T
end

import ModeCouplingTheory.evaluate_kernel


function evaluate_kernel(kernel::MyWeirdKernel, F, t)
    out = Diagonal(zeros(eltype(F), size(F)))
    evaluate_kernel!(out, kernel, F, t)
    return out
end

function evaluate_kernel(kernel::MyWeirdKernel, F::Number, t)
    return kernel.α * exp(-t)   
end

function evaluate_kernel!(out, kernel::MyWeirdKernel, F::AbstractMatrix, t)
    out .= kernel.α * exp(-t)
end

function evaluate_kernel!(out, kernel::MyWeirdKernel, F::AbstractVector, t)
    out.diag .= kernel.α.diag .* exp(-t)
end


solver = TimeDoublingSolver(Δt = 10^-4, t_max=10.0^5, ismutable=false)

kernel1 = MyWeirdKernel(Diagonal(ones(10)))
α = 0.0; β = 1.0; γ = Diagonal(ones(10)); δ = zeros(10); F0 = ones(10); F1 = zeros(10)
problem1 = MemoryEquation(α, β, γ, δ, F0, F1, kernel1)
sol1 = solve(problem1, solver)

kernel2 = MyWeirdKernel(Diagonal(ones(10)))
α = 0.0; β = 1.0; γ = Diagonal(ones(10)); δ = Diagonal(zeros(10)); F0 = Diagonal(ones(10)); F1 = Diagonal(zeros(10))
problem2 = MemoryEquation(α, β, γ, δ, F0, F1, kernel2)
sol2 = solve(problem2, solver)

kernel3 = MyWeirdKernel(1.0)
α = 0.0; β = 1.0; γ = 1.0; δ = 0.0; F0 = 1.0; F1 = 0.0
problem3 = MemoryEquation(α, β, γ, δ, F0, F1, kernel3)
sol3 = solve(problem3, solver)

kernel4 = MyWeirdKernel(Diagonal(ones(SMatrix{2,2,Float64,4},10)))
α = 0.0; β = 1.0; γ = Diagonal(ones(SMatrix{2,2,Float64,4},10)); δ = Diagonal(zeros(SMatrix{2,2,Float64,4},10)); F0 = Diagonal(ones(SMatrix{2,2,Float64,4},10)); F1 = Diagonal(zeros(SMatrix{2,2,Float64,4},10))
problem4 = MemoryEquation(α, β, γ, δ, F0, F1, kernel4)
sol4 = solve(problem4, solver)

dat1 = get_F(sol1, :, 1)
dat2 = get_F(sol2, :, (1,1))
dat3 = get_F(sol3, :)
dat4 = get_F(sol4, :, (1,1), (1,1))

@test allequal(dat1, dat2)
@test allequal(dat1, dat3)
@test allequal(dat1, dat4)


