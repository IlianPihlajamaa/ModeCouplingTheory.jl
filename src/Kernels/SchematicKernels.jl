"""
    ExponentiallyDecayingKernel{T<:Number} <: MemoryKernel

Scalar kernel with fields `ν` and `τ` which when called returns `ν exp(-t/τ)`.
"""
struct ExponentiallyDecayingKernel{T1<:Number,T2<:Number} <: MemoryKernel
    ν::T1
    τ::T2
end

function evaluate_kernel(kernel::ExponentiallyDecayingKernel, F::Number, t)
    return kernel.ν * exp(-t / kernel.τ)
end


"""
    SchematicF1Kernel{T<:Number} <: MemoryKernel

Scalar kernel with field `ν` which when called returns `ν F`.
"""
struct SchematicF1Kernel{T<:Number} <: MemoryKernel
    ν::T
end

function evaluate_kernel(kernel::SchematicF1Kernel, F::Number, t)
    ν = kernel.ν
    return ν * F
end

"""
    SchematicF2Kernel{T<:Number} <: MemoryKernel

Scalar kernel with field `ν` which when called returns `ν F^2`.
"""
struct SchematicF2Kernel{T<:Number} <: MemoryKernel
    ν::T
end

function evaluate_kernel(kernel::SchematicF2Kernel, F::Number, t)
    ν = kernel.ν
    return ν * F^2
end

"""
    SchematicF2Kernel{T<:Number} <: MemoryKernel

Scalar tagged particle kernel with field `ν` which when called returns `ν F*Fs`.
"""
struct TaggedSchematicF2Kernel{T1,T2,T3} <: MemoryKernel
    ν::T1
    tDict::T2
    F::T3
end

function TaggedSchematicF2Kernel(ν, sol)
    tDict = Dict(zip(sol.t, eachindex(sol.t)))
    return TaggedSchematicF2Kernel(ν, tDict, sol.F)
end

function evaluate_kernel(kernel::TaggedSchematicF2Kernel, Fs::Number, t)
    ν = kernel.ν
    F = kernel.F[kernel.tDict[t]]
    return ν * F * Fs
end

"""
    SjogrenKernel

Memory kernel that implements the kernel `K[1] = ν1 F[1]^2`, `K[2] = ν2 F[1] F[2]`. Consider using Static Vectors for performance. 
"""
struct SjogrenKernel{T} <: MemoryKernel
    ν1::T
    ν2::T
end

function evaluate_kernel(kernel::SjogrenKernel, F, t)
    return Diagonal(@SVector [kernel.ν1 * F[1]^2, kernel.ν2 * F[1] * F[2]])
end


"""
    SchematicF1Kernel{T<:Number} <: MemoryKernel

Scalar kernel with fields `ν1`, `ν2`, and `ν3` which when called returns `ν1 * F^1 + ν2 * F^2 + ν3 * F^3`.
"""
struct SchematicF123Kernel{T<:Number} <: MemoryKernel
    ν1::T
    ν2::T
    ν3::T
end

function evaluate_kernel(kernel::SchematicF123Kernel, F::Number, t)
    return kernel.ν1 * F^1 + kernel.ν2 * F^2 + kernel.ν3 * F^3
end


"""
    SchematicDiagonalKernel{T<:Union{SVector, Vector}} <: MemoryKernel

Matrix kernel with field `ν` which when called returns `Diagonal(ν .* F .^ 2)`, i.e., it implements a non-coupled system of SchematicF2Kernels.
"""
struct SchematicDiagonalKernel{T<:Union{SVector,Vector}} <: MemoryKernel
    ν::T
    SchematicDiagonalKernel(ν::T) where {T<:Union{SVector,Vector}} = eltype(ν) <: Number ? new{T}(ν) : error("element type of this kernel must be a number")
end

function evaluate_kernel(kernel::SchematicDiagonalKernel, F::Union{SVector,Vector}, t)
    ν = kernel.ν
    return Diagonal(ν .* F .^ 2)
end

function evaluate_kernel!(out::Diagonal, kernel::SchematicDiagonalKernel, F::Vector, t)
    ν = kernel.ν
    diag = out.diag
    @. diag = ν * F^2
end

"""
    SchematicMatrixKernel{T<:Union{SVector, Vector}} <: MemoryKernel

Matrix kernel with field `ν` which when called returns `ν * F * Fᵀ`, i.e., it implements Kαβ = ν*Fα*Fβ.
"""
struct SchematicMatrixKernel{T<:AbstractMatrix} <: MemoryKernel
    ν::T
    SchematicMatrixKernel(ν::T) where {T} = eltype(ν) <: Number ? new{T}(ν) : error("element type of this kernel must be a number")
end

function evaluate_kernel(kernel::SchematicMatrixKernel, F::Union{SVector,Vector}, t)
    ν = kernel.ν
    return ν * F * F'
end

function evaluate_kernel!(out::Matrix, kernel::SchematicMatrixKernel, F::Vector, t)
    ν = kernel.ν
    out .= zero(eltype(out))
    for i in eachindex(F)
        for j in eachindex(F)
            for k in eachindex(F)
                out[i, j] += ν[i, k] * F[k] * F[j]
            end
        end
    end
end

struct InterpolatingKernel{T} <: MemoryKernel
    M::T
end

"""
    InterpolatingKernel(t, M; k=1)

Uses the package `Dierckx` to provide a kernel that interpolates the data M defined on grid t using spline interpolation of degree `k`. 
"""
function InterpolatingKernel(t, M; k=1)
    InterpolatingKernel(Spline1D(t, M, k=k))
end

function evaluate_kernel(kernel::InterpolatingKernel, F::Number, t)
    return kernel.M(t)
end