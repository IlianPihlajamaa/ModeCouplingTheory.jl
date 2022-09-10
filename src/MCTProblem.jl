abstract type MCTProblem end

struct LinearMCTProblem{T1,T2,T3,A,B,C} <: MCTProblem
    α::T1
    β::T2
    γ::T3
    F₀::A
    ∂ₜF₀::A
    K₀::B
    kernel::C
end


"""
    LinearMCTProblem(α, β, γ, F₀::T, ∂ₜF₀::T, kernel::MemoryKernel) where T

# Arguments:
* `α`: coefficient in front of the second derivative term. If `α` and `F₀` are both vectors, `α` will automatically be converted to a diagonal matrix, to make them compatible.
* `β`: coefficient in front of the first derivative term. If `β` and `F₀` are both vectors, `β` will automatically be converted to a diagonal matrix, to make them compatible.
* `γ`: coefficient in front of the second derivative term. If `γ` and `F₀` are both vectors, `γ` will automatically be converted to a diagonal matrix, to make them compatible.
* `F₀`: initial condition of F(t)
* `∂ₜF₀` initial condition of the derivative of F(t)
* `kernel` instance of a `MemoryKernel` that when called on F₀ and t=0, evaluates to the initial condition of the memory kernel.
"""
function LinearMCTProblem(α, β, γ, F₀, ∂ₜF₀, kernel::MemoryKernel)
    K₀ = evaluate_kernel(kernel, F₀, 0.0)
    FKeltype = eltype(K₀ * F₀)
    F₀ = FKeltype.(F₀) # make sure F0 has the right eltype
    ∂ₜF₀ = FKeltype.(∂ₜF₀)
    Tnew = typeof(F₀)


    if Tnew <: Number && α isa Number && β isa Number && γ isa Number
        return LinearMCTProblem(α, β, γ, F₀, ∂ₜF₀, K₀, kernel)
    end

    if Tnew <: Vector
        if α isa Number
            A = α * I
        elseif α isa Vector
            A = Diagonal(α)
        else
            A = copy(α)
        end
        if β isa Number
            B = β * I
        elseif β isa Vector
            B = Diagonal(β)
        else
            B = copy(β)
        end
        if γ isa Number
            C = γ * I
        elseif γ isa Vector
            C = Diagonal(γ)
        else
            C = copy(γ)
        end
        return LinearMCTProblem(A, B, C, F₀, ∂ₜF₀, K₀, kernel)
    end
    if Tnew <: SVector
        if α isa Number
            A = α * I
        elseif α isa SVector
            A = Diagonal(α)
        else
            A = α
        end
        if β isa Number
            B = β * I
        elseif β isa SVector
            B = Diagonal(β)
        else
            B = β
        end
        if γ isa Number
            C = γ * I
        elseif γ isa SVector
            C = Diagonal(γ)
        else
            C = copy(γ)
        end
        return LinearMCTProblem(A, B, C, F₀, ∂ₜF₀, K₀, kernel)
    end
    LinearMCTProblem(α, β, γ, F₀, ∂ₜF₀, K₀, kernel)
end

struct StackedMCTProblem{P <: Tuple} <: MCTProblem
    problems::P
end

import Base.+
+(a::MCTProblem, b::MCTProblem) = StackedMCTProblem((a, b))
+(a::StackedMCTProblem, b::MCTProblem) = StackedMCTProblem((a.problems..., b))
+(a::MCTProblem, b::StackedMCTProblem) = StackedMCTProblem((a, b.problems...))
+(a::StackedMCTProblem, b::StackedMCTProblem) = StackedMCTProblem((a.problems..., b.problems...))


function Base.show(io::IO, ::MIME"text/plain", p::LinearMCTProblem) 
    println(io, "Linear MCT equation object:")
    println(io, "   α F̈ + β Ḟ + γF + ∫K(τ)Ḟ(t-τ) = 0")
    println(io, "in which α is a $(typeof(p.α)),")
    println(io, "         β is a $(typeof(p.β)),")
    println(io, "         γ is a $(typeof(p.γ)),")
    println(io, "  and K(t) is a $(typeof(p.kernel)).")
end

function Base.show(io::IO, t::MIME"text/plain", p::StackedMCTProblem) 
    println(io, "Stack of $(length(p.problems)) MCT Problems:")
    for problem in p.problems
        println(io)
        show(io, t, problem)
    end
end