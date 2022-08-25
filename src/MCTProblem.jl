struct MCTProblem{T1, T2, T3, A, B}
    α::T1
    β::T2
    γ::T3
    F₀::A
    ∂ₜF₀::A
    K₀::B
    Ftype::DataType
    Kerneltype::DataType
    FK_elementtype::DataType
end


"""
    MCTProblem(α, β, γ, F₀::T, ∂ₜF₀::T, kernel::MemoryKernel) where T

Constructor of the `MCTProblem` type. It requires that `F₀` and `∂ₜF₀` have the same type.

# Arguments:
* `α`: coefficient in front of the second derivative term. If `α` and `F₀` are both vectors, `α` will automatically be converted to a diagonal matrix, to make them compatible.
* `β`: coefficient in front of the first derivative term. If `β` and `F₀` are both vectors, `β` will automatically be converted to a diagonal matrix, to make them compatible.
* `γ`: coefficient in front of the second derivative term. If `γ` and `F₀` are both vectors, `γ` will automatically be converted to a diagonal matrix, to make them compatible.
* `F₀`: initial condition of F(t)
* `∂ₜF₀` initial condition of the derivative of F(t)
* `kernel` instance of a `MemoryKernel` that when called on F₀ and t=0, evaluates to the initial condition of the memory kernel.

"""
function MCTProblem(α, β, γ, F₀::T, ∂ₜF₀::T, kernel::MemoryKernel) where T

    Tnew = typeof(F₀)
    K₀ = evaluate_kernel(kernel, F₀, 0.0)
    Ftype = typeof(K₀*F₀)
    Ktype = typeof(K₀)

    
    FK_element_type = eltype(Ftype)

    if Tnew<:Number && α isa Number && β isa Number && γ isa Number
        return MCTProblem(α, β, γ, F₀, ∂ₜF₀, K₀, Ftype, Ktype, FK_element_type)
    end

    if Tnew<:Vector 
        if α isa Number
            A = α*I
        elseif α isa Vector
            A = Diagonal(α)
        else
            A = copy(α)
        end
        if β isa Number
            B = β*I
        elseif β isa Vector
            B = Diagonal(β)
        else
            B = copy(β)
        end
        if γ isa Number
            C = γ*I
        elseif γ isa Vector
            C = Diagonal(γ)
        else
            C = copy(γ)
        end
        return MCTProblem(A, B, C, F₀, ∂ₜF₀, K₀, Ftype, Ktype, FK_element_type)
    end
    if Tnew<:SVector 
        if α isa Number
            A = α*I
        elseif α isa SVector
            A = Diagonal(α)
        else
            A = α
        end
        if β isa Number
            B = β*I
        elseif β isa SVector
            B = Diagonal(β)
        else
            B = β
        end
        if γ isa Number
            C = γ*I
        elseif γ isa SVector
            C = Diagonal(γ)
        else
            C = copy(γ)
        end
        return MCTProblem(A, B, C, F₀, ∂ₜF₀, K₀, Ftype, Ktype, FK_element_type)
    end
    MCTProblem(α, β, γ, F₀, ∂ₜF₀, K₀, Ftype, Ktype, FK_element_type)
end
