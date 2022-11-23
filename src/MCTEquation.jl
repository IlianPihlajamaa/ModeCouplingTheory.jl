abstract type MCTEquation end

mutable struct LinearMCTCoefficients{TA,TB,TC,TD}
    α::TA
    β::TB
    γ::TC
    δ::TD
end

"""
    LinearMCTCoefficients(a, b, c, d, F₀)

Constructor of the LinearMCTCoefficients struct, which holds the values of the coefficients α, β, γ and δ. It will convert common cases automatically to make the types compatible.
For example, if α and F0 are both given as vectors, α is converted to a Diagonal matrix. 
"""
function LinearMCTCoefficients(a, b, c, d, F₀)
    a = clean_mulitplicative_coefficient(a, F₀)
    b = clean_mulitplicative_coefficient(b, F₀)
    c = clean_mulitplicative_coefficient(c, F₀)
    d = clean_additive_coefficient(d, F₀)
    return LinearMCTCoefficients(a, b, c, d)
end

clean_mulitplicative_coefficient(coeff::Vector, F₀::Vector) = Diagonal(coeff)
clean_mulitplicative_coefficient(coeff::SVector, F₀::SVector) = Diagonal(coeff)
clean_mulitplicative_coefficient(coeff::Vector, F₀::SVector) = Diagonal(SVector{length(coeff)}(coeff))
clean_mulitplicative_coefficient(coeff::Number, F₀::Vector) = coeff * I
clean_mulitplicative_coefficient(coeff::Number, F₀::SVector) = coeff * I
clean_mulitplicative_coefficient(coeff, F₀) = coeff

clean_additive_coefficient(coeff::Number, F₀::SVector) = coeff .+ F₀ * zero(eltype(F₀))
clean_additive_coefficient(coeff::Number, F₀::Vector) = coeff .+ F₀ * zero(eltype(F₀))
clean_additive_coefficient(coeff::SMatrix, F₀::Vector{<:SMatrix}) = Ref(coeff) .+ F₀ .* Ref(zero(eltype(F₀)))
clean_additive_coefficient(coeff, F₀) = coeff

struct LinearMCTEquation{T,A,B,C,D} <: MCTEquation
    coeffs::T
    F₀::A
    ∂ₜF₀::A
    K₀::B
    kernel::C
    update_coefficients!::D
end


"""
    LinearMCTEquation(α, β, γ, F₀::T, ∂ₜF₀::T, kernel::MemoryKernel) where T

# Arguments:
* `α`: coefficient in front of the second derivative term. If `α` and `F₀` are both vectors, `α` will automatically be converted to a diagonal matrix, to make them compatible.
* `β`: coefficient in front of the first derivative term. If `β` and `F₀` are both vectors, `β` will automatically be converted to a diagonal matrix, to make them compatible.
* `γ`: coefficient in front of the second derivative term. If `γ` and `F₀` are both vectors, `γ` will automatically be converted to a diagonal matrix, to make them compatible.
* `δ`: coefficient in front of the constant term. δ and F0 must by types that can be added together. If δ is a number and F0 is a vector, this conversion will be done automatically.
* `F₀`: initial condition of F(t)
* `∂ₜF₀` initial condition of the derivative of F(t)
* `kernel` instance of a `MemoryKernel` that when called on F₀ and t=0, evaluates to the initial condition of the memory kernel.
"""
function LinearMCTEquation(α, β, γ, δ, F₀, ∂ₜF₀, kernel::MemoryKernel; update_coefficients! = (coeffs, t) -> nothing)
    K₀ = evaluate_kernel(kernel, F₀, 0.0)
    FKeltype = eltype(K₀ * F₀)
    F₀ = FKeltype.(F₀)
    ∂ₜF₀ = FKeltype.(∂ₜF₀)
    coeffs = LinearMCTCoefficients(α, β, γ, δ, F₀)
    update_coefficients!(coeffs, 0.0)
    LinearMCTEquation(coeffs, F₀, ∂ₜF₀, K₀, kernel, update_coefficients!)
end

function Base.show(io::IO, ::MIME"text/plain", p::LinearMCTEquation)
    println(io, "Linear MCT equation object:")
    println(io, "   α F̈ + β Ḟ + γF + δ + ∫K(τ)Ḟ(t-τ) = 0")
    println(io, "in which α is a $(typeof(p.coeffs.α)),")
    println(io, "         β is a $(typeof(p.coeffs.β)),")
    println(io, "         γ is a $(typeof(p.coeffs.γ)),")
    println(io, "         δ is a $(typeof(p.coeffs.δ)),")
    println(io, "  and K(t) is a $(typeof(p.kernel)).")
end