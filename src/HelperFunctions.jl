"""
    regula_falsi(x0::Float64, x1::Float64, f; accuracy=10^-10, max_iterations=10^4)

Implements a simple "regular falsi" search to find a point x where
f(x)=0 within the given accuracy, starting in the interval [x0,x1].
For purists, f(x) should have exactly one root in the given interval,
and the scheme will then converge to that. The present code is a bit
more robust, at the expense of not guaranteeing convergence strictly.

Returns a point x such that approximately f(x)==0 (unless the maximum
number of iterations has been exceeded). If exactly one of
the roots of f(x) lie in the initially given interval [x0,x1],
the returned x will be the approximation to that root.

# Arguments:
* `x0`: lower bound of the initial search interval
* `x1`: upper bound of the initial search interval
* `f`: a function accepting one real value, return one real value
"""
function regula_falsi(x0, x1, f; accuracy=10^-10, max_iterations::Integer=10^4)
    iterations = 0
    xa, xb = x0, x1
    fa, fb = f(x0), f(x1)
    dx = xb-xa
    xguess = (xa + xb)/2
    while iterations==0 || (dx > accuracy && iterations <= max_iterations)
        # regula falsi: estimate the zero of f(x) by the secant,
        # check f(xguess) and use xguess as one of the new endpoints of
        # the interval, such that f has opposite signs at the endpoints
        # (and hence a root of f(x) should be inside the interval)
        xguess = xa - (dx / (fb-fa)) * fa
        fguess = f(xguess)
        # we catch the cases where the secant extrapolates outside the interval
        if xguess < xa
            xb,fb = xa,fa
            xa,fa = xguess,fguess
        elseif xguess > xb
            xa,fa = xb,fb
            xb,fb = xguess,fguess
        elseif (fguess>0 && fa<0) || (fguess<0 && fa>0)
            # f(xguess) and f(a) have opposite signs => search in [xa,xguess]
            xb,fb = xguess,fguess
        else
            # f(xguess) and f(b) have opposite signs => search in [xxguess,xb]
            xa,fa = xguess,fguess
        end
        iterations += 1
        dx = xb - xa
    end
    return xguess
end


isimmutabletype(x) = !ismutabletype(x)
check_if_diag(::Diagonal) = true
check_if_diag(::Any) = false

""""
    mymul!(c,a,b,α,β)

prescribes how types used in this solver should be multiplied in place. In particular, it performs
C.= β*C .+ α*a*b. defaults to mul!(c,a,b,α,β)
"""
mymul!(c, a, b, α, β) = mul!(c, a, b, α, β)
function mymul!(c::Vector{SMatrix{Ns,Ns,T,Ns2}}, a::Number, b::Vector{SMatrix{Ns,Ns,T,Ns2}}, α::Number, β::Number) where {Ns,Ns2,T}
    α2 = T(α)
    β2 = T(β)
    for ik in eachindex(c)
        c[ik] = β2 * c[ik] + α2 * a * b[ik]
    end
end

function mymul!(c::Vector{SMatrix{Ns,Ns,T,Ns2}}, a::UniformScaling, b::Vector{SMatrix{Ns,Ns,T,Ns2}}, α::Number, β::Number) where {Ns,Ns2,T}
    α2 = T(α)
    β2 = T(β)
    aλ = a.λ
    for ik in eachindex(c)
        c[ik] = β2 * c[ik] + α2 * aλ * b[ik]
    end
end

function mymul!(c::Vector{SMatrix{Ns,Ns,T,Ns2}}, a::Diagonal{SMatrix{Ns,Ns,T,Ns2},Vector{SMatrix{Ns,Ns,T,Ns2}}}, b::Vector{SMatrix{Ns,Ns,T,Ns2}}, α::Number, β::Number) where {Ns,Ns2,T}
    α2 = T(α)
    β2 = T(β)
    adiag = a.diag
    for ik in eachindex(c)
        c[ik] = β2 * c[ik] + α2 * adiag[ik] * b[ik]
    end
end

"""
    find_error(F_new::T, F_old::T) where T

Finds the error between a new and old iteration of F. The returned scalar will be compared 
to the tolerance to establish convergence. 
"""
function find_error(F_new::T, F_old::T) where {T}
    return maximum(abs.(F_new - F_old))
end

function find_error(F_new::T, F_old::T) where {T<:Vector}
    error = zero(eltype(eltype(F_old)))
    for i in eachindex(F_old)
        new_error = abs(maximum(F_new[i] - F_old[i]))
        if new_error > error
            error = new_error
        end
    end
    return error
end

function find_error(F_new::Number, F_old::Number)
    return abs(F_new - F_old)
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


"""
    `convert_multicomponent_structure_factor(Sk_in::Matrix{Vector{T}})`` 

This function takes a matrix of vectors Sk_in where each vector 
contains structure factor information for different species 
at different k-points and converts it into a vector of SMatrix.

Input:

 - Sk_in::Matrix{Vector{T}} - A square matrix of vectors, where each vector contains structure factor information for a specific set of species at different k-points.

Output:

 - Sk_out - A Vector of SMatrix where each SMatrix contains structure factor information for different species at a single k-point.

Example

```julia
Ns = 2;
Nk = 3;
S11 = [1,2,3]; S21 = [4,5,6]; S22 = [8,9,10];
S = [zeros(Nk) for i=1:2, j=1:2];
S[1,1] = S11; S[1,2] = S21; S[2,1] = S21; S[2,2] = S22;
convert_multicomponent_structure_factor(S)

    3-element Vector{StaticArraysCore.SMatrix{2, 2, Float64, 4}}:
    [1.0 4.0; 4.0 8.0]
    [2.0 5.0; 5.0 9.0]
    [3.0 6.0; 6.0 10.0]
```

"""
function convert_multicomponent_structure_factor(Sk_in::Matrix{Vector{T}}) where T 
    if !(size(Sk_in, 1) == size(Sk_in, 2))
        error("the input must be a square matrix of species")
    end
    Ns = size(Sk_in, 1)
    Nk = length(Sk_in[1,1])
    if !all(length.(Sk_in) .== Nk)
        error("There are not an equal number of k-points for every specie")
    end
    for α = 1:Ns
        for β = 1:α-1
            if !(all(Sk_in[α,β] .≈ Sk_in[β, α]))
                error("The structure factor is not symmetric in species $α and $β")
            end
        end
    end
    Sk_out = Vector{SMatrix{Ns, Ns, T, Ns*Ns}}()
    for i = 1:Nk
        Ski = SMatrix{Ns,Ns}(getindex.(Sk_in, i))
        push!(Sk_out, Ski)
    end
    return Sk_out
end
