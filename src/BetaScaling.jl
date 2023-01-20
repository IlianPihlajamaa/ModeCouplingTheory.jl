# here we have a few modifications that allow for the solver
# to be used in cases where a memory kernel is not separately
# defined, nor stored
#
# these could be moved to a separate file or integrated with
# TimeDoublingSolver.jl, because semantically they belong there
#
# not sure how much sense this makes in general, but it comes
# in handy for the beta-scaling equation, requires little code
# to be modified, and allocates less arrays where only zeros will
# be stored, and is a generic modification that other models can
# potentially reuse (corrections-to-beta scaling etc.)
# in the hope to increase reusability, I kept some arrays that
# are not used by the beta solver (marked with a "not used" comment
# below)
# of course, we could go with just allocating dummy arrays for K and K_I
# in the beta solver
mutable struct FuchsNoMemTempStruct{T,T2,T3,VT,TT,SC} <: AbstractFuchsTempStruct
    F_temp::VT
    F_I::VT
    C1::T3
    C1_temp::T3
    C2::T
    C3::T
    temp_vec::TT
    F_old::T
    temp_mat::T2
    solve_cache::SC
    inplace::Bool
    start_time::Float64
end

function initialize_temporary_arrays!(equation::AbstractMemoryEquation, solver::TimeDoublingSolver, kernel, temp_arrays::FuchsNoMemTempStruct)
    initialize_F_temp!(equation, solver, temp_arrays)
    initialize_integrals!(equation, solver, temp_arrays)
end

# version of update_K_and_F! that doesn't update the kernel
# intended for models that don't need to calculate the kernel separately
function update_K_and_F!(equation::AbstractMemoryEquation, solver::TimeDoublingSolver, kernel, temp_arrays::FuchsNoMemTempStruct, it::Int)
    update_F!(equation, solver, temp_arrays, it)
end

function update_integrals!(temp_arrays::FuchsNoMemTempStruct, it::Int)
    F_I = temp_arrays.F_I
    F_temp = temp_arrays.F_temp
    F_I[it] = (F_temp[it] + F_temp[it-1]) / 2
end

function allocate_results!(t_array, F_array, K_array, solver::TimeDoublingSolver, temp_arrays::FuchsNoMemTempStruct; istart=2(solver.N) + 1, iend=4(solver.N))
    N = solver.N
    δt = solver.Δt / (4N)
    for it = istart:iend
        t = δt * it
        push!(t_array, t)
        push!(F_array, deepcopy(temp_arrays.F_temp[it]))
    end
end

function new_time_mapping!(equation::AbstractMemoryEquation, solver::TimeDoublingSolver, temp_arrays::FuchsNoMemTempStruct)
    F = temp_arrays.F_temp
    F_I = temp_arrays.F_I
    N = solver.N
    for j = 1:N
        F_I[j] = (F_I[2j] + F_I[2j-1]) / 2
        F[j] = F[2j]
    end
    for j = (N+1):2*N
        # Flenner/Szamel version:
        #F_I[j] = (F_I[2j] + 4 * F_I[2j-1] + F_I[2j-2]) / 6
        # Hofacker/Fuchs version:
        #F_I[j] = (F_I[2j] + F_I[2j-1]) / 2
        # Hofacker/Fuchs that does not require extra update_integrals!
        F_I[j] = (F[2j] + 2*F[2j-1] + F[2j-2]) / 4
        F[j] = F[2j]
    end
    for j = 2N+1:4N
        F_I[j] = equation.F₀ * zero(eltype(eltype(F_I)))
        F[j] = equation.F₀ * zero(eltype(eltype(F)))
    end
    solver.Δt *= 2
end


# here starts the code for BetaScaling.jl proper



mutable struct BetaScalingEquationCoefficients{T}
    λ::T
    σ::T
    t₀::T
    δ::T
    δ_times_t::T
    a::T
    b::T
end

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
function regula_falsi(x0::Float64, x1::Float64, f; accuracy::Float64=10^-10, max_iterations::Integer=10^4)
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

function BetaScalingEquationCoefficients(λ, σ, t₀, δ)
    exponent_func = (x) -> (SpecialFunctions.gamma(1-x)/SpecialFunctions.gamma(1-2x))*SpecialFunctions.gamma(1-x) - λ
    a = regula_falsi(0.2, 0.3, exponent_func)
    b = - regula_falsi(-0.5, -0.1, exponent_func)
    return BetaScalingEquationCoefficients(λ, σ, t₀, δ, Float64(0.0), a, b)
end


struct BetaScalingEquation{T,A,B,C,D} <: AbstractMemoryEquation
    coeffs::T
    F₀::A
    K₀::B
    kernel::C
    update_coefficients!::D
end

"""
    BetaScalingEquation(λ, σ, t₀, δ=0.0)

Defines the β-scaling equation of MCT, which is a scalar equation for
a single scaling function g(t), determined by scalar parameters
λ (the MCT exponent parameter, 1/2<=λ<1), the distance parameter to
the glass transition σ, and (for convenience) an arbitrary time scale
t₀ that just shifts the results. Optionally, a "hopping parameter"
δ can be given (defaults to zero).

The MCT exponents a and b will be automatically calculated from λ.
"""
function BetaScalingEquation(λ::Float64, σ::Float64, t₀::Float64; δ::Float64=0.0)
    BetaScalingEquation(BetaScalingEquationCoefficients(λ, σ, t₀, δ), Float64(0.0), nothing, nothing, (coeffs,t) -> coeffs.δ_times_t = coeffs.δ*t)
end

function Base.show(io::IO, ::MIME"text/plain", p::BetaScalingEquation)
    println(io, "MCT beta-scaling object:")
    println(io, "   σ - δ t + λ (g(t))² = ∂ₜ∫g(t-τ)g(τ)dτ")
    println(io, "with real-valued parameters.")
end

function allocate_temporary_arrays(equation::BetaScalingEquation, solver::TimeDoublingSolver)
    start_time = time()
    F_temp = Float64[]
    F_I = Float64[]
    temp_arrays = FuchsNoMemTempStruct(F_temp, F_I, Float64(0.0), Float64(0.0), Float64(0.0), Float64(0.0), nothing, Float64(0.0), nothing, nothing, false, start_time)
    for _ in 1:4*solver.N
        push!(temp_arrays.F_temp, 1.)
        push!(temp_arrays.F_I, 1.)
    end
    return temp_arrays
end

"""
    initialize_F_temp!(equation::BetaScalingEquation, solver::TimeDoublingSolver, temp_arrays::FuchsNoMemTempStruct)

Fills the first 2N entries of the temporary arrays needed for solving the
β-scaling equation with a an adapted Fuchs scheme.

In particular, this initializes g(t) = (t/t₀)^-a.

"""
function initialize_F_temp!(equation::BetaScalingEquation, solver::TimeDoublingSolver, temp_arrays::FuchsNoMemTempStruct)
    N = solver.N
    δt = solver.Δt / (4 * N)
    t₀ = equation.coeffs.t₀
    a = equation.coeffs.a

    for it = 1:2N
        temp_arrays.F_temp[it] = (δt*it/t₀)^-a
    end
end

"""
    initialize_integrals!(equation::BetaScalingEquation, solver::TimeDoublingSolver, temp_arrays::FuchsNoMemTempStruct)

Initializes the integrals over the first 2N time points for the solution
of the β-scaling equation, using the known critical decay law as the
short-time asymptote.

"""
function initialize_integrals!(equation::BetaScalingEquation, solver::TimeDoublingSolver, temp_arrays::FuchsNoMemTempStruct)
    F_I = temp_arrays.F_I
    F_temp = temp_arrays.F_temp
    N = solver.N
    δt = solver.Δt / (4 * N)
    t₀ = equation.coeffs.t₀
    a = equation.coeffs.a

    F1 = (t₀/δt)^a / (1-a)
    F_I[1] = F1
    for it = 2:2N
        F_I[it] = F1 * (it^(1-a) - (it-1)^(1-a))
    end
end

"""
    update_Fuchs_parameters!(equation::BetaScalingEquation, solver::TimeDoublingSolver, temp_arrays::FuchsNoMemTempStruct, it::Int)

Updates the parameters that are needed to solve the β-scaling equation
numerically with Fuchs' scheme.
"""
function update_Fuchs_parameters!(equation::BetaScalingEquation, solver::TimeDoublingSolver, temp_arrays::FuchsNoMemTempStruct, it::Int)
    N = solver.N
    i2 = 2N
    δt = solver.Δt / (4N)
    F_I = temp_arrays.F_I
    F = temp_arrays.F_temp
    equation.update_coefficients!(equation.coeffs, δt*it)
    λ = equation.coeffs.λ
    σ = equation.coeffs.σ
    δ_times_t = equation.coeffs.δ_times_t

    temp_arrays.C1 = 2*F_I[1]
    temp_arrays.C2 = λ

    c3 = -F[it-i2]*F[i2] + 2*F[it-1]*F_I[1]
    c3 += σ - δ_times_t 
    @inbounds for j = 2:i2
        c3 += (F[it-j] - F[it-j+1]) * F_I[j]
    end
    @inbounds for j = 2:it-i2
        c3 += (F[it-j] - F[it-j+1]) * F_I[j]
    end
    #@inbounds if it-i2 != i2
    #    c3 += (F[i2] - F[i2+1]) * F_I[it-i2]
    #end
    temp_arrays.C3 = c3
end

function update_F!(::BetaScalingEquation, ::TimeDoublingSolver, temp_arrays::AbstractFuchsTempStruct, it::Int)
    c1 = temp_arrays.C1
    c2 = temp_arrays.C2
    c3 = temp_arrays.C3
    # F_old = temp_arrays.F_temp[it]
    temp_arrays.F_temp[it] = c1/(2c2)-sqrt((c1/(2c2))^2-c3/c2)
    #temp_arrays.F_temp[it] = c1 \ (F_old*F_old*c2 + c3)
end

