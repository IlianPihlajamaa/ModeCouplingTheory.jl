
"""
    find_relaxation_time(t, F; threshold=exp(-1), mode=:log)

Finds the time at which a signal first decreases below some `threshold`. Uses interpolation to find an accurate result.
if it never decreases below the threshold, returns `Inf`.

# Arguments:
* `t`: a vector of time values at which the signal `F` is known
* `F`: the signal
* `threshold`: see above
* `mode`: uses interpolation either in a logarithmic or linear space. 
"""
function find_relaxation_time(t, F::Vector{<:Number}; threshold=exp(-1), mode=:log)
    Nt = length(t)
    for it = 1:Nt
        if all(F[it]/F[1] .> threshold)
            continue
        end
        if it == 1
            return zero(eltype(t))
        end
        y0 = F[it]/F[1]
        y1 = F[it-1]/F[1]
        y = threshold
        if mode == :log
            x0 = log(t[it])
            x1 = log(t[it-1])
            return exp(((y-y0)*(x1-x0)+x0*(y1-y0))/(y1-y0))
        elseif mode == :lin
            x0 = t[it]
            x1 = t[it-1]
            return ((y-y0)*(x1-x0)+x0*(y1-y0))/(y1-y0)
        else
            error("the mode ':$mode' is not know. it must be either :lin, :log.")
        end
    end
    return Inf
end
