
struct ActiveMCTKernel <: MemoryKernel
    k_array ::Vector{Float64}
    prefactor ::Float64
    J ::Array{Float64, 3}
    V2 ::Array{Float64, 3} 
end

"""
    ActiveMCTKernel(ρ, k_array, wk, w0, Sk, dim=3)

Implements the following single-component active MCT kernel:
M(k,t) =  ρ / (2(2π)^dim ) ∫ dq w(k) V(k,q)^2 F(q,t) F(k-q,t)

# Arguments:

* ρ: number density
* k_array: array of wavevectors at which to evaluate all quantities
* wk: steady-state velocity correlations ( w(k) )
* w0: "pure" velocity correlations ( w(k→∞) )
* Sk: steady-state structure factor
* dim: dimensionality (dim=2 or dim=3)
"""
function ActiveMCTKernel(ρ, k_array, wk, w0, Sk, dim=3)
    @assert dim == 2 || dim == 3 "This kernel has been tested for dim=2 and dim=3"

    Δk = k_array[2] - k_array[1];
    prefactor = Δk^2* ρ / (2*(2*π)^dim) * surface_d_dim_unit_sphere(dim-1);
    Nk = length(k_array);
    
    mCk = @. 1/ρ * ( 1 - wk / (w0 * Sk) );
    V2  = zeros(Nk, Nk, Nk);
    J   = zeros(Nk, Nk, Nk);

    for i=1:Nk, j=1:Nk, l=abs(j-i)+1:min(i+j-1,Nk)
        k = k_array[i];
        q = k_array[j];
        p = k_array[l];

        cq = mCk[j];
        cp = mCk[l];

        V = cq/(2*k)*(k^2 + q^2 - p^2) + cp/(2*k)*(k^2 + p^2 - q^2);
        V2[l,j,i] = wk[i] * V^2;
        J[l,j,i] = 2*p*q/((2*k)^(dim-2))*( (q+p-k)*(k+p-q)*(k+q-p)*(k+p+q) )^((dim-3)/2);
    end
    return ActiveMCTKernel(k_array, prefactor, J, V2)
end

function evaluate_kernel!(out::Diagonal, kernel::ActiveMCTKernel, F, t)
    out.diag .= zero(eltype(out.diag))
    k_array = kernel.k_array
    Nk = length(k_array)
    
    for i = 1:Nk, j = 1:Nk, l=abs(j-i)+1:min(i+j-1,Nk)
        out.diag[i] += kernel.J[l,j,i] * kernel.V2[l, j, i] * F[j] * F[l]
    end
    out.diag .*= kernel.prefactor
end

function evaluate_kernel(kernel::ActiveMCTKernel, F, t)
    out = Diagonal(similar(F))
    evaluate_kernel!(out, kernel, F, t)
    return out
end


######################################################################

struct TaggedActiveMCTKernel <: MemoryKernel
    k_array ::Vector{Float64}
    prefactor ::Float64
    J ::Array{Float64, 3}
    V2 ::Array{Float64, 3}
    Fsol
end

"""
M(k,t) =  ρ w0 / ((2π)^dim ) ∫ dq  V(k,q)^2 F(q,t) Fs(k-q,t)
V(k,q) = 1/k (k * q) c(q)
"""
function TaggedActiveMCTKernel(ρ, k_array, w0, wk, Sk, Fsol)
    Δk = k_array[2] - k_array[1];
    dim=3;
    prefactor = ρ * Δk^2 / ((2*π)^dim) * surface_d_dim_unit_sphere(dim-1);
    Nk = length(k_array);
    
    mCk = @. 1/ρ * ( 1 - wk / (w0 * Sk) );
    V2  = zeros(Nk, Nk, Nk);
    J   = zeros(Nk, Nk, Nk);

    for i=1:Nk, j=1:Nk, l=abs(j-i)+1:min(i+j-1,Nk)
        k = k_array[i];
        q = k_array[j];
        p = k_array[l];
        cp = mCk[l];
        V = cp/(2*k)*(k^2 + p^2 - q^2);
        V2[l,j,i] = w0 * V^2;
        J[l,j,i] = 2*p*q/((2*k)^(dim-2))*( (q+p-k)*(k+p-q)*(k+q-p)*(k+p+q) )^((dim-3)/2);
    end

    return TaggedActiveMCTKernel(k_array, prefactor, J, V2, Fsol)
end

function evaluate_kernel(kernel::TaggedActiveMCTKernel, Fs, t)
    out = Diagonal(similar(Fs))
    evaluate_kernel!(out, kernel, Fs, t)
    return out
end

function evaluate_kernel!(out::Diagonal, kernel::TaggedActiveMCTKernel, Fs, t)
    out.diag .= zero(eltype(out.diag))
    k_array = kernel.k_array
    Nk = length(k_array)

    # get index of time
    time = findall(x -> x == t, get_t(kernel.Fsol))[1]; 
    F_col = get_F(kernel.Fsol, time, :)

    for i = 1:Nk, j = 1:Nk, l=abs(j-i)+1:min(i+j-1,Nk)
        out.diag[i] += kernel.J[l,j,i] * kernel.V2[l, j, i] * F_col[l] * Fs[j]
    end

    out.diag .*= kernel.prefactor
end

