struct ActiveMCTKernel <: MemoryKernel
    k_array ::Vector{Float64}
    prefactor ::Float64
    J ::Array{Float64, 3}
    V2 ::Array{Float64, 3} 
end

"""
    ActiveMCTKernel(ρ, k_array, wk, w0, Sk, dim=3)

Implements the following active MCT kernel:

M(k,t) =  ρ / (2(2π)^dim ) ∫ dq w(k) V(k,q)^2 F(q,t) F(k-q,t)

with the vertices:

V(k,q) = c(q) * (k * q)/k + c(p) (k * p)/k

# Arguments:

* ρ: number density
* k_array: array of wavevectors at which to evaluate the kernel
* wk: steady-state velocity correlations ( w(k) ) - this depends on k
* w0: local velocity correlations ( w(∞) ) - this is a constant
* Sk: steady-state structure factor
* dim: dimensionality of the problem (the default `dim=3`)
"""
function ActiveMCTKernel(ρ, k_array, wk, w0, Sk, dim=3)
    Δk = k_array[2] - k_array[1];
    prefactor = Δk^2* ρ / (2*(2*π)^dim) * surface_d_dim_unit_sphere(dim-1);
    Nk = length(k_array);
    
    mCk = @. 1/ρ * ( 1 - wk / (w0 * Sk) );
    @assert length(mCk) == Nk
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
    
    @inbounds for i = 1:Nk, j = 1:Nk, l=abs(j-i)+1:min(i+j-1,Nk)
        out.diag[i] += kernel.J[l,j,i] * kernel.V2[l, j, i] * F[j] * F[l]
    end
    out.diag .*= kernel.prefactor
end

function evaluate_kernel(kernel::ActiveMCTKernel, F, t)
    out = Diagonal(similar(F))
    evaluate_kernel!(out, kernel, F, t)
    return out
end

struct TaggedActiveMCTKernel{V,F,VA,FF,tD} <: MemoryKernel
    k_array ::V
    prefactor ::F
    J ::VA
    V2 ::VA
    Fsol ::FF
    tDict ::tD
end

"""
    TaggedActiveMCTkernel(ρ, k_array, wk, w0, Sk, Fsol, dim)

This function implements the following kernel for the self-intermediate scattering function:

M(k,t) =  ρ w0 / ((2π)^dim ) ∫ dq  ( (k*p)/(2k) * c(p) )^2 F(q,t) Fs(k-q,t)

# Arguments

* ρ: number density
* k_array: k-values at which to evaluate the kernel
* wk: steady-state velocity correlations ( w(k) ) - this depends on k
* w0: local velocity correlations ( w(∞) ) - this is a constant
* Sk: steady-state structure factor
* Fsol: solution of the collective mode-coupling equation
* dim: dimensionality of the problem (the default `dim=3`)
"""
function TaggedActiveMCTKernel(ρ, k_array, wk, w0, Sk, Fsol, dim=3)
    Δk = k_array[2] - k_array[1];
    prefactor = ρ * Δk^2 / ((2*π)^dim) * surface_d_dim_unit_sphere(dim-1);
    Nk = length(k_array);
    tDict = Dict(zip(Fsol.t, eachindex(Fsol.t)));
    
    mCk = @. 1/ρ * ( 1 - wk / (w0 * Sk) );
    @assert length(mCk) == Nk
    V2  = zeros(Nk, Nk, Nk);
    J   = zeros(Nk, Nk, Nk);

    @inbounds for i=1:Nk, j=1:Nk, l=abs(j-i)+1:min(i+j-1,Nk)
        k = k_array[i];
        q = k_array[j];
        p = k_array[l];
        
        V2[l,j,i] = ( mCk[l]/(2*k)*(k^2 + p^2 - q^2) )^2 * w0;
        J[l,j,i] = 2*p*q/((2*k)^(dim-2))*( (q+p-k)*(k+p-q)*(k+q-p)*(k+p+q) )^((dim-3)/2);
    end

    return TaggedActiveMCTKernel(k_array, prefactor, J, V2, Fsol, tDict)
end

function evaluate_kernel(kernel::TaggedActiveMCTKernel, Fs, t)
    out = Diagonal(similar(Fs))
    evaluate_kernel!(out, kernel, Fs, t)
    return out
end

function evaluate_kernel!(out::Diagonal, kernel::TaggedActiveMCTKernel, Fs, t)
    out.diag .= zero(eltype(out.diag));
    Nk = length(kernel.k_array);

    F_col = get_F(kernel.Fsol, kernel.tDict[t], :)

    for i = 1:Nk, j = 1:Nk, l=abs(j-i)+1:min(i+j-1,Nk)
        out.diag[i] += kernel.J[l,j,i] * kernel.V2[l,j,i] * F_col[l] * Fs[j]
    end

    out.diag .*= kernel.prefactor
end
