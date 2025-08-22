struct ActiveMultiComponentKernel{Fl,Ve,VM,V3,VJ,X} <: MemoryKernel
    prefactor ::Fl
    k_array ::Ve
    wk ::VM
    DCF ::VM
    V2 ::V3
    J ::VJ
    ρ ::X
    wkinv ::VM
    F_arr::Array{Float64,3}
    M_arr::Array{Float64,3}
    temp::Matrix{Float64}
end

"""
    ActiveMultiComponentKernel(ρₐ, k_array, wk, w0, Sk, dim)

Implements the following multi-component active MCT kernel:

M{αβ}(k,t) = 1 / (2*(2π)^dim) ∑{μνμ'ν'λ} ∫ dq F{μμ'}(q,t) F{νν'}(k-q,t) V{μνα}(k,q) V{μ'ν'λ}(k,q) [w{λβ}(k)]^(-1)

where Greek indices {...} denote species labels and the expression for the vertices V{μνα}(k,q) is given in the documentation.
Note: the input data (wk, Sk) should be given in the format Vector{ Matrix }, with the Vector having length Nk
and the Matrix having size Ns x Ns. Nk is the number of k-points and Ns is the number of species in the mixture.

# Arguments:

* ρₐ: number densities of each species (a vector of length Ns)
* k_array: k-values at which to evaluate the kernel (a vector of length Nk)
* wk: velocity correlations of each species (a vector of Ns x Ns matrices)
* w0: local velocity correlations (a Ns x Ns matrix)
* Sk: partial structure factors (a vector of Ns x Ns matrices)
* dim: dimensionality of the kernel (the default `dim=3`)
"""
function ActiveMultiComponentKernel(ρₐ, k_array, wk, w0, Sk, dim = 3)
    Nk = length(k_array);
    Ns = size(Sk[1],1);

    Δk = k_array[2] - k_array[1];
    prefactor = Δk^2 * surface_d_dim_unit_sphere(dim-1) /(2 * (2*π)^dim);

    J = zeros(Nk, Nk, Nk);
    V2 = zeros(Nk, Nk, Nk, Ns, Ns, Ns);
    DCF = 0 .* similar(Sk);

    for i=1:Nk
        DCF[i] = I(Ns) - inv(w0) * wk[i] * inv(Sk[i]);
    end

    for i=1:Nk, j=1:Nk, l=abs(j-i)+1:min(i+j-1,Nk)
        k = k_array[i]; q = k_array[j]; p = k_array[l];
       
        kdotq = (k^2 + q^2 - p^2)/(2*k);
        kdotp = (k^2 + p^2 - q^2)/(2*k);

        for μ=1:Ns, ν=1:Ns, α=1:Ns, γ = 1:Ns
            if γ == ν
                V2[l,j,i,μ,ν,α] += kdotq * wk[i][α,γ] * DCF[j][γ,μ] / sqrt(ρₐ[γ]);
            end
            if γ == μ
                V2[l,j,i,μ,ν,α] += kdotp * wk[i][α,γ] * DCF[l][γ,ν] / sqrt(ρₐ[γ]);
            end
        end

        J[l,j,i] = 2*p*q/((2*k)^(dim-2))*( (q+p-k)*(k+p-q)*(k+q-p)*(k+p+q) )^((dim-3)/2);
    end

    return ActiveMultiComponentKernel(prefactor, k_array, wk, DCF, V2, J, ρₐ, inv.(wk),
                                       zeros(Float64, Ns, Ns, Nk), zeros(Float64, Ns, Ns, Nk), zeros(Float64, Ns, Ns))
end

function evaluate_kernel!(out::Diagonal, kernel::ActiveMultiComponentKernel, F, t)
    k_array = kernel.k_array;
    V2 = kernel.V2;
    J = kernel.J;
    wk = kernel.wk;
    wkinv = kernel.wkinv;
   
    Nk = length(k_array)
    Ns = size(wk[1],1);

    @assert size(wk) == size(F)
    @assert length(wk) == Nk

    F_arr = kernel.F_arr # F_arr[μ, μ2, j] = F{μμ'}(q,t) for q=k_array[j]
    M_arr = kernel.M_arr # M_arr[α, β, i] = M{αβ}(k_array[i],t) for k_array[i]
    M_arr .= 0.0  # reset M_arr to zero

    for μ=1:Ns, μ2=1:Ns, j=1:Nk
        F_arr[μ, μ2, j] = F[j][μ, μ2]
    end
    
    temp = kernel.temp
    temp .= 0.0

    @tullio grad=false M_arr[α, β, i] = V2[l,j,i,μ,ν,α] * V2[l,j,i,μ2,ν2,β] * F_arr[μ, μ2, j] * F_arr[ν, ν2, l] * J[l,j,i]
    
    for i in 1:Nk
        for α in 1:Ns, β in 1:Ns
            temp[α, β] = 0.0
            for γ in 1:Ns
                temp[α, β] += (M_arr[α, γ, i] * wkinv[i][γ, β])
            end
        end
        out.diag[i] = SMatrix{Ns, Ns, Float64, Ns*Ns}(temp) * kernel.prefactor  # store the result in the diagonal of
    end
end

function evaluate_kernel(kernel::ActiveMultiComponentKernel, F, t)
    out = 0.0 .* Diagonal(similar(F));
    evaluate_kernel!(out, kernel, F, t)
    return out
end