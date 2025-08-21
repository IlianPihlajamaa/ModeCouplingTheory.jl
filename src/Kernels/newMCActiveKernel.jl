
struct ActiveMultiComponentKernel2{Fl,Ve,VM,VM2,V3,VJ,X} <: MemoryKernel
    prefactor ::Fl
    k_array ::Ve
    wk ::VM
    DCF ::VM2
    V2 ::V3
    J ::VJ
    ρ ::X
end

"""
    ActiveMultiComponentKernel2(ρₐ, k_array, wk, w0, Sk, dim)

Implements the following multi-component active MCT kernel:

M{αβ}(k,t) = ρ / (2*(2π)^dim) ∑{μνμ'ν'λ} ∫ dq F{μμ'}(q,t) F{νν'}(k-q,t) w{αλ}(k) V{μ'ν'λ}(k,q) V{μνβ}(k,q)

where Greek indices {...} denote species labels and the expression for the vertices V are given in the documentation.
Note: the input data should be given in the format of Vector{ Matrix }, with the Vector having length Nk
and the Matrix having size Ns x Ns. Nk is the number of k-points and Ns is the number of species in the mixture.

# Arguments:

* ρₐ: number densities of each species (a vector of length Ns)
* k_array: k-values at which to evaluate the kernel (a vector of length Nk)
* wk: velocity correlations of each species (a vector of Ns x Ns matrices) 
* w0: local velocity correlations (a Ns x Ns matrix)
* Sk: partial structure factors ( a vector of Ns x Ns matrices)
* dim: dimensionality of the kernel (the default `dim=3`)
"""
function ActiveMultiComponentKernel2(ρₐ, k_array, wk, w0, Sk, dim = 3)
    Nk = length(k_array);
    Ns = size(Sk[1],1);

    Δk = k_array[2] - k_array[1];
    prefactor = Δk^2 * surface_d_dim_unit_sphere(dim-1) /(2 * (2*π)^dim);

    J = zeros(Nk, Nk, Nk);
    V2 = zeros(Nk, Nk, Nk, Ns, Ns, Ns);
    DCF = 0 .* similar(Sk);

    # for i=1:Nk  # old definition (equation 5.24)
    #     c_dcf = zeros(Ns, Ns);

    #     for α=1:Ns, β=1:Ns
    #         kronecker = 0.0;

    #         if α==β
    #             kronecker = 1.0;
    #         end

    #         for γ=1:Ns, ϵ=1:Ns
    #             c_dcf[α,β] += kronecker - w0_inv[α,γ] * wk[i][γ,ϵ] * S_inv[i][ϵ,β]
    #         end
    #     end
    #     DCF[i] = c_dcf;
    # end

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

        # jacobian: same as single-component
        J[l,j,i] = 2*p*q/((2*k)^(dim-2))*( (q+p-k)*(k+p-q)*(k+q-p)*(k+p+q) )^((dim-3)/2);
        # J[l,j,i] = p * q / k; # 3D only (for testing)
    end

    return ActiveMultiComponentKernel2(prefactor, k_array, wk, DCF, V2, J, ρₐ)
end

function evaluate_kernel!(out::Diagonal, kernel::ActiveMultiComponentKernel2, F, t)
    k_array = kernel.k_array;
    V2 = kernel.V2;
    J = kernel.J;
    wk = kernel.wk;
    
    Nk = length(k_array)
    Ns = size(wk[1],1);
    mk = zeros(Ns, Ns) ::Matrix{Float64}

    @assert size(wk) == size(F)
    @assert length(wk) == Nk

    @inbounds for i=1:Nk
        mk = zeros(Ns, Ns) ::Matrix{Float64}; # resets per k=point; eq. 5.27
        wk_inv = inv(wk[i]);

        for j=1:Nk, l=abs(j-i)+1:min(i+j-1,Nk)  # integral over q-vector
            F_q = F[j];
            F_p = F[l];

            for α=1:Ns, β=1:Ns
                for μ = 1:Ns, ν = 1:Ns, μ2=1:Ns, ν2 = 1:Ns
                    mk[α,β] += V2[l,j,i,μ,ν,α] * V2[l,j,i,μ2,ν2,β] * F_q[μ,μ2] * F_p[ν,ν2] * J[l,j,i]
                end
            end
        end
        out.diag[i] = (mk * wk_inv) .* kernel.prefactor
        # println(out.diag[i])
    end

end

function evaluate_kernel(kernel::ActiveMultiComponentKernel2, F, t)
    out = 0.0 .* Diagonal(similar(F));
    evaluate_kernel!(out, kernel, F, t)
    return out
end
