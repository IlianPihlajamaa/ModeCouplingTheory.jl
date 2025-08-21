
struct ActiveMultiComponentKernel{Fl,Ve,VM,VM2,VJ,X,x2} <: MemoryKernel
    prefactor ::Fl
    k_array ::Ve
    wk ::VM
    DCF ::VM2
    J ::VJ
    ρ ::X
    x :: x2
end

"""
    ActiveMultiComponentKernel(ρₐ, k_array, wk, w0, Sk, dim)

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
function ActiveMultiComponentKernel(ρₐ, k_array, wk, w0, Sk, dim = 3)
    ρ_tot = sum(ρₐ);
    xₐ = ρₐ ./ ρ_tot;
    Nk = length(k_array);
    Ns = size(Sk[1],1);

    Δk = k_array[2] - k_array[1];
    prefactor = Δk^2 /((2*π)^dim * 2) * surface_d_dim_unit_sphere(dim-1);

    J = zeros(Nk, Nk, Nk);
    V2 = zeros(Nk, Nk, Nk, 2);
    DCF = 0 .* similar(Sk);

    S_inv = inv.(Sk);
    w0_inv = inv(w0);

    # @inbounds for i=1:Nk  # calculate DCF
    #     test = zeros(Ns, Ns);

    #     for α=1:Ns, β=1:Ns
    #         kronecker = 0.0;
    #         if α == β  # there's probably a better way to do this
    #             kronecker = 1.0;
    #         end

    #         test[α,β] = kronecker;  # this works

    #         for γ=1:Ns, δ=1:Ns
    #             test[α,β] -= w0_inv[α,γ] * wk[i][γ,δ] * S_inv[i][δ,β];
    #             # test[α,β] += kronecker - w0_inv[α,γ]*wk[i][γ,δ]*S_inv[i][δ,β];  # equation 5.24
    #         end
    #     end
    #     test -= w0_inv * wk[i] * S_inv[i]

    #     DCF[i] = test ./ ρ_tot;
    # end

    # new definition! (without extra ρ's)
    for i=1:Nk
        DCF[i] = ( I(Ns) - w0_inv * wk[i] * S_inv[i] );
    end

    for i=1:Nk, j=1:Nk, l=abs(j-i)+1:min(i+j-1,Nk)
        k = k_array[i];
        q = k_array[j];
        p = k_array[l];
        
        # jacobian: same as single-component
        J[l,j,i] = 2*p*q/((2*k)^(dim-2))*( (q+p-k)*(k+p-q)*(k+q-p)*(k+p+q) )^((dim-3)/2);
    end
    return ActiveMultiComponentKernel(prefactor, k_array, wk, DCF, J, ρₐ, xₐ)
end

function evaluate_kernel!(out::Diagonal, kernel::ActiveMultiComponentKernel, F, t)
    k_array = kernel.k_array;

    DCF = kernel.DCF;
    wk = kernel.wk;
    ρ = kernel.ρ;

    Nk = length(k_array)
    Ns = size(DCF[1],1);
    mk = zeros(Ns, Ns) ::Matrix{Float64}

    @assert size(DCF) == size(wk) == size(F)
    @assert length(DCF) == Nk

    @inbounds for i=1:Nk
        mk = zeros(Ns, Ns) ::Matrix{Float64}; # resets per k=point

        for j=1:Nk, l=abs(j-i)+1:min(i+j-1,Nk)
            k = k_array[i];
            q = k_array[j];
            p = k_array[l];
            
            v2_1 = 1/(2*k)*(k^2 + q^2 - p^2);  # k dot q
            v2_2 = 1/(2*k)*(k^2 + p^2 - q^2);  # k dot p
            J = kernel.J[l,j,i];

            for α = 1:Ns, β = 1:Ns
                F_j_ab = F[j][α,β];
                F_l_ab = F[l][α,β];

                for ν = 1:Ns
                    dcf_j_bv = DCF[j][β,ν];
                    dcf_l_bv = DCF[l][β,ν];
                    dcf_l_av = DCF[l][α,ν];
                    F_l_av = F[l][α,ν];
                    F_l_bv = F[l][ν,β];

                    for μ = 1:Ns
                        dcf_j_au = DCF[j][α,μ];
                        dcf_j_bu = DCF[j][β,μ];
                        F_j_uv = F[j][μ,ν];
                        F_j_bu = F[j][μ,β];
                        F_j_au = F[j][α,μ];                   
                        F_l_uv = F[l][μ,ν];

                        dcf_l_au = DCF[l][α,μ];
                        dcf_l_bv = DCF[l][β,ν];

                        one = v2_1 * v2_1 * dcf_j_au * dcf_j_bv * F_j_uv * F_l_ab;
                        two = v2_1 * v2_2 * dcf_j_au * dcf_l_bv * F_j_bu * F_l_av;
                        three = v2_1 * v2_2 * dcf_l_av * dcf_j_bu * F_j_au * F_l_bv;
                        four = v2_2 * v2_2 * dcf_l_au * dcf_l_bv * F_j_ab * F_l_uv;

                        mk[α,β] += J * (one + two + three + four) / sqrt(ρ[α]*ρ[β]);
                    end
                end
            end
        end
        out.diag[i] = kernel.prefactor .* wk[i] * mk;
    end
end

function evaluate_kernel(kernel::ActiveMultiComponentKernel, F, t)
    out = 0.0 .* Diagonal(similar(F));
    evaluate_kernel!(out, kernel, F, t)
    return out
end
