
struct ActiveMultiComponentKernel{Fl,Ve,VM,VM2,V3,VJ,X} <: MemoryKernel
    prefactor ::Fl
    k_array ::Ve
    wk ::VM
    DCF ::VM2
    V2 ::V3
    J ::VJ
    x ::X
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
    prefactor = zeros(Ns, Ns);

    for α = 1:Ns, β = 1:Ns  # check this expression
        prefactor[α,β] = Δk^2 * ρ_tot /((2*π)^dim * 2 * sqrt(xₐ[α] * xₐ[β])) * surface_d_dim_unit_sphere(dim-1);
    end

    J = zeros(Nk, Nk, Nk);
    V2 = zeros(Nk, Nk, Nk, 2);
    DCF = 0 .* similar(Sk);

    S_inv = inv.(Sk);
    w0_inv = inv(w0);

    @inbounds for i=1:Nk  # calculate DCF
        test = zeros(Ns, Ns);

        for α=1:Ns, β=1:Ns
            kronecker = 0.0;
            if α == β  # there's probably a better way to do this
                kronecker = 1.0;
            end

            test[α,β] = kronecker / xₐ[α];  # this works
            # test[α,β] = kronecker;  # this doesn't

            for γ=1:Ns, δ=1:Ns
                test[α,β] -= w0_inv[α,γ] * wk[i][γ,δ] * S_inv[i][δ,β];
                # test[α,β] += kronecker - w0_inv[α,γ]*wk[i][γ,δ]*S_inv[i][δ,β];  # vincent's definition
                # test[α,β] += kronecker / xₐ[α] - w0_inv[α,γ]*wk[i][γ,δ]*S_inv[i][δ,β];  # this definitely doesn't work
            end
        end

        DCF[i] = test ./ ρ_tot;
    end

    for i=1:Nk, j=1:Nk, l=abs(j-i)+1:min(i+j-1,Nk)
        k = k_array[i];
        q = k_array[j];
        p = k_array[l];
        
        # vertices: only the prefactors
        V2[l,j,i,1] = 1/(2*k)*(k^2 + q^2 - p^2);  # (k * q) / k
        V2[l,j,i,2] = 1/(2*k)*(k^2 + p^2 - q^2);  # (k * p) / k

        # jacobian: same as single-component
        J[l,j,i] = 2*p*q/((2*k)^(dim-2))*( (q+p-k)*(k+p-q)*(k+q-p)*(k+p+q) )^((dim-3)/2);
    end
    return ActiveMultiComponentKernel(prefactor, k_array, wk, DCF, V2, J, xₐ)
end

function evaluate_kernel!(out::Diagonal, kernel::ActiveMultiComponentKernel, F, t)
    k_array = kernel.k_array;

    DCF = kernel.DCF;
    V2 = kernel.V2;
    wk = kernel.wk;

    Nk = length(k_array)
    Ns = size(DCF[1],1);
    mk = zeros(Ns, Ns) ::Matrix{Float64}

    @assert size(DCF) == size(wk) == size(F)
    @assert length(DCF) == Nk

    @inbounds for i=1:Nk
        mk = zeros(Ns, Ns) ::Matrix{Float64}; # resets per k=point

        for j=1:Nk, l=abs(j-i)+1:min(i+j-1,Nk)
            v2_1 = V2[l,j,i,1]; v2_2 = V2[l,j,i,2]
            J = kernel.J[l,j,i]

            for β = 1:Ns, λ = 1:Ns
                pref = kernel.prefactor[λ,β];
                F_j_ab = F[j][λ,β];
                F_l_ab = F[l][λ,β];

                for ν = 1:Ns
                    dcf_j_bv = DCF[j][β,ν];
                    dcf_l_bv = DCF[l][β,ν];
                    dcf_l_av = DCF[l][λ,ν];
                    F_l_av = F[l][λ,ν];
                    F_l_bv = F[l][β,ν];

                    for μ = 1:Ns
                        dcf_j_au = DCF[j][λ,μ];
                        dcf_j_bu = DCF[j][β,μ];
                        dcf_l_bu = DCF[l][β,μ];
                        F_j_uv = F[j][μ,ν];
                        F_j_bu = F[j][β,μ];
                        F_j_au = F[j][λ,μ];                   
                        F_l_uv = F[l][μ,ν];

                        for α=1:Ns
                            wk_ay = wk[i][α,λ];

                            one = v2_1 * v2_1 * dcf_j_au * dcf_j_bv * F_j_uv * F_l_ab;
                            two = v2_1 * v2_2 * dcf_j_au * dcf_l_bv * F_j_bu * F_l_av;
                            three = v2_1 * v2_2 * dcf_l_av * dcf_j_bu * F_j_au * F_l_bv;
                            four = v2_2 * v2_2 * dcf_l_av * dcf_l_bu * F_j_ab * F_l_uv;

                            mk[α,β] += J * pref * wk_ay * (one + two + three + four);
                        end
                    end
                end
            end
        end
        out.diag[i] = mk;
    end
end

function evaluate_kernel(kernel::ActiveMultiComponentKernel, F, t)
    out = 0.0 .* Diagonal(similar(F));
    evaluate_kernel!(out, kernel, F, t)
    return out
end
