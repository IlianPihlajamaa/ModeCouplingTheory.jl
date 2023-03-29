Ns = 2;
Nk = 3;
S11 = [1,2,3]; S21 = [4,5,6]; S22 = [8,9,10];
S = [zeros(Nk) for i=1:2, j=1:2];
S[1,1] = S11; S[1,2] = S21; S[2,1] = S21; S[2,2] = S22;
@test all(convert_multicomponent_structure_factor(S)[2] .== [2 5; 5 9])