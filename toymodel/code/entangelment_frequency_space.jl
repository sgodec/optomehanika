using Random:default_rng, seed!
using LaTeXStrings
using CairoMakie
using ProgressMeter
using LinearAlgebra
using MatrixEquations

MT = Makie.MathTeXEngine
mt_fonts_dir = joinpath(dirname(pathof(MT)), "..", "assets", "fonts", "NewComputerModern")

set_theme!(fonts = (
regular = joinpath(mt_fonts_dir, "NewCM10-Regular.otf"),
bold = joinpath(mt_fonts_dir, "NewCM10-Bold.otf")
) )
###################################################################################################################################
#constants
const h = 6.62607015e-34     
const kB = 1.380649e-23
# System parameters
omega_b::Float64 =2π * 1e5 
omega_L::Float64 =2π * 3e15
gamma::Float64   =2π * 1e-5
kappaA::Float64  =2π * 58e3
kappaB::Float64  =2π * 58e3
gA::Float64      =2π * 11.3e3
gB::Float64      =2π * 3.3e3
deltaA::Float64  =2π * 1e5  
deltaB::Float64  =2π * -1e5
@show (2*(kappaA) /( omega_b * 4))^2

# Environment
T::Float64 = 300
n_b     =  1 /( exp( (h * omega_b )/ (2π * kB * T) ) -1) 
n_A, n_B =  1 /( exp( (h * omega_L )/ (2π * kB * T) ) -1), 1 /( exp( (h * omega_L )/ (2π * kB * T) ) -1) 
@show n_b,n_A,n_B
###################################################################################################################################
#definition of matrices(RWA)
A_rwa = [
    gamma/2   -omega_b   0     -gA     0     gB;
    omega_b   gamma/2    gA     0     gB    0;
    0         -gA        kappaA/2  -deltaA  0     0;
    gA        0          deltaA    kappaA/2 0     0;
    0         gB         0         0         kappaB/2 -deltaB;
    gB        0          0         0         deltaB   kappaB/2
]


H = im * I(6)

# compute M(ω)= A-ωH
Mω(A, ω) = A_rwa - ω*H

#
G = Diagonal([sqrt(gamma), sqrt(gamma), sqrt(kappaA), sqrt(kappaA), sqrt(kappaB), sqrt(kappaB)])

# Noise correlation matrix D
D_b = [n_b + 0.5  im/2;
       -im/2      n_b + 0.5]
D_A = [n_A + 0.5  im/2;
       -im/2      n_A + 0.5]
D_B = [n_B + 0.5  im/2;
       -im/2      n_B + 0.5]
#
# Block-diagonal D
D = zeros(ComplexF64, 6, 6)
D[1:2,1:2] = D_b
D[3:4,3:4] = D_A
D[5:6,5:6] = D_B

# compute spectra without explicit inversion 
function compute_spectra(A, D, ω_vec)
    S_list = Vector{Matrix{ComplexF64}}(undef, length(ω_vec))
    for (i, ω) in enumerate(ω_vec)
        M = Mω(A, ω)
        X = M \ D
        S = X / M'   #inv(M)*D*inv(M)'
        S_list[i] = S
    end
    return S_list
end

function compute_output_spectra(A, D, G, ω_vec)
    S_list = Vector{Matrix{ComplexF64}}(undef, length(ω_vec))
    Id = Matrix{ComplexF64}(I, size(A,1), size(A,1))
    Id[1,1] = 0
    Id[2,2] = 0
    G_new = Diagonal([-1, -1, sqrt(kappaA), sqrt(kappaA), sqrt(kappaB), sqrt(kappaB)])

    for (i, ω) in enumerate(ω_vec)
        M = Mω(A, ω)
        # K = I - G M^{-1} G
        K = Id - G_new * (M \ G)
        S_list[i] = K * D * K'
    end
    return S_list
end

function compute_filtered_spectra(A, D, G, ω_vec)
    #convolution function
    function gaussian(omega, center=-2 *pi * 10^5, width=2pi*1e4)
        prefactor = sqrt(2 * pi / (width * sqrt(pi)))  # normalization for V = 1
        return prefactor * exp(-(omega - center)^2 / (2 * width^2))
    end

    function F_gauss(omega)
        Cw = gaussian(omega)
        C_mw_conj = conj(xi_gaussian(-omega))
        
        M = Matrix{ComplexF64}(I, 6, 6)
        M[3,3] = 1/sqrt(2) * (Cw+C_mw_conj);            M[3,4] = 1/sqrt(2) * im * (Cw-C_mw_conj)
        M[4,3] = 1/sqrt(2) * im * (C_mw_conj-Cw);       M[4,4] = 1/sqrt(2) * (Cw+C_mw_conj)

        M[5,5] = 1/sqrt(2) * (Cw+C_mw_conj);            M[5,6] = 1/sqrt(2) * im * (Cw-C_mw_conj)
        M[6,5] = 1/sqrt(2) * im * (C_mw_conj-Cw);      M[6,6] = 1/sqrt(2) * (Cw+C_mw_conj)
        
        return M
    end

    dω = ω_vec[2] - ω_vec[1]
    S_list = Vector{Matrix{ComplexF64}}(undef, length(ω_vec))
    
    Id_mask = Matrix{ComplexF64}(I, 6, 6)
    Id_mask[1:2, 1:2] .= 0 
    G_new = Diagonal([-1.0, -1.0, sqrt(kappaA), sqrt(kappaA), sqrt(kappaB), sqrt(kappaB)])

    for (i, ω) in enumerate(ω_vec)
        M_mat = A - ω * (im * I)
        
        #K = I - G * inv(M) * G
        K = Id_mask - G_new * (M_mat \ G)
        
        # S_out = K * D * K'
        S_out = K * D * K'
        
        # Apply Gaussian filter transformation
        Mx = M_xi_B(ω)
        S_list[i] = Mx * S_out * Mx'
    end
    return S_list
end

ω_list = range(-1e8, stop=1e8, length=1000000)
dω = ω_list[2]-ω_list[1]
#S_rwa_list = compute_spectra(A_rwa, G*D*G', ω_list) 
#S_rwa_list = compute_output_spectra(A_rwa, D, G, ω_list)
S_rwa_list = compute_filtered_spectra(A_rwa, D, G, ω_list)

V =  real(dω .* sum([1/2 *(S + transpose(S)) for S in S_rwa_list]) ./ (2 * pi))
@show  dω * sum(xi_gaussian.(ω_list).^2) / (2 * pi)


@show real(V)
PSD_Qb = [real(S[1,1]) for S in S_rwa_list]  # mechanical position
PSD_QA = [real(S[3,3]) for S in S_rwa_list]  # cavity A quadrature
PSD_QB = [real(S[5,5]) for S in S_rwa_list]  # cavity B quadrature
PSD_PB = [real(S[6,6]) for S in S_rwa_list]  

###################################################################################################################################

function log_negativity(V_full, m1, m2)
    #two mode covariance matrix
    idxs = [2m1-1, 2m1, 2m2-1, 2m2]
    M = V_full[idxs, idxs]

    # block matrices
    A = M[1:2,1:2]
    B = M[3:4,3:4]
    C = M[1:2,3:4]

    # determinants
    detA = det(A)
    detB = det(B)
    detC = det(C)
    detM = det(M)

    Delta_tilde = detA + detB - 2*detC

    # symplectic eigenvalues
    term = sqrt(Delta_tilde^2 - 4*detM)

    nu_minus = sqrt((Delta_tilde - term)/2)
    nu_plus  = sqrt((Delta_tilde + term)/2)

    # logarithmic negativity
    L = max(0, -log(2*nu_minus))
         
    Omega = [ 0  1   0  0   0  0
             -1  0   0  0   0  0
              0  0   0  1   0  0
              0  0  -1  0   0  0
              0  0   0  0   0  1
              0  0   0  0  -1  0 ]

    @show eigvals(V + 1im*Omega/2)

    return L, nu_minus, nu_plus
end

E_bA = log_negativity(real.(V), 1, 2)   
E_bB = log_negativity(real.(V), 1, 3)   
E_AB = log_negativity(real.(V), 2, 3)   

n = 1/2 *(V[1,1]+V[2,2]-1)
println("Logarithmic negativity b-A: ", E_bA)
println("Logarithmic negativity b-B: ", E_bB)
println("Logarithmic negativity A-B: ", E_AB)
println("Average n_b: ", n)
####################################################################################################################################ploting

ax = [
    Axis(fig[1,i], 
         width = 350, 
         height = 350,
         xticklabelsize = 16,
         yticklabelsize = 16,
         xlabelsize = 20,
         ylabelsize = 20,
         titlesize = 22) for i in 1:1
]

ars = ["Q_b", "P_b", "Q_A", "P_A", "Q_B", "P_B"]

fig = Figure(size = (1500, 1400))

axes = [Axis(fig[i, j], 
             title = i == 1 ? vars[j] : "",        
             ylabel = j == 1 ? vars[i] : "",      
             xticklabelsvisible = i == 6,        
             yticklabelsvisible = j == 1,
             xticklabelsize = 16,
             yticklabelsize = 16,
             xlabelsize = 20,
             ylabelsize = 20,
             titlesize = 22
             )  for i in 1:6, j in 1:6]
for a in axes
    a.xgridvisible = true
    a.ygridvisible = true
    a.xgridstyle = :dash
    a.ygridstyle = :dash
    a.xminorgridvisible = true
    a.yminorgridvisible = true
    a.xminorticksvisible = true
    a.yminorticksvisible = true
end

for i in 1:6
    for j in 1:6
        ax = axes[i, j]
        y_data = [real(S[i,j]) for S in S_rwa_list]
        step = max(1, length(ω_list) ÷ 5000)
        
        lines!(ax, ω_list[1:step:end], y_data[1:step:end], 
               color = i == j ? :black : :crimson, 
               linewidth = 1.5)

        xlims!(ax,(-1.5e6, 1.5e6)) 

    end
end

linkaxes!(axes...)
colgap!(fig.layout, 10)
rowgap!(fig.layout, 10)

display(fig)
save("power_density_out.pdf",fig)
