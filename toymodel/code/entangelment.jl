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

# Drift matrix A (RWA)
A = [
    -gamma/2   omega_b   0       gA      0      -gB;
    -omega_b  -gamma/2  -gA      0       -gB      0;
     0         gA     -kappaA/2 deltaA   0       0;
    -gA        0      -deltaA  -kappaA/2 0       0;
     0        -gB      0        0     -kappaB/2 deltaB;
     -gB        0       0        0     -deltaB  -kappaB/2
]

# Noise coupling matrix G
G = Diagonal([sqrt(gamma), sqrt(gamma), sqrt(kappaA), sqrt(kappaA), sqrt(kappaB), sqrt(kappaB)])

# Noise correlation matrix D
D_b = [n_b + 0.5  im/2;
       -im/2      n_b + 0.5]
D_A = [n_A + 0.5  im/2;
       -im/2      n_A + 0.5]
D_B = [n_B + 0.5  im/2;
       -im/2      n_B + 0.5]

# Block-diagonal D
D = zeros(ComplexF64, 6, 6)
D[1:2,1:2] = D_b
D[3:4,3:4] = D_A
D[5:6,5:6] = D_B

# Lyapunov term: Q = 0.5*(G*D*G' + (G*D*G')')
Q = real(0.5 * (G*D*G' + (G*D*G')'))

# Stability
evals = eigvals(A)
real_parts = real.(evals)
is_stable = all(real_parts .< 0)

if is_stable
    println("STABLE")
else
    println("UNSTABLE")
end

# Solve Lyapunov
V = lyapc(A, Q) 
@show V

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

E_bA = log_negativity(V, 1, 2)   
E_bB = log_negativity(V, 1, 3)   
E_AB = log_negativity(V, 2, 3)   

n = 1/2 *(V[1,1]+V[2,2]-1)
println("Logarithmic negativity b-A: ", E_bA)
println("Logarithmic negativity b-B: ", E_bB)
println("Logarithmic negativity A-B: ", E_AB)
println("Average n_b: ", n)

V_out = zeros(4,4)

idx_A = [3,4]
idx_B = [5,6]

V_out[1:2,1:2] = kappaA * V[idx_A, idx_A] + (n_A + 0.5)*I(2)
V_out[3:4,3:4] = kappaB * V[idx_B, idx_B] + (n_B + 0.5)*I(2)
V_out[1:2,3:4] = sqrt(kappaA*kappaB) * V[idx_A, idx_B]
V_out[3:4,1:2] = V_out[1:2,3:4]'
println("Logarithmic negativity A_out-B_out: ", log_negativity(V_out,1,2))

idx_b = [1,2]   
idx_B = [5,6]   

V_bB_out = zeros(4,4)

V_bB_out[1:2,1:2] = V[idx_b, idx_b]

V_bB_out[3:4,3:4] = kappaB .* V[idx_B, idx_B] + (n_B + 0.5) * I(2)

V_bB_out[1:2,3:4] = sqrt(kappaB) .* V[idx_b, idx_B]
V_bB_out[3:4,1:2] = V_bB_out[1:2,3:4]'

println("Logarithmic negativity  b-B_out: ", log_negativity(V_bB_out, 1, 2))
############################################################################################################################################
function plot()
    fig = Figure(size = (1000 ,500))
    ax = [
        Axis(fig[1,i], 
             width = 350, 
             height = 350,
             xticklabelsize = 16,
             yticklabelsize = 16,
             xlabelsize = 20,
             ylabelsize = 20,
             titlesize = 22) for i in 1:2
    ]
    for a in ax
        a.xgridvisible = true
        a.ygridvisible = true
        a.xgridstyle = :dash
        a.ygridstyle = :dash
        a.xminorgridvisible = true
        a.yminorgridvisible = true
        a.xminorticksvisible = true
        a.yminorticksvisible = true
    end

    gBs = exp.(range(log((2.72)^(3)), stop=log(2.72 *  gB), length=20))
    deltaAs = exp.(range(log(2.72^8), stop=log(2.72 * omega_b), length=1000))

    colors = cgrad(:viridis, length(gBs),categorical = true,rev =true)
    for (i, gB) in enumerate(gBs)
        n_avg = []
        logneg = []
        deltaAs_final =[]
        for deltaA in deltaAs
            A = [
                -gamma/2   omega_b   0       gA      0      -gB;
                -omega_b  -gamma/2  -gA      0       -gB      0;
                 0         gA     -kappaA/2 deltaA   0       0;
                -gA        0      -deltaA  -kappaA/2 0       0;
                 0        -gB      0        0     -kappaB/2 deltaB;
                 -gB        0       0        0     -deltaB  -kappaB/2
            ]

            evals = eigvals(A)
            real_parts = real.(evals)
            is_stable = all(real_parts .< 0)
            if is_stable 
                V = lyapc(A, Q) 
                push!(logneg,log_negativity(V,1,3)[1])
                push!(n_avg, 0.5 * (V[1,1] + V[2,2] - 1))
                push!(deltaAs_final,deltaA)
            end
        end
        scatter!(ax[1], deltaAs_final./ omega_b, n_avg, color = colors[i],markersize = 3)
        scatter!(ax[2], deltaAs_final./ omega_b, logneg, color = colors[i],markersize = 3)
    end
    vlines!(ax[1],1,color = :black)
    vlines!(ax[2],1,color = :black)
    ax[1].xlabel = L"\Delta_A / \omega_b"
    ax[1].ylabel = L"\text{Average phonon number} \langle n_b \rangle"
    ax[1].title = L"\text{Phonon number vs } \Delta_A"

    ax[2].xlabel = L"\Delta_A / \omega_b"
    ax[2].ylabel = L"\text{Log negativity}" 
    ax[2].title = L"\text{Log negativity vs } \Delta_A"
    ax[1].xscale = log
    ax[1].yscale = log
    ax[2].xscale = log
    ax[2].yscale = log

    Colorbar(fig[1, 3], limits=(minimum(gBs), maximum(gBs)), width = 20, colormap=colors, label=L"$g_B$", vertical=true,ticklabelsize=18, labelsize=20)

    save("delta_A_g_b.png", fig; px_per_unit = 3)  
    return fig
end

function plot1()
    fig = Figure(size = (1000 ,500))
    ax = [
        Axis(fig[1,i], 
             width = 350, 
             height = 350,
             xticklabelsize = 16,
             yticklabelsize = 16,
             xlabelsize = 20,
             ylabelsize = 20,
             titlesize = 22) for i in 1:2
    ]
    for a in ax
        a.xgridvisible = true
        a.ygridvisible = true
        a.xgridstyle = :dash
        a.ygridstyle = :dash
        a.xminorgridvisible = true
        a.yminorgridvisible = true
        a.xminorticksvisible = true
        a.yminorticksvisible = true
    end

    gAs = exp.(range(log(2.72^3), stop=log(2.72*gA), length=40))
    deltaBs = -1 .* exp.(range(log(2.27^8), stop=log(10 * omega_b), length=1000))

    colors = cgrad(:viridis, length(gAs), categorical = true,rev=true)
    for (i, gA) in enumerate(gAs)
        n_avg = []
        logneg = []
        deltaBs_final =[]
        for deltaB in deltaBs
            A = [
                -gamma/2   omega_b   0       gA      0      -gB;
                -omega_b  -gamma/2  -gA      0       -gB      0;
                 0         gA     -kappaA/2 deltaA   0       0;
                -gA        0      -deltaA  -kappaA/2 0       0;
                 0        -gB      0        0     -kappaB/2 deltaB;
                 -gB        0       0        0     -deltaB  -kappaB/2
            ]

            evals = eigvals(A)
            real_parts = real.(evals)
            is_stable = all(real_parts .< 0)
            if is_stable 
                V = lyapc(A, Q) 
                push!(logneg,log_negativity(V,1,3)[1])
                push!(n_avg, 0.5 * (V[1,1] + V[2,2] - 1))
                push!(deltaBs_final,deltaB)
            end
        end
        if length(deltaBs_final) > 0
            scatter!(ax[1], -1. * deltaBs_final./ omega_b, n_avg, color = colors[i],markersize = 3)
            scatter!(ax[2], -1. *deltaBs_final./ omega_b, logneg, color = colors[i],markersize = 3)
        end
    end
    #hline!(ax[1],1,deltaAs ./ omega_b,color = :black)
    vlines!(ax[1],1,color = :black)
    vlines!(ax[2],1,color = :black)
    ax[1].xlabel = L"\Delta_B / \omega_b"
    ax[1].ylabel = L"\text{Average phonon number} \langle n \rangle"
    ax[1].title = L"\text{Phonon number vs } \Delta_B"
    Colorbar(fig[1, 3], limits=(minimum(gAs), maximum(gAs)), width = 20, colormap=colors, label=L"$g_A$", vertical=true,ticklabelsize=18, labelsize=20)

    ax[2].xlabel = L"\Delta_B / \omega_b"
    ax[2].ylabel = L"\text{Log negativity}" 
    ax[2].title = L"\text{Log negativity vs } \Delta_B"
    ax[1].xscale = log
    ax[1].yscale = log
    ax[2].xscale = log
    ax[2].yscale = log
    save("delta_B_g_b.png", fig; px_per_unit = 3)  
    return fig
end

