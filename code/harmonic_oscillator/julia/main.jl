using Random
using LaTeXStrings
using CairoMakie
using ProgressMeter
using LinearAlgebra
using Statistics
using FFTW
using StatsBase

MT = Makie.MathTeXEngine
mt_fonts_dir = joinpath(dirname(pathof(MT)), "..", "assets", "fonts", "NewComputerModern")

set_theme!(fonts = (
regular = joinpath(mt_fonts_dir, "NewCM10-Regular.otf"),
bold = joinpath(mt_fonts_dir, "NewCM10-Bold.otf")
) )

###################################################################################################################################
#Simulation of harmonic oscillator coupled to heatbath adding damping and heating term
#Mode of quantum harmonic oscillator can be expressed as \dot{a} = - i \omega_0 a - \gamma/2 + \sqrt{\gamma} a_{in} 
#where \braket{a^{\dag}(t) a(t')} = \bar{n} \delta(t-t') and \braket{a(t) a^{\dag}(t')} = (\bar{n}+1) \delta(t-t')
#in classical limit we get symetrised version of this \braket{a^*(t)a(t')} = (\bar{n} + 1/2) \delta(t-t') 
#We simulate this noise by simple scheme:
#a_{n+1} = a_n + \left(-i \omega_0 - \frac{\gamma}{2}\right) a_n \, \Delta t + \sigma \, dW_n
# where dW_n is a Gaussian random variable:
#dW_n = \frac{1}{\sqrt{2}} \left( \mathrm{randn}() + i \, \mathrm{randn}() \right) \sqrt{\Delta t}
###################################################################################################################################

#Physics parameters
const hbar = 6.62607015e-34 #J⋅Hz^−1
const k_b = 1.381e-23 #m^2 kg s^−1 K^−1
const hbar_dev_kb = hbar/k_b
T = 0.01 #K
omega_m = 2 * pi * 1000      # 1 kHz
gamma   = 2 * pi * 10        # 1 Hz damping
nbar    = 1 / (exp( hbar_dev_kb * omega_m / T) - 1) #average phonons ≈ 10e8             
nbar = 1 # we fix this so we are working at small values
sigma = sqrt(gamma * (nbar + 1/2)) #amplitude of noise
@show nbar

#Simulation parameters
dt = 1e-6                   
N  = 100_000                   
repeat = 1000   #repeated experiment          
burn = 1        #amount we throw away        

###################################################################################################################################
function simulate_euler()
    a = zeros(ComplexF64, N, repeat)
    a[1,:] .= sqrt(nbar) 

    dW = sqrt(dt/2) .* (randn(N, repeat) .+ im .* randn(N, repeat))

    drift = (-im*omega_m - gamma/2) * dt

    @inbounds for i in 2:N
        a[i, :] .= a[i-1, :] .+ drift .* a[i-1, :] .+ sigma .* dW[i, :]
    end

    X = real.(a)
    P = imag.(a)

    return X, P
end

function simulate_exp_euler()
    drift = -im*omega_m - gamma/2
    E = exp(drift * dt)                # exact drift propagator

    a = zeros(ComplexF64, N, repeat)
    a[1,:] .= sqrt(nbar)

    dW = sqrt(dt/2) .* (randn(N, repeat) .+ im .* randn(N, repeat))

    for i in 2:N
        a[i,:] = E .* a[i-1,:] .+ sigma .* dW[i,:]
    end

    X = real.(a)
    P = imag.(a)

    return X, P
end

function simulate_srk2()
    #rk method more of order 1
    f(a) = (-im*omega_m - gamma/2) .* a

    a = zeros(ComplexF64, N, repeat)
    a[1,:] .= randn(repeat) .* sqrt(nbar) 

    dW = sqrt(dt/2) .* (randn(N, repeat) .+ im .* randn(N, repeat))

    @showprogress for i in 2:N
        @inbounds begin
        a_prev = a[i-1,:]

        k1 = f(a_prev)
        η = randn(repeat) .+ im .* randn(repeat)  
        k2 = f(a_prev .+ k1*dt .+ sigma*sqrt(dt)*η)

        a[i,:] = a_prev .+ 0.5*(k1 + k2)*dt .+ sigma .* dW[i,:]
        end
    end

    return real.(a), imag.(a)
end

#calculating power specter
function psd_single(x::AbstractVector, dt)
     N = length(x)
    fs = 1/dt

    # Remove mean
    x = x .- mean(x)

    # FFT
    Xf = fft(x)

    Xf_shifted = fftshift(Xf)

    # centered
    f_shifted = (-N÷2:N÷2-1) .* (fs/N)

    Sx_shifted = (1 / (fs * N)) .* abs2.(Xf_shifted)
    S_sym = 0.5 .* (Sx_shifted .+ reverse(Sx_shifted))
    return f_shifted, S_sym
end

function analytic_psd(f)
    ω = 2π .* f
    χ = 1 ./ (-im .* (ω .- omega_m) .+ gamma/2)
    S = abs.(χ).^2 .* (gamma * (nbar + 1/2))
    return S
end

function average_psd(X::AbstractMatrix, dt)
    N, R = size(X)

    f, Sref = psd_single(view(X,:,1), dt)
    Ssum = zero(Sref)

    for r = 1:R
        _, Sr = psd_single(view(X,:,r), dt)
        Ssum .+= Sr
    end

    return f, Ssum ./ R
end

function simulate_and_plot()
    println("Simulating $repeat trajectories...")
    #choose method
    @time Xss, Pss = simulate_srk2()

    println("Simulated ⟨n⟩ = ",mean(Xss[end,:].^2 .+ Pss[end,:].^2))

    println("Expected ⟨n⟩ + 1/2      = ", nbar + 0.5)

    x_traj = mean(Xss,dims=2)[:] ./ sqrt(2)
    p_traj = mean(Pss,dims=2)[:] ./ sqrt(2)
    t = (1:length(x_traj)) * dt

    fig = Figure(size = (1750 ,500))

    ax = [
        Axis(fig[1,i], 
             width = 350, 
             height = 350,
             xticklabelsize = 16,
             yticklabelsize = 16,
             xlabelsize = 20,
             ylabelsize = 20,
             titlesize = 22) for i in 1:4
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

    cmap = :tab10         # or :viridis, :acton
    α = 0.3
    colors = cgrad(cmap, 10)   

    for r in 1:10
        lines!(ax[1], t, Xss[:, r] ./ √2, color = colors[r], alpha = α )
        lines!(ax[2], Xss[:, r] ./ √2, Pss[:, r] ./ √(2), color = colors[r], alpha = α)
    end

    ax[1].xlabel = L"t"
    ax[1].ylabel = L"X"
    ax[1].title = "X(t) (10 anambles)"

    ylims!(ax[2], -2, 2)  
    xlims!(ax[2], -2, 2)  

    #lines!(ax[1], t, x_traj / √2, label="<X>", color = :black,linestyle = :dash)
    #lines!(ax[2], x_traj ./ √2 , p_traj, color = :black,linestyle = :dash)
    #
    ax[2].xlabel = L"X"
    ax[2].ylabel = L"P"
    ax[2].title = "Phase space (X,P) 10 ansambles"

    x = vec(Xss) ./ √2 
    y = vec(Pss) ./ √2 

    h = fit(Histogram, (x, y), nbins=(50, 50))
    heatmap!(ax[3], h.edges[1], h.edges[2], h.weights; colormap=:viridis)
    ylims!(ax[3], -2, 2)  
    xlims!(ax[3], -2, 2)  
    ax[3].xlabel = L"\rho_X"
    ax[3].ylabel = L"\rho_P"
    ax[3].title = "Phase space histogram (X,P)"
    #Colorbar(
    #         fig[1, 3],
    #         limits=(0, 1),
    #         colormap=:viridis,
    #         vertical=true,
    #         ticklabelsize=14,
    #         labelsize=18
    #        )

    println("Computing PSD…")

    freqs, PSD_num = average_psd(Xss .+ im .* Pss, dt)

    PSD_analytic = analytic_psd(freqs)


    pos_idx = freqs .> 0
    lines!(ax[4], freqs[pos_idx], PSD_num[pos_idx], 
       label="Simulation", 
       color=:black, linewidth=6)

    lines!(ax[4], freqs[pos_idx], PSD_analytic[pos_idx],
       label=L"$|\frac{1}{-i(\omega-\omega_m)+\gamma/2}|^2 \gamma (\bar{n}+1/2)$",
       color=:firebrick4, linewidth=3, linestyle=:dash)

    vlines!(ax[4], omega_m/ (2*pi), color=:darkgoldenrod1, linestyle=:dot, linewidth=2, label = L"\frac{\omega_m}{2π}")
    
    ax[4].yscale = log10
    xlims!(ax[4], 10, 10^5)  
    ax[4].xscale = log10
    xlims!(ax[4], 10, 10^5)  
    ylims!(ax[4], 10^-10, 10^(-1))  

    ax[4].ylabel = L"S_{XX}"
    ax[4].xlabel = L"frequency"
    ax[4].title = "Power density specter"

    axislegend(ax[4]; position=:lb,       
           tellheight=false,         
           tellwidth=false,          
           labelsize=14)            

    function trapz(x, y)
        sum( (y[1:end-1] .+ y[2:end]) .* diff(x) ) / 2
    end

    PSD_integral = trapz(freqs, PSD_num)

    println("Simulated integral Sxx = ",PSD_integral)
    println("Theoretical integral  Sxx = ", (nbar+1/2))

    save("fig.png", fig)
end

