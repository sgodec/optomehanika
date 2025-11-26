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
gamma   = 2 * pi * 10      # 1 Hz damping
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

function simulate_srk2(mode::Int)
    #srk2 method is the one we will be using for simulations (Heun method-https://arxiv.org/pdf/2508.19040)
    #we define two different regimes with or without RWK approximation
    #1) No approximation
    
    if mode == 0
        g(a) = (-im*omega_m - gamma/2) .* a .+ gamma/2 .*  conj.(a) 

        a = zeros(ComplexF64, N, repeat)
        a[1,:] .= randn(repeat) .* sqrt(nbar/2) +randn(repeat) .* sqrt(nbar/2)

        dW = sqrt(dt) .* (im .* randn(N, repeat))
        @showprogress for i in 2:N
            @inbounds begin
            a_prev = a[i-1,:]
            ΔW = dW[i,:]

            k1 = g(a_prev)
            k2 = g(a_prev .+ k1*dt .+ sigma .* ΔW )

            a[i,:] = a_prev .+ 0.5*(k1 + k2)*dt .+ sigma .* ΔW
            end
        end
    end

    #2) RWK approximation
    if mode == 1
        f(a) = (-im*omega_m - gamma/2) .* a 

        a = zeros(ComplexF64, N, repeat)
        a[1,:] .= randn(repeat) .* sqrt(nbar/2) +randn(repeat) .* sqrt(nbar/2)

        dW = sqrt(dt/2) .* (randn(N,repeat) .+ im .* randn(N, repeat))

        @showprogress for i in 2:N
            @inbounds begin
            a_prev = a[i-1,:]
            ΔW = dW[i,:]

            k1 = f(a_prev)
            k2 = f(a_prev .+ k1*dt .+ sigma .* ΔW )

            a[i,:] = a_prev .+ 0.5*(k1 + k2)*dt .+ sigma .* ΔW
            end
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

function analytic_psd_qq_rwp(f)
    ω = 2π .* f
    χ_qq= (gamma/2 .- im .* ω)  ./ (omega_m^2 .- ω.^2 .- im .* gamma .* ω  .+ (gamma/2)^2 )
    χ_pp= (omega_m)  ./ (omega_m^2 .- ω.^2 .- im .* gamma .* ω  .+ (gamma/2)^2 )
    S = (abs.(χ_pp).^2 .+ abs.(χ_qq).^2) .* (gamma * (nbar + 1/2))
    return S
end

function analytic_psd_qq(f)
    ω = 2π .* f
    χ= (omega_m)  ./ (omega_m^2 .- ω.^2 .- im .* gamma .* ω ) 
    S = (abs.(χ).^2) .* (2 * gamma * (nbar + 1/2))
    return S
end

function analytic_psd_pp(f)
    ω = 2π .* f
    χ= (-im .* ω) ./ (omega_m^2 .- ω.^2 .- im .* gamma .* ω ) 
    S = (abs.(χ).^2) .* (2 * gamma * (nbar + 1/2))
    return S
end

function average_psd(X::AbstractMatrix, dt)
    N, R = size(X)

    f, Sref = psd_single(view(X,:,1), dt)
    Ssum = zero(Sref)

    @inbounds for r = 1:R
        _, Sr = psd_single(view(X,:,r), dt)
        Ssum .+= Sr
    end

    return f, Ssum ./ R
end

###################################################################################################################################
#PLOTING

function simulate_and_plot1(mode::Int)
    println("Simulating $repeat trajectories...")
    #choose method
    Xss, Pss = simulate_srk2(mode)

    println("Simulated ⟨n⟩ = ",mean(Xss[end,:].^2 .+ Pss[end,:].^2))

    println("Expected ⟨n⟩ + 1/2      = ", nbar + 0.5)

    x_traj = mean(Xss,dims=2)[:] .* sqrt(2)
    p_traj = mean(Pss,dims=2)[:] .* sqrt(2)
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
        lines!(ax[1], t, Xss[:, r] .* √2, color = colors[r], alpha = α )
        lines!(ax[2], Xss[:, r] .* √2, Pss[:, r] .* √(2), color = colors[r], alpha = α)
    end

    ax[1].xlabel = L"t"
    ax[1].ylabel = L"X"
    ax[1].title = "X(t) (10 anambles)"

    ylims!(ax[2], -4, 4)  
    xlims!(ax[2], -4, 4)  

    #lines!(ax[1], t, x_traj / √2, label="<X>", color = :black,linestyle = :dash)
    #lines!(ax[2], x_traj ./ √2 , p_traj, color = :black,linestyle = :dash)
    #
    ax[2].xlabel = L"X"
    ax[2].ylabel = L"P"
    ax[2].title = "Phase space (X,P) 10 ansambles"

    x = vec(Xss) .* √2 
    y = vec(Pss) .* √2 

    h = fit(Histogram, (x, y), nbins=(50, 50))
    heatmap!(ax[3], h.edges[1], h.edges[2], h.weights; colormap=:deepsea)
    ylims!(ax[3], -4, 4)  
    xlims!(ax[3], -4, 4)  
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

    freqs, PSD_num = average_psd(Xss  .+ im .* Pss , dt)

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
    xlims!(ax[4], 10^2, 10^4)  
    ax[4].xscale = log10
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

    save("fig0.png", fig)
    GC.gc()
end

function simulate_and_plot2()

    fig = Figure(size = (1500 ,500))

    ax = [
        Axis(fig[1,i], 
             width = 350, 
             height = 350,
             xticklabelsize = 16,
             yticklabelsize = 16,
             xlabelsize = 20,
             ylabelsize = 20,
             titlesize = 22) for i in 1:3
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
    title = ["ω_m/γ = 200 " ,"ω_m/γ = 20 " ,"ω_m/γ = 2"]
    for (i,g) in enumerate([2*pi*5,2*pi*50,2*pi*500])
        global gamma = g
        global sigma = sqrt(gamma * (nbar + 1/2))

        println("Simulating $repeat trajectories...")
        #choose method
        Xss_0, Pss_0 = simulate_srk2(0)
        Xss_1, Pss_1 = simulate_srk2(1)


        println("Simulated  0 ⟨n⟩ = ",mean(Xss_0[end,:].^2 .+ Pss_0[end,:].^2))
        println("Simulated  1 ⟨n⟩ = ",mean(Xss_1[end,:].^2 .+ Pss_1[end,:].^2))

        println("Expected ⟨n⟩ + 1/2      = ", nbar + 0.5)
        println("Computing PSD…")

        freqs_0, PSD_num_0 = average_psd(Xss_0 .* √2 .+ im .* 0, dt)
        freqs_1, PSD_num_1 = average_psd(Xss_1 .* √2 .+ im .* 0, dt)

        pos_idx_0 = freqs_0 .> 0
        pos_idx_1 = freqs_1 .> 0

        PSD_analytic_qq = analytic_psd_qq(freqs_0)
        PSD_analytic_qq_rwp = analytic_psd_qq_rwp(freqs_0)

        lines!(ax[i], freqs_0[pos_idx_0], PSD_num_0[pos_idx_0], 
           label="No RWA", 
           color=:teal,alpha= 0.5, linewidth=3)

        lines!(ax[i], freqs_1[pos_idx_1], PSD_num_1[pos_idx_1], 
           label="RWA", 
           color=:firebrick4, alpha = 0.5, linewidth=3) 

        lines!(ax[i], freqs_0[pos_idx_0], PSD_analytic_qq[pos_idx_0],
           color=:teal, linewidth=3, linestyle=:dash)

        lines!(ax[i], freqs_0[pos_idx_0], PSD_analytic_qq_rwp[pos_idx_0],
           color=:firebrick4, linewidth=3, linestyle=:dash)
        ax[i].yscale = log10
        xlims!(ax[i], 10^2, 10^4)  
        ax[i].xscale = log10
        ylims!(ax[i], 10^-10, 10^(-1))  

        ax[i].ylabel = L"S_{QQ}"
        ax[i].xlabel = L"frequency"
        ax[i].title = "S_QQ  $(title[i])"
        vlines!(ax[i], omega_m/ (2*pi), color=:darkgoldenrod1, linestyle=:dot, linewidth=2, label = L"\omega_m / 2π")

        GC.gc()
    end

    #axislegend(ax[1]; position=:lb, tellheight=false, tellwidth=false, labelsize=14)            
    fig[1, 4] = Legend(fig, ax[3],labelsize = 18,titlesize = 16)

    save("fig1.png", fig)
end

function simulate_and_plot3()

    fig = Figure(size = (1500 ,500))

    ax = [
        Axis(fig[1,i], 
             width = 350, 
             height = 350,
             xticklabelsize = 16,
             yticklabelsize = 16,
             xlabelsize = 20,
             ylabelsize = 20,
             titlesize = 22) for i in 1:3
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
    title = ["ω_m/γ = 200 " ,"ω_m/γ = 20 " ,"ω_m/γ = 2"]
    for (i,g) in enumerate([2*pi*5,2*pi*50,2*pi*500])
        global gamma = g
        global sigma = sqrt(gamma * (nbar + 1/2))

        println("Simulating $repeat trajectories...")
        #choose method
        Xss_0, Pss_0 = simulate_srk2(0)
        Xss_1, Pss_1 = simulate_srk2(1)


        println("Simulated  0 ⟨n⟩ = ",mean(Xss_0[end,:].^2 .+ Pss_0[end,:].^2))
        println("Simulated  1 ⟨n⟩ = ",mean(Xss_1[end,:].^2 .+ Pss_1[end,:].^2))

        println("Expected ⟨n⟩ + 1/2      = ", nbar + 0.5)
        println("Computing PSD…")

        freqs_0, PSD_num_0 = average_psd(Pss_0 .* √2 .+ im .* 0, dt)
        freqs_1, PSD_num_1 = average_psd(Pss_1 .* √2 .+ im .* 0, dt)

        pos_idx_0 = freqs_0 .> 0
        pos_idx_1 = freqs_1 .> 0

        PSD_analytic_pp = analytic_psd_pp(freqs_0)
        PSD_analytic_qq_rwp = analytic_psd_qq_rwp(freqs_0)

        lines!(ax[i], freqs_0[pos_idx_0], PSD_num_0[pos_idx_0], 
           label="No RWA", 
           color=:teal,alpha= 0.5, linewidth=3)

        lines!(ax[i], freqs_1[pos_idx_1], PSD_num_1[pos_idx_1], 
           label="RWA", 
           color=:firebrick4, alpha = 0.5, linewidth=3) 

        lines!(ax[i], freqs_0[pos_idx_0], PSD_analytic_pp[pos_idx_0],
           color=:teal, linewidth=3, linestyle=:dash)

        lines!(ax[i], freqs_0[pos_idx_0], PSD_analytic_qq_rwp[pos_idx_0],
           color=:firebrick4, linewidth=3, linestyle=:dash)
        ax[i].yscale = log10
        xlims!(ax[i], 10^2, 10^4)  
        ax[i].xscale = log10
        ylims!(ax[i], 10^-10, 10^(-1))  

        ax[i].ylabel = L"S_{PP}"
        ax[i].xlabel = L"frequency"
        ax[i].title = "S_PP  $(title[i])"
        vlines!(ax[i], omega_m/ (2*pi), color=:darkgoldenrod1, linestyle=:dot, linewidth=2, label = L"\omega_m / 2π")

        GC.gc()
    end

    #axislegend(ax[1]; position=:lb, tellheight=false, tellwidth=false, labelsize=14)            
    fig[1, 4] = Legend(fig, ax[3],labelsize = 18,titlesize = 16)

    save("fig2.png", fig)
end
