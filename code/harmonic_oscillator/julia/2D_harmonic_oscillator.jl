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
###################################################################################################################################

#Physics parameters
const hbar = 6.62607015e-34 #J⋅Hz^−1
const k_b = 1.381e-23 #m^2 kg s^−1 K^−1
const hbar_dev_kb = hbar/k_b
T = 0.01 #K

###################################################################################################################################
function simulate_2d_srk2(mode::Int, omega_1::Float64, omega_2::Float64,gamma::Float64, g_coupling::Float64,nbar::Float64,dt::Float64,N::Int,repeat::Int)
    sigma = sqrt(gamma * (nbar + 1/2)) #amplitude of noise

    if mode == 0
        g(a, b) = (
            (-im*omega_1 - gamma/2) .* a .+ gamma/2 .* conj.(a) .- im*g_coupling .* b,
            (-im*omega_2 - gamma/2) .* b .+ gamma/2 .* conj.(b) .- im*g_coupling .* a
        )

        a = zeros(ComplexF64, N, repeat)
        b = zeros(ComplexF64, N, repeat)
        a[1,:] .= randn(repeat) .* sqrt(nbar/2) .+ im .* randn(repeat) .* sqrt(nbar/2)
        b[1,:] .= randn(repeat) .* sqrt(nbar/2) .+ im .* randn(repeat) .* sqrt(nbar/2)

        dW_a = sqrt(dt) .* (im .* randn(N, repeat))
        dW_b = sqrt(dt) .* (im .* randn(N, repeat))
        
        @showprogress for i in 2:N
            @inbounds begin
            a_prev, b_prev = a[i-1,:], b[i-1,:]
            ΔW_a, ΔW_b = dW_a[i,:], dW_b[i,:]

            k1_a, k1_b = g(a_prev, b_prev)
            k2_a, k2_b = g(a_prev .+ k1_a*dt .+ sigma .* ΔW_a, 
                           b_prev .+ k1_b*dt .+ sigma .* ΔW_b)

            a[i,:] = a_prev .+ 0.5*(k1_a + k2_a)*dt .+ sigma .* ΔW_a
            b[i,:] = b_prev .+ 0.5*(k1_b + k2_b)*dt .+ sigma .* ΔW_b
            end
        end

    elseif mode == 1
        f(a, b) = (
            (-im*omega_1 - gamma/2) .* a .- im*g_coupling .* (b) +conj(b),
            (-im*omega_2 - gamma/2) .* b .- im*g_coupling .* (a) +conj(a)
        )

        a = zeros(ComplexF64, N, repeat)
        b = zeros(ComplexF64, N, repeat)
        a[1,:] .= (randn(repeat)  .+ im .* randn(repeat)) .* sqrt(nbar/2)
        b[1,:] .= (randn(repeat)  .+ im .* randn(repeat)) .* sqrt(nbar/2)

        dW_a = sqrt(dt/2) .* (randn(N,repeat) .+ im .* randn(N, repeat))

        dW_b = sqrt(dt/2) .* (randn(N,repeat) .+ im .* randn(N, repeat))
        
        @showprogress for i in 2:N
            @inbounds begin
            a_prev, b_prev = a[i-1,:], b[i-1,:]
            ΔW_a, ΔW_b = dW_a[i,:], dW_b[i,:]

            k1_a, k1_b = f(a_prev, b_prev)
            k2_a, k2_b = f(a_prev .+ k1_a*dt .+ sigma .* ΔW_a, 
                           b_prev .+ k1_b*dt .+ sigma .* ΔW_b)

            a[i,:] = a_prev .+ 0.5*(k1_a + k2_a)*dt .+ sigma .* ΔW_a
            b[i,:] = b_prev .+ 0.5*(k1_b + k2_b)*dt .+ sigma .* ΔW_b
            end
        end
    end

    return real.(a), imag.(a), real.(b), imag.(b)
end

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

function compute_cross_correlation(x1, x2,dt, max_lag=1000)
    x1_single = x1[:, 1]
    x2_single = x2[:, 1]
    
    cross_corr = crosscor(x1_single, x2_single, -max_lag:max_lag)
    lags = (-max_lag:max_lag) .* dt
    return lags, cross_corr
end
function cross_psd_single(x::AbstractVector, y::AbstractVector, dt)
    N = length(x)
    fs = 1/dt

    # Remove means
    x = x .- mean(x)
    y = y .- mean(y)

    # FFT
    Xf = fft(x)
    Yf = fft(y)

    # Cross-PSD calculation (similar structure to your PSD)
    Sxy_shifted = (1 / (fs * N)) .* (conj.(Xf) .* Yf)
    
    # Shift to center
    Sxy_shifted = fftshift(Sxy_shifted)
    
    # Frequency axis (centered)
    f_shifted = (-N÷2:N÷2-1) .* (fs/N)

    return f_shifted, Sxy_shifted
end

function average_cross_psd(X::AbstractMatrix, Y::AbstractMatrix, dt)
    N, R = size(X)

    # Get reference from first trajectory
    f, Sxy_ref = cross_psd_single(view(X, :, 1), view(Y, :, 1), dt)
    Sxy_sum_real = zero(real.(Sxy_ref))
    Sxy_sum_imag = zero(imag.(Sxy_ref))

    @inbounds for r = 1:R
        _, Sxy = cross_psd_single(view(X, :, r), view(Y, :, r), dt)
        Sxy_sum_real .+= real.(Sxy)
        Sxy_sum_imag .+= imag.(Sxy)
    end

    # Average and recombine as complex
    Sxy_avg = complex.(Sxy_sum_real ./ R, Sxy_sum_imag ./ R)
    
    return f, Sxy_avg
end

function simulate_and_plot(mode::Int)
    omega_1 = 2 * pi * (1000+500)      # 1 kHz
    omega_2 = 2 * pi * 1000      # 1 kHz
    gamma   = 2 * pi * 10      # 1 Hz damping
    nbar    = 1. 
    g_coupling = 2 * pi * 50.

    dt = 1e-6                   
    N  = 100_000                   
    repeat = 1000   #repeated experiment          

    println("Simulating $repeat trajectories, omega_1 = $omega_1, omega_2 = $omega_2,g = $g_coupling ")
    Xss, Pss, Xbb, Pbb = simulate_2d_srk2(1, omega_1, omega_2,gamma, g_coupling,nbar,dt,N,repeat)

    println("Simulated ⟨n⟩ = ",mean(Xss[end,:].^2 .+ Pss[end,:].^2))

    println("Expected ⟨n⟩ + 1/2      = ", nbar + 0.5)

    x_traj = mean(Xss,dims=2)[:] .* sqrt(2)
    p_traj = mean(Pss,dims=2)[:] .* sqrt(2)
    t = (1:length(x_traj)) * dt

    fig = Figure(size = (1350 ,500))

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
    

    lags, cross_corr = compute_cross_correlation(Xss, Xbb,dt, 1000)
    lines!(ax[1], lags, cross_corr, color=:purple, linewidth=2)

    ax[1].title = "Cross correlation <X_A X_B>"
    ax[1].xlabel = L"Time\ lag\ \tau"
    ax[1].ylabel = L"C_{AB}(\tau)"

    hlines!(ax[1], [0.0], color=:black, linestyle=:solid, linewidth=0.5)

    println("Computing Cross PSD…")

    freqs_cross, CPSD_avg = average_cross_psd(Xss, Xbb, dt)

    pos_idx_cross = freqs_cross .> 0

    #lines!(ax[3], freqs_cross[pos_idx_cross], real.(CPSD_avg[pos_idx_cross]), 
           #label="Real part", linewidth=2, color=:teal)
    #lines!(ax[3], freqs_cross[pos_idx_cross], imag.(CPSD_avg[pos_idx_cross]), 
           #label="Imag part", linewidth=2, color=:red)
    lines!(ax[2], freqs_cross[pos_idx_cross], abs.(CPSD_avg[pos_idx_cross]), 
           label="Magnitude", linewidth=3, color=:black, linestyle=:dash)

    axislegend(ax[2]; position=:rt, tellheight=false, tellwidth=false, labelsize=14)

    ax[2].title = "Cross PSD"
    ax[2].xlabel = L"Frequency\ (Hz)"
    ax[2].ylabel = L"S_{AB}(f)"
    xlims!(ax[2], 10^2, 10^4)  
    ax[2].xscale = log10
    ylims!(ax[2], 10^-10, 10^(-1))  
    ax[2].yscale = log10

    ω_avg = (omega_1 + omega_2)/2
    ω_split = sqrt(((omega_1 - omega_2)/2)^2 + g_coupling^2)
    ω_plus = ω_avg + ω_split
    ω_minus = ω_avg - ω_split

    vlines!(ax[2], [ω_plus/(2π), ω_minus/(2π)], color=[:cyan,:orange],
            linestyle=:dash, linewidth=3)

    println("Computing PSD…")

    freqs_a, PSD_a = average_psd(Xss .+ im .* Pss, dt)
    freqs_b, PSD_b = average_psd(Xbb .+ im .* Pbb, dt)

    pos_idx = freqs_a .> 0
    lines!(ax[3], freqs_a[pos_idx], PSD_a[pos_idx], label="PSD A", linewidth=3, color = :teal)
    lines!(ax[3], freqs_b[pos_idx], PSD_b[pos_idx], label="PSD B", linewidth=3, color = :firebrick4)


    vlines!(ax[3], [ω_plus/(2π), ω_minus/(2π)], color=[:cyan,:orange], 
            linestyle=:dash, linewidth=3)
    
    ax[3].yscale = log10
    xlims!(ax[3], 10^2, 10^4)  
    ax[3].xscale = log10
    ylims!(ax[3], 10^-10, 10^(-1))  

    ax[3].ylabel = L"S_{XX}"
    ax[3].xlabel = L"frequency"
    ax[3].title = "PSD (A,B)"

    axislegend(ax[3]; position=:lb,       
           tellheight=false,         
           tellwidth=false,          
           labelsize=14)            

    save("fig0_2d.png", fig)
    GC.gc()
end
