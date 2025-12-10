import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 22,
    "axes.titlesize": 24,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})

##################################################################################################################################
#code used for ploting simulated dynamics and powerspectrum
def plot_coordinates(t,sol):
    plt.figure(figsize=(14,10))

    plt.plot(t, sol[0,0, :], label='x(t)')
    plt.plot(t, sol[1,0, :], label='y(t)')
    plt.plot(t, sol[2,0, :], label='z(t)')

    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Translational Motion of Nanoparticle')
    plt.legend()
    plt.grid(True)
    plt.savefig("fig0.png", dpi=300) 
    print("Ploted coordinates, saved as fig0")

    return 0

def plot_angle(t,sol):
    plt.figure(figsize=(14,10))

    plt.plot(t, sol[3,0, :], label='alpha(t)')
    plt.plot(t, sol[4,0, :], label='beta(t)')
    plt.plot(t, sol[5,0, :], label='gamma(t)')

    plt.xlabel('Time')
    plt.ylabel('angle')
    plt.title('Eulare angles of Nanoparticle')
    plt.legend()
    plt.grid(True)
    plt.savefig("fig1.png", dpi=300) 
    plt.figure(figsize=(10,6))
    print("Ploted angles, saved as fig1")

    return 0

def plot_H(t,sol_H):
    plt.figure(figsize=(14,10))
    plt.plot(t, sol_H, label='H(t)')

    plt.xlabel('Time')
    plt.ylabel('H')
    plt.title('Energy(t)')
    plt.legend()
    plt.grid(True)
    plt.savefig("fig2.png", dpi=300) 
    print("Ploted hamiltonian fig2")

    return 0

    x = sol[1,0,:]

def plot_power(dt, sol):
    variable_names = [
    r'$x$', r'$y$', r'$z$',
    r'$\alpha$', r'$\beta$', r'$\gamma$',
    r'$p_x$', r'$p_y$', r'$p_z$',
    r'$p_\alpha$', r'$p_\beta$', r'$p_\gamma$'
    ]
    n_vars = len(variable_names)
    
    plt.figure(figsize=(14,10))
    
    for i in range(n_vars):
        data = sol[i, :, :] 
        
        P_list = []
        for repeat in range(data.shape[0]):
            signal_i = data[repeat, :] - np.mean(data[repeat, :])
            f, P = signal.periodogram(signal_i, fs=1/dt)
            P_list.append(P)
        
        P_mean = np.mean(P_list, axis=0)
        
        positive = f > 0
        plt.plot(f[positive], P_mean[positive], label=variable_names[i])
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power spectral density')

    plt.yscale("log")
    plt.xlim((0,5*10**6))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title("Power Spectrum ")
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc = "upper right")
    plt.savefig("fig3.png", dpi=300)

    print("Ploted powerspectrum at fig3")
    plt.show()
    return 0

def plot_combine(dt,t,sol):
    print("ploting combined")
    fig = plt.figure(figsize=(14, 14), layout='constrained')
    axs = fig.subplot_mosaic([["powerspectrum", "powerspectrum"],
                              ["position", "angle"],
                              ["conjugate_position", "conjugate_angle"]])

    variable_names = [
        r'$x$', r'$y$', r'$z$',
        r'$\alpha$', r'$\beta$', r'$\gamma$',
        r'$p_x$', r'$p_y$', r'$p_z$',
        r'$p_\alpha$', r'$p_\beta$', r'$p_\gamma$'
    ]
    
    n_vars = len(variable_names)
    
    #colors = [
    #"#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    #"#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#1abc9c", "#f0ad4e"
    #]

    colors = [
        "#000000", 
        "#E69F00", 
        "#56B4E9", 
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#999999",
        "#006400",
        "#008080",
        "#808000",
    ]
    
    for i in range(n_vars):
        data = sol[i, :, :]  #
        P_list = []
        for repeat in range(data.shape[0]):
            signal_i = data[repeat, :] - np.mean(data[repeat, :])
            f, P = signal.periodogram(signal_i, fs=1/dt)
            P_list.append(P)
        P_mean = np.mean(P_list, axis=0)
        positive = f > 0
        axs["powerspectrum"].plot(f[positive] / 1000, P_mean[positive], label=variable_names[i], color=colors[i], lw=1.5)
    print("ploted powerspectrum")
    
    axs["powerspectrum"].set_title("Power Spectrum", fontsize=18)
    axs["powerspectrum"].set_xlabel("Frequency [kHz]", fontsize=14)
    axs["powerspectrum"].set_ylabel("Power [units²/Hz]", fontsize=14)
    axs["powerspectrum"].set_yscale("log")
    axs["powerspectrum"].set_xlim(0, 3e3)
    axs["powerspectrum"].grid(True, which="both", ls="--", lw=0.5)
    axs["powerspectrum"].legend(fontsize=14, loc="upper right", ncol=2)
    
    def plot_time(ax, indices, title):
        for idx in indices:
            data = sol[idx, :, :]
            mean_signal = np.mean(data, axis=0)
            std_signal = np.std(data, axis=0)
            ax.plot(t, mean_signal, label=variable_names[idx], color=colors[idx], lw=1.5)
            ax.fill_between(t, mean_signal-std_signal, mean_signal+std_signal, color=colors[idx], alpha=0.2)
        ax.set_xlabel("Time [s]", fontsize=14)
        ax.set_ylabel("Signal", fontsize=14)
        ax.set_title(title, fontsize=18)
        ax.grid(True, ls="--", lw=0.5)
        ax.legend(fontsize=10,loc = "upper right")

    print("ploted signal")
    
    plot_time(axs["position"], [0,1,2], "Position Dynamics")
    plot_time(axs["angle"], [3,4,5], "Angle Dynamics")
    plot_time(axs["conjugate_position"], [6,7,8], "Conjugate Momentum Dynamics")
    plot_time(axs["conjugate_angle"], [9,10,11], "Conjugate Angle Dynamics")
    
    plt.savefig("figfinal.png", dpi=300)

    return 0

