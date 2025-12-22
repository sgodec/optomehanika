import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from matplotlib.animation import FuncAnimation

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
    mean_signal = np.mean(sol_H,axis= 0)
    std_signal = np.std(sol_H,axis= 0)
    plt.plot(t,mean_signal , label='H(t)',color = "teal")
    plt.fill_between(t, mean_signal-std_signal, mean_signal+std_signal, color="teal", alpha=0.2)

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
    n_vars = len(variable_names)
    
    plt.figure(figsize=(14,10))
    
    for i in range(n_vars // 2):
        data = sol[i, :, :] 
        
        f, P_all = signal.periodogram(data - np.mean(data, axis=1, keepdims=True), fs=1/dt, axis=1)

        P_mean = np.mean(P_all, axis=0)
        
        positive = f > 0
        plt.plot(f[positive] / 1000, P_mean[positive], label=variable_names[i],color = colors[i])

    

    plt.yscale("log")
    plt.xlim((0,3*10**3))
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Power")
    plt.title("Power Spectrum ")
    plt.grid(True)
    plt.tight_layout()
    plt.legend(fontsize = 16,loc = "upper right")
    plt.savefig("fig3.png", dpi=300)
    print("Ploted powerspectrum at fig3")
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
    
    for i in range(n_vars // 2):
        data = sol[i, :, :]  #
        f, P_all = signal.periodogram(data - np.mean(data, axis=1, keepdims=True), fs=1/dt, axis=1)
        P_mean = np.mean(P_all, axis=0)
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

    plot_time(axs["position"], [0,1,2], "Position Dynamics")
    plot_time(axs["angle"], [3,4,5], "Angle Dynamics")
    plot_time(axs["conjugate_position"], [6,7,8], "Conjugate Momentum Dynamics")
    plot_time(axs["conjugate_angle"], [9,10,11], "Conjugate Angle Dynamics")
    print("ploted signal")
    
    
    plt.savefig("figfinal.png", dpi=300)
    print("saved as figfinal.png")

    return 0

def plot_global_orientation(sol):
    c_a = np.cos(sol[3,0,:])
    s_a = np.sin(sol[3,0,:])
    c_b = np.cos(sol[4,0,:])
    s_b = np.sin(sol[4,0,:])
    c_g = np.cos(sol[5,0,:])
    s_g = np.sin(sol[5,0,:])
    # euler angles transformed to global coordinate
    X1x = c_a * c_b * c_g - s_a * s_g
    X1y = c_a * s_g + c_b * c_g * s_a
    X1z =  -c_g * s_b

    X2x = -c_g * s_a  - c_a *c_b * s_g
    X2y = c_a * c_g - c_b * s_a * s_g
    X2z =  s_b * s_g

    X3x = c_a * s_b 
    X3y = s_a * s_b 
    X3z =  c_b

    print(np.allclose(
    X1x*X2x + X1y*X2y + X1z*X2z, 0, atol=1e-6
    ))


    fig = plt.figure(figsize=(12, 4))

    axes = [
        fig.add_subplot(131, projection='3d'),
        fig.add_subplot(132, projection='3d'),
        fig.add_subplot(133, projection='3d')
    ]

    datasets = [
        (X1x, X1y, X1z),
        (X2x, X2y, X2z),
        (X3x, X3y, X3z)
    ]

    def plot_sphere(ax, radius=1.0, alpha=0.15):
        u = np.linspace(0, 2*np.pi, 50)
        v = np.linspace(0, np.pi, 50)

        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones_like(u), np.cos(v))

        ax.plot_surface(x, y, z, color='gray', alpha=alpha, linewidth=0)


    for ax, (x, y, z) in zip(axes, datasets):
        plot_sphere(ax, radius=1.0)
         
        ax.plot(x, y, z, linewidth=1, color='blue')

        ax.scatter(x[0], y[0], z[0], color='green', s=20)

        ax.scatter(x[-1], y[-1], z[-1], color='red', s=20)

        ax.set_box_aspect([1,1,1]) 

    plt.tight_layout()
    plt.show()
    return 0

def animate_orientation(sol,skip = 10**3,interval = 100,end = 10**5):
    c_a = np.cos(sol[3,0,:end:skip])
    s_a = np.sin(sol[3,0,:end:skip])
    c_b = np.cos(sol[4,0,:end:skip])
    s_b = np.sin(sol[4,0,:end:skip])
    c_g = np.cos(sol[5,0,:end:skip])
    s_g = np.sin(sol[5,0,:end:skip])
    
    X1x = c_a * c_b * c_g - s_a * s_g
    X1y = c_a * s_g + c_b * c_g * s_a
    X1z =  -c_g * s_b

    X2x = -c_g * s_a  - c_a *c_b * s_g
    X2y = c_a * c_g - c_b * s_a * s_g
    X2z =  s_b * s_g

    X3x = c_a * s_b 
    X3y = s_a * s_b 
    X3z =  c_b


    T = len(X1x)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    #unit sphere
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="gray", alpha=0.08, linewidth=0)

    q1 = ax.quiver(0, 0, 0, X1x[0], X1y[0], X1z[0], color="r", linewidth=3)
    q2 = ax.quiver(0, 0, 0, X2x[0], X2y[0], X2z[0], color="g", linewidth=3)
    q3 = ax.quiver(0, 0, 0, X3x[0], X3y[0], X3z[0], color="b", linewidth=3)

    def update(i):
        nonlocal q1, q2, q3

        q1.remove()
        q2.remove()
        q3.remove()

        q1 = ax.quiver(0, 0, 0, X1x[i], X1y[i], X1z[i], color="r", linewidth=3)
        q2 = ax.quiver(0, 0, 0, X2x[i], X2y[i], X2z[i], color="g", linewidth=3)
        q3 = ax.quiver(0, 0, 0, X3x[i], X3y[i], X3z[i], color="b", linewidth=3)

        return q1, q2, q3
    ani = FuncAnimation(fig, update, frames=T, interval=interval,blit=True)

    ani.save(
    "animation.mp4",
    writer="ffmpeg",
    fps=30,
    dpi=300,
    extra_args=["-pix_fmt", "yuv420p"])

    plt.show()
    return ani



