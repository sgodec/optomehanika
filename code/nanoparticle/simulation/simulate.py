import numpy as np
import numba as nb
import time
from symbolic.hamiltonian_flow import generate_functions_from_hamiltonian
from integrator.integrate import rk2_ti_step, rk4_ti_step
from plotting.plots  import plot_coordinates, plot_angle, plot_H, plot_power, plot_combine
###################################################################################################################################
params = {
    'rho': 2200,
    'a': 2*80e-9,
    'h': 2*80e-9,
    'epsilon0' : 8.85e-12,
    'c' : 3e8,
    'T' : 300,
    'lambda' : 1064e-9,
    'P' : 200e-3,
    'z_R': 2.95e-6,
    'a1': 1.0,
    'psi': 0,
}
params['V'] = 2 * params['a']**2 * params['h'] / 3
a_regular = (3 * params['V'] / (2**0.5))**(1/3)
r = a_regular / 2
params["r"] = r
params['m'] =params['rho']*params['V'];
params['I0'] = (2/5) * params['r']**2 * params['m']
params['w0'] = params['lambda']/(1.3)
params['z_R'] = np.pi * params['w0']**2 / params['lambda'];
params['I1']= 2*((params['a']**4 * params['h']) / 60 + (params['a']**2 * params['h']**3) / 30)  * params['rho'];
params['I2']= 2*((params['a']**4 * params['h']) / 60 + (params['a']**2 * params['h']**3) / 30)  * params['rho'];
params['I3']= (params['a']**4 * params['h']) / 15
params['chi1'] = np.sqrt((-params['I1']+ params['I2'] + params['I3'])/ params['I0'])
params['chi2'] = np.sqrt((params['I1'] - params['I2'] + params['I3'])/ params['I0'])
params['chi3'] = np.sqrt((params['I1'] + params['I2'] - params['I3'])/ params['I0'])
kb = 1.38e-23
Pressure = 100 * 10  # in Pa units
# gas particles
mg = 29 * 1.66e-27
vg = np.sqrt((8 / np.pi) * kb * params['T'] / mg)
ng = Pressure / (kb * params['T'])
gamma = ((4 * np.pi / 3) * mg * ng * r**2) * vg * (1 + np.pi / 8) / params['m']
params['dampingT'] = gamma * 0.1
params['dampingR'] = gamma  * 0.1

params['noiseT'] = 2 * kb * params['T'] * params['dampingT'] * 0.05
params['noiseR'] = 2 * kb * params['T'] * params['dampingR'] * 0.05
#params['I2'] = params['I1']
#params['I3'] = 2 *params['I2']

#params['chi1'] = params['chi2']
#params['chi3'] = 1/2 * params['chi2']

###################################################################################################################################
#simulation function
@nb.njit
def run(x0,h,M,N,fi,gi,noise_all):
    solution = np.zeros((12,M,N))
    solution[:,:,0] = x0

    for i in range(1,N):
        solution[:,:,i] = rk4_ti_step(solution[:,:,i-1],0,h,noise_all[:,:,i],fi,gi)
    return solution
###################################################################################################################################

H, step , noise , generate_random, f = generate_functions_from_hamiltonian(params)

def simulate_opto(N,M,dt,state0 = [1e-7, 1e-7, 1e-7, 0.1, np.pi/2-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0]):
    x0 = np.zeros((12,M))
    t = np.arange(N) * dt
    for i in range(M):
        x0[:,i] = state0

    noise_all = generate_random(M,N)  
    start = time.time()
    sol = run(x0,dt,M,N,step,noise,noise_all)
    stop = time.time()
    print(stop-start)

    return t,sol,np.array(H(sol))


