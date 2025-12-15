import numpy as np
import numba as nb
from numba import prange
import sympy as sp
from sympy import symbols, Function, Matrix, cos, sin, exp, I, pi, sqrt, Rational, lambdify
##################################################################################################################################
''' This is a program which is used to calculate Hamiltonian flow d\dt (q,p) =  (𝜕H/𝜕p, -𝜕H/𝜕q) '''
##################################################################################################################################
#Physical constants
c0 = 299792458          # speed of light in m/s
epsilon0 = 8.854187817e-12  # vacuum permittivity in F/m
hbar = 1.05457182e-34 # hbar
##################################################################################################################################
# we start by creating hamiltonian given parameters defined in main
def create_hamiltonian(params):
    #define needed symbols
    t = symbols('t', real=True)

    # Define coordinates and conjugate momenta
    x = Function('x')(t)
    y = Function('y')(t)
    z = Function('z')(t)

    p_x = Function('p_x')(t)
    p_y = Function('p_y')(t)
    p_z = Function('p_z')(t)

    #(Euler angles ZYZ convention)

    alpha = Function('alpha')(t)
    beta = Function('beta')(t)
    gamma = Function('gamma')(t)

    p_alpha = Function('p_alpha')(t)
    p_beta = Function('p_beta')(t)
    p_gamma = Function('p_gamma')(t)

    # Nanoparticle properties 
    V = params['V']
    m = params['m']

    # Laser parameters 
    P = params['P']

    # Beam parameters 
    w0 = params['w0']
    z_R = params['z_R']
    a1 = params['a1']

    # Susceptibility tensor components
    chi1 = params['chi1']
    chi2 = params['chi2']
    chi3 = params['chi3']

    # Inertia tensor components
    I1 = params['I1']
    I2 = params['I2']
    I3 = params['I3']

    # Polarization angle
    psi = params['psi']
    bx = cos(psi)
    by = sin(psi)

    # mode function and intensity

    def beam_width(z_val):
        return w0 * sqrt(1 + (z_val/z_R)**2)

    def intensity(x_val, y_val, z_val):
        w_z = beam_width(z_val)
        return (w0 / w_z)**2 * exp(-2 * (x_val**2/a1 + y_val**2*a1)/w_z**2)

    I_beam = intensity(x, y, z)
    
    # Rotational kinetic energy
    sin_b = sin(beta)
    cos_b = cos(beta)
    sin_g = sin(gamma)
    cos_g = cos(gamma)

    T_rot_term1 = (1/(I1 * (sin_b + 1e-6)**2)) * ((p_alpha - p_gamma * cos_b) * cos_g - p_beta * sin_b * sin_g)**2
    T_rot_term2 = (1/(I2 * (sin_b + 1e-6)**2)) * ((p_alpha - p_gamma * cos_b) * sin_g + p_beta * sin_b * cos_g)**2
    T_rot_term3 = (1/I3) * p_gamma**2

    T_rot = Rational(1,2) * (T_rot_term1 + T_rot_term2 + T_rot_term3)

    # Gradient potential
    term1_bx = chi1 * (cos(alpha)*cos(beta)*cos(gamma) - sin(alpha)*sin(gamma))**2
    term2_bx = chi2 * (cos(alpha)*cos(beta)*sin(gamma) + sin(alpha)*cos(gamma))**2
    term3_bx = chi3 * cos(alpha)**2 * sin(beta)**2
    bx_part = bx**2 * (term1_bx + term2_bx + term3_bx)

    term1_by = chi1 * (sin(alpha)*cos(beta)*cos(gamma) + cos(alpha)*sin(gamma))**2
    term2_by = chi2 * (cos(alpha)*cos(gamma) - sin(alpha)*cos(beta)*sin(gamma))**2
    term3_by = chi3 * sin(alpha)**2 * sin(beta)**2
    by_part = by**2 * (term1_by + term2_by + term3_by)

    H_gradient = - (V * P) / (c0 * pi * w0**2 ) * I_beam * (bx_part + by_part)

    # Translational kinetic energy
    T_trans = (p_x**2 + p_y**2 + p_z**2) / (2 * m)

    # Total Hamiltonian
    H = T_trans + T_rot + H_gradient

    #List of all coordinates and conjugate momenta
    q_list = [x, y, z, alpha, beta, gamma]
    p_list = [p_x, p_y, p_z, p_alpha, p_beta, p_gamma]
    return H, q_list, p_list

def hamilton_flow(H, q_list, p_list):
    "d\dt (q,p) =  (𝜕H/𝜕p, -𝜕H/𝜕q)"
    dq_dt_list = [sp.diff(H, p) for p in p_list]
    dp_dt_list = [-sp.diff(H, q) for q in q_list]
    
    return dq_dt_list, dp_dt_list

def hamilton_step(params):
    ''' return propagator function d/dt [q,p] = f(q,p)) '''

    H, q_list, p_list = create_hamiltonian(params)
    dq_dt_list, dp_dt_list = hamilton_flow(H,q_list,p_list)

    #create function that takes in  q_x...,p_gamma and return derivative presumablly it should be numpy function
    expr =dq_dt_list + dp_dt_list
    propagator = lambdify(q_list+p_list,expr,"numpy")
    H_func =lambdify(q_list+p_list,H,"numpy")
    return H_func, propagator

###################################################################################################################################
#end of symbolic calculation
#here we define helper function because propragaotr returns list not np.array which we need for numba
@nb.njit()
def stack(list_of_array):
    shape = (len(list_of_array),) + list_of_array[0].shape
    stacked_array = np.empty(shape)
    for j in prange(len(list_of_array)):
        stacked_array[j] = list_of_array[j]
    return stacked_array

# Here we generate numpy function derived from Hamiltonian 
def generate_functions_from_hamiltonian(params):
    m = params['m']
    V = params['V']

    P = params['P']
    lamb = params['lambda_']

    w0 = params['w0']
    z_R = params['z_R']
    a1 = params['a1']

    I1 = params['I1']
    I2 = params['I2']
    I3 = params['I3']

    chi1 = params['chi1']
    chi2 = params['chi2']
    chi3 = params['chi3']

    Gamma_c = params['dampingT']
    Gamma_a =params['dampingR']

    Noise_c = params["noiseT"]
    Noise_a = params["noiseR"]

    psi = params['psi']
    bx = np.cos(psi)
    by = np.sin(psi)

    sigma_R = np.pi**2 * V**2 / lamb**4
    sigma_L = np.pi * w0**2 / 2
    omega_L = 2 * np.pi * c0 / lamb
    Gamma_s = sigma_R * P / (sigma_L * omega_L * hbar)

    H_func,propagator = hamilton_step(params)

    f = nb.njit(propagator)
    h = nb.njit(H_func)

    @nb.njit
    def g(x):
        return stack((f(x[0], x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11])))   

    @nb.njit
    def H(x):
        return h(x[0], x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11])

    @nb.njit
    def noise(x,random):
        # matrix M^t R = 
        #-sin_b * cos_g, sin_b * sin_g , cos_b 
        #-sin_g * cos_g, cos_g , 0
        # 0,0,1
        sin_b = np.sin(x[4,:])  
        cos_b = np.cos(x[4,:])  
        sin_g = np.sin(x[5,:])  
        cos_g = np.cos(x[5,:])  

        dV_alpha = random[9,:].copy()
        dV_beta = random[10,:].copy()
        dV_gamma = random[11,:].copy()

        random[9,:] = -sin_b * cos_g * dV_alpha + sin_b * sin_g * dV_beta + cos_b * dV_gamma
        random[10,:] = sin_g * dV_alpha + cos_g * dV_beta
        return random

    @nb.njit
    def noise_faster(x, random):
        n = x.shape[1]

        for j in range(n):
            sb = np.sin(x[4, j])
            cb = np.cos(x[4, j])
            sg = np.sin(x[5, j])
            cg = np.cos(x[5, j])

            dVa = random[9, j]
            dVb = random[10, j]
            dVg = random[11, j]

            random[9, j]  = -sb * cg * dVa + sb * sg * dVb + cb * dVg
            random[10, j] =  sg * dVa + cg * dVb

        return random

    @nb.njit
    def determenistic_scattering(x):
        sin_a = np.sin(x[3,:])
        sin_2a = np.sin(2 * x[3,:])
        cos_a = np.cos(x[3,:])
        cos_2a = np.cos(2 * x[3,:])
        sin_b = np.sin(x[4,:])  
        cos_b = np.cos(x[4,:])  
        cos_2b = np.cos(2 * x[4,:])  
        sin_g = np.sin(x[5,:])  
        sin_2g = np.sin(2 * x[5,:])  
        cos_g = np.cos(x[5,:])  
        cos_2g = np.cos(2 * x[5,:])  

        w_z =  w0 * np.sqrt(1 + (x[2,:]/z_R)**2)
        u02 =  ((w0 / w_z)**2 * np.exp(-2 * (x[0,:]**2/a1 + x[1,:]**2*a1)/w_z**2))**2

        result  = np.zeros(x.shape)

        #dp determenistic_scattering
        result[8] =  16 * np.pi**2 * hbar * Gamma_s / (3 *lamb) * u02 * (1/2 * (by**2 - bx**2) * sin_2a * cos_b * sin_2g * (chi1**2 - chi2**2)  - 1/16 * cos_2g * (chi1**2-chi2**2) * (2 * (bx**2 -by**2) * cos_2a * (cos_2b+3) + 4 * sin_b**2)- 1/8 * (chi1**2 +  chi2**2 - 2 * chi3**2) * (2 * (bx**2-by**2)* cos_2a * sin_b**2 - cos_2b) + 3/8 * (chi1**2 + chi2**2) + 1/4 * chi3**2)

        #dalpha determenistic_scattering
        result[9] = 2 * np.pi * hbar * Gamma_s * bx * by / (3)* u02 * (-2 * sin_b**2 * cos_2g * (chi1 - chi2) * (chi1 + chi2 - 2* chi3) + cos_2b * (chi1**2 + 2*chi3 * (chi1 + chi2)- 4 * chi1 * chi2 + chi2**2-2*chi3**2 + 3 *chi1**2 -2* chi3 * (chi1 + chi2) - 4 * chi1 * chi2 + 3 * chi2**2 + 2 * chi3**2))

        #dbeta determenistic_scattering
        result[10] = 8 * np.pi * bx * by * hbar * Gamma_s / (3) * u02 * (sin_b * sin_g * cos_g * (chi1 -chi2) * (chi1 + chi2 - 2* chi3))

        #dgamma determenistic_scattering
        result[11] = 8 * np.pi * bx * by * hbar * Gamma_s / (3) * u02 * (cos_b * (chi1-chi2)**2)

        return result 

    @nb.njit
    def damping(x):
        result = np.zeros_like(x)
        for i in range(6, 9):
            result[i] = -Gamma_c * x[i]
        for i in range(9, 12):
            result[i] = -Gamma_a * x[i]
        return result

    

    @nb.njit
    def step(x):
        return g(x) + damping(x) + determenistic_scattering(x)

        
    @nb.njit
    def step_faster(x):
        out = g(x)

        # damping 
        for i in range(6, 9):
            out[i] -= Gamma_c * x[i]

        for i in range(9, 12):
            out[i] -= Gamma_a * x[i]

        sin_a = np.sin(x[3,:])
        sin_2a = np.sin(2 * x[3,:])
        cos_a = np.cos(x[3,:])
        cos_2a = np.cos(2 * x[3,:])
        sin_b = np.sin(x[4,:])  
        cos_b = np.cos(x[4,:])  
        cos_2b = np.cos(2 * x[4,:])  
        sin_g = np.sin(x[5,:])  
        sin_2g = np.sin(2 * x[5,:])  
        cos_g = np.cos(x[5,:])  
        cos_2g = np.cos(2 * x[5,:])  

        w_z =  w0 * np.sqrt(1 + (x[2,:]/z_R)**2)
        u02 =  ((w0 / w_z)**2 * np.exp(-2 * (x[0,:]**2/a1 + x[1,:]**2*a1)/w_z**2))**2


        #dp determenistic_scattering
        out[8] +=  16 * np.pi**2 * hbar * Gamma_s / (3 *lamb) * u02 * (1/2 * (by**2 - bx**2) * sin_2a * cos_b * sin_2g * (chi1**2 - chi2**2)  - 1/16 * cos_2g * (chi1**2-chi2**2) * (2 * (bx**2 -by**2) * cos_2a * (cos_2b+3) + 4 * sin_b**2)- 1/8 * (chi1**2 +  chi2**2 - 2 * chi3**2) * (2 * (bx**2-by**2)* cos_2a * sin_b**2 - cos_2b) + 3/8 * (chi1**2 + chi2**2) + 1/4 * chi3**2)

        #dalpha determenistic_scattering
        out[9] += 2 * np.pi * hbar * Gamma_s * bx * by / (3)* u02 * (-2 * sin_b**2 * cos_2g * (chi1 - chi2) * (chi1 + chi2 - 2* chi3) + cos_2b * (chi1**2 + 2*chi3 * (chi1 + chi2)- 4 * chi1 * chi2 + chi2**2-2*chi3**2 + 3 *chi1**2 -2* chi3 * (chi1 + chi2) - 4 * chi1 * chi2 + 3 * chi2**2 + 2 * chi3**2))

        #dbeta determenistic_scattering
        out[10] += 8 * np.pi * bx * by * hbar * Gamma_s / (3) * u02 * (sin_b * sin_g * cos_g * (chi1 -chi2) * (chi1 + chi2 - 2* chi3))

        #dgamma determenistic_scattering
        out[11] += 8 * np.pi * bx * by * hbar * Gamma_s / (3) * u02 * (cos_b * (chi1-chi2)**2)

        return out


    @nb.njit
    def generate_random(M,N):
        noise_all = np.zeros((12,M,N))

        noise_all[6:12,:,:] = np.random.randn(6,M,N)
        noise_all[6:9,:,:] = np.sqrt( m * Noise_c) *noise_all[6:9,:,:]

        noise_all[9,:,:] = np.sqrt(I1 * Noise_a) * noise_all[9,:,:]
        noise_all[10,:,:] = np.sqrt(I2 * Noise_a) * noise_all[10,:,:]
        noise_all[11,:,:] = np.sqrt(I3 * Noise_a) * noise_all[11,:,:]
        return noise_all

    return H,step_faster,noise_faster,generate_random,f,#step_faster,noise_faster

def build_params(overrides=None):
    '''
    Builds parameters for simulation
    
    Parameter
    
    overrides:dict
            defines base parameters from which others are computed.
    
    Returns
    
    params:dict
            parameters
    '''

    base = dict(
        rho=2200,
        a=2*80e-9,
        h=2*80e-9,
        T=300,
        lambda_=1064e-9,
        epsilon0=8.85e-12,
        c=3e8,
        P=200e-3,
        a1=1.0,
        psi=np.pi/3,
        Pressure=1.0,
    )

    if overrides:
        base.update(overrides)

    p = dict(base)

    
    p['V'] = 2 * p['a']**2 * p['h'] / 3
    a_reg = (3 * p['V'] / np.sqrt(2))**(1/3)
    p['r'] = a_reg / 2
    p['m'] = p['rho'] * p['V']
    p['I0'] = (2/5) * p['m'] * p['r']**2

    p['w0'] = p['lambda_'] / 1.3
    p['z_R'] = np.pi * p['w0']**2 / p['lambda_']

    a, h, rho = p['a'], p['h'], p['rho']
    I = 2 * ((a**4 * h)/60 + (a**2 * h**3)/30) * rho
    p['I1'] = I
    p['I2'] = I
    p['I3'] = (a**4 * h)/15

    p['chi1'] = np.sqrt((-p['I1'] + p['I2'] + p['I3']) / p['I0'])
    p['chi2'] = np.sqrt(( p['I1'] - p['I2'] + p['I3']) / p['I0'])
    p['chi3'] = np.sqrt(( p['I1'] + p['I2'] - p['I3']) / p['I0'])

    kb = 1.38e-23
    mg = 29 * 1.66e-27

    vg = np.sqrt((8/np.pi) * kb * p['T'] / mg)
    ng = p['Pressure'] / (kb * p['T'])
    gamma = ((4*np.pi/3) * mg * ng * p['r']**2) * vg * (1 + np.pi/8) / p['m']

    p['dampingT'] = gamma 
    p['dampingR'] = gamma 
    p['noiseT'] = 2 * kb * p['T'] * p['dampingT'] 
    p['noiseR'] = 2 * kb * p['T'] * p['dampingR'] 

    return p
params = build_params()
params
#H,step,noise,generate_random,f,step_faster,noise_faster  = generate_functions_from_hamiltonian(params)
#import time
#M = 10
#x =np.random.randn(12,M)
#step(x)
#step_faster(x)
#start = time.time()
#step(x)
#stop = time.time()
#print(stop-start)
#start = time.time()
#step_faster(x)
#stop = time.time()
#print(stop-start)
#print(sum(step(x) == step_faster(x)))
#noise(x,x)
#noise_faster(x,x)
#start = time.time()
#noise(x,x)
#stop = time.time()
#print(stop-start)
#start = time.time()
#noise_faster(x,x)
#stop = time.time()
#print(stop-start)
#print(sum(noise_faster(x,x) == noise(x,x)))



