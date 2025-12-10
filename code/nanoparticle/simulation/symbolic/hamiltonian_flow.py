import numpy as np
import numba as nb
from numba import prange
import sympy as sp
from sympy import symbols, Function, Matrix, cos, sin, exp, I, pi, sqrt, Rational, lambdify
##################################################################################################################################
''' This is a program which is used to calculate Hamiltonian flow d\dt (q,p) =  (𝜕H/𝜕p, -𝜕H/𝜕q) '''
##################################################################################################################################
#Physical constants
c = 299792458          # speed of light in m/s
epsilon0 = 8.854187817e-12  # vacuum permittivity in F/m
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

    H_gradient = - (V * P) / (c * pi * w0**2 ) * I_beam * (bx_part + by_part)

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
    I1 = params['I1']
    I2 = params['I2']
    I3 = params['I3']
    Gamma_c = params['dampingT']
    Gamma_a =params['dampingR']
    Noise_c = params["noiseT"]
    Noise_a = params["noiseR"]


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
    def damping(x):
        result  = np.zeros(x.shape)
        result[6:9] = - Gamma_c * x[6:9]
        result[9:12] = - Gamma_a * x[9:12]
        return result 

    @nb.njit
    def step(x):
        return g(x) - damping(x)

    @nb.njit
    def generate_random(M,N):
        noise_all = np.zeros((12,M,N))

        noise_all[6:12,:,:] = np.random.randn(6,M,N)
        noise_all[6:9,:,:] = np.sqrt( m * Noise_c) *noise_all[6:9,:,:]

        noise_all[9,:,:] = np.sqrt(I1 * Noise_a) * noise_all[9,:,:]
        noise_all[10,:,:] = np.sqrt(I2 * Noise_a) * noise_all[10,:,:]
        noise_all[11,:,:] = np.sqrt(I3 * Noise_a) * noise_all[11,:,:]
        return noise_all

    return H,step,noise,generate_random,f




