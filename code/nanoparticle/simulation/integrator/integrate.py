import numpy as np
import numba as nb
import time
###################################################################################################################################
#Writen numericall recepies for simulating SDE source https://people.math.sc.edu/burkardt/m_src/stochastic_rk/stochastic_rk.html

@nb.njit
def rk2_ti_step(x, t, h, noise, fi, gi):
    '''
    Perform a single time-independent second-order Runge–Kutta step
    for a stochastic differential equation.

    Advances the state x by one time step h for the SDE
        dx = f(x) dt + g(x) dW,
    where f(x) is the deterministic drift and g(x) is the state-dependent
    noise amplitude.

    Parameters
    x : ndarray
        Current state vector.
    t : float
        Current time (not used explicitly)
    h : float
        Time step size.
    noise : ndarray
        Wiener increment dW for the current time step.
    fi : callable
        Deterministic drift function f(x).
    gi : callable
        Noise amplitude function g(x), which may depend on the state.

    Returns
    xstar : ndarray
        Update
    '''

    a21 = 1.0
    a31 = 0.5
    a32 = 0.5

    q1 = 2.0
    q2 = 2.0

    x1 = x
    w1 = gi(x,noise) * np.sqrt ( q1 / h )
    k1 = h * fi ( x1 ) + h  * w1
    
    x2 = x1 + a21 * k1
    w2 = gi(x,noise) * np.sqrt ( q2 / h )
    k2 = h * fi ( x2 ) + h * w2

    xstar = x1 + a31 * k1 + a32 * k2
    return xstar

@nb.njit
def rk4_ti_step(x, t, h, noise, fi, gi):
    '''
    Perform a single time-independent fourth-order Runge–Kutta step
    for a stochastic differential equation.

    Advances the state x by one time step h for the SDE
        dx = f(x) dt + g(x) dW,
    where f(x) is the deterministic drift and g(x) is the state-dependent
    noise amplitude.

    Parameters
    x : ndarray
        Current state vector.
    t : float
        Current time (not used explicitly)
    h : float
        Time step size.
    noise : ndarray
        Wiener increment dW for the current time step.
    fi : callable
        Deterministic drift function f(x).
    gi : callable
        Noise amplitude function g(x), which may depend on the state.

    Returns
    xstar : ndarray
        Update
    '''
    a21 =   2.71644396264860
    a31 = - 6.95653259006152
    a32 =   0.78313689457981
    a41 =   0.0
    a42 =   0.48257353309214
    a43 =   0.26171080165848
    a51 =   0.47012396888046
    a52 =   0.36597075368373
    a53 =   0.08906615686702
    a54 =   0.07483912056879

    q1 =   2.12709852335625
    q2 =   2.73245878238737
    q3 =  11.22760917474960
    q4 =  13.36199560336697

    x1 = x
  
    w1 = gi(x1,noise) * np.sqrt ( q1 / h )
    k1 = h * fi (x1) + h *  w1

    x2 = x1 + a21 * k1

    w2 = gi(x2,noise) * np.sqrt ( q2 / h )
    k2 = h * fi (x2) + h * w2

    x3 = x1 + a31 * k1 + a32 * k2

    w3 = gi(x3,noise) * np.sqrt ( q3 / h )
    k3 = h * fi (x3) + h * w3

    x4 = x1 + a41 * k1 + a42 * k2

    w4 = gi(x4,noise) * np.sqrt ( q4 / h )
    k4 = h * fi (x4) + h * w4

    xstar = x1 + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4
    return xstar    




        

