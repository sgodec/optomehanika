import numpy as np
import numba as nb
###################################################################################################################################
#Writen numericall recepies for simulating SDE source https://people.math.sc.edu/burkardt/m_src/stochastic_rk/stochastic_rk.html
#This functions are addapted in such way that they work for vector SDE where components of vector may be correlated and scalled differently

@nb.njit
def rk2_ti_step(x, t, h, noise1, noise2, step, transform_noise):
    '''
    inputs of function are
    x; state (12,M) ndarray
    t; doesn't matter (time independent)
    h; time step (dt)
    noise1,2; (12,M) ndarray, independently generated noises that are already scaled by (sqrt(2 kb T gamma)
    (questionable choise to not do it in transform_noise however this is it for now)
    step; callable returns (12,M) ndarray ,deterministic part of SDE dx= step(x)dt + dnoise 
    transform_noise; callable input noise returns scaled and correlated noise if there is no correlation transform_noise = id 
    '''
    a21 = 1.0
    a31 = 0.5
    a32 = 0.5

    q1 = 2.0
    q2 = 2.0

    x1 = x
    #we devide noise by sqrt(h) 
    w1 = transform_noise(x1,noise1) * np.sqrt ( q1 / h )
    #we multiply noise by h so the noise part is scaled by sqrt(h)
    k1 = h * step( x1 ) + h  * w1
    
    x2 = x1 + a21 * k1
    w2 = transform_noise(x2,noise2) * np.sqrt ( q2 / h )
    k2 = h * step( x2 ) + h * w2

    xstar = x1 + a31 * k1 + a32 * k2
    return xstar

@nb.njit
def rk4_ti_step(x, t, h, noise1, noise2, noise3, noise4, step, transform_noise):
    '''
    inputs of function are
    x; state (12,M) ndarray
    t; doesn't matter (time independent)
    h; time step (dt)
    noise1,2,3,4; (12,M) ndarray, independently generated noises that are already scaled by (sqrt(2 kb T gamma)
    (questionable choise to not do it in transform_noise however this is it for now)
    step; callable returns (12,M) ndarray ,deterministic part of SDE dx= step(x)dt + dnoise 
    transform_noise; callable input noise returns scaled and correlated noise if there is no correlation transform_noise = id 
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
  
    w1 = transform_noise(x1,noise1) * np.sqrt ( q1 / h )
    k1 = h * step (x1) + h *  w1

    x2 = x1 + a21 * k1

    w2 = transform_noise(x2,noise2) * np.sqrt ( q2 / h )
    k2 = h * step (x2) + h * w2

    x3 = x1 + a31 * k1 + a32 * k2

    w3 = transform_noise(x3,noise3) * np.sqrt ( q3 / h )
    k3 = h * step (x3) + h * w3

    x4 = x1 + a41 * k1 + a42 * k2

    w4 = transform_noise(x4,noise4) * np.sqrt ( q4 / h )
    k4 = h * step (x4) + h * w4

    xstar = x1 + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4

    return xstar    




        

