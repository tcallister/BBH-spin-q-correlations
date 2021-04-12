import numpy as np
from scipy.special import erf,erfc

mMin = 5.
zMax = 3.

def asym(x):
    return -np.exp(-x**2)/np.sqrt(np.pi)/x*(1.-1./(2.*x**2))

def calculate_Gaussian(x, mu, sigma2, low, high): 
    a = (low-mu)/np.sqrt(2*sigma2)
    b = (high-mu)/np.sqrt(2*sigma2)
    norm = np.sqrt(sigma2*np.pi/2)*(-erf(a) + erf(b))

    if np.any(norm==0):
        badInds = np.where(norm==0)
        norm[badInds] = (np.sqrt(sigma2*np.pi/2)*(-asym(a) + asym(b)))[badInds]

    y = (1.0/norm)*np.exp((-1.0*(x-mu)*(x-mu))/(2.*sigma2))
    return y

def injection_weights(m1_det,m2_det,s1z_det,s2z_det,z_det):

    # Load mock detections
    ref_m_min = 2.
    ref_m_max = 100.
    ref_a1 = -2.35
    ref_a2 = 2.

    # Derived quantities
    q_det = m2_det/m1_det
    mtot_det = m1_det+m2_det
    X_det = (m1_det*s1z_det + m2_det*s2z_det)/(m1_det+m2_det)

    ref_p_z = np.power(1.+z_det,2.-1.)
    ref_p_m1 = np.power(m1_det,ref_a1)
    ref_p_m2 = (1.+ref_a2)*np.power(m2_det,ref_a2)/(m1_det**(1.+ref_a2) - ref_m_min**(1.+ref_a2))

    ref_p_xeff = np.zeros(X_det.size)
    for i in range(ref_p_xeff.size):

        X = X_det[i]
        q = q_det[i]
        
        if X<-(1.-q)/(1.+q):
            ref_p_xeff[i] = (1./(2.*q))*(1.+q)*(1.+X)*(1.+q)/2.
            
        elif X>(1.-q)/(1.+q):
            ref_p_xeff[i] = (1./(2.*q))*(1.+q)*(1.-X)*(1.+q)/2.
            
        else:
            ref_p_xeff[i] = (1.+q)/2.

    pop_reweight = 1./(ref_p_xeff*ref_p_m1*ref_p_m2*ref_p_z)
    pop_reweight[m1_det<mMin] = 0.
    pop_reweight[m2_det<mMin] = 0.
    return pop_reweight
