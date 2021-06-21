import numpy as np
from scipy.special import erf,erfc

def asym(x):
    return -np.exp(-x**2)/np.sqrt(np.pi)/x*(1.-1./(2.*x**2))

def calculate_Gaussian(x, mu, sigma2, low, high): 
    a = (low-mu)/np.sqrt(2*sigma2)
    b = (high-mu)/np.sqrt(2*sigma2)
    norm = np.sqrt(sigma2*np.pi/2)*(-erf(a) + erf(b))

    # If difference in error functions produce zero, overwrite with asymptotic expansion
    if np.any(norm==0):
        badInds = np.where(norm==0)
        norm[badInds] = (np.sqrt(sigma2*np.pi/2)*(-asym(a) + asym(b)))[badInds]

    # If differences remain zero, then our domain of interest (-1,1) is so many std. deviations
    # from the mean that our parametrization is unphysical. In this case, discount this hyperparameter.
    # This amounts to an additional condition in our hyperprior
    # NaNs occur when norm is infinitesimal, like 1e-322, such that 1/norm is set to inf and the exponential term is zero
    y = (1.0/norm)*np.exp((-1.0*(x-mu)*(x-mu))/(2.*sigma2))
    if np.any(norm==0) or np.any(y!=y):
        return np.zeros(x.size)

    else:
        y[x<low] = 0
        y[x>high] = 0
        return y

def calculate_Gaussian_2D(x,y,mu_x,sigma2_x,mu_y,sigma2_y,low_x,high_x,low_y,high_y):

    xgrid = np.linspace(low_x,high_x,100)
    ygrid = np.linspace(low_y,high_y,100)
    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]

    X,Y = np.meshgrid(xgrid,ygrid)
    Z = np.exp(-0.5*(np.square(X-mu_x)/sigma2_x + np.square(Y-mu_y)/sigma2_y))
    norm = np.sum(Z)*dx*dy

    z = (1./norm)*np.exp(-0.5*(np.square(x-mu_x)/sigma2_x + np.square(y-mu_y)/sigma2_y))
    return z

def injection_weights(m1_det,m2_det,s1z_det,s2z_det,z_det,mMin=5):

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

def mock_injection_weights(m1_det,m2_det,s1z_det,s2z_det,z_det,mMin=5.):

    # Population from which injections are drawn
    ref_a1 = -2
    ref_a2 = -4
    ref_m0 = 40
    ref_mMin = 5.
    ref_mMax = 100.
    ref_bq = 0.5
    ref_kappa = 2.
    ref_mu = 0.
    ref_chi = 1.

    # Derived quantities
    q_det = m2_det/m1_det
    mtot_det = m1_det+m2_det
    X_det = (m1_det*s1z_det + m2_det*s2z_det)/(m1_det+m2_det)

    ref_p_z = np.power(1.+z_det,ref_kappa-1.)
    ref_p_m1 = np.zeros(m1_det.size)
    ref_p_m1[m1_det<ref_m0] = np.power(m1_det[m1_det<ref_m0]/ref_m0,ref_a1)
    ref_p_m1[m1_det>=ref_m0] = np.power(m1_det[m1_det>=ref_m0]/ref_m0,ref_a2)

    ref_p_m2 = np.power(m2_det,ref_bq)/(m1_det**(1.+ref_bq) - ref_mMin**(1.+ref_bq))
    ref_p_xeff = calculate_Gaussian(X_det,ref_mu,ref_chi**2,-1,1)

    pop_reweight = 1./(ref_p_xeff*ref_p_m1*ref_p_m2*ref_p_z)
    pop_reweight[m1_det<mMin] = 0.
    pop_reweight[m2_det<mMin] = 0.
    return pop_reweight
