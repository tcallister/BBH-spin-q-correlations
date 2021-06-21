import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
import json
import glob

c = 3.0e8          # m/s
H_0 = 67270.0      # m/s/MPc
Omega_M = 0.3156 # unitless
Omega_Lambda = 1.0-Omega_M

def Hz(z):
    return H_0*np.sqrt(Omega_M*(1.+z)**3.+Omega_Lambda)

def calculate_pBilby(m1,m2,z):

    # Uniform in (mc,q) to (m1,m2)
    eta = (m1*m2)/(m1+m2)**2.
    Mc =  eta**(3./5.)*(m1+m2)
    Jac = Mc/m1**2.
    pBilby = (1.+z)**2*Jac
    return pBilby

def calculate_pASTRO(z, dl):
    dc = dl/(1.+z) # comoving distance 
    dVc_dz = 4*np.pi*c*(dc**2.)/Hz(z) # comoving volume 
    pASTRO = np.power(1.+z,1.7)*dVc_dz
    return pASTRO

def Calculate_Xeff(m1, m2, a1, a2, costilt1, costilt2):
    Xeff = (m1*a1*costilt1 + m2*a2*costilt2)/(m1+m2)
    return Xeff

from scipy.special import spence as PL
def prior(q,x):
    
    x = np.reshape(np.abs(x),-1)
    
    pdfs = np.zeros(x.size)
    caseA = (x<(1.-q)/(1.+q))*(x<q/(1.+q))
    caseB = (x<(1.-q)/(1.+q))*(x>q/(1.+q))
    caseC = (x>(1.-q)/(1.+q))*(x<q/(1.+q))
    caseD = (x>(1.-q)/(1.+q))*(x>q/(1.+q))*(x<1./(1.+q))
    caseE = (x>(1.-q)/(1.+q))*(x>q/(1.+q))*(x>1./(1.+q))                           
    
    x_A = x[caseA]
    x_B = x[caseB]
    x_C = x[caseC]
    x_D = x[caseD]
    x_E = x[caseE]
    
    pdfs[caseA] = -((1.+q)/(4.*q))*(\
            2.*(1.+q)*x_A*np.arctanh((1.+1./q)*x_A)\
            + q*(-4.+np.log(q*q-(1+q)*(1+q)*x_A*x_A))\
            - (1.+q)*x_A*PL(1+q/x_A/(1.+q)+0j)\
            + (1.+q)*x_A*PL(1-q/x_A/(1.+q)+0j)\
        )
    
    pdfs[caseB] = -((1.+q)/(4.*q))*(\
            2.*(1.+q)*x_B*np.arctanh(q/(1.+q)/x_B)\
            + q*(-4. + np.log(x_B**2.*(1+q)**2.-q**2.))\
            - (1+q)*x_B*(
                PL(1+q/(1+q)/x_B+0j)
                -PL(1-q/(1+q)/x_B+0j))
        )
    
    pdfs[caseC] = ((1.+q)/(4.*q))*(\
            (1. + (1.+q)*x_C*np.log((1.+q)*x_C))*np.log(q/(1.-x_C*(1+q)))\
            + (1.+q)*(2.-2*x_C+x_C*np.log((x_C*(1+q)-q)*(x_C*(1+q)-1)/q))\
            -q*np.log(q-(1+q)*x_C)\
            + (1.+q)*x_C*PL(1./x_C/(1.+q)+0j)\
            - (1.+q)*x_C*PL(1-q/(1+q)/x_C+0j)
        )
    
    pdfs[caseD] = -((1.+q)/(4.*q))*(\
            2.*(1.+q)*(x_D-1.) \
            + (1.+q)*x_D*np.log(q) \
            + q*np.log((1.+q)*x_D-q) \
            - (1. + (1.+q)*x_D*np.log((1.+q)*x_D))*np.log(q/(1.-(1.+q)*x_D)) \
            - (1.+q)*x_D*np.log((1.+q)**2.*(x_D-x_D**2)-q) \
            + (1.+q)*x_D*PL(1.-q/(1.+q)/x_D+0j) \
            - (1.+q)*x_D*PL(1./(1.+q)/x_D+0j)
        )
                               
    pdfs[caseE] = ((1.+q)/(4.*q))*(\
            2.*(1.+q)*(1.-x_E) \
            - q*np.log((1.+q)*x_E-q) \
            + (1. + (1.+q)*x_E*np.log((1.+q)*x_E))*np.log(q/((1.+q)*x_E-1.)) \
            + (1.+q)*x_E*np.log((q+(1+q)**2.*(x_E**2-x_E))/q) \
            - (1.+q)*x_E*PL(1.-q/(1.+q)/x_E+0j) \
            + (1.+q)*x_E*PL(1./(1.+q)/x_E+0j)
        )
    
    return np.real(pdfs)

def alignedPrior(q,xs,aMax):

    # Ensure that `xs` is an array and take absolute value
    xs = np.reshape(xs,-1)

    # Set up various piecewise cases
    pdfs = np.zeros(xs.size)
    caseA = (xs>aMax*(1.-q)/(1.+q))*(xs<=aMax)
    caseB = (xs<-aMax*(1.-q)/(1.+q))*(xs>=-aMax)
    caseC = (xs>=-aMax*(1.-q)/(1.+q))*(xs<=aMax*(1.-q)/(1.+q))

    # Select relevant effective spins
    x_A = xs[caseA]
    x_B = xs[caseB]
    x_C = xs[caseC]

    q_A = q[caseA]
    q_B = q[caseB]
    q_C = q[caseC]

    pdfs[caseA] = (1.+q_A)**2.*(aMax-x_A)/(4.*q_A*aMax**2)
    pdfs[caseB] = (1.+q_B)**2.*(aMax+x_B)/(4.*q_B*aMax**2)
    pdfs[caseC] = (1.+q_C)/(2.*aMax)

    return pdfs

bilby_output_files = np.sort(glob.glob('output/job_?????_result.json'))
for f in bilby_output_files:

    key = f.split('_')[1]
    print(key)

    with open(f,'r') as jf:
        result = json.load(jf)

    mA_SF = np.array(result['posterior']['content']['mass_1_source'])
    mB_SF = np.array(result['posterior']['content']['mass_2_source'])
    DL = np.array(result['posterior']['content']['luminosity_distance'])
    z = np.array(result['posterior']['content']['redshift'])
    Xeff = np.array(result['posterior']['content']['chi_eff'])

    m1_SF = np.maximum(mA_SF,mB_SF)
    m2_SF = np.minimum(mA_SF,mB_SF)
    q = m2_SF/m1_SF
            
    # Downselect to a reasonable number of samples
    nSamps = min(5000,DL.size)
    sampleDraws = np.random.choice(np.arange(DL.size),size=nSamps,replace=False)
    m1_SF = m1_SF[sampleDraws]
    m2_SF = m2_SF[sampleDraws]
    DL = DL[sampleDraws]
    z = z[sampleDraws]
    q = q[sampleDraws]
    Xeff = Xeff[sampleDraws]
    
    Xeff_priors = alignedPrior(q,Xeff,1.)

    # Redshift and mass priors
    #pAstro = calculate_pASTRO(z,DL)
    #pAstro[pAstro<0] = 0 # if pASTRO < 0, make pASTRO = 0
    #p_bilby = calculate_pBilby(m1_SF*(1.+z),m2_SF*(1.+z),z)
    #weights = 1./p_bilby
    #weights = np.ones(Xeff.size)
    old_pz_prior = (1.+z)**(-1.)*np.power(1.+z,2.)
    weights = 1./old_pz_prior
    
    preprocDict = {'z':z,\
                    'weights':weights,\
                    'm1':m1_SF,\
                    'm2':m2_SF,\
                    'Xeff':Xeff,\
                    'Xeff_priors':Xeff_priors,\
                   }
    np.save('tmp_5k/job_{0}.npy'.format(key),preprocDict)
