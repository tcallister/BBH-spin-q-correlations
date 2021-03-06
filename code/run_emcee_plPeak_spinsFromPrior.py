import numpy as np
import glob
import emcee as mc
import h5py
from scipy.stats import gaussian_kde
from scipy.special import erf,erfc
import sys
from astropy.cosmology import Planck13
import astropy.units as u

# Precompute differential comoving volumes across reference grid
def dVdz(z):
    return 4.*np.pi*Planck13.differential_comoving_volume(z).to(u.Gpc**3/u.sr).value
z_grid = np.linspace(0,3,300)
dVdz_grid = dVdz(z_grid)

mMin = 5.
zMax = 3.

# -- Set prior bounds --
lmbda_min = -5
lmbda_max = 4
mMax_min = 60.
mMax_max = 100.
m0_min = 20
m0_max = 100
sigM_min = 1.
sigM_max = 10.
fPeak_min = 0
fPeak_max = 1.
bq_min = -2.
bq_max = 10.
sig_kappa = 6.
mu0_min = -1.
mu0_max = 1.
log_sigma0_min = -1.5
log_sigma0_max = 0.5
alpha_min = -2.5
alpha_max = 1.
beta_min = -2.
beta_max = 1.5

# Dicts with samples: 
sampleDict = np.load("/home/thomas.callister/RedshiftDistributions/spin-evolution/input/sampleDict_spinsFromPrior.pickle")

"""
for key in sampleDict:
    Xeffs = sampleDict[key]['Xeff']
    sampleDict[key]['Xeff'] = np.median(Xeffs) - (Xeffs - np.median(Xeffs))
"""

# Load mock detections
ref_m_min = 2.
ref_m_max = 100.
ref_a1 = -2.35
ref_a2 = 2.

mockDetections = h5py.File('/home/thomas.callister/RedshiftDistributions/spin-evolution/input/o3a_bbhpop_inj_info.hdf','r')
ifar_1 = mockDetections['injections']['ifar_gstlal'].value
ifar_2 = mockDetections['injections']['ifar_pycbc_bbh'].value
ifar_3 = mockDetections['injections']['ifar_pycbc_full'].value
detected = (ifar_1>1) + (ifar_2>1) + (ifar_3>1)
m1_det = mockDetections['injections']['mass1_source'].value[detected]
m2_det = mockDetections['injections']['mass2_source'].value[detected]
s1z_det = mockDetections['injections']['spin1z'].value[detected]
s2z_det = mockDetections['injections']['spin2z'].value[detected]
z_det = mockDetections['injections']['redshift'].value[detected]

mockDetectionsO1O2 = h5py.File('/home/thomas.callister/RedshiftDistributions/spin-evolution/input/injections_O1O2an_spin.h5','r')
m1_det = np.append(m1_det,mockDetectionsO1O2['mass1_source'])
m2_det = np.append(m2_det,mockDetectionsO1O2['mass2_source'])
s1z_det = np.append(s1z_det,mockDetectionsO1O2['spin1z'])
s2z_det = np.append(s2z_det,mockDetectionsO1O2['spin2z'])
z_det = np.append(z_det,mockDetectionsO1O2['redshift'])

# Derived quantities
q_det = m2_det/m1_det
mtot_det = m1_det+m2_det
X_det = (m1_det*s1z_det + m2_det*s2z_det)/(m1_det+m2_det)
dVdz_det = dVdz(z_det)

ref_p_z = dVdz_det*np.power(1.+z_det,2.-1.)
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

def calculate_Gaussian(x, mu, sigma2, low, high): 
    norm = np.sqrt(sigma2*np.pi/2)*(-erf((low-mu)/np.sqrt(2*sigma2)) + erf((high-mu)/np.sqrt(2*sigma2)))
    y = (1.0/norm)*np.exp((-1.0*(x-mu)*(x-mu))/(2.*sigma2))
    y[y!=y] = 0.
    return y

# -- Log posterior function -- 
def logposterior(c):

    # Read parameters
    lmbda = c[0]
    mMax = c[1]
    m0 = c[2]
    sigM = c[3]
    fPeak = c[4]
    bq = c[5]
    kappa = c[6]
    mu0 = c[7]
    log_sigma0 = c[8]
    alpha = c[9]
    beta = c[10]

    # Flat priors, reject samples past boundaries
    if lmbda<lmbda_min or lmbda>lmbda_max or mMax<mMax_min or mMax>mMax_max or m0<m0_min or m0>m0_max or sigM<sigM_min or sigM>sigM_max or fPeak<fPeak_min or fPeak>fPeak_max or bq<bq_min or bq>bq_max or mu0<mu0_min or mu0>mu0_max or log_sigma0<log_sigma0_min or log_sigma0>log_sigma0_max or alpha<alpha_min or alpha>alpha_max or beta<beta_min or beta>beta_max:
        return -np.inf

    # If sample in prior range, evaluate
    else:

        logP = 0.
        logP += -kappa**2./(2.*sig_kappa**2.)

        nEvents = len(sampleDict)

        # Reweight injection probabilities
        mu_q = mu0 + alpha*(q_det-0.5)
        log_sigma_q = log_sigma0 + beta*(q_det-0.5)
        p_det_Xeff = calculate_Gaussian(X_det, mu_q, 10.**(2.*log_sigma_q),-1.,1.)
        p_det_m2 = (1.+bq)*np.power(m2_det,bq)/(np.power(m1_det,1.+bq)-mMin**(1.+bq))
        p_det_z = dVdz_det*np.power(1.+z_det,kappa-1.)

        # PLPeak distribution on p(m1)
        p_det_m1_pl = (1.+lmbda)*m1_det**lmbda/(mMax**(1.+lmbda) - mMin**(1.+lmbda))
        p_det_m1_pl[m1_det>mMax] = 0
        p_det_m1_peak = np.exp(-0.5*(m1_det-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
        p_det_m1 = fPeak*p_det_m1_peak + (1.-fPeak)*p_det_m1_pl

        # Construct full weighting factors
        det_weights = p_det_Xeff*p_det_m1*p_det_m2*p_det_z*pop_reweight
        det_weights[np.where(det_weights!=det_weights)] = 0.
        if np.max(det_weights)==0:
            return -np.inf
        Nsamp = np.sum(det_weights)/np.max(det_weights)
        if Nsamp<=4*nEvents:
            print("Insufficient mock detections:",c)
            return -np.inf
        log_detEff = -nEvents*np.log(np.sum(det_weights))
        logP += log_detEff

        # Loop across samples
        for event in sampleDict:

            # Grab samples
            m1_sample = sampleDict[event]['m1']
            m2_sample = sampleDict[event]['m2']
            X_sample = sampleDict[event]['Xeff']
            z_sample = sampleDict[event]['z']
            Xeff_prior = sampleDict[event]['Xeff_priors']
            weights = sampleDict[event]['weights']
            q_sample = m2_sample/m1_sample
            
            # Chi probability - Gaussian: P(chi_eff | mu, sigma2)
            mu_q = mu0 + alpha*(q_sample-0.5)
            log_sigma_q = log_sigma0 + beta*(q_sample-0.5)
            p_Chi = calculate_Gaussian(X_sample, mu_q, 10.**(2.*log_sigma_q),-1.,1.)

            # p(m1)
            p_m1_pl = (1.+lmbda)*m1_sample**lmbda/(mMax**(1.+lmbda) - mMin**(1.+lmbda))
            p_m1_pl[m1_sample>mMax] = 0.
            p_m1_peak = np.exp(-0.5*(m1_sample-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
            p_m1 = fPeak*p_m1_peak + (1.-fPeak)*p_m1_pl
            old_m1_prior = np.ones(m1_sample.size)
            
            # p(m2)
            p_m2 = (1.+bq)*np.power(m2_sample,bq)/(np.power(m1_sample,1.+bq)-mMin**(1.+bq))
            #old_m2_prior = 1./(m1_sample-mMin)
            old_m2_prior = np.ones(m2_sample.size)
            p_m2[m2_sample<mMin]=0

            # p(z)
            #p_z = np.power(1.+z_sample,kappa-1.)/p_z_norm
            p_z = np.power(1.+z_sample,kappa-1.)
            old_pz_prior = (1.+z_sample)**(2.7-1.)
            
            # Evaluate marginalized likelihood
            nSamples = p_Chi.size
            pEvidence = np.sum(p_Chi*p_m1*p_m2*p_z*weights/Xeff_prior/old_m1_prior/old_m2_prior/old_pz_prior)/nSamples
            
            # Summation
            logP += np.log(pEvidence)
            """
            if logP!=logP:
                inds = np.where(p_Chi==np.inf)
                print((sigma0*np.exp(beta*(mtot_sample/30.-1)))[inds])
                print((mu0 + alpha*(mtot_sample/30.-1))[inds])
            """
        if logP!=logP:
            print("!!!!",c,logP)
            logP = -np.inf

        return logP
    
# -- Running mcmc --     
if __name__=="__main__":

    nWalkers = 32
    output = "/home/thomas.callister/RedshiftDistributions/spin-evolution/code/spin_with_q/emcee_samples_qDependence_plPeak_spinsFromPrior"

    # Search for existing chains
    old_chains = np.sort(glob.glob("{0}_r??.npy".format(output)))

    # If no chain already exists, begin a new one
    if len(old_chains)==0:

        run_version = 0

        # Initialize walkers from random positions in mu-sigma2 parameter space
        initial_lmbdas = np.random.random(nWalkers)*(-2.)
        initial_mMaxs = np.random.random(nWalkers)*20.+80.
        initial_m0s = np.random.random(nWalkers)*10.+30
        initial_sigMs = np.random.random(nWalkers)*4+1.
        initial_fs = np.random.random(nWalkers)
        initial_bqs = np.random.random(nWalkers)*2.
        initial_ks = np.random.normal(size=nWalkers,loc=0,scale=1)+2.
        initial_mu0s = np.random.random(nWalkers)*0.05
        initial_sigma0s = np.random.random(nWalkers)*0.5-1.
        initial_alphas = np.random.random(nWalkers)*0.05
        initial_betas = np.random.random(nWalkers)*1.
        initial_walkers = np.transpose([initial_lmbdas,initial_mMaxs,initial_m0s,initial_sigMs,initial_fs,initial_bqs,initial_ks,initial_mu0s,initial_sigma0s,initial_alphas,initial_betas])

    # Otherwise resume existing chain
    else:

        # Load existing file and iterate run version
        old_chain = np.load(old_chains[-1])
        run_version = int(old_chains[-1][-6:-4])+1

        # Strip off any trailing zeros due to incomplete run
        goodInds = np.where(old_chain[0,:,0]!=0.0)[0]
        old_chain = old_chain[:,goodInds,:]

        # Initialize new walker locations to final locations from old chain
        initial_walkers = old_chain[:,-1,:]
    
    print('Initial walkers:')
    print(initial_walkers)
    
    # Dimension of parameter space
    dim = 11

    # Run
    nSteps = 10000
    sampler = mc.EnsembleSampler(nWalkers,dim,logposterior,threads=16)
    for i,result in enumerate(sampler.sample(initial_walkers,iterations=nSteps)):
        if i%10==0:
            np.save("{0}_r{1:02d}.npy".format(output,run_version),sampler.chain)
    np.save("{0}_r{1:02d}.npy".format(output,run_version),sampler.chain)
