import numpy as np
import json
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

def calculate_Gaussian(x, mu, sigma2, low, high): 
    norm = np.sqrt(sigma2*np.pi/2)*(-erf((low-mu)/np.sqrt(2*sigma2)) + erf((high-mu)/np.sqrt(2*sigma2)))
    y = (1.0/norm)*np.exp((-1.0*(x-mu)*(x-mu))/(2.*sigma2))
    y[y!=y] = 0.
    return y

mMin = 5.
zMax = 3.

# -- Set prior bounds --
lmbda1_min = -5
lmbda1_max = 4
lmbda2_min = -12
lmbda2_max = -1
m0_min = 10
m0_max = 100
bq_min = -2.
bq_max = 10.
sig_kappa = 6.
mu_min = -1.
mu_max = 1.
log_sigma_min = -1.5
log_sigma_max = 0.5
alpha_min = -1.
alpha_max = 1.
beta_min = -2.
beta_max = 1.5

# Dicts with samples: 
bad = ["30043","46612","23185","44306","18338","47535","03055","18548"]
sampleDict = {}
singleEvents = np.sort(glob.glob('/home/thomas.callister/RedshiftDistributions/spin-evolution/code/injections/tmp/job*'))
for i,eventFile in enumerate(singleEvents):
    key = eventFile.split('_')[1].split('.')[0]
    if key in bad:
        print(key)
        continue
    else:
        dataDict = np.load(eventFile,allow_pickle=True)[()]
        sampleDict[i] = dataDict

# Load mock detections
ref_a1 = -2
ref_a2 = -4
ref_m0 = 40
ref_mMin = 5.
ref_mMax = 100.
ref_bq = 0.5
ref_kappa = 2.
ref_mu = 0.
ref_chi = 1.

with open('/home/thomas.callister/RedshiftDistributions/spin-evolution/code/injections/population.json','r') as jf:
    mockDetections = json.load(jf)
m1_det = np.array(mockDetections['m1'].values())
m2_det = np.array(mockDetections['m2'].values())
s1z_det = np.array(mockDetections['a1'].values())
s2z_det = np.array(mockDetections['a2'].values())
z_det = np.array(mockDetections['z'].values())

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


# -- Log posterior function -- 
def logposterior(c):

    # Read parameters
    lmbda1 = c[0]
    lmbda2 = c[1]
    m0 = c[2]
    bq = c[3]
    kappa = c[4]
    mu = c[5]
    log_sigma = c[6]
    #alpha = c[7]
    #beta = c[8]

    # Flat priors, reject samples past boundaries
    if lmbda1<lmbda1_min or lmbda1>lmbda1_max or lmbda2<lmbda2_min or lmbda2>lmbda2_max or m0<m0_min or m0>m0_max or bq<bq_min or bq>bq_max or mu<mu_min or mu>mu_max or log_sigma<log_sigma_min or log_sigma>log_sigma_max:
        return -np.inf

    #or alpha<alpha_min or alpha>alpha_max or beta<beta_min or beta>beta_max:

    # If sample in prior range, evaluate
    else:

        logP = 0.
        logP += -kappa**2./(2.*sig_kappa**2.)

        nEvents = len(sampleDict)

        # Precompute normalization factors for p(z) and p(m1)
        p_m1_norm = (1.+lmbda1)*(1.+lmbda2)/(m0*(lmbda2-lmbda1)-mMin*np.power(mMin/m0,lmbda1)*(1.+lmbda2))

        # Reweight injection probabilities
        #mu_q = mu0 + alpha*(q_det-0.5)
        #log_sigma_q = log_sigma0 + beta*(q_det-0.5)
        p_det_Xeff = calculate_Gaussian(X_det, mu, 10.**(2.*log_sigma),-1.,1.)
        p_det_m2 = (1.+bq)*np.power(m2_det,bq)/(np.power(m1_det,1.+bq)-mMin**(1.+bq))
        p_det_z = np.power(1.+z_det,kappa-1.)

        # Broken power-law distribution on p(m1)
        p_det_m1 = np.ones(m1_det.size)
        low_m_dets = m1_det<m0
        high_m_dets = m1_det>=m0
        p_det_m1[low_m_dets] = p_m1_norm*np.power(m1_det[low_m_dets]/m0,lmbda1)
        p_det_m1[high_m_dets] = p_m1_norm*np.power(m1_det[high_m_dets]/m0,lmbda2)

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
            #mu_q = mu0 + alpha*(q_sample-0.5)
            #log_sigma_q = log_sigma0 + beta*(q_sample-0.5)
            p_Chi = calculate_Gaussian(X_sample, mu, 10.**(2.*log_sigma),-1.,1.)

            # p(m1)
            p_m1 = np.ones(m1_sample.size)
            low_ms = m1_sample<m0
            high_ms = m1_sample>=m0
            p_m1[low_ms] = p_m1_norm*np.power(m1_sample[low_ms]/m0,lmbda1)
            p_m1[high_ms] = p_m1_norm*np.power(m1_sample[high_ms]/m0,lmbda2)
            
            # p(m2)
            p_m2 = (1.+bq)*np.power(m2_sample,bq)/(np.power(m1_sample,1.+bq)-mMin**(1.+bq))
            p_m2[m2_sample<mMin]=0

            # p(z)
            # Extra (1+z)**2 comes from uniform priors on *detector* frame masses
            #old_pz_prior = (1.+z_sample)**(-1.)*np.power(1.+z_sample,2.)
            old_pz_prior = (1.+z_sample)**(-1.)*np.power(1.+z_sample,2.)
            p_z = np.power(1.+z_sample,kappa-1.)
            
            # Evaluate marginalized likelihood
            nSamples = p_Chi.size
            pEvidence = np.sum(p_Chi*p_m1*p_m2*p_z*weights/Xeff_prior/old_pz_prior)/nSamples
            
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
    output = "/home/thomas.callister/RedshiftDistributions/spin-evolution/code/injections/emcee_samples_injection_qDependence"

    # Search for existing chains
    old_chains = np.sort(glob.glob("{0}_r??.npy".format(output)))

    # If no chain already exists, begin a new one
    if len(old_chains)==0:

        run_version = 0

        # Initialize walkers from random positions in mu-sigma2 parameter space
        initial_lmbdas1 = np.random.random(nWalkers)*(-2.)-1.
        initial_lmbdas2 = np.random.random(nWalkers)*(-2.)-5.
        initial_m0s = np.random.random(nWalkers)*10.+35
        initial_bqs = np.random.random(nWalkers)*1.-0.5
        initial_ks = np.random.normal(size=nWalkers,loc=0,scale=1)+2.
        initial_mu0s = np.random.random(nWalkers)*0.1
        initial_sigma0s = np.random.random(nWalkers)*0.2-0.5
        #initial_alphas = np.random.random(nWalkers)*0.5
        #initial_betas = np.random.random(nWalkers)*0.1
        initial_walkers = np.transpose([initial_lmbdas1,initial_lmbdas2,initial_m0s,initial_bqs,initial_ks,initial_mu0s,initial_sigma0s])
        #initial_alphas,initial_betas])

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
    dim = 7

    # Run
    nSteps = 10000
    sampler = mc.EnsembleSampler(nWalkers,dim,logposterior,threads=16)
    for i,result in enumerate(sampler.sample(initial_walkers,iterations=nSteps)):
        if i%10==0:
            np.save("{0}_r{1:02d}.npy".format(output,run_version),sampler.chain)
    np.save("{0}_r{1:02d}.npy".format(output,run_version),sampler.chain)
