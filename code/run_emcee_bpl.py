import numpy as np
import glob
import emcee as mc
import h5py
import sys
from support import *
from likelihoods import *

# -- Set prior bounds --
priorDict = {
    'lmbda1':(-5,4),
    'lmbda2':(-12,-1),
    'm0':(10,100),
    'bq':(-2,10),
    'sig_kappa':6.,
    'mu0':(-1,1),
    'log_sigma0':(-1.5,0.5),
    'alpha':(-2.5,1),
    'beta':(-2,1.5)
    }

# Dicts with samples: 
sampleDict = np.load("/home/thomas.callister/RedshiftDistributions/spin-evolution/input/sampleDict_analyticPrior.pickle")

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

pop_reweight = injection_weights(m1_det,m2_det,s1z_det,s2z_det,z_det)

injectionDict = {
        'm1':m1_det,
        'm2':m2_det,
        's1z':s1z_det,
        's2z':s2z_det,
        'z':z_det,
        'weights':pop_reweight
        }

nWalkers = 32
output = "/home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/code/output/emcee_samples_BPL"

# Search for existing chains
old_chains = np.sort(glob.glob("{0}_r??.npy".format(output)))

# If no chain already exists, begin a new one
if len(old_chains)==0:

    run_version = 0

    # Initialize walkers from random positions in mu-sigma2 parameter space
    initial_lmbdas1 = np.random.random(nWalkers)*(-2.)
    initial_lmbdas2 = np.random.random(nWalkers)-7.
    initial_m0s = np.random.random(nWalkers)*10.+30
    initial_bqs = np.random.random(nWalkers)*2.
    initial_ks = np.random.normal(size=nWalkers,loc=0,scale=1)+2.
    initial_mu0s = np.random.random(nWalkers)*0.05
    initial_sigma0s = np.random.random(nWalkers)*0.5-1.
    initial_alphas = np.random.random(nWalkers)*0.05
    initial_betas = np.random.random(nWalkers)*1.
    initial_walkers = np.transpose([initial_lmbdas1,initial_lmbdas2,initial_m0s,initial_bqs,initial_ks,initial_mu0s,initial_sigma0s,initial_alphas,initial_betas])

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
dim = 9

# Run
nSteps = 10000
sampler = mc.EnsembleSampler(nWalkers,dim,logp_brokenPowerLaw,args=[sampleDict,injectionDict,priorDict],threads=2)
for i,result in enumerate(sampler.sample(initial_walkers,iterations=nSteps)):
    if i%10==0:
        np.save("{0}_r{1:02d}.npy".format(output,run_version),sampler.chain)
np.save("{0}_r{1:02d}.npy".format(output,run_version),sampler.chain)
