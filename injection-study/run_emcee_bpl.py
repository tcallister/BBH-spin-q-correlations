import numpy as np
import glob
import emcee as mc
import h5py
import json
import sys
sys.path.append("/home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/code/")
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
    'beta':(-2,1.5),
    'mMin':5.
    }

# Dicts with samples: 
bad = []
sampleDict = {}
singleEvents = np.sort(glob.glob('/home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/injection-study/tmp/job*'))
for i,eventFile in enumerate(singleEvents):
    key = eventFile.split('_')[1].split('.')[0]
    if key in bad:
        print(key)
        continue
    else:
        dataDict = np.load(eventFile,allow_pickle=True)[()]
        sampleDict[i] = dataDict

with open('/home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/injection-study/population.json','r') as jf:
    mockDetections = json.load(jf)
m1_det = np.array(mockDetections['m1'].values())
m2_det = np.array(mockDetections['m2'].values())
s1z_det = np.array(mockDetections['a1'].values())
s2z_det = np.array(mockDetections['a2'].values())
z_det = np.array(mockDetections['z'].values())

pop_reweight = mock_injection_weights(m1_det,m2_det,s1z_det,s2z_det,z_det,mMin=priorDict['mMin'])

injectionDict = {
        'm1':m1_det,
        'm2':m2_det,
        's1z':s1z_det,
        's2z':s2z_det,
        'z':z_det,
        'weights':pop_reweight
        }

nWalkers = 32
output = "/home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/injection-study/emcee_samples_injection_bpl"

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
sampler = mc.EnsembleSampler(nWalkers,dim,logp_brokenPowerLaw,args=[sampleDict,injectionDict,priorDict],threads=16)
for i,result in enumerate(sampler.sample(initial_walkers,iterations=nSteps)):
    if i%10==0:
        np.save("{0}_r{1:02d}.npy".format(output,run_version),sampler.chain)
np.save("{0}_r{1:02d}.npy".format(output,run_version),sampler.chain)
