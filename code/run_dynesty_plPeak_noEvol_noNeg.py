import numpy as np
import glob
import emcee as mc
import h5py
import sys
import os
from support import *
from dynesty_likelihoods import *
from multiprocessing import Pool
import dynesty

job = sys.argv[1]

# -- Set prior bounds --
priorDict = {
    'lmbda':(-5,4),
    'mMax':(60,100),
    'm0':(20,100),
    'sigM':(1,10),
    'fPeak':(0,1),
    'bq':(-2,10),
    'sig_kappa':6.,
    'mu0':(0,1),
    'log_sigma0':(-1.5,0.5),
    'mMin':5.
    }

# Dicts with samples: 
sampleDict = np.load("/home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/input/sampleDict.pickle",allow_pickle=True)

mockDetections = h5py.File('/home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/input/o3a_bbhpop_inj_info.hdf','r')
ifar_1 = mockDetections['injections']['ifar_gstlal'].value
ifar_2 = mockDetections['injections']['ifar_pycbc_bbh'].value
ifar_3 = mockDetections['injections']['ifar_pycbc_full'].value
detected = (ifar_1>1) + (ifar_2>1) + (ifar_3>1)
m1_det = mockDetections['injections']['mass1_source'].value[detected]
m2_det = mockDetections['injections']['mass2_source'].value[detected]
s1z_det = mockDetections['injections']['spin1z'].value[detected]
s2z_det = mockDetections['injections']['spin2z'].value[detected]
z_det = mockDetections['injections']['redshift'].value[detected]

mockDetectionsO1O2 = h5py.File('/home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/input/injections_O1O2an_spin.h5','r')
m1_det = np.append(m1_det,mockDetectionsO1O2['mass1_source'])
m2_det = np.append(m2_det,mockDetectionsO1O2['mass2_source'])
s1z_det = np.append(s1z_det,mockDetectionsO1O2['spin1z'])
s2z_det = np.append(s2z_det,mockDetectionsO1O2['spin2z'])
z_det = np.append(z_det,mockDetectionsO1O2['redshift'])

pop_reweight = injection_weights(m1_det,m2_det,s1z_det,s2z_det,z_det,mMin=priorDict['mMin'])

injectionDict = {
        'm1':m1_det,
        'm2':m2_det,
        's1z':s1z_det,
        's2z':s2z_det,
        'z':z_det,
        'weights':pop_reweight
        }

tmpfile = "/home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/code/dynesty_output/dynesty_results_noEvol_noNeg_job{0}.resume".format(job)
if os.path.isfile(tmpfile):
    print("Resuming from checkpoint")
    sampler = np.load(tmpfile,allow_pickle=True)[()]
    sampler.rstate = np.random
    newPool = Pool(16)
    sampler.pool = newPool
    sampler.M = newPool.map
    sampler.nqueue = -1

else:
    sampler = dynesty.NestedSampler(logp_powerLawPeak_noEvol_noNeg, priorTransform_powerLawPeak_noEvol, 9,
                nlive = 8000, ptform_args=[priorDict], logl_args=[sampleDict,injectionDict,priorDict],
                sample='unif',
                pool = Pool(16), queue_size=16, bound='multi', first_update={'min_eff':10,'min_ncall':50000})

pbar, print_func = sampler._get_print_func(None, True)
for it, res in enumerate(sampler.sample(dlogz=0.1)):

    ncall = sampler.ncall
    (worst, ustar, vstar, loglstar, logvol, logwt, logz, logzvar, 
        h, nc, worst_it, boundidx, bounditer, eff,delta_logz) = res
    ncall += nc
    if delta_logz > 1e6:
        delta_logz = np.inf
    if logz <= -1e6:
        logz = -np.inf

    print_func(res,
               sampler.it - 1,
               ncall,
               dlogz=0.1,
               logl_max=np.inf)

    if it%1000==0:
        print("saving")
        np.save(tmpfile,sampler)

sampler.add_final_live()
res = sampler.results
np.save('/home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/code/dynesty_output/dynesty_results_noEvol_noNeg_job{0}'.format(job),res)


"""
sampler = dynesty.NestedSampler(logp_powerLawPeak_noEvol_noNeg, priorTransform_powerLawPeak_noEvol, 9, 
                nlive = 8000, ptform_args=[priorDict], logl_args=[sampleDict,injectionDict,priorDict],
                pool = Pool(16), queue_size=16, bound='multi', first_update={'min_eff':10,'min_ncall':5000})

sampler.run_nested(dlogz=0.1)
res = sampler.results
np.save('/home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/code/dynesty_output/dynesty_results_noEvol_noNeg_n8000',res)
"""
