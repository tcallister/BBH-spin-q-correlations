import numpy as np
from support import *

mMin = 5.
zMax = 3.

def inPrior_brokenPowerLaw(c,priorDict):

    # Read parameters
    lmbda1 = c[0]
    lmbda2 = c[1]
    m0 = c[2]
    bq = c[3]
    kappa = c[4]
    mu0 = c[5]
    log_sigma0 = c[6]
    alpha = c[7]
    beta = c[8]

    if lmbda1<priorDict['lmbda1'][0] or lmbda1>priorDict['lmbda1'][1]:
        return False
    elif lmbda2<priorDict['lmbda2'][0] or lmbda2>priorDict['lmbda2'][1]:
        return False
    elif m0<priorDict['m0'][0] or m0>priorDict['m0'][1]:
        return False
    elif bq<priorDict['bq'][0] or bq>priorDict['bq'][1]:
        return False
    elif mu0<priorDict['mu0'][0] or mu0>priorDict['mu0'][1]:
        return False
    elif log_sigma0<priorDict['log_sigma0'][0] or log_sigma0>priorDict['log_sigma0'][1]:
        return False
    elif alpha<priorDict['alpha'][0] or alpha>priorDict['alpha'][1]:
        return False
    elif beta<priorDict['beta'][0] or beta>priorDict['beta'][1]:
        return False
    else:
        return True

def logp_brokenPowerLaw(c,sampleDict,injectionDict,priorDict):

    # Flat priors, reject samples past boundaries
    if not inPrior_brokenPowerLaw(c,priorDict):
        return -np.inf

    # If sample in prior range, evaluate
    else:

        # Read parameters
        lmbda1 = c[0]
        lmbda2 = c[1]
        m0 = c[2]
        bq = c[3]
        kappa = c[4]
        mu0 = c[5]
        log_sigma0 = c[6]
        alpha = c[7]
        beta = c[8]

        logP = 0.
        logP += -kappa**2./(2.*priorDict['sig_kappa']**2.)

        # Precompute normalization factors for p(z) and p(m1)
        p_m1_norm = (1.+lmbda1)*(1.+lmbda2)/(m0*(lmbda2-lmbda1)-mMin*np.power(mMin/m0,lmbda1)*(1.+lmbda2))

        # Unpack injections
        m1_det = injectionDict['m1']
        m2_det = injectionDict['m2']
        s1z_det = injectionDict['s1z']
        s2z_det = injectionDict['s2z']
        z_det = injectionDict['z']
        pop_reweight = injectionDict['weights']
        q_det = m2_det/m1_det
        mtot_det = m1_det+m2_det
        X_det = (m1_det*s1z_det + m2_det*s2z_det)/(m1_det+m2_det)

        # Reweight injection probabilities
        mu_q = mu0 + alpha*(q_det-0.5)
        log_sigma_q = log_sigma0 + beta*(q_det-0.5)
        p_det_Xeff = calculate_Gaussian(X_det, mu_q, 10.**(2.*log_sigma_q),-1.,1.)
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

        nEvents = len(sampleDict)
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
            p_m1 = np.ones(m1_sample.size)
            low_ms = m1_sample<m0
            high_ms = m1_sample>=m0
            p_m1[low_ms] = p_m1_norm*np.power(m1_sample[low_ms]/m0,lmbda1)
            p_m1[high_ms] = p_m1_norm*np.power(m1_sample[high_ms]/m0,lmbda2)
            old_m1_prior = np.ones(m1_sample.size)
            
            # p(m2)
            p_m2 = (1.+bq)*np.power(m2_sample,bq)/(np.power(m1_sample,1.+bq)-mMin**(1.+bq))
            old_m2_prior = np.ones(m2_sample.size)
            p_m2[m2_sample<mMin]=0

            # p(z)
            p_z = np.power(1.+z_sample,kappa-1.)
            old_pz_prior = (1.+z_sample)**(2.7-1.)
            
            # Evaluate marginalized likelihood
            nSamples = p_Chi.size
            pEvidence = np.sum(p_Chi*p_m1*p_m2*p_z*weights/Xeff_prior/old_m1_prior/old_m2_prior/old_pz_prior)/nSamples
            
            # Summation
            logP += np.log(pEvidence)

        return logP
