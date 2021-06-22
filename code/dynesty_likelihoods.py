import numpy as np
from scipy.special import erfinv
from support import *

def priorTransform_powerLawPeak(c,priorDict):

    # Read parameters
    lmbda = c[0]*(priorDict['lmbda'][1]-priorDict['lmbda'][0]) + priorDict['lmbda'][0]
    mMax = c[1]*(priorDict['mMax'][1]-priorDict['mMax'][0]) + priorDict['mMax'][0]
    m0 = c[2]*(priorDict['m0'][1]-priorDict['m0'][0]) + priorDict['m0'][0]
    sigM = c[3]*(priorDict['sigM'][1]-priorDict['sigM'][0]) + priorDict['sigM'][0]
    fPeak = c[4]*(priorDict['fPeak'][1]-priorDict['fPeak'][0]) + priorDict['fPeak'][0]
    bq = c[5]*(priorDict['bq'][1]-priorDict['bq'][0]) + priorDict['bq'][0]
    kappa = np.sqrt(2.)*priorDict['sig_kappa']*erfinv(-1.+2.*c[6])
    mu0 = c[7]*(priorDict['mu0'][1]-priorDict['mu0'][0]) + priorDict['mu0'][0]
    log_sigma0 = c[8]*(priorDict['log_sigma0'][1]-priorDict['log_sigma0'][0]) + priorDict['log_sigma0'][0]
    alpha = c[9]*(priorDict['alpha'][1]-priorDict['alpha'][0]) + priorDict['alpha'][0]
    beta = c[10]*(priorDict['beta'][1]-priorDict['beta'][0]) + priorDict['beta'][0]

    return([lmbda,mMax,m0,sigM,fPeak,bq,kappa,mu0,log_sigma0,alpha,beta])

def logp_powerLawPeak(c,sampleDict,injectionDict,priorDict):

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

    logP = 0.

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

    mMin = priorDict['mMin']

    # Reweight injection probabilities
    mu_q = mu0 + alpha*(q_det-0.5)
    log_sigma_q = log_sigma0 + beta*(q_det-0.5)
    p_det_Xeff = calculate_Gaussian(X_det, mu_q, 10.**(2.*log_sigma_q),-1.,1.)
    p_det_m2 = (1.+bq)*np.power(m2_det,bq)/(np.power(m1_det,1.+bq)-mMin**(1.+bq))
    p_det_z = np.power(1.+z_det,kappa-1.)

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

    nEvents = len(sampleDict)
    Nsamp = np.sum(det_weights)/np.max(det_weights)
    if Nsamp<=4*nEvents:
        #print("Insufficient mock detections:",c)
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

def logp_powerLawPeak_noNeg(c,sampleDict,injectionDict,priorDict):

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

    logP = 0.

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

    mMin = priorDict['mMin']

    # Reweight injection probabilities
    mu_q = mu0 + alpha*(q_det-0.5)
    log_sigma_q = log_sigma0 + beta*(q_det-0.5)
    p_det_Xeff = calculate_Gaussian(X_det, mu_q, 10.**(2.*log_sigma_q),0.,1.)
    p_det_m2 = (1.+bq)*np.power(m2_det,bq)/(np.power(m1_det,1.+bq)-mMin**(1.+bq))
    p_det_z = np.power(1.+z_det,kappa-1.)

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

    nEvents = len(sampleDict)
    Nsamp = np.sum(det_weights)/np.max(det_weights)
    if Nsamp<=4*nEvents:
        #print("Insufficient mock detections:",c)
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
        p_Chi = calculate_Gaussian(X_sample, mu_q, 10.**(2.*log_sigma_q),0.,1.)

        # p(m1)
        p_m1_pl = (1.+lmbda)*m1_sample**lmbda/(mMax**(1.+lmbda) - mMin**(1.+lmbda))
        p_m1_pl[m1_sample>mMax] = 0.
        p_m1_peak = np.exp(-0.5*(m1_sample-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
        p_m1 = fPeak*p_m1_peak + (1.-fPeak)*p_m1_pl
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

def priorTransform_powerLawPeak_noEvol(c,priorDict):

    # Read parameters
    lmbda = c[0]*(priorDict['lmbda'][1]-priorDict['lmbda'][0]) + priorDict['lmbda'][0]
    mMax = c[1]*(priorDict['mMax'][1]-priorDict['mMax'][0]) + priorDict['mMax'][0]
    m0 = c[2]*(priorDict['m0'][1]-priorDict['m0'][0]) + priorDict['m0'][0]
    sigM = c[3]*(priorDict['sigM'][1]-priorDict['sigM'][0]) + priorDict['sigM'][0]
    fPeak = c[4]*(priorDict['fPeak'][1]-priorDict['fPeak'][0]) + priorDict['fPeak'][0]
    bq = c[5]*(priorDict['bq'][1]-priorDict['bq'][0]) + priorDict['bq'][0]
    kappa = np.sqrt(2.)*priorDict['sig_kappa']*erfinv(-1.+2.*c[6])
    mu0 = c[7]*(priorDict['mu0'][1]-priorDict['mu0'][0]) + priorDict['mu0'][0]
    log_sigma0 = c[8]*(priorDict['log_sigma0'][1]-priorDict['log_sigma0'][0]) + priorDict['log_sigma0'][0]

    return([lmbda,mMax,m0,sigM,fPeak,bq,kappa,mu0,log_sigma0])

def logp_powerLawPeak_noEvol(c,sampleDict,injectionDict,priorDict):

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

    logP = 0.

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

    mMin = priorDict['mMin']

    # Reweight injection probabilities
    p_det_Xeff = calculate_Gaussian(X_det, mu0, 10.**(2.*log_sigma0),-1.,1.)
    p_det_m2 = (1.+bq)*np.power(m2_det,bq)/(np.power(m1_det,1.+bq)-mMin**(1.+bq))
    p_det_z = np.power(1.+z_det,kappa-1.)

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

    nEvents = len(sampleDict)
    Nsamp = np.sum(det_weights)/np.max(det_weights)
    if Nsamp<=4*nEvents:
        #print("Insufficient mock detections:",c)
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
        p_Chi = calculate_Gaussian(X_sample, mu0, 10.**(2.*log_sigma0),-1.,1.)

        # p(m1)
        p_m1_pl = (1.+lmbda)*m1_sample**lmbda/(mMax**(1.+lmbda) - mMin**(1.+lmbda))
        p_m1_pl[m1_sample>mMax] = 0.
        p_m1_peak = np.exp(-0.5*(m1_sample-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
        p_m1 = fPeak*p_m1_peak + (1.-fPeak)*p_m1_pl
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

def logp_powerLawPeak_noEvol_noNeg(c,sampleDict,injectionDict,priorDict):

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

    logP = 0.

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

    mMin = priorDict['mMin']

    # Reweight injection probabilities
    p_det_Xeff = calculate_Gaussian(X_det, mu0, 10.**(2.*log_sigma0),0.,1.)
    p_det_m2 = (1.+bq)*np.power(m2_det,bq)/(np.power(m1_det,1.+bq)-mMin**(1.+bq))
    p_det_z = np.power(1.+z_det,kappa-1.)

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

    nEvents = len(sampleDict)
    Nsamp = np.sum(det_weights)/np.max(det_weights)
    if Nsamp<=4*nEvents:
        #print("Insufficient mock detections:",c)
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
        p_Chi = calculate_Gaussian(X_sample, mu0, 10.**(2.*log_sigma0),0.,1.)

        # p(m1)
        p_m1_pl = (1.+lmbda)*m1_sample**lmbda/(mMax**(1.+lmbda) - mMin**(1.+lmbda))
        p_m1_pl[m1_sample>mMax] = 0.
        p_m1_peak = np.exp(-0.5*(m1_sample-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
        p_m1 = fPeak*p_m1_peak + (1.-fPeak)*p_m1_pl
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

