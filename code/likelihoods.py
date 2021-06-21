import numpy as np
from support import *

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
        mMin = priorDict['mMin']
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

def inPrior_powerLawPeak(c,priorDict):

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

    if lmbda<priorDict['lmbda'][0] or lmbda>priorDict['lmbda'][1]:
        return False
    elif mMax<priorDict['mMax'][0] or mMax>priorDict['mMax'][1]:
        return False
    elif m0<priorDict['m0'][0] or m0>priorDict['m0'][1]:
        return False
    elif sigM<priorDict['sigM'][0] or sigM>priorDict['sigM'][1]:
        return False
    elif fPeak<priorDict['fPeak'][0] or fPeak>priorDict['fPeak'][1]:
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

def logp_powerLawPeak(c,sampleDict,injectionDict,priorDict):

    # Flat priors, reject samples past boundaries
    if not inPrior_powerLawPeak(c,priorDict):
        return -np.inf

    # If sample in prior range, evaluate
    else:

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
        logP += -kappa**2./(2.*priorDict['sig_kappa']**2.)

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

        if logP!=logP:
            print("!!!!!!!",c)

        return logP

def inPrior_powerLawPeak_gaussianQ(c,priorDict):

    # Read parameters
    lmbda = c[0]
    mMax = c[1]
    m0 = c[2]
    sigM = c[3]
    fPeak = c[4]
    mu_q = c[5]
    sig_q = c[6]
    kappa = c[7]
    mu0 = c[8]
    log_sigma0 = c[9]
    alpha = c[10]
    beta = c[11]

    if lmbda<priorDict['lmbda'][0] or lmbda>priorDict['lmbda'][1]:
        return False
    elif mMax<priorDict['mMax'][0] or mMax>priorDict['mMax'][1]:
        return False
    elif m0<priorDict['m0'][0] or m0>priorDict['m0'][1]:
        return False
    elif sigM<priorDict['sigM'][0] or sigM>priorDict['sigM'][1]:
        return False
    elif fPeak<priorDict['fPeak'][0] or fPeak>priorDict['fPeak'][1]:
        return False
    elif mu_q<priorDict['mu_q'][0] or mu_q>priorDict['mu_q'][1]:
        return False
    elif sig_q<priorDict['sig_q'][0] or sig_q>priorDict['sig_q'][1]:
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

def logp_powerLawPeak_gaussianQ(c,sampleDict,injectionDict,priorDict):

    # Flat priors, reject samples past boundaries
    if not inPrior_powerLawPeak_gaussianQ(c,priorDict):
        return -np.inf

    # If sample in prior range, evaluate
    else:

        # Read parameters
        lmbda = c[0]
        mMax = c[1]
        m0 = c[2]
        sigM = c[3]
        fPeak = c[4]
        mu_q = c[5]
        sig_q = c[6]
        kappa = c[7]
        mu0 = c[8]
        log_sigma0 = c[9]
        alpha = c[10]
        beta = c[11]

        logP = 0.
        logP += -kappa**2./(2.*priorDict['sig_kappa']**2.)

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
        mu_chi = mu0 + alpha*(q_det-0.5)
        log_sigma_chi = log_sigma0 + beta*(q_det-0.5)
        p_det_Xeff = calculate_Gaussian(X_det, mu_chi, 10.**(2.*log_sigma_chi),-1.,1.)
        p_det_m2 = calculate_Gaussian(q_det,mu_q,sig_q,(mMin/m1_det),1.)/m1_det
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
            mu_chi = mu0 + alpha*(q_sample-0.5)
            log_sigma_chi = log_sigma0 + beta*(q_sample-0.5)
            p_Chi = calculate_Gaussian(X_sample, mu_chi, 10.**(2.*log_sigma_chi),-1.,1.)

            # p(m1)
            p_m1_pl = (1.+lmbda)*m1_sample**lmbda/(mMax**(1.+lmbda) - mMin**(1.+lmbda))
            p_m1_pl[m1_sample>mMax] = 0.
            p_m1_peak = np.exp(-0.5*(m1_sample-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
            p_m1 = fPeak*p_m1_peak + (1.-fPeak)*p_m1_pl
            old_m1_prior = np.ones(m1_sample.size)
            
            # p(m2)
            #p_m2 = (1.+bq)*np.power(m2_sample,bq)/(np.power(m1_sample,1.+bq)-mMin**(1.+bq))
            p_m2 = calculate_Gaussian(q_sample,mu_q,sig_q,(mMin/m1_sample),1.)/m1_sample
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

        if logP!=logP:
            print("!!!!!!!",c)

        return logP

def inPrior_powerLawPeak_bivariateGaussian(c,priorDict):

    # Read parameters
    lmbda = c[0]
    mMax = c[1]
    m0 = c[2]
    sigM = c[3]
    fPeak = c[4]
    f_lowSpin = c[5]
    mu_q_lowSpin = c[6]
    logsig_q_lowSpin = c[7]
    mu_q_highSpin = c[8]
    logsig_q_highSpin = c[9]
    kappa = c[10]
    mu_chi_lowSpin = c[11]
    logsig_chi_lowSpin = c[12]
    mu_chi_highSpin = c[13]
    logsig_chi_highSpin = c[14]

    if lmbda<priorDict['lmbda'][0] or lmbda>priorDict['lmbda'][1]:
        return False
    elif mMax<priorDict['mMax'][0] or mMax>priorDict['mMax'][1]:
        return False
    elif m0<priorDict['m0'][0] or m0>priorDict['m0'][1]:
        return False
    elif sigM<priorDict['sigM'][0] or sigM>priorDict['sigM'][1]:
        return False
    elif fPeak<priorDict['fPeak'][0] or fPeak>priorDict['fPeak'][1]:
        return False
    elif f_lowSpin<priorDict['f_lowSpin'][0] or f_lowSpin>priorDict['f_lowSpin'][1]:
        return False
    elif mu_q_lowSpin<priorDict['mu_q_lowSpin'][0] or mu_q_lowSpin>priorDict['mu_q_lowSpin'][1]:
        return False
    elif logsig_q_lowSpin<priorDict['logsig_q_lowSpin'][0] or logsig_q_lowSpin>priorDict['logsig_q_lowSpin'][1]:
        return False
    elif mu_q_highSpin<priorDict['mu_q_highSpin'][0] or mu_q_highSpin>priorDict['mu_q_highSpin'][1]:
        return False
    elif logsig_q_highSpin<priorDict['logsig_q_highSpin'][0] or logsig_q_highSpin>priorDict['logsig_q_highSpin'][1]:
        return False
    elif mu_chi_lowSpin<priorDict['mu_chi_lowSpin'][0] or mu_chi_lowSpin>priorDict['mu_chi_lowSpin'][1]:
        return False
    elif logsig_chi_lowSpin<priorDict['logsig_chi_lowSpin'][0] or logsig_chi_lowSpin>priorDict['logsig_chi_lowSpin'][1]:
        return False
    elif mu_chi_highSpin<priorDict['mu_chi_highSpin'][0] or mu_chi_highSpin>priorDict['mu_chi_highSpin'][1] or mu_chi_highSpin<mu_chi_lowSpin:
        return False
    elif logsig_chi_highSpin<priorDict['logsig_chi_highSpin'][0] or logsig_chi_highSpin>priorDict['logsig_chi_highSpin'][1]:
        return False
    else:
        return True

def logp_powerLawPeak_bivariateGaussian(c,sampleDict,injectionDict,priorDict):

    # Flat priors, reject samples past boundaries
    if not inPrior_powerLawPeak_bivariateGaussian(c,priorDict):
        return -np.inf

    # If sample in prior range, evaluate
    else:

        # Read parameters
        lmbda = c[0]
        mMax = c[1]
        m0 = c[2]
        sigM = c[3]
        fPeak = c[4]
        f_lowSpin = c[5]
        mu_q_lowSpin = c[6]
        logsig_q_lowSpin = c[7]
        mu_q_highSpin = c[8]
        logsig_q_highSpin = c[9]
        kappa = c[10]
        mu_chi_lowSpin = c[11]
        logsig_chi_lowSpin = c[12]
        mu_chi_highSpin = c[13]
        logsig_chi_highSpin = c[14]

        logP = 0.
        logP += -kappa**2./(2.*priorDict['sig_kappa']**2.)

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
        p_det_Xeff_m2_lowSpin = calculate_Gaussian_2D(X_det,q_det,mu_chi_lowSpin,10.**(2.*logsig_chi_lowSpin),\
                mu_q_lowSpin,10.**(2.*logsig_q_lowSpin),-1.,1.,0.,1.)/m1_det
        p_det_Xeff_m2_highSpin = calculate_Gaussian_2D(X_det,q_det,mu_chi_highSpin,10.**(2.*logsig_chi_highSpin),\
                mu_q_highSpin,10.**(2.*logsig_q_highSpin),-1.,1.,0.,1.)/m1_det
        p_det_Xeff_m2 = f_lowSpin*p_det_Xeff_m2_lowSpin + (1.-f_lowSpin)*p_det_Xeff_m2_highSpin
        p_det_Xeff_m2[m2_det<mMin] = 0
        p_det_z = np.power(1.+z_det,kappa-1.)

        # PLPeak distribution on p(m1)
        p_det_m1_pl = (1.+lmbda)*m1_det**lmbda/(mMax**(1.+lmbda) - mMin**(1.+lmbda))
        p_det_m1_pl[m1_det>mMax] = 0
        p_det_m1_peak = np.exp(-0.5*(m1_det-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
        p_det_m1 = fPeak*p_det_m1_peak + (1.-fPeak)*p_det_m1_pl

        # Construct full weighting factors
        det_weights = p_det_Xeff_m2*p_det_m1*p_det_z*pop_reweight
        det_weights[np.where(det_weights!=det_weights)] = 0.
        if np.max(det_weights)==0:
            return -np.inf

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
            p_Xeff_m2_lowSpin = calculate_Gaussian_2D(X_sample,q_sample,mu_chi_lowSpin,10.**(2.*logsig_chi_lowSpin),\
                    mu_q_lowSpin,10.**(2.*logsig_q_lowSpin),-1.,1.,0.,1.)/m1_sample
            p_Xeff_m2_highSpin = calculate_Gaussian_2D(X_sample,q_sample,mu_chi_highSpin,10.**(2.*logsig_chi_highSpin),\
                    mu_q_highSpin,10.**(2.*logsig_q_highSpin),-1.,1.,0.,1.)/m1_sample
            p_Xeff_m2 = f_lowSpin*p_Xeff_m2_lowSpin + (1.-f_lowSpin)*p_Xeff_m2_highSpin
            p_Xeff_m2[m2_sample<mMin] = 0.

            # p(m1)
            p_m1_pl = (1.+lmbda)*m1_sample**lmbda/(mMax**(1.+lmbda) - mMin**(1.+lmbda))
            p_m1_pl[m1_sample>mMax] = 0.
            p_m1_peak = np.exp(-0.5*(m1_sample-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
            p_m1 = fPeak*p_m1_peak + (1.-fPeak)*p_m1_pl
            old_m1_prior = np.ones(m1_sample.size)
            old_m2_prior = np.ones(m2_sample.size)
            
            # p(z)
            p_z = np.power(1.+z_sample,kappa-1.)
            old_pz_prior = (1.+z_sample)**(2.7-1.)
            
            # Evaluate marginalized likelihood
            nSamples = p_z.size
            pEvidence = np.sum(p_Xeff_m2*p_m1*p_z*weights/Xeff_prior/old_m1_prior/old_m2_prior/old_pz_prior)/nSamples

            # Summation
            logP += np.log(pEvidence)

        if logP!=logP:
            print("!!!!!!!",c)

        print(f_lowSpin,logP)
        return logP

def inPrior_powerLawPeak_variableChiMin(c,priorDict):

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
    chi_min = c[11]

    if lmbda<priorDict['lmbda'][0] or lmbda>priorDict['lmbda'][1]:
        return False
    elif mMax<priorDict['mMax'][0] or mMax>priorDict['mMax'][1]:
        return False
    elif m0<priorDict['m0'][0] or m0>priorDict['m0'][1]:
        return False
    elif sigM<priorDict['sigM'][0] or sigM>priorDict['sigM'][1]:
        return False
    elif fPeak<priorDict['fPeak'][0] or fPeak>priorDict['fPeak'][1]:
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
    elif chi_min<priorDict['chi_min'][0] or chi_min>priorDict['chi_min'][1]:
        return False
    else:
        return True

def logp_powerLawPeak_variableChiMin(c,sampleDict,injectionDict,priorDict):

    # Flat priors, reject samples past boundaries
    if not inPrior_powerLawPeak_variableChiMin(c,priorDict):
        return -np.inf

    # If sample in prior range, evaluate
    else:

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
        chi_min = c[11]

        logP = 0.
        logP += -kappa**2./(2.*priorDict['sig_kappa']**2.)

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
        #min_bounds = np.zeros(X_det.size)
        #min_bounds[q_det>=0.75] = chi_min
        p_det_Xeff = calculate_Gaussian(X_det, mu_q, 10.**(2.*log_sigma_q),chi_min,1.)
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
            #min_bounds = np.zeros(X_sample.size)
            #min_bounds[q_sample>=0.75] = chi_min
            p_Chi = calculate_Gaussian(X_sample, mu_q, 10.**(2.*log_sigma_q),chi_min,1.)

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

        if logP!=logP:
            print("!!!!!!!",c)

        return logP

def inPrior_powerLawPeak_noEvol(c,priorDict):

    # Read parameters
    lmbda = c[0]
    mMax = c[1]
    m0 = c[2]
    sigM = c[3]
    fPeak = c[4]
    bq = c[5]
    kappa = c[6]
    mu = c[7]
    log_sigma = c[8]

    if lmbda<priorDict['lmbda'][0] or lmbda>priorDict['lmbda'][1]:
        return False
    elif mMax<priorDict['mMax'][0] or mMax>priorDict['mMax'][1]:
        return False
    elif m0<priorDict['m0'][0] or m0>priorDict['m0'][1]:
        return False
    elif sigM<priorDict['sigM'][0] or sigM>priorDict['sigM'][1]:
        return False
    elif fPeak<priorDict['fPeak'][0] or fPeak>priorDict['fPeak'][1]:
        return False
    elif bq<priorDict['bq'][0] or bq>priorDict['bq'][1]:
        return False
    elif mu<priorDict['mu'][0] or mu>priorDict['mu'][1]:
        return False
    elif log_sigma<priorDict['log_sigma'][0] or log_sigma>priorDict['log_sigma'][1]:
        return False
    else:
        return True

def logp_powerLawPeak_noEvol(c,sampleDict,injectionDict,priorDict):

    # Flat priors, reject samples past boundaries
    if not inPrior_powerLawPeak_noEvol(c,priorDict):
        return -np.inf

    # If sample in prior range, evaluate
    else:

        # Read parameters
        lmbda = c[0]
        mMax = c[1]
        m0 = c[2]
        sigM = c[3]
        fPeak = c[4]
        bq = c[5]
        kappa = c[6]
        mu = c[7]
        log_sigma = c[8]

        logP = 0.
        logP += -kappa**2./(2.*priorDict['sig_kappa']**2.)

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
        p_det_Xeff = calculate_Gaussian(X_det, mu, 10.**(2.*log_sigma),-1.,1.)
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
            p_Chi = calculate_Gaussian(X_sample, mu, 10.**(2.*log_sigma),-1.,1.)

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

        if logP!=logP:
            print("!!!!!!!",c)

        return logP

def inPrior_powerLawPeak_noEvol_variableChiMin(c,priorDict):

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
    chi_min = c[9]

    if lmbda<priorDict['lmbda'][0] or lmbda>priorDict['lmbda'][1]:
        return False
    elif mMax<priorDict['mMax'][0] or mMax>priorDict['mMax'][1]:
        return False
    elif m0<priorDict['m0'][0] or m0>priorDict['m0'][1]:
        return False
    elif sigM<priorDict['sigM'][0] or sigM>priorDict['sigM'][1]:
        return False
    elif fPeak<priorDict['fPeak'][0] or fPeak>priorDict['fPeak'][1]:
        return False
    elif bq<priorDict['bq'][0] or bq>priorDict['bq'][1]:
        return False
    elif mu0<priorDict['mu0'][0] or mu0>priorDict['mu0'][1]:
        return False
    elif log_sigma0<priorDict['log_sigma0'][0] or log_sigma0>priorDict['log_sigma0'][1]:
        return False
    elif chi_min<priorDict['chi_min'][0] or chi_min>priorDict['chi_min'][1]:
        return False
    else:
        return True

def logp_powerLawPeak_noEvol_variableChiMin(c,sampleDict,injectionDict,priorDict):

    # Flat priors, reject samples past boundaries
    if not inPrior_powerLawPeak_noEvol_variableChiMin(c,priorDict):
        return -np.inf

    # If sample in prior range, evaluate
    else:

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
        chi_min = c[9]

        logP = 0.
        logP += -kappa**2./(2.*priorDict['sig_kappa']**2.)

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
        p_det_Xeff = calculate_Gaussian(X_det, mu0, 10.**(2.*log_sigma0),chi_min,1.)
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
            p_Chi = calculate_Gaussian(X_sample, mu0, 10.**(2.*log_sigma0),chi_min,1.)

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

        if logP!=logP:
            print("!!!!!!!",c)

        return logP
