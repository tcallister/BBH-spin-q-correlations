import numpy as np
from pycbc.filter import matchedfilter
from pycbc.waveform import get_fd_waveform
from pycbc.detector import Detector
from pycbc.psd import analytical
from pycbc.psd import read
from astropy.cosmology import Planck13,z_at_value
from scipy.special import erf,erfinv
import astropy.units as u
import sys
import pandas as pd

def dVdz(z):
    return 4.*np.pi*Planck13.differential_comoving_volume(z).to(u.Gpc**3/u.sr).value

# Prepare detector object
H1 = Detector("H1")
L1 = Detector("L1")
V1 = Detector("V1")
psd_H1 = read.from_txt('./aligo_O3actual_H1.txt',4096,1,10,is_asd_file=True)
psd_L1 = read.from_txt('./aligo_O3actual_L1.txt',4096,1,10,is_asd_file=True)
psd_V1 = read.from_txt('./avirgo_O3actual.txt',4096,1,10,is_asd_file=True)

# Arrays to hold injection values
saved_m1s = np.array([])
saved_m2s = np.array([])
saved_s1zs = np.array([])
saved_s2zs = np.array([])
saved_zs = np.array([])
saved_DLs = np.array([])
saved_incs = np.array([])
saved_ras = np.array([])
saved_decs = np.array([])
saved_snrs = np.array([])

# Choose hyperparameters governing true distribution
a1 = -2
a2 = -4
m0 = 35
mMin = 5
mMax = 100
bq = 0.5
kappa = 2.
mu_chi = 0.
sig_chi = 1.

# Precompute normalization factors for m1 distribution
p_m1_norm = (1+a1)*(1+a2)/((a2-a1)*m0 + (1.+a1)*mMax*np.power(mMax/m0,a2) - (1.+a2)*mMin*np.power(mMin/m0,a1))
c0 = p_m1_norm*(m0-mMin*np.power(mMin/m0,a1))/(1.+a1)

# Also prepare interpolation grid for redshifts
z_grid = np.linspace(0.,2,1000)
dVdz_grid = dVdz(z_grid)
p_z_grid = dVdz_grid*(1.+z_grid)**(kappa-1.)
p_z_norm = np.trapz(p_z_grid,z_grid)
p_z_grid /= p_z_norm
cdf_grid = np.cumsum(p_z_grid)*(z_grid[1]-z_grid[0])

horizon_component_masses = np.logspace(0.5,2.5,10)
horizon_zs = np.zeros(horizon_component_masses.size)
for i,m in enumerate(horizon_component_masses):

    # Select initial trial distance
    DL = 50.

    trial_snr = np.inf
    while trial_snr>8.:
        DL = DL*1.5
        hp, hc = get_fd_waveform(approximant="IMRPhenomD", mass1=m, mass2=m,
                                    spin1z=0.95, spin2z=0.95,
                                    inclination=0., distance=DL,
                                    f_lower=15, delta_f=1., f_final=4096.)
        sqSNR1 = matchedfilter.overlap(hp,hp,psd=psd_H1,low_frequency_cutoff=15.,normalized=False)
        sqSNR2 = matchedfilter.overlap(hp,hp,psd=psd_L1,low_frequency_cutoff=15.,normalized=False)
        sqSNR3 = matchedfilter.overlap(hp,hp,psd=psd_V1,low_frequency_cutoff=15.,normalized=False)
        trial_snr = np.sqrt(sqSNR1+sqSNR2+sqSNR3)
        print(m,trial_snr,DL,z_at_value(Planck13.luminosity_distance,DL*u.Mpc))

    horizon_zs[i] = z_at_value(Planck13.luminosity_distance,DL*u.Mpc)
    print(m,horizon_zs[i])

# Loop
n_det = 0
n_trials = 0
n_hopeless = 0
while n_det<50000:

    n_trials += 1

    # Random primary mass
    c = np.random.random()
    if c<c0:
        m1 = np.power(mMin**(1.+a1) + (1.+a1)*m0**a1*c/p_m1_norm,1./(1+a1))
    else:
        m1 = np.power(m0**(1.+a2) + (1.+a2)*m0**a2*(c-c0)/p_m1_norm,1./(1+a2))

    # Random m2
    c_m2 = np.random.random()
    m2 = np.power(mMin**(1.+bq) + c_m2*(m1**(1.+bq)-mMin**(1.+bq)),1./(1.+bq))

    # Random redshift
    cz = np.random.random()
    z = np.interp(cz,cdf_grid,z_grid)
    DL = Planck13.luminosity_distance(z).to(u.Mpc).value

    # Compare against precomputed horizons
    z_reject = np.interp((m1+m2)*(1.+z),2.*horizon_component_masses,horizon_zs)
    if z>z_reject:
        n_hopeless += 1
        continue

    # Random effective spin
    cx = np.random.random()
    xmin = -1
    xmax = 1
    chi_eff = mu_chi+np.sqrt(2.*sig_chi**2)*erfinv(\
                                    erf((xmin-mu_chi)/(np.sqrt(2.*sig_chi**2))) \
                                    + cx*(erf((xmax-mu_chi)/(np.sqrt(2.*sig_chi**2))) - erf((xmin-mu_chi)/(np.sqrt(2.*sig_chi**2))) )\
                                    )

    # For lack of a better choice, assign each component spin this same value
    s1z = chi_eff
    s2z = chi_eff

    # Extrinsic parameters
    ra = 2.*np.pi*np.random.random()
    dec = np.arccos(2.*np.random.random()-1.) + np.pi/2.
    pol = 2.*np.pi*np.random.random()
    inc = np.arccos(2.*np.random.random()-1.)

    # Generate waveform
    hp, hc = get_fd_waveform(approximant="IMRPhenomD", mass1=m1*(1.+z), mass2=m2*(1.+z),
                                    spin1z=s1z, spin2z=s2z,
                                    inclination=inc, distance=DL,
                                    f_lower=15, delta_f=1.)

    # Project onto detectors
    time = 1126259642.413
    Hp, Hx = H1.antenna_pattern(ra, dec, pol, time)
    Lp, Lx = L1.antenna_pattern(ra, dec, pol, time)
    Vp, Vx = V1.antenna_pattern(ra, dec, pol, time)
    s1 = Hp*hp + Hx*hc
    s2 = Lp*hp + Lx*hc
    s3 = Vp*hp + Vx*hc

    # Compute network SNR
    try:
        sqSNR1 = matchedfilter.overlap(s1,s1,psd=psd_H1,low_frequency_cutoff=15.,normalized=False)
        sqSNR2 = matchedfilter.overlap(s2,s2,psd=psd_L1,low_frequency_cutoff=15.,normalized=False)
        sqSNR3 = matchedfilter.overlap(s3,s3,psd=psd_V1,low_frequency_cutoff=15.,normalized=False)
    except ValueError:
        print(s1)
        print(m1,m2,z)
        break
    snr = np.sqrt(sqSNR1+sqSNR2+sqSNR3)

    if snr>=10.:

        n_det += 1

        # Record
        saved_m1s = np.append(saved_m1s,m1)
        saved_m2s = np.append(saved_m2s,m2)
        saved_s1zs = np.append(saved_s1zs,s1z)
        saved_s2zs = np.append(saved_s2zs,s2z)
        saved_zs = np.append(saved_zs,z)
        saved_DLs = np.append(saved_DLs,DL)
        saved_ras = np.append(saved_ras,ra)
        saved_decs = np.append(saved_decs,dec)
        saved_incs = np.append(saved_incs,inc)
        saved_snrs = np.append(saved_snrs,snr)

        print(n_trials,n_det,n_hopeless)

populationDict = {\
        'm1':saved_m1s,\
        'm2':saved_m2s,\
        'a1':saved_s1zs,\
        'a2':saved_s2zs,\
        'z':saved_zs,\
        'Dl':saved_DLs,\
        'ra':saved_ras,\
        'dec':saved_decs,\
        'inc':saved_incs,\
        'snr':saved_snrs,\
        'seed':np.random.randint(1000000,size=n_det)\
        }

df = pd.DataFrame(populationDict)
df.to_json('population.json')
