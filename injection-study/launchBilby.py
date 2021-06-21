import numpy as np
import bilby
import argparse
import sys
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-json',help='Json file with population instantiation')
parser.add_argument('-job',help='Job number',type=int)
parser.add_argument('-outdir',help="Output directory")
args = parser.parse_args()

# Load dataframe and select injection
injections = pd.read_json(args.json)
injection = injections.iloc[args.job]
print(injection)

# Set the duration and sampling frequency of the data segment that we're
# going to inject the signal into
duration = 8.
sampling_frequency = 2048.

# Specify the output directory and the name of the simulation.
label = "job_{0:05d}".format(args.job)
bilby.core.utils.setup_logger(outdir=args.outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(int(injection.seed))

# We are going to inject a binary black hole waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# spins of both black holes (a, tilt, phi), etc.
#tilt_1=0,\
#tilt_2=0,\
#phi_12=0,\
#phi_jl=0,\
injection_parameters = dict(
    mass_1=injection.m1*(1.+injection.z),\
    mass_2=injection.m2*(1.+injection.z),\
    chi_1=injection.a1,\
    chi_2=injection.a2,\
    luminosity_distance=injection.Dl,\
    theta_jn=injection.inc,\
    psi=2.*np.pi*np.random.random(),\
    phase=2.*np.pi*np.random.random(),\
    geocent_time=1126259642.413,\
    ra=injection.ra,\
    dec=injection.dec)

# Fixed arguments passed into the source model
waveform_arguments = dict(waveform_approximant='IMRPhenomD',
                                  reference_frequency=50., minimum_frequency=15.)

# Create the waveform_generator using a LAL BinaryBlackHole source function
# the generator will convert all the parameters
waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration, sampling_frequency=sampling_frequency,
                frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                        waveform_arguments=waveform_arguments)

# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])

ifos[0].power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file="/home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/injection-study/aligo_O3actual_H1.txt")
ifos[1].power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file="/home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/injection-study/aligo_O3actual_L1.txt")
ifos[2].power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file="/home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/injection-study/avirgo_O3actual.txt")

ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency, duration=duration,
                start_time=injection_parameters['geocent_time'] - 3)

ifos.inject_signal(waveform_generator=waveform_generator,
                           parameters=injection_parameters)

# For this analysis, we implemenet the standard BBH priors defined, except for
# the definition of the time prior, which is defined as uniform about the
# injected value.
# Furthermore, we decide to sample in chirp mass and mass ratio, due to the
# preferred shape for the associated posterior distributions.
#highMasses = [12755, 47652, 49140, 49382]
lowMasses = [28327]
#if args.job in highMasses:
#    priors = bilby.gw.prior.BBHPriorDict("/home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/injection-study/prior_highMass.prior")
if args.job in lowMasses:
    priors = bilby.gw.prior.BBHPriorDict("/home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/injection-study/prior_lowMass.prior")
else:
    priors =  bilby.gw.prior.BBHPriorDict("/home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/injection-study/prior.prior")
priors['geocent_time'] = bilby.core.prior.Uniform(minimum=injection_parameters['geocent_time'] - 0.1,maximum=injection_parameters['geocent_time'] + 0.1,name='geocent_time', latex_label='$t_c$', unit='$s$')

# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveoform generator, as well the priors.
# The explicit time, distance, and phase marginalizations are turned on to
# improve convergence, and the parameters are recovered by the conversion
# function.
likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=ifos, waveform_generator=waveform_generator, priors=priors,
                distance_marginalization=True, phase_marginalization=True, time_marginalization=True)

# Run sampler. In this case we're going to use the `cpnest` sampler
# Note that the maxmcmc parameter is increased so that between each iteration of
# the nested sampler approach, the walkers will move further using an mcmc
# approach, searching the full parameter space.
# The conversion function will determine the distance, phase and coalescence
# time posteriors in post processing.
result = bilby.run_sampler(
            likelihood=likelihood, priors=priors, sampler='dynesty', nlive=2000,
                injection_parameters=injection_parameters, outdir=args.outdir,
                    label=label,conversion_function=bilby.gw.conversion.generate_all_bbh_parameters)


