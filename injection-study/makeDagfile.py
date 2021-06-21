import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.special import erf
import sys
import pandas as pd

def calculate_Gaussian(x, mu, sigma2, low, high): 
    norm = np.sqrt(sigma2*np.pi/2)*(-erf((low-mu)/np.sqrt(2*sigma2)) + erf((high-mu)/np.sqrt(2*sigma2)))
    y = (1.0/norm)*np.exp((-1.0*(x-mu)*(x-mu))/(2.*sigma2))
    y[y!=y] = 0.
    return y

injections = pd.read_json('./population.json')
injections.sort_index(inplace=True)
n_total = len(injections)

# Read and reweight spins
inj_spins = np.array((injections.m1*injections.a1 + injections.m2*injections.a2)/(injections.m1+injections.m2))
old_weights = calculate_Gaussian(inj_spins,0.,1.,-1.,1)
new_weights = calculate_Gaussian(inj_spins,0.05,0.15**2.,-1.,1.)

draw_weights = new_weights/old_weights
to_inject = np.random.choice(range(n_total),size=50,p=draw_weights/np.sum(draw_weights),replace=False)
np.savetxt('injlist.txt',to_inject,fmt="%d")

dagfile='./condor/bilby.dag'
with open(dagfile,'w') as df:

    for i in to_inject:

        df.write('JOB {0} /home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/injection-study/condor/bilby.sub\n'.format(i))
        df.write('VARS {0} jobNumber="{0}" json="/home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/injection-study/population.json" outdir="/home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/injection-study/output/"\n\n'.format(i))
