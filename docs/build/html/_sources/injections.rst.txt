Mock Population Study
=====================

As an end-to-end test for possible systematics, we perform a mock population study in which we draw events from a simulated population, run parameter estimation over these signals, and hierarchically infer the properties of the simulated parent population.
By design, our mock population has **no intrinsic correlations** between mass ratios and spin; accordingly, we correctly find no correlations between the :math:`\chi_\mathrm{eff}` and :math:`q` distributions of the parent population.

Generating a mock population
----------------------------

First, a mock population of detected events is generated via

.. code-block:: bash

    $ python genPopulation.py

This script repeatedly draws events from a parent population, computes an expected network signal-to-noise ratio (SNR) using estimates of Hanford, Livingston, and Virgo's O3a sensitivity, and saves as "detectable" those events with network SNRs above 10.
This processes repeats until we obtain :math:`5\times10^4` such detectable events.

To help expedite this processes, this script eliminates "hopeless" proposals (those that have no chance of being detected) by calculating a horizon redshift (stored in the array :code:`horizon_zs`) as a function of binary mass.
Proposed events that lie beyond this horizon are immediately discarded, without having to generate waveforms and compute waveform inner products.

The parent population is hardcoded as follows:

.. code-block:: python

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

Proposed primary masses follow a broken power law with indices :code:`a1` and :code:`a2`,

.. math::

    p(m_1) \propto
        \begin{cases}
        m_1^{a_1} & (m_\mathrm{min} \leq m_1 \leq m_0) \\
        m_1^{a_2} & (m_0 < m_1 m_\mathrm{max})
        \end{cases}

secondary masses follow a power law with index :code:`bq`,

.. math::

    p(m_2|m_1) \propto m_2^{\beta_q} \quad (m_\mathrm{min}\leq m_2 \leq m_1)

redshifts are distributed as

.. math::

    p(z) \propto \frac{1}{1+z} \frac{dV_c}{dz} (1+z)^\kappa

and :math:`\chi_\mathrm{eff}` values are normally distributed with mean :code:`mu_chi` and standard deviation :code:`sig_chi`.
Since we only care about :math:`\chi_\mathrm{eff}` and not the underlying component spins, for a given event both of the aligned spin components are set to the proposed :math:`\chi_\mathrm{eff}` value, such that

.. math::

    \begin{aligned}
    \chi_\mathrm{eff}
        &= \frac{s_{1,z} + q s_{2,z}}{1+q} \\
        &= \frac{\chi_\mathrm{eff} + q \chi_\mathrm{eff}}{1+q} \\
        &= \chi_\mathrm{eff}.
    \end{aligned}

.. Note::

    The :math:`\chi_\mathrm{eff}` distribution described here is **not** the :math:`\chi_\mathrm{eff}` distribution that we will ultimately adopt as our injected population; see :ref:`Choosing an injection set` below.

The output of :code:`genPopulation.py` is a file :code:`population.json` containing listing the extrinsic and intrinsic properties of detectable events, as well as a random seed to be used later when generating Gaussian noise as part of parameter estimation.
The file is most easily parsed as a :code:`pandas.DataFrame` object:

.. code-block:: python

    >>> import pandas as pd
    >>> event_list = pd.read_json('population.json')
    >>> event_list.columns
    Index([u'Dl', u'a1', u'a2', u'dec', u'inc', u'm1', u'm2', u'ra', u'seed',
       u'snr', u'z'],
      dtype='object')

Events are labeled by indices ranging from 0 to 49999.
To obtain the properties of a specific index (say, index 11), do

.. code-block:: python

    >>> event = event_list.loc[11]
    Dl        1261.271988
    a1           0.207897
    a2           0.207897
    dec          3.239911
    inc          2.719737
    m1          13.374225
    m2          12.097571
    ra           1.412080
    seed    891333.000000
    snr         10.269079
    z            0.243488
    Name: 11.0, dtype: float64

Choosing an injection set
-------------------------

.. note::

    The following workflow is structured primarily to enable usage of the Caltech LIGO computing cluster.
    Running in a different environment will likely require some modification of the steps described below.

Once the file :file:`population.json` has been created, we now wish to downselect to a catalog of 50 events that we will inject into simulated data and perform parameter estimation on.
This is accomplished by running the script

.. code-block:: bash

    $ python makeDagfile.py

This script randomly draws a subset of 50 events from the proposed population.
Note that, to ensure good coverage of the full interval :math:`-1\leq \chi_\mathrm{eff}\leq 1`, the proposed events in :file:`population.json` were drawn from a very broad :math:`chi_\mathrm{eff}` distribution, centered at zero with a standard deviation of one.
Observationally, however, we know that the :math:`\chi_\mathrm{eff}` distribution is much narrower.
To produce an injected catalog corresponding to a more realistic spin distribution, the random draws in :file:`makeDagfile.py` are *weighted* such that our final catalog corresponds to a BBH population with

.. math::

    p(\chi_\mathrm{eff}) ~ N(\chi_\mathrm{eff}|\mu = 0.05, \sigma = 0.15)

Running :file:`makeDagfile.py` produces two output files:

:file:`injlist.txt`
    This file contains the list of 50 events selected from :file:`population.json` for injection and parameter estimation.

    .. code-block:: bash

        $ head injlist.txt 
        20456
        12838
        11417
        15322
        23031
        2688
        46717
        2405
        15356
        48467

:file:`condor/bilby.dag`
    This is a corresponding dagfile that, on the LIGO Caltech cluster, can be submitted to HTCondor to run parameter estimation over these 50 events.

Running parameter estimation
----------------------------

To launch parameter estimation with HTCondor on the LIGO Caltech cluster, do

.. code-block:: bash

    $ cd condor/
    $ condor_submit_dag bilby.dag

Parameter estimation is done using the *Bilby* package [1]_ in conjunction with the *dynesty* nested sampler [2]_, and is launched with the script :file:`launchBilby.py`.
This script takes three arguments:

.. code-block:: bash

    $ python launchBilby.py --h
    usage: launchBilby.py [-h] [-json JSON] [-job JOB] [-outdir OUTDIR]

    optional arguments:
      -h, --help      show this help message and exit
      -json JSON      Json file with population instantiation
      -job JOB        Job number
      -outdir OUTDIR  Output directory

For example, to begin parameter estimation over event number 20456 (the first entry in :file:`injlist.txt` above), do

.. code-block:: bash

    $ python launchBilby.py -json ./populations.json -job 20456 -outdir ./output/

Inside :file:`launchBilby.py`, the relevant injection information is read from :file:`populations.json`, a corresponding waveform generated, and added into Gaussian noise consistent with Hanford, Livingston, and Virgo O3 PSDs.
Parameter estimation is then launched, with output written to the :file:`injection-study/output/` folder.
By default, all events are run with the same broad prior specified in :file:`prior.prior`:

.. literalinclude:: ../injection-study/prior.prior

Under this broad prior, however, *dynesty* may have a hard time finding the likelihood peak for low-mass injections falling near the prior boundary.
You should always be sure to check posteriors and make sure they look well-behaved.
For example, the following code block reads out and plots the posterior of one such poorly-behaving run:

.. code-block:: python

    import json
    import numpy as np 
    import matplotlib.pyplot as plt
    from corner import corner

    # Read file
    f = './outout/job_03055_result.json'
    with open(f,'r') as jf:
        result = json.load(jf)
        
    # Extract injected parameters
    inj_chi = result['injection_parameters']['chi_eff']
    inj_q = result['injection_parameters']['mass_ratio']
    inj_m1 = result['injection_parameters']['mass_1_source']
    inj_m2 = result['injection_parameters']['mass_2_source']
    inj_z = result['injection_parameters']['redshift']['content']
    inj_Mc = (inj_q/(1.+inj_q)**2)**(3./5.)*(inj_m1+inj_m2)
    inj_params = [inj_chi,inj_q,inj_Mc,inj_z]

    # Extract posterior samples
    chiEff = np.array(result['posterior']['content']['chi_eff'])
    m1 = np.array(result['posterior']['content']['mass_1_source'])
    m2 = np.array(result['posterior']['content']['mass_2_source'])
    z = np.array(result['posterior']['content']['redshift'])
    s1z =  np.array(result['posterior']['content']['spin_1z'])
    s2z =  np.array(result['posterior']['content']['spin_2z'])

    # Convert to mass ratio and chirp mass
    q = m2/m1
    Mc = (q/(1.+q)**2)**(3./5.)*(m1+m2)

    # Make a corner plot of the posterior samples
    ndim = 4
    labels = [r"$\chi_\mathrm{eff}$",r"$q$",r"$\mathcal{M}_c$",r"$z$"]
    fig = corner(np.transpose([chiEff,q,Mc,z]),labels=labels,fontsize=18)
    axes = np.array(fig.axes).reshape((ndim, ndim))

    # Overplot injected values as horizontal/vertical lines
    for i in range(ndim):
        ax = axes[i, i]
        ax.axvline(inj_params[i], color="black")
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(inj_params[xi], color="black")
            ax.axhline(inj_params[yi], color="black")

    plt.show()

.. image:: ./images/bad_corner.pdf
    :width: 600
    :align: center
    :class: with-border

Note how the injected chirp mass and redshift fall essentially *on* the prior boundary, and our sampler has completely missed the region where the injection lies.
In this case, such events should be rerun with restricted priors.
You might try, for instance, editing the :code:`highMasses` and :code:`lowMasses` arrays inside :code:`launchBilby.py`, which specify events to be run with alternative priors restricted to higher and lower chirp mass ranges:

.. code-block:: python

    ...
    highMasses = []
    lowMasses = [03055]
    if args.job in highMasses:
        priors = bilby.gw.prior.BBHPriorDict("/home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/injection-study/prior_highMass.prior")
    if args.job in lowMasses:
        priors = bilby.gw.prior.BBHPriorDict("/home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/injection-study/prior_lowMass.prior")
    else:
        priors =  bilby.gw.prior.BBHPriorDict("/home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/injection-study/prior.prior")
    ...

Here I have pointed low mass events to an existing prior file, but new/bespoke priors can be made as needed.
In the case of event 03055, running with a restricted chirp mass prior yields a much more reasonable posterior:

.. image:: ./images/good_corner.pdf
    :width: 600
    :align: center
    :class: with-border

Note that the consistent usage in :file:`launchBilby.py` of the random seed associated with this event in :file:`population.json` ensures that the *random noise realizations* will be consistent across different PE trials.

Hierarchical inference
----------------------

Running hierarchical inference over our parameter estimation results consists of two steps.

First, do

.. code-block:: bash

    $ python prep_hierarchical_inference.py

This script loops across each injection, reads in posteriors and downselects to a reasonable number of samples, calculates prior weights, and saves these intermediate results to the directory :file:`injection-study/tmp/`.

After this preparation step, hierarchical inference is performed with *emcee* [3]_ by running one or both of the following scripts:

* :file:`run_emcee_plPeak.py`
   * *Number of parameters*: 11
   * *Mass model*: Mixture between power law and Gaussian components for :math:`p(m_1)`; power law for :math:`p(m_2|m_1)`
   * *Spin model*: Normal distribution for :math:`p(\chi_\mathrm{eff}|q)`, truncated on :math:`-1 \leq \chi_\mathrm{eff} \leq 1`
   * *Spin vs. mass ratio correlation*: Yes

* :file:`run_emcee_bpl.py`
   * *Number of parameters*: 9
   * *Mass model*: Broken power law for :math:`p(m_1)`; power law for :math:`p(m_2|m_1)`
   * *Spin model*: Normal distribution for :math:`p(\chi_\mathrm{eff}|q)`, truncated on :math:`-1 \leq \chi_\mathrm{eff} \leq 1`
   * *Spin vs. mass ratio correlation*: Yes
    
Output from the first script, :file:`run_emcee_plpeak.py`, is adopted as our fiducial result presented in the paper.
Note, though, that the mass model presumed in this script is deliberaly *mismatched* to the injected mass distribution, which takes the form of a broken power law.
The second script, meanwhile, adopts a mass model consistent with our injected population.

As in the case of our hierarchical inference over true LIGO-Virgo events (see :ref:`Code`), both of the above scripts invoke the likelihoods defined in :file:`code/likelihoods.py`.
Additionally, both scripts adopt the checkpointing system described in :ref:`Running the inference`.

Raw sample chains from *emcee* are saved in files of the form :file:`emcee_samples_plPeak_r00.npy`, in which is stored an array of size :code:`(n,N,m)`, where :code:`n` is the number of walkers used in *emcee*, :code:`N` is the number of steps taken by each walker, and :code:`m` is the dimensionality of our model. 
The final step is to post-process this output via

.. code-block:: bash

    $ python ../code/post_processing.py emcee_samples_plPeak_r00.npy 

which will create a new file :file:`processed_emcee_samples_plPeak_r00.npy` that has been downsampled and collapsed to a 2D array of independent posterior samples.
See :ref:`Post-processing` for more information and examples concerning these output files.

.. [1] https://lscsoft.docs.ligo.org/bilby/
.. [2] https://dynesty.readthedocs.io/en/latest/
.. [3] https://emcee.readthedocs.io/en/stable/

