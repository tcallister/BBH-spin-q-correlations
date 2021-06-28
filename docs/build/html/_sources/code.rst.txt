Code
====

Hierarchical Bayesian inference of the BBH population is performed using code in this directory.

Auxiliary functions
-------------------

The file :file:`support.py` contains the following auxiliary functions

.. table::
   :align: left

   ==================================  =========================
   Method                              Description
   ==================================  =========================
   :func:`calculate_Gaussian`          Probability density associated with a 1D
                                       truncated Gaussian.
   :func:`calculate_Gaussian_2D`       Probability density associated with a 2D
                                       truncated Gaussian.
   :func:`injection_weights`           Weights used to calculate the LIGO/Virgo detection efficiency.
   :func:`mock_injection_weights`      Selection function weights for mock population study.
   ==================================  =========================

As discussed in Appendix A of the paper text, selection effects are undone via calculation of the LIGO/Virgo detection efficiency :math:`\xi(\Lambda)`, the fraction of BBHs successfully detected given a population described by :math:`\Lambda`.
In practice, this is done via a Monte Carlo averaging of successfully recovered software injections:

.. math::

    \xi(\Lambda) \propto \left\langle \frac{p(\theta|\Lambda)}{p_\mathrm{inj}(\theta)}\right\rangle_{\mathrm{found\,injections}}.

Here, :math:`p(\theta|\Lambda)` is the probability that a BBH with parameters :math:`\theta` arises under our proposed population :math:`\Lambda`, while :math:`p_\mathrm{inj}(\theta)` is the probability of this BBH arising from the reference population from which injections are drawn.

The function :func:`injection_weights` computes the :math:`w(\theta) = 1/p_\mathrm{inj}(\theta)` associated with the found injections downloaded from the LIGO/Virgo O2 and O3a data releases (see :ref:`Loading external data products`).
In particular, weights are calculated assuming that injections are drawn from a population described by:

.. math::

    p_\mathrm{inj}(m_1) \propto m_1^{-2.35} \quad (2\,M_\odot \leq m_1 \leq 100\,M_\odot)

.. math::

    p_\mathrm{inj}(m_2|m_1) \propto m_2^2 \qquad (2\,M_\odot \leq m_2 \leq m_1)

.. math::

    p_\mathrm{inj}(z) \propto \frac{1}{1+z} \frac{dVc}{dz} (1+z)^2

.. math::

    p_\mathrm{inj}(\chi_{1,z}) = p_\mathrm{inj}(\chi_{2,z}) = \mathrm{const.} \quad (-1\leq \chi_{1/2,z} \leq 1)

Similarly, we will later perform an end-to-end injection and recovery of a mock BBH population, including the application and subsequent removal of selection effects.
The function :func:`mock_injection_weights` will analogously compute the (inverse) injection probabilities associated with our found injections in this study.

Likelihood definitions
----------------------

The likelihood functions for each model considered in the paper text are implemented in :file:`likelihoods.py`.
Each expects four arguments:

* :code:`c`: Array containing proposed hyperparameter values.
* :code:`sampleDict`: Dictionary of preprocessed samples, as constructed via :file:`preprocess_samples.ipynb` (see :ref:`Preprocessing`).
* :code:`injectionDict`: Dictionary of successfully-found software injections, together with weight factors as discussed above, used to quantify detection efficiency. Expected format is

    .. code-block:: python

        injectionDict = {
            'm1':primary_mass_values,
            'm2':secondary_mass_values,
            's1z':primary_mass_spins_z_component,
            's2z':secondary_mass_spins_z_component,
            'z':redshifts,
            'weights':injection_weights
            }

* :code:`priorDict`: Dictionary containing information about our priors. For all parameters except :code:`kappa` (redshift evolution) and :code:`mMin` (minimum black hole mass), entries are tuples listing min/max values. For :code:`kappa`, meanwhile, we pass a single parameter governing the width of our normal prior distribution. The key :code:`mMin`, meanwhile, is used to prescribe a fixed minimum BH mass. An example, copied from :code:`run_emcee_plPeak.py` (see `here <https://github.com/tcallister/BBH-spin-q-correlations/blob/main/code/run_emcee_plPeak.py>`__):

.. _SO: http://stackoverflow.com/

    .. code-block:: python

        priorDict = {
            'lmbda':(-5,4),
            'mMax':(60,100),
            'm0':(20,100),
            'sigM':(1,10),
            'fPeak':(0,1),
            'bq':(-2,10),
            'sig_kappa':6.,
            'mu0':(-1,1),
            'log_sigma0':(-1.5,0.5),
            'alpha':(-2.5,1),
            'beta':(-2,1.5),
            'mMin':5.
            }

The following likelihood models are implemented:

:code:`logp_brokenPowerLaw`
   * *Number of parameters*: 9
   * *Mass model*: Broken power law for :math:`p(m_1)`; power law for :math:`p(m_2|m_1)`
   * *Spin model*: Normal distribution for :math:`p(\chi_\mathrm{eff}|q)`, truncated on :math:`-1 \leq \chi_\mathrm{eff} \leq 1`
   * *Spin vs. mass ratio correlation*: Yes

:code:`logp_powerLawPeak`
   * *Number of parameters*: 11
   * *Mass model*: Mixture between power law and Gaussian components for :math:`p(m_1)`; power law for :math:`p(m_2|m_1)`
   * *Spin model*: Normal distribution for :math:`p(\chi_\mathrm{eff}|q)`, truncated on :math:`-1 \leq \chi_\mathrm{eff} \leq 1`
   * *Spin vs. mass ratio correlation*: Yes

:code:`logp_powerLawPeak_gaussianQ`
   * *Number of parameters*: 12
   * *Mass model*: Mixture between power law and Gaussian components for :math:`p(m_1)`; Gaussian for :math:`p(m_2|m_1)`
   * *Spin model*: Normal distribution for :math:`p(\chi_\mathrm{eff}|q)`, truncated on :math:`-1 \leq \chi_\mathrm{eff} \leq 1`
   * *Spin vs. mass ratio correlation*: Yes

:code:`logp_powerLawPeak_bivariateGaussian`
   * *Number of parameters*: 15
   * *Mass model*: Mixture between power law and Gaussian components for :math:`p(m_1)`. The joint distribution :math:`p(\chi_\mathrm{eff},q|m_1)` is described as a mixture between two bivariate Gaussians, each with their own means and covariance matrices.
   * *Spin model*: See above
   * *Spin vs. mass ratio correlation*: Yes (implicitly)

:code:`logp_powerLawPeak_variableChiMin`
   * *Number of parameters*: 12
   * *Mass model*: Mixture between power law and Gaussian components for :math:`p(m_1)`; Gaussian for :math:`p(m_2|m_1)`
   * *Spin model*: Normal distribution for :math:`p(\chi_\mathrm{eff}|q)`, truncated on the variable range :math:`\chi_\mathrm{min} \leq \chi_\mathrm{eff} \leq 1`
   * *Spin vs. mass ratio correlation*: Yes

:code:`logp_powerLawPeak_noEvol`
   * *Number of parameters*: 9
   * *Mass model*: Mixture between power law and Gaussian components for :math:`p(m_1)`; power law for :math:`p(m_2|m_1)`
   * *Spin model*: Normal distribution for :math:`p(\chi_\mathrm{eff})`, truncated on :math:`-1 \leq \chi_\mathrm{eff} \leq 1`
   * *Spin vs. mass ratio correlation*: No

:code:`logp_powerLawPeak_noEvol_variableChiMin`
   * *Number of parameters*: 10
   * *Mass model*: Mixture between power law and Gaussian components for :math:`p(m_1)`; power law for :math:`p(m_2|m_1)`
   * *Spin model*: Normal distribution for :math:`p(\chi_\mathrm{eff})`, truncated on the variable range :math:`\chi_\mathrm{min} \leq \chi_\mathrm{eff} \leq 1`
   * *Spin vs. mass ratio correlation*: No

Running the inference
---------------------

We use the MCMC package *emcee* [1]_ to perform the majority of our inference and parameter estimation.
Inference with *emcee* is done by calling these scripts, which invoke the likelihoods listed above together with various subsets of events.
Unless stated otherwise, the events used are the 44 BBHs in GWTC-2 with false alarm rates below one per year, excluding GW190814.

.. list-table::
    :widths: 10 10 20
    :header-rows: 1

    * - Script
      - Likelihood
      - Sample
    * - :file:`run_emcee_bpl.py`
      - :func:`logp_brokenPowerLaw`
      - Default
    * - :file:`run_emcee_plPeak.py`
      - :func:`logp_powerLawPeak`
      - Default
    * - :file:`run_emcee_plPeak_O3only.py`
      - :func:`logp_powerLawPeak`
      - Excludes the 10 O1+O2 BBHs
    * - :file:`run_emcee_plPeak_bivariateGaussian.py`
      - :func:`logp_powerLawPeak_bivariateGaussian`
      - Default
    * - :file:`run_emcee_plPeak_gaussianQ.py`
      - :func:`logp_powerLawPeak_gaussianQ`
      - Default
    * - :file:`run_emcee_plPeak_no190412.py`
      - :func:`logp_powerLawPeak`
      - Excludes GW190412
    * - :file:`run_emcee_plPeak_no190412_no190517.py`
      - :func:`logp_powerLawPeak`
      - Excludes GW190412 and GW190517
    * - :file:`run_emcee_plPeak_no190517.py`
      - :func:`logp_powerLawPeak`
      - Excludes GW190517
    * - :file:`run_emcee_plPeak_noEvol.py`
      - :func:`logp_powerLawPeak_noEvol`
      - Default
    * - :file:`run_emcee_plPeak_noEvol_no190412.py`
      - :func:`logp_powerLawPeak_noEvol`
      - Excludes GW190412
    * - :file:`run_emcee_plPeak_variableChiMin.py`
      - :func:`logp_powerLawPeak_variableChiMin`
      - Default
    * - :file:`run_emcee_plPeak_w190814.py`
      - :func:`logp_powerLawPeak`
      - Includes GW190814

.. note::

    The scripts above are written to run on the Caltech LIGO computing cluster, and by default may attempt to initiate a large number of parallel processes.
    If running scripts locally, you may wish to reduce the number of threads prepared when defining an :code:`emcee.EnsembleSampler` object.
    For example, changing

    .. code-block:: python

        sampler = mc.EnsembleSampler(
            nWalkers,dim,logp_powerLawPeak_bivariateGaussian,
            args=[sampleDict,injectionDict,priorDict],
            threads=12)

    to

    .. code-block:: python

        sampler = mc.EnsembleSampler(
            nWalkers,dim,logp_powerLawPeak_bivariateGaussian,
            args=[sampleDict,injectionDict,priorDict],
            threads=4)
    
    reduces the number of parallel processes from 12 to 4.

These scripts all implement a somewhat crude checkpointing.
Running any of the :file:`run_emcee_*.py` scripts will create a file of the form :file:`~/code/output/emcee_samples_*_r00.npy` storing posterior samples.
This file is regularly overwritten during the sampling, so that a run in progress can be monitored.
If a run is interrupted and :file:`run_emcee_*.py` restarted, it will search for an existing output file.
If one exists, then the final walker positions from this previous halted run will be loaded and used to initialize new walker locations.
Subsequent output will now be written to :file:`~/code/output/emcee_samples_*_r01.npy`.
If this run is in turn interrupted and relaunched, output will be written to :file:`~/code/output/emcee_samples_*_r02.npy`, etc.
    
Post-processing
---------------

The output files above contain raw output in the form of an :code:`(n,N,m)` dimensional array, where :code:`n` is the number of parallel walkers used, :code:`N` is the number of MCMC steps, and :code:`m` the number of dimensions in the particular likelihood model.

.. code-block:: python

    >>> import numpy as np
    >>> output = np.load('emcee_samples_plPeak_r00.npy')
    >>> output.shape
    (32, 10000, 11)

To obtain a final set of uncorrelated posterior samples, the script :file:`post_processing.py` can be used to (i) discard a burn-in periord from each walker, (ii) downsample each chain, and (iii) collapse and combine samples from different walkers.
By default, this script burns the first third of every chain and downsamples by twice the maximum autocorrelation length (maximized across all walkers and varaibles), but it's always best to inspect raw chains and verify that these choices are appropriate.

.. code-block:: bash

    $ cd ~/code/output/
    $ python ../post_processing.py emcee_samples_plPeak_r00.npy 

    Shape of sample chain:
    (32, 10000, 11)
    Shape of burned chain:
    (32, 6667, 11)
    Mean correlation length:
    110.4984881768458
    Shape of downsampled chain:
    (32, 31, 11)
    Shape of downsampled chain post-flattening:
    (992, 11)

The output of :file:`post_processing.py` is a new file, of the form :file:`~/code/output/processed_emcee_samples_plPeak_r00.npy` containing a 2D array with the burned, downsampled, and collapsed posterior samples.

.. code-block:: python

    >>> import numpy as np
    >>> processed_output = np.load('processed_emcee_samples_plPeak_r00.npy')
    >>> processed_output.shape
    (992, 11)

Evaluating evidences
--------------------

Although we primarily use *emcee* [1]_ for our parameter estimation, we also use the *dynesty* nested sampler [2]_ to calculate Bayes factors between several different models.
Nested sampling requires specification of priors in a slightly different way, and so we re-implement likelihoods for models for which we want evidences in the file :file:`dynesty_likelihoods.py`.
The following four models are defined, two of which are effectively copies of the models defined above in :ref:`Likelihood definitions`

:code:`logp_powerLawPeak`
   * *Number of parameters*: 11
   * *Mass model*: Mixture between power law and Gaussian components for :math:`p(m_1)`; power law for :math:`p(m_2|m_1)`
   * *Spin model*: Normal distribution for :math:`p(\chi_\mathrm{eff}|q)`, truncated on :math:`-1 \leq \chi_\mathrm{eff} \leq 1`
   * *Spin vs. mass ratio correlation*: Yes

:code:`logp_powerLawPeak_noNeg`
   * *Number of parameters*: 11
   * *Mass model*: Mixture between power law and Gaussian components for :math:`p(m_1)`; power law for :math:`p(m_2|m_1)`
   * *Spin model*: Normal distribution for :math:`p(\chi_\mathrm{eff}|q)`, truncated on :math:`0 \leq \chi_\mathrm{eff} \leq 1`
   * *Spin vs. mass ratio correlation*: Yes

:code:`logp_powerLawPeak_noEvol`
   * *Number of parameters*: 9
   * *Mass model*: Mixture between power law and Gaussian components for :math:`p(m_1)`; power law for :math:`p(m_2|m_1)`
   * *Spin model*: Normal distribution for :math:`p(\chi_\mathrm{eff})`, truncated on :math:`-1 \leq \chi_\mathrm{eff} \leq 1`
   * *Spin vs. mass ratio correlation*: No

:code:`logp_powerLawPeak_noEvol_noNeg`
   * *Number of parameters*: 9
   * *Mass model*: Mixture between power law and Gaussian components for :math:`p(m_1)`; power law for :math:`p(m_2|m_1)`
   * *Spin model*: Normal distribution for :math:`p(\chi_\mathrm{eff})`, truncated on :math:`0 \leq \chi_\mathrm{eff} \leq 1`
   * *Spin vs. mass ratio correlation*: No

Each function expects the same four arguments discussed in :ref:`Likelihood definitions`, and inference with these four models is done by running the following four scripts:

* :file:`run_dynesty_plPeak.py`
* :file:`run_dynesty_plPeak_noEvol.py`
* :file:`run_dynesty_plPeak_noEvol_noNeg`
* :file:`run_dynesty_plPeak_noNeg.py`

.. note::

    As in :ref:`Running the inference`, these scripts are by default set up to launch a decent number of parallel processes.
    If you wish to reduce this number, then change the lines

    .. code-block:: python

        if os.path.isfile(tmpfile):
            ...
            newPool = Pool(16)
            ...    

        else:
            sampler = dynesty.NestedSampler(
                ...
                pool = Pool(16), queue_size=16,
                ...

    to

    .. code-block:: python

        if os.path.isfile(tmpfile):
            ...
            newPool = Pool(4)
            ...    

        else:
            sampler = dynesty.NestedSampler(
                ...
                pool = Pool(4), queue_size=4,
                ...

    to reduce the number of parallel processes from 16 to 4, for example.

To assess uncertainties on reported evidences, we run each of the above scripts several times in parallel.
Each script therefore expects a unique job number, e.g.

.. code-block:: python

    python run_dynesty_plPeak.py 1

will run this script with a jobnumber of 1.

Temporary checkpointing files are periodically written to the :code:`~/code/dynesty_output/` directory.
Running the example above, for instance, will yield a checkpoint file :code:`~/code/dynesty_output/dynesty_results_job1.resume.npy`.

Each of the above scripts, when run, will write temporary checkpoint files to the directory :file:`~/code/dynesty_output/`.
If your run is interrupted, then upon being relaunched *dynesty* will pick back up from the last written checkpoint file.

The end result will be an output file of the form :file:`~/code/dynesty_output/dynesty_results_job1.npy`.
See the *dynesy* doumentation [2]_ for detailed instructions regarding how to work with this output.
For our purposes, the final log-evidence for a given model/job can be quickly read out via

.. code-block:: python

    import numpy as np 
    output_file = "dynesty_results_job1.npy"
    logz = np.load(output_file,allow_pickle=True)[()]['logz'][-1]
   
.. [1] https://emcee.readthedocs.io/en/stable/
.. [2] https://dynesty.readthedocs.io/en/latest/

