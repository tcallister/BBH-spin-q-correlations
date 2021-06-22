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

Running the inference:
----------------------

Inference is done by calling these scripts, which invoke the likelihoods listed above together with various subsets of events.
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




Burning and downsampling: :code:`post_processing.py`
----------------------------------------------------
