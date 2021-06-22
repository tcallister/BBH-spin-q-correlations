Input
=====

The contents of this directory fetch various external data products (posterior samples, detector sensitivity estimates, etc.) and perform preprocessing in preparation for hierarchical inference of the BBH population.

Loading external data products
------------------------------

To load all external sensitivity products, use

.. code-block:: bash

   . copyData.sh

This will download and unzip the following files and directories:

* **o3a_bbhpop_inj_info.hdf**:
    Record of successfully recovered software injections of BBH signals into LIGO/Virgo O3a data [1]_.
* **injections_O1O2an_spin.h5**:
    Record of mock BBH signals deemed detectable (based on a semi-analytic sensitivity estimation) in O1 and O2 [2]_.
* **all_posterior_samples/**:
    Parameter estimation results associated with new compact binary detections in O3a [3]_.
* **GWTC-1_sample_release/**:
    Parameter estimation results for compact binary detections made in O1 and O2 [4]_.

Preprocessing
-------------

Next, open and run the following Jupyter notebook:

.. code-block:: bash

   jupyter notebook preprocess_samples.ipynb 

This notebook unpacks the posterior samples downloaded above, downsamples to a smaller number of samples per event, and computes the prior weights on masses, spins, and redshift imposed during parameter estimation.
Two new files will be created:

* **sampleDict_w190814.pickle**:
    A pickled dictionary containing posterior samples and their prior weights for every BBH candidate in GWTC-2 with a false alarm rate below one per year, including the outlier event GW190814.
* **sampleDict.pickle**:
    As above, but without GW190814.

These files can be accessed in a Python session via

.. code-block:: python

    >>> import numpy as np
    >>> preprocessed_samples = np.load('sampleDict.pickle')

    >>> # Data for individual events accessed as keys
    >>> print(preprocessed_samples.keys())

    [u'S190731aa',
     u'S190514n',
     u'S190828l',
     u'GW151012',
     u'GW170814',
     ...

    >>> # Nested dictionary containing samples and weights for each individual event
    >>> print(preprocessed_samples['GW150914'].keys())

    [u'Xeff', u'cost1', u'a1', u'Xeff_priors', u'cost2', u'a2', u'm1', u'm2', u'weights', u'z']

The above dictionary keys denote the following quantities:

* :code:`Xeff`: Effective inspiral spin posterior samples
* :code:`a1`: Posterior samples on the dimensionless spin magnitude of the primary BH
* :code:`a2`: Posterior samples on the dimensionless spin magnitude of the secondary BH
* :code:`cost1`: Posterior samples on the spin-orbit (cosine) tilt angle of the primary BH
* :code:`cost2`: Posterior samples on the spin-orbit (cosine) tilt angle of the secondary BH
* :code:`m1`: Posterior samples on the mass of the primary BH
* :code:`m2`: Posterior samples on the mass of the secondary BH
* :code:`z`: Posterior samples on the redshift of the BBH
* :code:`Xeff_priors`: Priors associated with the effective spin samples, corresponding to uniform and isotropic component spin priors.
* :code:`weights`: Additional weights with which to convert these posterior samples from a prior that is uniform in detector-frame mass and and Euclidean volume to a new prior that is uniform in *source*-frame mass and that follows an astrophysically-reasonable redshift evolution:

    .. math::

        p(z) \propto \frac{1}{1+z} \frac{dV_c}{dz} (1+z)^{2.7}

  where :math:`\frac{dV_c}{dz}` is the differential comoving volume per unit redshift.



.. [1] https://dcc.ligo.org/LIGO-P2000217/public
.. [2] https://dcc.ligo.org/LIGO-P2000434/public
.. [3] https://dcc.ligo.org/LIGO-P2000223/public
.. [4] https://dcc.ligo.org/LIGO-P1800370/public
