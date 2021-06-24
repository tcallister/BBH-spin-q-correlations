.. BBH-spin-q-correlations documentation master file, created by
   sphinx-quickstart on Mon Jun 21 11:22:47 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Companion code for Callister+ 2021 ("Who Ordered That?")
========================================================

.. image:: ../code/figures/reweighted_posterior_scatter_yesEvolution.pdf
    :width: 500
    :align: center
    :class: with-border

This page describes the code used to produce results presented in `ArXiv:2106.00521`_
In this paper, we sought to measure the degree of correlation between the mass ratio :math:`q` and effective inspiral spins :math:`\chi_\mathrm{eff}` of the binary black hole population observed by the Advanced LIGO and Virgo experiments.

We found that, at high confidence, unequal-mass events (low :math:`q`) systematically have larger :math:`\chi_\mathrm{eff}`, such that these parameters are anticorrelated.
Applying this population fit as a new prior on the properties of LIGO/Virgo's BBH detections yields the reweighted posteriors shown in the figure above.

.. _`ArXiv:2106.00521`: http://arxiv.org/abs/2106.00521

Contents:

.. toctree::
   :maxdepth: 1

   input
   code
   injections


