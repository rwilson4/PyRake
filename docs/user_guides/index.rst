User Guides
===========

:doc:`population_mean` is a tutorial. It calibrates a single survey
with :class:`~pyrake.Rake`, first staying close to baseline weights and
then minimizing Kish's design effect, traces the efficient frontier
between the two, checks covariate balance, and estimates the population
mean with each set of weights.

The usage guide is a set of recipes for the two main workflows:
calibrating weights, whether for a single sample with
:class:`~pyrake.Rake` or for treatment and control groups jointly with
:class:`~pyrake.JointCalibrator`, and estimating with them. It covers
population means, ratios, treatment effects in experiments and
observational studies, linear combinations of estimators, double
sampling, and balance plots, and ends with a complete survey analysis of
an A/B test. ``pyrake.usage()`` returns the same text.

.. toctree::
   :maxdepth: 1

   population_mean
   usage
