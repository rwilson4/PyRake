User Guides
===========

:doc:`population_mean` is a tutorial. It calibrates a single survey
with :class:`~pyrake.Rake`, first staying close to baseline weights and
then minimizing Kish's design effect, traces the efficient frontier
between the two, checks covariate balance, and estimates the population
mean with each set of weights.

:doc:`treatment_effect` is a tutorial for a randomized experiment
measured by a survey. It calibrates the treatment and control groups
together with :class:`~pyrake.JointCalibrator`, for internal validity
(the groups balanced against each other) and external validity (each
balanced against the population), traces the external
validity-variance frontier and its knee, and estimates the treatment
effect with :class:`~pyrake.SimpleDifferenceEstimator`, including a
sensitivity analysis.

:doc:`classifier_metrics` estimates a classifier's precision and
recall in a target population when its labeled holdout set is not
representative. It calibrates the holdout set with
:class:`~pyrake.Rake`, including to the model's score distribution,
which is known for the whole population, treats each metric as a ratio
of population means with :class:`~pyrake.RatioEstimator`, plots the
precision and recall curves with confidence bands, and closes with a
sensitivity analysis.

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
   treatment_effect
   classifier_metrics
   usage
