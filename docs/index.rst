PyRake
======

PyRake calculates balancing weights that adjust a survey, or one arm of
an experiment, to look like the population it is meant to represent,
and then uses those weights for estimation and inference. The weights
solve a convex optimization problem that trades off three goals:

- **Reduce bias**, by up-weighting under-represented respondents.
- **Improve representativeness**, by balancing important covariates
  against their population means, exactly or to within a tolerance.
- **Control variance**, by bounding the mean squared weight, which
  bounds the design effect.

Concretely, PyRake solves

.. math::

   \begin{array}{ll}
   \mbox{minimize}   & D(w, v) \\
   \mbox{subject to} & \frac{1}{M} X^T w = \mu \\
                     & \| \frac{1}{M} Z^T w - \nu \|_\infty \leq \psi \\
                     & \frac{1}{M} \| w \|_2^2 \leq \phi \\
                     & w_{\min} \leq w \leq w_{\max},
   \end{array}

where :math:`D` is a distance from baseline weights :math:`v` (squared
:math:`\ell_2`, KL divergence, or Huber), :math:`X` holds covariates to
balance exactly and :math:`Z` covariates to balance approximately. With
the KL divergence and only marginal constraints this is classical
raking [DS40]_, and with exact balance it is entropy balancing [Hai12]_;
bounding the mean squared weight, as stable balancing weights do
[Zub15]_, limits the variance the weights cost. The default imbalance
tolerance, :math:`\psi = 1.5 / \sqrt{M}`, follows [CC65]_.

The optimization runs on `Cvxium <https://convexanalytics.com/cvxium/>`_,
whose interior-point solvers [BV04]_ exploit the problem's structure, so
a solve with thousands of respondents takes milliseconds.

The library provides:

- **Calibration**: :class:`~pyrake.Rake` for a single sample, and
  :class:`~pyrake.JointCalibrator` for treatment and control groups
  calibrated together, for internal validity (balance between the
  groups) and external validity (balance with the population), with a
  fallback cascade when the full problem is infeasible.
- **Tradeoffs**: :class:`~pyrake.EfficientFrontier` and
  :class:`~pyrake.ExternalValidityVarianceFrontier` trace the
  bias-variance and balance-variance Pareto frontiers and find their
  knees.
- **Estimation**: population means (:class:`~pyrake.IPWEstimator`,
  inverse propensity weighting after [HT52]_, :class:`~pyrake.SIPWEstimator`,
  and their augmented, doubly robust versions), ratios, and treatment
  effects (:class:`~pyrake.ATEEstimator` and its relatives), each with
  variance, confidence intervals, p-values, the bootstrap, and a
  sensitivity analysis for hidden bias [ZSB19]_ [Fog23]_. The
  design-based view of survey inference these follow is set out in
  [Wil25]_.
- **Diagnostics**: balance plots against the population
  (:func:`~pyrake.plot_balance`) and between groups
  (:func:`~pyrake.plot_balance_2_sample`), with a Wilcoxon signed-rank
  test of overall balance [HW99]_.

Full citations are on the :doc:`references` page.

Installation
------------

.. code-block:: bash

    pip install pyrake-survey

The package installs as ``pyrake-survey`` and imports as ``pyrake``.

Quickstart
----------

Draw a survey from a population in which younger and urban people
respond more often, so the sample over-represents them. The naive
sample mean is biased:

.. doctest::

    >>> import numpy as np
    >>> import pyrake
    >>> rng = np.random.default_rng(0)
    >>> N, M = 100_000, 500
    >>> age = rng.normal(45, 15, N)
    >>> urban = rng.random(N) < 0.6
    >>> outcome = 3.0 + 0.04 * age + 1.5 * urban + rng.normal(0, 1, N)
    >>> p_respond = 1 / (1 + np.exp(0.03 * (age - 45) - 0.8 * urban + 5))
    >>> sample = rng.choice(N, size=M, replace=False, p=p_respond / p_respond.sum())
    >>> print(f"population {outcome.mean():.3f}, naive {outcome[sample].mean():.3f}")
    population 5.702, naive 5.572

Calibrate the weights so the sample matches the population's mean age
and urban share exactly, with a mean squared weight of at most 2:

.. doctest::

    >>> X = np.column_stack([age[sample], urban[sample]])
    >>> mu = np.array([age.mean(), urban.mean()])
    >>> rake = pyrake.Rake(distance=pyrake.KLDivergence(), X=X, mu=mu, phi=2.0)
    >>> result = rake.solve()
    >>> w = result.solution
    >>> print(result.status)
    SolveStatus.OPTIMAL
    >>> bool(np.allclose(X.T @ w / M, mu))
    True
    >>> print(f"design effect {np.mean(w ** 2):.3f}")
    design effect 1.471

The weights imply propensity scores, which an estimator turns into a
point estimate and a confidence interval:

.. doctest::

    >>> estimator = pyrake.SIPWEstimator(
    ...     propensity_scores=(M / N) / w, outcomes=outcome[sample]
    ... )
    >>> lo, hi = estimator.confidence_interval(alpha=0.05)
    >>> print(f"{estimator.point_estimate():.3f} ({lo:.3f}, {hi:.3f})")
    5.640 (5.494, 5.786)

The calibrated estimate moves toward the population mean, and its
interval covers it. The :doc:`user_guides/usage` covers joint
calibration of treatment and control groups, treatment effects,
ratios, linear combinations of estimators, and sensitivity analysis;
``pyrake.usage()`` returns the same guide.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user_guides/index
   api/index
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
