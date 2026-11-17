"""Population estimators."""

from __future__ import annotations

import math
from collections.abc import Generator
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.optimize import linprog

from pyrake.estimation.base_classes import (
    Estimand,
    SimpleEstimand,
    WeightingEstimator,
)
from pyrake.estimation.sensitivity_solvers import (
    solve_bounded_linear_fractional_program,
)


class PopulationMean(SimpleEstimand):
    """Mean over the full target population.

    Parameters
    ----------
     propensity_scores : (M,) ``npt.NDArray[np.float64]``
        Estimated response propensity for each of the ``M`` sample members.

    """

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Calculate weights based on propensity scores.

        Returns
        -------
         weights : (M,) ``npt.NDArray[np.float64]``
            Inverse propensity weight for each of the ``M`` sample members.

        """
        return np.ones_like(self.propensity_scores) / self.propensity_scores

    def sensitivity_region(
        self,
        gamma: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate a sensitivity region around baseline weights.

        Parameters
        ----------
         gamma : float
            The Gamma factor. Must be >= 1.0.

        Returns
        -------
         lower, upper : (M,) ``npt.NDArray[np.float64]``
            Elementwise lower and upper weight bounds.

        """
        weights = self.weights
        wl = weights / math.sqrt(gamma) + (1.0 - 1.0 / math.sqrt(gamma))
        wu = weights * math.sqrt(gamma) - (math.sqrt(gamma) - 1.0)
        return wl, wu

    @staticmethod
    def normalizing_factor(sample_size: int, population_size: int) -> float:
        """Calculate normalizing factor.

        Parameters
        ----------
         sample_size : int
            Number of units in the sample.
         population_size : int
            Number of units in the target population.

        Returns
        -------
         factor : float
            Size of the target population.

        """
        return population_size


class NonRespondentMean(SimpleEstimand):
    """Mean among everyone who did not respond.

    The non-respondent mean is the mean response among everyone who did *not* respond.
    The weights for this estimand are (1 - π) / π, where π are the propensity scores.

    Parameters
    ----------
     propensity_scores : (M,) ``npt.NDArray[np.float64]``
        Estimated response propensity for each of the ``M`` sample members.

    """

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Calculate weights based on propensity scores.

        Returns
        -------
         weights : (M,) ``npt.NDArray[np.float64]``
            Non-response weight for each of the ``M`` sample members.

        """
        return (
            np.ones_like(self.propensity_scores) - self.propensity_scores
        ) / self.propensity_scores

    def sensitivity_region(
        self,
        gamma: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate a sensitivity region around baseline weights.

        Parameters
        ----------
         gamma : float
            The Gamma factor. Must be >= 1.0.

        Returns
        -------
         lower, upper : (M,) ``npt.NDArray[np.float64]``
            Elementwise lower and upper weight bounds.

        """
        weights = self.weights
        wl = weights / math.sqrt(gamma)
        wu = weights * math.sqrt(gamma)
        return wl, wu

    @staticmethod
    def normalizing_factor(sample_size: int, population_size: int) -> float:
        """Calculate normalizing factor.

        Parameters
        ----------
         sample_size : int
            Number of units in the sample.
         population_size : int
            Number of units in the target population.

        Returns
        -------
         factor : float
            Number of non-respondents in the target population.

        """
        return population_size - sample_size


class SampleMean(SimpleEstimand):
    """Mean among the population of respondents.

    The sample mean is the mean response among the population of respondents. The
    weights are simply 1, and thus involve no uncertainty.

    Parameters
    ----------
     propensity_scores : (M,) ``npt.NDArray[np.float64]``
        Estimated response propensity for each of the ``M`` sample members.

    """

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Calculate weights based on propensity scores.

        Returns
        -------
         weights : (M,) ``npt.NDArray[np.float64]``
            Unit weight for each of the ``M`` sample members.

        """
        return np.ones_like(self.propensity_scores)

    def sensitivity_region(
        self,
        gamma: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate a sensitivity region around baseline weights.

        Parameters
        ----------
         gamma : float
            The Gamma factor. Must be >= 1.0.

        Returns
        -------
         lower, upper : (M,) ``npt.NDArray[np.float64]``
            Elementwise lower and upper weight bounds.

        """
        weights = self.weights
        return weights, weights

    @staticmethod
    def normalizing_factor(sample_size: int, population_size: int) -> float:
        """Calculate normalizing factor.

        Parameters
        ----------
         sample_size : int
            Number of units in the sample.
         population_size : int
            Number of units in the target population.

        Returns
        -------
         factor : float
            Number of units in the sample.

        """
        return sample_size


class MeanEstimator(WeightingEstimator):
    """Base class for estimating a mean in some population.

    Parameters
    ----------
     propensity_scores : list_like
        Propensity scores.
     outcomes : list_like
        Outcomes or responses. Should have the same length as ``propensity_scores``.
     estimand : Estimand, optional
        An ``Estimand`` object representing the thing to be estimated. Defaults to
        ``PopulationMean``.
     estimand_class : type, optional
        Estimand class to construct from ``propensity_scores`` when ``estimand`` is not
        given. Defaults to ``PopulationMean``.
     kwargs : dict, optional
        Extra arguments forwarded to the estimator constructor.

    """

    def __init__(
        self,
        propensity_scores: list[float] | npt.NDArray[np.float64],
        outcomes: list[float | int] | npt.NDArray[np.float64 | np.int64],
        estimand: Estimand | None = None,
        estimand_class: type[SimpleEstimand] | None = None,
        **kwargs: Any,
    ) -> None:
        if len(propensity_scores) != len(outcomes):
            raise ValueError(
                "Must have equal numbers of propensity scores and outcomes."
            )

        if np.min(propensity_scores) <= 0.0 or np.max(propensity_scores) > 1.0:
            raise ValueError("Propensitiy scores must be strictly positive and <= 1")

        self.propensity_scores: npt.NDArray[np.float64] = np.asarray(propensity_scores)
        if estimand is None:
            estimand = (estimand_class or PopulationMean)(self.propensity_scores)
        self.estimand: Estimand = estimand
        self.weights: npt.NDArray[np.float64] = self.estimand.weights
        self.outcomes: npt.NDArray[np.float64] = np.asarray(outcomes)
        self.extra_args: dict[str, Any] = kwargs

    def min_ess_bound(
        self,
        gamma: float,
        method: Literal["floor", "exact"] = "floor",
    ) -> float:
        """Worst-case effective sample size over the Gamma sensitivity region.

        Returns the smallest Kish effective sample size

           ESS(w) = (1ᵀw)²/‖w‖²

        attainable anywhere in the Gamma sensitivity region [``wl``, ``wu``]. The Kish
        effective sample size measures how evenly the weights spread the estimate's
        leverage; the normal approximation behind the sensitivity bounds leans on its
        being large. Its worst case over the region is a ratio with a rank-one numerator
        over a separable denominator, so the minimizer is a vertex of the box: heavy
        units raised to ``wu`` and light units lowered to ``wl``, the arrangement that
        most concentrates the weights. Because both ``wl`` and ``wu`` increase with the
        nominal weight, the optimal split is a threshold on the sorted weights, found by
        sweeping all thresholds in a single pass (``method="exact"``).

        A coarser bound needs no search at all, taking the numerator at the region's
        lowest corner and the denominator at its highest (``method="floor"``):

           min over wl ≤ w ≤ wu of ESS(w) ≥ (Σᵢ wlᵢ)²/Σᵢ wuᵢ².

        Both are computable before any outcome is examined. At gamma=1 the region
        collapses to a point and either method returns the exact effective sample size
        of the nominal weights; as gamma grows the region widens and the value
        decreases.

        Parameters
        ----------
         gamma : float
            The Gamma factor. Must be >= 1.0. See ``sensitivity_analysis``.
         method : ["floor", "exact"], optional
            "floor" (default) returns the conservative closed-form lower bound; "exact"
            returns the actual worst-case minimum via the threshold sweep.

        Returns
        -------
         ess : float
            The worst-case effective sample size over the region (a lower bound when
            ``method="floor"``).

        Raises
        ------
         ValueError
            If ``gamma`` < 1 or ``method`` is not recognized.

        References
        ----------
         - Wilson, Bob. 2025. "Design-Based Inference and Sensitivity Analysis for
           Survey Sampling." https://adventuresinwhy.com/post/survey_sampling/

        """
        if gamma < 1.0:
            raise ValueError("`gamma` must be >= 1")

        wl, wu = self.estimand.sensitivity_region(gamma)
        # Orient the box: sub-unit nominal weights can invert the affine bounds.
        w_lo = np.minimum(wl, wu)
        w_hi = np.maximum(wl, wu)

        if method == "floor":
            numerator = float(np.sum(np.maximum(w_lo, 0.0))) ** 2
            denominator = float(np.sum(np.square(w_hi)))
            return numerator / denominator
        elif method == "exact":
            # The minimizer sets the heaviest units to their upper bound and the rest
            # to their lower bound. Sort by upper bound (which increases with the
            # nominal weight) and sweep every threshold via prefix sums: for the kth
            # split the top k units sit at ``w_hi`` and the remainder at ``w_lo``.
            order = np.argsort(-w_hi)
            hi = w_hi[order]
            lo = w_lo[order]
            zero = np.zeros(1)
            cum_hi_sum = np.concatenate((zero, np.cumsum(hi)))
            cum_hi_sq = np.concatenate((zero, np.cumsum(np.square(hi))))
            cum_lo_sum = np.concatenate((zero, np.cumsum(lo)))
            cum_lo_sq = np.concatenate((zero, np.cumsum(np.square(lo))))
            s1 = cum_hi_sum + (cum_lo_sum[-1] - cum_lo_sum)
            s2 = cum_hi_sq + (cum_lo_sq[-1] - cum_lo_sq)
            ess = np.where(
                s2 > 0.0, np.square(s1) / np.where(s2 > 0.0, s2, 1.0), np.inf
            )
            return float(np.min(ess))
        else:
            raise ValueError(f"Unrecognized input {method=:}")

    def pvalue(
        self,
        null_value: float,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ) -> float:
        """Calculate a p-value.

        Computes a p-value against the null hypothesis:
            H0: Ȳ = ``null_value``,
        where Ȳ is the mean in the target population.

        Parameters
        ----------
         null_value : float
            The hypothesized mean.
         alternative : ["two-sided", "less", "greater"], optional
            What kind of test:
              - "two-sided": H0: Ȳ = ``null_value`` vs Halt: Ȳ <>
                ``null_value``.
              - "greater": H0: Ȳ <= ``null_value`` vs Halt: Ȳ > ``null_value``.
              - "less": H0: Ȳ >= ``null_value`` vs Halt: Ȳ < ``null_value``.
            For example, specifying alternative = "greater" returns a p-value that
            quantifies the strength of evidence against the null hypothesis that the
            mean is less than ``null_value``, in favor of the alternative hypothesis
            that the mean is actually greater than ``null_value``. Defaults to
            "two-sided".

        Returns
        -------
         p : float
            P-value.

        Raises
        ------
         ValueError
            If ``alternative`` is not recognized (propagated from ``pvalue``).

        """
        return super().pvalue(null_value=null_value, alternative=alternative)

    def confidence_interval(
        self,
        alpha: float = 0.10,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ) -> tuple[float, float]:
        """Calculate confidence interval on mean in target population.

        Parameters
        ----------
         alpha : float, optional
            P-value threshold, e.g. specify alpha=0.05 for a 95% confidence interval.
            Defaults to 0.10, corresponding to a 90% confidence interval.
         alternative : ["two-sided", "less", "greater"], optional
            What kind of test:
              - "two-sided": H0: Ȳ = ``null_value`` vs Halt: Ȳ <>
                ``null_value``.
              - "greater": H0: Ȳ <= ``null_value`` vs Halt: Ȳ > ``null_value``.
              - "less": H0: Ȳ >= ``null_value`` vs Halt: Ȳ < ``null_value``.
            Defaults to "two-sided".

        Returns
        -------
         lb, ub : float
            Lower and upper bounds on the confidence interval. For one-sided intervals,
            only one of these will be finite:
              - "two-sided": lb and ub both finite
              - "greater": lb finite, ub = np.inf
              - "less": lb = -np.inf, ub finite

        Raises
        ------
         ValueError
            If ``alternative`` is not recognized (propagated from
            ``confidence_interval``).

        """
        return super().confidence_interval(alpha=alpha, alternative=alternative)

    def resample(
        self,
        B: int,
        seed: None | (
            int
            | list[int]
            | np.random.SeedSequence
            | np.random.BitGenerator
            | np.random.Generator
        ) = None,
    ) -> Generator[MeanEstimator, None, None]:
        """Yield a sequence of resampled estimators.

        Parameters
        ----------
         B : int
            Number of bootstrap replications to run.
         seed : int, list_like, etc
            A seed for ``numpy.random.default_rng``. See that documentation for details.

        Yields
        ------
         estimator : ``MeanEstimator``
            Bootstrap-resampled estimator.

        """
        rng = np.random.default_rng(seed)
        for _ in range(B):
            # Resample with replacement
            bootstrap_indices = rng.choice(
                range(len(self.weights)),
                size=len(self.weights),
                replace=True,
            )
            bootstrap_propensities = self.propensity_scores[bootstrap_indices]
            bootstrap_outcomes = self.outcomes[bootstrap_indices]
            # Implementation note for AIPW and SAIPW estimators: `self.outcomes` is
            # actually the adjusted outcomes, outcome minus prediction, and
            # `self.extra_args["predicted_outcomes"]` is actually just a vector of all
            # zeros to make the constructor happy. So there is no need to resample those
            # predicted outcomes.

            yield self.__class__(
                propensity_scores=bootstrap_propensities,
                outcomes=bootstrap_outcomes,
                estimand=self.estimand.resample(bootstrap_indices),
                **self.extra_args,
            )


class IPWEstimator(MeanEstimator):
    """Inverse Propensity Weighted Estimator.

    Parameters
    ----------
     propensity_scores : list_like
        Propensity scores.
     outcomes : list_like
        Outcomes or responses. Should have the same length as ``propensity_scores``.
     population_size : int
        Size of target population.
     estimand : Estimand, optional
        An ``Estimand`` object representing the thing to be estimated. Defaults to
        ``PopulationMean``.
     estimand_class : type, optional
        Estimand class to construct from ``propensity_scores`` when ``estimand`` is not
        given. Defaults to ``PopulationMean``.
     kwargs : dict, optional
        Extra arguments forwarded to the estimator constructor.

    Notes
    -----
    The IPW estimator is:

       Ŷ_IPW = (1/N) Σᵢ wᵢyᵢ,

    where wᵢ = 1/πᵢ, N is the target population size, and πᵢ is the propensity score.

    Details on this estimator, including sensitivity analysis and links to academic
    resources, can be found in (Wilson, 2025)

    References
    ----------
     - Wilson, Bob. 2025. "Design-Based Inference and Sensitivity Analysis for
       Survey Sampling." https://adventuresinwhy.com/post/survey_sampling/

    """

    def __init__(
        self,
        propensity_scores: list[float] | npt.NDArray[np.float64],
        outcomes: list[float | int] | npt.NDArray[np.float64 | np.int64],
        population_size: int,
        estimand: Estimand | None = None,
        estimand_class: type[SimpleEstimand] | None = None,
        **kwargs: Any,
    ) -> None:
        if estimand is not None and not isinstance(estimand, PopulationMean):
            # When estimating something other than the PopulationMean, the weights are
            # different, so the point estimate and variance are different. The point
            # estimate is easy, but the variance is tricky. Punting for now.
            raise NotImplementedError(
                "Implementation has not been checked except for PopulationMean"
            )

        if estimand_class is not None and estimand_class != PopulationMean:
            raise NotImplementedError(
                "Implementation has not been checked except for PopulationMean"
            )

        super().__init__(
            propensity_scores=propensity_scores,
            outcomes=outcomes,
            estimand=estimand,
            estimand_class=estimand_class,
            population_size=population_size,
            **kwargs,
        )
        self.population_size = population_size

    def point_estimate(self) -> float:
        """Calculate a point estimate.

        Returns
        -------
         estimate : float
            Inverse-propensity-weighted estimate of the population total divided by the
            normalizing factor.

        """
        nf = self.estimand.normalizing_factor(
            sample_size=len(self.weights), population_size=self.population_size
        )
        return np.dot(self.weights, self.outcomes) / nf

    def variance(self) -> float:
        """Calculate the variance.

        Per (Wilson 2025, §2), an estimator for the variance is:

           Σᵢ(((n/N)/πᵢ) × yᵢ − pe)² / (n × (n − 1)),

        where pe is the point estimate and n is the sample size.

        Returns
        -------
         variance : float
            Estimated sampling variance of the point estimate.

        References
        ----------
         - Wilson, Bob. 2025. "Design-Based Inference and Sensitivity Analysis for
           Survey Sampling." https://adventuresinwhy.com/post/survey_sampling/

        """
        n = len(self.weights)
        n_over_N = n / self.population_size
        return float(
            np.mean(
                np.square(
                    n_over_N * self.weights * self.outcomes - self.point_estimate()
                )
            )
            / (n - 1)
        )

    def _sensitivity_variance(self, gamma: float) -> tuple[float, float]:
        """Return variance at the worst-case lower- and upper-bound weights.

        The lower-bound weights assign each observation the weight from the sensitivity
        region that minimises the estimate (``wl`` when outcomes >= 0, ``wu``
        otherwise), mirroring ``sensitivity_analysis``. The upper-bound weights are the
        element-wise opposite. Variance is then evaluated using the IPW formula at each
        of these weight vectors, implementing the heuristic from (Wilson 2025, §3).

        This implementation is inherited by ``AIPWEstimator``, where ``self.outcomes``
        holds residuals rather than raw outcomes; the formula is identical in both
        cases.

        Parameters
        ----------
         gamma : float
            The Gamma factor. Must be >= 1.0.

        Returns
        -------
         lb_var, ub_var : float
            Variance at the lower- and upper-bound sensitivity weights.

        """
        wl, wu = self.estimand.sensitivity_region(gamma)
        lb_weights = np.where(self.outcomes >= 0, wl, wu)
        ub_weights = np.where(self.outcomes >= 0, wu, wl)
        n = len(lb_weights)
        n_over_N = n / self.population_size
        lb_pe = np.sum(self.outcomes * lb_weights) / self.population_size
        ub_pe = np.sum(self.outcomes * ub_weights) / self.population_size
        lb_var = float(
            np.mean(np.square(n_over_N * lb_weights * self.outcomes - lb_pe)) / (n - 1)
        )
        ub_var = float(
            np.mean(np.square(n_over_N * ub_weights * self.outcomes - ub_pe)) / (n - 1)
        )
        return lb_var, ub_var

    def sensitivity_analysis(
        self,
        gamma: float = 6.0,
        outcome_proxies: list[float] | npt.NDArray[np.float64] | None = None,
        proxy_mean: float | None = None,
        alpha: float = 0.10,
    ) -> tuple[float, float]:
        """Perform a sensitivity analysis.

        Calculate a range of point estimates implied by the Gamma factor.

        Parameters
        ----------
         gamma : float, optional
            The Gamma factor. Must be >= 1.0, with 1.0 indicating perfect propensity
            scores. Defaults to 6. See Notes.
         outcome_proxies : array_like, optional
            A proxy for the outcome, observed for every unit in the sample. When
            supplied together with ``proxy_mean``, an additional constraint is added to
            the sensitivity region: the IPW estimate of the proxy mean must lie within
            ``tol`` of ``proxy_mean``, where ``tol`` is the half-width from
            ``confidence_interval``. Weights that violate this constraint are excluded
            from the sensitivity region, typically reducing the width of the sensitivity
            interval. Must have the same length as ``outcomes``. Requires ``proxy_mean``
            to also be specified.
         proxy_mean : float, optional
            Known population mean of the outcome proxy. Requires ``outcome_proxies`` to
            also be specified.
         alpha : float, optional
            Significance level used to compute the proxy tolerance ``tol`` via
            ``scipy.stats.norm.isf`` and the point-estimate variance. Defaults to 0.10,
            matching the default of ``confidence_interval``.

        Returns
        -------
         lb, ub : float
            Lower and upper bounds on the point estimate.

        Raises
        ------
         ValueError
            If only one of ``outcome_proxies`` and ``proxy_mean`` is supplied, if
            ``gamma`` < 1, or if the proxy constraints admit no feasible weights.

        Notes
        -----
        See the Notes in ``WeightingEstimator`` for overall context. Without an outcome
        proxy there is a simple closed-form calculation for the sensitivity interval.
        See (Wilson 2025) for details. When an outcome proxy is supplied, the
        constrained optimisation is solved as a linear programme via
        ``scipy.optimize.linprog``.

        References
        ----------
         - Wilson, Bob. 2025. "Design-Based Inference and Sensitivity Analysis for
           Survey Sampling." https://adventuresinwhy.com/post/survey_sampling/

        """
        if (outcome_proxies is None) != (proxy_mean is None):
            raise ValueError(
                "`outcome_proxies` and `proxy_mean` must be provided together."
            )

        if gamma < 1.0:
            raise ValueError("`gamma` must be >= 1")

        if gamma == 1.0:
            return self.point_estimate(), self.point_estimate()

        wl, wu = self.estimand.sensitivity_region(gamma)

        if outcome_proxies is None:
            # Closed-form: element-wise weight selection.
            lb = np.where(self.outcomes >= 0, wl, wu)
            ub = np.where(self.outcomes >= 0, wu, wl)
            return (
                np.sum(self.outcomes * lb) / self.population_size,
                np.sum(self.outcomes * ub) / self.population_size,
            )

        # LP formulation with proxy constraint.
        p = np.asarray(outcome_proxies, dtype=float)
        assert proxy_mean is not None  # Satisfy type-checker
        pm = float(proxy_mean)
        zcrit = stats.norm.isf(alpha / 2)
        tol = zcrit * math.sqrt(self.variance())
        N = self.population_size

        # Proxy constraint: N*(pm - tol) <= p^T w <= N*(pm + tol)
        A_ub = np.vstack([p, -p])
        b_ub = np.array([N * (pm + tol), -N * (pm - tol)])
        bounds = list(zip(wl.tolist(), wu.tolist(), strict=True))

        lb_res = linprog(
            c=self.outcomes, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs"
        )
        ub_res = linprog(
            c=-self.outcomes, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs"
        )

        if lb_res.status != 0 or ub_res.status != 0:
            raise ValueError(
                "Proxy constraints are infeasible: no weight vector in the "
                "sensitivity region is consistent with the supplied proxy_mean."
            )

        return float(lb_res.fun) / N, float(-ub_res.fun) / N


class AIPWEstimator(IPWEstimator):
    """Augmented Inverse Propensity Weighted Estimator.

    Parameters
    ----------
     propensity_scores : list_like
        Propensity scores.
     outcomes : list_like
        Outcomes or responses. Should have the same length as ``propensity_scores``.
     predicted_outcomes : list_like
        Predicted outcomes corresponding to each response. Should have the same length
        as ``propensity_scores``.
     mean_predicted_outcome : float
        Mean of predicted outcomes in target population.
     population_size : int
        Size of target population.
     estimand : Estimand, optional
        An ``Estimand`` object representing the thing to be estimated. Defaults to
        ``PopulationMean``.
     estimand_class : type, optional
        Estimand class to construct from ``propensity_scores`` when ``estimand`` is not
        given. Defaults to ``PopulationMean``.
     kwargs : dict, optional
        Extra arguments forwarded to the estimator constructor.

    Notes
    -----
    The AIPW estimator for a mean is:

       Ŷ_AIPW = μ̄ + (1/N) Σᵢ wᵢ × (yᵢ − μ(xᵢ)),

    where wᵢ = 1/πᵢ, N is the target population size, and πᵢ is the propensity score (as
    in ``IPWEstimator``). μ(xᵢ) is a *prediction* of the outcome, yᵢ, for individual i,
    made on the basis of covariates, x, observed for everyone in the target population
    (including everyone in the sample), and μ̄ is the average predicted outcome in the
    target population. (Since we observe x for everyone in the target population, we can
    predict the outcome for everyone in the population, and μ̄ is the average of these
    predictions.)

    The intuition is, if we have a good model for the outcome, we can use it to predict
    the target population mean directly, then adjust this prediction with an estimator
    based on the residuals in the sample. Details on this estimator, including
    sensitivity analysis and links to academic resources, can be found in (Wilson, 2025)

    References
    ----------
     - Wilson, Bob. 2025. "Design-Based Inference and Sensitivity Analysis for
       Survey Sampling." https://adventuresinwhy.com/post/survey_sampling/

    """

    def __init__(
        self,
        propensity_scores: list[float] | npt.NDArray[np.float64],
        outcomes: list[float | int] | npt.NDArray[np.float64 | np.int64],
        predicted_outcomes: list[float] | npt.NDArray[np.float64],
        mean_predicted_outcome: float,
        population_size: int,
        estimand: Estimand | None = None,
        estimand_class: type[SimpleEstimand] | None = None,
        **kwargs: Any,
    ) -> None:
        if len(propensity_scores) != len(predicted_outcomes):
            raise ValueError(
                "Must have equal numbers of propensity scores and predicted outcomes."
            )

        super().__init__(
            propensity_scores=propensity_scores,
            # Here, the outcomes have the interpretation of residuals, actual outcome minus
            # predicted outcome. With good predictions, these residuals tend to be smaller
            # than the raw outcomes, leading to smaller variance and lower sensitivity to
            # hidden biases.
            outcomes=np.asarray(outcomes) - np.asarray(predicted_outcomes),
            estimand=estimand,
            estimand_class=estimand_class,
            population_size=population_size,
            predicted_outcomes=np.zeros_like(np.asarray(propensity_scores)),
            mean_predicted_outcome=mean_predicted_outcome,
        )

        self.mean_predicted_outcome = mean_predicted_outcome

    def point_estimate(self) -> float:
        """Calculate a point estimate.

        Returns
        -------
         estimate : float
            Mean predicted outcome plus the IPW estimate of the mean residual.

        """
        return self.mean_predicted_outcome + super().point_estimate()

    def variance(self) -> float:
        """Calculate the variance.

        Per (Wilson 2025, §2), an estimator for the variance is:

           Σᵢ(wᵢ × yᵢ − pe)² / (n × (n − 1)),

        where pe is the point estimate and n is the sample size.

        Returns
        -------
         variance : float
            Estimated sampling variance of the point estimate.

        References
        ----------
         - Wilson, Bob. 2025. "Design-Based Inference and Sensitivity Analysis for
           Survey Sampling." https://adventuresinwhy.com/post/survey_sampling/

        """
        n_over_N = len(self.weights) / self.population_size
        return float(
            np.mean(
                np.square(
                    n_over_N * self.weights * self.outcomes - super().point_estimate()
                )
            )
            / (len(self.weights) - 1)
        )

    def sensitivity_analysis(
        self,
        gamma: float = 6.0,
        outcome_proxies: list[float] | npt.NDArray[np.float64] | None = None,
        proxy_mean: float | None = None,
        alpha: float = 0.10,
    ) -> tuple[float, float]:
        """Perform a sensitivity analysis.

        Calculate a range of point estimates implied by the Gamma factor.

        Parameters
        ----------
         gamma : float, optional
            The Gamma factor. Must be >= 1.0, with 1.0 indicating perfect propensity
            scores. Defaults to 6. See Notes.
         outcome_proxies : array_like, optional
            A proxy for the outcome with a known population mean. See
            ``IPWEstimator.sensitivity_analysis`` for details. The proxy constraint
            applies to the same (AIPW) weights.
         proxy_mean : float, optional
            Known population mean of the outcome proxy.
         alpha : float, optional
            Significance level for the proxy tolerance. Defaults to 0.10.

        Returns
        -------
         lb, ub : float
            Lower and upper bounds on the point estimate.

        Raises
        ------
         ValueError
            If ``gamma`` < 1, if only one of ``outcome_proxies`` and ``proxy_mean`` is
            supplied, or if the proxy constraints admit no feasible weights (propagated
            from ``IPWEstimator.sensitivity_analysis``).

        Notes
        -----
        See the Notes in ``WeightingEstimator`` for overall context. In our case, there
        is a simple closed-form calculation for the sensitivity interval. See (Wilson
        2025) for details.

        References
        ----------
         - Wilson, Bob. 2025. "Design-Based Inference and Sensitivity Analysis for
           Survey Sampling." https://adventuresinwhy.com/post/survey_sampling/

        """
        if gamma < 1.0:
            raise ValueError("`gamma` must be >= 1")

        if gamma == 1.0:
            return self.point_estimate(), self.point_estimate()

        lb, ub = super().sensitivity_analysis(
            gamma=gamma,
            outcome_proxies=outcome_proxies,
            proxy_mean=proxy_mean,
            alpha=alpha,
        )
        return (
            self.mean_predicted_outcome + lb,
            self.mean_predicted_outcome + ub,
        )


class SIPWEstimator(MeanEstimator):
    """Stabilized Inverse Propensity Weighted Estimator.

    Parameters
    ----------
     propensity_scores : list_like
        Propensity scores.
     outcomes : list_like
        Outcomes or responses. Should have the same length as ``propensity_scores``.
     binary_outcomes : bool, optional
        Indicates whether the outcomes are binary. If ``outcomes`` contains values other
        than 0 and 1, we conclude the outcomes are *not* binary; if all values are 0 and
        1 we conclude outcomes *are* binary. The only place this matters is in the
        sensitivity analysis, which is faster for binary outcomes than in general.
     estimand : Estimand, optional
        An ``Estimand`` object representing the thing to be estimated. Defaults to
        ``PopulationMean``.
     estimand_class : type, optional
        Estimand class to construct from ``propensity_scores`` when ``estimand`` is not
        given. Defaults to ``PopulationMean``.
     kwargs : dict, optional
        Extra arguments forwarded to the estimator constructor.

    Notes
    -----
    The SIPW estimator is:

       Ŷ_SIPW = Σᵢ wᵢyᵢ / Σᵢ wᵢ.

    This is literally the first thing people think of when they hear "weighted average",
    fancy name notwithstanding. The weights are the inverse propensity scores. Since we
    divide by the sum of the weights, we can scale them by an arbitrary multiplicative
    constant without changing point estimates or p-values, but this *does* affect
    sensitivity analysis, so it is best to use unscaled weights. More details about this
    estimator, including its variance, may be found in (Sarig, Galili, and Eilat 2023).

    For details on sensitivity analysis for this estimator, see (Zhao, Small, and
    Bhattarcharya 2019).

    References
    ----------
     - Sarig, Tal, Tal Galili, and Roee Eilat. "balance--a Python package for balancing
       biased data samples." arXiv preprint arXiv:2307.06024 (2023).
     - Zhao, Qingyuan, Dylan S Small, and Bhaswar B Bhattacharya. 2019. "Sensitivity
       Analysis for Inverse Probability Weighting Estimators via the Percentile
       Bootstrap." Journal of the Royal Statistical Society Series B: Statistical
       Methodology 81 (4). Oxford University Press: pg. 735--61.

    """

    def __init__(
        self,
        propensity_scores: list[float] | npt.NDArray[np.float64],
        outcomes: list[float | int] | npt.NDArray[np.float64 | np.int64],
        estimand: Estimand | None = None,
        estimand_class: type[SimpleEstimand] | None = None,
        binary_outcomes: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            propensity_scores=propensity_scores,
            outcomes=outcomes,
            estimand=estimand,
            estimand_class=estimand_class,
            **kwargs,
        )

        # Default to binary outcomes unless `outcomes` includes other values, or the
        # user specifically says otherwise.
        if binary_outcomes and np.all(np.isin(outcomes, [0, 1])):
            self.binary_outcomes: bool = True
        else:
            self.binary_outcomes = False

    def point_estimate(self) -> float:
        """Calculate a point estimate.

        Returns
        -------
         estimate : float
            Self-normalized weighted mean of the outcomes.

        """
        return np.dot(self.weights, self.outcomes) / np.sum(self.weights)

    def variance(self) -> float:
        """Calculate the variance.

        Per (Sarig, Galili, Eilat 2023, Appendix D.2), an estimator for the variance is:

           Σᵢ wᵢ² × (yᵢ − pe)² / (Σᵢ wᵢ)²,

        where pe is the point estimate.

        Returns
        -------
         variance : float
            Estimated sampling variance of the point estimate.

        """
        return np.sum(
            np.square(self.weights) * np.square(self.outcomes - self.point_estimate())
        ) / (np.sum(self.weights) ** 2)

    def _sensitivity_weights(
        self, gamma: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return weight vectors that achieve the sensitivity lower and upper bounds.

        For binary outcomes the optimal weights are determined element-wise:
        observations with outcome 1 receive ``wl`` for the lower bound and ``wu`` for
        the upper bound; observations with outcome 0 receive the reverse.

        For non-binary outcomes, (Zhao, Small, and Bhattarcharya 2019) calculates

           inf Σᵢwᵢyᵢ/Σᵢwᵢ

        over wlᵢ ≤ wᵢ ≤ wuᵢ as the lower bound, and the supremum as the upper bound.
        They propose a fast algorithm requiring a single pass over the sorted data. For
        the lower bound, observations are processed in ascending outcome order: flipping
        each weight from ``wl`` to ``wu`` increases the denominator faster than the
        numerator (low outcome * high weight), decreasing the ratio. The algorithm stops
        at the first flip that fails to improve. The cutoff index records how many
        observations were flipped at the optimum, and the weight vector is reconstructed
        from it. The upper bound uses the same approach in descending outcome order.

        Parameters
        ----------
         gamma : float
            The Gamma factor. Must be >= 1.0.

        Returns
        -------
         lb_weights, ub_weights : (M,) ``npt.NDArray[np.float64]``
            Weight vectors attaining the lower and upper sensitivity bounds.

        References
        ----------
         - Zhao, Qingyuan, Dylan S Small, and Bhaswar B Bhattacharya. 2019.
           "Sensitivity Analysis for Inverse Probability Weighting Estimators
           via the Percentile Bootstrap." Journal of the Royal Statistical
           Society Series B: Statistical Methodology 81 (4): 735--61.

        """
        wl, wu = self.estimand.sensitivity_region(gamma)

        if self.binary_outcomes:
            return (
                np.where(self.outcomes == 1, wl, wu),
                np.where(self.outcomes == 1, wu, wl),
            )

        # Non-binary lower bound: process observations in ascending outcome
        # order, flipping each weight from wl to wu while the ratio decreases.
        idx_asc = np.argsort(self.outcomes)
        w = np.array(wl)
        zeta = np.sum(w)
        lmbda = np.dot(self.outcomes, w)
        lb_star = lmbda / zeta
        lb_cutoff = 0  # number of observations flipped at optimum
        for ii in range(len(self.outcomes)):
            w[idx_asc[ii]] = wu[idx_asc[ii]]
            zeta += wu[idx_asc[ii]] - wl[idx_asc[ii]]
            lmbda += self.outcomes[idx_asc[ii]] * (wu[idx_asc[ii]] - wl[idx_asc[ii]])
            if lmbda < lb_star * zeta:
                lb_star = lmbda / zeta
                lb_cutoff = ii + 1
            else:
                break
        lb_weights = np.array(wl)
        lb_weights[idx_asc[:lb_cutoff]] = wu[idx_asc[:lb_cutoff]]

        # Non-binary upper bound: process observations in descending outcome
        # order, flipping each weight from wl to wu while the ratio increases.
        idx_desc = idx_asc[::-1]
        w = np.array(wl)
        zeta = np.sum(w)
        lmbda = np.dot(self.outcomes, w)
        ub_star = lmbda / zeta
        ub_cutoff = 0
        for ii in range(len(self.outcomes)):
            w[idx_desc[ii]] = wu[idx_desc[ii]]
            zeta += wu[idx_desc[ii]] - wl[idx_desc[ii]]
            lmbda += self.outcomes[idx_desc[ii]] * (wu[idx_desc[ii]] - wl[idx_desc[ii]])
            if lmbda > ub_star * zeta:
                ub_star = lmbda / zeta
                ub_cutoff = ii + 1
            else:
                break
        ub_weights = np.array(wl)
        ub_weights[idx_desc[:ub_cutoff]] = wu[idx_desc[:ub_cutoff]]

        return lb_weights, ub_weights

    def _sensitivity_variance(self, gamma: float) -> tuple[float, float]:
        """Return variance at the worst-case lower- and upper-bound weights.

        Calls ``_sensitivity_weights`` to obtain the weight vectors, then evaluates the
        SIPW variance formula at each, implementing the heuristic from (Wilson 2025,
        §3).

        This implementation is inherited by ``SAIPWEstimator``, where ``self.outcomes``
        holds residuals; the formula is identical.

        Parameters
        ----------
         gamma : float
            The Gamma factor. Must be >= 1.0.

        Returns
        -------
         lb_var, ub_var : float
            Variance at the lower- and upper-bound sensitivity weights.

        """
        lb_weights, ub_weights = self._sensitivity_weights(gamma)
        lb_pe = np.dot(self.outcomes, lb_weights) / np.sum(lb_weights)
        ub_pe = np.dot(self.outcomes, ub_weights) / np.sum(ub_weights)
        lb_var = float(
            np.sum(np.square(lb_weights) * np.square(self.outcomes - lb_pe))
            / np.sum(lb_weights) ** 2
        )
        ub_var = float(
            np.sum(np.square(ub_weights) * np.square(self.outcomes - ub_pe))
            / np.sum(ub_weights) ** 2
        )
        return lb_var, ub_var

    def sensitivity_analysis(
        self,
        gamma: float = 6.0,
        outcome_proxies: list[float] | npt.NDArray[np.float64] | None = None,
        proxy_mean: float | None = None,
        alpha: float = 0.10,
    ) -> tuple[float, float]:
        """Perform a sensitivity analysis.

        Calculate a range of point estimates implied by the Gamma factor. See
        ``_sensitivity_weights`` for the unconstrained algorithm.

        Parameters
        ----------
         gamma : float, optional
            The Gamma factor. Must be >= 1.0, with 1.0 indicating perfect propensity
            scores. Defaults to 6. See Notes in ``WeightingEstimator``.
         outcome_proxies : array_like, optional
            A proxy for the outcome with a known population mean. When supplied together
            with ``proxy_mean``, an additional constraint is added to the sensitivity
            region: the SIPW estimate of the proxy mean must lie within ``tol`` of
            ``proxy_mean``, where ``tol`` is the half-width from
            ``confidence_interval``. For the SIPW estimator this is ``(wᵀproxy)/(wᵀ1)``
            in [pm − tol, pm + tol], which is linear after the Charnes-Cooper
            transformation. Weights that violate this constraint are excluded from the
            sensitivity region, typically reducing the width of the sensitivity
            interval. Must have the same length as ``outcomes``. Requires ``proxy_mean``
            to also be specified.
         proxy_mean : float, optional
            Known population mean of the outcome proxy. Requires ``outcome_proxies`` to
            also be specified.
         alpha : float, optional
            Significance level for the proxy tolerance ``tol`` via
            ``scipy.stats.norm.isf`` and the point-estimate variance. Defaults to 0.10,
            matching the default of ``confidence_interval``.

        Returns
        -------
         lb, ub : float
            Lower and upper bounds on the point estimate.

        Raises
        ------
         ValueError
            If only one of ``outcome_proxies`` and ``proxy_mean`` is supplied, if
            ``gamma`` < 1, or if the proxy constraints admit no feasible weights.

        """
        if (outcome_proxies is None) != (proxy_mean is None):
            raise ValueError(
                "`outcome_proxies` and `proxy_mean` must be provided together."
            )

        if gamma < 1.0:
            raise ValueError("`gamma` must be >= 1")

        if gamma == 1.0:
            return self.point_estimate(), self.point_estimate()

        if outcome_proxies is None:
            lb_weights, ub_weights = self._sensitivity_weights(gamma)
            return (
                np.dot(self.outcomes, lb_weights) / np.sum(lb_weights),
                np.dot(self.outcomes, ub_weights) / np.sum(ub_weights),
            )

        # Charnes-Cooper LP with proxy constraint.
        wl, wu = self.estimand.sensitivity_region(gamma)
        p = np.asarray(outcome_proxies, dtype=float)
        assert proxy_mean is not None  # Satisfy type-checker
        pm = float(proxy_mean)
        zcrit = stats.norm.isf(alpha / 2)
        tol = zcrit * math.sqrt(self.variance())
        n = len(self.outcomes)

        # Variables: x = [w_cc (n), s (1)]
        # Objective: outcomes^T w_cc  (s has zero coefficient)
        c_obj = np.append(self.outcomes, 0.0)

        # Equality: sum(w_cc) = 1  ->  [ones; 0] @ x = 1
        A_eq = np.append(np.ones(n), 0.0).reshape(1, n + 1)
        b_eq = np.array([1.0])

        # Box inequalities: -w_cc + wl*s <= 0,  w_cc - wu*s <= 0
        top = np.hstack([-np.eye(n), wl.reshape(-1, 1)])
        bottom = np.hstack([np.eye(n), -wu.reshape(-1, 1)])

        # Proxy constraints (after CC substitution; valid since sum(w_cc)=1):
        # p^T w_cc <= pm + tol  ->  (p - (pm+tol))^T w_cc <= 0
        # p^T w_cc >= pm - tol  ->  ((pm-tol) - p)^T w_cc <= 0
        proxy_upper = np.append(p - (pm + tol), 0.0).reshape(1, n + 1)
        proxy_lower = np.append((pm - tol) - p, 0.0).reshape(1, n + 1)

        A_ub = np.vstack([top, bottom, proxy_upper, proxy_lower])
        b_ub = np.zeros(2 * n + 2)

        # Bounds: w_cc unconstrained (enforced by A_ub), s >= 0
        bounds = [(-np.inf, np.inf)] * n + [(0.0, np.inf)]

        lb_res = linprog(
            c=c_obj,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        ub_res = linprog(
            c=-c_obj,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if lb_res.status != 0 or ub_res.status != 0:
            raise ValueError(
                "Proxy constraints are infeasible: no weight vector in the "
                "sensitivity region is consistent with the supplied proxy_mean."
            )

        return float(lb_res.fun), float(-ub_res.fun)


class SAIPWEstimator(SIPWEstimator):
    """Stabilized Augmented Inverse Propensity Weighted Estimator.

    Parameters
    ----------
     propensity_scores : list_like
        Propensity scores.
     outcomes : list_like
        Outcomes or responses. Should have the same length as ``propensity_scores``.
     predicted_outcomes : list_like
        Predicted outcomes corresponding to each response. Should have the same length
        as ``propensity_scores``.
     mean_predicted_outcome : float
        Mean of predicted outcomes in target population.
     binary_outcomes : bool, optional
        Indicates whether the outcomes are binary. If ``outcomes`` contains values other
        than 0 and 1, we conclude the outcomes are *not* binary; if all values are 0 and
        1 we conclude outcomes *are* binary. The only place this matters is in the
        sensitivity analysis, which is faster for binary outcomes than in general.
     estimand : Estimand, optional
        An ``Estimand`` object representing the thing to be estimated. Defaults to
        ``PopulationMean``.
     estimand_class : type, optional
        Estimand class to construct from ``propensity_scores`` when ``estimand`` is not
        given. Defaults to ``PopulationMean``.
     kwargs : dict, optional
        Extra arguments forwarded to the estimator constructor.

    Notes
    -----
    The SAIPW estimator augments the SIPW estimator with a model for the outcome, just
    as the AIPW estimator does. See notes from those two estimators.

    """

    def __init__(
        self,
        propensity_scores: list[float] | npt.NDArray[np.float64],
        outcomes: list[float | int] | npt.NDArray[np.float64 | np.int64],
        predicted_outcomes: list[float | int] | npt.NDArray[np.float64 | np.int64],
        mean_predicted_outcome: float,
        estimand: Estimand | None = None,
        estimand_class: type[SimpleEstimand] | None = None,
        binary_outcomes: bool = True,
        **kwargs: Any,
    ) -> None:
        if len(propensity_scores) != len(predicted_outcomes):
            raise ValueError(
                "Must have equal numbers of propensity scores and predicted outcomes."
            )

        super().__init__(
            propensity_scores=propensity_scores,
            outcomes=np.asarray(outcomes) - np.asarray(predicted_outcomes),
            estimand=estimand,
            estimand_class=estimand_class,
            # Even with binary outcomes, the adjusted outcomes are not binary.
            # Because of this, sensitivity analysis is slower!
            binary_outcomes=False,
            predicted_outcomes=np.zeros_like(np.asarray(propensity_scores)),
            mean_predicted_outcome=mean_predicted_outcome,
        )

        self.mean_predicted_outcome = mean_predicted_outcome

    def point_estimate(self) -> float:
        """Calculate a point estimate.

        Returns
        -------
         estimate : float
            Mean predicted outcome plus the SIPW estimate of the mean residual.

        """
        return self.mean_predicted_outcome + super().point_estimate()

    def variance(self) -> float:
        """Calculate the variance.

        Per (Sarig, Galili, Eilat 2023, Appendix D.2), an estimator for the variance is:

           Σᵢ wᵢ² × (yᵢ − pe)² / (Σᵢ wᵢ)²,

        where pe is the point estimate.

        Returns
        -------
         variance : float
            Estimated sampling variance of the point estimate.

        """
        return np.sum(
            np.square(self.weights)
            * np.square(self.outcomes - super().point_estimate())
        ) / (np.sum(self.weights) ** 2)

    def sensitivity_analysis(
        self,
        gamma: float = 6.0,
        outcome_proxies: list[float] | npt.NDArray[np.float64] | None = None,
        proxy_mean: float | None = None,
        alpha: float = 0.10,
    ) -> tuple[float, float]:
        """Perform a sensitivity analysis.

        Calculate a range of point estimates implied by the Gamma factor.

        Parameters
        ----------
         gamma : float, optional
            The Gamma factor. Must be >= 1.0, with 1.0 indicating perfect propensity
            scores. Defaults to 6. See Notes.
         outcome_proxies : array_like, optional
            A proxy for the outcome with a known population mean. See
            ``SIPWEstimator.sensitivity_analysis`` for details. The proxy constraint
            applies to the same (SAIPW) weights.
         proxy_mean : float, optional
            Known population mean of the outcome proxy.
         alpha : float, optional
            Significance level for the proxy tolerance. Defaults to 0.10.

        Returns
        -------
         lb, ub : float
            Lower and upper bounds on the point estimate.

        Raises
        ------
         ValueError
            If ``gamma`` < 1, if only one of ``outcome_proxies`` and ``proxy_mean`` is
            supplied, or if the proxy constraints admit no feasible weights (propagated
            from ``SIPWEstimator.sensitivity_analysis``).

        Notes
        -----
        See the Notes in ``WeightingEstimator`` and ``SIPWEstimator`` for overall
        context.

        """
        if gamma < 1.0:
            raise ValueError("`gamma` must be >= 1")

        if gamma == 1.0:
            return self.point_estimate(), self.point_estimate()

        lb, ub = super().sensitivity_analysis(
            gamma=gamma,
            outcome_proxies=outcome_proxies,
            proxy_mean=proxy_mean,
            alpha=alpha,
        )
        return (
            self.mean_predicted_outcome + lb,
            self.mean_predicted_outcome + ub,
        )


class RatioEstimator(WeightingEstimator):
    """Estimate the ratio of two means in some population.

    Parameters
    ----------
     propensity_scores : list_like
        Propensity scores.
     numerator_outcomes, denominator_outcomes : list_like
        Outcomes or responses corresponding to numerator and denominator. Should have
        the same length as ``propensity_scores``.
     estimand : Estimand, optional
        An ``Estimand`` object representing the thing to be estimated. Defaults to
        ``PopulationMean``.
     estimand_class : type, optional
        Estimand class to construct from ``propensity_scores`` when ``estimand`` is not
        given. Defaults to ``PopulationMean``.
     estimator_class : ["IPW", "AIPW", "SIPW", "SAIPW"], optional
        What kind of estimator to use. Defaults to "SIPW". Currently only "SIPW" is
        supported.
     numerator_estimator_class, denominator_estimator_class: ["IPW", "AIPW", "SIPW", "SAIPW"], optional
        If you really want to use different estimator classes for numerator vs
        denominator, you can. Defaults to ``estimator_class``. Currently only "SIPW" is
        supported.
     numerator_kwargs, denominator_kwargs, kwargs : dict_like
        Extra kwargs to pass to the estimators.

    """

    def __init__(
        self,
        propensity_scores: list[float] | npt.NDArray[np.float64],
        numerator_outcomes: list[float | int] | npt.NDArray[np.float64 | np.int64],
        denominator_outcomes: list[float | int] | npt.NDArray[np.float64 | np.int64],
        estimand: Estimand | None = None,
        estimand_class: type[SimpleEstimand] | None = None,
        estimator_class: Literal["IPW", "AIPW", "SIPW", "SAIPW"] = "SIPW",
        numerator_estimator_class: (
            Literal["IPW", "AIPW", "SIPW", "SAIPW"] | None
        ) = None,
        denominator_estimator_class: (
            Literal["IPW", "AIPW", "SIPW", "SAIPW"] | None
        ) = None,
        numerator_kwargs: dict[str, Any] | None = None,
        denominator_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        propensity_scores = np.asarray(propensity_scores)
        numerator_outcomes = np.asarray(numerator_outcomes)
        denominator_outcomes = np.asarray(denominator_outcomes)

        numerator_kwargs_nn = dict(**kwargs)
        if numerator_kwargs is not None:
            numerator_kwargs_nn.update(**numerator_kwargs)

        denominator_kwargs_nn = dict(**kwargs)
        if denominator_kwargs is not None:
            denominator_kwargs_nn.update(**denominator_kwargs)

        classes = {
            "IPW": IPWEstimator,
            "AIPW": AIPWEstimator,
            "SIPW": SIPWEstimator,
            "SAIPW": SAIPWEstimator,
        }

        self.numerator_estimator: MeanEstimator = classes[
            numerator_estimator_class or estimator_class
        ](
            propensity_scores=propensity_scores,
            outcomes=numerator_outcomes,
            estimand=estimand,
            estimand_class=estimand_class,
            **numerator_kwargs_nn,
        )

        self.denominator_estimator: MeanEstimator = classes[
            denominator_estimator_class or estimator_class
        ](
            propensity_scores=propensity_scores,
            outcomes=denominator_outcomes,
            estimand=estimand,
            estimand_class=estimand_class,
            **denominator_kwargs_nn,
        )

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Weights from the numerator estimator.

        Returns
        -------
         weights : (M,) ``npt.NDArray[np.float64]``
            Weight for each of the ``M`` sample members.

        """
        return self.numerator_estimator.weights

    @property
    def propensity_scores(self) -> npt.NDArray[np.float64]:
        """Propensity scores from the numerator estimator.

        Returns
        -------
         propensity_scores : (M,) ``npt.NDArray[np.float64]``
            Estimated response propensity for each sample member.

        """
        return self.numerator_estimator.propensity_scores

    def point_estimate(self) -> float:
        """Calculate a point estimate.

        Returns
        -------
         estimate : float
            Ratio of the numerator and denominator point estimates.

        """
        return (
            self.numerator_estimator.point_estimate()
            / self.denominator_estimator.point_estimate()
        )

    def variance(self) -> float:
        """Calculate the variance.

        Uses the delta method to calculate the variance.

        Returns
        -------
         variance : float
            Estimated sampling variance of the ratio estimate.

        """
        mu_num = self.numerator_estimator.point_estimate()
        var_num = self.numerator_estimator.variance()
        mu_den = self.denominator_estimator.point_estimate()
        var_den = self.denominator_estimator.variance()

        w = self.weights
        x = self.numerator_estimator.outcomes
        y = self.denominator_estimator.outcomes
        cov_num_den = np.sum(np.square(w) * (x - mu_num) * (y - mu_den)) / (
            np.sum(w) ** 2.0
        )

        return max(
            0.0,
            (
                ((1.0 / mu_den) ** 2.0) * var_num
                + ((mu_num / (mu_den**2.0)) ** 2.0) * var_den
                - (2.0 * mu_num / (mu_den**3.0)) * cov_num_den
            ),
        )

    def sensitivity_analysis(
        self,
        gamma: float = 6.0,
        outcome_proxies: list[float] | npt.NDArray[np.float64] | None = None,
        proxy_mean: float | None = None,
        alpha: float = 0.10,
    ) -> tuple[float, float]:
        """Perform a sensitivity analysis.

        Calculate a range of point estimates implied by the Gamma factor.

        Parameters
        ----------
         gamma : float, optional
            The Gamma factor. Must be >= 1.0, with 1.0 indicating perfect propensity
            scores. Defaults to 6. See Notes.
         outcome_proxies : array_like, optional
            Ignored by this estimator; proxy constraints are not supported here.
         proxy_mean : float, optional
            Ignored by this estimator.
         alpha : float, optional
            Ignored by this estimator.

        Returns
        -------
         lb, ub : float
            Lower and upper bounds on the point estimate.

        Raises
        ------
         ValueError
            If ``gamma`` < 1.

        Notes
        -----
        See the Notes in ``WeightingEstimator`` for overall context.

        We calculate

           inf Σᵢwᵢxᵢ/Σᵢwᵢyᵢ

        over wlᵢ ≤ wᵢ ≤ wuᵢ as the lower bound of the sensitivity interval, and the
        supremum as the upper bound. These are linear fractional programs subject to
        linear inequality constraints. They may be transformed to equivalent linear
        programs and solved efficiently, as described in Boyd and Vandenberghe (2004).

        References
        ----------
         - Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge
           University Press, 2004.

        """
        if gamma < 1.0:
            raise ValueError("`gamma` must be >= 1")

        if gamma == 1.0:
            return self.point_estimate(), self.point_estimate()

        weights = self.weights
        wl = weights / math.sqrt(gamma) + (1.0 - 1.0 / math.sqrt(gamma))
        wu = weights * math.sqrt(gamma) - (math.sqrt(gamma) - 1.0)

        x_num = self.numerator_estimator.outcomes
        y_den = self.denominator_estimator.outcomes

        lb_res = solve_bounded_linear_fractional_program(c=x_num, y=y_den, wl=wl, wu=wu)
        ub_res = solve_bounded_linear_fractional_program(
            c=-x_num, y=y_den, wl=wl, wu=wu
        )

        return lb_res.objective_value, -ub_res.objective_value

    def resample(
        self,
        B: int,
        seed: None | (
            int
            | list[int]
            | np.random.SeedSequence
            | np.random.BitGenerator
            | np.random.Generator
        ) = None,
    ) -> Generator[RatioEstimator, None, None]:
        """Yield a sequence of resampled estimators.

        Parameters
        ----------
         B : int
            Number of bootstrap replications to run.
         seed : int, list_like, etc
            A seed for ``numpy.random.default_rng``. See that documentation for details.

        Yields
        ------
         estimator : ``RatioEstimator``
            Bootstrap-resampled estimator.

        """
        rng = np.random.default_rng(seed)
        for _ in range(B):
            # Resample with replacement
            bootstrap_indices = rng.choice(
                range(len(self.weights)),
                size=len(self.weights),
                replace=True,
            )
            bootstrap_propensities = self.propensity_scores[bootstrap_indices]
            bootstrap_numerator_outcomes = self.numerator_estimator.outcomes[
                bootstrap_indices
            ]
            bootstrap_denominator_outcomes = self.denominator_estimator.outcomes[
                bootstrap_indices
            ]
            # Implementation note for AIPW and SAIPW estimators: `self.outcomes` is
            # actually the adjusted outcomes, outcome minus prediction, and
            # `self.extra_args["predicted_outcomes"]` is actually just a vector of all
            # zeros to make the constructor happy. So there is no need to resample those
            # predicted outcomes.

            # Calculate sensitivity interval for this bootstrap sample
            yield self.__class__(
                propensity_scores=bootstrap_propensities,
                numerator_outcomes=bootstrap_numerator_outcomes,
                denominator_outcomes=bootstrap_denominator_outcomes,
                estimand=self.numerator_estimator.estimand.resample(bootstrap_indices),
                numerator_kwargs=self.numerator_estimator.extra_args,
                denominator_kwargs=self.denominator_estimator.extra_args,
            )
