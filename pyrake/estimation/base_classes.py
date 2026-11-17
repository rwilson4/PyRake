"""Base classes for estimation."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from scipy import stats


class Estimand(ABC):
    """Base class for things that can be estimated."""

    @property
    @abstractmethod
    def weights(self) -> npt.NDArray[np.float64]:
        """Calculate weights based on propensity scores.

        Returns
        -------
         weights : (M,) ``npt.NDArray[np.float64]``
            Weight for each of the ``M`` sample members.

        """
        ...

    @abstractmethod
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
        pass

    @abstractmethod
    def resample(self, bootstrap_indices: npt.NDArray[np.int64]) -> Estimand:
        """Create a new Estimand object based on resampled indices.

        Parameters
        ----------
         bootstrap_indices : (B,) ``npt.NDArray[np.int64]``
            Row positions to retain in the resampled estimand.

        Returns
        -------
         estimand : ``Estimand``
            New estimand built from the indexed rows.

        """
        pass

    @staticmethod
    @abstractmethod
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
            Multiplicative factor mapping sample totals to population totals.

        """


class SimpleEstimand(Estimand):
    """Estimand determined by a single vector of propensity scores.

    Parameters
    ----------
     propensity_scores : (M,) ``npt.NDArray[np.float64]``
        Estimated response propensity for each of the ``M`` sample members.

    """

    def __init__(self, propensity_scores: npt.NDArray[np.float64]) -> None:
        self.propensity_scores = propensity_scores

    def resample(self, bootstrap_indices: npt.NDArray[np.int64]) -> SimpleEstimand:
        """Create a new Estimand object based on resampled indices.

        Parameters
        ----------
         bootstrap_indices : (B,) ``npt.NDArray[np.int64]``
            Row positions to retain in the resampled estimand.

        Returns
        -------
         estimand : ``SimpleEstimand``
            New estimand of the same class built from the indexed rows.

        """
        return self.__class__(self.propensity_scores[bootstrap_indices])


class WeightingEstimator(ABC):
    """Estimator that reweights sample members to a target population."""

    @abstractmethod
    def point_estimate(self) -> float:
        """Calculate a point estimate.

        Returns
        -------
         estimate : float
            Estimated value of the target quantity.

        """

    @abstractmethod
    def variance(self) -> float:
        """Calculate the variance.

        Returns
        -------
         variance : float
            Sampling variance of the point estimate.

        """

    def _sensitivity_variance(self, gamma: float) -> tuple[float, float]:
        """Return variance evaluated at the worst-case sensitivity weights.

        Returns ``(lb_var, ub_var)`` where ``lb_var`` is the variance at the weights
        that achieve the lower sensitivity bound and ``ub_var`` is the variance at the
        weights that achieve the upper bound.

        The default falls back to the nominal variance for both bounds. Subclasses that
        can identify the worst-case weight vectors override this to implement the
        heuristic from (Wilson 2025, §3): minimize the test-statistic numerator and use
        those same weights in the denominator.

        Parameters
        ----------
         gamma : float
            The Gamma factor. Must be >= 1.0.

        Returns
        -------
         lb_var, ub_var : float
            Variance at the lower- and upper-bound sensitivity weights.

        References
        ----------
        - Wilson, Bob. 2025. "Design-Based Inference and Sensitivity Analysis for Survey
          Sampling." https://adventuresinwhy.com/post/survey_sampling/

        """
        v = self.variance()
        return v, v

    @abstractmethod
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
            A proxy for the outcome with a known population mean. When supplied together
            with ``proxy_mean``, the sensitivity region gains a constraint keeping the
            proxy estimate within tolerance of ``proxy_mean``, which typically narrows
            the interval. Defaults to None (no proxy constraint).
         proxy_mean : float, optional
            Known population mean of the outcome proxy. Must be supplied together with
            ``outcome_proxies``. Defaults to None.
         alpha : float, optional
            Significance level for the proxy tolerance. Defaults to 0.10.

        Returns
        -------
         lb, ub : float
            Lower and upper bounds on the point estimate.

        Raises
        ------
         ValueError
            Raised by implementations when ``gamma`` < 1, when only one of
            ``outcome_proxies`` and ``proxy_mean`` is supplied, or when the proxy
            constraints admit no feasible weights.

        Notes
        -----
        (Zhao, Small, and Bhattarcharya 2019) posits a model:

           1/Λ < (πᵢ / (1 - πᵢ)) / (π̂ᵢ / (1 - π̂ᵢ)) < Λ,

        where πᵢ is the true (unknown) propensity score for person i and π̂ᵢ is its
        estimate. That is, Λ controls how "wrong" the estimated propensity scores are.
        For sufficiently large values of Λ, this model will always hold, but we have no
        way of knowing what value of Λ is "good enough". So instead we recommend trying
        a variety of Λ values and plot the results.

        As discussed in (Fogarty 2023), this model is connected to Rosenbaum's
        sensitivity analysis via Λ = √Γ. To foster connection with that work (and
        because ``lambda`` is a reserved keyword in python!), we take Gamma as input to
        this calculation, and then calculate Λ as √Γ.

        While no value of Gamma is guaranteed to be good enough, a value of Gamma=6 is a
        good candidate for default reporting. In his work on sensitivity analysis, Paul
        Rosenbaum often cites Hammond's meta-analysis on the connection between smoking
        and cancer, which was sensitive to biases above Gamma = 6, yet was persuasive in
        establishing that smoking causes cancer. While biases may exceed Gamma = 6, this
        value balances rigor and practicality.

        The constraint on the propensity scores translates into a constraint on the
        weights, but this depends on the estimator. The "true" weights satisfy:
           wlᵢ ≤ wᵢ ≤ wuᵢ.

        (Zhao, Small, and Bhattarcharya 2019) calculates the infimum and supremum of the
        estimator over this range of weights. Details vary depending on the estimator.

        References
        ----------
         - Zhao, Qingyuan, Dylan S Small, and Bhaswar B Bhattacharya. 2019. "Sensitivity
           Analysis for Inverse Probability Weighting Estimators via the Percentile
           Bootstrap." Journal of the Royal Statistical Society Series B: Statistical
           Methodology 81 (4). Oxford University Press: pg. 735--61.
         - Fogarty, Colin. 2023. "Sensitivity Analysis." In Handbook of Matching and
           Weighting Adjustments for Causal Inference, , pg. 553--82. Chapman and
           Hall/CRC.

        """

    @abstractmethod
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
    ) -> Generator[WeightingEstimator, None, None]:
        """Yield a sequence of B bootstrap-resampled estimators.

        Parameters
        ----------
         B : int
            Number of bootstrap replications.
         seed : int, list_like, etc
            A seed for ``numpy.random.default_rng``.

        Yields
        ------
         estimator : ``WeightingEstimator``
            Bootstrap-resampled estimator.

        """

    def expanded_confidence_interval(
        self,
        alpha: float = 0.10,
        gamma: float = 6.0,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        bootstrap: bool = True,
        B: int = 1_000,
        seed: None | (
            int
            | list[int]
            | np.random.SeedSequence
            | np.random.BitGenerator
            | np.random.Generator
        ) = None,
    ) -> tuple[float, float]:
        """Calculate an expanded confidence interval.

        The expanded confidence interval combines uncertainty from stochastic sampling
        and from propensity scores estimated with error.

        Parameters
        ----------
         alpha : float, optional
            P-value threshold, e.g. specify alpha=0.05 for a 95% confidence interval.
            Defaults to 0.10, corresponding to a 90% confidence interval.
         gamma : float, optional
            The Gamma factor. Must be >= 1.0, with 1.0 indicating perfect propensity
            scores. Defaults to 6. See Notes in ``sensitivity_analysis``.
         alternative : ["two-sided", "less", "greater"], optional
            What kind of test:
              - "two-sided": H0: theta = ``null_value`` vs Halt:
                theta <> ``null_value``.
              - "greater": H0: theta <= ``null_value`` vs Halt: theta > ``null_value``.
              - "less": H0: theta >= ``null_value`` vs Halt: theta < ``null_value``.
            Defaults to "two-sided".
         bootstrap : bool, optional
            If True (default), use the percentile bootstrap (Zhao, Small, and
            Bhattacharya, 2019) to account for sampling variability in the sensitivity
            bounds. If False, use the normal approximation: the ECI is the sensitivity
            interval expanded by a multiple of the standard error on each side, where
            the standard error is evaluated at the worst-case sensitivity weights (the
            heuristic from Wilson 2025, §3). See ``adjusted_pvalue`` for additional
            notes.
         B : int, optional
            Number of bootstrap replications to run. Ignored when ``bootstrap=False``.
            Defaults to 1_000.
         seed : int, list_like, etc
            A seed for ``numpy.random.default_rng``. Ignored when ``bootstrap=False``.
            See that documentation for details.

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
            If ``gamma`` < 1 or ``alternative`` is not recognized.

        References
        ----------
         - Wilson, Bob. 2025. "Design-Based Inference and Sensitivity Analysis for
           Survey Sampling." https://adventuresinwhy.com/post/survey_sampling/
         - Zhao, Qingyuan, Dylan S Small, and Bhaswar B Bhattacharya. 2019. "Sensitivity
           Analysis for Inverse Probability Weighting Estimators via the Percentile
           Bootstrap." Journal of the Royal Statistical Society Series B: Statistical
           Methodology 81 (4). Oxford University Press: pg. 735--61.

        """
        if gamma < 1.0:
            raise ValueError("`gamma` must be >= 1")

        if gamma == 1.0:
            return self.confidence_interval(alpha=alpha, alternative=alternative)

        if not bootstrap:
            sen_lb, sen_ub = self.sensitivity_analysis(gamma=gamma)
            lb_var, ub_var = self._sensitivity_variance(gamma)
            if alternative == "two-sided":
                zcrit = stats.norm.isf(alpha / 2)
            else:
                zcrit = stats.norm.isf(alpha)

            if alternative == "greater":
                lb = float(sen_lb - zcrit * math.sqrt(lb_var))
                return lb, np.inf
            elif alternative == "less":
                ub = float(sen_ub + zcrit * math.sqrt(ub_var))
                return -np.inf, ub
            elif alternative == "two-sided":
                lb = float(sen_lb - zcrit * math.sqrt(lb_var))
                ub = float(sen_ub + zcrit * math.sqrt(ub_var))
                return lb, ub
            else:
                raise ValueError(f"Unrecognized input {alternative=:}")

        lb_bootstrap = np.zeros(B)
        ub_bootstrap = np.zeros(B)
        for b, est in enumerate(self.resample(B, seed)):
            lb_bootstrap[b], ub_bootstrap[b] = est.sensitivity_analysis(gamma=gamma)

        if alternative == "two-sided":
            lb = float(np.percentile(lb_bootstrap, 100 * alpha / 2))
            ub = float(np.percentile(ub_bootstrap, 100 * (1 - alpha / 2)))
        elif alternative == "less":
            lb = -np.inf
            ub = float(np.percentile(ub_bootstrap, 100 * (1 - alpha)))
        elif alternative == "greater":
            lb = float(np.percentile(lb_bootstrap, 100 * alpha))
            ub = np.inf
        else:
            raise ValueError(f"Unrecognized input {alternative=:}")

        return lb, ub

    def pvalue(
        self,
        null_value: float,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ) -> float:
        """Calculate a p-value.

        Computes a p-value against the null hypothesis:
            H0: theta = null_value,
        where theta is the quantity being estimated

        Parameters
        ----------
         null_value : float
            The hypothesized theta.
         alternative : ["two-sided", "less", "greater"], optional
            What kind of test:
              - "two-sided": H0: theta = ``null_value`` vs Halt:
                theta <> ``null_value``.
              - "greater": H0: theta <= ``null_value`` vs Halt: theta > ``null_value``.
              - "less": H0: theta >= ``null_value`` vs Halt: theta < ``null_value``.
            For example, specifying alternative = "greater" returns a p-value that
            quantifies the strength of evidence against the null hypothesis that theta
            is less than ``null_value``, in favor of the alternative hypothesis that
            theta is actually greater than ``null_value``. Defaults to "two-sided".

        Returns
        -------
         p : float
            P-value.

        Raises
        ------
         ValueError
            If ``alternative`` is not recognized.

        """
        pe = self.point_estimate()
        se = math.sqrt(self.variance())

        t = (pe - null_value) / se

        if alternative == "greater":
            return float(stats.norm.sf(t))
        elif alternative == "less":
            return float(stats.norm.cdf(t))
        elif alternative == "two-sided":
            p_greater = float(stats.norm.sf(t))
            p_less = float(stats.norm.cdf(t))
            return min(1.0, 2.0 * min(p_less, p_greater))
        else:
            raise ValueError(f"Unrecognized input {alternative=:}")

    def adjusted_pvalue(
        self,
        null_value: float,
        gamma: float = 6.0,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        bootstrap: bool = True,
        B: int = 1_000,
        seed: None | (
            int
            | list[int]
            | np.random.SeedSequence
            | np.random.BitGenerator
            | np.random.Generator
        ) = None,
    ) -> float:
        """Calculate a p-value adjusted for hidden bias.

        The adjusted p-value accounts for both sampling uncertainty and uncertainty from
        propensity scores estimated with error. It is the smallest alpha for which the
        expanded confidence interval at level alpha excludes ``null_value``, making it
        the dual of ``expanded_confidence_interval`` in the same sense that ``pvalue``
        is the dual of ``confidence_interval``.

        Parameters
        ----------
         null_value : float
            The hypothesized theta.
         gamma : float, optional
            The Gamma factor. Must be >= 1.0, with 1.0 indicating perfect propensity
            scores. Defaults to 6. See Notes in ``sensitivity_analysis``.
         alternative : ["two-sided", "less", "greater"], optional
            What kind of test:
              - "two-sided": H0: theta = ``null_value`` vs Halt:
                theta <> ``null_value``.
              - "greater": H0: theta <= ``null_value`` vs Halt: theta > ``null_value``.
              - "less": H0: theta >= ``null_value`` vs Halt: theta < ``null_value``.
            Defaults to "two-sided".
         bootstrap : bool, optional
            If True (default), use the percentile bootstrap to account for sampling
            variability in the sensitivity bounds. If False, use the normal
            approximation: the test statistic is formed from the sensitivity bound and
            the standard error evaluated at the worst-case sensitivity weights (the
            heuristic from Wilson 2025, §3). This provides a lower bound on the adjusted
            p-value, but simulations suggest it's tight enough to be practically useful.
         B : int, optional
            Number of bootstrap replications to run. Ignored when ``bootstrap=False``.
            Defaults to 1_000.
         seed : int, list_like, etc
            A seed for ``numpy.random.default_rng``. Ignored when ``bootstrap=False``.
            See that documentation for details.

        Returns
        -------
         p : float
            Adjusted p-value.

        Raises
        ------
         ValueError
            If ``gamma`` < 1 or ``alternative`` is not recognized.

        Notes
        -----
        When ``bootstrap=True``, B bootstrap replicates are drawn. For each replicate,
        ``sensitivity_analysis`` computes the range of point estimates consistent with
        gamma-level hidden bias. The adjusted p-value is then the fraction of bootstrap
        replicates whose sensitivity interval does not rule out ``null_value``:

          - "greater": fraction of bootstrap lower bounds that do not exceed
            ``null_value``, i.e. mean(lb_bootstrap <= ``null_value``).
          - "less": fraction of bootstrap upper bounds not below ``null_value``,
            i.e. mean(ub_bootstrap > ``null_value``).
          - "two-sided": twice the minimum of the two one-sided adjusted p-values,
            capped at 1.

        When ``bootstrap=False``, the normal approximation is used instead: the test
        statistic is formed from the sensitivity bound in the numerator and the standard
        error at the worst-case weights in the denominator. Because the true infimum of
        the test statistic is intractable (the statistic is not convex in the weights),
        this uses the heuristic from (Wilson 2025, §3): find the weights that minimize
        the numerator, then evaluate the standard error at those same weights.
        Simulations show this closely tracks the bootstrap-based adjusted p-value.

        When gamma=1 the sensitivity interval collapses to the point estimate, and the
        adjusted p-value reduces to the standard p-value.

        References
        ----------
         - Wilson, Bob. 2025. "Design-Based Inference and Sensitivity Analysis for
           Survey Sampling." https://adventuresinwhy.com/post/survey_sampling/
         - Zhao, Qingyuan, Dylan S Small, and Bhaswar B Bhattacharya. 2019. "Sensitivity
           Analysis for Inverse Probability Weighting Estimators via the Percentile
           Bootstrap." Journal of the Royal Statistical Society Series B: Statistical
           Methodology 81 (4): 735--61.

        """
        if gamma < 1.0:
            raise ValueError("`gamma` must be >= 1")

        if gamma == 1.0:
            return self.pvalue(null_value=null_value, alternative=alternative)

        if not bootstrap:
            sen_lb, sen_ub = self.sensitivity_analysis(gamma=gamma)
            lb_var, ub_var = self._sensitivity_variance(gamma)

            if alternative == "greater":
                return float(stats.norm.sf((sen_lb - null_value) / math.sqrt(lb_var)))
            elif alternative == "less":
                return float(stats.norm.sf((null_value - sen_ub) / math.sqrt(ub_var)))
            elif alternative == "two-sided":
                p_greater = float(
                    stats.norm.sf((sen_lb - null_value) / math.sqrt(lb_var))
                )
                p_less = float(stats.norm.sf((null_value - sen_ub) / math.sqrt(ub_var)))
                return min(1.0, 2.0 * min(p_greater, p_less))
            else:
                raise ValueError(f"Unrecognized input {alternative=:}")

        lb_bootstrap = np.zeros(B)
        ub_bootstrap = np.zeros(B)
        for b, est in enumerate(self.resample(B, seed)):
            lb_bootstrap[b], ub_bootstrap[b] = est.sensitivity_analysis(gamma=gamma)

        if alternative == "greater":
            return float(np.mean(lb_bootstrap <= null_value))
        elif alternative == "less":
            return float(np.mean(ub_bootstrap > null_value))
        elif alternative == "two-sided":
            p_greater = float(np.mean(lb_bootstrap <= null_value))
            p_less = float(np.mean(ub_bootstrap > null_value))
            return min(1.0, 2.0 * min(p_greater, p_less))
        else:
            raise ValueError(f"Unrecognized input {alternative=:}")

    def gamma_star(
        self,
        null_value: float,
        alpha: float = 0.05,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        bootstrap: bool = True,
        B: int = 1_000,
        seed: None | (
            int
            | list[int]
            | np.random.SeedSequence
            | np.random.BitGenerator
            | np.random.Generator
        ) = None,
        gamma_upper: float = 6.0,
        tol: float = 1e-3,
    ) -> float:
        """Find the smallest Gamma at which the result is no longer significant.

        Searches for the smallest Gamma >= 1 such that the adjusted p-value is >=
        ``alpha`` (i.e., the result is no longer statistically significant).

        Parameters
        ----------
         null_value : float
            The hypothesized theta.
         alpha : float, optional
            Significance threshold. Defaults to 0.05.
         alternative : ["two-sided", "less", "greater"], optional
            What kind of test:
              - "two-sided": H0: theta = ``null_value`` vs Halt:
                theta <> ``null_value``.
              - "greater": H0: theta <= ``null_value`` vs Halt: theta > ``null_value``.
              - "less": H0: theta >= ``null_value`` vs Halt: theta < ``null_value``.
            Defaults to "two-sided".
         bootstrap : bool, optional
            If True (default), use the percentile bootstrap. If False, use the normal
            approximation. See ``adjusted_pvalue`` for details.
         B : int, optional
            Number of bootstrap replications. Ignored when ``bootstrap=False``. Defaults
            to 1_000.
         seed : int, list_like, etc, optional
            A seed for ``numpy.random.default_rng``. Ignored when ``bootstrap=False``.
            Passing a fixed seed is recommended when ``bootstrap=True`` because it
            ensures the adjusted p-value is monotone in Gamma, which makes the binary
            search more reliable.
         gamma_upper : float, optional
            Initial upper bound for the exponential search phase. If the adjusted
            p-value is still significant here, the search doubles this value repeatedly
            until a bracket is found. Defaults to 6.
         tol : float, optional
            Convergence tolerance on Gamma. The returned value is accurate to within
            ``tol``. Defaults to 1e-3.

        Returns
        -------
         gamma_star : float
            The smallest Gamma >= 1 at which the result is not statistically
            significant. Returns 1.0 if the result is already not significant at
            Gamma=1.

        Raises
        ------
         ValueError
            If ``gamma_upper`` < 1, or if ``alternative`` is not recognized (propagated
            from ``pvalue`` and ``adjusted_pvalue``).

        Notes
        -----
        The adjusted p-value is monotonically non-decreasing in Gamma: larger Gamma
        widens the sensitivity interval, making it harder to reject the null hypothesis.
        This guarantees that a unique crossing point exists and that binary search
        converges reliably.

        The search proceeds in two phases. First, an exponential search starting at
        ``gamma_upper`` doubles the candidate until a Gamma is found where the adjusted
        p-value exceeds ``alpha``. Then bisection narrows the bracket to within ``tol``.

        When ``bootstrap=True``, pass a fixed ``seed`` to ensure reproducible results
        and to preserve monotonicity across binary search iterations.

        References
        ----------
         - Zhao, Qingyuan, Dylan S Small, and Bhaswar B Bhattacharya. 2019. "Sensitivity
           Analysis for Inverse Probability Weighting Estimators via the Percentile
           Bootstrap." Journal of the Royal Statistical Society Series B: Statistical
           Methodology 81 (4): 735--61.

        """
        gamma_max = 1_000.0

        if gamma_upper < 1.0:
            raise ValueError("`gamma_upper` must be >= 1")

        def apvalue(gamma: float) -> float:
            return self.adjusted_pvalue(
                null_value=null_value,
                gamma=gamma,
                alternative=alternative,
                bootstrap=bootstrap,
                B=B,
                seed=seed,
            )

        if self.pvalue(null_value=null_value, alternative=alternative) >= alpha:
            return 1.0

        # Exponential search: double gamma_upper until adjusted p-value >= alpha.
        lo, hi = 1.0, gamma_upper
        while apvalue(hi) < alpha:
            if hi > gamma_max:
                # The result is robust enough that no finite Gamma explains it
                # away; return inf to signal no crossing was found.
                return math.inf
            lo = hi
            hi *= 2

        # Bisect to find the crossing within tol.
        while hi - lo > tol:
            mid = (lo + hi) / 2.0
            if apvalue(mid) < alpha:
                lo = mid
            else:
                hi = mid

        return hi

    def confidence_interval(
        self,
        alpha: float = 0.10,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ) -> tuple[float, float]:
        """Calculate confidence interval on a quantity, theta.

        Parameters
        ----------
         alpha : float, optional
            P-value threshold, e.g. specify alpha=0.05 for a 95% confidence interval.
            Defaults to 0.10, corresponding to a 90% confidence interval.
         alternative : ["two-sided", "less", "greater"], optional
            What kind of test:
              - "two-sided": H0: theta = ``null_value`` vs Halt:
                theta <> ``null_value``.
              - "greater": H0: theta <= ``null_value`` vs Halt: theta > ``null_value``.
              - "less": H0: theta >= ``null_value`` vs Halt: theta < ``null_value``.
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
            If ``alternative`` is not recognized.

        """
        pe = self.point_estimate()
        se = math.sqrt(self.variance())
        if alternative == "two-sided":
            zcrit = stats.norm.isf(alpha / 2)
        else:
            zcrit = stats.norm.isf(alpha)

        if alternative == "greater":
            lb = float(pe - zcrit * se)
            return lb, np.inf
        elif alternative == "less":
            ub = float(pe + zcrit * se)
            return -np.inf, ub
        elif alternative == "two-sided":
            lb = float(pe - zcrit * se)
            ub = float(pe + zcrit * se)
            return lb, ub
        else:
            raise ValueError(f"Unrecognized input {alternative=:}")

    def plot_sensitivity(
        self,
        gamma_lower: float = 1.0,
        gamma_upper: float = 6.0,
        num_points: int = 50,
        alpha: float = 0.10,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        bootstrap: bool = True,
        B: int = 1_000,
        null_value: float | None = None,
        title: str | None = None,
        ylabel: str = "Sensitivity Interval",
        ytick_format: str = ".0%",
        axis_label_size: int | None = None,
        tick_label_size: int | None = None,
        legend_label_size: int | None = None,
        legend_placement: str | None = None,
        ax: Axes | None = None,
    ) -> tuple[pd.DataFrame, Axes]:
        """Plot sensitivity of estimates to hidden biases.

        Parameters
        ----------
         gamma_lower, gamma_upper : float, optional
            Range of Gamma values to test. Defaults to 1 and 6 resp.
         num_points : int, optional
            Number of Gammas to simulate and plot. Defaults to 50.
         alpha : float, optional
            P-value threshold, e.g. specify alpha=0.05 for a 95% confidence interval.
            Defaults to 0.10, corresponding to a 90% confidence interval.
         alternative : ["two-sided", "less", "greater"], optional
            What kind of test:
              - "two-sided": H0: Ȳ = ``null_value`` vs Halt: Ȳ <>
                ``null_value``.
              - "greater": H0: Ȳ <= ``null_value`` vs Halt: Ȳ > ``null_value``.
              - "less": H0: Ȳ >= ``null_value`` vs Halt: Ȳ < ``null_value``.
         bootstrap : bool, optional
            Passed to ``expanded_confidence_interval``. If True (default), use the
            percentile bootstrap. If False, use the normal approximation (much faster,
            especially when plotting many gamma values).
         B : int, optional
            Number of bootstrap replications per gamma value. Ignored when
            ``bootstrap=False``. Defaults to 1_000.
         null_value : float, optional
            If specified, compute ``gamma_star`` (the smallest Gamma at which the result
            is no longer significant against this null) and annotate the plot with a
            vertical line at that point. No annotation is added if the result is already
            non-significant at Gamma=1 (``gamma_star`` == 1) or if ``gamma_star``
            exceeds ``gamma_upper``.
         title, ylabel : str, optional
            Title/ylabel for plot. If not specified, no title is included.
         ytick_format : str, optional
            How to format the y-axis tick labels. Defaults to ".0%", corresponding to a
            percentage.
         axis_label_size, tick_label_size, legend_label_size : int, optional
            Font size for axis labels and title; tick mark labels; and legend entries,
            resp.
         legend_placement : str, optional
            Legend placement. Chosen automatically if not specified.
         ax : Axes, optional
            If specified, plot everything on these axes.

        Returns
        -------
         df : pandas DataFrame
            A DataFrame with the lower and upper bounds corresponding to each gamma.
         ax : Axes
            The plot axes.

        Raises
        ------
         ValueError
            If the gamma range is empty, fewer than two points are requested, or
            ``alternative`` is not recognized.

        """
        if gamma_lower < 1 or gamma_upper <= gamma_lower:
            raise ValueError("Must have 1 <= gamma_lower < gamma_upper")

        if num_points < 2:
            raise ValueError("Must plot at least 2 points")

        gammas = np.linspace(gamma_lower, gamma_upper, num_points)
        sensitivities_lb = np.zeros_like(gammas)
        sensitivities_ub = np.zeros_like(gammas)
        eci_lb = np.zeros_like(gammas)
        eci_ub = np.zeros_like(gammas)
        for ii, gamma in enumerate(gammas):
            (
                sensitivities_lb[ii],
                sensitivities_ub[ii],
            ) = self.sensitivity_analysis(gamma)
            (
                eci_lb[ii],
                eci_ub[ii],
            ) = self.expanded_confidence_interval(
                alpha=alpha,
                gamma=gamma,
                alternative=alternative,
                bootstrap=bootstrap,
                B=B,
                seed=42,
            )

        # Create a dataframe for plotting
        ci_lb, ci_ub = self.confidence_interval(alpha=alpha, alternative=alternative)
        df = pd.DataFrame(
            {
                "Gamma": gammas,
                "ECI Lower Bound": eci_lb,
                "Sen Lower Bound": sensitivities_lb,
                "Point Estimate": self.point_estimate(),
                "Sen Upper Bound": sensitivities_ub,
                "ECI Upper Bound": eci_ub,
            }
        )

        # Melt the dataframe for plotting
        if alternative == "two-sided":
            value_vars = [
                "Point Estimate",
                "Sen Lower Bound",
                "Sen Upper Bound",
                "ECI Lower Bound",
                "ECI Upper Bound",
            ]
        elif alternative == "greater":
            value_vars = [
                "Point Estimate",
                "Sen Lower Bound",
                "ECI Lower Bound",
            ]
        elif alternative == "less":
            value_vars = [
                "Point Estimate",
                "Sen Upper Bound",
                "ECI Upper Bound",
            ]
        else:
            raise ValueError(f"Unrecognized input {alternative=:}")

        df_melt = pd.melt(
            df,
            id_vars="Gamma",
            value_vars=value_vars,
        )
        color_map = {
            "Point Estimate": "black",
            "Sen Lower Bound": "blue",
            "Sen Upper Bound": "blue",
            "ECI Lower Bound": "orange",
            "ECI Upper Bound": "orange",
        }

        if ax is None:
            fig, ax = plt.subplots()
        else:
            plt.sca(ax)

        sns.lineplot(
            x="Gamma",
            y="value",
            hue="variable",
            data=df_melt,
            palette=color_map,
            legend=False,  # Don't show legend
            ax=ax,
        )

        if alternative == "two-sided":
            ax.plot(
                df["Gamma"],
                np.full_like(df["Gamma"], ci_lb),
                color="black",
                linestyle="--",
            )
            ax.plot(
                df["Gamma"],
                np.full_like(df["Gamma"], ci_ub),
                color="black",
                linestyle="--",
            )
            plt.fill_between(
                gammas, eci_lb, sensitivities_lb, alpha=0.2, color="orange"
            )
            plt.fill_between(
                gammas,
                sensitivities_lb,
                sensitivities_ub,
                alpha=0.2,
                color="blue",
            )
            plt.fill_between(
                gammas, sensitivities_ub, eci_ub, alpha=0.2, color="orange"
            )
        elif alternative == "greater":
            ax.plot(
                df["Gamma"],
                np.full_like(df["Gamma"], ci_lb),
                color="black",
                linestyle="--",
            )
            plt.fill_between(
                gammas, eci_lb, sensitivities_lb, alpha=0.2, color="orange"
            )
            plt.fill_between(
                gammas,
                sensitivities_lb,
                np.full_like(sensitivities_lb, self.point_estimate()),
                alpha=0.2,
                color="blue",
            )
        elif alternative == "less":
            ax.plot(
                df["Gamma"],
                np.full_like(df["Gamma"], ci_ub),
                color="black",
                linestyle="--",
            )
            plt.fill_between(
                gammas,
                np.full_like(sensitivities_lb, self.point_estimate()),
                sensitivities_ub,
                alpha=0.2,
                color="blue",
            )
            plt.fill_between(
                gammas, sensitivities_ub, eci_ub, alpha=0.2, color="orange"
            )

        ax.set_xlim(gamma_lower, gamma_upper)
        ax.set_xlabel("Gamma")
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title, fontsize=axis_label_size)

        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, pos: ("{0:" + ytick_format + "}").format(y))
        )

        if axis_label_size is not None:
            ax.xaxis.label.set_fontsize(axis_label_size)
            ax.yaxis.label.set_fontsize(axis_label_size)

        if tick_label_size is not None:
            ax.tick_params(axis="both", which="major", labelsize=tick_label_size)

        legend_handles = [
            Line2D([0], [0], color="black", linestyle="-", label="Point Estimate"),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                label=f"{(1 - alpha):.0%} Confidence Interval",
            ),
            Line2D(
                [0],
                [0],
                color="blue",
                linestyle="-",
                label="Sensitivity Interval",
            ),
            Line2D(
                [0],
                [0],
                color="orange",
                linestyle="-",
                label="Expanded Confidence Interval",
            ),
        ]

        if legend_placement is None:
            if alternative == "less":
                legend_placement = "upper left"
            else:
                legend_placement = "lower left"

        ax.legend(
            handles=legend_handles,
            loc=legend_placement,
            fontsize=legend_label_size,
        )

        if null_value is not None:
            assert alternative in (
                "two-sided",
                "less",
                "greater",
            )  # Satisfy type-checker
            gs = self.gamma_star(
                null_value=null_value,
                alpha=alpha,
                alternative=alternative,
                bootstrap=bootstrap,
                B=B,
                seed=42,
                gamma_upper=gamma_upper,
            )
            if gs > 1.0 and gs <= gamma_upper:
                ax.axvline(x=gs, color="black", linestyle=":")
                ax.annotate(
                    f"$\\Gamma^* = {gs:.2f}$",
                    xy=(gs, ax.get_ylim()[0]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    color="black",
                    fontsize=tick_label_size,
                )

        plt.tight_layout()
        return df, ax

    def min_ess_bound(
        self,
        gamma: float,
        method: Literal["floor", "exact"] = "floor",
    ) -> float:
        """Worst-case effective sample size over the Gamma sensitivity region.

        Returns the smallest Kish effective sample size attainable anywhere in the Gamma
        sensitivity region (``method="exact"``), or a cheap conservative lower bound on
        it (``method="floor"``). The default raises, since this is only defined for
        estimators whose sensitivity region is a box of weights. See
        ``MeanEstimator.min_ess_bound`` for the implementation.

        Parameters
        ----------
         gamma : float
            The Gamma factor. Must be >= 1.0. See ``sensitivity_analysis``.
         method : ["floor", "exact"], optional
            "floor" (default) returns the conservative closed-form lower bound; "exact"
            returns the actual worst-case minimum via a vertex sweep.

        Returns
        -------
         ess : float
            The worst-case effective sample size over the region (or a lower bound).

        Raises
        ------
         NotImplementedError
            Always raised here; only mean estimators implement this method.

        """
        raise NotImplementedError(
            "min_ess_bound is only implemented for mean estimators "
            "(IPW/SIPW/AIPW/SAIPW)."
        )

    def plot_min_ess(
        self,
        gamma_lower: float = 1.0,
        gamma_upper: float = 6.0,
        num_points: int = 50,
        method: Literal["floor", "exact"] = "floor",
        reference_ess: float | None = None,
        title: str | None = None,
        ylabel: str = "Minimum Effective Sample Size",
        ytick_format: str = ",.0f",
        axis_label_size: int | None = None,
        tick_label_size: int | None = None,
        legend_label_size: int | None = None,
        legend_placement: str | None = None,
        ax: Axes | None = None,
    ) -> tuple[pd.DataFrame, Axes]:
        """Plot the worst-case effective sample size against hidden bias.

        Plots the conservative lower bound on the smallest Kish effective sample size
        ESS(w) = (1ᵀw)²/‖w‖² attainable anywhere in the Gamma sensitivity region, as a
        function of Gamma. This is the effective-sample-size analogue of
        ``plot_sensitivity``: it shows how much the leverage of the weights can collapse
        across the region as the assumed hidden bias grows.

        At ``gamma_lower = 1`` the region collapses to a point, so the curve begins at
        the exact effective sample size of the nominal weights; as Gamma grows the
        region widens and the bound decreases. The bound is computed by
        ``min_ess_bound``, which uses the conservative floor derived in (Wilson 2025):
        the numerator at the region's lowest corner over the denominator at its highest.

        Parameters
        ----------
         gamma_lower, gamma_upper : float, optional
            Range of Gamma values to plot. Defaults to 1 and 6 resp.
         num_points : int, optional
            Number of Gammas to evaluate and plot. Defaults to 50.
         method : ["floor", "exact"], optional
            Passed to ``min_ess_bound``. "floor" (default) plots the conservative
            closed-form lower bound; "exact" plots the actual worst-case minimum.
         reference_ess : float, optional
            If specified, draw a horizontal dashed line at this effective sample size
            (e.g. a reliability floor below which the normal approximation is suspect).
            No line is drawn if not specified.
         title, ylabel : str, optional
            Title/ylabel for plot. If ``title`` is not specified, no title is included.
         ytick_format : str, optional
            How to format the y-axis tick labels. Defaults to ",.0f" (integer counts
            with thousands separators).
         axis_label_size, tick_label_size, legend_label_size : int, optional
            Font size for axis labels and title; tick mark labels; and legend entries,
            resp.
         legend_placement : str, optional
            Legend placement. Defaults to "upper right".
         ax : Axes, optional
            If specified, plot everything on these axes.

        Returns
        -------
         df : pandas DataFrame
            A DataFrame with the minimum effective sample size bound for each gamma.
         ax : Axes
            The plot axes.

        Raises
        ------
         ValueError
            If the gamma range is empty or fewer than two points are requested.

        References
        ----------
         - Wilson, Bob. 2025. "Design-Based Inference and Sensitivity Analysis for
           Survey Sampling." https://adventuresinwhy.com/post/survey_sampling/

        """
        if gamma_lower < 1 or gamma_upper <= gamma_lower:
            raise ValueError("Must have 1 <= gamma_lower < gamma_upper")

        if num_points < 2:
            raise ValueError("Must plot at least 2 points")

        gammas = np.linspace(gamma_lower, gamma_upper, num_points)
        min_ess = np.array(
            [self.min_ess_bound(gamma, method=method) for gamma in gammas]
        )

        df = pd.DataFrame(
            {
                "Gamma": gammas,
                "Min ESS Bound": min_ess,
            }
        )

        if ax is None:
            fig, ax = plt.subplots()
        else:
            plt.sca(ax)

        ax.plot(df["Gamma"], df["Min ESS Bound"], color="blue", linestyle="-")
        plt.fill_between(gammas, 0.0, min_ess, alpha=0.2, color="blue")

        legend_handles = [
            Line2D(
                [0],
                [0],
                color="blue",
                linestyle="-",
                label="Min ESS bound",
            ),
        ]

        if reference_ess is not None:
            ax.plot(
                df["Gamma"],
                np.full_like(gammas, reference_ess),
                color="black",
                linestyle="--",
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle="--",
                    label=f"ESS = {reference_ess:,.0f}",
                )
            )

        ax.set_xlim(gamma_lower, gamma_upper)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("Gamma")
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title, fontsize=axis_label_size)

        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, pos: ("{0:" + ytick_format + "}").format(y))
        )

        if axis_label_size is not None:
            ax.xaxis.label.set_fontsize(axis_label_size)
            ax.yaxis.label.set_fontsize(axis_label_size)

        if tick_label_size is not None:
            ax.tick_params(axis="both", which="major", labelsize=tick_label_size)

        if legend_placement is None:
            legend_placement = "upper right"

        ax.legend(
            handles=legend_handles,
            loc=legend_placement,
            fontsize=legend_label_size,
        )

        plt.tight_layout()
        return df, ax
