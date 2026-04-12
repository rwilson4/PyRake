"""Linear combination estimator."""

from collections.abc import Generator
from typing import Literal

import numpy as np

from .base_classes import WeightingEstimator


class LinearCombinationEstimator(WeightingEstimator):
    r"""Affine combination of WeightingEstimators.

    Estimates:

        theta = affine + sum_i c_i * theta_i

    where each theta_i is estimated by the corresponding component estimator.
    Component estimators may be MeanEstimators, TreatmentEffectEstimators, or
    other LinearCombinationEstimators.

    Sensitivity analysis is valid because the objective is separable across
    components: hidden bias in each component contributes independently, and
    the worst-case bounds combine linearly (with sign-aware bound selection).

    Parameters
    ----------
     terms : list of (float, WeightingEstimator) pairs
        Coefficients and their associated estimators.
     affine : float, optional
        Constant offset. Defaults to 0.0.

    Notes
    -----
    Variance is computed assuming independence across component estimators:

        var(theta) = sum_i c_i^2 * var(theta_i).

    This is exact when components are fit on disjoint datasets (e.g. different
    treatment arms or independent studies). If components share data, variance
    will be understated and variance-based methods (confidence_interval,
    pvalue) will be anti-conservative. Bootstrap-based methods
    (expanded_confidence_interval) carry the same independence assumption
    since each component is resampled independently.

    """

    def __init__(
        self,
        terms: list[tuple[float, WeightingEstimator]],
        affine: float = 0.0,
    ) -> None:
        self.terms = terms
        self.affine = affine

    def point_estimate(self) -> float:
        """Calculate the point estimate."""
        return self.affine + sum(c * e.point_estimate() for c, e in self.terms)

    def variance(self) -> float:
        """Calculate the variance.

        Assumes independence across component estimators. See class docstring.

        """
        return sum(c**2 * e.variance() for c, e in self.terms)

    def sensitivity_analysis(self, gamma: float = 6.0) -> tuple[float, float]:
        r"""Perform a sensitivity analysis.

        The analysis is separable across components. For coefficient c and
        component sensitivity interval [lb_i, ub_i]:

          - c >= 0: contributes c * lb_i to the overall lower bound and
            c * ub_i to the overall upper bound.
          - c < 0: contributes c * ub_i to the overall lower bound and
            c * lb_i to the overall upper bound.

        The affine term shifts both bounds by the same constant.

        Parameters
        ----------
         gamma : float, optional
            The Gamma factor. Must be >= 1.0, with 1.0 indicating perfect
            propensity scores. Defaults to 6. See WeightingEstimator.

        Returns
        -------
         lb, ub : float
            Lower and upper bounds on the point estimate.

        """
        lb = self.affine
        ub = self.affine
        for c, e in self.terms:
            e_lb, e_ub = e.sensitivity_analysis(gamma)
            if c >= 0:
                lb += c * e_lb
                ub += c * e_ub
            else:
                lb += c * e_ub
                ub += c * e_lb
        return lb, ub

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
    ) -> Generator["LinearCombinationEstimator", None, None]:
        """Yield a sequence of resampled estimators.

        Each of the B bootstrap iterations yields a new
        LinearCombinationEstimator whose components are independently
        resampled from their respective component estimators.

        Parameters
        ----------
         B : int
            Number of bootstrap replications.
         seed : int, list_like, etc
            A seed for numpy.random.default_rng. See that documentation.

        """
        if not self.terms:
            # No stochastic components; yield B identical copies so that
            # this estimator composes correctly inside another LCE.
            for _ in range(B):
                yield LinearCombinationEstimator(terms=[], affine=self.affine)
            return

        resamplers = [e.resample(B, seed=seed) for _, e in self.terms]
        for resampled_components in zip(*resamplers, strict=False):
            yield LinearCombinationEstimator(
                terms=[
                    (c, est)
                    for (c, _), est in zip(
                        self.terms, resampled_components, strict=False
                    )
                ],
                affine=self.affine,
            )

    def adjusted_pvalue(
        self,
        null_value: float,
        gamma: float = 6.0,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        B: int = 1_000,
        seed: None | (
            int
            | list[int]
            | np.random.SeedSequence
            | np.random.BitGenerator
            | np.random.Generator
        ) = None,
    ) -> float:
        r"""Calculate a p-value adjusted for hidden bias via the percentile bootstrap.

        See `WeightingEstimator.adjusted_pvalue` for full documentation.

        When gamma=1, returns the standard p-value. When there are no stochastic
        components, the bootstrap produces a degenerate distribution concentrated
        at the point estimate, and the adjusted p-value is 1 if the null value is
        within the sensitivity interval and 0 otherwise.

        """
        if gamma < 1.0:
            raise ValueError("`gamma` must be >= 1")

        if gamma == 1.0:
            return self.pvalue(null_value=null_value, alternative=alternative)

        return super().adjusted_pvalue(
            null_value=null_value,
            gamma=gamma,
            alternative=alternative,
            B=B,
            seed=seed,
        )

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
        r"""Calculate an expanded confidence interval.

        Each bootstrap replicate independently resamples every component
        estimator and recomputes the sensitivity interval for the resulting
        linear combination (exploiting the separability property). Percentiles
        of those bootstrap sensitivity bounds are returned as the expanded
        confidence interval. With ``bootstrap=False`` the normal approximation
        is used instead: the sensitivity interval is expanded by
        :math:`z \cdot \text{se}` on each side, with no resampling required.

        Parameters
        ----------
         alpha : float, optional
            P-value threshold. Defaults to 0.10 (90% interval).
         gamma : float, optional
            The Gamma factor. Must be >= 1.0. Defaults to 6.
         alternative : ["two-sided", "less", "greater"], optional
            Defaults to "two-sided".
         bootstrap : bool, optional
            If True (default), use the percentile bootstrap. If False, use
            the normal approximation (no resampling).
         B : int, optional
            Number of bootstrap replications. Ignored when
            ``bootstrap=False``. Defaults to 1_000.
         seed : int, list_like, etc
            A seed for numpy.random.default_rng. Ignored when
            ``bootstrap=False``.

        Returns
        -------
         lb, ub : float
            Lower and upper bounds on the confidence interval. For one-sided
            intervals, only one of these will be finite.

        """
        if gamma < 1.0:
            raise ValueError("`gamma` must be >= 1")

        if gamma == 1.0 or not self.terms:
            return self.confidence_interval(alpha=alpha, alternative=alternative)

        if not bootstrap:
            return super().expanded_confidence_interval(
                alpha=alpha,
                gamma=gamma,
                alternative=alternative,
                bootstrap=False,
            )

        lb_bootstrap = np.zeros(B)
        ub_bootstrap = np.zeros(B)
        for b, lce in enumerate(self.resample(B, seed)):
            lb_bootstrap[b], ub_bootstrap[b] = lce.sensitivity_analysis(gamma=gamma)

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
