"""Linear combination estimator."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from pyrake.estimation.base_classes import (
    WeightingEstimator,
)


class LinearCombinationEstimator(WeightingEstimator):
    """Affine combination of WeightingEstimators.

    Estimates:

       θ = affine + Σᵢ cᵢθᵢ,

    where each θᵢ is estimated by the corresponding component estimator. Component
    estimators may be ``MeanEstimator``, ``TreatmentEffectEstimator``, or other
    ``LinearCombinationEstimator`` objects.

    Sensitivity analysis is valid because the objective is separable across components:
    hidden bias in each component contributes independently, and the worst-case bounds
    combine linearly (with sign-aware bound selection).

    Parameters
    ----------
     terms : list of (float, WeightingEstimator) pairs
        Coefficients and their associated estimators.
     affine : float, optional
        Constant offset. Defaults to 0.0.

    Notes
    -----
    Variance is computed assuming independence across component estimators:

       var(θ) = Σᵢ cᵢ²var(θᵢ).

    This is exact when components are fit on disjoint datasets (e.g. different treatment
    arms or independent studies). If components share data, variance will be understated
    and variance-based methods (``confidence_interval``, ``pvalue``) will be
    anti-conservative. Bootstrap-based methods (``expanded_confidence_interval``) carry
    the same independence assumption since each component is resampled independently.

    """

    def __init__(
        self,
        terms: list[tuple[float, WeightingEstimator]],
        affine: float = 0.0,
    ) -> None:
        self.terms = terms
        self.affine = affine

    def point_estimate(self) -> float:
        """Calculate the point estimate.

        Returns
        -------
         estimate : float
            Offset plus the coefficient-weighted sum of component estimates.

        """
        return self.affine + sum(c * e.point_estimate() for c, e in self.terms)

    def variance(self) -> float:
        """Calculate the variance.

        Assumes independence across component estimators. See class docstring.

        Returns
        -------
         variance : float
            Coefficient-weighted sum of component variances.

        """
        return sum(c**2 * e.variance() for c, e in self.terms)

    def _sensitivity_variance(self, gamma: float) -> tuple[float, float]:
        """Return variance at the worst-case lower- and upper-bound weights.

        Mirrors the sign-aware bound selection in ``sensitivity_analysis``: for
        coefficient c and component variance pair (e_lb_var, e_ub_var):

          - c >= 0: contributes c**2 * e_lb_var to the overall lower-bound variance and
            c**2 * e_ub_var to the upper-bound variance.
          - c < 0: contributes c**2 * e_ub_var to the lower-bound variance and c**2 *
            e_lb_var to the upper-bound variance.

        Parameters
        ----------
         gamma : float
            The Gamma factor. Must be >= 1.0.

        Returns
        -------
         lb_var, ub_var : float
            Variance at the lower- and upper-bound sensitivity weights.

        """
        lb_var = 0.0
        ub_var = 0.0
        for c, e in self.terms:
            e_lb_var, e_ub_var = e._sensitivity_variance(gamma)
            if c >= 0:
                lb_var += c**2 * e_lb_var
                ub_var += c**2 * e_ub_var
            else:
                lb_var += c**2 * e_ub_var
                ub_var += c**2 * e_lb_var
        return lb_var, ub_var

    def sensitivity_analysis(
        self,
        gamma: float = 6.0,
        outcome_proxies: list[float] | npt.NDArray[np.float64] | None = None,
        proxy_mean: float | None = None,
        alpha: float = 0.10,
        *,
        proxy_kwargs: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """Perform a sensitivity analysis.

        The analysis is separable across components. For coefficient c and component
        sensitivity interval [lb_i, ub_i]:
          - c >= 0: contributes c * lb_i to the overall lower bound and c * ub_i to the
            overall upper bound.
          - c < 0: contributes c * ub_i to the overall lower bound and c * lb_i to the
            overall upper bound.

        The affine term shifts both bounds by the same constant.

        Parameters
        ----------
         gamma : float, optional
            The Gamma factor. Must be >= 1.0, with 1.0 indicating perfect propensity
            scores. Defaults to 6. See ``WeightingEstimator``.
         outcome_proxies : array_like, optional
            Accepted for override compatibility with ``WeightingEstimator``; must be
            None. Use ``proxy_kwargs`` instead.
         proxy_mean : float, optional
            Accepted for override compatibility with ``WeightingEstimator``; must be
            None. Use ``proxy_kwargs`` instead.
         alpha : float, optional
            Significance level for the proxy tolerance. Defaults to 0.10.
         proxy_kwargs : list of dict, optional
            Per-component keyword arguments forwarded to each component's
            ``sensitivity_analysis``. Must have the same length as ``terms``. Each dict
            is unpacked into the corresponding component's call, so the keys must match
            that component's signature. Use an empty dict ``{}`` for components that
            need no proxy arguments. For example, a ``MeanEstimator`` component accepts
            ``outcome_proxies`` and ``proxy_mean``, while a ``TreatmentEffectEstimator``
            accepts ``control_outcome_proxies``, ``treated_outcome_proxies``, and
            ``proxy_mean``.
         kwargs : dict, optional
            Accepted for override compatibility with ``WeightingEstimator``; must be
            empty.

        Returns
        -------
         lb, ub : float
            Lower and upper bounds on the point estimate.

        Raises
        ------
         TypeError
            If ``outcome_proxies``, ``proxy_mean``, or unexpected keyword arguments
            are supplied.
         ValueError
            If ``proxy_kwargs`` disagrees with ``terms`` in length, or if ``gamma`` < 1
            (propagated from the component ``sensitivity_analysis`` calls).

        """
        if outcome_proxies is not None or proxy_mean is not None:
            raise TypeError(
                "`outcome_proxies` and `proxy_mean` are not supported here; use "
                "`proxy_kwargs`."
            )
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}.")
        if proxy_kwargs is not None and len(proxy_kwargs) != len(self.terms):
            raise ValueError(
                f"proxy_kwargs has {len(proxy_kwargs)} elements "
                f"but there are {len(self.terms)} terms."
            )

        lb = self.affine
        ub = self.affine
        for i, (c, e) in enumerate(self.terms):
            kw: dict[str, Any] = proxy_kwargs[i] if proxy_kwargs is not None else {}
            e_lb, e_ub = e.sensitivity_analysis(gamma, alpha=alpha, **kw)
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
    ) -> Generator[LinearCombinationEstimator, None, None]:
        """Yield a sequence of resampled estimators.

        Each of the B bootstrap iterations yields a new LinearCombinationEstimator whose
        components are independently resampled from their respective component
        estimators.

        Parameters
        ----------
         B : int
            Number of bootstrap replications.
         seed : int, list_like, etc
            A seed for ``numpy.random.default_rng``. See that documentation.

        Yields
        ------
         estimator : ``LinearCombinationEstimator``
            Bootstrap-resampled estimator.

        """
        if not self.terms:
            # No stochastic components; yield B identical copies so that
            # this estimator composes correctly inside another LCE.
            for _ in range(B):
                yield LinearCombinationEstimator(terms=[], affine=self.affine)
            return

        if isinstance(seed, np.random.SeedSequence):
            ss = seed
        elif isinstance(seed, (np.random.Generator, np.random.BitGenerator)):
            rng = np.random.default_rng(seed)
            ss = np.random.SeedSequence(int(rng.integers(2**63)))
        else:
            ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(len(self.terms))
        resamplers = [
            e.resample(B, seed=cs)
            for cs, (_, e) in zip(child_seeds, self.terms, strict=True)
        ]
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

        Each bootstrap replicate independently resamples every component estimator and
        recomputes the sensitivity interval for the resulting linear combination
        (exploiting the separability property). Percentiles of those bootstrap
        sensitivity bounds are returned as the expanded confidence interval. With
        ``bootstrap=False`` the normal approximation is used instead: the sensitivity
        interval is expanded by a multiple of the standard error on each side, with no
        resampling required.

        Parameters
        ----------
         alpha : float, optional
            P-value threshold. Defaults to 0.10 (90% interval).
         gamma : float, optional
            The Gamma factor. Must be >= 1.0. Defaults to 6.
         alternative : ["two-sided", "less", "greater"], optional
            Defaults to "two-sided".
         bootstrap : bool, optional
            If True (default), use the percentile bootstrap. If False, use the normal
            approximation (no resampling).
         B : int, optional
            Number of bootstrap replications. Ignored when ``bootstrap=False``. Defaults
            to 1_000.
         seed : int, list_like, etc
            A seed for ``numpy.random.default_rng``. Ignored when ``bootstrap=False``.

        Returns
        -------
         lb, ub : float
            Lower and upper bounds on the confidence interval. For one-sided intervals,
            only one of these will be finite.

        Raises
        ------
         ValueError
            If ``gamma`` < 1 or ``alternative`` is not recognized.

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
