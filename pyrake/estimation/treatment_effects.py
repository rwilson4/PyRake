"""Treatment effect estimators."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes

from pyrake.estimation.base_classes import (
    Estimand,
    SimpleEstimand,
    WeightingEstimator,
)
from pyrake.estimation.population import (
    AIPWEstimator,
    IPWEstimator,
    MeanEstimator,
    NonRespondentMean,
    PopulationMean,
    RatioEstimator,
    SAIPWEstimator,
    SampleMean,
    SIPWEstimator,
)


class DoubleSamplingEstimand(Estimand):
    """Estimand combining two sampling mechanisms.

    Parameters
    ----------
     estimand1, estimand2 : ``SimpleEstimand``
        The two sampling-mechanism estimands whose weights multiply.

    """

    def __init__(self, estimand1: SimpleEstimand, estimand2: SimpleEstimand) -> None:
        self.estimand1 = estimand1
        self.estimand2 = estimand2

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Calculate weights based on propensity scores.

        Returns
        -------
         weights : (M,) ``npt.NDArray[np.float64]``
            Elementwise product of the two mechanism weight vectors.

        """
        return self.estimand1.weights * self.estimand2.weights

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
            Elementwise products of the two mechanisms' bound vectors.

        """
        wl1, wu1 = self.estimand1.sensitivity_region(gamma)
        wl2, wu2 = self.estimand2.sensitivity_region(gamma)
        return wl1 * wl2, wu1 * wu2

    def resample(
        self, bootstrap_indices: npt.NDArray[np.int64]
    ) -> DoubleSamplingEstimand:
        """Create a new Estimand object based on resampled indices.

        Parameters
        ----------
         bootstrap_indices : (B,) ``npt.NDArray[np.int64]``
            Row positions to retain in the resampled estimand.

        Returns
        -------
         estimand : ``DoubleSamplingEstimand``
            New estimand built from the indexed rows of both mechanisms.

        """
        return DoubleSamplingEstimand(
            self.estimand1.resample(bootstrap_indices),
            self.estimand2.resample(bootstrap_indices),
        )

    @staticmethod
    def normalizing_factor(sample_size: int, population_size: int) -> float:
        """Calculate normalizing factor.

        Raises
        ------
         NotImplementedError
            Always raised; double sampling has no single normalizing factor.

        """
        raise NotImplementedError("Not implemented")


class TreatmentEffectEstimator(WeightingEstimator):
    """Estimator for the difference between treated and control means.

    Parameters
    ----------
     control_estimator, treated_estimator : ``MeanEstimator`` or ``RatioEstimator``
        Estimators for the control- and treated-group quantities. The point estimate is
        their difference.

    """

    def __init__(
        self,
        control_estimator: MeanEstimator | RatioEstimator,
        treated_estimator: MeanEstimator | RatioEstimator,
    ) -> None:
        self.control_estimator = control_estimator
        self.treated_estimator = treated_estimator

    def point_estimate(self) -> float:
        """Calculate a point estimate.

        Returns
        -------
         estimate : float
            Treated point estimate minus control point estimate.

        """
        return (
            self.treated_estimator.point_estimate()
            - self.control_estimator.point_estimate()
        )

    def variance(self) -> float:
        """Calculate the variance.

        Returns
        -------
         variance : float
            Sum of the treated and control sampling variances.

        """
        return self.treated_estimator.variance() + self.control_estimator.variance()

    def _sensitivity_variance(self, gamma: float) -> tuple[float, float]:
        """Return variance at the worst-case lower- and upper-bound weights.

        The lower bound of the treatment-effect sensitivity interval is ``treated_lb -
        control_ub``, so the variance at the lower-bound weights is the sum of the
        treated lower-bound variance and the control upper-bound variance. The upper
        bound is the reverse.

        Parameters
        ----------
         gamma : float
            The Gamma factor. Must be >= 1.0.

        Returns
        -------
         lb_var, ub_var : float
            Variance at the lower- and upper-bound sensitivity weights.

        """
        treated_lb_var, treated_ub_var = self.treated_estimator._sensitivity_variance(
            gamma
        )
        control_lb_var, control_ub_var = self.control_estimator._sensitivity_variance(
            gamma
        )
        return treated_lb_var + control_ub_var, treated_ub_var + control_lb_var

    def pvalue(
        self,
        null_value: float = 0.0,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ) -> float:
        """Calculate a p-value.

        Computes a p-value against the null hypothesis:
            H0: TE = ``null_value``,
        where TE is the treatment effect.

        Parameters
        ----------
         null_value : float
            The hypothesized treatment effect.
         alternative : ["two-sided", "less", "greater"], optional
            What kind of test:
              - "two-sided": H0: TE = ``null_value`` vs Halt: TE <>
                ``null_value``.
              - "greater": H0: TE <= ``null_value`` vs Halt: TE > ``null_value``.
              - "less": H0: TE >= ``null_value`` vs Halt: TE < ``null_value``.
            For example, specifying alternative = "greater" returns a p-value that
            quantifies the strength of evidence against the null hypothesis that the
            treatment effect is less than ``null_value``, in favor of the alternative
            hypothesis that the treatment effect is actually greater than
            ``null_value``. Defaults to "two-sided".

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

    def adjusted_pvalue(
        self,
        null_value: float = 0.0,
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

        See ``WeightingEstimator.adjusted_pvalue`` for full documentation.

        Parameters
        ----------
         null_value : float, optional
            The hypothesized treatment effect. Defaults to 0.0.
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
         p : float
            Adjusted p-value.

        Raises
        ------
         ValueError
            If ``gamma`` < 1 or ``alternative`` is not recognized (propagated from
            ``WeightingEstimator.adjusted_pvalue``).

        """
        return super().adjusted_pvalue(
            null_value=null_value,
            gamma=gamma,
            alternative=alternative,
            bootstrap=bootstrap,
            B=B,
            seed=seed,
        )

    def confidence_interval(
        self,
        alpha: float = 0.10,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ) -> tuple[float, float]:
        """Calculate confidence interval on the treatment effect.

        Parameters
        ----------
         alpha : float, optional
            P-value threshold, e.g. specify alpha=0.05 for a 95% confidence interval.
            Defaults to 0.10, corresponding to a 90% confidence interval.
         alternative : ["two-sided", "less", "greater"], optional
            What kind of test:
              - "two-sided": H0: TE = ``null_value`` vs Halt: TE <>
                ``null_value``.
              - "greater": H0: TE <= ``null_value`` vs Halt: TE > ``null_value``.
              - "less": H0: TE >= ``null_value`` vs Halt: TE < ``null_value``.
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

    def sensitivity_analysis(
        self,
        gamma: float = 6.0,
        outcome_proxies: list[float] | npt.NDArray[np.float64] | None = None,
        proxy_mean: float | None = None,
        alpha: float = 0.10,
        *,
        control_outcome_proxies: list[float] | npt.NDArray[np.float64] | None = None,
        treated_outcome_proxies: list[float] | npt.NDArray[np.float64] | None = None,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """Perform a sensitivity analysis.

        Calculate a range of point estimates implied by the Gamma factor.

        Parameters
        ----------
         gamma : float, optional
            The Gamma factor. Must be >= 1.0, with 1.0 indicating perfect propensity
            scores. Defaults to 6. See Notes.
         outcome_proxies : array_like, optional
            Accepted for override compatibility with ``WeightingEstimator``; must be
            None. Use ``control_outcome_proxies`` and ``treated_outcome_proxies``
            instead.
         proxy_mean : float, optional
            Known population mean of the outcome proxy. Required when either
            ``control_outcome_proxies`` or ``treated_outcome_proxies`` is supplied.
         alpha : float, optional
            Significance level for the proxy tolerance. Defaults to 0.10.
         control_outcome_proxies : array_like, optional
            A proxy for the outcome observed for every control-group unit, with a known
            population mean given by ``proxy_mean``. When supplied (together with
            ``treated_outcome_proxies`` and ``proxy_mean``), a constraint is added to
            the control-arm sensitivity region requiring the weighted proxy estimate to
            be within the estimator's confidence-interval half-width of ``proxy_mean``.
            Must have the same length as the control-group outcomes. All three of
            ``control_outcome_proxies``, ``treated_outcome_proxies``, and ``proxy_mean``
            must be provided together.
         treated_outcome_proxies : array_like, optional
            A proxy for the outcome observed for every treated-group unit. See
            ``control_outcome_proxies`` for details.
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
            If ``outcome_proxies`` or unexpected keyword arguments are supplied.
         ValueError
            If proxies are supplied without ``proxy_mean``, if ``proxy_mean`` is
            supplied without proxies, or if ``gamma`` < 1 (propagated from the arm-level
            ``sensitivity_analysis`` calls).

        Notes
        -----
        The underlying sensitivity analysis is separable in the control vs treated
        individuals, so we perform two separate sensitivity analyses and combine the
        results.

        """
        if outcome_proxies is not None:
            raise TypeError(
                "`outcome_proxies` is not supported here; use "
                "`control_outcome_proxies` and `treated_outcome_proxies`."
            )
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}.")
        has_proxies = (
            control_outcome_proxies is not None or treated_outcome_proxies is not None
        )
        if has_proxies and proxy_mean is None:
            raise ValueError(
                "`proxy_mean` is required when `control_outcome_proxies` or "
                "`treated_outcome_proxies` is provided."
            )
        if proxy_mean is not None and not has_proxies:
            raise ValueError(
                "`proxy_mean` was provided but neither `control_outcome_proxies` "
                "nor `treated_outcome_proxies` was specified."
            )

        control_lb, control_ub = self.control_estimator.sensitivity_analysis(
            gamma=gamma,
            outcome_proxies=control_outcome_proxies,
            proxy_mean=proxy_mean,
            alpha=alpha,
        )
        treated_lb, treated_ub = self.treated_estimator.sensitivity_analysis(
            gamma=gamma,
            outcome_proxies=treated_outcome_proxies,
            proxy_mean=proxy_mean,
            alpha=alpha,
        )

        return treated_lb - control_ub, treated_ub - control_lb

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
    ) -> Generator[TreatmentEffectEstimator, None, None]:
        """Yield a sequence of resampled estimators.

        Parameters
        ----------
         B : int
            Number of bootstrap replications to run.
         seed : int, list_like, etc
            A seed for ``numpy.random.default_rng``. See that documentation for details.

        Yields
        ------
         estimator : ``TreatmentEffectEstimator``
            Bootstrap-resampled estimator.

        """
        # Annotated so mypy keeps the union rather than joining the two generator
        # types to their common base, WeightingEstimator.
        control_resamples: Iterator[MeanEstimator | RatioEstimator] = (
            self.control_estimator.resample(B, seed=seed)
        )
        treated_resamples: Iterator[MeanEstimator | RatioEstimator] = (
            self.treated_estimator.resample(B, seed=seed)
        )
        for control_est, treated_est in zip(
            control_resamples, treated_resamples, strict=False
        ):
            yield TreatmentEffectEstimator(
                control_estimator=control_est,
                treated_estimator=treated_est,
            )

    def plot_sensitivity(
        self,
        gamma_lower: float = 1.0,
        gamma_upper: float = 6.0,
        num_points: int = 50,
        alpha: float = 0.10,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        bootstrap: bool = True,
        B: int = 1_000,
        null_value: float | None = 0.0,
        title: str | None = None,
        ylabel: str = "Sensitivity Interval",
        ytick_format: str = ".0%",
        axis_label_size: int | None = None,
        tick_label_size: int | None = None,
        legend_label_size: int | None = None,
        legend_placement: str | None = None,
        ax: Axes | None = None,
    ) -> tuple[pd.DataFrame, Axes]:
        """Plot like ``WeightingEstimator.plot_sensitivity`` with ``null_value`` 0."""
        return super().plot_sensitivity(
            gamma_lower=gamma_lower,
            gamma_upper=gamma_upper,
            num_points=num_points,
            alpha=alpha,
            alternative=alternative,
            bootstrap=bootstrap,
            B=B,
            null_value=null_value,
            title=title,
            ylabel=ylabel,
            ytick_format=ytick_format,
            axis_label_size=axis_label_size,
            tick_label_size=tick_label_size,
            legend_label_size=legend_label_size,
            legend_placement=legend_placement,
            ax=ax,
        )


class SimpleDifferenceEstimator(TreatmentEffectEstimator):
    """Simple difference in means.

    Parameters
    ----------
     control_outcomes, treated_outcomes : list_like
        Outcomes or responses for control and treatment groups, resp.
     control_sampling_propensity_scores : list_like, optional
        Propensity scores for the sampling mechanism for the control group. See Notes.
     treated_sampling_propensity_scores : list_like, optional
        Propensity scores for the sampling mechanism for the treated group. See Notes.
     sampling_estimand_class : str, optional
        One of "PopulationMean", "NonRespondentMean", or "SampleMean". Population for
        which to estimate the average treatment effect. Used only when sampling
        propensity scores are specified. Defaults to "PopulationMean" in that case. See
        Notes.
     control_denominator_outcomes, treated_denominator_outcomes : list_like, optional
        If specified, the metric of interest is considered a ratio (with
        ``control_outcomes`` and ``treated_outcomes`` treated as the numerators), and
        interest is in the effect of treatment on that ratio. See Notes in
        ``ATEEstimator`` for details.
     kwargs : dict, optional
        Extra arguments forwarded to the estimator constructor.

    Notes
    -----
    When called without sampling propensity scores, this class computes the average
    response in treatment, minus the average response in control. No weights, just
    straight averages. This may be helpful to see how different the adjusted estimator
    is from the simplest comparison of treatment and control, to see how biased this
    simple estimator is.

    In situations where the treatment is randomly assigned in the population (but
    potentially not in the sample), this estimator can also be used to generalize
    estimated treatment effects from the sample to the population (Tipton and Hartman,
    2023). The basic strategy here is to use the control responses to estimate the
    average potential outcome Y(0) in the population, and the treated responses to
    estimate the average Y(1). Since treatment is randomly assigned in the population,
    the population average treatment effect is just the difference in these means.

    References
    ----------
    - Tipton, Elizabeth and Erin Hartman. 2023. "Generalizability and
      Transportability." In Handbook of Matching and Weighting Adjustments for Causal
      Inference, , pg. 39–59. Chapman and Hall/CRC.

    """

    def __init__(
        self,
        control_outcomes: list[float | int] | npt.NDArray[np.float64 | np.int64],
        treated_outcomes: list[float | int] | npt.NDArray[np.float64 | np.int64],
        control_sampling_propensity_scores: (
            list[float] | npt.NDArray[np.float64] | None
        ) = None,
        treated_sampling_propensity_scores: (
            list[float] | npt.NDArray[np.float64] | None
        ) = None,
        sampling_estimand_class: Literal[
            "PopulationMean", "NonRespondentMean", "SampleMean"
        ] = "PopulationMean",
        control_denominator_outcomes: (
            list[float | int] | npt.NDArray[np.float64 | np.int64] | None
        ) = None,
        treated_denominator_outcomes: (
            list[float | int] | npt.NDArray[np.float64 | np.int64] | None
        ) = None,
        **kwargs: Any,
    ) -> None:
        control_outcomes = np.asarray(control_outcomes)
        treated_outcomes = np.asarray(treated_outcomes)

        if control_sampling_propensity_scores is None:
            control_estimand: Estimand = SampleMean(
                np.ones_like(control_outcomes, dtype=np.float64)
            )
        else:
            control_estimand = DoubleSamplingEstimand(
                SampleMean(np.ones_like(control_outcomes, dtype=np.float64)),
                PopulationMean(np.asarray(control_sampling_propensity_scores)),
            )

        if treated_sampling_propensity_scores is None:
            treated_estimand: Estimand = SampleMean(
                np.ones_like(treated_outcomes, dtype=np.float64)
            )
        else:
            treated_estimand = DoubleSamplingEstimand(
                SampleMean(np.ones_like(treated_outcomes, dtype=np.float64)),
                PopulationMean(np.asarray(treated_sampling_propensity_scores)),
            )

        if (
            control_denominator_outcomes is not None
            and treated_denominator_outcomes is not None
        ):
            # why: asarray preserves the float|int dtype the signature needs.
            control_denominator_outcomes = cast(
                npt.NDArray[np.float64 | np.int64],
                np.asarray(control_denominator_outcomes),
            )
            treated_denominator_outcomes = cast(
                npt.NDArray[np.float64 | np.int64],
                np.asarray(treated_denominator_outcomes),
            )
            control_estimator: MeanEstimator | RatioEstimator = RatioEstimator(
                propensity_scores=np.ones_like(control_outcomes, dtype=np.float64),
                numerator_outcomes=control_outcomes,
                denominator_outcomes=control_denominator_outcomes,
                estimand=control_estimand,
            )
            treated_estimator: MeanEstimator | RatioEstimator = RatioEstimator(
                propensity_scores=np.ones_like(treated_outcomes, dtype=np.float64),
                numerator_outcomes=treated_outcomes,
                denominator_outcomes=treated_denominator_outcomes,
                estimand=treated_estimand,
            )
        elif (
            control_denominator_outcomes is not None
            or treated_denominator_outcomes is not None
        ):
            raise ValueError(
                "Must specify neither or both `control_denominator_outcomes` and `treated_denominator_outcomes`"
            )
        else:
            control_estimator = SIPWEstimator(
                propensity_scores=np.ones_like(control_outcomes, dtype=np.float64),
                outcomes=control_outcomes,
                estimand=control_estimand,
            )
            treated_estimator = SIPWEstimator(
                propensity_scores=np.ones_like(treated_outcomes, dtype=np.float64),
                outcomes=treated_outcomes,
                estimand=treated_estimand,
            )

        super().__init__(
            control_estimator=control_estimator,
            treated_estimator=treated_estimator,
        )


class ATEEstimator(TreatmentEffectEstimator):
    """Estimator for the average treatment effect.

    Parameters
    ----------
     control_propensity_scores, treated_propensity_scores : list_like
        Propensity scores for control and treatment groups, resp.
     control_outcomes, treated_outcomes : list_like
        Outcomes or responses for control and treatment groups, resp. Should have the
        same lengths as the corresponding ``propensity_scores``.
     estimator_class : ["IPW", "AIPW", "SIPW", "SAIPW"], optional
        What kind of estimator to use. Defaults to "SIPW".
     control_estimator_class : ["IPW", "AIPW", "SIPW", "SAIPW"], optional
        Estimator class for the control arm. Defaults to ``estimator_class``.
     treated_estimator_class : ["IPW", "AIPW", "SIPW", "SAIPW"], optional
        Estimator class for the treated arm. Defaults to ``estimator_class``.
     control_sampling_propensity_scores : list_like, optional
        Propensity scores for the sampling mechanism for the control group. See Notes.
     treated_sampling_propensity_scores : list_like, optional
        Propensity scores for the sampling mechanism for the treated group. See Notes.
     sampling_estimand_class : str, optional
        One of "PopulationMean", "NonRespondentMean", or "SampleMean". Population for
        which to estimate the average treatment effect. Used only when sampling
        propensity scores are specified. Defaults to "PopulationMean" in that case. See
        Notes.
     control_denominator_outcomes, treated_denominator_outcomes : list_like, optional
        If specified, the metric of interest is considered a ratio (with
        ``control_outcomes`` and ``treated_outcomes`` treated as the numerators), and
        interest is in the effect of treatment on that ratio. See Notes.
     control_kwargs, treated_kwargs, kwargs : dict_like
        Extra kwargs to pass to the estimators. ``kwargs`` are passed to both
        estimators, while ``control_kwargs`` is only passed to the control estimator and
        ``treated_kwargs`` is only passed to the treated estimator.

    Notes
    -----
    Weighting estimators for the average treatment effect tend to look like:

       ÂTE = Σ_{i∈T} wᵢyᵢ/Σ_{i∈T} wᵢ − Σ_{i∈C} wᵢyᵢ/Σ_{i∈C} wᵢ,

    where sums run over treated (T) and control (C) units, wᵢ = 1/πᵢ for the treated
    group, and 1/(1 − πᵢ) for the control group, and πᵢ is the propensity score. This is
    really just the difference between two ``MeanEstimator`` objects, which makes
    calculations (including sensitivity analysis) straightforward.

    When the units in the study are sampled from a population, and the sample is not a
    simple random sample, we may wish to estimate treatment effects in the population,
    which involves elements of both observational causal inference and survey
    non-response. If sampling propensity scores are available, in additional to
    treatment assignment propensity scores, we can adjust for both non-response bias and
    treatment selection bias.

    Sometimes we are interested in a treatment effect in a subpopulation that is
    identifiable only in the sample. For example, people may self-identify as belonging
    to that population in the sample. We can estimate the population mean among the
    sub-population as a ratio of two ``MeanEstimator`` objects as follows. Let tᵢ be 1
    if person i belongs to the sub-population, and 0 otherwise. We observe tᵢ only for
    respondents. Let yᵢ be the metric of interest. Then the average value of the metric
    among the sub-population is (sums run over i = 1, ..., N):

       Σᵢ tᵢyᵢ/Σᵢ tᵢ = (1/N) Σᵢ tᵢyᵢ/(1/N) Σᵢ tᵢ,

    which is the ratio of the population means of tᵢ × yᵢ and tᵢ. Inference (including
    sensitivity analysis) for such ratios is provided by ``RatioEstimator``.

    We may wish to estimate the impact of a treatment on an outcome in a subpopulation
    identifiable only in the sample. Let (y0ᵢ, y1ᵢ) be potential outcomes, and assume
    subpopulation membership is not influenced by treatment, so that tᵢ does not have
    associated potential outcomes. The quantity of interest is (sums run over i = 1,
    ..., N):

       Σᵢ tᵢ(y1ᵢ − y0ᵢ)/Σᵢ tᵢ
          = (1/N) Σᵢ tᵢy1ᵢ/(1/N) Σᵢ tᵢ − (1/N) Σᵢ tᵢy0ᵢ/(1/N) Σᵢ tᵢ.

    By specifying tᵢ × yᵢ as the ``control_outcomes`` and ``treated_outcomes`` (the
    numerator), and tᵢ as the ``control_denominator_outcomes`` and
    ``treated_denominator_outcomes``, inference (including sensitivity analysis) for
    this use case is supported.

    """

    def __init__(
        self,
        control_propensity_scores: list[float] | npt.NDArray[np.float64],
        control_outcomes: list[float | int] | npt.NDArray[np.float64 | np.int64],
        treated_propensity_scores: list[float] | npt.NDArray[np.float64],
        treated_outcomes: list[float | int] | npt.NDArray[np.float64 | np.int64],
        estimator_class: Literal["IPW", "AIPW", "SIPW", "SAIPW"] = "SIPW",
        control_estimator_class: None | (
            Literal["IPW", "AIPW", "SIPW", "SAIPW"]
        ) = None,
        treated_estimator_class: None | (
            Literal["IPW", "AIPW", "SIPW", "SAIPW"]
        ) = None,
        control_sampling_propensity_scores: (
            list[float] | npt.NDArray[np.float64] | None
        ) = None,
        treated_sampling_propensity_scores: (
            list[float] | npt.NDArray[np.float64] | None
        ) = None,
        sampling_estimand_class: Literal[
            "PopulationMean", "NonRespondentMean", "SampleMean"
        ] = "PopulationMean",
        control_denominator_outcomes: (
            list[float | int] | npt.NDArray[np.float64 | np.int64] | None
        ) = None,
        treated_denominator_outcomes: (
            list[float | int] | npt.NDArray[np.float64 | np.int64] | None
        ) = None,
        control_kwargs: dict[str, Any] | None = None,
        treated_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        control_propensity_scores = np.asarray(control_propensity_scores)
        control_outcomes = np.asarray(control_outcomes)
        treated_propensity_scores = np.asarray(treated_propensity_scores)
        treated_outcomes = np.asarray(treated_outcomes)

        control_kwargs_nn = dict(**kwargs)
        if control_kwargs is not None:
            control_kwargs_nn.update(**control_kwargs)

        treated_kwargs_nn = dict(**kwargs)
        if treated_kwargs is not None:
            treated_kwargs_nn.update(**treated_kwargs)

        classes = {
            "IPW": IPWEstimator,
            "AIPW": AIPWEstimator,
            "SIPW": SIPWEstimator,
            "SAIPW": SAIPWEstimator,
        }

        sampling_estimand_classes: dict[
            str,
            type[PopulationMean | NonRespondentMean | SampleMean],
        ] = {
            "PopulationMean": PopulationMean,
            "NonRespondentMean": NonRespondentMean,
            "SampleMean": SampleMean,
        }
        sampling_estimand: type[PopulationMean | NonRespondentMean | SampleMean] = (
            sampling_estimand_classes[sampling_estimand_class]
        )

        if control_sampling_propensity_scores is None:
            control_estimand: Estimand = PopulationMean(
                np.ones_like(control_propensity_scores) - control_propensity_scores
            )
        else:
            control_estimand = DoubleSamplingEstimand(
                PopulationMean(
                    np.ones_like(control_propensity_scores) - control_propensity_scores
                ),
                sampling_estimand(np.asarray(control_sampling_propensity_scores)),
            )

        if treated_sampling_propensity_scores is None:
            treated_estimand: Estimand = PopulationMean(
                np.asarray(treated_propensity_scores, dtype=np.float64)
            )
        else:
            treated_estimand = DoubleSamplingEstimand(
                PopulationMean(np.asarray(treated_propensity_scores, dtype=np.float64)),
                sampling_estimand(np.asarray(treated_sampling_propensity_scores)),
            )

        if (
            control_denominator_outcomes is not None
            and treated_denominator_outcomes is not None
        ):
            # why: asarray preserves the float|int dtype the signature needs.
            control_denominator_outcomes = cast(
                npt.NDArray[np.float64 | np.int64],
                np.asarray(control_denominator_outcomes),
            )
            treated_denominator_outcomes = cast(
                npt.NDArray[np.float64 | np.int64],
                np.asarray(treated_denominator_outcomes),
            )
            control_estimator = RatioEstimator(
                propensity_scores=control_propensity_scores,
                numerator_outcomes=control_outcomes,
                denominator_outcomes=control_denominator_outcomes,
                estimand=control_estimand,
                estimator_class=(control_estimator_class or estimator_class),
                **control_kwargs_nn,
            )
            treated_estimator = RatioEstimator(
                propensity_scores=treated_propensity_scores,
                numerator_outcomes=treated_outcomes,
                denominator_outcomes=treated_denominator_outcomes,
                estimand=treated_estimand,
                estimator_class=(treated_estimator_class or estimator_class),
                **treated_kwargs_nn,
            )
        elif (
            control_denominator_outcomes is not None
            or treated_denominator_outcomes is not None
        ):
            raise ValueError(
                "Must specify neither or both `control_denominator_outcomes` and `treated_denominator_outcomes`"
            )
        else:
            control_estimator = classes[control_estimator_class or estimator_class](
                propensity_scores=np.ones_like(control_propensity_scores)
                - control_propensity_scores,
                outcomes=control_outcomes,
                estimand=control_estimand,
                **control_kwargs_nn,
            )
            treated_estimator = classes[treated_estimator_class or estimator_class](
                propensity_scores=treated_propensity_scores,
                outcomes=treated_outcomes,
                estimand=treated_estimand,
                **treated_kwargs_nn,
            )

        super().__init__(
            control_estimator=control_estimator,
            treated_estimator=treated_estimator,
        )


class ATTEstimator(TreatmentEffectEstimator):
    """Estimator for the average effect of treatment on the treated (ATT).

    Parameters
    ----------
     control_propensity_scores : list_like
        Propensity scores for control group.
     control_outcomes, treated_outcomes : list_like
        Outcomes or responses for control and treatment groups, resp. Should have the
        same lengths as the corresponding ``propensity_scores``.
     estimator_class : ["IPW", "AIPW", "SIPW", "SAIPW"], optional
        What kind of estimator to use. Defaults to "SIPW".
     control_estimator_class : ["IPW", "AIPW", "SIPW", "SAIPW"], optional
        Estimator class for the control arm. Defaults to ``estimator_class``.
     treated_estimator_class : ["IPW", "AIPW", "SIPW", "SAIPW"], optional
        Estimator class for the treated arm. Defaults to ``estimator_class``.
     control_sampling_propensity_scores : list_like, optional
        Propensity scores for the sampling mechanism for the control group. See Notes.
     treated_sampling_propensity_scores : list_like, optional
        Propensity scores for the sampling mechanism for the treated group. See Notes.
     sampling_estimand_class : str, optional
        One of "PopulationMean", "NonRespondentMean", or "SampleMean". Population for
        which to estimate the average treatment effect. Used only when sampling
        propensity scores are specified. Defaults to "PopulationMean" in that case. See
        Notes.
     control_denominator_outcomes, treated_denominator_outcomes : list_like, optional
        If specified, the metric of interest is considered a ratio (with
        ``control_outcomes`` and ``treated_outcomes`` treated as the numerators), and
        interest is in the effect of treatment on that ratio. See Notes in
        ``ATEEstimator`` for details.
     control_kwargs, treated_kwargs, kwargs : dict_like
        Extra kwargs to pass to the estimators. ``kwargs`` are passed to both
        estimators, while ``control_kwargs`` is only passed to the control estimator and
        ``treated_kwargs`` is only passed to the treated estimator.

    Notes
    -----
    Weighting estimators for the average treatment effect tend to look like:

       ÂTT = Σ_{i∈T} yᵢ/nT − Σ_{i∈C} wᵢyᵢ/Σ_{i∈C} wᵢ,

    where sums run over treated (T) and control (C) units, nT is the number of treated
    units, wᵢ = πᵢ/(1 − πᵢ), and πᵢ is the propensity score. This is really just the
    difference between two ``MeanEstimator`` objects, which makes calculations
    (including sensitivity analysis) straightforward.

    """

    def __init__(
        self,
        control_propensity_scores: list[float] | npt.NDArray[np.float64],
        control_outcomes: list[float | int] | npt.NDArray[np.float64 | np.int64],
        treated_outcomes: list[float | int] | npt.NDArray[np.float64 | np.int64],
        estimator_class: Literal["IPW", "AIPW", "SIPW", "SAIPW"] = "SIPW",
        control_estimator_class: None | (
            Literal["IPW", "AIPW", "SIPW", "SAIPW"]
        ) = None,
        treated_estimator_class: None | (
            Literal["IPW", "AIPW", "SIPW", "SAIPW"]
        ) = None,
        control_sampling_propensity_scores: (
            list[float] | npt.NDArray[np.float64] | None
        ) = None,
        treated_sampling_propensity_scores: (
            list[float] | npt.NDArray[np.float64] | None
        ) = None,
        sampling_estimand_class: Literal[
            "PopulationMean", "NonRespondentMean", "SampleMean"
        ] = "PopulationMean",
        control_denominator_outcomes: (
            list[float | int] | npt.NDArray[np.float64 | np.int64] | None
        ) = None,
        treated_denominator_outcomes: (
            list[float | int] | npt.NDArray[np.float64 | np.int64] | None
        ) = None,
        control_kwargs: dict[str, Any] | None = None,
        treated_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        control_propensity_scores = np.asarray(control_propensity_scores)
        control_outcomes = np.asarray(control_outcomes)
        treated_outcomes = np.asarray(treated_outcomes)

        control_kwargs_nn = dict(**kwargs)
        if control_kwargs is not None:
            control_kwargs_nn.update(**control_kwargs)

        treated_kwargs_nn = dict(**kwargs)
        if treated_kwargs is not None:
            treated_kwargs_nn.update(**treated_kwargs)

        classes = {
            "IPW": IPWEstimator,
            "AIPW": AIPWEstimator,
            "SIPW": SIPWEstimator,
            "SAIPW": SAIPWEstimator,
        }

        sampling_estimand_classes: dict[
            str,
            type[PopulationMean | NonRespondentMean | SampleMean],
        ] = {
            "PopulationMean": PopulationMean,
            "NonRespondentMean": NonRespondentMean,
            "SampleMean": SampleMean,
        }
        sampling_estimand: type[PopulationMean | NonRespondentMean | SampleMean] = (
            sampling_estimand_classes[sampling_estimand_class]
        )

        if control_sampling_propensity_scores is None:
            control_estimand: Estimand = NonRespondentMean(
                np.ones_like(control_propensity_scores) - control_propensity_scores
            )
        else:
            control_estimand = DoubleSamplingEstimand(
                NonRespondentMean(
                    np.ones_like(control_propensity_scores) - control_propensity_scores
                ),
                sampling_estimand(np.asarray(control_sampling_propensity_scores)),
            )

        if treated_sampling_propensity_scores is None:
            treated_estimand: Estimand = SampleMean(
                np.ones_like(treated_outcomes, dtype=np.float64)
            )
        else:
            treated_estimand = DoubleSamplingEstimand(
                SampleMean(np.ones_like(treated_outcomes, dtype=np.float64)),
                sampling_estimand(np.asarray(treated_sampling_propensity_scores)),
            )

        if (
            control_denominator_outcomes is not None
            and treated_denominator_outcomes is not None
        ):
            # why: asarray preserves the float|int dtype the signature needs.
            control_denominator_outcomes = cast(
                npt.NDArray[np.float64 | np.int64],
                np.asarray(control_denominator_outcomes),
            )
            treated_denominator_outcomes = cast(
                npt.NDArray[np.float64 | np.int64],
                np.asarray(treated_denominator_outcomes),
            )
            control_estimator = RatioEstimator(
                propensity_scores=np.ones_like(control_propensity_scores)
                - control_propensity_scores,
                numerator_outcomes=control_outcomes,
                denominator_outcomes=control_denominator_outcomes,
                estimand=control_estimand,
                estimator_class=(control_estimator_class or estimator_class),
                **control_kwargs_nn,
            )
            treated_estimator = RatioEstimator(
                propensity_scores=np.ones_like(treated_outcomes, dtype=np.float64),
                numerator_outcomes=treated_outcomes,
                denominator_outcomes=treated_denominator_outcomes,
                estimand=treated_estimand,
                estimator_class=(treated_estimator_class or estimator_class),
                **treated_kwargs_nn,
            )
        elif (
            control_denominator_outcomes is not None
            or treated_denominator_outcomes is not None
        ):
            raise ValueError(
                "Must specify neither or both `control_denominator_outcomes` and `treated_denominator_outcomes`"
            )
        else:
            control_estimator = classes[control_estimator_class or estimator_class](
                propensity_scores=np.ones_like(control_propensity_scores)
                - control_propensity_scores,
                outcomes=control_outcomes,
                estimand=control_estimand,
                **control_kwargs_nn,
            )
            treated_estimator = classes[treated_estimator_class or estimator_class](
                propensity_scores=np.ones_like(treated_outcomes, dtype=np.float64),
                outcomes=treated_outcomes,
                estimand=treated_estimand,
                **treated_kwargs_nn,
            )

        super().__init__(
            control_estimator=control_estimator,
            treated_estimator=treated_estimator,
        )


class ATCEstimator(TreatmentEffectEstimator):
    """Estimator for the average effect of treatment on the control units (ATC).

    Parameters
    ----------
     treated_propensity_scores : list_like
        Propensity scores for the treatment group.
     control_outcomes, treated_outcomes : list_like
        Outcomes or responses for control and treatment groups, resp. Should have the
        same lengths as the corresponding ``propensity_scores``.
     estimator_class : ["IPW", "AIPW", "SIPW", "SAIPW"], optional
        What kind of estimator to use. Defaults to "SIPW".
     control_estimator_class : ["IPW", "AIPW", "SIPW", "SAIPW"], optional
        Estimator class for the control arm. Defaults to ``estimator_class``.
     treated_estimator_class : ["IPW", "AIPW", "SIPW", "SAIPW"], optional
        Estimator class for the treated arm. Defaults to ``estimator_class``.
     control_sampling_propensity_scores : list_like, optional
        Propensity scores for the sampling mechanism for the control group. See Notes.
     treated_sampling_propensity_scores : list_like, optional
        Propensity scores for the sampling mechanism for the treated group. See Notes.
     sampling_estimand_class : str, optional
        One of "PopulationMean", "NonRespondentMean", or "SampleMean". Population for
        which to estimate the average treatment effect. Used only when sampling
        propensity scores are specified. Defaults to "PopulationMean" in that case. See
        Notes.
     control_denominator_outcomes, treated_denominator_outcomes : list_like, optional
        If specified, the metric of interest is considered a ratio (with
        ``control_outcomes`` and ``treated_outcomes`` treated as the numerators), and
        interest is in the effect of treatment on that ratio. See Notes in
        ``ATEEstimator`` for details.
     control_kwargs, treated_kwargs, kwargs : dict_like
        Extra kwargs to pass to the estimators. ``kwargs`` are passed to both
        estimators, while ``control_kwargs`` is only passed to the control estimator and
        ``treated_kwargs`` is only passed to the treated estimator.

    Notes
    -----
    Weighting estimators for the average treatment effect tend to look like:

       ÂTC = Σ_{i∈T} wᵢyᵢ/Σ_{i∈T} wᵢ − Σ_{i∈C} yᵢ/nC,

    where sums run over treated (T) and control (C) units, nC is the number of control
    units, wᵢ = (1 − πᵢ)/πᵢ, and πᵢ is the propensity score. This is really just the
    difference between two ``MeanEstimator`` objects, which makes calculations
    (including sensitivity analysis) straightforward.

    """

    def __init__(
        self,
        control_outcomes: list[float | int] | npt.NDArray[np.float64 | np.int64],
        treated_propensity_scores: list[float] | npt.NDArray[np.float64],
        treated_outcomes: list[float | int] | npt.NDArray[np.float64 | np.int64],
        estimator_class: Literal["IPW", "AIPW", "SIPW", "SAIPW"] = "SIPW",
        control_estimator_class: Literal["IPW", "AIPW", "SIPW", "SAIPW"] | None = None,
        treated_estimator_class: Literal["IPW", "AIPW", "SIPW", "SAIPW"] | None = None,
        control_sampling_propensity_scores: (
            list[float] | npt.NDArray[np.float64] | None
        ) = None,
        treated_sampling_propensity_scores: (
            list[float] | npt.NDArray[np.float64] | None
        ) = None,
        sampling_estimand_class: Literal[
            "PopulationMean", "NonRespondentMean", "SampleMean"
        ] = "PopulationMean",
        control_denominator_outcomes: (
            list[float | int] | npt.NDArray[np.float64 | np.int64] | None
        ) = None,
        treated_denominator_outcomes: (
            list[float | int] | npt.NDArray[np.float64 | np.int64] | None
        ) = None,
        control_kwargs: dict[str, Any] | None = None,
        treated_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        control_outcomes = np.asarray(control_outcomes)
        treated_propensity_scores = np.asarray(treated_propensity_scores)
        treated_outcomes = np.asarray(treated_outcomes)

        control_kwargs_nn = dict(**kwargs)
        if control_kwargs is not None:
            control_kwargs_nn.update(**control_kwargs)

        treated_kwargs_nn = dict(**kwargs)
        if treated_kwargs is not None:
            treated_kwargs_nn.update(**treated_kwargs)

        classes = {
            "IPW": IPWEstimator,
            "AIPW": AIPWEstimator,
            "SIPW": SIPWEstimator,
            "SAIPW": SAIPWEstimator,
        }

        sampling_estimand_classes: dict[
            str,
            type[PopulationMean | NonRespondentMean | SampleMean],
        ] = {
            "PopulationMean": PopulationMean,
            "NonRespondentMean": NonRespondentMean,
            "SampleMean": SampleMean,
        }
        sampling_estimand: type[PopulationMean | NonRespondentMean | SampleMean] = (
            sampling_estimand_classes[sampling_estimand_class]
        )

        if control_sampling_propensity_scores is None:
            control_estimand: Estimand = SampleMean(
                np.ones_like(control_outcomes, dtype=np.float64)
            )
        else:
            control_estimand = DoubleSamplingEstimand(
                SampleMean(np.ones_like(control_outcomes, dtype=np.float64)),
                sampling_estimand(np.asarray(control_sampling_propensity_scores)),
            )

        if treated_sampling_propensity_scores is None:
            treated_estimand: Estimand = NonRespondentMean(
                np.asarray(treated_propensity_scores, dtype=np.float64)
            )
        else:
            treated_estimand = DoubleSamplingEstimand(
                NonRespondentMean(
                    np.asarray(treated_propensity_scores, dtype=np.float64)
                ),
                sampling_estimand(np.asarray(treated_sampling_propensity_scores)),
            )

        if (
            control_denominator_outcomes is not None
            and treated_denominator_outcomes is not None
        ):
            # why: asarray preserves the float|int dtype the signature needs.
            control_denominator_outcomes = cast(
                npt.NDArray[np.float64 | np.int64],
                np.asarray(control_denominator_outcomes),
            )
            treated_denominator_outcomes = cast(
                npt.NDArray[np.float64 | np.int64],
                np.asarray(treated_denominator_outcomes),
            )
            control_estimator = RatioEstimator(
                propensity_scores=np.ones_like(control_outcomes, dtype=np.float64),
                numerator_outcomes=control_outcomes,
                denominator_outcomes=control_denominator_outcomes,
                estimand=control_estimand,
                estimator_class=(control_estimator_class or estimator_class),
                **control_kwargs_nn,
            )
            treated_estimator = RatioEstimator(
                propensity_scores=treated_propensity_scores,
                numerator_outcomes=treated_outcomes,
                denominator_outcomes=treated_denominator_outcomes,
                estimand=treated_estimand,
                estimator_class=(treated_estimator_class or estimator_class),
                **treated_kwargs_nn,
            )
        elif (
            control_denominator_outcomes is not None
            or treated_denominator_outcomes is not None
        ):
            raise ValueError(
                "Must specify neither or both `control_denominator_outcomes` and `treated_denominator_outcomes`"
            )
        else:
            control_estimator = classes[control_estimator_class or estimator_class](
                propensity_scores=np.ones_like(control_outcomes, dtype=np.float64),
                outcomes=control_outcomes,
                estimand=control_estimand,
                **control_kwargs_nn,
            )
            treated_estimator = classes[treated_estimator_class or estimator_class](
                propensity_scores=treated_propensity_scores,
                outcomes=treated_outcomes,
                estimand=treated_estimand,
                **treated_kwargs_nn,
            )

        super().__init__(
            control_estimator=control_estimator,
            treated_estimator=treated_estimator,
        )


class TreatmentEffectRatioEstimator(WeightingEstimator):
    """Estimate the ratio of two treatment effects in some population.

    Parameters
    ----------
     control_propensity_scores, treated_propensity_scores : list_like
        Propensity scores for control and treatment groups, respectively.
     numerator_control_outcomes, numerator_treated_outcomes : list_like
        Numerator outcomes for control and treated groups.
     denominator_control_outcomes, denominator_treated_outcomes : list_like
        Denominator outcomes for control and treated groups.
     mean_estimator_class : ["IPW", "AIPW", "SIPW", "SAIPW"], optional
        Estimator class to use for all sub-estimators. Defaults to "SIPW".
     treatment_effect_estimator_class : [SimpleDifference, ATE, ATT, ATC], optional
        Estimator class for treatment effects. Defaults to "ATE".
     control_sampling_propensity_scores : list_like, optional
        Propensity scores for the sampling mechanism for the control group. See Notes.
     treated_sampling_propensity_scores : list_like, optional
        Propensity scores for the sampling mechanism for the treated group. See Notes.
     sampling_estimand_class : str, optional
        One of "PopulationMean", "NonRespondentMean", or "SampleMean". Population for
        which to estimate the average treatment effect. Used only when sampling
        propensity scores are specified. Defaults to "PopulationMean" in that case. See
        Notes.
     numerator_control_kwargs, numerator_treated_kwargs : dict, optional
        Extra kwargs to pass to the numerator control and treated estimators.
     denominator_control_kwargs, denominator_treated_kwargs : dict, optional
        Extra kwargs to pass to the denominator control and treated estimators.

    Notes
    -----
    Sensitivity analysis and expanded confidence intervals are not implemented because
    the ratio of differences of linear-fractional problems is not quasilinear and cannot
    be transformed to a linear program easily.

    """

    def __init__(
        self,
        control_propensity_scores: list[float] | npt.NDArray[np.float64],
        treated_propensity_scores: list[float] | npt.NDArray[np.float64],
        numerator_control_outcomes: list[float] | npt.NDArray[np.float64],
        numerator_treated_outcomes: list[float] | npt.NDArray[np.float64],
        denominator_control_outcomes: list[float] | npt.NDArray[np.float64],
        denominator_treated_outcomes: list[float] | npt.NDArray[np.float64],
        mean_estimator_class: Literal["IPW", "AIPW", "SIPW", "SAIPW"] = "SIPW",
        treatment_effect_estimator_class: Literal[
            "SimpleDifference", "ATE", "ATT", "ATC"
        ] = "ATE",
        control_sampling_propensity_scores: (
            list[float] | npt.NDArray[np.float64] | None
        ) = None,
        treated_sampling_propensity_scores: (
            list[float] | npt.NDArray[np.float64] | None
        ) = None,
        sampling_estimand_class: Literal[
            "PopulationMean", "NonRespondentMean", "SampleMean"
        ] = "PopulationMean",
        numerator_control_kwargs: dict[str, Any] | None = None,
        numerator_treated_kwargs: dict[str, Any] | None = None,
        denominator_control_kwargs: dict[str, Any] | None = None,
        denominator_treated_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # Convert inputs to arrays
        control_propensity_scores = np.asarray(control_propensity_scores)
        treated_propensity_scores = np.asarray(treated_propensity_scores)
        numerator_control_outcomes = np.asarray(numerator_control_outcomes)
        numerator_treated_outcomes = np.asarray(numerator_treated_outcomes)
        denominator_control_outcomes = np.asarray(denominator_control_outcomes)
        denominator_treated_outcomes = np.asarray(denominator_treated_outcomes)

        # Classes dictionary for estimators
        classes = {
            "SimpleDifference": SimpleDifferenceEstimator,
            "ATE": ATEEstimator,
            "ATT": ATTEstimator,
            "ATC": ATCEstimator,
        }
        EstimatorClass = classes[treatment_effect_estimator_class]

        self.numerator_estimator: TreatmentEffectEstimator = EstimatorClass(
            control_propensity_scores=control_propensity_scores,
            control_outcomes=numerator_control_outcomes,
            treated_propensity_scores=treated_propensity_scores,
            treated_outcomes=numerator_treated_outcomes,
            estimator_class=mean_estimator_class,
            control_sampling_propensity_scores=control_sampling_propensity_scores,
            treated_sampling_propensity_scores=treated_sampling_propensity_scores,
            sampling_estimand_class=sampling_estimand_class,
            control_kwargs=numerator_control_kwargs,
            treated_kwargs=numerator_treated_kwargs,
        )

        self.denominator_estimator: TreatmentEffectEstimator = EstimatorClass(
            control_propensity_scores=control_propensity_scores,
            control_outcomes=denominator_control_outcomes,
            treated_propensity_scores=treated_propensity_scores,
            treated_outcomes=denominator_treated_outcomes,
            estimator_class=mean_estimator_class,
            control_sampling_propensity_scores=control_sampling_propensity_scores,
            treated_sampling_propensity_scores=treated_sampling_propensity_scores,
            sampling_estimand_class=sampling_estimand_class,
            control_kwargs=denominator_control_kwargs,
            treated_kwargs=denominator_treated_kwargs,
        )

    def point_estimate(self) -> float:
        """Calculate a point estimate.

        Returns
        -------
         estimate : float
            Ratio of the numerator and denominator treatment-effect estimates.

        """
        return (
            self.numerator_estimator.point_estimate()
            / self.denominator_estimator.point_estimate()
        )

    def variance(self) -> float:
        """Calculate the variance of the ratio estimator using the delta method.

        Returns
        -------
         variance : float
            Estimated sampling variance of the ratio estimate.

        Raises
        ------
         ValueError
            If any sub-estimator is itself a ``RatioEstimator``.

        """
        # Numerator treatment effect and variance
        numerator_control_estimator = self.numerator_estimator.control_estimator
        denominator_control_estimator = self.denominator_estimator.control_estimator
        numerator_treated_estimator = self.numerator_estimator.treated_estimator
        denominator_treated_estimator = self.denominator_estimator.treated_estimator
        if (
            isinstance(numerator_control_estimator, RatioEstimator)
            or isinstance(denominator_control_estimator, RatioEstimator)
            or isinstance(numerator_treated_estimator, RatioEstimator)
            or isinstance(denominator_treated_estimator, RatioEstimator)
        ):
            raise ValueError(
                "Ratios of treatment effects of RatioEstimators not supported."
            )

        mu_num = self.numerator_estimator.point_estimate()
        var_num = self.numerator_estimator.variance()
        mu_den = self.denominator_estimator.point_estimate()
        var_den = self.denominator_estimator.variance()

        wc = numerator_control_estimator.weights
        mu_num_c = numerator_control_estimator.point_estimate()
        mu_den_c = denominator_control_estimator.point_estimate()
        xc = numerator_control_estimator.outcomes
        yc = denominator_control_estimator.outcomes
        cov_num_den_c = np.sum(np.square(wc) * (xc - mu_num_c) * (yc - mu_den_c)) / (
            np.sum(wc) ** 2.0
        )

        wt = numerator_treated_estimator.weights
        mu_num_t = numerator_treated_estimator.point_estimate()
        mu_den_t = denominator_treated_estimator.point_estimate()
        xt = numerator_treated_estimator.outcomes
        yt = denominator_treated_estimator.outcomes
        cov_num_den_t = np.sum(np.square(wt) * (xt - mu_num_t) * (yt - mu_den_t)) / (
            np.sum(wt) ** 2.0
        )

        cov_num_den = cov_num_den_c + cov_num_den_t

        return (
            ((1.0 / mu_den) ** 2.0) * var_num
            + ((mu_num / (mu_den**2.0)) ** 2.0) * var_den
            - (2.0 * mu_num / (mu_den**3.0)) * cov_num_den
        )

    def sensitivity_analysis(
        self,
        gamma: float = 6.0,
        outcome_proxies: list[float] | npt.NDArray[np.float64] | None = None,
        proxy_mean: float | None = None,
        alpha: float = 0.10,
    ) -> tuple[float, float]:
        """Raise NotImplementedError (sensitivity analysis is unsupported).

        Raises
        ------
         NotImplementedError
            Always raised; the ratio of differences of linear-fractional problems is not
            quasilinear.

        """
        raise NotImplementedError(
            "Sensitivity analysis is not implemented for TreatmentEffectRatioEstimator."
        )

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
    ) -> Generator[TreatmentEffectRatioEstimator, None, None]:
        """Raise NotImplementedError (resampling is unsupported).

        Raises
        ------
         NotImplementedError
            Always raised; resampling is not implemented for
            ``TreatmentEffectRatioEstimator``.

        """
        raise NotImplementedError(
            "resample is not implemented for TreatmentEffectRatioEstimator."
        )

    def expanded_confidence_interval(
        self,
        alpha: float = 0.10,
        gamma: float = 6.0,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        bootstrap: bool = True,
        B: int = 1_000,
        seed: (
            int
            | list[int]
            | np.random.SeedSequence
            | np.random.BitGenerator
            | np.random.Generator
            | None
        ) = None,
    ) -> tuple[float, float]:
        """Raise NotImplementedError (expanded confidence intervals unsupported).

        Raises
        ------
         NotImplementedError
            Always raised; expanded confidence intervals are not implemented for
            ``TreatmentEffectRatioEstimator``.

        """
        raise NotImplementedError(
            "Expanded confidence interval is not implemented for TreatmentEffectRatioEstimator."
        )
