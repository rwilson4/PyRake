"""Treatment effect estimators."""

from collections.abc import Generator
from typing import Any, Literal, Type

import numpy as np
import numpy.typing as npt

from .base_classes import Estimand, SimpleEstimand, WeightingEstimator
from .population import (
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
    """Base class when there are two sampling mechanisms."""

    def __init__(self, estimand1: SimpleEstimand, estimand2: SimpleEstimand) -> None:
        self.estimand1 = estimand1
        self.estimand2 = estimand2

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Calculate weights based on propensity scores."""
        return self.estimand1.weights * self.estimand2.weights

    def sensitivity_region(
        self,
        gamma: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate a sensitivity region around baseline weights."""
        wl1, wu1 = self.estimand1.sensitivity_region(gamma)
        wl2, wu2 = self.estimand2.sensitivity_region(gamma)
        return wl1 * wl2, wu1 * wu2

    def resample(
        self, bootstrap_indices: npt.NDArray[np.int64]
    ) -> "DoubleSamplingEstimand":
        """Create a new Estimand object based on resampled indices."""
        return DoubleSamplingEstimand(
            self.estimand1.resample(bootstrap_indices),
            self.estimand2.resample(bootstrap_indices),
        )

    @staticmethod
    def normalizing_factor(sample_size: int, population_size: int) -> float:
        """Calculate normalizing factor."""
        raise NotImplementedError("Not implemented")


class TreatmentEffectEstimator(WeightingEstimator):
    """Base class for estimating treatment effects."""

    def __init__(
        self,
        control_estimator: MeanEstimator | RatioEstimator,
        treated_estimator: MeanEstimator | RatioEstimator,
    ) -> None:
        self.control_estimator = control_estimator
        self.treated_estimator = treated_estimator

    def point_estimate(self) -> float:
        """Calculate a point estimate."""
        return (
            self.treated_estimator.point_estimate()
            - self.control_estimator.point_estimate()
        )

    def variance(self) -> float:
        """Calculate the variance."""
        return self.treated_estimator.variance() + self.control_estimator.variance()

    def pvalue(
        self,
        null_value: float = 0.0,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ) -> float:
        r"""Calculate a p-value.

        Computes a p-value against the null hypothesis:
            H0: TE = null_value,
        where TE is the treatment effect.

        Parameters
        ----------
         null_value : float
            The hypothesized treatment effect.
         alternative : ["two-sided", "less", "greater"], optional
            What kind of test:
              - "two-sided": H0: TE = `null_value` vs Halt: TE <> `null_value`.
              - "greater": H0: TE <= `null_value` vs Halt: TE > `null_value`.
              - "less": H0: TE >= `null_value` vs Halt: TE < `null_value`.
            For example, specifying alternative = "greater" returns a p-value that quantifies
            the strength of evidence against the null hypothesis that the treatment
            effect is less than `null_value`, in favor of the alternative hypothesis
            that the treatment effect is actually greater than `null_value`. Defaults to
            "two-sided".

        Returns
        -------
         p : float
            P-value.

        """
        return super().pvalue(null_value=null_value, alternative=alternative)

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
              - "two-sided": H0: TE = `null_value` vs Halt: TE <> `null_value`.
              - "greater": H0: TE <= `null_value` vs Halt: TE > `null_value`.
              - "less": H0: TE >= `null_value` vs Halt: TE < `null_value`.
            Defaults to "two-sided".

        Returns
        -------
         lb, ub : float
            Lower and upper bounds on the confidence interval. For one-sided intervals,
            only one of these will be finite:
              - "two-sided": lb and ub both finite
              - "greater": lb = -np.inf, ub = finite
              - "less": lb finite, ub = np.inf

        """
        return super().confidence_interval(alpha=alpha, alternative=alternative)

    def sensitivity_analysis(self, gamma: float = 6.0) -> tuple[float, float]:
        r"""Perform a sensitivity analysis.

        Calculate a range of point estimates implied by the Gamma factor.

        Parameters
        ----------
         gamma : float, optional
            The Gamma factor. Must be >= 1.0, with 1.0 indicating perfect propensity
            scores. Defaults to 6. See Notes.

        Returns
        -------
         lb, ub : float
            Lower and upper bounds on the point estimate.

        Notes
        -----
        The underlying sensitivity analysis is separable in the control vs treated
        individuals, so we perform two separate sensitivity analyses and combine the
        results.

        """
        control_lb, control_ub = self.control_estimator.sensitivity_analysis(
            gamma=gamma
        )
        treated_lb, treated_ub = self.treated_estimator.sensitivity_analysis(
            gamma=gamma
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
    ) -> Generator[tuple[WeightingEstimator, WeightingEstimator], None, None]:
        """Yield a sequence of resampled estimators.

        Parameters
        ----------
         B : int
            Number of bootstrap replications to run.
         seed : int, list_like, etc
            A seed for numpy.random.default_rng. See that documentation for details.

        """
        for control_estimator, treatment_estimator in zip(
            self.control_estimator.resample(B, seed=seed),
            self.treated_estimator.resample(B, seed=seed),
        ):
            yield control_estimator, treatment_estimator

    def expanded_confidence_interval(
        self,
        alpha: float = 0.10,
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
    ) -> tuple[float, float]:
        r"""Calculate an expanded confidence interval.

        The expanded confidence interval combines uncertainty from stochastic sampling
        and from propensity scores estimated with error.

        Parameters
        ----------
         alpha : float, optional
            P-value threshold, e.g. specify alpha=0.05 for a 95% confidence interval.
            Defaults to 0.10, corresponding to a 90% confidence interval.
         gamma : float, optional
            The Gamma factor. Must be >= 1.0, with 1.0 indicating perfect propensity
            scores. Defaults to 6. See Notes in `sensitivity_analysis`.
         alternative : ["two-sided", "less", "greater"], optional
            What kind of test:
              - "two-sided": H0: TE = `null_value` vs Halt: TE <> `null_value`.
              - "greater": H0: TE <= `null_value` vs Halt: TE > `null_value`.
              - "less": H0: TE >= `null_value` vs Halt: TE < `null_value`.
            Defaults to "two-sided".
         B : int, optional
            Number of bootstrap replications to run. Defaults to 1_000.
         seed : int, list_like, etc
            A seed for numpy.random.default_rng. See that documentation for details.

        Returns
        -------
         lb, ub : float
            Lower and upper bounds on the confidence interval. For one-sided intervals,
            only one of these will be finite:
              - "two-sided": lb and ub both finite
              - "greater": lb finite, ub = np.inf
              - "less": lb = -np.inf, ub finite

        Notes
        -----
        Unlike `sensitivity_analysis`, we cannot recycle the expanded confidence
        intervals of the original estimators. The appropriate bootstrap statistics
        (which we then take quantiles of to calculate the expanded confidence interval)
        are treated_lb - control_ub for the lower bound of the expanded confidence
        interval, and treated_ub - control_lb for the upper bound of the expanded
        confidence interval, where, e.g. treated_lb is the lower bound of the
        sensitivity interval for the treated estimator. When calculating the expanded
        confidence interval, the control and treated estimators are intertwined, unlike
        in the sensitivity analysis.

        """
        if gamma < 1.0:
            raise ValueError("`gamma` must be >= 1")

        if gamma == 1.0:
            return self.confidence_interval(alpha=alpha, alternative=alternative)

        lb_bootstrap = np.zeros(B)
        ub_bootstrap = np.zeros(B)
        for b, (control_estimator, treated_estimator) in enumerate(
            self.resample(B, seed)
        ):
            # Calculate sensitivity interval for this bootstrap sample
            control_lb, control_ub = control_estimator.sensitivity_analysis(gamma=gamma)
            treated_lb, treated_ub = treated_estimator.sensitivity_analysis(gamma=gamma)
            lb_bootstrap[b] = treated_lb - control_ub
            ub_bootstrap[b] = treated_ub - control_lb

        # Calculate the expanded confidence interval as percentiles of the bootstrap estimates
        if alternative == "two-sided":
            lb = float(np.percentile(lb_bootstrap, 100 * alpha / 2))
            ub = float(np.percentile(ub_bootstrap, 100 * (1 - (alpha / 2))))
        elif alternative == "less":
            lb = -np.inf
            ub = float(np.percentile(ub_bootstrap, 100 * (1 - alpha)))
        elif alternative == "greater":
            lb = float(np.percentile(lb_bootstrap, 100 * alpha))
            ub = np.inf
        else:
            raise ValueError(f"Unrecognized input {alternative=:}")

        return lb, ub


class SimpleDifferenceEstimator(TreatmentEffectEstimator):
    """Simple difference in means.

    Parameters
    ----------
     control_outcomes, treated_outcomes : list_like
        Outcomes or responses for control and treatment groups, resp.
     control_sampling_propensity_scores, control_sampling_propensity_scores : list_like, optional
        Propensity scores for the sampling mechanism for control and treatment groups,
        resp. See Notes.
     sampling_estimand_class : ["PopulationMean", "NonRespondentMean", "SampleMean"], optional
        Population for which to estimate the average treatment effect. Used only when
        sampling propensity scores are specified. Defaults to "PopulationMean" in that
        case. See Notes.
     control_denominator_outcomes, treated_denominator_outcomes : list_like, optional
        If specified, the metric of interest is considered a ratio (with
        `control_outcomes` and `treated_outcomes` are treated as the numerators), and
        interest is in the effect of treatment on that ratio. See Notes in ATEEstimator
        for details.

    Notes
    -----
    When called without sampling propensity scores, this class computes the average
    response in treatment, minus the average response in control. No weights, just
    straight averages. This may be helpful to see how different the adjusted estimator
    is from the simplest comparison of treatment and control, to see how biased this
    simple estimator is.

    In situations where the treatment is randomly assigned in the population (but
    potentially not in the sample), this estimator can also be used generalize estimated
    treatment effects from the sample to the population (Tipton and Hartman, 2023). The
    basic strategy here is to use the control responses to estimate the average
    potential outcome Y(0) in the population, and the treated responses to estimate the
    average Y(1). Since treatment is randomly assigned in the population, the population
    average treatment effect is just the difference in these means.

    References
    ----------
    - Tipton, Elizabeth and Erin Hartman. 2023. "Generalizability and Transportability."
      In Handbook of Matching and Weighting Adjustments for Causal Inference, , pg.
      39–59. Chapman and Hall/CRC.

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
            control_denominator_outcomes = np.asarray(control_denominator_outcomes)
            treated_denominator_outcomes = np.asarray(treated_denominator_outcomes)
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
    r"""Estimator for the average treatment effect.

    Parameters
    ----------
     control_propensity_scores, treated_propensity_scores : list_like
        Propensity scores for control and treatment groups, resp.
     control_outcomes, treated_outcomes : list_like
        Outcomes or responses for control and treatment groups, resp. Should have the
        same lengths as the corresponding `propensity_scores`.
     estimator_class : ["IPW", "AIPW", "SIPW", "SAIPW"], optional
        What kind of estimator to use. Defaults to "SIPW".
     control_estimator_class, treated_estimator_class: ["IPW", "AIPW", "SIPW", "SAIPW"], optional
        If you really want to use different estimator classes for treatment vs control,
        you can. Defaults to `estimator_class`.
     control_sampling_propensity_scores, treated_sampling_propensity_scores : list_like, optional
        Propensity scores for the sampling mechanism for control and treatment groups,
        resp. See Notes.
     sampling_estimand_class : ["PopulationMean", "NonRespondentMean", "SampleMean"], optional
        Population for which to estimate the average treatment effect. Used only when
        sampling propensity scores are specified. Defaults to "PopulationMean" in that
        case. See Notes.
     control_denominator_outcomes, treated_denominator_outcomes : list_like, optional
        If specified, the metric of interest is considered a ratio (with
        `control_outcomes` and `treated_outcomes` are treated as the numerators), and
        interest is in the effect of treatment on that ratio. See Notes.
     control_kwargs, treated_kwargs, kwargs : dict_like
        Extra kwargs to pass to the estimators. `kwargs` are passed to both estimators,
        while `control_kwargs` is only passed to the control estimator and
        `treated_kwargs` is only passed to the treated estimator.

    Notes
    -----
    Weighting estimators for the average treatment effect tend to look like:
                    \sum_{i \in T} w_i * y_i      \sum_{i \in C} w_i * y_i
       \hat{ATE} = -------------------------  -  -------------------------,
                      \sum_{i \in T} w_i            \sum_{i \in C} w_i
    where w_i = 1 / pi_i for the treated group, and 1 / (1 - pi_i) for the control
    group, and pi_i is the propensity score. This is really just the difference between
    two MeanEstimators, which makes calculations (including sensitivity analysis)
    straightforward.

    When the units in the study are sampled from a population, and the sample is not a
    simple random sample, we may wish to estimate treatment effects in the population,
    which involves elements of both observational causal inference and survey
    non-response. If sampling propensity scores are available, in additional to
    treatment assignment propensity scores, we can adjust for both non-response bias and
    treatment selection bias.

    Sometimes we are interested in a treatment effect in a subpopulation that is
    identifiable only in the sample. For example, people may self-identify as belonging
    to that population in the sample. We can estimate the population mean among the
    sub-population as a ratio of two MeanEstimators as follows. Let t_i be 1 if person i
    belongs to the sub-population, and 0 otherwise. We observe t_i only for respondents.
    Let y_i be the metric of interest. Then the average value of the metric among the
    sub-population is:
        \sum_{i=1}^N t_i * y_i     (1/N) \sum_{i=1}^N t_i * y_i
        ----------------------  =  ----------------------------,
        \sum_{i=1}^N t_i           (1/N) \sum_{i=1}^N t_i
    which is the ratio of the population means of t_i * y_i and t_i. Inference
    (including sensitivity analysis) for such ratios is provided by RatioEstimator.

    We may wish to estimate the impact of a treatment on an outcome in a subpopulation
    identifiable only in the sample. Let (y0_i, y1_i) be potential outcomes, and assume
    subpopulation membership is not influenced by treatment, so that t_i does not have
    associated potential outcomes. The quantity of interest is
        \sum_{i=1}^N t_i * (y1_i - y0_i)     (1/N) \sum_{i=1}^N t_i * y1_i     (1/N) \sum_{i=1}^N t_i * y0_i
        --------------------------------  =  -----------------------------  -  -----------------------------.
        \sum_{i=1}^N t_i                     (1/N) \sum_{i=1}^N t_i            (1/N) \sum_{i=1}^N t_i

    By specifying t_i * y_i as the `control_outcomes` and `treated_outcomes` (the
    numerator), and t_i as the `control_denominator_outcomes` and
    `treated_denominator_outcomes`, inference (including sensitivity analysis) for this
    use case is supported.

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
            Type[PopulationMean | NonRespondentMean | SampleMean],
        ] = {
            "PopulationMean": PopulationMean,
            "NonRespondentMean": NonRespondentMean,
            "SampleMean": SampleMean,
        }
        sampling_estimand: Type[PopulationMean | NonRespondentMean | SampleMean] = (
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
            treated_estimand: Estimand = PopulationMean(treated_propensity_scores)
        else:
            treated_estimand = DoubleSamplingEstimand(
                PopulationMean(treated_propensity_scores),
                sampling_estimand(np.asarray(treated_sampling_propensity_scores)),
            )

        if (
            control_denominator_outcomes is not None
            and treated_denominator_outcomes is not None
        ):
            control_denominator_outcomes = np.asarray(control_denominator_outcomes)
            treated_denominator_outcomes = np.asarray(treated_denominator_outcomes)
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
    r"""Estimator for the average effect of treatment on the treated (ATT).

    Parameters
    ----------
     control_propensity_scores : list_like
        Propensity scores for control group.
     control_outcomes, treated_outcomes : list_like
        Outcomes or responses for control and treatment groups, resp. Should have the
        same lengths as the corresponding `propensity_scores`.
     estimator_class : ["IPW", "AIPW", "SIPW", "SAIPW"], optional
        What kind of estimator to use. Defaults to "SIPW".
     control_estimator_class, treated_estimator_class: ["IPW", "AIPW", "SIPW", "SAIPW"], optional
        If you really want to use different estimator classes for treatment vs control,
        you can. Defaults to `estimator_class`.
     control_sampling_propensity_scores, treated_sampling_propensity_scores : list_like, optional
        Propensity scores for the sampling mechanism for control and treatment groups,
        resp. See Notes.
     sampling_estimand_class : ["PopulationMean", "NonRespondentMean", "SampleMean"], optional
        Population for which to estimate the average treatment effect. Used only when
        sampling propensity scores are specified. Defaults to "PopulationMean" in that
        case. See Notes.
     control_denominator_outcomes, treated_denominator_outcomes : list_like, optional
        If specified, the metric of interest is considered a ratio (with
        `control_outcomes` and `treated_outcomes` are treated as the numerators), and
        interest is in the effect of treatment on that ratio. See Notes in ATEEstimator
        for details.
     control_kwargs, treated_kwargs, kwargs : dict_like
        Extra kwargs to pass to the estimators. `kwargs` are passed to both estimators,
        while `control_kwargs` is only passed to the control estimator and
        `treated_kwargs` is only passed to the treated estimator.

    Notes
    -----
    Weighting estimators for the average treatment effect tend to look like:
                    \sum_{i \in T} y_i      \sum_{i \in C} w_i * y_i
       \hat{ATT} = --------------------  -  -------------------------,
                           nT                  \sum_{i \in C} w_i
    where w_i = pi_i / (1 - pi_i) and pi_i is the propensity score. This is really just
    the difference between two MeanEstimators, which makes calculations (including
    sensitivity analysis) straightforward.

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
            Type[PopulationMean | NonRespondentMean | SampleMean],
        ] = {
            "PopulationMean": PopulationMean,
            "NonRespondentMean": NonRespondentMean,
            "SampleMean": SampleMean,
        }
        sampling_estimand: Type[PopulationMean | NonRespondentMean | SampleMean] = (
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
            control_denominator_outcomes = np.asarray(control_denominator_outcomes)
            treated_denominator_outcomes = np.asarray(treated_denominator_outcomes)
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
    r"""Estimator for the average effect of treatment on the control units (ATC).

    Parameters
    ----------
     treated_propensity_scores : list_like
        Propensity scores for the treatment group.
     control_outcomes, treated_outcomes : list_like
        Outcomes or responses for control and treatment groups, resp. Should have the
        same lengths as the corresponding `propensity_scores`.
     estimator_class : ["IPW", "AIPW", "SIPW", "SAIPW"], optional
        What kind of estimator to use. Defaults to "SIPW".
     control_estimator_class, treated_estimator_class: ["IPW", "AIPW", "SIPW", "SAIPW"], optional
        If you really want to use different estimator classes for treatment vs control,
        you can. Defaults to `estimator_class`.
     control_sampling_propensity_scores, treated_sampling_propensity_scores : list_like, optional
        Propensity scores for the sampling mechanism for control and treatment groups,
        resp. See Notes.
     sampling_estimand_class : ["PopulationMean", "NonRespondentMean", "SampleMean"], optional
        Population for which to estimate the average treatment effect. Used only when
        sampling propensity scores are specified. Defaults to "PopulationMean" in that
        case. See Notes.
     control_denominator_outcomes, treated_denominator_outcomes : list_like, optional
        If specified, the metric of interest is considered a ratio (with
        `control_outcomes` and `treated_outcomes` are treated as the numerators), and
        interest is in the effect of treatment on that ratio. See Notes in ATEEstimator
        for details.
     control_kwargs, treated_kwargs, kwargs : dict_like
        Extra kwargs to pass to the estimators. `kwargs` are passed to both estimators,
        while `control_kwargs` is only passed to the control estimator and
        `treated_kwargs` is only passed to the treated estimator.

    Notes
    -----
    Weighting estimators for the average treatment effect tend to look like:
                    \sum_{i \in T} w_i * y_i      \sum_{i \in C} y_i
       \hat{ATC} = -------------------------  -  --------------------,
                      \sum_{i \in T} w_i                nC
    where w_i = (1 - pi_i) / pi_i and pi_i is the propensity score. This is really just
    the difference between two MeanEstimators, which makes calculations (including
    sensitivity analysis) straightforward.

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
            Type[PopulationMean | NonRespondentMean | SampleMean],
        ] = {
            "PopulationMean": PopulationMean,
            "NonRespondentMean": NonRespondentMean,
            "SampleMean": SampleMean,
        }
        sampling_estimand: Type[PopulationMean | NonRespondentMean | SampleMean] = (
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
            treated_estimand: Estimand = NonRespondentMean(treated_propensity_scores)
        else:
            treated_estimand = DoubleSamplingEstimand(
                NonRespondentMean(treated_propensity_scores),
                sampling_estimand(np.asarray(treated_sampling_propensity_scores)),
            )

        if (
            control_denominator_outcomes is not None
            and treated_denominator_outcomes is not None
        ):
            control_denominator_outcomes = np.asarray(control_denominator_outcomes)
            treated_denominator_outcomes = np.asarray(treated_denominator_outcomes)
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
     control_sampling_propensity_scores, treated_sampling_propensity_scores : list_like, optional
        Propensity scores for the sampling mechanism for control and treatment groups,
        resp. See Notes.
     sampling_estimand : ["PopulationMean", "NonRespondentMean", "SampleMean"], optional
        Population for which to estimate the average treatment effect. Used only when
        sampling propensity scores are specified. Defaults to "PopulationMean" in that
        case. See Notes.
     numerator_control_kwargs, numerator_treated_kwargs, denominator_control_kwargs, denominator_treated_kwargs : dict, optional
        Extra kwargs to pass to the respective estimators.

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
        """Calculate a point estimate."""
        return (
            self.numerator_estimator.point_estimate()
            / self.denominator_estimator.point_estimate()
        )

    def variance(self) -> float:
        """Calculate the variance of the ratio estimator using the delta method."""
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

    def sensitivity_analysis(self, gamma: float = 6.0) -> tuple[float, float]:
        """Not implemented."""
        raise NotImplementedError(
            "Sensitivity analysis is not implemented for TreatmentEffectRatioEstimator."
        )

    def expanded_confidence_interval(
        self,
        alpha: float = 0.10,
        gamma: float = 6.0,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
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
        """Not implemented."""
        raise NotImplementedError(
            "Expanded confidence interval is not implemented for TreatmentEffectRatioEstimator."
        )
