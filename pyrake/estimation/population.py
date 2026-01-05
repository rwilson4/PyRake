"""Population estimators."""

import math
from collections.abc import Generator
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt
from scipy import optimize

from .base_classes import Estimand, SimpleEstimand, WeightingEstimator


class PopulationMean(SimpleEstimand):
    """Population mean."""

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Calculate weights based on propensity scores."""
        return np.ones_like(self.propensity_scores) / self.propensity_scores

    def sensitivity_region(
        self,
        gamma: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate a sensitivity region around baseline weights."""
        weights = self.weights
        wl = weights / math.sqrt(gamma) + (1.0 - 1.0 / math.sqrt(gamma))
        wu = weights * math.sqrt(gamma) - (math.sqrt(gamma) - 1.0)
        return wl, wu

    @staticmethod
    def normalizing_factor(sample_size: int, population_size: int) -> float:
        """Calculate normalizing factor."""
        return population_size


class NonRespondentMean(SimpleEstimand):
    """Non-respondent mean.

    The non-respondent mean is the mean response among everyone who did *not* respond.
    The weights for this estimand are (1 - pi) / pi, where pi are the propensity scores.

    """

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Calculate weights based on propensity scores."""
        return (
            np.ones_like(self.propensity_scores) - self.propensity_scores
        ) / self.propensity_scores

    def sensitivity_region(
        self,
        gamma: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate a sensitivity region around baseline weights."""
        weights = self.weights
        wl = weights / math.sqrt(gamma)
        wu = weights * math.sqrt(gamma)
        return wl, wu

    @staticmethod
    def normalizing_factor(sample_size: int, population_size: int) -> float:
        """Calculate normalizing factor."""
        return population_size - sample_size


class SampleMean(SimpleEstimand):
    """Sample mean.

    The sample mean is the mean response among the population of respondents. The
    weights are simply 1, and thus involve no uncertainty.

    """

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Calculate weights based on propensity scores."""
        return np.ones_like(self.propensity_scores)

    def sensitivity_region(
        self,
        gamma: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate a sensitivity region around baseline weights."""
        weights = self.weights
        return weights, weights

    @staticmethod
    def normalizing_factor(sample_size: int, population_size: int) -> float:
        """Calculate normalizing factor."""
        return sample_size


class MeanEstimator(WeightingEstimator):
    """Base class for estimating a mean in some population.

    Parameters
    ----------
     propensity_scores : list_like
        Propensity scores.
     outcomes : list_like
        Outcomes or responses. Should have the same length as `propensity_scores`.
     estimand : Estimand, optional
        An Estimand object representing the thing to be estimated. Defaults to
        PopulationMean.

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

    def pvalue(
        self,
        null_value: float,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ) -> float:
        r"""Calculate a p-value.

        Computes a p-value against the null hypothesis:
            H0: \bar{Y} = null_value,
        where \bar{Y} is the mean in the target population.

        Parameters
        ----------
         null_value : float
            The hypothesized mean.
         alternative : ["two-sided", "less", "greater"], optional
            What kind of test:
              - "two-sided": H0: \bar{Y} = `null_value` vs Halt: \bar{Y} <>
                `null_value`.
              - "greater": H0: \bar{Y} <= `null_value` vs Halt: \bar{Y} > `null_value`.
              - "less": H0: \bar{Y} >= `null_value` vs Halt: \bar{Y} < `null_value`.
            For example, specifying alternative = "greater" returns a p-value that quantifies
            the strength of evidence against the null hypothesis that the mean is less
            than `null_value`, in favor of the alternative hypothesis that the mean is
            actually greater than `null_value`. Defaults to "two-sided".

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
        r"""Calculate confidence interval on mean in target population.

        Parameters
        ----------
         alpha : float, optional
            P-value threshold, e.g. specify alpha=0.05 for a 95% confidence interval.
            Defaults to 0.10, corresponding to a 90% confidence interval.
         alternative : ["two-sided", "less", "greater"], optional
            What kind of test:
              - "two-sided": H0: \bar{Y} = `null_value` vs Halt: \bar{Y} <>
                `null_value`.
              - "greater": H0: \bar{Y} <= `null_value` vs Halt: \bar{Y} > `null_value`.
              - "less": H0: \bar{Y} >= `null_value` vs Halt: \bar{Y} < `null_value`.
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
    ) -> Generator[Self, None, None]:
        """Yield a sequence of resampled estimators.

        Parameters
        ----------
         B : int
            Number of bootstrap replications to run.
         seed : int, list_like, etc
            A seed for numpy.random.default_rng. See that documentation for details.

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
              - "two-sided": H0: \bar{Y} = `null_value` vs Halt: \bar{Y} <>
                `null_value`.
              - "greater": H0: \bar{Y} <= `null_value` vs Halt: \bar{Y} > `null_value`.
              - "less": H0: \bar{Y} >= `null_value` vs Halt: \bar{Y} < `null_value`.
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

        """
        if gamma < 1.0:
            raise ValueError("`gamma` must be >= 1")

        if gamma == 1.0:
            return self.confidence_interval(alpha=alpha, alternative=alternative)

        lb_bootstrap = np.zeros(B)
        ub_bootstrap = np.zeros(B)
        for b, est in enumerate(self.resample(B, seed)):
            (
                lb_bootstrap[b],
                ub_bootstrap[b],
            ) = est.sensitivity_analysis(gamma=gamma)

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


class IPWEstimator(MeanEstimator):
    r"""Inverse Propensity Weighted Estimator.

    Parameters
    ----------
     propensity_scores : list_like
        Propensity scores.
     outcomes : list_like
        Outcomes or responses. Should have the same length as `propensity_scores`.
     population_size : int
        Size of target population.
     estimand : Estimand, optional
        An Estimand object representing the thing to be estimated. Defaults to
        PopulationMean.

    Notes
    -----
    The IPW estimator is:
        \hat{Y}_IPW = (1/N) \sum_i w_i * y_i,
    where w_i = 1 / pi_i, N is the target population size, and pi_i is the propensity
    score.

    Details on this estimator, including sensitivity analysis and links to academic
    resources, can be found in (Wilson, 2025)

    References
    ----------
     - Wilson, Bob. 2025. "Design-Based Inference and Sensitivity Analysis for Survey
       Sampling." https://adventuresinwhy.com/post/survey_sampling/

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
        """Calculate a point estimate."""
        nf = self.estimand.normalizing_factor(
            sample_size=len(self.weights), population_size=self.population_size
        )
        return np.dot(self.weights, self.outcomes) / nf

    def variance(self) -> float:
        r"""Calculate the variance.

        Per (Wilson 2025, §2), an estimator for the variance is:
                   \sum_i (((n/N) / pi_i) * y_i - pe)^2
                  --------------------------------------,
                                n * (n - 1)
        where pe is the point estimate and n is the sample size.

        References
        ----------
         - Wilson, Bob. 2025. "Design-Based Inference and Sensitivity Analysis for Survey
           Sampling." https://adventuresinwhy.com/post/survey_sampling/

        """
        n = len(self.weights)
        n_over_N = n / self.population_size
        return np.mean(
            np.square(n_over_N * self.weights * self.outcomes - self.point_estimate())
        ) / (n - 1)

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
        See the Notes in WeightingEstimator for overall context. In our case, there is a
        simple closed-form calculation for the sensitivity interval. See (Wilson 2025)
        for details.

        References
        ----------
         - Wilson, Bob. 2025. "Design-Based Inference and Sensitivity Analysis for Survey
           Sampling." https://adventuresinwhy.com/post/survey_sampling/

        """
        if gamma < 1.0:
            raise ValueError("`gamma` must be >= 1")

        if gamma == 1.0:
            return self.point_estimate(), self.point_estimate()

        wl, wu = self.estimand.sensitivity_region(gamma)
        lb = np.where(self.outcomes >= 0, wl, wu)
        ub = np.where(self.outcomes >= 0, wu, wl)
        return (
            np.sum(self.outcomes * lb) / self.population_size,
            np.sum(self.outcomes * ub) / self.population_size,
        )


class AIPWEstimator(IPWEstimator):
    r"""Augmented Inverse Propensity Weighted Estimator.

    Parameters
    ----------
     propensity_scores : list_like
        Propensity scores.
     outcomes : list_like
        Outcomes or responses. Should have the same length as `propensity_scores`.
     predicted_outcomes : list_like
        Predicted outcomes corresponding to each response. Should have the same length
        as `propensity_scores`.
     mean_predicted_outcome : float
        Mean of predicted outcomes in target population.
     population_size : int
        Size of target population.

    Notes
    -----
    The AIPW estimator for a mean is:
        \hat{Y}_AIPW = \bar{\mu} + (1/N) \sum_i w_i * (y_i - \mu(x_i)),
    where w_i = 1 / pi_i, N is the target population size, and pi_i is the propensity
    score (as in IPWEstimator). \mu(x_i) is a *prediction* of the outcome, y_i, for
    individual i, made on the basis of covariates, x, observed for everyone in the
    target population (including everyone in the sample), and \bar{\mu} is the average
    predicted outcome in the target population. (Since we observe x for everyone in the
    target population, we can predict the outcome for everyone in the population, and
    \bar{\mu} is the average of these predictions.)

    The intuition is, if we have a good model for the outcome, we can use it to predict
    the target population mean directly, then adjust this prediction with an estimator
    based on the residuals in the sample. Details on this estimator, including
    sensitivity analysis and links to academic resources, can be found in (Wilson, 2025)

    References
    ----------
     - Wilson, Bob. 2025. "Design-Based Inference and Sensitivity Analysis for Survey
       Sampling." https://adventuresinwhy.com/post/survey_sampling/

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
        """Calculate a point estimate."""
        return self.mean_predicted_outcome + super().point_estimate()

    def variance(self) -> float:
        r"""Calculate the variance.

        Per (Wilson 2025, §2), an estimator for the variance is:
                   \sum_i (w_i * y_i - pe)^2
                  ---------------------------,
                           n * (n - 1)
        where pe is the point estimate and n is the sample size.

        References
        ----------
         - Wilson, Bob. 2025. "Design-Based Inference and Sensitivity Analysis for Survey
           Sampling." https://adventuresinwhy.com/post/survey_sampling/

        """
        n_over_N = len(self.weights) / self.population_size
        return np.mean(
            np.square(
                n_over_N * self.weights * self.outcomes - super().point_estimate()
            )
        ) / (len(self.weights) - 1)

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
        See the Notes in WeightingEstimator for overall context. In our case, there is a
        simple closed-form calculation for the sensitivity interval. See (Wilson 2025)
        for details.

        References
        ----------
         - Wilson, Bob. 2025. "Design-Based Inference and Sensitivity Analysis for Survey
           Sampling." https://adventuresinwhy.com/post/survey_sampling/

        """
        if gamma < 1.0:
            raise ValueError("`gamma` must be >= 1")

        if gamma == 1.0:
            return self.point_estimate(), self.point_estimate()

        lb, ub = super().sensitivity_analysis(gamma=gamma)
        return (
            self.mean_predicted_outcome + lb,
            self.mean_predicted_outcome + ub,
        )


class SIPWEstimator(MeanEstimator):
    r"""Stabilized Inverse Propensity Weighted Estimator.

    Parameters
    ----------
     propensity_scores : list_like
        Propensity scores.
     outcomes : list_like
        Outcomes or responses. Should have the same length as `propensity_scores`.
     binary_outcomes : bool, optional
        Indicates whether the outcomes are binary. If `outcomes` contains values other
        than 0 and 1, we conclude the outcomes are *not* binary; if all values are 0 and
        1 we conclude outcomes *are* binary. The only place this matters is in the
        sensitivity analysis, which is faster for binary outcomes than in general.

    Notes
    -----
    The SIPW estimator is:
                       \sum_i w_i * y_i
       \hat{Y}_SIPW = -------------------
                         \sum_i w_i
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
        """Calculate a point estimate."""
        return np.dot(self.weights, self.outcomes) / np.sum(self.weights)

    def variance(self) -> float:
        r"""Calculate the variance.

        Per (Sarig, Galili, Eilat 2023, Appendix D.2), an estimator for the variance is:
                   \sum_i w_i^2 * (y_i - pe)^2
                  -----------------------------,
                        (\sum_i w_i)^2
        where pe is the point estimate.

        """
        return np.sum(
            np.square(self.weights) * np.square(self.outcomes - self.point_estimate())
        ) / (np.sum(self.weights) ** 2)

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
        See the Notes in WeightingEstimator for overall context.

        (Zhao, Small, and Bhattarcharya 2019) calculates
                 \sum_i w_i * y_i
           inf -------------------
                   \sum_i w_i
        over wli <= w_i <= wui as the lower bound of the sensitivity interval, and the
        supremum as the upper bound. These are linear fractional programs subject to
        linear inequality constraints.

        They propose a fast algorithm for calculating the infimum and supremum that
        requires iterating over the data just once. When the outcome is binary, we can
        do even better: the infimum is achieved by setting weights corresponding to
        negative outcomes to their maximum value (wui) and weights corresponding to
        positive outcomes to their minimum value (wli). Reverse this to calculate the
        supremum.

        References
        ----------
         - Zhao, Qingyuan, Dylan S Small, and Bhaswar B Bhattacharya. 2019. "Sensitivity
           Analysis for Inverse Probability Weighting Estimators via the Percentile
           Bootstrap." Journal of the Royal Statistical Society Series B: Statistical
           Methodology 81 (4). Oxford University Press: pg. 735--61.

        """
        if gamma < 1.0:
            raise ValueError("`gamma` must be >= 1")

        if gamma == 1.0:
            return self.point_estimate(), self.point_estimate()

        wl, wu = self.estimand.sensitivity_region(gamma)
        if self.binary_outcomes:
            lb = np.where(self.outcomes == 1, wl, wu)
            ub = np.where(self.outcomes == 1, wu, wl)
            return (
                np.dot(self.outcomes, lb) / np.sum(lb),
                np.dot(self.outcomes, ub) / np.sum(ub),
            )

        idx = np.argsort(self.outcomes)
        w = np.array(wl)
        zeta = np.sum(w)
        lmbda = np.dot(self.outcomes, w)
        lb_star = lmbda / zeta
        for ii in range(len(self.outcomes)):
            w[idx[ii]] = wu[idx[ii]]
            zeta += wu[idx[ii]] - wl[idx[ii]]
            lmbda += self.outcomes[idx[ii]] * (wu[idx[ii]] - wl[idx[ii]])
            if lmbda < lb_star * zeta:
                lb_star = lmbda / zeta
            else:
                break

        idx = idx[::-1]
        w = np.array(wl)
        zeta = np.sum(w)
        lmbda = np.dot(self.outcomes, w)
        ub_star = lmbda / zeta
        for ii in range(len(self.outcomes)):
            w[idx[ii]] = wl[idx[ii]]
            zeta += wu[idx[ii]] - wl[idx[ii]]
            lmbda += self.outcomes[idx[ii]] * (wu[idx[ii]] - wl[idx[ii]])
            if lmbda > ub_star * zeta:
                ub_star = lmbda / zeta
            else:
                break

        return lb_star, ub_star


class SAIPWEstimator(SIPWEstimator):
    """Stabilized Augmented Inverse Propensity Weighted Estimator.

    Parameters
    ----------
     propensity_scores : list_like
        Propensity scores.
     outcomes : list_like
        Outcomes or responses. Should have the same length as `propensity_scores`.
     predicted_outcomes : list_like
        Predicted outcomes corresponding to each response. Should have the same length
        as `propensity_scores`.
     mean_predicted_outcome : float
        Mean of predicted outcomes in target population.
     binary_outcomes : bool, optional
        Indicates whether the outcomes are binary. If `outcomes` contains values other
        than 0 and 1, we conclude the outcomes are *not* binary; if all values are 0 and
        1 we conclude outcomes *are* binary. The only place this matters is in the
        sensitivity analysis, which is faster for binary outcomes than in general.

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
        """Calculate a point estimate."""
        return self.mean_predicted_outcome + super().point_estimate()

    def variance(self) -> float:
        r"""Calculate the variance.

        Per (Sarig, Galili, Eilat 2023, Appendix D.2), an estimator for the variance is:
                   \sum_i w_i^2 * (y_i - pe)^2
                  -----------------------------,
                        (\sum_i w_i)^2
        where pe is the point estimate.

        """
        return np.sum(
            np.square(self.weights)
            * np.square(self.outcomes - super().point_estimate())
        ) / (np.sum(self.weights) ** 2)

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
        See the Notes in WeightingEstimator and SIPWEstimator for overall context.

        """
        if gamma < 1.0:
            raise ValueError("`gamma` must be >= 1")

        if gamma == 1.0:
            return self.point_estimate(), self.point_estimate()

        lb, ub = super().sensitivity_analysis(gamma=gamma)
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
        the same length as `propensity_scores`.
     estimand : Estimand, optional
        An Estimand object representing the thing to be estimated. Defaults to
        PopulationMean.
     estimator_class : ["IPW", "AIPW", "SIPW", "SAIPW"], optional
        What kind of estimator to use. Defaults to "SIPW". Currently only "SIPW" is supported.
     numerator_estimator_class, denominator_estimator_class: ["IPW", "AIPW", "SIPW", "SAIPW"], optional
        If you really want to use different estimator classes for numerator vs
        denominator, you can. Defaults to `estimator_class`. Currently only "SIPW" is
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
        """Wrapper for weights."""
        return self.numerator_estimator.weights

    @property
    def propensity_scores(self) -> npt.NDArray[np.float64]:
        """Wrapper for propensity scores."""
        return self.numerator_estimator.propensity_scores

    def point_estimate(self) -> float:
        """Calculate a point estimate."""
        return (
            self.numerator_estimator.point_estimate()
            / self.denominator_estimator.point_estimate()
        )

    def variance(self) -> float:
        """Calculate the variance.

        Uses the delta method to calculate the variance.

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
        See the Notes in WeightingEstimator for overall context.

        We calculate
                 \sum_i w_i * x_i
           inf -------------------
                 \sum_i w_i * y_i
        over wli <= w_i <= wui as the lower bound of the sensitivity interval, and the
        supremum as the upper bound. These are linear fractional programs subject to
        linear inequality constraints. They may be transformed to equivalent linear
        programs and solved efficiently, as described in Boyd and Vandenberghe (2004).

        References
        ----------
         - Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge
           University Press, 2004.

        """
        weights = self.weights
        wl = weights / math.sqrt(gamma) + (1.0 - 1.0 / math.sqrt(gamma))
        wu = weights * math.sqrt(gamma) - (math.sqrt(gamma) - 1.0)
        G = np.vstack([-np.eye(len(weights)), np.eye(len(weights))])
        h = np.concatenate([-wl, wu])

        lb_res = optimize.linprog(
            c=np.append(self.numerator_estimator.outcomes, 0),
            A_ub=np.hstack([G, -h.reshape(-1, 1)]),
            b_ub=np.zeros_like(h),
            A_eq=np.append(self.denominator_estimator.outcomes, 0).reshape(1, -1),
            b_eq=np.array([1]),
        )

        ub_res = optimize.linprog(
            c=np.append(-self.numerator_estimator.outcomes, 0),
            A_ub=np.hstack([G, -h.reshape(-1, 1)]),
            b_ub=np.zeros_like(h),
            A_eq=np.append(self.denominator_estimator.outcomes, 0).reshape(1, -1),
            b_eq=np.array([1]),
        )

        return lb_res.fun, -ub_res.fun

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
    ) -> Generator[Self, None, None]:
        """Yield a sequence of resampled estimators.

        Parameters
        ----------
         B : int
            Number of bootstrap replications to run.
         seed : int, list_like, etc
            A seed for numpy.random.default_rng. See that documentation for details.

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
              - "two-sided": H0: \bar{Y} = `null_value` vs Halt: \bar{Y} <>
                `null_value`.
              - "greater": H0: \bar{Y} <= `null_value` vs Halt: \bar{Y} > `null_value`.
              - "less": H0: \bar{Y} >= `null_value` vs Halt: \bar{Y} < `null_value`.
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

        """
        if gamma < 1.0:
            raise ValueError("`gamma` must be >= 1")

        if gamma == 1.0:
            return self.confidence_interval(alpha=alpha, alternative=alternative)

        lb_bootstrap = np.zeros(B)
        ub_bootstrap = np.zeros(B)
        for b, est in enumerate(self.resample(B, seed)):
            (
                lb_bootstrap[b],
                ub_bootstrap[b],
            ) = est.sensitivity_analysis(gamma=gamma)

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
