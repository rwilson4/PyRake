"""Estimate things."""

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

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

    @staticmethod
    @abstractmethod
    def weights_from_propensity_scores(
        propensity_scores: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Calculate weights based on propensity scores."""
        pass

    @staticmethod
    @abstractmethod
    def sensitivity_region(
        weights: npt.NDArray[np.float64],
        gamma: float,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate a sensitivity region around baseline weights."""
        pass


class PopulationMean(Estimand):
    """Population mean."""

    @staticmethod
    def weights_from_propensity_scores(
        propensity_scores: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Calculate weights based on propensity scores."""
        return np.ones_like(propensity_scores) / propensity_scores

    @staticmethod
    def sensitivity_region(
        weights: npt.NDArray[np.float64],
        gamma: float,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate a sensitivity region around baseline weights."""
        wl = weights / math.sqrt(gamma) + (1.0 - 1.0 / math.sqrt(gamma))
        wu = weights * math.sqrt(gamma) - (math.sqrt(gamma) - 1.0)
        return wl, wu


class NonRespondentMean(Estimand):
    """Non-respondent mean.

    The non-respondent mean is the mean response among everyone who did *not* respond.
    The weights for this estimand are (1 - pi) / pi, where pi are the propensity scores.

    """

    @staticmethod
    def weights_from_propensity_scores(
        propensity_scores: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Calculate weights based on propensity scores."""
        return (np.ones_like(propensity_scores) - propensity_scores) / propensity_scores

    @staticmethod
    def sensitivity_region(
        weights: npt.NDArray[np.float64],
        gamma: float,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate a sensitivity region around baseline weights."""
        wl = weights / math.sqrt(gamma)
        wu = weights * math.sqrt(gamma)
        return wl, wu


class SampleMean(Estimand):
    """Sample mean.

    The sample mean is the mean response among the population of respondents. The
    weights are simply 1, and thus involve no uncertainty.

    """

    @staticmethod
    def weights_from_propensity_scores(
        propensity_scores: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Calculate weights based on propensity scores."""
        return np.ones_like(propensity_scores)

    @staticmethod
    def sensitivity_region(
        weights: npt.NDArray[np.float64],
        gamma: float,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate a sensitivity region around baseline weights."""
        return weights, weights


class WeightingEstimator(ABC):
    """Base class for weighting estimators."""

    @abstractmethod
    def point_estimate(self) -> float:
        """Calculate a point estimate."""

    @abstractmethod
    def variance(self) -> float:
        """Calculate the variance."""

    @abstractmethod
    def sensitivity_analysis(self, gamma: float = 6.0) -> Tuple[float, float]:
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
        (Zhao, Small, and Bhattarcharya 2019) posits a model:
               1             pi_i / (1 - pi_i)
            ------  < ------------------------------- < Lambda,
            Lambda     \hat{pi}_i / (1 - \hat{pi}_i)

        where pi_i is the true (typically unknown) propensity score for person i and
        \hat{pi}_i is the estimated propensity score. That is, Lambda controls how
        "wrong" the estimated propensity scores are. For sufficiently large values of
        Lambda, this model will always hold, but we have no way of knowing what value of
        Lambda is "good enough". So instead we recommend trying a variety of Lambda values
        and plot the results.

        As discussed in (Fogarty 2023), this model is connected to Rosenbaum's
        sensitivity analysis via Lambda = sqrt(Gamma). To foster connection with that
        work (and because lambda is a reserved keyword in python!), we take Gamma as
        input to this calculation, and then calculate Lambda as sqrt(Gamma).

        While no value of Gamma is guaranteed to be good enough, a value of Gamma=6 is a
        good candidate for default reporting. In his work on sensitivity analysis, Paul
        Rosenbaum often cites Hammond's meta-analysis on the connection between smoking
        and cancer, which was sensitive to biases above Gamma = 6, yet was persuasive in
        establishing that smoking causes cancer. While biases may exceed Gamma = 6, this
        value balances rigor and practicality.

        The constraint on the propensity scores translates into a constraint on the
        weights, but this depends on the estimator. The "true" weights satisfy:
            wli <= w_i <= wui.

        (Zhao, Small, and Bhattarcharya 2019) calculates the infimum and supremum of the
        estimator over this range of weights. Details vary depending on the estimator.

        References
        ----------
         - Zhao, Qingyuan, Dylan S Small, and Bhaswar B Bhattacharya. 2019. "Sensitivity
           Analysis for Inverse Probability Weighting Estimators via the Percentile
           Bootstrap." Journal of the Royal Statistical Society Series B: Statistical
           Methodology 81 (4). Oxford University Press: pg. 735–61.
         - Fogarty, Colin. 2023. “Sensitivity Analysis.” In Handbook of Matching and
           Weighting Adjustments for Causal Inference, , pg. 553–82. Chapman and
           Hall/CRC.

        """

    @abstractmethod
    def expanded_confidence_interval(
        self,
        alpha: float = 0.10,
        gamma: float = 6.0,
        side: Literal["two-sided", "lesser", "greater"] = "two-sided",
        B: int = 1_000,
        seed: Optional[
            Union[
                int,
                List[int],
                np.random.SeedSequence,
                np.random.BitGenerator,
                np.random.Generator,
            ]
        ] = None,
    ) -> Tuple[float, float]:
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
         side : ["two-sided", "lesser", "greater"], optional
            What kind of test:
              - "two-sided": H0: theta = `null_value` vs Halt: theta <> `null_value`.
              - "greater": H0: theta <= `null_value` vs Halt: theta > `null_value`.
              - "lesser": H0: theta >= `null_value` vs Halt: theta < `null_value`.
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
              - "lesser": lb = -np.inf, ub finite

        """

    def pvalue(
        self,
        null_value: float,
        side: Literal["two-sided", "lesser", "greater"] = "two-sided",
    ) -> float:
        """Calculate a p-value.

        Computes a p-value against the null hypothesis:
            H0: theta = null_value,
        where theta is the quantity being estimated

        Parameters
        ----------
         null_value : float
            The hypothesized theta.
         side : ["two-sided", "lesser", "greater"], optional
            What kind of test:
              - "two-sided": H0: theta = `null_value` vs Halt: theta <> `null_value`.
              - "greater": H0: theta <= `null_value` vs Halt: theta > `null_value`.
              - "lesser": H0: theta >= `null_value` vs Halt: theta < `null_value`.
            For example, specifying side = "greater" returns a p-value that quantifies
            the strength of evidence against the null hypothesis that theta is less than
            `null_value`, in favor of the alternative hypothesis that theta is actually
            greater than `null_value`. Defaults to "two-sided".

        Returns
        -------
         p : float
            P-value.

        """
        pe = self.point_estimate()
        se = math.sqrt(self.variance())

        t = (pe - null_value) / se

        if side == "greater" or side == "two-sided":
            p_greater = stats.norm.cdf(t)
            if side == "greater":
                return p_greater

        if side == "lesser" or side == "two-sided":
            p_lesser = stats.norm.sf(t)
            if side == "lesser":
                return p_lesser

        if side == "two-sided":
            return min(1.0, 2.0 * min(p_lesser, p_greater))

        raise ValueError(f"Unrecognized input {side=:}")

    def confidence_interval(
        self,
        alpha: float = 0.10,
        side: Literal["two-sided", "lesser", "greater"] = "two-sided",
    ) -> Tuple[float, float]:
        """Calculate confidence interval on a quantity, theta.

        Parameters
        ----------
         alpha : float, optional
            P-value threshold, e.g. specify alpha=0.05 for a 95% confidence interval.
            Defaults to 0.10, corresponding to a 90% confidence interval.
         side : ["two-sided", "lesser", "greater"], optional
            What kind of test:
              - "two-sided": H0: theta = `null_value` vs Halt: theta <> `null_value`.
              - "greater": H0: theta <= `null_value` vs Halt: theta > `null_value`.
              - "lesser": H0: theta >= `null_value` vs Halt: theta < `null_value`.
            Defaults to "two-sided".

        Returns
        -------
         lb, ub : float
            Lower and upper bounds on the confidence interval. For one-sided intervals,
            only one of these will be finite:
              - "two-sided": lb and ub both finite
              - "greater": lb = -np.inf, ub = finite
              - "lesser": lb finite, ub = np.inf

        """
        pe = self.point_estimate()
        se = math.sqrt(self.variance())
        if side == "two-sided":
            zcrit = stats.norm.isf(alpha / 2)
        else:
            zcrit = stats.norm.isf(alpha)

        if side == "lesser" or side == "two-sided":
            lb = pe - zcrit * se
            if side == "lesser":
                return lb, np.inf

        if side == "greater" or side == "two-sided":
            ub = pe + zcrit * se
            if side == "greater":
                return -np.inf, ub

        if side == "two-sided":
            return lb, ub

        raise ValueError(f"Unrecognized input {side=:}")

    def plot_sensitivity(
        self,
        gamma_lower: float = 1.0,
        gamma_upper: float = 6.0,
        num_points: int = 50,
        alpha: float = 0.10,
        side: Literal["two-sided", "lesser", "greater"] = "two-sided",
        B: int = 1_000,
        title: Optional[str] = None,
        ylabel: str = "Sensitivity Interval",
        ytick_format: str = ".0%",
        axis_label_size: Optional[int] = None,
        tick_label_size: Optional[int] = None,
        legend_label_size: Optional[int] = None,
        ax: Optional[Axes] = None,
    ) -> Tuple[pd.DataFrame, Axes]:
        r"""Plot sensitivity of estimates to hidden biases.

        Parameters
        ----------
         gamma_lower, gamma_upper : float, optional
            Range of Gamma values to test. Defaults to 1 and 6 resp.
         num_points : int, optional
            Number of Gammas to simulate and plot. Defaults to 50.
         alpha : float, optional
            P-value threshold, e.g. specify alpha=0.05 for a 95% confidence interval.
            Defaults to 0.10, corresponding to a 90% confidence interval.
         side : ["two-sided", "lesser", "greater"], optional
            What kind of test:
              - "two-sided": H0: \bar{Y} = `null_value` vs Halt: \bar{Y} <>
                `null_value`.
              - "greater": H0: \bar{Y} <= `null_value` vs Halt: \bar{Y} > `null_value`.
              - "lesser": H0: \bar{Y} >= `null_value` vs Halt: \bar{Y} < `null_value`.
         B : int, optional
            Number of bootstrap replications to run. Defaults to 1_000.
         title, ylabel : str, optional
            Title/ylabel for plot. If not specified, no title is included.
         ytick_format : str, optional
            How to format the y-axis tick labels. Defaults to ".0%", corresponding to a
            percentage.
         axis_label_size, tick_label_size, legend_label_size : int, optional
            Font size for axis labels and title; tick mark labels; and legend entries,
            resp.
         ax : Axes, optional
            If specified, plot everything on these axes.

        Returns
        -------
         df : pandas DataFrame
            A DataFrame with the lower and upper bounds corresponding to each gamma.
         ax : Axes
            The plot axes.

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
                alpha=alpha, gamma=gamma, side=side, B=B, seed=42
            )

        # Create a dataframe for plotting
        ci_lb, ci_ub = self.confidence_interval(alpha=alpha, side=side)
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
        if side == "two-sided":
            value_vars = [
                "Point Estimate",
                "Sen Lower Bound",
                "Sen Upper Bound",
                "ECI Lower Bound",
                "ECI Upper Bound",
            ]
        elif side == "lesser":
            value_vars = [
                "Point Estimate",
                "Sen Lower Bound",
                "ECI Lower Bound",
            ]
        elif side == "greater":
            value_vars = [
                "Point Estimate",
                "Sen Upper Bound",
                "ECI Upper Bound",
            ]
        else:
            raise ValueError(f"Unrecognized input {side=:}")

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

        sns.lineplot(
            x="Gamma",
            y="value",
            hue="variable",
            data=df_melt,
            palette=color_map,
            legend=False,  # Don't show legend
            ax=ax,
        )

        if side == "two-sided":
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
        elif side == "lesser":
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
        elif side == "greater":
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
            ax.xaxis.label.set_size(axis_label_size)
            ax.yaxis.label.set_size(axis_label_size)

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
        ax.legend(
            handles=legend_handles,
            loc="upper left" if side == "greater" else "lower left",
            fontsize=legend_label_size,
        )

        plt.tight_layout()
        return df, ax


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
        propensity_scores: Union[List[float], npt.NDArray[np.float64]],
        outcomes: Union[
            List[Union[float, int]], npt.NDArray[Union[np.float64, np.int64]]
        ],
        estimand: Optional[Estimand] = None,
        **kwargs,
    ) -> None:
        if len(propensity_scores) != len(outcomes):
            raise ValueError(
                "Must have equal numbers of propensity scores and outcomes."
            )

        if np.min(propensity_scores) <= 0.0 or np.max(propensity_scores) > 1.0:
            raise ValueError("Propensitiy scores must be strictly positive and <= 1")

        self.propensity_scores: npt.NDArray[np.float64] = np.asarray(propensity_scores)
        self.estimand: Estimand = estimand or PopulationMean()
        self.weights: npt.NDArray[np.float64] = (
            self.estimand.weights_from_propensity_scores(self.propensity_scores)
        )
        self.outcomes: npt.NDArray[np.float64] = np.asarray(outcomes)
        self.extra_args: Dict[str, Any] = kwargs

    def pvalue(
        self,
        null_value: float,
        side: Literal["two-sided", "lesser", "greater"] = "two-sided",
    ) -> float:
        r"""Calculate a p-value.

        Computes a p-value against the null hypothesis:
            H0: \bar{Y} = null_value,
        where \bar{Y} is the mean in the target population.

        Parameters
        ----------
         null_value : float
            The hypothesized mean.
         side : ["two-sided", "lesser", "greater"], optional
            What kind of test:
              - "two-sided": H0: \bar{Y} = `null_value` vs Halt: \bar{Y} <>
                `null_value`.
              - "greater": H0: \bar{Y} <= `null_value` vs Halt: \bar{Y} > `null_value`.
              - "lesser": H0: \bar{Y} >= `null_value` vs Halt: \bar{Y} < `null_value`.
            For example, specifying side = "greater" returns a p-value that quantifies
            the strength of evidence against the null hypothesis that the mean is less
            than `null_value`, in favor of the alternative hypothesis that the mean is
            actually greater than `null_value`. Defaults to "two-sided".

        Returns
        -------
         p : float
            P-value.

        """
        return super().pvalue(null_value=null_value, side=side)

    def confidence_interval(
        self,
        alpha: float = 0.10,
        side: Literal["two-sided", "lesser", "greater"] = "two-sided",
    ) -> Tuple[float, float]:
        r"""Calculate confidence interval on mean in target population.

        Parameters
        ----------
         alpha : float, optional
            P-value threshold, e.g. specify alpha=0.05 for a 95% confidence interval.
            Defaults to 0.10, corresponding to a 90% confidence interval.
         side : ["two-sided", "lesser", "greater"], optional
            What kind of test:
              - "two-sided": H0: \bar{Y} = `null_value` vs Halt: \bar{Y} <>
                `null_value`.
              - "greater": H0: \bar{Y} <= `null_value` vs Halt: \bar{Y} > `null_value`.
              - "lesser": H0: \bar{Y} >= `null_value` vs Halt: \bar{Y} < `null_value`.
            Defaults to "two-sided".

        Returns
        -------
         lb, ub : float
            Lower and upper bounds on the confidence interval. For one-sided intervals,
            only one of these will be finite:
              - "two-sided": lb and ub both finite
              - "greater": lb = -np.inf, ub = finite
              - "lesser": lb finite, ub = np.inf

        """
        return super().confidence_interval(alpha=alpha, side=side)

    def expanded_confidence_interval(
        self,
        alpha: float = 0.10,
        gamma: float = 6.0,
        side: Literal["two-sided", "lesser", "greater"] = "two-sided",
        B: int = 1_000,
        seed: Optional[
            Union[
                int,
                List[int],
                np.random.SeedSequence,
                np.random.BitGenerator,
                np.random.Generator,
            ]
        ] = None,
    ) -> Tuple[float, float]:
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
         side : ["two-sided", "lesser", "greater"], optional
            What kind of test:
              - "two-sided": H0: \bar{Y} = `null_value` vs Halt: \bar{Y} <>
                `null_value`.
              - "greater": H0: \bar{Y} <= `null_value` vs Halt: \bar{Y} > `null_value`.
              - "lesser": H0: \bar{Y} >= `null_value` vs Halt: \bar{Y} < `null_value`.
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
              - "lesser": lb = -np.inf, ub finite

        """
        if gamma < 1.0:
            raise ValueError("`gamma` must be >= 1")

        if gamma == 1.0:
            return self.confidence_interval(alpha=alpha, side=side)

        lb_bootstrap = np.zeros(B)
        ub_bootstrap = np.zeros(B)
        rng = np.random.default_rng(seed)
        for b in range(B):
            # Resample with replacement
            bootstrap_indices = rng.choice(
                range(len(self.propensity_scores)),
                size=len(self.propensity_scores),
                replace=True,
            )
            bootstrap_propensities = self.propensity_scores[bootstrap_indices]
            bootstrap_outcomes = self.outcomes[bootstrap_indices]
            # Implementation note for AIPW and SAIPW estimators: `self.outcomes` is
            # actually the adjusted outcomes, outcome minus prediction, and
            # `self.extra_args["predicted_outcomes"]` is actually just a vector of all
            # zeros to make the constructor happy. So there is no need to resample those
            # predicted outcomes.

            # Calculate sensitivity interval for this bootstrap sample
            (
                lb_bootstrap[b],
                ub_bootstrap[b],
            ) = self.__class__(
                propensity_scores=bootstrap_propensities,
                outcomes=bootstrap_outcomes,
                estimand=self.estimand,
                **self.extra_args,
            ).sensitivity_analysis(gamma=gamma)

        # Calculate the expanded confidence interval as percentiles of the bootstrap estimates
        if side == "two-sided":
            lb = float(np.percentile(lb_bootstrap, 100 * alpha / 2))
            ub = float(np.percentile(ub_bootstrap, 100 * (1 - (alpha / 2))))
        elif side == "lesser":
            lb = -np.inf
            ub = float(np.percentile(ub_bootstrap, 100 * (1 - alpha)))
        elif side == "greater":
            lb = float(np.percentile(lb_bootstrap, 100 * alpha))
            ub = np.inf
        else:
            raise ValueError(f"Unrecognized input {side=:}")

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
        propensity_scores: Union[List[float], npt.NDArray[np.float64]],
        outcomes: Union[
            List[Union[float, int]], npt.NDArray[Union[np.float64, np.int64]]
        ],
        population_size: int,
        estimand: Optional[Estimand] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            propensity_scores=propensity_scores,
            outcomes=outcomes,
            estimand=estimand,
            population_size=population_size,
            **kwargs,
        )
        self.population_size = population_size

    def point_estimate(self) -> float:
        """Calculate a point estimate."""
        return np.dot(self.weights, self.outcomes) / self.population_size

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
        n_over_N = len(self.propensity_scores) / self.population_size
        return np.mean(
            np.square(n_over_N * self.weights * self.outcomes - self.point_estimate())
        ) / (len(self.propensity_scores) - 1)

    def sensitivity_analysis(self, gamma: float = 6.0) -> Tuple[float, float]:
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

        wl, wu = self.estimand.sensitivity_region(self.weights, gamma)
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
        propensity_scores: Union[List[float], npt.NDArray[np.float64]],
        outcomes: Union[
            List[Union[float, int]], npt.NDArray[Union[np.float64, np.int64]]
        ],
        predicted_outcomes: Union[List[float], npt.NDArray[np.float64]],
        mean_predicted_outcome: float,
        population_size: int,
        estimand: Optional[Estimand] = None,
        **kwargs,
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
        n_over_N = len(self.propensity_scores) / self.population_size
        return np.mean(
            np.square(
                n_over_N * self.weights * self.outcomes - super().point_estimate()
            )
        ) / (len(self.propensity_scores) - 1)

    def sensitivity_analysis(self, gamma: float = 6.0) -> Tuple[float, float]:
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
       Methodology 81 (4). Oxford University Press: pg. 735–61.

    """

    def __init__(
        self,
        propensity_scores: Union[List[float], npt.NDArray[np.float64]],
        outcomes: Union[
            List[Union[float, int]], npt.NDArray[Union[np.float64, np.int64]]
        ],
        estimand: Optional[Estimand] = None,
        binary_outcomes: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            propensity_scores=propensity_scores,
            outcomes=outcomes,
            estimand=estimand,
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

    def sensitivity_analysis(self, gamma: float = 6.0) -> Tuple[float, float]:
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
           Methodology 81 (4). Oxford University Press: pg. 735–61.

        """
        if gamma < 1.0:
            raise ValueError("`gamma` must be >= 1")

        if gamma == 1.0:
            return self.point_estimate(), self.point_estimate()

        wl, wu = self.estimand.sensitivity_region(self.weights, gamma)
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
        propensity_scores: Union[List[float], npt.NDArray[np.float64]],
        outcomes: Union[
            List[Union[float, int]], npt.NDArray[Union[np.float64, np.int64]]
        ],
        predicted_outcomes: Union[
            List[Union[float, int]], npt.NDArray[Union[np.float64, np.int64]]
        ],
        mean_predicted_outcome: float,
        estimand: Optional[Estimand] = None,
        binary_outcomes: bool = True,
        **kwargs,
    ) -> None:
        if len(propensity_scores) != len(predicted_outcomes):
            raise ValueError(
                "Must have equal numbers of propensity scores and predicted outcomes."
            )

        super().__init__(
            propensity_scores=propensity_scores,
            outcomes=np.asarray(outcomes) - np.asarray(predicted_outcomes),
            estimand=estimand,
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

    def sensitivity_analysis(self, gamma: float = 6.0) -> Tuple[float, float]:
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


class TreatmentEffectEstimator(WeightingEstimator):
    """Base class for estimating treatment effects."""

    def __init__(
        self,
        control_estimator: MeanEstimator,
        treated_estimator: MeanEstimator,
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
        side: Literal["two-sided", "lesser", "greater"] = "two-sided",
    ) -> float:
        r"""Calculate a p-value.

        Computes a p-value against the null hypothesis:
            H0: TE = null_value,
        where TE is the treatment effect.

        Parameters
        ----------
         null_value : float
            The hypothesized treatment effect.
         side : ["two-sided", "lesser", "greater"], optional
            What kind of test:
              - "two-sided": H0: TE = `null_value` vs Halt: TE <> `null_value`.
              - "greater": H0: TE <= `null_value` vs Halt: TE > `null_value`.
              - "lesser": H0: TE >= `null_value` vs Halt: TE < `null_value`.
            For example, specifying side = "greater" returns a p-value that quantifies
            the strength of evidence against the null hypothesis that the treatment
            effect is less than `null_value`, in favor of the alternative hypothesis
            that the treatment effect is actually greater than `null_value`. Defaults to
            "two-sided".

        Returns
        -------
         p : float
            P-value.

        """
        # This is hacky but side has the opposite interpretation here compared to in
        # MeanEstimator:
        if side == "two-sided":
            return super().pvalue(null_value=null_value, side=side)
        elif side == "lesser":
            return super().pvalue(null_value=null_value, side="greater")

        return super().pvalue(null_value=null_value, side="lesser")

    def confidence_interval(
        self,
        alpha: float = 0.10,
        side: Literal["two-sided", "lesser", "greater"] = "two-sided",
    ) -> Tuple[float, float]:
        """Calculate confidence interval on the treatment effect.

        Parameters
        ----------
         alpha : float, optional
            P-value threshold, e.g. specify alpha=0.05 for a 95% confidence interval.
            Defaults to 0.10, corresponding to a 90% confidence interval.
         side : ["two-sided", "lesser", "greater"], optional
            What kind of test:
              - "two-sided": H0: TE = `null_value` vs Halt: TE <> `null_value`.
              - "greater": H0: TE <= `null_value` vs Halt: TE > `null_value`.
              - "lesser": H0: TE >= `null_value` vs Halt: TE < `null_value`.
            Defaults to "two-sided".

        Returns
        -------
         lb, ub : float
            Lower and upper bounds on the confidence interval. For one-sided intervals,
            only one of these will be finite:
              - "two-sided": lb and ub both finite
              - "greater": lb = -np.inf, ub = finite
              - "lesser": lb finite, ub = np.inf

        """
        return super().confidence_interval(alpha=alpha, side=side)

    def sensitivity_analysis(self, gamma: float = 6.0) -> Tuple[float, float]:
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

    def expanded_confidence_interval(
        self,
        alpha: float = 0.10,
        gamma: float = 6.0,
        side: Literal["two-sided", "lesser", "greater"] = "two-sided",
        B: int = 1_000,
        seed: Optional[
            Union[
                int,
                List[int],
                np.random.SeedSequence,
                np.random.BitGenerator,
                np.random.Generator,
            ]
        ] = None,
    ) -> Tuple[float, float]:
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
         side : ["two-sided", "lesser", "greater"], optional
            What kind of test:
              - "two-sided": H0: TE = `null_value` vs Halt: TE <> `null_value`.
              - "greater": H0: TE <= `null_value` vs Halt: TE > `null_value`.
              - "lesser": H0: TE >= `null_value` vs Halt: TE < `null_value`.
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
              - "lesser": lb = -np.inf, ub finite

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
            return self.confidence_interval(alpha=alpha, side=side)

        lb_bootstrap = np.zeros(B)
        ub_bootstrap = np.zeros(B)
        rng = np.random.default_rng(seed)
        for b in range(B):
            # Resample with replacement
            control_bootstrap_indices = rng.choice(
                range(len(self.control_estimator.propensity_scores)),
                size=len(self.control_estimator.propensity_scores),
                replace=True,
            )
            treated_bootstrap_indices = rng.choice(
                range(len(self.treated_estimator.propensity_scores)),
                size=len(self.treated_estimator.propensity_scores),
                replace=True,
            )

            # Implementation note for AIPW and SAIPW estimators:
            # `self.control_estimator.outcomes` is actually the adjusted outcomes,
            # outcome minus prediction, and
            # `self.control_estimator.extra_args["predicted_outcomes"]` is actually just
            # a vector of all zeros to make the constructor happy. So there is no need
            # to resample those predicted outcomes.
            control_estimator = self.control_estimator.__class__(
                propensity_scores=self.control_estimator.propensity_scores[
                    control_bootstrap_indices
                ],
                outcomes=self.control_estimator.outcomes[control_bootstrap_indices],
                estimand=self.control_estimator.estimand,
                **self.control_estimator.extra_args,
            )
            treated_estimator = self.treated_estimator.__class__(
                propensity_scores=self.treated_estimator.propensity_scores[
                    treated_bootstrap_indices
                ],
                outcomes=self.treated_estimator.outcomes[treated_bootstrap_indices],
                estimand=self.treated_estimator.estimand,
                **self.treated_estimator.extra_args,
            )

            # Calculate sensitivity interval for this bootstrap sample
            control_lb, control_ub = control_estimator.sensitivity_analysis(gamma=gamma)
            treated_lb, treated_ub = treated_estimator.sensitivity_analysis(gamma=gamma)
            lb_bootstrap[b] = treated_lb - control_ub
            ub_bootstrap[b] = treated_ub - control_lb

        # Calculate the expanded confidence interval as percentiles of the bootstrap estimates
        if side == "two-sided":
            lb = float(np.percentile(lb_bootstrap, 100 * alpha / 2))
            ub = float(np.percentile(ub_bootstrap, 100 * (1 - (alpha / 2))))
        elif side == "lesser":
            lb = -np.inf
            ub = float(np.percentile(ub_bootstrap, 100 * (1 - alpha)))
        elif side == "greater":
            lb = float(np.percentile(lb_bootstrap, 100 * alpha))
            ub = np.inf
        else:
            raise ValueError(f"Unrecognized input {side=:}")

        return lb, ub


class SimpleDifferenceEstimator(TreatmentEffectEstimator):
    """Simple difference in means.

    Parameters
    ----------
     control_outcomes, treated_outcomes : list_like
        Outcomes or responses for control and treatment groups, resp.

    Notes
    -----
    This class simple computes the average response in treatment, minus the average
    response in control. No weights, just straight averages. This may be helpful to see
    how different the adjusted estimator is from the simplest comparison of treatment
    and control, to see how biased this simple estimator is.

    """

    def __init__(
        self,
        control_outcomes: Union[
            List[Union[float, int]], npt.NDArray[Union[np.float64, np.int64]]
        ],
        treated_outcomes: Union[
            List[Union[float, int]], npt.NDArray[Union[np.float64, np.int64]]
        ],
        **kwargs,
    ) -> None:
        control_outcomes = np.asarray(control_outcomes)
        treated_outcomes = np.asarray(treated_outcomes)

        super().__init__(
            control_estimator=SIPWEstimator(
                propensity_scores=np.ones_like(control_outcomes, dtype=np.float64),
                outcomes=control_outcomes,
                estimand=SampleMean(),
            ),
            treated_estimator=SIPWEstimator(
                propensity_scores=np.ones_like(treated_outcomes, dtype=np.float64),
                outcomes=treated_outcomes,
                estimand=SampleMean(),
            ),
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

    """

    def __init__(
        self,
        control_propensity_scores: Union[List[float], npt.NDArray[np.float64]],
        control_outcomes: Union[
            List[Union[float, int]], npt.NDArray[Union[np.float64, np.int64]]
        ],
        treated_propensity_scores: Union[List[float], npt.NDArray[np.float64]],
        treated_outcomes: Union[
            List[Union[float, int]], npt.NDArray[Union[np.float64, np.int64]]
        ],
        estimator_class: Literal["IPW", "AIPW", "SIPW", "SAIPW"] = "SIPW",
        control_estimator_class: Optional[
            Literal["IPW", "AIPW", "SIPW", "SAIPW"]
        ] = None,
        treated_estimator_class: Optional[
            Literal["IPW", "AIPW", "SIPW", "SAIPW"]
        ] = None,
        control_kwargs: Optional[Dict[str, Any]] = None,
        treated_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
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

        super().__init__(
            control_estimator=classes[control_estimator_class or estimator_class](
                propensity_scores=np.ones_like(control_propensity_scores)
                - control_propensity_scores,
                outcomes=control_outcomes,
                estimand=PopulationMean(),
                **control_kwargs_nn,
            ),
            treated_estimator=classes[treated_estimator_class or estimator_class](
                propensity_scores=treated_propensity_scores,
                outcomes=treated_outcomes,
                estimand=PopulationMean(),
                **treated_kwargs_nn,
            ),
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
        control_propensity_scores: Union[List[float], npt.NDArray[np.float64]],
        control_outcomes: Union[
            List[Union[float, int]], npt.NDArray[Union[np.float64, np.int64]]
        ],
        treated_outcomes: Union[
            List[Union[float, int]], npt.NDArray[Union[np.float64, np.int64]]
        ],
        estimator_class: Literal["IPW", "AIPW", "SIPW", "SAIPW"] = "SIPW",
        control_estimator_class: Optional[
            Literal["IPW", "AIPW", "SIPW", "SAIPW"]
        ] = None,
        treated_estimator_class: Optional[
            Literal["IPW", "AIPW", "SIPW", "SAIPW"]
        ] = None,
        control_kwargs: Optional[Dict[str, Any]] = None,
        treated_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
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

        super().__init__(
            control_estimator=classes[control_estimator_class or estimator_class](
                propensity_scores=np.ones_like(control_propensity_scores)
                - control_propensity_scores,
                outcomes=control_outcomes,
                estimand=NonRespondentMean(),
                **control_kwargs_nn,
            ),
            treated_estimator=classes[treated_estimator_class or estimator_class](
                propensity_scores=np.ones_like(treated_outcomes),
                outcomes=treated_outcomes,
                estimand=SampleMean(),
                **treated_kwargs_nn,
            ),
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
        control_outcomes: Union[
            List[Union[float, int]], npt.NDArray[Union[np.float64, np.int64]]
        ],
        treated_propensity_scores: Union[List[float], npt.NDArray[np.float64]],
        treated_outcomes: Union[
            List[Union[float, int]], npt.NDArray[Union[np.float64, np.int64]]
        ],
        estimator_class: Literal["IPW", "AIPW", "SIPW", "SAIPW"] = "SIPW",
        control_estimator_class: Optional[
            Literal["IPW", "AIPW", "SIPW", "SAIPW"]
        ] = None,
        treated_estimator_class: Optional[
            Literal["IPW", "AIPW", "SIPW", "SAIPW"]
        ] = None,
        control_kwargs: Optional[Dict[str, Any]] = None,
        treated_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
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

        super().__init__(
            control_estimator=classes[control_estimator_class or estimator_class](
                propensity_scores=np.ones_like(control_outcomes),
                outcomes=control_outcomes,
                estimand=SampleMean(),
                **control_kwargs_nn,
            ),
            treated_estimator=classes[treated_estimator_class or estimator_class](
                propensity_scores=treated_propensity_scores,
                outcomes=treated_outcomes,
                estimand=NonRespondentMean(),
                **treated_kwargs_nn,
            ),
        )


class PATEEstimator(TreatmentEffectEstimator):
    """Estimator for the population average treatment effect.

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
     control_kwargs, treated_kwargs, kwargs : dict_like
        Extra kwargs to pass to the estimators. `kwargs` are passed to both estimators,
        while `control_kwargs` is only passed to the control estimator and
        `treated_kwargs` is only passed to the treated estimator.

    Notes
    -----
    When units in the study are a non-representative sample from some larger population,
    estimating the treatment effect in the population involves elements of non-response
    bias in addition to bias from non-random assignment. PATEEstimator addresses both of
    these, weighting responses in both treatment and control to match the target
    population. This is sometimes called "generalization" in the literature, e.g.
    (Tipton and Hartman 2023).

    The propensity scores used by PATEEstimator should not be the probability of
    accepting treatment, as is the case with ATEEstimator, but rather the probability of
    being in the sample. The intuition here is that by reweighting the control units, we
    can estimate the population average of the control potential outcome, Y(0). Doing
    the same for the treated units estimates the population average of the treated
    potential outcome, Y(1). The PATE is then the difference of these estimates.

    References
    ----------
    - Tipton, Elizabeth and Erin Hartman. 2023. "Generalizability and Transportability."
      In Handbook of Matching and Weighting Adjustments for Causal Inference, , pg.
      39–59. Chapman and Hall/CRC.

    """

    def __init__(
        self,
        control_propensity_scores: Union[List[float], npt.NDArray[np.float64]],
        control_outcomes: Union[
            List[Union[float, int]], npt.NDArray[Union[np.float64, np.int64]]
        ],
        treated_propensity_scores: Union[List[float], npt.NDArray[np.float64]],
        treated_outcomes: Union[
            List[Union[float, int]], npt.NDArray[Union[np.float64, np.int64]]
        ],
        estimator_class: Literal["IPW", "AIPW", "SIPW", "SAIPW"] = "SIPW",
        control_estimator_class: Optional[
            Literal["IPW", "AIPW", "SIPW", "SAIPW"]
        ] = None,
        treated_estimator_class: Optional[
            Literal["IPW", "AIPW", "SIPW", "SAIPW"]
        ] = None,
        control_kwargs: Optional[Dict[str, Any]] = None,
        treated_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
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

        super().__init__(
            control_estimator=classes[control_estimator_class or estimator_class](
                propensity_scores=control_propensity_scores,
                outcomes=control_outcomes,
                estimand=PopulationMean(),
                **control_kwargs_nn,
            ),
            treated_estimator=classes[treated_estimator_class or estimator_class](
                propensity_scores=treated_propensity_scores,
                outcomes=treated_outcomes,
                estimand=PopulationMean(),
                **treated_kwargs_nn,
            ),
        )


class TransportEstimator(TreatmentEffectEstimator):
    """Estimator for the out-of-sample average treatment effect.

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
     control_kwargs, treated_kwargs, kwargs : dict_like
        Extra kwargs to pass to the estimators. `kwargs` are passed to both estimators,
        while `control_kwargs` is only passed to the control estimator and
        `treated_kwargs` is only passed to the treated estimator.

    Notes
    -----
    When interest is in the average treatment effect in a population distinct from the
    population from which the units in the study were sampled, the average treatment
    effect in the target population may be very different from the sample average
    treatment effect. Indeed, without additional assumptions, there is no guarantee we
    can learn anything about the effect of treatment on the target population, since the
    study involved a distinct population. For example, there is no reason to think we
    can learn about the effect of a treatment on women based on an experiment conducted
    on men, unless we expect the effect to be "transportable" (Tipton and Hartman 2023).

    If the variation in treatment effect is explained by observed covariates and the
    sample and target populations have good overlap in these covariates, then
    reweighting units to match the target population can provide an estimate of the
    average treatment effect in that population. This should be preceded by an
    exploration of treatment effect variation to identify those covariates that most
    strongly predict heterogeneity in treatment effect, and a comparison of the sample
    and target population in terms of these covariates.

    For example, domain expertise may lead us to expect that age explains most of the
    variation in treatment effect, and this may be supported by the data. Domain
    expertise may lead us to expect similar treatment effects for men and women of the
    same age, but since women were not included in the example experiment, the data
    cannot validate this. But this would then lead us to expect that the only reason the
    average treatment effect in the target population differs from that in the study, is
    if the age distribution of women differed from men. Reweighting the observations in
    the experiment to match the distribution of age in the target population would allow
    us to estimate the ATE.

    The propensity scores used by TransportEstimator should not be the probability of
    accepting treatment, as is the case with ATEEstimator, but rather the probability of
    being in the sample. The intuition here is that by reweighting the control units, we
    can estimate the average of the control potential outcome, Y(0), in the target
    population. Doing the same for the treated units estimates the average of the
    treated potential outcome, Y(1). The out-of-sample ATE is then the difference of
    these estimates.

    References
    ----------
    - Tipton, Elizabeth and Erin Hartman. 2023. "Generalizability and Transportability."
      In Handbook of Matching and Weighting Adjustments for Causal Inference, , pg.
      39–59. Chapman and Hall/CRC.

    """

    def __init__(
        self,
        control_propensity_scores: Union[List[float], npt.NDArray[np.float64]],
        control_outcomes: Union[
            List[Union[float, int]], npt.NDArray[Union[np.float64, np.int64]]
        ],
        treated_propensity_scores: Union[List[float], npt.NDArray[np.float64]],
        treated_outcomes: Union[
            List[Union[float, int]], npt.NDArray[Union[np.float64, np.int64]]
        ],
        estimator_class: Literal["IPW", "AIPW", "SIPW", "SAIPW"] = "SIPW",
        control_estimator_class: Optional[
            Literal["IPW", "AIPW", "SIPW", "SAIPW"]
        ] = None,
        treated_estimator_class: Optional[
            Literal["IPW", "AIPW", "SIPW", "SAIPW"]
        ] = None,
        control_kwargs: Optional[Dict[str, Any]] = None,
        treated_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
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

        super().__init__(
            control_estimator=classes[control_estimator_class or estimator_class](
                propensity_scores=control_propensity_scores,
                outcomes=control_outcomes,
                estimand=NonRespondentMean(),
                **control_kwargs_nn,
            ),
            treated_estimator=classes[treated_estimator_class or estimator_class](
                propensity_scores=treated_propensity_scores,
                outcomes=treated_outcomes,
                estimand=NonRespondentMean(),
                **treated_kwargs_nn,
            ),
        )
