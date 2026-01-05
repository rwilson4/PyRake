"""Base classes for estimation."""

import math
from abc import ABC, abstractmethod, abstractproperty
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
from typing_extensions import Self


class Estimand(ABC):
    """Base class for things that can be estimated."""

    @abstractproperty
    def weights(self) -> npt.NDArray[np.float64]:
        """Calculate weights based on propensity scores."""
        pass

    @abstractmethod
    def sensitivity_region(
        self,
        gamma: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate a sensitivity region around baseline weights."""
        pass

    @abstractmethod
    def resample(self, bootstrap_indices: npt.NDArray[np.int64]) -> Self:
        """Create a new Estimand object based on resampled indices."""
        pass

    @staticmethod
    @abstractmethod
    def normalizing_factor(sample_size: int, population_size: int) -> float:
        """Calculate normalizing factor."""


class SimpleEstimand(Estimand):
    """Simple estimands."""

    def __init__(self, propensity_scores: npt.NDArray[np.float64]) -> None:
        self.propensity_scores = propensity_scores

    def resample(self, bootstrap_indices: npt.NDArray[np.int64]) -> Self:
        """Create a new Estimand object based on resampled indices."""
        return self.__class__(self.propensity_scores[bootstrap_indices])


class WeightingEstimator(ABC):
    """Base class for weighting estimators."""

    @abstractmethod
    def point_estimate(self) -> float:
        """Calculate a point estimate."""

    @abstractmethod
    def variance(self) -> float:
        """Calculate the variance."""

    @abstractmethod
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
         - Fogarty, Colin. 2023. "Sensitivity Analysis." In Handbook of Matching and
           Weighting Adjustments for Causal Inference, , pg. 553–82. Chapman and
           Hall/CRC.

        """

    @abstractmethod
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
              - "two-sided": H0: theta = `null_value` vs Halt: theta <> `null_value`.
              - "greater": H0: theta <= `null_value` vs Halt: theta > `null_value`.
              - "less": H0: theta >= `null_value` vs Halt: theta < `null_value`.
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
              - "two-sided": H0: theta = `null_value` vs Halt: theta <> `null_value`.
              - "greater": H0: theta <= `null_value` vs Halt: theta > `null_value`.
              - "less": H0: theta >= `null_value` vs Halt: theta < `null_value`.
            For example, specifying alternative = "greater" returns a p-value that quantifies
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

        if alternative == "greater" or alternative == "two-sided":
            p_greater = stats.norm.sf(t)
            if alternative == "greater":
                return p_greater

        if alternative == "less" or alternative == "two-sided":
            p_less = stats.norm.cdf(t)
            if alternative == "less":
                return p_less

        if alternative == "two-sided":
            return min(1.0, 2.0 * min(p_less, p_greater))

        raise ValueError(f"Unrecognized input {alternative=:}")

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
              - "two-sided": H0: theta = `null_value` vs Halt: theta <> `null_value`.
              - "greater": H0: theta <= `null_value` vs Halt: theta > `null_value`.
              - "less": H0: theta >= `null_value` vs Halt: theta < `null_value`.
            Defaults to "two-sided".

        Returns
        -------
         lb, ub : float
            Lower and upper bounds on the confidence interval. For one-sided intervals,
            only one of these will be finite:
              - "two-sided": lb and ub both finite
              - "greater": lb finite, ub = np.inf
              - "less": lb = -np.inf, ub finite

        """
        pe = self.point_estimate()
        se = math.sqrt(self.variance())
        if alternative == "two-sided":
            zcrit = stats.norm.isf(alpha / 2)
        else:
            zcrit = stats.norm.isf(alpha)

        if alternative == "greater" or alternative == "two-sided":
            lb = pe - zcrit * se
            if alternative == "greater":
                return lb, np.inf

        if alternative == "less" or alternative == "two-sided":
            ub = pe + zcrit * se
            if alternative == "less":
                return -np.inf, ub

        if alternative == "two-sided":
            return lb, ub

        raise ValueError(f"Unrecognized input {alternative=:}")

    def plot_sensitivity(
        self,
        gamma_lower: float = 1.0,
        gamma_upper: float = 6.0,
        num_points: int = 50,
        alpha: float = 0.10,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        B: int = 1_000,
        title: str | None = None,
        ylabel: str = "Sensitivity Interval",
        ytick_format: str = ".0%",
        axis_label_size: int | None = None,
        tick_label_size: int | None = None,
        legend_label_size: int | None = None,
        legend_placement: str | None = None,
        ax: Axes | None = None,
    ) -> tuple[pd.DataFrame, Axes]:
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
         alternative : ["two-sided", "less", "greater"], optional
            What kind of test:
              - "two-sided": H0: \bar{Y} = `null_value` vs Halt: \bar{Y} <>
                `null_value`.
              - "greater": H0: \bar{Y} <= `null_value` vs Halt: \bar{Y} > `null_value`.
              - "less": H0: \bar{Y} >= `null_value` vs Halt: \bar{Y} < `null_value`.
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
                alpha=alpha, gamma=gamma, alternative=alternative, B=B, seed=42
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

        plt.tight_layout()
        return df, ax
