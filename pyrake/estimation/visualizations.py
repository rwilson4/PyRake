"""Estimation visualizations."""

from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib import ticker
from matplotlib.axes import Axes

from .base_classes import WeightingEstimator


def meta_analysis(
    estimators: dict[str, WeightingEstimator],
    null_min: float,
    null_max: float,
    num_points: int = 1_000,
    alpha: float = 0.10,
    title: str | None = None,
    xlabel: str | None = None,
    xtick_format: str = ".0%",
    axis_label_size: int | None = None,
    tick_label_size: int | None = None,
    legend_label_size: int | None = None,
    legend_placement: str | None = None,
    ax: Axes | None = None,
) -> tuple[float, tuple[float, float], Axes]:
    """Combine estimates from multiple platforms.

    Parameters
    ----------
     estimators : dict_like
         Estimators for each platform. Keys should be the name of the platform, and
         value should be a WeightingEstimator corresponding to a particular metric.
     null_min, null_max : float, optional
         Range of null hypotheses to test.
     num_points : int, optional
         Number of hypotheses to check. Defaults to 1_000.
     alpha : float, optional
         Threshold for statistical significance. Defaults to 0.10.
     title : str, optional
         Optional title for figure.
     xlabel : str, optional
         Optional xlabel for figure. Defaults to "Null Hypothesis".
     xtick_format : str, optional
         Format for tick marks on x-axis. Defaults to ".0%", leading to xticks like
         "50%".
     axis_label_size, tick_label_size, legend_label_size : int, optional
         Font sizes for various figure elements.
     legend_placement : str, optional
         Where to put the legend.
     ax : Axes, optional
         Where to plot the figure.

    Returns
    -------
     point_estimate : float
         Combined point estimate.
     confidence_interval : tuple[float, float]
         Combined confidence interval.
     ax : Axes
         The figure.

    """
    null_values = np.linspace(null_min, null_max, num_points)
    pvals = {
        f"{platform} ({alternative})": np.array(
            [
                estimator.pvalue(
                    null_value=null_value,
                    alternative=cast("Literal['less', 'greater']", alternative),
                )
                for null_value in null_values
            ]
        )
        for platform, estimator in estimators.items()
        for alternative in ["less", "greater"]
    }
    pvals["Null Hypothesis"] = null_values
    df_meta = pd.DataFrame(pvals)

    def fisher(pvals: list[float]) -> float:
        return stats.chi2.sf(-2 * np.log(np.asarray(pvals)).sum(), 2 * len(pvals))

    # Combine pvals across all platforms using Fisher's method of meta-analysis.
    for alternative in ["less", "greater"]:
        df_meta[f"Combined ({alternative})"] = df_meta.apply(
            lambda row, alternative=alternative: fisher(
                [row[f"{platform} ({alternative})"] for platform in estimators]
            ),
            axis=1,
        )

    # Calculate a two-sided p-value based on both one-side p-values.
    for platform in [*list(estimators.keys()), "Combined"]:
        df_meta[platform] = df_meta.apply(
            lambda row, platform=platform: min(
                1.0, 2 * min(row[f"{platform} (less)"], row[f"{platform} (greater)"])
            ),
            axis=1,
        )

    df_meta = df_meta[["Null Hypothesis", *list(estimators.keys()), "Combined"]]

    # Calculate max pvals for each platform so we know how tall the vertical lines should be.
    max_pvals = df_meta.values[:, 1:].max(axis=0)

    # Calculate the point estimate as the null hypothesis with maximal p-value.
    # If there are multiple hypotheses with that same p-value, use the midpoint.
    pe_low = df_meta["Null Hypothesis"].loc[df_meta["Combined"].values.argmax()]  # type: ignore[union-attr]
    pe_high = df_meta["Null Hypothesis"].loc[
        len(null_values) - 1 - df_meta["Combined"].values[::-1].argmax()  # type: ignore[union-attr]
    ]
    point_estimate = 0.5 * (pe_low + pe_high)

    # Find the first and last null hypotheses with p-values larger than the threshold.
    ci_low = df_meta["Null Hypothesis"].loc[
        (df_meta["Combined"] >= alpha).values.argmax()  # type: ignore[union-attr]
    ]
    ci_high = df_meta["Null Hypothesis"].loc[
        len(null_values) - 1 - (df_meta["Combined"].values[::-1] >= alpha).argmax()  # type: ignore[union-attr,operator]
    ]

    # Plot the p-value curves for each platform and the combination.
    if ax is None:
        _, ax = plt.subplots()
    else:
        plt.sca(ax)

    df_meta = df_meta.set_index("Null Hypothesis").unstack().reset_index()
    df_meta.columns = ["Platform", "Null Hypothesis", "P-Value"]
    sns.lineplot(
        data=df_meta,
        x="Null Hypothesis",
        y="P-Value",
        hue="Platform",
        linewidth=3,
        ax=ax,
    )

    # Show the significance threshold and the point estimates.
    xlim = plt.xlim()
    ylim = plt.ylim()
    ax.hlines(y=alpha, xmin=xlim[0], xmax=xlim[1], colors="black")
    ax.vlines(
        x=[estimator.point_estimate() for estimator in estimators.values()]
        + [point_estimate],
        ymin=max(0.0, ylim[0]),
        ymax=[min(1.0, ylim[1], max_pval) for max_pval in max_pvals],
        colors=["blue", "orange", "green"],
    )
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    # Format plot.
    if title is not None:
        ax.set_title(title, fontsize=axis_label_size)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if axis_label_size is not None:
        ax.xaxis.label.set_fontsize(axis_label_size)
        ax.yaxis.label.set_fontsize(axis_label_size)

    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: ("{0:" + xtick_format + "}").format(x))
    )
    if tick_label_size is not None:
        ax.tick_params(axis="both", which="major", labelsize=tick_label_size)

    ax.legend(fontsize=legend_label_size, loc=legend_placement)

    return point_estimate, (ci_low, ci_high), ax
