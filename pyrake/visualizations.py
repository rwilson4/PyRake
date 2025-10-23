"""Visualizations."""

from collections import namedtuple
from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib import ticker
from matplotlib.axes import Axes

from .estimators import WeightingEstimator

WilcoxonResult = namedtuple(
    "WilcoxonResult", ["statistic", "zstatistic", "pvalue", "log_pvalue"]
)


def wilcoxon_signed_rank(
    d: npt.NDArray[np.float64], w: Optional[npt.NDArray[np.float64]] = None
) -> WilcoxonResult:
    """Perform Wilcoxon's signed rank test.

    Parameters
    ----------
     d : list_like
        The data.
     w : list_like, optional
        Weights. See Notes.

    Returns
    -------
     statistic : float
        The signed rank statistic, T+.
     zstatistic : float
        The signed rank statistic, minus its expected value under H0, divided by the
        square root of its variance.
     pvalue : float
        P-value.

    Notes
    -----
    Per (Hollander and Wolfe, 1999, §3.1), Wilcoxon's signed rank test addresses the
    one-sample location problem: it tests the null hypothesis that the values d are
    symmetrically distributed around 0. To test some other center, subtract it from d
    before calling this function. The test ranks the absolute values of the elements,
    and then sums the ranks for positive entries.

    This function supports weighted observations. Instead of summing the ranks for
    positive entries, we calculate the weighted sum, where the weights are scaled to
    have mean 1. The unweighted test is equivalent to using a weight of 1 for all
    observations. I haven't seen much discussion of weighted tests in the literature, so
    this is a bit of a heuristic.

    References
    ----------
    Hollander, Myles and Douglas A. Wolfe. "Nonparametric Statistical Methods". 2nd
    edition. (1999) Wiley Series in Probability and Statistics.

    """
    if len(d.shape) != 1:
        raise ValueError("`d` must be 1d")

    nzi = np.logical_and(~np.isnan(d), d != 0)
    d_nz = d[nzi]
    count = len(d_nz)

    r = stats.rankdata(np.abs(d_nz))
    if w is None:
        T_plus = np.sum((d_nz > 0) * r)
    elif d.shape != w.shape:
        raise ValueError("`d` and `w` must have the same shape")
    else:
        T_plus = np.sum((d_nz > 0) * w[nzi] * r) / np.mean(w[nzi])

    mn = count * (count + 1.0) * 0.25
    se = count * (count + 1.0) * (2.0 * count + 1.0)

    # Handle ties
    replist, repnum = stats.find_repeats(r)
    if repnum.size != 0:
        se -= 0.5 * (repnum * (repnum - 1) * (repnum + 1)).sum()

    se = np.sqrt(se / 24)

    z = (T_plus - mn) / se
    pval = 2.0 * stats.norm.sf(abs(z))
    log_pval = np.log(2.0) + stats.norm.logsf(abs(z))
    return WilcoxonResult(T_plus, z, pval, log_pval)


def plot_balance(
    X: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    covariates: List[str],
    weights: Dict[str, npt.NDArray[np.float64]],
    test: Optional[
        Union[Literal["z", "wilcoxon"], List[Literal["z", "wilcoxon"]]]
    ] = None,
    sigma: Optional[npt.NDArray[np.float64]] = None,
    signed: bool = True,
    title: Optional[str] = None,
    quantify_imbalance: bool = True,
    imbalance_pval_threshold: float = 0.2,
    legend_placement: str = "lower right",
    ax: Optional[Axes] = None,
) -> Tuple[pd.DataFrame, Axes]:
    r"""Plot one-sample z-statistics for covariates.

    The one-sample z-statistic tests a null hypothesis regarding the mean or center of
    symmetry of the sample. For covariates without outliers, this can be as simple as
    the difference in sample mean and population mean, but for covariates with fat
    tails, it's better to use a non-parametric test for location.

    Parameters
    ----------
     X : npt.NDArray
        X_ij = feature j for respondent i.
     mu : npt.NDArray
        Mean (or median; see Notes) of covariates in target population.
     covariates : List[str]
        Feature names.
     weights : Dict[str, array]
        One or more sets of weights.
     test : ["z", "wilcoxon"] or list of such options, optional
        Test to apply when comparing sample to population, for each covariate. The
        default behavior is to use simple z tests, which just compare the weighted
        sample mean to the population mean. For continuous covariates with potentially
        large values, means can be unstable, so we also support using the one-sample
        Wilcoxon signed rank test. See Notes. This input can be specified for each
        covariate separately.
     sigma : npt.NDArray, optional
        Sqrt of population variance. Only used with `test='z'`. If not specified, the
        (sqrt of the) sample variance of `X` will be used.
     signed : bool, optional
        If True (default), plot the signed z-statistics. Otherwise, plot the absolute
        value of the z-statistics, which results in more condensed plots.
     title : str, optional
        A title for the plot.
     quantify_imbalance : bool, optional
        If True (default), annotate legend with the imbalance metric for each set of
        weights. Larger values indicate greater imbalance. See Notes.
     imbalance_pval_threshold : float, optional
        P-value threshold for calculating imbalance metric. See Notes. Should be in (0,
        1]. Defaults to 0.2.
     legend_placement : str, optional
        Where to place the legend. Defaults to "lower right".
     ax : matplotlib Axes, optional
        If specified, plot on `ax`.

    Returns
    -------
     df_balance : pandas DataFrame
        The z-statistics for each covariate and each set of weights.
     ax : Axes
        The figure.

    Notes
    -----
    The Wilcoxon signed rank test computes the difference between the sample values and
    mu, which should be the population median for this test. The test statistic is the
    weighted sum of the ranks corresponding to sample values larger than the population
    median.

    The imbalance is quantified by aggregating the p-values for each covariate.  Any
    p-values less than `imbalance_pval_threshold` represent covariates with large
    imbalance; other covariates are ignored. The product of these p-values is a measure
    of imbalance, but to avoid numeric underflow when working with very small p-values
    (and very out-of-balance covariates), we report the logarithm of the product of
    p-values, which is equivalent to the sum of logs of the p-values. We negate the
    result to get a non-negative metric, with larger values indicating greater
    imbalance. When `quantify_imbalance` is True, the resulting metrics are included in
    the plot legend.

    """
    if title is None:
        title = "Covariate Balance with Target Population"

    M, p = X.shape
    if len(mu) != p or len(covariates) != p:
        raise ValueError(
            "Dimension mismatch: `X` should have one column for every entry in `mu`"
        )

    if test is None:
        test_nn: List[str] = ["z"] * p
    elif isinstance(test, list):
        if len(test) != p:
            raise ValueError("`test` must be scalar or same length as `mu`")
        for t in test:
            if t not in ("z", "wilcoxon"):
                raise ValueError("Entries of `test` must be either 'z' or 'wilcoxon'")
        test_nn = test
    elif test not in ("z", "wilcoxon"):
        raise ValueError("`test` must be either 'z' or 'wilcoxon'")
    else:
        test_nn = [test] * p

    if sigma is None:
        sigma = np.sqrt(np.var(X, ddof=1, axis=0))
    elif len(sigma) != p:
        raise ValueError(
            "Dimension mismatch: `sigma` should have one entry for each entry in `mu`"
        )

    methods = []
    z_stats = []
    pvals = []
    log_pvals = []
    for method, w in weights.items():
        for ip in range(p):
            if test_nn[ip] == "z":
                z = (np.dot(X[:, ip], w) / M - mu[ip]) * np.sqrt(M) / sigma[ip]
                pval = 2.0 * stats.norm.sf(abs(z))
                log_pval = np.log(2.0) + stats.norm.logsf(abs(z))
            elif test_nn[ip] == "wilcoxon":
                _, z, pval, log_pval = wilcoxon_signed_rank(X[:, ip] - mu[ip], w)
            else:
                raise ValueError(f"Unrecognized test '{test_nn[ip]}'")

            methods.append(method)
            z_stats.append(z)
            pvals.append(pval)
            log_pvals.append(log_pval)

    df_balance = pd.DataFrame(
        {
            "Method": methods,
            "covariate": list(covariates) * len(weights),
            "delta": z_stats,
            "pval": pvals,
            "log_pval": log_pvals,
        }
    )

    df_balance["abs_delta"] = df_balance["delta"].abs()
    if quantify_imbalance:
        if imbalance_pval_threshold <= 0.0 or imbalance_pval_threshold > 1.0:
            raise ValueError(
                "`imbalance_pval_threshold` should be strictly positive and <= 1.0"
            )

        imbalance = df_balance.groupby("Method")["log_pval"].apply(
            lambda log_pvals: -np.mean(
                np.where(
                    log_pvals <= np.log(imbalance_pval_threshold),
                    log_pvals,
                    0.0,
                )
            )
        )
        for key in weights.keys():
            df_balance["Method"] = np.where(
                df_balance["Method"] == key,
                f"{key} (imbalance: {imbalance.loc[key]:,.0f})",
                df_balance["Method"],
            )

    # Plot the results
    if ax is None:
        _, ax = plt.subplots(figsize=(8, len(covariates) / 2))
    else:
        plt.sca(ax)

    sns.set()
    if signed:
        sns.scatterplot(data=df_balance, x="delta", y="covariate", hue="Method")
        plt.axvline(x=-1.96, color="black")
        plt.axvline(x=1.96, color="black")
    else:
        sns.scatterplot(data=df_balance, x="abs_delta", y="covariate", hue="Method")
        plt.axvline(x=1.96, color="black")
    plt.title(title)
    plt.xlabel("Standardized Difference in Means")
    plt.ylabel("")
    plt.legend(loc=legend_placement)

    return df_balance, ax


def plot_balance_2_sample(
    X1: npt.NDArray,
    X2: npt.NDArray,
    covariates: List[str],
    weights1: Dict[str, npt.NDArray],
    weights2: Dict[str, npt.NDArray],
    signed: bool = True,
    title: Optional[str] = None,
    quantify_imbalance: bool = True,
    imbalance_pval_threshold: float = 0.2,
    legend_placement: str = "lower right",
    ax: Optional[Axes] = None,
) -> Tuple[pd.DataFrame, Axes]:
    r"""Plot z-statistics for covariates.

    If we were testing the null hypothesis that the sample average equals the (known)
    population average, then the z-statistic for each covariate would be:
       (1/M) * X_i^T * w - mu
       ----------------------,
           sqrt(sigma^2)
    where X_i is the vector for feature i and sigma^2 is the estimated variance of
        (1/M) * X_i^T * w = (1/M^2) * \sum_j w_j^2 (X_ij - nu_i)^2.
    If X is whitened, (X_ij - nu_i)^2 = 1 for all i, so this is
    sigma^2 = (1 / M) * mean squared weight. When mean squared weight is 1 (like with a
    simple average), sigma^2 = 1 / M and the z-statistic is:
       sqrt(M) * ((1/M) * X_i^T * w - mu)
    In order to have a consistent notion of balance across all weights (regardless of
    mean-square), this is the formula we use.

    Parameters
    ----------
     X1, X2 : npt.NDArray
        X_ij = feature j for respondent i.
     covariates : List[str]
        Feature names.
     weights1, weights2 : Dict[str, array]
        One or more sets of weights.
     signed : bool, optional
        If True (default), plot the signed z-statistics. Otherwise, plot the two-sided
        z-statistics, which are always >= 0. This results in more condensed plots.
     title : str, optional
        A title for the plot.
     quantify_imbalance : bool, optional
        If True (default), annotate legend with the imbalance metric for each set of
        weights. Larger values indicate greater imbalance. See Notes.
     imbalance_pval_threshold : float, optional
        P-value threshold for calculating imbalance metric. See Notes. Should be in (0,
        1]. Defaults to 0.2.
     legend_placement : str, optional
        Where to place the legend. Defaults to "lower right".
     ax : matplotlib Axes, optional
        If specified, plot on `ax`.

    Returns
    -------
     df_balance : pandas DataFrame
        The z-statistics for each covariate and each set of weights.
     ax : Axes
        The figure.

    Notes
    -----
    The imbalance is quantified by aggregating the p-values for each covariate.
    Presently we are just doing a z-test on the difference in means, but a future state
    might use something non-parametric.

    Any p-values less than `imbalance_pval_threshold` represent covariates with large
    imbalance; other covariates are ignored. The product of these p-values is a measure
    of imbalance, but to avoid numeric underflow when working with very small p-values
    (and very out-of-balance covariates), we report the logarithm of the product of
    p-values, which is equivalent to the sum of logs of the p-values. We negate the
    result to get a non-negative metric, with larger values indicating greater
    imbalance.

    When `quantify_imbalance` is True, the resulting metrics are included in the plot
    legend.

    """
    if set(weights1.keys()) != set(weights2.keys()):
        raise ValueError("`weights1` and `weights2` must have consistent keys.")

    if title is None:
        title = "Covariate Balance with Target Population"

    M1 = X1.shape[0]
    M2 = X2.shape[0]

    df_balance = (
        pd.DataFrame(
            {
                k: ((1 / M2) * (X2.T @ w2) - (1 / M1) * (X1.T @ weights1[k]))
                / np.sqrt(1.0 / M1 + 1.0 / M2)
                for k, w2 in weights2.items()
            },
            index=covariates,
        )
        .unstack()
        .reset_index()
    )
    df_balance.columns = ["Method", "covariate", "delta"]
    df_balance["abs_delta"] = df_balance["delta"].abs()
    if quantify_imbalance:
        if imbalance_pval_threshold <= 0.0 or imbalance_pval_threshold > 1.0:
            raise ValueError(
                "`imbalance_pval_threshold` should be strictly positive and <= 1.0"
            )

        imbalance = df_balance.groupby("Method")["abs_delta"].apply(
            lambda group: -np.mean(
                np.where(
                    (log_pval := np.log(2.0) + stats.norm.logsf(group))
                    <= np.log(imbalance_pval_threshold),
                    log_pval,
                    0.0,
                )
            )
        )
        for key in weights1.keys():
            df_balance["Method"] = np.where(
                df_balance["Method"] == key,
                f"{key} (imbalance: {imbalance.loc[key]:,.0f})",
                df_balance["Method"],
            )

    # Plot the results
    if ax is None:
        _, ax = plt.subplots(figsize=(8, len(covariates) / 2))
    else:
        plt.sca(ax)

    sns.set()
    if signed:
        sns.scatterplot(data=df_balance, x="delta", y="covariate", hue="Method")
        plt.axvline(x=-1.96, color="black")
        plt.axvline(x=1.96, color="black")
    else:
        sns.scatterplot(data=df_balance, x="abs_delta", y="covariate", hue="Method")
        plt.axvline(x=1.96, color="black")
    plt.title(title)
    plt.xlabel("Standardized Difference in Means")
    plt.ylabel("")
    plt.legend(loc=legend_placement)

    return df_balance, ax


def meta_analysis(
    estimators: Dict[str, WeightingEstimator],
    null_min: float,
    null_max: float,
    num_points: int = 1_000,
    alpha: float = 0.10,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    xtick_format: str = ".0%",
    axis_label_size: Optional[int] = None,
    tick_label_size: Optional[int] = None,
    legend_label_size: Optional[int] = None,
    legend_placement: Optional[str] = None,
    ax: Optional[Axes] = None,
) -> Tuple[float, Tuple[float, float], Axes]:
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
     confidence_interval : Tuple[float, float]
         Combined confidence interval.
     ax : Axes
         The figure.

    """
    null_values = np.linspace(null_min, null_max, num_points)
    pvals = {
        f"{platform} ({side})": np.array(
            [
                estimator.pvalue(null_value=null_value, side=side)
                for null_value in null_values
            ]
        )
        for platform, estimator in estimators.items()
        for side in ["lesser", "greater"]
    }
    pvals["Null Hypothesis"] = null_values
    df_meta = pd.DataFrame(pvals)

    def fisher(pvals: List[float]) -> float:
        return stats.chi2.sf(-2 * np.log(np.asarray(pvals)).sum(), 2 * len(pvals))

    # Combine pvals across all platforms using Fisher's method of meta-analysis.
    for side in ["lesser", "greater"]:
        df_meta[f"Combined ({side})"] = df_meta.apply(
            lambda row, side=side: fisher(
                [row[f"{platform} ({side})"] for platform in estimators.keys()]
            ),
            axis=1,
        )

    # Calculate a two-sided p-value based on both one-side p-values.
    for platform in list(estimators.keys()) + ["Combined"]:
        df_meta[platform] = df_meta.apply(
            lambda row, platform=platform: min(
                1.0, 2 * min(row[f"{platform} (lesser)"], row[f"{platform} (greater)"])
            ),
            axis=1,
        )

    df_meta = df_meta[["Null Hypothesis"] + list(estimators.keys()) + ["Combined"]]

    # Calculate max pvals for each platform so we know how tall the vertical lines should be.
    max_pvals = df_meta.values[:, 1:].max(axis=0)

    # Calculate the point estimate as the null hypothesis with maximal p-value.
    # If there are multiple hypotheses with that same p-value, use the midpoint.
    pe_low = df_meta["Null Hypothesis"].loc[df_meta["Combined"].values.argmax()]
    pe_high = df_meta["Null Hypothesis"].loc[
        len(null_values) - 1 - df_meta["Combined"].values[::-1].argmax()
    ]
    point_estimate = 0.5 * (pe_low + pe_high)

    # Find the first and last null hypotheses with p-values larger than the threshold.
    ci_low = df_meta["Null Hypothesis"].loc[
        (df_meta["Combined"] >= alpha).values.argmax()
    ]
    ci_high = df_meta["Null Hypothesis"].loc[
        len(null_values) - 1 - (df_meta["Combined"].values[::-1] >= alpha).argmax()
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
        ax.xaxis.label.set_size(axis_label_size)
        ax.yaxis.label.set_size(axis_label_size)

    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: ("{0:" + xtick_format + "}").format(x))
    )
    if tick_label_size is not None:
        ax.tick_params(axis="both", which="major", labelsize=tick_label_size)

    ax.legend(fontsize=legend_label_size, loc=legend_placement)

    return point_estimate, (ci_low, ci_high), ax
