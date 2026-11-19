"""Balance plots and the rank test behind them.

``plot_balance`` and ``plot_balance_2_sample`` draw standardized differences between
weighted sample means and their targets; ``wilcoxon_signed_rank`` is the one-sample rank
alternative offered wherever ``plot_balance`` takes ``test``.

"""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Mapping
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib.axes import Axes

WilcoxonResult = namedtuple(
    "WilcoxonResult", ["statistic", "zstatistic", "pvalue", "log_pvalue"]
)


def wilcoxon_signed_rank(
    d: npt.NDArray[np.float64], w: npt.NDArray[np.float64] | None = None
) -> WilcoxonResult:
    """Perform Wilcoxon's signed rank test.

    Parameters
    ----------
     d : npt.NDArray[np.float64]
        The data, one-dimensional.
     w : npt.NDArray[np.float64] | None, optional
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
     log_pvalue : float
        Natural log of the p-value, for aggregating without underflow.

    Raises
    ------
     ValueError
        When ``d`` is not one-dimensional, or ``w`` is given with a shape different from
        ``d``.

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


def _resolve_balance_means(
    means: Mapping[str, npt.NDArray[np.float64] | pd.Series],
    weights: dict[str, npt.NDArray[np.float64]],
    covariates: list[str],
) -> dict[str, npt.NDArray[np.float64]]:
    """Validate model means; named Series may cover only some covariates.

    Parameters
    ----------
     means : Mapping[str, npt.NDArray[np.float64] | pd.Series]
        Model-estimated means keyed by method name.
     weights : dict[str, npt.NDArray[np.float64]]
        Weight sets keyed by method name; names must not overlap ``means``.
     covariates : list[str]
        Feature names, in order.

    Returns
    -------
     resolved : dict[str, npt.NDArray[np.float64]]
        Finite covariate vectors in ``covariates`` order, one per method.

    Raises
    ------
     ValueError
        When a method name appears in both ``means`` and ``weights``, a Series has
        unknown or duplicate covariates, or any mean is not a finite covariate vector.

    """
    if set(means) & set(weights):
        raise ValueError("`means` and `weights` must have distinct method names")
    resolved = {}
    for name, mean in means.items():
        if isinstance(mean, pd.Series):
            if (
                mean.empty
                or not mean.index.is_unique
                or not set(mean.index) <= set(covariates)
            ):
                raise ValueError(f"Unknown or duplicate mean covariates for {name!r}")
            if not np.isfinite(mean.to_numpy(dtype=float)).all():
                raise ValueError(f"Means for {name!r} must be finite")
            values = mean.reindex(covariates).to_numpy(dtype=float)
        else:
            values = np.asarray(mean, dtype=float)
            if values.shape != (len(covariates),) or not np.isfinite(values).all():
                raise ValueError(
                    f"Means for {name!r} must be a finite covariate vector"
                )
        resolved[name] = values
    return resolved


def plot_balance(
    X: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    covariates: list[str],
    weights: dict[str, npt.NDArray[np.float64]],
    test: None | (Literal["z", "wilcoxon"] | list[Literal["z", "wilcoxon"]]) = None,
    sigma: npt.NDArray[np.float64] | None = None,
    signed: bool = True,
    title: str | None = None,
    quantify_imbalance: bool = True,
    imbalance_pval_threshold: float = 0.2,
    legend_placement: str = "lower right",
    ax: Axes | None = None,
    means: Mapping[str, npt.NDArray[np.float64] | pd.Series] | None = None,
) -> tuple[pd.DataFrame, Axes]:
    """Plot one-sample z-statistics for covariates.

    The one-sample z-statistic tests a null hypothesis regarding the mean or center of
    symmetry of the sample. For covariates without outliers, this can be as simple as
    the difference in sample mean and population mean, but for covariates with fat
    tails, it's better to use a non-parametric test for location.

    Parameters
    ----------
     X : npt.NDArray[np.float64]
        X_ij = feature j for respondent i.
     mu : npt.NDArray[np.float64]
        Mean (or median; see Notes) of covariates in target population.
     covariates : list[str]
        Feature names.
     weights : dict[str, npt.NDArray[np.float64]]
        One or more sets of weights.
     test : ["z", "wilcoxon"] or list of such options, optional
        Test to apply when comparing sample to population, for each covariate. The
        default behavior is to use simple z tests, which just compare the weighted
        sample mean to the population mean. For continuous covariates with potentially
        large values, means can be unstable, so we also support using the one-sample
        Wilcoxon signed rank test. See Notes. This input can be specified for each
        covariate separately.
     sigma : npt.NDArray[np.float64], optional
        Sqrt of population variance. Only used with ``test='z'``. If not specified, the
        (sqrt of the) sample variance of ``X`` will be used.
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
        If specified, plot on ``ax``.
     means : dict of covariate vectors, optional
        Additional methods' model-estimated means, in covariate order and original
        units. A Series may name a subset of covariates; omitted entries are not plotted
        or counted in that method's imbalance. Method names must not overlap weights.
        Only the 'z' diagnostic supports means; there are no synthetic respondent
        weights. Uses the same nominal M and sigma as weighted means. These coordinates
        need not be inferential z-statistics for a model-based estimate.

    Returns
    -------
     df_balance : pandas DataFrame
        The z-statistics for each covariate and each set of weights.
     ax : Axes
        The figure.

    Raises
    ------
     ValueError
        On dimension mismatches between ``X``, ``mu``, ``covariates``, ``sigma`` and
        ``test``; on an unknown test name; when model means accompany a non-'z'
        diagnostic; or when ``imbalance_pval_threshold`` falls outside (0, 1].

    Notes
    -----
    The Wilcoxon signed rank test computes the difference between the sample values and
    mu, which should be the population median for this test. The test statistic is the
    weighted sum of the ranks corresponding to sample values larger than the population
    median.

    The imbalance is quantified by aggregating the p-values for each covariate. Any
    p-values less than ``imbalance_pval_threshold`` represent covariates with large
    imbalance; other covariates are ignored. The product of these p-values is a measure
    of imbalance, but to avoid numeric underflow when working with very small p-values
    (and very out-of-balance covariates), we report the logarithm of the product of
    p-values, which is equivalent to the sum of logs of the p-values. We negate the
    result to get a non-negative metric, with larger values indicating greater
    imbalance. When ``quantify_imbalance`` is True, the resulting metrics are included
    in the plot legend.

    """
    if title is None:
        title = "Covariate Balance with Target Population"

    M, p = X.shape
    if len(mu) != p or len(covariates) != p:
        raise ValueError(
            "Dimension mismatch: `X` should have one column for every entry in `mu`"
        )

    test_nn: list[str]
    if test is None:
        test_nn = ["z"] * p
    elif isinstance(test, list):
        if len(test) != p:
            raise ValueError("`test` must be scalar or same length as `mu`")
        for t in test:
            if t not in ("z", "wilcoxon"):
                raise ValueError("Entries of `test` must be either 'z' or 'wilcoxon'")
        test_nn = list(test)
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

    assert sigma is not None
    resolved_means = _resolve_balance_means(
        {} if means is None else means, weights, covariates
    )
    if resolved_means and any(t != "z" for t in test_nn):
        raise ValueError(
            "Model means support only the mean-difference ('z') diagnostic"
        )
    methods = []
    names = []
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
            names.append(covariates[ip])
            z_stats.append(z)
            pvals.append(pval)
            log_pvals.append(log_pval)

    for method, mean in resolved_means.items():
        for ip in range(p):
            if np.isnan(mean[ip]):
                continue
            z = (mean[ip] - mu[ip]) * np.sqrt(M) / sigma[ip]
            methods.append(method)
            names.append(covariates[ip])
            z_stats.append(z)
            pvals.append(2.0 * stats.norm.sf(abs(z)))
            log_pvals.append(np.log(2.0) + stats.norm.logsf(abs(z)))
    method_names = list(weights) + list(resolved_means)
    df_balance = pd.DataFrame(
        {
            "Method": methods,
            "covariate": names,
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

        # 0.0 - x rather than -x: a perfectly balanced method's mean is 0.0, and its
        # negation, -0.0, would print in the legend as "-0".
        imbalance = df_balance.groupby("Method")["log_pval"].apply(
            lambda log_pvals: 0.0
            - np.mean(
                np.where(
                    log_pvals <= np.log(imbalance_pval_threshold),
                    log_pvals,
                    0.0,
                )
            )
        )
        for key in method_names:
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
    X1: npt.NDArray[np.float64],
    X2: npt.NDArray[np.float64],
    covariates: list[str],
    weights1: dict[str, npt.NDArray[np.float64]],
    weights2: dict[str, npt.NDArray[np.float64]],
    sigma1: npt.NDArray[np.float64] | None = None,
    sigma2: npt.NDArray[np.float64] | None = None,
    signed: bool = True,
    title: str | None = None,
    quantify_imbalance: bool = True,
    imbalance_pval_threshold: float = 0.2,
    legend_placement: str = "lower right",
    ax: Axes | None = None,
    means1: Mapping[str, npt.NDArray[np.float64] | pd.Series] | None = None,
    means2: Mapping[str, npt.NDArray[np.float64] | pd.Series] | None = None,
) -> tuple[pd.DataFrame, Axes]:
    """Plot z-statistics for covariates.

    If we were testing the null hypothesis that the two populations have equal covariate
    averages, then the z-statistic for each covariate would be:
       (1/M2) X2ᵀw2 - (1/M1) X1ᵀw1
       ---------------------------.
         √(σ1² / M1 + σ2² / M2)

    Parameters
    ----------
     X1 : npt.NDArray[np.float64]
        X_ij = feature j for respondent i in population 1.
     X2 : npt.NDArray[np.float64]
        X_ij = feature j for respondent i in population 2.
     covariates : list[str]
        Feature names.
     weights1 : dict[str, npt.NDArray[np.float64]]
        One or more sets of weights for population 1.
     weights2 : dict[str, npt.NDArray[np.float64]]
        One or more sets of weights for population 2.
     sigma1 : npt.NDArray[np.float64], optional
        Square root of variances for population 1, for putting all covariates on the
        same scale. See below for which values to pass.
     sigma2 : npt.NDArray[np.float64], optional
        Square root of variances for population 2. The exact values to pass here depend
        on context, so use judgment about which z-statistic you want to calculate. The
        key here is to use the same variances regardless of which weights are applied,
        so you could use variances of each sample, or variances of the respective
        populations.
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
        If specified, plot on ``ax``.
     means1 : dict of covariate vectors, optional
        Model means for additional matching method names for population 1, distinct from
        weights.
     means2 : dict of covariate vectors, optional
        Model means for population 2, with keys matching ``means1``. Uses (mean2 -
        mean1) on the same nominal-count/population-SD scale as weighted means; no
        weights or posterior tests are implied.

    Returns
    -------
     df_balance : pandas DataFrame
        The z-statistics for each covariate and each set of weights.
     ax : Axes
        The figure.

    Raises
    ------
     ValueError
        When the weight or mean method keys disagree across groups, ``X1`` and ``X2``
        differ in columns, or ``imbalance_pval_threshold`` falls outside (0, 1].

    Notes
    -----
    The imbalance is quantified by aggregating the p-values for each covariate.
    Presently we are just doing a z-test on the difference in means, but a future state
    might use something non-parametric.

    Any p-values less than ``imbalance_pval_threshold`` represent covariates with large
    imbalance; other covariates are ignored. The product of these p-values is a measure
    of imbalance, but to avoid numeric underflow when working with very small p-values
    (and very out-of-balance covariates), we report the logarithm of the product of
    p-values, which is equivalent to the sum of logs of the p-values. We negate the
    result to get a non-negative metric, with larger values indicating greater
    imbalance.

    When ``quantify_imbalance`` is True, the resulting metrics are included in the plot
    legend.

    """
    if set(weights1.keys()) != set(weights2.keys()):
        raise ValueError("`weights1` and `weights2` must have consistent keys.")

    if title is None:
        title = "Covariate Balance with Target Population"

    if X1.shape[1] != X2.shape[1]:
        raise ValueError("X1 and X2 must have the same columns")

    M1 = X1.shape[0]
    M2 = X2.shape[0]

    if sigma1 is None:
        sigma1 = np.ones((X1.shape[1],))

    if sigma2 is None:
        sigma2 = np.ones((X2.shape[1],))

    means1 = {} if means1 is None else means1
    means2 = {} if means2 is None else means2
    if set(means1) != set(means2):
        raise ValueError("`means1` and `means2` must have consistent keys")
    resolved_means1 = _resolve_balance_means(means1, weights1, covariates)
    resolved_means2 = _resolve_balance_means(means2, weights2, covariates)
    differences = {
        k: ((X2.T @ w2) / M2 - (X1.T @ weights1[k]) / M1)
        / np.sqrt(np.square(sigma1) / M1 + np.square(sigma2) / M2)
        for k, w2 in weights2.items()
    }
    differences.update(
        {
            k: (mean2 - resolved_means1[k])
            / np.sqrt(np.square(sigma1) / M1 + np.square(sigma2) / M2)
            for k, mean2 in resolved_means2.items()
        }
    )
    df_balance = pd.DataFrame(differences, index=covariates).unstack().reset_index()
    df_balance.columns = ["Method", "covariate", "delta"]
    omitted = {
        (name, covariates[ip])
        for name in resolved_means1
        for ip in range(len(covariates))
        if np.isnan(resolved_means1[name][ip]) or np.isnan(resolved_means2[name][ip])
    }
    df_balance = df_balance.loc[
        [
            (method, covariate) not in omitted
            for method, covariate in zip(
                df_balance["Method"], df_balance["covariate"], strict=True
            )
        ]
    ]
    df_balance["abs_delta"] = df_balance["delta"].abs()
    if quantify_imbalance:
        if imbalance_pval_threshold <= 0.0 or imbalance_pval_threshold > 1.0:
            raise ValueError(
                "`imbalance_pval_threshold` should be strictly positive and <= 1.0"
            )

        imbalance = df_balance.groupby("Method")["abs_delta"].apply(
            # 0.0 - x rather than -x, as in plot_balance, so balance prints as "0".
            lambda group: 0.0
            - np.mean(
                np.where(
                    (log_pval := np.log(2.0) + stats.norm.logsf(group))
                    <= np.log(imbalance_pval_threshold),
                    log_pval,
                    0.0,
                )
            )
        )
        for key in differences:
            if key not in imbalance.index:
                continue
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
