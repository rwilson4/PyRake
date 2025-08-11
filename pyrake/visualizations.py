"""Visualizations."""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib.figure import Figure


def plot_balance(
    X: npt.NDArray,
    mu: npt.NDArray,
    covariates: List[str],
    weights: Dict[str, npt.NDArray],
    signed: bool = True,
    title: Optional[str] = None,
    quantify_imbalance: bool = True,
    imbalance_pval_threshold: float = 0.2,
    legend_placement: str = "lower right",
) -> Tuple[pd.DataFrame, Figure]:
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
     X : npt.NDArray
        X_ij = feature j for respondent i.
     mu : npt.NDArray
        Mean of covariates in target population.
     covariates : List[str]
        Feature names.
     weights : Dict[str, array]
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

    Returns
    -------
     df_balance : pandas DataFrame
        The z-statistics for each covariate and each set of weights.
     fig : Figure
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
    if title is None:
        title = "Covariate Balance with Target Population"

    M = X.shape[0]
    df_balance = (
        pd.DataFrame(
            {k: ((1 / M) * X.T.dot(w) - mu) * np.sqrt(M) for k, w in weights.items()},
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
            lambda group: -np.sum(
                np.log(
                    np.where(
                        (pval := stats.norm.sf(group) * 2.0)
                        <= imbalance_pval_threshold,
                        pval,
                        1.0,
                    )
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
    fig = plt.figure(figsize=(8, len(covariates) / 2))
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

    return df_balance, fig
