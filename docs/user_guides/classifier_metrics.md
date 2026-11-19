# Precision and Recall in a Target Population

A classifier's precision and recall are usually measured on a labeled
holdout set. When the labeled data were collected
non-representatively, as they almost always are, those numbers describe
the holdout set and not the population the model will be deployed on.
This guide shows how to estimate precision and recall in the target
population, with confidence intervals, by weighting the holdout set
with {class}`~pyrake.Rake` and treating each metric as a ratio of
population means with {class}`~pyrake.RatioEstimator`.

1. **Precision and recall are ratios of population means**, which is
   what makes the weighting machinery apply.
2. **Calibration weights** for the holdout set, balanced on the
   features and on the model's own score distribution, which is known
   exactly for the whole population.
3. **Point estimates and intervals** at a threshold, and then
   **precision and recall curves** across thresholds.
4. A **sensitivity analysis** for selection the features do not
   explain.

It builds on the {doc}`population_mean` guide, which introduces
calibration for a single sample.

```{testsetup}
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#52514e",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
    "xtick.color": "#898781",
    "ytick.color": "#898781",
    "legend.frameon": False,
    "lines.linewidth": 2,
})
```

## Precision and recall as ratios

Let $Y_i \in \{0, 1\}$ be the true label of unit $i$ in a population
of $N$, $S_i$ the model's score, and $F_i(t) = 1\{S_i \geq t\}$ the
indicator that the model flags it at threshold $t$. In the population,

$$
\mathrm{precision}(t) = \frac{\sum_i Y_i F_i(t)}{\sum_i F_i(t)}
= \frac{\bar{Y F}}{\bar{F}},
\qquad
\mathrm{recall}(t) = \frac{\sum_i Y_i F_i(t)}{\sum_i Y_i}
= \frac{\bar{Y F}}{\bar{Y}},
$$

where bars denote population means. Each is a ratio of two population
means of quantities we observe for every labeled unit: true positives
$Y F$, flags $F$, and positives $Y$. Weights that make the labeled
sample represent the population make both weighted means, and so their
ratio, estimate the population's. {class}`~pyrake.RatioEstimator`
computes the ratio of the two weighted means and its delta-method
variance, which accounts for the strong correlation between numerator
and denominator.

The two metrics fail to transfer from a biased sample for different
reasons. Suppose labeled units are chosen on their features $x$ alone,
so $P(Y = 1 \mid x)$ is the same in the sample and the population
(*covariate shift*). Precision is then the average of
$P(Y = 1 \mid x)$ over the flagged region, taken under the wrong
distribution of $x$. Recall is the share of positives that land in the
flagged region, and the sample's positives are concentrated wherever
labeling was concentrated. If labeling instead depends on the label
itself (*label shift*), recall transfers, since $P(x \mid Y)$ is
unchanged, while precision moves with the prevalence. Weighting on
features corrects covariate shift; label shift is what the sensitivity
analysis at the end is for.

## The example model

A platform hosts 100,000 posts and wants to flag spam. For every post
it knows whether the account is new, the number of links, the length
in hundreds of words, whether it was posted at night, and a phrase
score from a text model. We simulate the population, so that we know
the true answer:

```{doctest}
>>> import numpy as np
>>> import pyrake
>>> from scipy.special import expit
>>> rng = np.random.default_rng(1)
>>> N = 100_000
>>> new = (rng.random(N) < 0.2).astype(float)
>>> links = rng.poisson(1.0, N).astype(float)
>>> length = rng.gamma(2.0, 1.0, N)
>>> night = (rng.random(N) < 0.3).astype(float)
>>> phrases = rng.normal(0, 1, N)
>>> eta = -4.5 + 1.5 * new + 0.6 * links - 0.4 * length + 0.8 * night + 1.4 * phrases
>>> spam = (rng.random(N) < expit(eta)).astype(float)
>>> print(f"population spam rate {spam.mean():.3f}")
population spam rate 0.050
```

Human reviewers labeled a sample of posts, but the review queue
favored posts from new accounts and posts with many links, which looked
suspicious. The reviewed posts hold about twice the population's share
of spam:

```{doctest}
>>> reviewed = np.flatnonzero(rng.random(N) < expit(-3.6 + 1.8 * new + 0.6 * links))
>>> print(f"{len(reviewed)} reviewed, spam rate {spam[reviewed].mean():.3f}")
9365 reviewed, spam rate 0.097
```

We hold out half the reviewed posts at random and fit a logistic
regression on the other half. Any model would do: what follows needs
only its scores.

```{doctest}
>>> from scipy.optimize import minimize
>>> holdout = rng.random(len(reviewed)) < 0.5
>>> train, test = reviewed[~holdout], reviewed[holdout]
>>> features = np.column_stack([np.ones(N), new, links, length, night, phrases])
>>> def neg_log_likelihood(beta):
...     eta = features[train] @ beta
...     return np.sum(np.logaddexp(0, eta)) - eta @ spam[train]
>>> beta = minimize(neg_log_likelihood, np.zeros(6), method="BFGS").x
>>> score = expit(features @ beta)
>>> M = len(test)
>>> y, s = spam[test], score[test]
>>> print(f"{M} held-out posts")
4686 held-out posts
```

The held-out set is a fair test of the model in the sense that it was
not trained on, but it was drawn by the same review queue, so it is no
more representative than the training set.

## Calibration weights

The covariates are known for every post, not just their means, and so
is anything computed from them. In particular we can score the whole
population, so the share of posts the model flags at any threshold is
known exactly. That is the denominator of precision. We calibrate the
held-out set to the population means of the five features and to the
population shares above a ladder of score cut points:

```{doctest}
>>> cuts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
>>> covariates = np.column_stack(
...     [new, links, length, night, phrases] + [score >= c for c in cuts]
... ).astype(float)
>>> rake = pyrake.Rake(
...     distance=pyrake.KLDivergence(),
...     X=covariates[test],
...     mu=covariates.mean(axis=0),
...     min_weight=M / N,
... )
>>> result = rake.solve()
>>> print(result.status)
SolveStatus.OPTIMAL
>>> w = result.solution
>>> print(f"design effect {np.mean(w ** 2):.2f}")
design effect 1.90
```

The KL divergence from uniform weights gives weights of the form
$\exp(\lambda^T x)$, the inverse of a logistic selection model when
selection probabilities are small, which suits a queue that picks posts
by a logistic-like score of their features. If you can model selection
directly (here we know which posts were reviewed, so we could), use the
fitted inverse propensities as the baseline `v`, as in the
{doc}`population_mean` guide. The lower bound `M / N` keeps every
implied propensity score at most 1.

The score bands cost almost nothing in design effect, since the score
is a function of the features already balanced. What they buy is exact
agreement on how many posts the model flags at each cut point, and
tighter recall estimates at high thresholds, where few positives are
flagged.

## Precision and recall at a threshold

With weights $w$, the implied propensity scores are $(M / N) / w_i$.
A small helper builds a {class}`~pyrake.RatioEstimator` for each
metric. Passing uniform weights gives the unweighted holdout estimate,
with its own interval, for comparison:

```{doctest}
>>> def precision_and_recall(t, weights):
...     flagged = (s >= t).astype(float)
...     def ratio(numerator, denominator):
...         return pyrake.RatioEstimator(
...             propensity_scores=(M / N) / weights,
...             numerator_outcomes=numerator,
...             denominator_outcomes=denominator,
...         )
...     return ratio(y * flagged, flagged), ratio(y * flagged, y)
>>> precision, recall = precision_and_recall(0.5, w)
>>> lo, hi = precision.confidence_interval(alpha=0.05)
>>> print(f"precision {precision.point_estimate():.3f} ({lo:.3f}, {hi:.3f})")
precision 0.591 (0.494, 0.688)
>>> lo, hi = recall.confidence_interval(alpha=0.05)
>>> print(f"recall    {recall.point_estimate():.3f} ({lo:.3f}, {hi:.3f})")
recall    0.145 (0.109, 0.181)
```

Across the cut points, against the unweighted holdout and the truth,
which only a simulation can supply:

```{doctest}
>>> def truth(t):
...     flagged = score >= t
...     tp = np.sum(spam * flagged)
...     return tp / np.sum(flagged), tp / np.sum(spam)
>>> print("   t   precision: holdout weighted true   recall: holdout weighted true")
   t   precision: holdout weighted true   recall: holdout weighted true
>>> for t in cuts:
...     p0, r0 = precision_and_recall(t, np.ones(M))
...     p1, r1 = precision_and_recall(t, w)
...     p, r = truth(t)
...     print(f"{t:4.1f} {p0.point_estimate():18.3f} {p1.point_estimate():8.3f} {p:5.3f}"
...           f" {r0.point_estimate():15.3f} {r1.point_estimate():8.3f} {r:5.3f}")
 0.1              0.267    0.238 0.238           0.810    0.657 0.643
 0.2              0.373    0.344 0.363           0.620    0.428 0.443
 0.3              0.472    0.445 0.467           0.479    0.307 0.316
 0.4              0.551    0.557 0.549           0.373    0.226 0.219
 0.5              0.605    0.591 0.629           0.259    0.145 0.152
 0.6              0.685    0.703 0.710           0.176    0.097 0.096
 0.7              0.818    0.848 0.808           0.125    0.060 0.056
```

Precision on the holdout is somewhat too high at low thresholds, where
the flagged region is broad and its spam rate follows the holdout's
inflated prevalence; at high thresholds it is roughly right, since
there it is governed by $P(Y = 1 \mid x)$ for the most suspicious
posts, which the review queue did not change. Recall on the holdout is
badly wrong at every threshold: at 0.5 it claims the model catches a
quarter of all spam, when it catches 15%. The review queue found the
spam that new accounts and link-heavy posts produce, which the model
scores highly, and missed spam from established accounts, which it
scores low. The weighted estimates recover both metrics to within
their standard errors.

Weighting is not free. Where the holdout's precision is nearly
unbiased, as at high thresholds here, the weighted estimate is no more
accurate and is noisier: its standard error at 0.6 is 0.058 against
0.044 unweighted. But nothing in the holdout alone says where it is
unbiased, and for recall the weights are indispensable.

## Precision and recall curves

The same helper, run over a fine grid of thresholds, traces both
curves. The bands are pointwise 95% confidence intervals for the
weighted estimates:

```{doctest}
>>> thresholds = np.arange(0.05, 0.751, 0.01)
>>> def curve(weights):
...     out = []
...     for t in thresholds:
...         for metric in precision_and_recall(t, weights):
...             lo, hi = metric.confidence_interval(alpha=0.05)
...             out.append((metric.point_estimate(), lo, hi))
...     return np.array(out).reshape(len(thresholds), 2, 3)
>>> unweighted, weighted = curve(np.ones(M)), curve(w)
>>> true = np.array([truth(t) for t in thresholds])
>>> fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
>>> for k, (ax, name) in enumerate(zip(axes, ["Precision", "Recall"])):
...     band = ax.fill_between(thresholds, weighted[:, k, 1], weighted[:, k, 2],
...                            color="#2a78d6", alpha=0.18, linewidth=0)
...     lines = ax.plot(thresholds, true[:, k], color="#0b0b0b", label="Population (true)")
...     lines = ax.plot(thresholds, weighted[:, k, 0], color="#2a78d6",
...                     label="Holdout, calibrated (95% CI)")
...     lines = ax.plot(thresholds, unweighted[:, k, 0], color="#eb6834",
...                     linestyle="--", label="Holdout, unweighted")
...     title = ax.set_title(name, loc="left", color="#0b0b0b")
...     label = ax.set_xlabel("Score threshold")
...     limits = ax.set_ylim(0, 1)
>>> legend = axes[0].legend(loc="lower right")
>>> fig.tight_layout()
```

```{image} images/classifier_metrics_curves.png
:alt: Precision and recall against the score threshold. Precision rises from about 0.15 to 0.85 and the unweighted holdout runs slightly above the truth at low thresholds; recall falls from about 0.8 to 0.04, and the unweighted holdout lies far above the truth throughout. The calibrated estimates track the truth inside their confidence bands.
```

Plotting precision against recall, with the threshold implicit, gives
the familiar precision-recall curve:

```{doctest}
>>> fig, ax = plt.subplots(figsize=(5, 4.5))
>>> lines = ax.plot(true[:, 1], true[:, 0], color="#0b0b0b", label="Population (true)")
>>> lines = ax.plot(weighted[:, 1, 0], weighted[:, 0, 0], color="#2a78d6",
...                 label="Holdout, calibrated")
>>> lines = ax.plot(unweighted[:, 1, 0], unweighted[:, 0, 0], color="#eb6834",
...                 linestyle="--", label="Holdout, unweighted")
>>> labels = ax.set_xlabel("Recall"), ax.set_ylabel("Precision")
>>> limits = ax.set_xlim(0, 1), ax.set_ylim(0, 1)
>>> legend = ax.legend(loc="upper right")
```

```{image} images/classifier_metrics_pr.png
:alt: Precision-recall curves. The unweighted holdout curve lies well to the right of the true population curve, overstating the recall available at any precision; the calibrated curve follows the truth, with some wiggle.
```

The unweighted curve sits well to the right of the truth: read off the
holdout, the model seems to offer half again as much recall, or more,
at any given precision as it does in the population. A threshold chosen to hit a
recall target on the holdout would miss it badly in deployment.

## Reading the intervals

The confidence intervals are Wald intervals from the delta method,
with the variance of each weighted mean computed by
{class}`~pyrake.SIPWEstimator`. A few properties matter here:

- **They ignore the calibration.** The variance treats the weights as
  fixed inverse propensities, not as the solution of a calibration
  problem, so it gives no credit for the variance calibration removes.
  For quantities the calibration constraints pin down, such as recall
  at high thresholds, where the score bands fix the flagged share,
  expect the intervals to be conservative. Nor do they account for
  estimating the weights at all.
- **They are pointwise.** Each band covers its own threshold with 95%
  probability; the curve as a whole is covered less often. The
  estimates at neighboring thresholds share almost all their data, so
  errors run along the curve rather than scattering around it, as the
  calibrated curves above show.
- **They degrade where few units are flagged.** The delta method needs
  enough flagged positives and negatives for the ratio to be
  approximately normal. When every flagged holdout unit is positive,
  the estimated precision is 1 with a variance of exactly zero, an
  interval that is certainly too narrow. The grid above stops at 0.75,
  where 46 holdout posts are still flagged. For marginal cases
  {meth}`~pyrake.RatioEstimator.resample` gives a bootstrap, which
  is better behaved near the boundaries but cannot rescue a precision
  estimated from a handful of flags.
- **Choose the threshold before estimating at it.** Picking the
  threshold that maximizes a metric on the holdout and then reporting
  that metric at it is optimistic, for the same reason a model's
  training error is. Choose the operating point on a separate split,
  or from the model's scores alone (for example, a target flag rate,
  which is known exactly in the population), then evaluate.

## Sensitivity analysis

The weights correct for selection on the five features. If reviewers
also chose posts on something the features miss (because a user
reported them, say, and reported posts are more often spam), the
weights are wrong, and in a direction no balance check can detect.
{meth}`~pyrake.RatioEstimator.sensitivity_analysis` bounds the
estimate under a model in which each post's true odds of being
reviewed differ from those the weights imply by at most a factor of
$\sqrt{\Gamma}$:

```{doctest}
>>> for gamma in [1.5, 2.0, 3.0]:
...     lo, hi = precision.sensitivity_analysis(gamma=gamma)
...     print(f"Gamma {gamma}: precision at 0.5 between {lo:.3f} and {hi:.3f}")
Gamma 1.5: precision at 0.5 between 0.507 and 0.671
Gamma 2.0: precision at 0.5 between 0.446 and 0.722
Gamma 3.0: precision at 0.5 between 0.365 and 0.785
```

These ranges describe the point estimate only; widen them by sampling
error with {meth}`~pyrake.estimation.base_classes.WeightingEstimator.expanded_confidence_interval`,
or plot them over a range of $\Gamma$ with
{meth}`~pyrake.estimation.base_classes.WeightingEstimator.plot_sensitivity`. The same calls work
for recall. Label shift is the extreme case: when labeling depends on
the label, $\Gamma$ must be large enough to cover it.

## Other metrics

Many classification metrics are ratios of population means and fit the
same pattern. With `flagged` and `y` as in the helper:

| Metric              | Numerator                 | Denominator         |
|---------------------|---------------------------|---------------------|
| Precision           | `y * flagged`             | `flagged`           |
| Recall (TPR)        | `y * flagged`             | `y`                 |
| False positive rate | `(1 - y) * flagged`       | `1 - y`             |
| Specificity         | `(1 - y) * (1 - flagged)` | `1 - y`             |
| F1 score            | `2 * y * flagged`         | `flagged + y`       |

The F1 score, $2 \cdot \mathrm{TP} / (2 \cdot \mathrm{TP} + \mathrm{FP}
+ \mathrm{FN})$, is the last row because
$\mathrm{FP} + \mathrm{FN} = \sum_i (F_i + Y_i) - 2 \cdot \mathrm{TP}$.
A metric that is not a ratio of means, such as the area under the ROC
curve, is not covered by this pattern, though the weighted curves above
can still be integrated for a point estimate.

To summarize:

- **Weight the evaluation set, not just the training set.** A model
  trained with weights and evaluated without them still reports the
  holdout's precision and recall.
- **Calibrate to the score distribution.** The model can be run on the
  whole population, so the flag rate at every threshold is known
  exactly. Include score bands among the calibration covariates.
- **Distrust recall most.** Under covariate shift precision is
  partly protected, since $P(Y = 1 \mid x)$ is unchanged, but recall
  depends on where the positives the sample never saw are, and errs
  wherever labeling was concentrated.
- **Report the sensitivity.** The weights cannot see selection on
  the label. Say how large a $\Gamma$ the conclusion survives.

## Future work

Three gaps this guide works around rather than closes:

- **A variance that credits the calibration.** The intervals here use
  {class}`~pyrake.SIPWEstimator`'s variance, which treats the weights
  as fixed. The standard alternative for calibration estimators
  linearizes the ratio, $z_i = (a_i - \hat{R} b_i) / \bar{b}_w$ for
  numerator $a$ and denominator $b$, and computes the variance from the
  residuals of a weighted regression of $z$ on the calibration
  covariates (Deville, J.-C., and Särndal, C.-E., "Calibration
  Estimators in Survey Sampling," *Journal of the American Statistical
  Association*, 87(418), 1992). In 40 repeated draws of this
  example's holdout set (not shown), the current intervals covered
  about right for precision but reported roughly twice the actual
  standard error for recall at high thresholds, where the score bands
  pin down the flagged share. The residual variance should remove most
  of that gap.
- **Simultaneous bands for the whole curve.** The bands are pointwise.
  A bootstrap of the largest standardized deviation across thresholds,
  $\max_t |\hat{\theta}^*(t) - \hat{\theta}(t)| / \widehat{\mathrm{se}}^*(t)$,
  would give a band that covers the whole curve with the stated
  probability. {meth}`~pyrake.RatioEstimator.resample` resamples with
  the weights held fixed; a faithful bootstrap would re-solve the
  calibration in each replicate, which costs a {class}`~pyrake.Rake`
  solve per replicate.
- **Label shift with a known prevalence.** When labeling depends on
  the label, feature weights cannot correct it. If the population
  prevalence is known, or can be bounded, from another source, it is a
  calibration target like any other: $Y$ is observed for every labeled
  unit, so it can be a column of `X` with its population mean in `mu`.
  This guide does not yet show it, or check how well it works when
  selection depends on both the features and the label. Relatedly,
  {meth}`~pyrake.RatioEstimator.sensitivity_analysis` ignores
  `outcome_proxies`, so a known prevalence cannot yet narrow the
  sensitivity bounds the way it can for
  {class}`~pyrake.SIPWEstimator`.
