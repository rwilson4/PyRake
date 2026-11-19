# Calibrating Weights for a Population Mean

A survey's respondents rarely look like the population they are meant
to represent. Some kinds of people are easier to reach, or more willing
to answer, than others, and when those same traits predict the outcome
the plain sample mean is biased. Calibration weights the respondents so
that weighted sample means of known covariates match their population
values, and the weighted mean of the outcome then estimates the
population mean.

Many sets of weights achieve the same balance, so calibration chooses
among them by minimizing some penalty. This guide uses
{class}`~pyrake.Rake` to work through the two classical choices and the
continuum between them:

1. Weights that balance the covariates exactly while staying as close
   as possible to a set of **baseline weights**, such as inverse
   propensity scores.
2. Weights that balance the covariates exactly while minimizing **Kish's
   design effect**, the variance inflation the weighting costs.
3. The **efficient frontier** of weights that trade those two goals
   off, and its knee.

It ends by estimating the population mean with each set of weights.
Treatment effects, where two groups are calibrated together, use
{class}`~pyrake.JointCalibrator` instead; see the {doc}`usage`.

```{testsetup}
import matplotlib
matplotlib.use("Agg")
```

## The example survey

We simulate a population of 200,000 adults, so that we know the true
answer. A census publishes the population means of age, the urban
share, and income. Two more traits matter, but their population
shares are unknown: think of flags appended to the sample file from a
commercial database, available for every person invited to the survey
but for no one else. One, `engaged`, raises both the chance of
responding and the outcome. The other, `email`, says we hold an email
address for the person, which earns them a reminder and so raises
their chance of responding, but has nothing to do with the outcome.

```{doctest}
>>> import numpy as np
>>> import pyrake
>>> from scipy.special import expit
>>> rng = np.random.default_rng(0)
>>> N = 200_000
>>> age = rng.uniform(18, 80, N)
>>> urban = (rng.random(N) < 0.6).astype(float)
>>> income = np.exp(3.8 + 0.01 * (age - 45) + 0.3 * urban + rng.normal(0, 0.5, N))
>>> engaged = (rng.random(N) < 0.4).astype(float)
>>> email = (rng.random(N) < 0.5).astype(float)
>>> outcome = 3 + 0.03 * age + 0.5 * urban + 0.6 * engaged + rng.normal(0, 1, N)
```

We invite a simple random sample of 3,000 people. Younger, urban,
engaged, and emailed people are more likely to respond:

```{doctest}
>>> n = 3_000
>>> invited = rng.choice(N, size=n, replace=False)
>>> F = np.column_stack(
...     [np.ones(n), age[invited] / 10, urban[invited], engaged[invited], email[invited]]
... )
>>> responded = rng.random(n) < expit(F @ [0.8, -0.5, 0.6, 1.2, 1.5])
>>> sample = invited[responded]
>>> M = len(sample)
>>> print(f"{M} respondents")
1426 respondents
```

The sample is too young, which pulls its mean down, and too engaged,
which pushes it up. On balance it undershoots:

```{doctest}
>>> print(f"population {outcome.mean():.3f}, sample {outcome[sample].mean():.3f}")
population 5.009, sample 4.934
```

## Baseline weights

Because we know the covariates of everyone invited, respondents and
nonrespondents alike, we can model the probability of responding. A
logistic regression, fit here by maximum likelihood with SciPy, gives
each respondent an estimated probability of being in the sample,
$\hat{\pi}_i$: the probability of being invited, $n / N$, times the
fitted probability of responding.

```{doctest}
>>> from scipy.optimize import minimize
>>> def neg_log_likelihood(beta):
...     eta = F @ beta
...     return np.sum(np.logaddexp(0, eta)) - eta @ responded
>>> beta = minimize(neg_log_likelihood, np.zeros(5), method="BFGS").x
>>> pi_hat = (n / N) * expit(F[responded] @ beta)
```

PyRake normalizes weights so that a mean weight of 1 is an ordinary
weighted average (rather than the Horvitz-Thompson convention of
weights summing to $N$). The weight corresponding to a propensity
score $\pi$ is $(M / N) / \pi$, so the baseline weights are

```{doctest}
>>> v = (M / N) / pi_hat
>>> print(f"mean weight {v.mean():.3f}")
mean weight 1.001
>>> print(f"baseline estimate {np.average(outcome[sample], weights=v):.3f}")
baseline estimate 5.000
```

The baseline weights remove the bias, since the model includes
`engaged`. They also vary more than they need to, since they correct
for `email` too, which is harmless for bias but costs variance. And they
do not reproduce the census: the weighted means of
age, urban, and income miss their population values. Calibration
fixes that.

The covariates to balance exactly, and their population means, are

```{doctest}
>>> X = np.column_stack([age[sample], urban[sample], income[sample]])
>>> mu = np.array([age.mean(), urban.mean(), income.mean()])
```

## Exact balance, close to the baseline

The first approach solves

$$
\begin{array}{ll}
\mbox{minimize}   & D(w, v) \\
\mbox{subject to} & \frac{1}{M} X^T w = \mu \\
                  & \frac{1}{M} \mathbf{1}^T w = 1 \\
                  & w \geq M / N,
\end{array}
$$

choosing, among all weights that balance $X$ exactly, those closest to
the baseline. The constraint on the mean weight is {class}`~pyrake.Rake`'s
default (`constrain_mean_weight_to=1`). The lower bound keeps every
implied propensity score, $(M/N) / w_i$, at most 1.

With the KL divergence,
$D(w, v) = \frac{1}{M} \sum_i \left( w_i \log (w_i / v_i) - w_i + v_i \right)$,
this is raking (Deming and Stephan, 1940) starting from the baseline
weights, or entropy balancing (Hainmueller, 2012) when the baseline is
uniform. The KL divergence keeps the weights positive and measures
departures from the baseline proportionally: each term is $v_i$ times a
function of the ratio $w_i / v_i$, so halving a weight costs the same,
relative to its baseline, whether the baseline is large or small.

```{doctest}
>>> rake = pyrake.Rake(
...     distance=pyrake.KLDivergence(v=v), X=X, mu=mu, min_weight=M / N
... )
>>> result = rake.solve()
>>> print(result.status)
SolveStatus.OPTIMAL
>>> w_close = result.solution
>>> bool(np.allclose(X.T @ w_close / M, mu))
True
```

{class}`~pyrake.SquaredL2` and {class}`~pyrake.Huber` distances are
also available, each taking the baseline as `v`. Omitting `v` measures
distance from uniform weights.

## Exact balance, minimum design effect

Weighting costs precision. Kish's approximate design effect,

$$
\mathrm{deff} = \frac{M \sum_i w_i^2}{\left( \sum_i w_i \right)^2},
$$

is the factor by which unequal weights inflate the variance of a
weighted mean relative to a simple random sample of the same size, when
the outcome is unrelated to the weights. Divide $M$ by it to get the
effective sample size. With the mean weight fixed at 1, the denominator
is $M^2$, and the design effect is just the mean squared weight,
$\frac{1}{M} \| w \|_2^2$.

That is exactly the {class}`~pyrake.SquaredL2` distance,
$\frac{1}{M} \| w - v \|_2^2$, from baseline weights of zero. So the
weights that balance $X$ exactly with the smallest possible design
effect solve

```{doctest}
>>> rake = pyrake.Rake(
...     distance=pyrake.SquaredL2(v=np.zeros(M)), X=X, mu=mu, min_weight=M / N
... )
>>> w_deff = rake.solve().solution
>>> print(f"design effect: close to baseline {np.mean(w_close ** 2):.3f}, "
...       f"minimized {np.mean(w_deff ** 2):.3f}")
design effect: close to baseline 1.800, minimized 1.190
```

These are the stable balancing weights of Zubizarreta (2015), with
exact rather than approximate balance. Full citations are on the
{doc}`../references` page.

```{note}
With the mean weight constrained, as it is by default, any baseline
that is a linear combination of the columns of $X$ and a constant gives
the same weights as a baseline of zero. Expanding
$\frac{1}{M} \| w - v \|_2^2 = \frac{1}{M} \| w \|_2^2
- \frac{2}{M} v^T w + \frac{1}{M} \| v \|_2^2$,
the middle term is fixed by the balance constraints whenever $v$ lies
in that span. In particular, `SquaredL2()` with its default uniform
baseline also minimizes the design effect. A baseline only shapes
squared-$\ell_2$ weights through the information it carries beyond the
calibration covariates, which is precisely what an inverse propensity
weight built on `engaged` does here.
```

## Checking balance

{func}`~pyrake.plot_balance` plots, for each covariate and each set of
weights, the z-statistic of the weighted sample mean against the
population mean: the difference divided by $\sigma / \sqrt{M}$, where
$\sigma$ is the population standard deviation. The vertical lines at
$\pm 1.96$ mark differences a simple random sample would show only 5%
of the time.

The census also publishes the share of adults 65 and older, which we
did not calibrate to, so it shows what exact balance on other
covariates does and does not buy:

```{doctest}
>>> import matplotlib.pyplot as plt
>>> population = np.column_stack([age, urban, income, age >= 65])
>>> fig, ax = plt.subplots(figsize=(8, 4))
>>> balance, ax = pyrake.plot_balance(
...     X=population[sample],
...     mu=population.mean(axis=0),
...     sigma=population.std(axis=0),
...     covariates=["Age", "Urban", "Income", "Age 65+"],
...     weights={
...         "Unweighted": np.ones(M),
...         "Baseline": v,
...         "Close to baseline": w_close,
...         "Minimum design effect": w_deff,
...     },
...     title="Covariate Balance",
...     ax=ax,
... )
>>> legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
```

```{image} images/population_mean_balance.png
:alt: Z-statistics of four covariates under four weightings. Without weights, age, urban and the 65+ share are far out of balance; both calibrated weightings put age, urban and income exactly on zero, and only the minimum-design-effect weights leave the 65+ share outside the lines.
```

Without weights, the sample is far too young and too urban. The
baseline weights correct most of that, but not exactly. Both calibrated
weightings put the three calibration covariates exactly on zero (the
two overlap there).

The 65+ share separates them. The weights closest to the baseline keep
the baseline's good balance on it, while the minimum-design-effect
weights leave it significantly short. Matching the mean age does not
match the shape of the age distribution. The baseline, being an inverse
logistic function of age, up-weights older respondents increasingly
steeply, much as the response mechanism down-weights them; the
minimum-design-effect weights are, apart from the lower bound, linear
in the calibration covariates, so they cannot. If the 65+ share
matters, it belongs in `X`.

This is the bias side of the tradeoff the next section makes explicit,
caught on a covariate we happen to be able to check. The legend's
imbalance score summarizes each weighting; see
{func}`~pyrake.plot_balance` for its definition.

No balance plot can show `engaged`, whose population share we do not
know. There the two calibrated weightings differ too, unseen.

## The efficient frontier

An estimator's mean squared error is its squared bias plus its
variance, and the two weightings so far each attend to one term. The
baseline weights are our best guess at weights that remove bias, so
distance from them stands in for bias that calibration cannot check,
such as that from `engaged` or the 65+ share. The design effect stands
in for variance. We may well accept a small increase in bias for a
large reduction in variance. Between the two extremes lies a family of
weights solving

$$
\begin{array}{ll}
\mbox{minimize}   & D(w, v) \\
\mbox{subject to} & \frac{1}{M} X^T w = \mu, \quad \frac{1}{M} \mathbf{1}^T w = 1 \\
                  & \frac{1}{M} \| w \|_2^2 \leq \phi \\
                  & w \geq M / N
\end{array}
$$

for each design-effect budget $\phi$. As $\phi$ shrinks from the
design effect of the closest weights to the smallest achievable, the
distance from baseline grows. {class}`~pyrake.EfficientFrontier` traces
this Pareto frontier, solving the problem over a grid of $\phi$
values:

```{doctest}
>>> rake = pyrake.Rake(
...     distance=pyrake.KLDivergence(v=v), X=X, mu=mu, min_weight=M / N
... )
>>> frontier = pyrake.EfficientFrontier(rake)
>>> results = frontier.trace(num_points=50)
```

Its two ends are `results.corners[0]`, the weights closest to the
baseline (our `w_close`), and `results.corners[1]`, the weights with
the smallest design effect (our `w_deff`). Each point carries its
weights in `solution` and its coordinates, distance then design effect,
in `objectives`.

The *knee* is the point farthest from the straight line joining the two
corners, where the curve bends most sharply. It does not depend on how
either axis is scaled.

```{doctest}
>>> knee = results.knee()
>>> distance, deff = knee.objectives
>>> print(f"knee: distance {distance:.4f}, design effect {deff:.3f}")
knee: distance 0.0306, design effect 1.334
>>> w_knee = knee.solution
```

The plot marks the knee with dashed guides and its tangent. Here we
also label the two corners:

```{doctest}
>>> ax = results.plot(
...     y_label="Design Effect",
...     annotations={
...         "Closest to baseline": results.corners[0],
...         "Minimum design effect": results.corners[1],
...     },
... )
```

```{image} images/population_mean_frontier.png
:alt: The efficient frontier of design effect against KL distance from the baseline weights, falling steeply and then flattening, with the knee marked.
```

The frontier is steep near the baseline: moving a little away from it
cuts the design effect from 1.80 to 1.33, three quarters of the
reduction available. Past the knee, further variance reduction costs
ever more distance. The knee is where the trade stops being cheap: a
small increase in bias buys a large reduction in variance up to it,
and not beyond. Every weighting on the curve balances age, urban,
and income exactly; the curve trades off only what the weights do
beyond that.

To use a particular budget rather than the knee, pass it to
{class}`~pyrake.Rake` as `phi`. A $\phi$ below the smallest achievable
design effect (`results.corners[1].objectives[1]`) is infeasible, and
`solve()` raises `cvxium.ProblemCertifiablyInfeasibleError`.

## Estimating the mean

Each set of weights implies propensity scores, $(M / N) / w_i$, which
{class}`~pyrake.SIPWEstimator` turns into an estimate of the population
mean with a confidence interval:

```{doctest}
>>> weightings = {
...     "Baseline": v,
...     "Close to baseline": w_close,
...     "Knee": w_knee,
...     "Minimum design effect": w_deff,
... }
>>> for label, w in weightings.items():
...     estimator = pyrake.SIPWEstimator(
...         propensity_scores=(M / N) / w, outcomes=outcome[sample]
...     )
...     lo, hi = estimator.confidence_interval(alpha=0.05)
...     print(f"{label:<22} {estimator.point_estimate():.3f} ({lo:.3f}, {hi:.3f})")
Baseline               5.000 (4.920, 5.079)
Close to baseline      4.990 (4.910, 5.070)
Knee                   5.020 (4.948, 5.091)
Minimum design effect  5.068 (5.000, 5.136)
>>> print(f"population mean        {outcome.mean():.3f}")
population mean        5.009
```

Every calibrated weighting removes the sample's bias from age, but
they part ways on `engaged`. The weights closest to the baseline keep
the propensity model's correction for it, and estimate the mean about
as well as the baseline does. The minimum-design-effect weights discard
that correction along with the harmless one for `email`, and drift
upward toward an over-engaged sample, by nearly two standard errors.
The knee sits between: it gives up a small part of the correction for
an interval about 11% narrower.

That narrowing is less than Kish's design effect suggests, since
$\sqrt{1.334 / 1.800} \approx 0.86$: the design effect describes an
outcome unrelated to the weights, and here part of the weights'
variation tracks `engaged`, which drives the outcome. That is
expected, and it is no reason to choose weights by interval width
instead. The design effect, like the balance constraints and the
distance from the baseline, depends only on the covariates, so the
whole frontier, and the choice of a point on it, can be settled before
anyone looks at an outcome. That is a legitimate design-based choice.
Picking the weights that make a particular outcome's interval
narrowest uses the outcome to choose the analysis, which is p-hacking.
(The table above uses the outcome only because this is a simulation,
to show the consequences of a choice already made.)

Bear in mind, too, that this is one simulated survey: the differences
between the estimates are of the order of their standard errors, so
they illustrate the mechanism rather than measure it.

The frontier makes the tradeoff explicit, but cannot settle it: its
horizontal axis measures distance from the baseline, not bias. How far
to move depends on how much you trust the propensity model relative to
the calibration covariates. A few guidelines:

- **Calibrate to what you know.** Every covariate with a known
  population mean that predicts the outcome or response belongs in
  `X` (or, for approximate balance within a tolerance $\psi$, in `Z`
  with `nu` and `psi`). Balance on it is then guaranteed at every point
  on the frontier, so it is no longer at stake in the tradeoff.
- **The knee is a default, not a verdict.** It is a reasonable choice
  when the baseline and the calibration covariates are about equally
  credible. If the baseline encodes information you trust, stay closer
  to it; if it is little more than a guess, move toward the minimum
  design effect.
- **Choose before you look.** Settle the covariates, the baseline,
  and the point on the frontier using the covariates alone, then
  estimate. The same weights then serve every outcome in the survey.
- **Keep the lower bound.** `min_weight=M / N` rules out implied
  propensity scores above 1, which no respondent can have, and
  {class}`~pyrake.SIPWEstimator` requires propensity scores in (0, 1].
