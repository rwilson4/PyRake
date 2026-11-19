# Calibrating Weights for a Treatment Effect

Randomizing a treatment makes the treated and control groups
comparable in the population. Measuring the outcome with a survey can
undo that. Only some of the people invited respond, the respondents
differ from the population, and if the treatment changes who responds,
the respondents in the two groups differ from each other as well.
Weighting has to repair two things:

- **Internal validity**: the weighted treated and control respondents
  should look like each other, so their difference measures the effect
  of the treatment rather than a difference between the people who
  answered.
- **External validity**: each weighted group should look like the
  target population, so the effect measured is the population's,
  rather than the respondents'.

{class}`~pyrake.JointCalibrator` calibrates the two groups' weights
together, holding both. This guide uses it on a simulated brand-lift
study:

1. **Joint calibration** for internal and external validity, and what
   internal validity alone gets you.
2. The **external validity-variance frontier**, which prices external
   balance in effective sample size, and its knee.
3. **Estimating the treatment effect** with
   {class}`~pyrake.SimpleDifferenceEstimator`, with a confidence
   interval, a p-value, and a sensitivity analysis.

It builds on the {doc}`population_mean` guide, which introduces
calibration for a single sample.

```{testsetup}
import matplotlib
matplotlib.use("Agg")
```

## The example study

A platform with 400,000 users runs an advertising campaign. Half the
users, chosen at random, are eligible to see it, and the other half are
held out. Afterwards the platform surveys 4,000 users from each group
and measures how favorably they view the brand. The platform knows
every user's age, whether they live in a city, whether they are a
heavy user of the platform, and their gender, so it knows the
population means of all four.

We simulate the population, so that we know the true answer. The
favorability score depends on all four traits, and the campaign raises
it more for younger and for heavy users:

```{doctest}
>>> import numpy as np
>>> import pyrake
>>> from scipy.special import expit
>>> rng = np.random.default_rng(2)
>>> N = 400_000
>>> age = rng.uniform(18, 80, N)
>>> urban = (rng.random(N) < 0.6).astype(float)
>>> heavy = (rng.random(N) < expit(1.0 - 0.06 * (age - 30))).astype(float)
>>> female = (rng.random(N) < 0.5).astype(float)
>>> baseline = 3 + 0.02 * age + 0.4 * urban + 1.0 * heavy + 0.2 * female
>>> effect = 0.5 - 0.01 * (age - 45) + 0.3 * heavy
>>> y0 = baseline + rng.normal(0, 1, N)
>>> y1 = y0 + effect
>>> treated = rng.random(N) < 0.5
>>> print(f"population average treatment effect {effect.mean():.3f}")
population average treatment effect 0.602
```

Here `y0` and `y1` are each user's potential outcomes, without and with
the campaign, and the estimand is the population average of their
difference.

Younger and urban users are more likely to respond in both groups. In
the treated group, heavy users are also more likely to respond: they
saw the campaign most, and recognize the brand when the survey
arrives.

```{doctest}
>>> def survey(group, lift):
...     invited = rng.choice(np.flatnonzero(group), size=4_000, replace=False)
...     eta = -0.3 - 0.04 * (age[invited] - 45) + 0.5 * urban[invited]
...     responded = rng.random(4_000) < expit(eta + lift * heavy[invited])
...     return invited[responded]
>>> control = survey(~treated, lift=0.0)
>>> treatment = survey(treated, lift=1.5)
>>> M1, M2 = len(control), len(treatment)
>>> N1, N2 = int((~treated).sum()), int(treated.sum())
>>> print(f"{M1} control and {M2} treated respondents")
1872 control and 2422 treated respondents
```

The plain difference in means overstates the effect, for two reasons:

```{doctest}
>>> y_control, y_treated = y0[control], y1[treatment]
>>> respondents = np.concatenate([control, treatment])
>>> print(f"difference in means {y_treated.mean() - y_control.mean():.3f}, "
...       f"average effect among respondents {effect[respondents].mean():.3f}")
difference in means 0.767, average effect among respondents 0.699
```

The gap between 0.767 and 0.699 is a failure of internal validity: the
treated respondents include more heavy users than the control
respondents, and heavy users score higher with or without the campaign.
The gap between 0.699 and the population's 0.602 is a failure of
external validity: the respondents are younger and heavier users than
the population, and the campaign works better on them.

## Joint calibration

PyRake indexes the control group by 1 and the treated group by 2. With
$Z_1$ and $Z_2$ the two groups' covariates, $w_1$ and $w_2$ their
weights, and $\nu$ the population means, {class}`~pyrake.JointCalibrator`
solves

$$
\begin{array}{ll}
\mbox{minimize}   & D([w_1; w_2], [v_1; v_2]) \\
\mbox{subject to} & \| \frac{1}{M_1} Z_1^T w_1 - \nu \|_\infty \leq \psi_1 \\
                  & \| \frac{1}{M_2} Z_2^T w_2 - \nu \|_\infty \leq \psi_2 \\
                  & \| \frac{1}{M_2} Z_2^T w_2 - \frac{1}{M_1} Z_1^T w_1 \|_\infty \leq \psi_3 \\
                  & \frac{1}{M_1} \mathbf{1}^T w_1 = 1, \quad \frac{1}{M_2} \mathbf{1}^T w_2 = 1 \\
                  & w_1 \geq M_1 / N_1, \quad w_2 \geq M_2 / N_2.
\end{array}
$$

The first two constraints are external validity, each group against
the population, and the third is internal validity, one group against
the other. Each group has its own mean weight of 1, so each weighted
mean is an ordinary weighted average. The lower bounds play the role
they play for {class}`~pyrake.Rake`: $N_b$ is the size of group $b$ in
the population, so $(M_b / N_b) / w_i$ is the implied probability that
a member of that group responded, and must not exceed 1. The argument
names follow the constraints: `Z1`, `nu1`, and `psi1` for the first,
`Z2`, `nu2`, and `psi2` for the second, and `Z3`, `Z4`, and `psi3` for
the third, which can balance different covariates than the external
constraints do, though here they are the same.

Because the campaign was randomized, each group is a random half of
the population, so the same population means serve as the target for
both. Standardizing the covariates by their population standard
deviations makes each band a bound on standardized mean differences:

```{doctest}
>>> X = np.column_stack([age, urban, heavy, female])
>>> mu, sigma = X.mean(axis=0), X.std(axis=0)
>>> Z1, Z2, nu = X[control] / sigma, X[treatment] / sigma, mu / sigma
```

Left unset, the bands default to Cochran's $\psi_b = 1.5 / \sqrt{M_b}$
for external validity, and its two-sample analogue
$\psi_3 = 1.5 \sqrt{1 / M_1 + 1 / M_2}$ for internal validity. A
weighted mean that close to its target is as close as a simple random
sample of the same size would usually be.

```{doctest}
>>> calibrator = pyrake.JointCalibrator(
...     distance=pyrake.KLDivergence(),
...     Z1=Z1, nu1=nu,
...     Z2=Z2, nu2=nu,
...     Z3=Z1, Z4=Z2,
...     min_weight1=M1 / N1,
...     min_weight2=M2 / N2,
... )
>>> joint = calibrator.solve_with_fallback()
>>> print(joint.fallback_message)
Full solution achieved.
>>> print(f"effective sample sizes: control {joint.control_effective_sample_size:.0f}, "
...       f"treated {joint.treatment_effective_sample_size:.0f}")
effective sample sizes: control 1689, treated 2086
```

The KL divergence from uniform baseline weights makes this a two-group
entropy balancing. As with {class}`~pyrake.Rake`, a baseline from a
response model can be passed as `v`, one vector for both groups,
control first.

Use {meth}`~pyrake.JointCalibrator.solve_with_fallback` rather than
`solve()`. When the constraints cannot all be met, it relaxes them in
order of importance and says so, rather than raising. Its
`fallback_status` is 0 when everything was achieved, as here. If the
external bands cannot be met it relaxes them first (status 2), holding
internal validity, and only if internal validity itself is out of reach
does it relax that (status 3). Internal validity comes first because
without it the estimate is not an effect of the treatment for any
population at all. The `solution` is the two groups' weights end to
end, control first.

For contrast, here are weights that ask for internal validity alone,
leaving out `Z1` and `Z2`:

```{doctest}
>>> internal = pyrake.JointCalibrator(
...     distance=pyrake.KLDivergence(),
...     Z3=Z1, Z4=Z2,
...     min_weight1=M1 / N1,
...     min_weight2=M2 / N2,
... ).solve_with_fallback()
```

## Checking balance

{func}`~pyrake.plot_balance_2_sample` checks internal validity. For
each covariate it plots the z-statistic of the difference between the
two groups' weighted means, scaled by
$\sigma \sqrt{1 / M_1 + 1 / M_2}$, the standard error of that
difference in two simple random samples:

```{doctest}
>>> import matplotlib.pyplot as plt
>>> covariates = ["Age", "Urban", "Heavy user", "Female"]
>>> weights = {
...     "Unweighted": np.ones(M1 + M2),
...     "Internal validity only": internal.solution,
...     "Joint": joint.solution,
... }
>>> fig, ax = plt.subplots(figsize=(8, 4))
>>> balance, ax = pyrake.plot_balance_2_sample(
...     X1=X[control],
...     X2=X[treatment],
...     covariates=covariates,
...     sigma1=sigma,
...     sigma2=sigma,
...     weights1={label: w[:M1] for label, w in weights.items()},
...     weights2={label: w[M1:] for label, w in weights.items()},
...     title="Internal Validity: Treated vs Control",
...     ax=ax,
... )
>>> legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
```

```{image} images/treatment_effect_internal.png
:alt: Z-statistics of the treated-minus-control difference for four covariates under three weightings. Unweighted, the treated respondents have significantly more heavy users and fewer urban ones; both calibrated weightings bring every difference inside the lines.
```

Unweighted, the treated respondents have significantly more heavy users
than the control respondents, and fewer urban ones. Both calibrated
weightings repair that.
{func}`~pyrake.plot_balance` checks external validity, one group at a
time:

```{doctest}
>>> fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
>>> for ax, rows, label, block in [
...     (axes[0], control, "Control", slice(None, M1)),
...     (axes[1], treatment, "Treated", slice(M1, None)),
... ]:
...     balance, ax = pyrake.plot_balance(
...         X=X[rows],
...         mu=mu,
...         sigma=sigma,
...         covariates=covariates,
...         weights={name: w[block] for name, w in weights.items()},
...         title=f"External Validity: {label}",
...         ax=ax,
...     )
>>> for ax in axes:
...     legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15))
```

```{image} images/treatment_effect_external.png
:alt: Z-statistics of each group's weighted covariate means against the population. Unweighted and with internal validity only, both groups are far too young and have far too many heavy and urban users; the joint weights bring every covariate in both groups inside the lines.
```

The weights for internal validity alone make the two groups look alike,
but not like the population: both stay far too young and too heavy in
their use of the platform. They even move the control group further
from the population on heavy use, since nothing stops them meeting the
treated group partway. The joint weights fix both.

## The external validity-variance frontier

External balance costs precision. Pooling the two groups, Kish's
effective sample size $(\sum_i w_i)^2 / \sum_i w_i^2$ is

```{doctest}
>>> print(f"joint {joint.combined_effective_sample_size:.0f}, "
...       f"internal validity only {internal.combined_effective_sample_size:.0f}")
joint 3773, internal validity only 4279
```

{class}`~pyrake.ExternalValidityVarianceFrontier` makes the price
explicit. Holding internal validity at $\psi_3$, it minimizes the
mean squared weight, $\frac{1}{M} (\|w_1\|_2^2 + \|w_2\|_2^2)$ with
$M = M_1 + M_2$, subject to external bands
$\psi_b = \phi / \sqrt{M_b}$, for a grid of scales $\phi$. A single
$\phi$ makes both groups equally stringent relative to their size, and
Cochran's band is $\phi = 1.5$.

```{doctest}
>>> frontier = pyrake.ExternalValidityVarianceFrontier.from_calibrator(calibrator)
>>> results = frontier.trace()
>>> loose, tight = results.corners
>>> print(f"no external balance: scale {loose.objectives[1]:.2f}, "
...       f"effective sample size {(M1 + M2) / loose.objectives[0]:.0f}")
no external balance: scale 15.79, effective sample size 4279
>>> print(f"tightest balance:    scale {tight.objectives[1]:.2f}, "
...       f"effective sample size {(M1 + M2) / tight.objectives[0]:.0f}")
tightest balance:    scale 0.00, effective sample size 3671
```

Each point's `objectives` are its variance inflation factor (the mean
squared weight, Kish's design effect) and its scale: the largest
external imbalance, in standard deviations, times $\sqrt{M_b}$. Because
each group's mean weight is 1, the effective sample size is $M$
divided by the first. The frontier uses the
{class}`~pyrake.SquaredL2` distance from zero, so each point is the
largest effective sample size achievable at its band; the joint
calibration minimized the KL divergence instead, so it can sit inside
the frontier, though here only barely.

The plot marks the knee, the point farthest from the chord between the
two corners. {meth}`~pyrake.ExternalValidityVarianceFrontier.evaluate_objectives`
places any other weights in the same coordinates:

```{doctest}
>>> ax = results.plot(
...     annotations={
...         "Cochran's band (joint)": frontier.evaluate_objectives(joint.solution),
...         "Internal validity only": frontier.evaluate_objectives(internal.solution),
...     },
... )
```

```{image} images/treatment_effect_frontier.png
:alt: The external validity-variance frontier, with covariate imbalance falling from about 16 to zero as the effective sample size falls from about 4,280 to about 3,670. The knee is marked near a scale of 7, and Cochran's band of 1.5 sits near the tight end.
```

The frontier runs from about 4,280 effective respondents with no
external balance to about 3,670 with exact balance. The first half of
the imbalance comes off cheaply; past the knee, each further unit costs
more sample.

{func}`~pyrake.solve_with_knee_fallback` turns that into a rule. It
traces a coarse frontier and calibrates at the looser of Cochran's
band and the knee, so it never pays past the knee for balance. It
exists for surveys where Cochran's band sits far out on the flat end of
the frontier; its documentation describes one where meeting the
tightest attainable band left an effective sample of 164, against 448
at the knee. Here it picks the knee:

```{doctest}
>>> knee = pyrake.solve_with_knee_fallback(calibrator)
>>> print(knee.fallback_status)
4
>>> print(knee.fallback_message)  # doctest: +NORMALIZE_WHITESPACE
External validity was relaxed on purpose: the conventional band of 1.5 was
attainable, but the chosen band sits at 7.28 (at worst 0.17 standard
deviations in either group) and tightening past it costs sample size faster
than it buys balance. Internal validity and the design effect were achieved
at that band.
>>> print(f"effective sample size {knee.combined_effective_sample_size:.0f}")
effective sample size 4111
```

Status 4 says the external band was loosened by choice, not because
Cochran's was out of reach. Only the band comes from the frontier: the
final solve still minimizes the calibrator's KL divergence, and still
holds internal validity.

## Estimating the treatment effect

The campaign was randomized in the population, so the population mean
of `y0` is the mean of the control group's population, and that of
`y1` is the mean of the treated group's. Each weighted group mean
estimates its own group's population mean, and the effect is their
difference. {class}`~pyrake.SimpleDifferenceEstimator` does exactly
that, given each group's implied response probabilities,
$(M_b / N_b) / w_i$:

```{doctest}
>>> def estimator(w):
...     return pyrake.SimpleDifferenceEstimator(
...         control_outcomes=y_control,
...         treated_outcomes=y_treated,
...         control_sampling_propensity_scores=(M1 / N1) / w[:M1],
...         treated_sampling_propensity_scores=(M2 / N2) / w[M1:],
...     )
>>> ate = estimator(joint.solution)
>>> lo, hi = ate.confidence_interval(alpha=0.05)
>>> print(f"effect {ate.point_estimate():.3f}, 95% interval ({lo:.3f}, {hi:.3f}), "
...       f"p-value {ate.pvalue():.1e}")
effect 0.631, 95% interval (0.559, 0.704), p-value 2.2e-65
```

Its variance is the sum of the two groups' variances, each that of a
stabilized inverse propensity weighted mean
({class}`~pyrake.SIPWEstimator`). Without sampling propensity scores it
is the plain difference in means. When treatment was not randomized,
{class}`~pyrake.ATEEstimator` and its relatives take treatment
propensity scores as well; see the {doc}`usage`.

The interval accounts for sampling error, not for error in the
weights. If calibrating on these four covariates leaves a hidden
difference between respondents and nonrespondents, the response
probabilities are wrong. The sensitivity analysis of Zhao, Small, and
Bhattacharya (2019) bounds the estimate when each implied response
probability may be off by a factor of up to $\Gamma$ in odds:

```{doctest}
>>> low, high = ate.sensitivity_analysis(gamma=1.5)
>>> print(f"Gamma = 1.5: effect between {low:.3f} and {high:.3f}")
Gamma = 1.5: effect between 0.266 and 0.998
```

Even then the campaign raised favorability.
{meth}`~pyrake.SimpleDifferenceEstimator.plot_sensitivity` traces
these bounds, with a confidence interval around them, over a range of
$\Gamma$. Full citations are on the {doc}`../references` page.

Now each set of weights. Because this is a simulation, we can also say
what each weighting is aiming at. Given who responded, the expected
value of each group's weighted mean replaces each outcome by its
expectation: `baseline` for a control respondent, and
`baseline + effect` for a treated one. That expectation, the target
below, moves only with the weights, not with the noise in the
outcomes, so it isolates the bias each weighting leaves:

```{doctest}
>>> weights["Knee"] = knee.solution
>>> for label, w in weights.items():
...     est = estimator(w)
...     lo, hi = est.confidence_interval(alpha=0.05)
...     target = (np.average(baseline[treatment] + effect[treatment], weights=w[M1:])
...               - np.average(baseline[control], weights=w[:M1]))
...     print(f"{label:<23} {est.point_estimate():.3f} ({lo:.3f}, {hi:.3f})"
...           f"  target {target:.3f}")
Unweighted              0.767 (0.698, 0.835)  target 0.774
Internal validity only  0.725 (0.656, 0.793)  target 0.732
Joint                   0.631 (0.559, 0.704)  target 0.624
Knee                    0.675 (0.605, 0.745)  target 0.676
>>> print(f"population effect       {effect.mean():.3f}")
population effect       0.602
```

Internal validity alone removes the gap between the groups but leaves
the estimate near the respondents' effect, far from the population's.
Only external validity moves it there. The joint weights' target,
0.624, is not exactly 0.602, since Cochran's band allows each
covariate an imbalance a simple random sample might show; exact
balance would remove the rest, at the price the frontier shows.

The knee did not pay here. It bought about 9% more effective sample
(4,111 against 3,773) and an interval 3% narrower, for a bias of 0.07,
two standard errors. In mean squared error that is more than three
times worse than Cochran's band. The frontier measures what balance costs, not what imbalance
does: the bias from an imbalance is the imbalance times how strongly
that covariate modifies the outcomes, and here age and heavy use modify
the effect itself. A covariate unrelated to the outcomes could be left
unbalanced for free. The frontier's horizontal axis cannot tell the two
apart.

This is not luck of the draw. In the other simulated surveys of this
population we tried, seeds 1, 3, and 4, the joint target was within
0.025 of the population effect every time, and the knee's 0.07 to 0.08
above it.

A few guidelines:

- **Internal validity first.** It is what makes the estimate an
  effect. {meth}`~pyrake.JointCalibrator.solve_with_fallback` relaxes
  external validity before internal validity, and its status says
  which it had to give up.
- **Read the frontier before taking the knee.** The knee is a good
  default when Cochran's band would cost much of the sample, which is
  the trap {func}`~pyrake.solve_with_knee_fallback` exists to avoid.
  When the band is cheap, as here, and the covariates plausibly modify
  the effect, the knee's extra precision can cost more in bias than it
  saves in variance.
- **Choose before you look.** The frontier, the band, and the weights
  depend only on the covariates, so settle them before estimating any
  effect, and use the same weights for every outcome in the study.
  Choosing the band that makes a particular effect's interval narrowest
  uses the outcome to choose the analysis. (The table above uses the
  outcomes only because this is a simulation.)
- **Balance the effect modifiers.** Covariates that predict how much
  the treatment helps, not just the outcome, matter most for external
  validity. If one matters but its population mean is unknown, no
  weighting can check it.
