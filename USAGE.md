# PyRake User Guide

This guide teaches an AI agent how to use PyRake to analyze surveys.
PyRake is a survey calibration and estimation library for causal
inference and observational studies. It solves constrained convex
optimization problems to find balancing weights, then uses those
weights for estimation and inference.

**For AI agents:** This guide provides recipes and conceptual
orientation, but every class and method in this library has a detailed
NumPy-style docstring documenting its parameters, return values, and
usage notes. Before calling any function, read its docstring via
`help()`:

```python
from pyrake import (
    JointCalibrator,
    SIPWEstimator,
)

help(JointCalibrator)
help(JointCalibrator.solve_with_fallback)
help(SIPWEstimator)
```

The docstrings are the authoritative reference for parameter names,
types, defaults, and edge cases.

The two main workflows are:

1. **Calibration**: Find weights that balance covariates between a
   sample and a target population (or between treatment and control
   groups).
2. **Estimation**: Use the calibrated weights (expressed as propensity
   scores) to estimate population means, treatment effects, ratios,
   and linear combinations, with proper variance estimation and
   sensitivity analysis.

All classes are accessible from the top-level `pyrake` namespace:

```python
import pyrake
```

## Calibration

### Single-Group Calibration with `Rake`

Use `Rake` when calibrating a single set of weights (e.g., one survey
sample to a known population).

```python
rake = pyrake.Rake(
    distance=pyrake.KLDivergence(),
    X=X,          # shape (M, p): covariates to balance exactly
    mu=mu,        # shape (p,): population means of X
    Z=Z,          # shape (M, q): covariates to balance approximately (optional)
    nu=nu,        # shape (q,): population means of Z (required if Z given)
    psi=psi,      # float or array: max imbalance for Z (optional; default 1.5/sqrt(M))
    phi=phi,      # float: upper bound on mean squared weight (optional)
    min_weight=M / N,  # lower bound on weights (scalar or array)
)
result = rake.solve()
weights = result.solution  # shape (M,)
```

**Key arguments:**

- `distance`: The objective function to minimize. Use
  `pyrake.KLDivergence()` (recommended for survey calibration),
  `pyrake.SquaredL2()`, or `pyrake.Huber()`. The distance is measured
  from baseline weights (default: all ones). Pass
  `pyrake.KLDivergence(v=baseline_weights)` to use custom baselines.
- `X`, `mu`: Covariates balanced *exactly*. Use sparingly — exact
  balance for many covariates inflates variance.
- `Z`, `nu`: Covariates balanced *approximately*, via an l-infinity
  constraint. This is the workhorse for most covariate balancing.
- `psi`: Controls the tightness of approximate balance. The default
  `1.5 / sqrt(M)` follows Cochran and Chambers (1965, section 3.1).
  If `Z` is whitened (divided by population standard deviations),
  then `psi = 1.96 / sqrt(M)` would constrain imbalances to be
  non-significant at alpha = 0.05.
- `phi`: Upper bound on the mean squared weight, `(1/M) * ||w||_2^2`.
  Controls variance inflation. When set deliberately infeasible
  (e.g., `phi=0.99`), the solver minimizes the norm, yielding a
  minimum-variance solution.
- `min_weight`: Lower bound on each weight. Setting this to
  `M / N` (sample size divided by population size) prevents
  implied propensity scores greater than 1.
- `constrain_mean_weight_to`: Constrains the average weight.
  Defaults to 1.0. Set to `None` to remove this constraint.

**Standardizing covariates:**

Always standardize `Z` and `nu` by dividing by population standard
deviations before passing them to `Rake` or `JointCalibrator`. This
puts all covariates on a common scale, making `psi` interpretable as
a z-statistic bound.

```python
MIN_SIGMA = 1e-6
sigma = np.clip(population_std, MIN_SIGMA, None)
Z_standardized = Z / sigma
nu_standardized = nu / sigma
```

### Joint Calibration with `JointCalibrator`

When analyzing treatment effects, jointly calibrate treatment and
control group weights using `JointCalibrator`. This achieves both
*internal validity* (balance between groups) and *external validity*
(balance with the target population).

```python
rake = pyrake.JointCalibrator(
    distance=pyrake.KLDivergence(),
    # External validity: each group vs population (approximate balance)
    Z1=Z_control_standardized,   # shape (M1, q): control covariates
    Z2=Z_treated_standardized,   # shape (M2, q): treated covariates
    nu1=nu_control / sigma,      # population means for control covariates
    nu2=nu_treated / sigma,      # population means for treated covariates
    # Internal validity: control vs treated (approximate balance)
    Z3=Z_control_standardized,   # same covariates, for between-group comparison
    Z4=Z_treated_standardized,
    # Optional: treated-only exact balance covariates
    X2=X_treated_only,           # shape (M2, p2)
    mu2=mu_treated_only,         # shape (p2,)
    # Bounds
    min_weight1=M1 / N,          # control lower bound
    min_weight2=M2 / N,          # treated lower bound
    # Variance control
    phi=0.99,                    # deliberately infeasible → minimize norm
    settings=cvxium.OptimizationSettings(
        outer=cvxium.OuterSettings(tolerance_soft=0.1),
    ),
)
```

**Constraint types:**

| Constraints   | Purpose                               | Arguments         |
|---------------|---------------------------------------|-------------------|
| X1, mu1       | Exact balance, control vs population  | External validity |
| X2, mu2       | Exact balance, treated vs population  | External validity |
| X3, X4        | Exact balance, control vs treated     | Internal validity |
| Z1, nu1, psi1 | Approx balance, control vs population | External validity |
| Z2, nu2, psi2 | Approx balance, treated vs population | External validity |
| Z3, Z4, psi3  | Approx balance, control vs treated    | Internal validity |

All constraint groups are optional. Specify only the ones needed. At
minimum, the mean-weight constraint (default: average weight = 1 for
each group) provides the required equality constraints.

**Default `psi` values** are set automatically:

- External validity: `psi1 = 1.5 / sqrt(M1)`, `psi2 = 1.5 / sqrt(M2)`
  (one-sample z-test scale).
- Internal validity: `psi3 = 1.5 * sqrt(1/M1 + 1/M2)` (two-sample
  z-test scale).

#### Solving with Fallback

Always use `solve_with_fallback()` instead of `solve()` for
`JointCalibrator`. It implements a graceful degradation cascade when
the full problem is infeasible:

```python
result = rake.solve_with_fallback()
```

The result is a `FallbackOptimizationResult` with:

- `result.solution`: The weight vector, shape `(M1 + M2,)`. The
  first `M1` entries are control weights; the remaining `M2` are
  treated weights.
- `result.fallback_status`: Integer code indicating what was achieved.
- `result.fallback_message`: Human-readable description.

**Fallback levels** (in order of preference):

| Status   | Description                                                                                        |
|----------|----------------------------------------------------------------------------------------------------|
| 0        | Full solution: all constraints satisfied                                                           |
| 1        | Phi relaxed: covariate balance achieved, but design effect (phi) minimized rather than constrained |
| 2        | Ambivalent: internal validity achieved, external validity maximized best-effort                    |
| 3        | Sad: internal validity was infeasible; internal imbalance minimized best-effort                    |
| (raises) | Fatal: basic bounds + equality constraints are infeasible                                          |

Internal validity should *always* be prioritized over external validity.

#### Extracting Propensity Scores from Weights

After solving, convert weights to propensity scores for use with
estimators:

```python
w = result.solution
w_control = w[:M1]         # control weights
w_treated = w[M1:]         # treated weights

# Propensity scores: pi = (M / N) / w
control_propensities = (M1 / N) / w_control
treated_propensities = (M2 / N) / w_treated
```

### Assessing Covariate Balance

Use `plot_balance` for external validity (one-sample: group vs
population) and `plot_balance_2_sample` for internal validity
(two-sample: control vs treated).

```python
# External validity: control group vs population
pyrake.plot_balance(
    X=df_control[covariates].values.astype(float),
    mu=population_means.values,
    covariates=covariate_names,
    sigma=population_stds.values,
    weights={
        "Unadjusted": np.ones(M1),
        "Calibrated": w_control,
    },
    title="External Validity: Control",
)

# Internal validity: control vs treated
pyrake.plot_balance_2_sample(
    X1=df_control[covariates].values.astype(float),
    X2=df_treated[covariates].values.astype(float),
    covariates=covariate_names,
    sigma1=population_stds.values,
    sigma2=population_stds.values,
    weights1={
        "Unadjusted": np.ones(M1),
        "Calibrated": w_control,
    },
    weights2={
        "Unadjusted": np.ones(M2),
        "Calibrated": w_treated,
    },
    title="Internal Validity: Control vs Treated",
)
```

The `weights` argument is a dictionary mapping strategy names to
weight arrays, allowing visual comparison of different weighting
strategies.

## Estimation

### Common Estimator Interface

All estimators share these methods:

```python
estimator.point_estimate()          # float
estimator.variance()                # float
estimator.confidence_interval(alpha=0.10)  # (lb, ub)
estimator.pvalue(null_value=0.0)    # float
estimator.sensitivity_analysis(gamma=6.0)  # (lb, ub)
estimator.expanded_confidence_interval(alpha=0.10, gamma=6.0)  # (lb, ub)
estimator.plot_sensitivity(gamma_lower=1.0, gamma_upper=6.0)
```

### Estimating Population Means with `SIPWEstimator`

`SIPWEstimator` (Stabilized Inverse Propensity Weighted) estimates
the weighted mean `sum(w_i * y_i) / sum(w_i)`. It does not require
knowing the population size.

```python
estimator = pyrake.SIPWEstimator(
    propensity_scores=propensities,  # shape (M,)
    outcomes=Y,                      # shape (M,)
)

pe = estimator.point_estimate()
ci = estimator.confidence_interval(alpha=0.05)
```

By default, `SIPWEstimator` uses the `PopulationMean` estimand,
which sets weights to `1 / propensity_scores`. This is the standard
choice for survey calibration: we want to estimate the population
mean, and the propensity scores describe how likely each person is
to appear in the sample.

For binary outcomes (e.g., "yes/no" survey responses), the default
`binary_outcomes=True` enables a faster sensitivity analysis
algorithm. Set `binary_outcomes=False` for continuous outcomes.

### Estimating Means within Sub-Populations with `RatioEstimator`

When analyzing a metric within a sub-population that is only
observable in the sample (e.g., people who like chocolate), use
`RatioEstimator`. We cannot calibrate weights specifically for
chocolate-lovers because we do not know who in the population
loves chocolate. Instead, we calibrate weights for the full sample
and estimate the sub-population mean as a ratio:

```
E[C * Y] / E[C]
```

where `C` is a binary indicator for the sub-population (1 if the
person likes chocolate, 0 otherwise), and `Y` is the outcome.

```python
ratio_estimator = pyrake.RatioEstimator(
    propensity_scores=propensities,
    numerator_outcomes=C * Y,    # C_i * Y_i for each respondent
    denominator_outcomes=C,      # C_i for each respondent
)

pe = ratio_estimator.point_estimate()   # estimates E[C*Y] / E[C]
ci = ratio_estimator.confidence_interval()
```

Variance is computed via the delta method, properly accounting for
correlation between numerator and denominator.

### Treatment Effects in a Properly-Randomized A/B Test

In a properly-randomized A/B test with survey-based measurement,
the causal effect is the difference between two population means
corresponding to the two potential outcomes, Y(0) and Y(1).
By randomization, the population mean of `Y(1)` equals the mean
in the treated group, and the population mean of `Y(0)` equals the
mean in the control group. So we estimate each group's population
mean via `SIPWEstimator` (adjusting for survey non-response), and
the treatment effect is their difference.

Use `SimpleDifferenceEstimator`:

```python
uplift = pyrake.SimpleDifferenceEstimator(
    control_outcomes=Y_control,
    treated_outcomes=Y_treated,
    control_sampling_propensity_scores=control_propensities,
    treated_sampling_propensity_scores=treated_propensities,
)

pe = uplift.point_estimate()       # treated mean - control mean
ci = uplift.confidence_interval()
pv = uplift.pvalue(null_value=0.0)
```

The `sampling_propensity_scores` adjust for survey non-response
(generalizing from respondents to the population). When omitted,
the estimator reduces to a simple difference in sample means.

**Important:** `SimpleDifferenceEstimator` is appropriate *only*
when treatment assignment is random (e.g., a proper A/B test). It
does not adjust for treatment selection bias.

### Treatment Effects Outside a Randomized Experiment

Outside a randomized experiment, we need *treatment propensity
scores* (how likely someone is to select treatment) in addition
to *response propensity scores* (how likely someone is to respond
to the survey). Use `ATEEstimator`, `ATTEstimator`, or
`ATCEstimator`:

```python
# Average Treatment Effect (ATE)
ate = pyrake.ATEEstimator(
    control_propensity_scores=treatment_propensities_control,
    control_outcomes=Y_control,
    treated_propensity_scores=treatment_propensities_treated,
    treated_outcomes=Y_treated,
    # Optional: also adjust for survey non-response
    control_sampling_propensity_scores=response_propensities_control,
    treated_sampling_propensity_scores=response_propensities_treated,
)
```

When both treatment and response propensity scores are present, the
estimator uses `DoubleSamplingEstimand` internally, which multiplies
the two sets of weights together.

**Choosing the right estimand:**

- **ATE** (`ATEEstimator`): Average effect across the whole
  population. Uses `PopulationMean` estimand for both groups. Control
  weights are `1/(1-pi)`, treated weights are `1/pi`.
- **ATT** (`ATTEstimator`): Average effect on the treated. Uses
  `NonRespondentMean` for control (weights = `(1-pi)/pi`) and
  `SampleMean` for treated (weights = 1). Only requires
  `control_propensity_scores`.
- **ATC** (`ATCEstimator`): Average effect on the control. Mirror of
  ATT. Only requires `treated_propensity_scores`.

All three support optional `control_denominator_outcomes` and
`treated_denominator_outcomes` for ratio-based treatment effects
(see below).

### Treatment Effects within Sub-Populations

To estimate treatment effects within a sub-population (e.g., among
chocolate lovers), take the *difference of two `RatioEstimator`s*.
Use the denominator outcomes feature of the treatment effect
estimators:

```python
# Uplift among chocolate lovers in a randomized A/B test
uplift_choc = pyrake.SimpleDifferenceEstimator(
    control_outcomes=C_control * Y_control,
    treated_outcomes=C_treated * Y_treated,
    control_denominator_outcomes=C_control,
    treated_denominator_outcomes=C_treated,
    control_sampling_propensity_scores=control_propensities,
    treated_sampling_propensity_scores=treated_propensities,
)
```

This internally creates `RatioEstimator`s for each group and takes
their difference.

### Cost Efficiency: Sentiment Lift per Cost

When analyzing marketing impact, we often want the cost per unit of
incremental sentiment. It is better to estimate **sentiment lift per
cost** via `TreatmentEffectRatioEstimator` and then invert, because
the ratio estimator provides proper variance estimates.

```python
lift_per_cost = pyrake.TreatmentEffectRatioEstimator(
    treatment_effect_estimator_class="SimpleDifference",
    control_propensity_scores=[0.5],  # irrelevant for SimpleDifference
    treated_propensity_scores=[0.5],  # irrelevant for SimpleDifference
    numerator_control_outcomes=Y_control,
    numerator_treated_outcomes=Y_treated,
    denominator_control_outcomes=spend_control,
    denominator_treated_outcomes=spend_treated,
    control_sampling_propensity_scores=control_propensities,
    treated_sampling_propensity_scores=treated_propensities,
)

pe = lift_per_cost.point_estimate()
ci = lift_per_cost.confidence_interval()

# Invert to get cost per unit of sentiment lift
cost_per_lift = 1.0 / pe
cost_per_lift_ci = (1.0 / ci[1], 1.0 / ci[0])  # flip bounds
```

Note: when using `treatment_effect_estimator_class="SimpleDifference"`,
the `control_propensity_scores` and `treated_propensity_scores`
arguments are ignored (they are only used for ATE/ATT/ATC), but must
still be provided. Pass `[0.5]` as a placeholder.

### Combining Estimates with `LinearCombinationEstimator`

`LinearCombinationEstimator` estimates `theta = offset + sum(a_i * theta_i)`
for independent sub-estimators. This is useful for:

1. **Meta-analysis**: Combining estimates across sub-groups, weighted
   by their population fractions.
2. **Comparing sub-populations**: Testing whether estimates differ
   between groups (e.g., men vs women).

#### Combining across cells/sub-groups

```python
# Per-cell estimators, one for each study cell
cell_estimators = [estimator_cell1, estimator_cell2, estimator_cell3]
cell_fractions = [0.4, 0.35, 0.25]  # sum to 1

total = pyrake.LinearCombinationEstimator(
    estimators=cell_estimators,
    multipliers=cell_fractions,
)

pe = total.point_estimate()     # weighted average
ci = total.confidence_interval()
```

#### Comparing sub-populations

```python
# Is the treatment effect different for men vs women?
men_vs_women = pyrake.LinearCombinationEstimator(
    estimators=[men_estimator, women_estimator],
    multipliers=[1.0, -1.0],  # men minus women
)

pe = men_vs_women.point_estimate()
ci = men_vs_women.confidence_interval()
pv = men_vs_women.pvalue(null_value=0.0)  # test H0: no difference
```

### Sensitivity Analysis

All estimators support sensitivity analysis following Rosenbaum's
framework, which quantifies robustness to hidden biases from
unobserved confounders. The `gamma` parameter controls how "wrong"
the propensity scores might be:

```python
# Point estimate bounds under hidden bias
lb, ub = estimator.sensitivity_analysis(gamma=6.0)

# Confidence interval that accounts for both sampling
# uncertainty and propensity score error
ci = estimator.expanded_confidence_interval(
    alpha=0.10, gamma=6.0, B=1_000
)

# Visual summary across a range of gamma values
df, ax = estimator.plot_sensitivity(
    gamma_lower=1.0,
    gamma_upper=6.0,
    title="Sensitivity to Hidden Bias",
)
```

A `gamma` of 6.0 is a reasonable default for reporting, following
Rosenbaum's convention based on Hammond's smoking-cancer
meta-analysis.

### `DoubleSamplingEstimand`

When there are two sources of selection — treatment assignment and
survey response — compose two `SimpleEstimand` objects:

```python
# Treatment propensity: how likely to be treated
treatment_estimand = pyrake.PopulationMean(treatment_propensities)

# Response propensity: how likely to respond to the survey
response_estimand = pyrake.PopulationMean(response_propensities)

# Combined: weights are the product
double_estimand = pyrake.DoubleSamplingEstimand(
    estimand1=treatment_estimand,
    estimand2=response_estimand,
)

# Use with any mean estimator
estimator = pyrake.SIPWEstimator(
    propensity_scores=np.ones(M),  # placeholder; weights come from estimand
    outcomes=Y,
    estimand=double_estimand,
)
```

In practice, the treatment effect estimators (`ATEEstimator`, etc.)
construct `DoubleSamplingEstimand` automatically when you pass
both treatment propensity scores and `*_sampling_propensity_scores`.

## Complete Example: Survey Analysis in an A/B Test

```python
import numpy as np
import pyrake
import cvxium

# --- Step 1: Prepare data ---
# df_control, df_treated: DataFrames of survey respondents
# covariates: list of column names to balance
# pop_means, pop_stds: population mean/std of covariates
# N: population size

M1 = len(df_control)
M2 = len(df_treated)

MIN_SIGMA = 1e-6
sigma = np.clip(pop_stds.values, MIN_SIGMA, None)

Z_control = df_control[covariates].values.astype(float) / sigma
Z_treated = df_treated[covariates].values.astype(float) / sigma
nu_control = pop_means.values / sigma
nu_treated = pop_means.values / sigma

# --- Step 2: Calibrate weights jointly ---
rake = pyrake.JointCalibrator(
    distance=pyrake.KLDivergence(),
    Z1=Z_control,
    Z2=Z_treated,
    Z3=Z_control,
    Z4=Z_treated,
    nu1=nu_control,
    nu2=nu_treated,
    min_weight1=M1 / N,
    min_weight2=M2 / N,
    phi=0.99,
    settings=cvxium.OptimizationSettings(
        outer=cvxium.OuterSettings(tolerance_soft=0.1),
    ),
)

result = rake.solve_with_fallback()
print(f"Fallback status: {result.fallback_status}")
print(result.fallback_message)

# --- Step 3: Extract propensity scores ---
w = result.solution
control_propensities = (M1 / N) / w[:M1]
treated_propensities = (M2 / N) / w[M1:]

# --- Step 4: Estimate treatment effect ---
Y_control = df_control["outcome"].values.astype(float)
Y_treated = df_treated["outcome"].values.astype(float)

uplift = pyrake.SimpleDifferenceEstimator(
    control_outcomes=Y_control,
    treated_outcomes=Y_treated,
    control_sampling_propensity_scores=control_propensities,
    treated_sampling_propensity_scores=treated_propensities,
)

print(f"Uplift: {uplift.point_estimate():.4f}")
print(f"95% CI: {uplift.confidence_interval(alpha=0.05)}")
print(f"p-value: {uplift.pvalue(null_value=0.0):.4f}")

# --- Step 5: Assess covariate balance ---
pyrake.plot_balance_2_sample(
    X1=df_control[covariates].values.astype(float),
    X2=df_treated[covariates].values.astype(float),
    covariates=covariates,
    sigma1=pop_stds.values,
    sigma2=pop_stds.values,
    weights1={"Calibrated": w[:M1]},
    weights2={"Calibrated": w[M1:]},
    title="Internal Validity",
)
```

## Summary of When to Use Each Estimator

| Scenario                                                   | Estimator                                                |
|------------------------------------------------------------|----------------------------------------------------------|
| Population mean from a single survey                       | `SIPWEstimator`                                          |
| Mean within an observable sub-population                   | `RatioEstimator`                                         |
| Treatment effect in a randomized A/B test                  | `SimpleDifferenceEstimator`                              |
| Treatment effect, observational study                      | `ATEEstimator` / `ATTEstimator` / `ATCEstimator`         |
| Treatment effect within a sub-population                   | Treatment effect estimator with `denominator_outcomes`   |
| Ratio of two treatment effects (e.g., sentiment lift per cost) | `TreatmentEffectRatioEstimator`                          |
| Combining estimates across sub-groups                      | `LinearCombinationEstimator`                             |
| Comparing sub-populations (e.g., men vs women)             | `LinearCombinationEstimator` with `[+1, -1]` multipliers |
