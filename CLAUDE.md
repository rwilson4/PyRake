# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when
working with code in this repository. It explains how to *modify*
this code. For guidance on how to *use* this library in
applications, see USAGE.md (also available at runtime via
`pyrake.usage()`).

## Overview

PyRake is a survey calibration and balancing weights library for
causal inference and observational studies. It solves constrained
convex optimization problems to find weights that balance covariates
between a sample and a target population while controlling variance,
then uses those weights for estimation and inference. The
optimization itself is delegated to
[Cvxium](https://pypi.org/project/cvxium/), an interior point
framework that began life as a PyRake module.

## Development Commands

```bash
# Create the virtual environment and install dependencies
uv sync

# THE FAST LOOP: skips the tests marked `slow` (~30s)
uv run python -m pytest -q -m "not slow"

# The other half, when you have touched the frontier code (~35s)
uv run python -m pytest -q -m "slow"

# Everything (~1 minute), which is what CI runs
uv run python -m pytest

# Find out what is slow before complaining about it
uv run python -m pytest -q --durations=20

# One file, one class, one method, one parameterization
uv run python -m pytest test/calibration/test_rake.py
uv run python -m pytest test/calibration/test_rake.py::RakeSolveTest
uv run python -m pytest test/calibration/test_rake.py::RakeSolveTest::test_solve_kl_divergence_0_M100_p20_no_phi

# Format, lint, and type-check (CI runs all three)
uv run python -m black pyrake/ test/
uv run python -m ruff check pyrake/ test/
uv run python -m mypy
```

When debugging a failure, narrow scope at increasing granularity
rather than re-running the whole suite: file → class → method → single
parameterization. PyRake tests are heavily parameterized across
problem sizes (M=13 to M=500, p=3 to p=100).

### The `slow` marker

Tests that trace a real Pareto frontier cost seconds rather than
milliseconds. They carry `@pytest.mark.slow` (registered in
`test/conftest.py`), which is what `-m "not slow"` selects against. **A
new test that traces a frontier, or otherwise runs for more than a
second or so, should be marked**; the fast loop is only worth having
if it stays fast. CI passes no `-m`, so marking a test does not reduce
the coverage that gates a merge.

### Performance guard

Cvxium's performance guard fails a test whose solver iteration counts
move materially, up *or* down, against `test/perf_baseline.json`.
`test/conftest.py` arms it by setting `CVXIUM_PERF_BASELINE` before
cvxium is imported; set the variable to an empty string to disable
it. If a change intentionally alters iteration counts, regenerate
with `CVXIUM_PERF_BASELINE_WRITE=1 uv run python -m pytest` and
explain why in the commit message. Counts below 8 iterations are not
guarded, and known-bimodal tests are listed under the baseline's
`__ignore__` key.

The baseline is generated on macOS (Accelerate); CI runs Ubuntu
(OpenBLAS) on mixed hardware. The JointCalibrator cascade's
linear-objective barrier solves often stop when a high-t centering step
fails on an ill-conditioned Newton system
([Cvxium #20](https://github.com/rwilson4/Cvxium/issues/20)), and which
step fails depends on the BLAS's rounding. Two settings in `ci.yml`
make Linux reproducible: `OPENBLAS_NUM_THREADS=1`, since multi-threaded
reductions sum in a run-dependent order, and `OPENBLAS_CORETYPE=Haswell`,
since OpenBLAS otherwise picks different kernels on different CPUs.
With them, every runner computes identically.

What remains is a fixed difference between Accelerate and OpenBLAS, of
about one centering step in six tests (four in `test_knee_fallback.py`,
two in `JointCalibratorFallbackTest`). Their baseline entries carry a
`"tolerance"` of 0.2 or 0.25, sized to that difference; regeneration
preserves it. Narrow or drop these once #20 is fixed. If a test starts
tripping the guard on CI but not locally, reproduce it with those two
variables set before widening anything. A band is the right fix only
when the counts differ deterministically by whole centering steps.

### Documentation

The docs site (https://convexanalytics.com/pyrake/) is a Sphinx project
in `docs/` with the pydata theme. The landing page's quickstart is a
doctest, the usage guide is USAGE.md included through MyST, and the API
pages are autodoc.

```bash
cd docs && uv run make html SPHINXOPTS="-W --keep-going"   # warnings are errors
cd docs && uv run make doctest                             # the quickstart
```

Docstrings stay in plain-text style: display equations and lists sit
directly under the prose, with no blank lines. A hook in `docs/conf.py`,
shared with Cvxium, translates them to reStructuredText at build time.
Draw ASCII-art matrices with underscores and pipes (`_  _` on top,
`|_  _|` on the bottom), not dashes: a border line that starts with `-`
reads as a list item. Keep a matrix out of a list item; introduce it
with a line of prose instead. If a docstring breaks the `-W` build, fix
the docstring or the hook rather than loosening `-W`.

`.github/workflows/deploy_docs.yml` builds and doctests on every change
to `docs/`, `pyrake/` or USAGE.md, and deploys over rsync from master
once the `DOCS_HOST`, `DOCS_USER`, and `DOCS_DEPLOY_KEY` secrets exist.
The key's `rrsync` forced command on the server fixes the destination,
`/srv/docs/convexanalytics/pyrake`.

### Releasing

PyRake is published to PyPI as **`pyrake-survey`** and imported as
`pyrake`. The name `pyrake` on PyPI belongs to an unrelated 2016 build
tool, which installs a module named `snake`, so the two don't clash.
Pushing a `v*` tag runs `.github/workflows/publish.yml`, which builds
with `uv build` and uploads through PyPI trusted publishing. To
release:

1. Bump `version` in `pyproject.toml`, run `uv lock`, and commit that
   to master.
2. Tag *that* commit and push the tag: `git tag vX.Y.Z && git push
   origin vX.Y.Z`. The workflow fails if the tag and the version
   disagree, since a tag pushed before the bump would build the old
   version, which PyPI then rejects as a duplicate.

PyPI versions can't be replaced, so check the build first with `uv
build` and an install of the wheel into a clean environment.

### Test conventions

The test suite uses `unittest.TestCase` classes, run by pytest; test
classes follow the `*Test` naming convention. For parameterization,
use `@parameterized.expand` and pass a name as the first element of
each tuple; the generated test ID is `<test_method>_<index>_<label>`
(e.g., `test_solve_kl_divergence_0_M100_p20_no_phi`).

## Architecture

### `pyrake/calibration/` — Finding weights

- **`distance_metrics.py`** — `Distance` (a Cvxium `Objective`) with
  implementations `SquaredL2`, `KLDivergence`, `Huber`. Each provides
  `evaluate`, `gradient`, and `hessian_diagonal`, and reports its
  Hessian to Cvxium as a `DiagonalSystem`.
- **`rake.py`** — `Rake`, the primary user-facing API. Minimizes
  D(w, v) subject to exact balance (1/M) X^T w = μ, approximate
  balance ‖(1/M) Z^T w − ν‖∞ ≤ ψ, a mean squared weight bound
  (1/M) ‖w‖₂² ≤ φ, and weight bounds. It assembles these as a Cvxium
  `Problem` from layered constraints (`EqualityConstraintLayer`,
  `BoundsConstraintLayer`, `ImbalanceConstraintLayer`,
  `SquaredNormConstraintLayer`) and solves it with `BarrierSolver`
  (`engine="ipm"`, the default) or `PrimalDualSolver`
  (`engine="pd"`). Cvxium builds and solves the Phase I feasibility
  problem itself.
- **`joint_calibrator.py`** — `JointCalibrator` extends `Rake` to
  jointly calibrate treatment and control weights for internal
  validity (balance between groups) and external validity (balance
  with the target population). `solve_with_fallback()` degrades
  gracefully through fallback levels when the full problem is
  infeasible, returning a `FallbackOptimizationResult`.
- **`frontier.py`** — `EfficientFrontier` traces the bias-variance
  Pareto frontier over φ; `ExternalValidityVarianceFrontier` traces
  external imbalance against variance. Includes knee detection,
  knee refinement, tangent slopes, and `solve_with_knee_fallback`.
- **`visualizations.py`** — `plot_balance` (one-sample),
  `plot_balance_2_sample`, and the Wilcoxon signed-rank test behind
  them.

### `pyrake/estimation/` — Using weights

- **Estimands**: `PopulationMean`, `NonRespondentMean`, `SampleMean`,
  `DoubleSamplingEstimand`.
- **Mean estimators**: `IPWEstimator`, `AIPWEstimator`,
  `SIPWEstimator`, `SAIPWEstimator`, `RatioEstimator` (delta method
  variance).
- **Treatment effect estimators**: `ATEEstimator`, `ATTEstimator`,
  `ATCEstimator`, `SimpleDifferenceEstimator`,
  `TreatmentEffectRatioEstimator`.
- **`LinearCombinationEstimator`** — affine combinations of
  independent estimators with delta-method variance.
- **`sensitivity_solvers.py`** — the linear fractional program behind
  `RatioEstimator`'s sensitivity analysis, as a Cvxium layer.
- **`visualizations.py`** — `meta_analysis` (Fisher's method across
  platforms).
- All estimators provide `point_estimate`, `variance`,
  `confidence_interval`, `pvalue`, `sensitivity_analysis`,
  `plot_sensitivity`, and `resample` (for the bootstrap).

## Key Design Patterns

- **Hessian structure exploitation**: Newton systems are solved in
  O(p³ + p²M) rather than O(M³) by exploiting the diagonal plus
  low-rank Hessian structure; Cvxium's `MatrixSystem` classes carry
  that structure.
- **Absolute imports**: All modules and test files use
  fully-qualified absolute imports rooted at `pyrake.` (e.g., `from
  pyrake.calibration.rake import Rake`).
- **Narrowing optionals**: use `pyrake._typing.none_throws` rather
  than a bare `assert x is not None` where an expression is needed;
  unlike an assert, it survives `python -O`.
- **Logging**: modules log under the `pyrake` namespace, which gets a
  `NullHandler` on import. `pyrake.enable_logging` wires `pyrake` and
  `cvxium` to one handler so solver output lands beneath the
  calibration stage that invoked it.

## Important Conventions

- **Weight normalization**: Mean weight of 1 corresponds to a true
  weighted average. This differs from standard Horvitz-Thompson
  notation, which uses weights summing to population size.
- **Variance constraint**: The constraint (1/M) ‖w‖₂² ≤ φ controls the
  design effect, which quantifies efficiency loss from
  unrepresentative samples.
- **Default ψ**: When not specified, maximum covariate imbalance
  defaults to 1.5/√M, following Cochran and Chambers (1965) guidance
  on acceptable balance.
- **Minimum weights**: Should generally be ≥ M/N (sample/population
  ratio), since weights < M/N correspond to impossible propensity
  scores > 1.

## Dependencies

numpy, scipy, pandas, matplotlib, seaborn, and cvxium, managed with
uv. Development tools: pytest, parameterized, black, ruff, mypy.

## Tests

Tests are in `test/`, organized by `calibration/` and `estimation/`
subdirectories matching the source layout:

| File                                     | Coverage                                                                                                                                                     |
|------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `calibration/test_distance_metrics.py`   | Distance evaluations, gradients, Hessians                                                                                                                    |
| `calibration/test_rake.py`               | Rake.solve() with all distance metrics; infeasible problems; imbalance constraints; JointCalibrator fallback scenarios; cross-engine (IPM vs PD) equivalence |
| `calibration/test_frontier.py`           | The frontier classes, the knee fallback and tangent slopes, cross-engine equivalence. Holds most of the suite's runtime; largely marked `slow`               |
| `calibration/test_frontier_seam.py`      | Frontiers that cross the seam where external balance must be relaxed                                                                                         |
| `calibration/test_knee_fallback.py`      | `solve_with_knee_fallback` and its degradation paths                                                                                                         |
| `calibration/test_convergence.py`        | Newton step budgets and full steps past quadratic convergence                                                                                                |
| `calibration/test_logging.py`            | `enable_logging` narration and the Cvxium log beneath it                                                                                                     |
| `calibration/test_model_mean_balance.py` | Balance plots against model-based means                                                                                                                      |
| `estimation/test_population.py`          | IPW, AIPW, SIPW, SAIPW population-mean estimators; sensitivity analysis; outcome-proxy-constrained sensitivity; ratio estimator                              |
| `estimation/test_treatment_effects.py`   | ATE/ATT/ATC treatment effect estimators; simple difference; treatment-effect ratio; outcome-proxy-constrained sensitivity                                    |
| `estimation/test_linear_combination.py`  | LinearCombinationEstimator                                                                                                                                   |
| `estimation/test_sensitivity_solvers.py` | The sensitivity linear fractional program, against `scipy.optimize.linprog`                                                                                  |
