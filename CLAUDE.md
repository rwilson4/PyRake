# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when
working with code in this repository.

## Project Overview

PyRake is a Python library for calculating balancing weights to adjust
surveys for non-response bias. It solves constrained optimization
problems that balance three objectives:
1. Reduce bias (by emphasizing under-represented respondents)
2. Improve representativeness (by balancing important covariates)
3. Reduce variance (by controlling weight sizes)

The library supports multiple distance metrics (KL Divergence, Squared
L2, Huber) and provides estimators for population means and causal
treatment effects.

## Development Commands

### Environment Setup
```bash
# Create virtual environment and install dependencies
poetry shell
poetry install --no-root
```

### Testing
```bash
# Run all tests
poetry run python -m pytest

# Run specific test file
poetry run python -m pytest test/test_rake.py

# Run specific test function
poetry run python -m pytest test/test_rake.py::test_rake_solve_kl_divergence

# Run tests with verbose output
poetry run python -m pytest -v
```

### Linting
```bash
# Check code formatting with ruff
poetry run python -m ruff check pyrake test

# Check types
poetry run mypy

# Check code formatting with black
poetry run python -m black --check pyrake/ test/

# Auto-format with black (if needed)
poetry run python -m black pyrake/ test/

# Run all quality checks together
poetry run black pyrake/ test/ && \
poetry run ruff check pyrake/ test/ --fix && \
poetry run mypy && \\
poetry run pytest
```

### Installation
```bash
pip install .
```

## Architecture

### Core Components

**Optimization** (`pyrake/optimization/`)
- Implements solvers for convex optimization problems.
- Uses interior point methods with Newton's method for solving
  constrained optimization
- Custom KKT system solvers for equality-constrained problems
- Settings managed via `OptimizationSettings` dataclass

**Calibration** (`pyrake/calibration/`):
- `Rake`: Main class for solving weight calibration problems with
  exact covariate balance constraints (equality), approximate balance
  constraints (l∞ norm), variance constraints (l2 norm), and lower
  bound constraints
- `JointCalibrator`: Extension for jointly calibrating treatment and
  control group weights to achieve both internal validity (balance
  between groups) and external validity (balance with population)
- Solves: minimize D(w, v) subject to (1/M)X^T w = μ, constraints on
  covariate imbalance, mean squared weight, and positivity
- `EfficientFrontier`: Traces out the bias-variance tradeoff by
  solving optimization problems for a range of variance constraints
  (φ)

**Estimation** (`pyrake/estimation/`):
- Hierarchy of estimator classes for population means and treatment
  effects:
  - Base: `MeanEstimator`, `WeightingEstimator`
  - Population means: `IPWEstimator` (Inverse Propensity Weighted),
    `SIPWEstimator` (Stabilized IPW), `AIPWEstimator` (Augmented IPW),
    `SAIPWEstimator` (Stabilized Augmented IPW)
  - Treatment effects: `TreatmentEffectEstimator`, `ATEEstimator`,
    `ATTEstimator`, `ATCEstimator`
- Compute point estimates, variance, confidence intervals, p-values
- Include sensitivity analysis methods for hidden bias (from Zhao,
  Small, and Bhattacharya, 2019)

### Key Design Patterns

**Optimization Strategy**:
- Interior point methods transform inequality constraints into
  logarithmic barrier functions
- Newton's method solves the "centering step" for each barrier
  parameter value
- Barrier parameter increases by `barrier_multiplier` each iteration
  until convergence
- Custom KKT solvers exploit problem structure (diagonal + low-rank
  Hessians)

**Constraint Hierarchy**:
1. **Exact balance** (equality constraints): Force (1/M)X^T w = μ for
   critical covariates
2. **Approximate balance** (l∞ constraints): Limit max imbalance
   ||(1/M)Z^T w - ν||∞ ≤ ψ for important covariates
3. **Implicit balance**: Propensity score weighting approximately
   balances all covariates used in the model

**Estimator Pattern**:
- Double robustness: Augmented estimators are unbiased if either
  propensity scores OR outcome model is correct
- Stabilization: Normalizing weights to sum to sample size ensures
  estimates stay within data range
- Sensitivity analysis: Explore robustness to unobserved confounding

## Important Conventions

- **Weight normalization**: Mean weight of 1 corresponds to a true
  weighted average. This differs from standard Horvitz-Thompson
  notation which uses weights summing to population size.
- **Variance constraint**: The constraint (1/M)||w||₂² ≤ φ controls
  the design effect, which quantifies efficiency loss from
  unrepresentative samples.
- **Default ψ**: When not specified, maximum covariate imbalance
  defaults to 1.5/√M, following Cochran and Chambers (1965) guidance
  on acceptable balance.
- **Minimum weights**: Should generally be ≥ M/N (sample/population
  ratio) since weights < M/N correspond to impossible propensity
  scores > 1.

## Testing Notes

- Tests use parametrized fixtures with different seeds, sample sizes,
  covariate dimensions, and variance constraints
- Test data generation simulates realistic propensity scores using
  beta distributions
- Tests verify both optimization convergence and correctness of
  statistical properties
