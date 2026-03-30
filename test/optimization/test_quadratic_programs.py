"""Tests for quadratic program solvers."""

import numpy as np
import pytest
from scipy import linalg

from pyrake.optimization import NewtonResult
from pyrake.optimization.quadratic_programs import QuadraticNewtonSolver


@pytest.mark.parametrize(
    "seed,n",
    [
        (101, 5),
        (201, 10),
        (301, 20),
        (401, 50),
        (501, 100),
    ],
)
def test_unconstrained_newton_quadratic(seed: int, n: int) -> None:
    """UnconstrainedNewtonSolver finds the exact minimizer of a quadratic in one step."""
    rng = np.random.default_rng(seed)

    # Q = A^T A + I ensures strict positive definiteness.
    A = rng.standard_normal((n, n))
    Q = A.T @ A + np.eye(n)
    c = rng.standard_normal(n)
    x0 = rng.standard_normal(n)

    x_star_expected = linalg.solve(Q, -c)

    solver = QuadraticNewtonSolver(Q=Q, c=c)
    result = solver.solve(x0=x0)

    assert isinstance(result, NewtonResult)
    np.testing.assert_allclose(result.solution, x_star_expected, rtol=1e-6, atol=1e-8)
    # Newton's method is exact for quadratics: one step reaches the optimum, then a
    # second iteration detects the near-zero Newton decrement and returns.
    assert result.nits == 2
