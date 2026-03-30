"""Tests for optimization solver classes."""

import numpy as np
import numpy.typing as npt
import pytest
from scipy import linalg

from pyrake.optimization import (
    NewtonResult,
    OptimizationSettings,
    UnconstrainedNewtonSolver,
)


class QuadraticNewtonSolver(UnconstrainedNewtonSolver):
    r"""Solve min 0.5 * x^T Q x + c^T x, where Q is PSD.

    Optimal solution: x* = -Q^{-1} c.
    """

    def __init__(
        self,
        Q: npt.NDArray[np.float64],
        c: npt.NDArray[np.float64],
        settings: OptimizationSettings | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self.Q = Q
        self.c = c
        self._Q_factor = linalg.cho_factor(Q)

    def newton_step(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve Q * delta_x = -grad_f0 = -(Q*x + c)."""
        return linalg.cho_solve(self._Q_factor, -(self.Q @ x + self.c))

    def evaluate_objective(self, x: npt.NDArray[np.float64]) -> float:
        return float(0.5 * x @ self.Q @ x + self.c @ x)

    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.Q @ x + self.c

    def hessian_vector_product(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return self.Q @ y


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
