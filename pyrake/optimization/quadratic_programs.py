"""Quadratic program solvers."""

import numpy as np
import numpy.typing as npt
from scipy import linalg

from .optimization import OptimizationSettings, UnconstrainedNewtonSolver


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
