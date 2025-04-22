"""Solve optimization problem."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from .distance_metrics import Distance
from .exceptions import ProblemInfeasibleError
from .numerical_helpers import (
    solve_kkt_system_hessian_diagonal_plus_rank_one,
)


@dataclass
class OptimizationSettings:
    """Optimization settings."""

    outer_tolerance: float = 1e-6
    barrier_multiplier: float = 10.0
    inner_tolerance: float = 1e-8
    max_inner_iterations: int = 100
    backtracking_alpha: float = 0.01
    backtracking_beta: float = 0.5


class Rake:
    """Solve optimization problem."""

    def __init__(
        self,
        distance: Distance,
        X: np.ndarray,
        mu: np.ndarray,
        phi: float,
        settings: Optional[OptimizationSettings] = None,
    ):
        M, p = X.shape
        assert mu.shape[0] == p

        self.distance = distance
        self.X = X
        self.mu = mu
        self.phi = phi
        if settings is None:
            self.settings = OptimizationSettings()
        else:
            self.settings = settings

    def interior_point(self):
        r"""Solve optimization problem.

        Uses an interior point method with a logarithmic barrier penalty to
        solve:
           minimize    D(w, v)
           subject to  (1/M) * X^T * w = \mu
                        \| w \|_2^2 \leq \phi
                        w >= 0,
        where D is a distance metric than penalizes deviations from baseline
        weights v.

        Returns
        -------
         w : vector
            The optimal weights.

        """
        p = self.X.shape[1]
        w = self.solve_phase1()
        t = 1.0
        for _ in range(20):
            if p / t < self.settings.outer_tolerance:
                return w
            w = self.centering_step(w, t)
            t *= self.settings.barrier_multiplier
        raise RuntimeError("Interior point method did not converge")

    def centering_step(self, w0: np.ndarray, t: float) -> np.ndarray:
        r"""Solve centering step.

        The centering step solves:
          minimize   ft(w) := t * D(w, v) - log(1 - (1/phi) \| w \|_2^2)
                              - \sum_i log(w_i)
          subject to (1 / M) * X^T * w = \mu.

        Parameters
        ----------
         w0 : vector
            Initial guess.
         v : vector

        Returns
        -------
         w : vector
            Solution to centering step.

        """
        w = w0.copy()
        for _ in range(self.settings.max_inner_iterations):
            delta_w = self.calculate_newton_step(w, t)

            # Check for convergence
            lmbda = self._newton_decrement(w, t, delta_w)
            if 0.5 * lmbda * lmbda < self.settings.inner_tolerance:
                return w

            # Update
            s = self.backtracking_line_search(w, delta_w, t)
            w += s * delta_w

        raise RuntimeError("Centering step did not converge")

    def calculate_newton_step(self, w: np.array, t: float) -> np.ndarray:
        r"""Calculate Newton step.

        Calculates Newton step for the "inner" problem:
          minimize   ft(w) := t * D(w, v) - log(1 - (1/phi) \| w \|_2^2)
                              - \sum_i log(w_i)
          subject to (1 / M) * X^T * w = \mu.


        Returns
        -------
         delta_w : vector
            Newton step.

        Notes
        -----
        The Newton step, delta_w, is the solution of the system:
           _       _   _       _     _         _
          | H   A^T | | delta_w |   | - grad_ft |
          | A    0  | |   xi    | = |      0    |
           -       -   -       -     -         -
        where H is the Hessian of ft evaluated at w, A = (1/M) * X^T, grad_f is
        the gradient of ft evaluated at w, and xi is a auxiliary variable that
        enforces the constraints. We use
        `solve_kkt_system_hessian_diagonal_plus_rank_one` to solve this system
        in O(p^3 + p^2 * M) time.

        """
        delta_w, _ = solve_kkt_system_hessian_diagonal_plus_rank_one(
            A=(1.0 / self.X.shape[0]) * self.X.T,
            g=-self._grad_ft(w, t),
            eta=self._hessian_ft_diagonal(w, t),
            zeta=self._hessian_ft_rank_one(w),
        )

        return delta_w

    def backtracking_line_search(
        self, w: np.ndarray, delta_w: np.ndarray, t: float
    ) -> float:
        """Perform backtracking line search.

        Parameters
        ----------
         w : np.ndarray
            Current estimate.
         delta_w : np.ndarray
            Descent direction.
         t : float
            Barrier penalty.

        Returns
        -------
         s : float
            Step modifier.

        """
        alpha = self.settings.backtracking_alpha
        beta = self.settings.backtracking_beta

        s = 1.0
        w_new = w + s * delta_w
        while np.any(w_new <= 0) or np.sum(w_new * w_new) >= self.phi:
            s *= beta
            w_new = w + s * delta_w

        fw = self._ft(w, t)
        grad_ft = self._grad_ft(w, t)
        grad_ft_dot_delta_w = np.dot(grad_ft, delta_w)

        while self._ft(w_new, t) > fw + alpha * s * grad_ft_dot_delta_w:
            s *= beta
            w_new = w + s * delta_w

        return s

    def solve_phase1(self) -> np.ndarray:
        r"""Find a feasible point.

        A point, w, is feasible if:
           (1/M) * X^T w = \mu
           \| w \|_2^2 \leq \phi
           w >= 0

        We look for such a point by solving:
           minimize   \| w \|_2^2
           subject to (1/M) * X^T w = \mu
                      w >= 0.

        If the solution satisfies \| w_star \|_2^2 \leq \phi, this point is
        feasible, otherwise the problem is infeasible.

        Returns
        -------
         w : np.ndarray
           Feasible point.

        """
        M = self.X.shape[0]
        res = minimize(
            objective=lambda w: np.dot(w, w),
            w0=np.ones((M,)),
            method="trust-constr",
            jac=lambda w: 2 * w,
            constraints=[
                LinearConstraint(
                    (1.0 / M) * (self.X.T),
                    self.mu,
                    self.mu,
                )
            ],
            bounds=Bounds(0, np.inf),
        )

        if not res.success:
            raise ProblemInfeasibleError("Phase I problem did not converge.")

        w = res.x
        if np.dot(w, w) > self.phi:
            raise ProblemInfeasibleError(
                "Minimum feasible norm exceeds variance budget (phi)."
            )

        return w

    def _ft(self, w: np.ndarray, t: float) -> float:
        """Calculate ft at w."""
        return (
            t * self.distance.evaluate(w)
            - np.log(1.0 - (1.0 / self.phi) * np.dot(w, w))
            - np.sum(np.log(w))
        )

    def _grad_ft(self, w: np.ndarray, t: float) -> np.ndarray:
        """Calculate gradient of ft at w."""
        den = 1.0 - (1.0 / self.phi) * np.dot(w, w)
        grad_constraints = w * ((2.0 / self.phi) / den) - 1.0 / w
        return t * self.distance.gradient(w) + grad_constraints

    def _hessian_ft_diagonal(self, w: np.ndarray, t: float) -> np.ndarray:
        """Calculate diagonal component of Hessian of ft at w."""
        den = 1.0 - (1.0 / self.phi) * np.dot(w, w)
        return (
            t * self.distance.hessian_diagonal(w)
            + ((2.0 / self.phi) / den) * np.ones_like(w)
            + 1.0 / (w * w)
        )

    def _hessian_ft_rank_one(self, w: np.ndarray) -> np.ndarray:
        """Calculate rank one component of Hessian of ft at w."""
        den = 1.0 - (1.0 / self.phi) * np.dot(w, w)
        return w * ((2.0 / self.phi) / den)

    def _newton_decrement(
        self, w: np.ndarray, t: float, delta_w: np.ndarray
    ) -> float:
        """Calculate Newton decrement.

        The Newton decrement is the square root of:
           delta_w * H * delta_w,
        where H is the Hessian. Our Hessian is diag(eta) + zeta * zeta^T, so H
        * delta_w is eta * delta_w + (zeta^T * delta_w) * zeta.

        """
        eta = self._hessian_ft_diagonal(w, t)
        zeta = self._hessian_ft_rank_one(w)
        hessian_times_dw = eta * delta_w + (np.dot(zeta, delta_w)) * zeta
        return np.sqrt(np.dot(delta_w, hessian_times_dw))
