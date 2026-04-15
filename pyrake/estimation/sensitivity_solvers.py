"""Cvxium solver for RatioEstimator sensitivity analysis."""

from typing import Any

import numpy as np
import numpy.typing as npt
from cvxium import (
    EqualityConstrainedInteriorPointMethodSolver,
    FeasibilitySolver,
    InteriorPointMethodSolver,
    OptimizationResult,
    OptimizationSettings,
    multiply_arrow_sparsity_pattern,
    solve_arrow_sparsity_pattern,
    solve_kkt_system,
)


class _PrecomputedFeasibility(FeasibilitySolver):
    """Trivial feasibility solver that returns a pre-computed feasible point."""

    def __init__(self, x0: npt.NDArray[np.float64]) -> None:
        self._x0 = x0
        super().__init__()

    @property
    def num_eq_constraints(self) -> int:
        """No equality constraints in the feasibility problem."""
        return 0

    @property
    def num_ineq_constraints(self) -> int:
        """No inequality constraints in the feasibility problem."""
        return 0

    def solve(
        self,
        x0: npt.NDArray[np.float64] | None = None,
        fully_optimize: bool = False,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Return the pre-computed feasible point."""
        return OptimizationResult(solution=self._x0)


class LinearFractionalProgramSolver(
    EqualityConstrainedInteriorPointMethodSolver,
    InteriorPointMethodSolver,
):
    r"""Solve a linear fractional program via its Charnes-Cooper LP reformulation.

    Solves:
        minimize    c^T [w; s]
        subject to  y^T w = 1
                    wl_i * s - w_i <= 0,  i = 1, ..., n
                    w_i - wu_i * s <= 0,  i = 1, ..., n

    This is the Charnes-Cooper transformation of the linear fractional program

        minimize    (c^T w) / (y^T w)
        subject to  wl_i <= w_i <= wu_i

    where w are weights and wl, wu are the bounds from the sensitivity region.

    The Hessian of the barrier objective has an arrow sparsity pattern (diagonal
    upper-left block plus a single dense row/column), enabling O(n) Newton steps.

    Parameters
    ----------
     c : (n,) array
        Objective vector for the weight variables. The scaling variable s has
        coefficient zero.
     y : (n,) array
        Denominator outcomes; defines the equality constraint y^T w = 1.
     wl, wu : (n,) array
        Lower and upper bounds on the original weights from the sensitivity region.
     settings : OptimizationSettings, optional
        Optimization settings.

    Attributes
    ----------
     x0 : (n+1,) array
        A strictly feasible starting point, computed analytically in __init__.

    Notes
    -----
    Hessian structure (barrier only; the objective is linear so its Hessian is zero):

        H = diag(eta)  zeta
            zeta^T     theta

    where, with alpha_i = w_i - wl_i * s and beta_i = wu_i * s - w_i:

        eta_j   = 1/alpha_j^2 + 1/beta_j^2
        zeta_j  = -wl_j/alpha_j^2 - wu_j/beta_j^2
        theta   = sum_j [wl_j^2/alpha_j^2 + wu_j^2/beta_j^2]

    """

    def __init__(
        self,
        c: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        wl: npt.NDArray[np.float64],
        wu: npt.NDArray[np.float64],
        settings: OptimizationSettings | None = None,
    ) -> None:
        """Initialize solver and compute a strictly feasible starting point."""
        self.c = np.asarray(c, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.wl = np.asarray(wl, dtype=float)
        self.wu = np.asarray(wu, dtype=float)

        n = len(self.c)
        # A x = b: [y^T, 0] @ [w; s] = 1
        self._A: npt.NDArray[np.float64] = np.append(self.y, 0.0).reshape(1, n + 1)
        self._b: npt.NDArray[np.float64] = np.array([1.0])

        # Feasible starting point: w0 = midpoint of bounds with a small inward push,
        # s0 chosen to satisfy y^T w = 1.  The nudge ensures strict feasibility even
        # when wl_i == wu_i (degenerate sensitivity region).
        w0 = (self.wl + self.wu) / 2.0 + 1e-8 * (self.wu - self.wl + 1e-10)
        sum0 = float(self.y @ w0)
        self.x0: npt.NDArray[np.float64] = np.append(w0 / sum0, 1.0 / sum0)

        super().__init__(
            phase1_solver=_PrecomputedFeasibility(self.x0),
            settings=settings,
        )

    # ------------------------------------------------------------------
    # Equality constraint properties

    @property
    def A(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Equality constraint matrix [y^T, 0], shape (1, n+1)."""
        return self._A

    @property
    def b(self) -> npt.NDArray[np.float64]:
        """Equality constraint right-hand side [1]."""
        return self._b

    # ------------------------------------------------------------------
    # Constraint counts

    @property
    def num_eq_constraints(self) -> int:
        """Number of equality constraints."""
        return 1

    @property
    def num_ineq_constraints(self) -> int:
        """Number of inequality constraints (2n: lower and upper bounds)."""
        return 2 * len(self.c)

    # ------------------------------------------------------------------
    # Objective

    def evaluate_objective(self, x: npt.NDArray[np.float64]) -> float:
        """Evaluate c^T w (the s variable has zero objective coefficient)."""
        return float(self.c @ x[:-1])

    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Gradient of objective: [c; 0]."""
        return np.append(self.c, 0.0)

    # ------------------------------------------------------------------
    # Inequality constraints

    def constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Evaluate all inequality constraints fi(x) <= 0.

        Returns a (2n,) array:
            f1_i = wl_i * s - w_i  (lower-bound constraint)
            f2_i = w_i - wu_i * s  (upper-bound constraint)
        """
        w, s = x[:-1], x[-1]
        return np.concatenate([self.wl * s - w, w - self.wu * s])

    def grad_constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Jacobian of constraints, shape (2n, n+1).

        The Jacobian is constant (does not depend on x):
            d(f1_i)/d(w_j) = -delta_{ij},  d(f1_i)/d(s) = wl_i
            d(f2_i)/d(w_j) = +delta_{ij},  d(f2_i)/d(s) = -wu_i
        """
        n = len(self.c)
        top = np.hstack([-np.eye(n), self.wl.reshape(-1, 1)])
        bottom = np.hstack([np.eye(n), -self.wu.reshape(-1, 1)])
        return np.vstack([top, bottom])

    def grad_constraints_multiply(
        self,
        x: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Compute J @ v without forming the full (2n, n+1) Jacobian.

        For v = [v_w; v_s] where v_w has length n and v_s is a scalar:
            (J v)[:n] = -v_w + v_s * wl
            (J v)[n:] =  v_w - v_s * wu
        """
        v_w, v_s = v[:-1], v[-1]
        return np.concatenate([-v_w + v_s * self.wl, v_w - v_s * self.wu])

    def grad_constraints_transpose_multiply(
        self,
        x: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Compute J^T @ v without forming the full (2n, n+1) Jacobian.

        For v = [v1; v2] (each of length n):
            (J^T v)[:n] = -v1 + v2
            (J^T v)[n]  = wl^T v1 - wu^T v2
        """
        n = len(self.c)
        v1, v2 = v[:n], v[n:]
        return np.append(-v1 + v2, float(self.wl @ v1 - self.wu @ v2))

    # ------------------------------------------------------------------
    # Hessian

    def _hessian_components(self, x: npt.NDArray[np.float64]) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        float,
    ]:
        """Compute arrow-sparsity-pattern Hessian components eta, zeta, theta."""
        w, s = x[:-1], x[-1]
        alpha = w - self.wl * s  # > 0 at feasible points
        beta = self.wu * s - w  # > 0 at feasible points
        inv_alpha2 = 1.0 / (alpha * alpha)
        inv_beta2 = 1.0 / (beta * beta)
        eta = inv_alpha2 + inv_beta2
        zeta = -self.wl * inv_alpha2 - self.wu * inv_beta2
        theta = float(
            np.sum(self.wl * self.wl * inv_alpha2 + self.wu * self.wu * inv_beta2)
        )
        return eta, zeta, theta

    def hessian_multiply(
        self,
        x: npt.NDArray[np.float64],
        t: float,
        y_vec: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute H_ft @ y_vec using the arrow sparsity pattern.

        The objective is linear so H_f0 = 0 and H_ft = H_barrier.
        """
        eta, zeta, theta = self._hessian_components(x)
        return multiply_arrow_sparsity_pattern(y_vec, eta, zeta, theta)

    # ------------------------------------------------------------------
    # Newton step

    def calculate_newton_step(
        self,
        x: npt.NDArray[np.float64],
        t: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Solve the KKT system for the Newton step.

        Exploits the arrow sparsity pattern of the Hessian to solve
        the (n+2) x (n+2) KKT system in O(n) time.
        """
        eta, zeta, theta = self._hessian_components(x)

        def hessian_solve(b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return solve_arrow_sparsity_pattern(b, eta, zeta, theta)

        return solve_kkt_system(
            A=self._A,
            g=-self.gradient_barrier(x, t),
            hessian_solve=hessian_solve,
        )

    # ------------------------------------------------------------------
    # Dual

    def evaluate_dual(
        self,
        lmbda: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
        x_star: npt.NDArray[np.float64],
    ) -> float:
        """Evaluate the Lagrangian at (x_star, lmbda, nu) as the dual value."""
        return float(
            self.evaluate_objective(x_star)
            + lmbda @ self.constraints(x_star)
            + nu @ (self._A @ x_star - self._b)
        )

    # ------------------------------------------------------------------
    # Backtracking line search

    def btls_keep_feasible(
        self,
        x: npt.NDArray[np.float64],
        delta_x: npt.NDArray[np.float64],
    ) -> float:
        """Compute the largest step s <= 1.0 such that x + s*delta_x is strictly feasible.

        Feasibility requires alpha_i(x + s*dx) > 0 and beta_i(x + s*dx) > 0:
            alpha_i + s * d_alpha_i > 0,  where d_alpha_i = dw_i - wl_i * ds
            beta_i  + s * d_beta_i  > 0,  where d_beta_i  = wu_i * ds - dw_i
        """
        w, s = x[:-1], x[-1]
        dw, ds = delta_x[:-1], delta_x[-1]

        alpha = w - self.wl * s
        beta = self.wu * s - w

        d_alpha = dw - self.wl * ds
        d_beta = self.wu * ds - dw

        max_step = 1.0
        mask_alpha = d_alpha < 0
        if np.any(mask_alpha):
            max_step = min(
                max_step, float(np.min(alpha[mask_alpha] / (-d_alpha[mask_alpha])))
            )

        mask_beta = d_beta < 0
        if np.any(mask_beta):
            max_step = min(
                max_step, float(np.min(beta[mask_beta] / (-d_beta[mask_beta])))
            )

        return 0.99 * max_step
