"""Phase I solvers."""

import time
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy import linalg

from .exceptions import ProblemInfeasibleError
from .numerical_helpers import (
    solve_arrow_sparsity_pattern_phase1,
    solve_diagonal_eta_inverse,
    solve_kkt_system,
)
from .optimization import (
    NewtonResult,
    OptimizationResult,
    OptimizationSettings,
    PhaseIInteriorPointSolver,
    PhaseISolver,
    ProblemCertifiablyInfeasibleError,
)


class EqualitySolver(PhaseISolver):
    """Find x satisfying A * x = b."""

    def __init__(
        self,
        A: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        phase1_solver: Optional["PhaseISolver"] = None,
        settings: Optional[OptimizationSettings] = None,
    ) -> None:
        super().__init__(phase1_solver=phase1_solver, settings=settings)
        p, _ = A.shape
        assert len(b.shape) == 1
        assert b.shape[0] == p
        self.A = A
        self.b = b

    @property
    def num_eq_constraints(self) -> int:
        """Count equality constraints."""
        return self.A.shape[0]

    @property
    def num_ineq_constraints(self) -> int:
        """Count inequality constraints."""
        return 0

    def solve(
        self,
        x0: Optional[npt.NDArray[np.float64]] = None,
        fully_optimize: bool = False,
        **kwargs,
    ) -> OptimizationResult:
        """Solve A * x = b.

        Parameters
        ----------
         x0 : vector, optional
            Initial guess.

        Returns
        -------
         res : OptimizationResult
            The solution.

        """
        if x0 is not None and np.all(np.abs(self.A @ x0 - self.b) < 1e-10):
            if self.settings.verbose:
                print("  Initial guess was feasible.")
            return OptimizationResult(solution=x0)

        # If A = U * s * Vh, then we seek to solve:
        # U * s * Vh * x = b
        #
        # Then s * Vh * x = U^T * b
        # Vh * x = (U^T * b) / s
        # x = Vh^T * (U^T * b) / s
        if self.settings.verbose:
            start_time = time.time()

        U, s, Vh = linalg.svd(self.A, full_matrices=False)
        rank = int(np.sum(s > 1e-10))
        U_r = U[:, 0:rank]
        if not np.allclose(U_r @ (U_r.T @ self.b), self.b):
            raise ProblemInfeasibleError("b was not in the range of A.")

        s_inv = np.zeros_like(s)
        s_inv[0:rank] = 1.0 / s[0:rank]

        x = Vh.T @ (s_inv * (U.T @ self.b))
        if self.settings.verbose:
            end_time = time.time()
            print(f"  SVD calculated in {1000 * (end_time - start_time):.03f} ms")
        return OptimizationResult(solution=x)


class EqualityWithBoundsSolver(PhaseIInteriorPointSolver):
    r"""Find x satisfying A * x = b and x > 0.

    We do this by solving:
      minimize   s
      subject to A * x = b
                 -x_i <= s
                 s <= s0 + eps,

    equivalent to the standard form:
      minimize   s
      subject to A * x = b
                 -x_i - s <= 0
                 s - (s0 + eps) <= 0.

    We initialize s as s0 := -x.min() + eps, guaranteeing the starting point is
    strictly feasible. The last constraint is needed to make the Hessian of the
    inner problem strictly positive definite. If the resulting s^\star is > 0, the
    original problem is infeasible.

    """

    def __init__(
        self,
        phase1_solver: Optional[EqualitySolver] = None,
        settings: Optional[OptimizationSettings] = None,
    ) -> None:
        if phase1_solver is None:
            raise ValueError("phase1_solver is required.")

        super().__init__(phase1_solver=phase1_solver, settings=settings)
        self.s0_plus_eps = 0.0

    @property
    def dimension(self) -> int:
        """Problem dimension."""
        return self.A.shape[1]

    @property
    def num_eq_constraints(self) -> int:
        """Count equality constraints."""
        return self.A.shape[0]

    @property
    def num_ineq_constraints(self) -> int:
        """Count inequality constraints."""
        return self.A.shape[1] + 1

    @property
    def A(self) -> npt.NDArray[np.float64]:
        """Wrap A."""
        if self.phase1_solver is None:
            raise ValueError("PhaseISolver not specified.")

        if not isinstance(self.phase1_solver, EqualitySolver):
            raise ValueError("PhaseISolver must be an EqualitySolver.")

        return self.phase1_solver.A

    @property
    def b(self) -> npt.NDArray[np.float64]:
        """Wrap b."""
        if self.phase1_solver is None:
            raise ValueError("PhaseISolver not specified.")

        if not isinstance(self.phase1_solver, EqualitySolver):
            raise ValueError("PhaseISolver must be an EqualitySolver.")

        return self.phase1_solver.b

    def is_feasible(self, x: npt.NDArray[np.float64]) -> bool:
        """Determine whether a feasible point has been found."""
        return x[-1] < 0

    def check_for_infeasibility(self, result: NewtonResult):
        """Check if infeasible."""
        if result.dual_value > 0:
            raise ProblemCertifiablyInfeasibleError(
                message=(
                    "Problem certifiably infeasible: dual value was "
                    f"{result.dual_value} > 0"
                ),
                result=result,
            )

    def augment_previous_solution(
        self, phase1_res: OptimizationResult, **kwargs
    ) -> npt.NDArray[np.float64]:
        """Initialize variable based on Phase I result."""
        x = np.zeros((len(phase1_res.solution) + 1,))
        x[0 : self.dimension] = phase1_res.solution
        eps = 1.0
        x[self.dimension] = -phase1_res.solution.min() + eps
        self.s0_plus_eps = x[self.dimension] + eps
        return x

    def finalize_solution(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """De-augment solution."""
        return x[0 : self.dimension]

    def initialize_barrier_parameter(self, x0: npt.NDArray[np.float64]) -> float:
        r"""Initialize barrier parameter.

        We try 2 different approaches to initializing the barrier parameter, based on
        the discussion in Boyd and Vandenberghe, section 11.3.1.
          1. Choose t0 so that m / t0 is of the same order as f0(x0) - p_star,
             where m is the number of inequality constraints. Since f0 = s, and the
             happy path is p_star < 0 for a feasible solution, f0(x0) - p_star > s0.
             Thus, we want m / t0 > s0, or t0 < m / s0.
          2. Choose t0 to minimize the deviation of x0 from x^\star(t0). This ends up
             being a least squares equation (Eq 11.12), and is quite a bit more
             complicated, but ends up having a fairly simple formula.

        """
        M = self.dimension
        t1 = self.num_ineq_constraints / max(self.settings.outer_tolerance, x0[M])
        t2 = np.sum(1.0 / (x0[0:M] + x0[M])) - M / 1.0
        return max(1.0, t1, t2)

    def calculate_newton_step(
        self,
        x: npt.NDArray[np.float64],
        t: float,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Calculate Newton step.

        Calculates Newton step for the "inner" problem:
          minimize   ft(w) := t * -s - \sum_i log(w_i - s)
          subject to A * w = b.

        Parameters
        ----------
         x : vector
            Current estimate, [w; s].
         t : float
            Barrier parameter.

        Returns
        -------
         delta_x : vector
            Newton step.
         nu : vector
            Lagrange multiplier associated with equality constraints.

        Notes
        -----
        The Newton step, delta_x, is the solution of the system:
           _       _   _       _     _         _
          | H   A^T | | delta_x |   | - grad_ft |
          | A    0  | |   nu    | = |      0    |
           -       -   -       -     -         -
        where H is the Hessian of ft evaluated at x, grad_f is the gradient of ft
        evaluated at x, and nu is the Lagrange multiplier associated with the equality
        constraints. We use `solve_kkt_system` to solve this system in O(p^3 + p^2 * M)
        time.

        """
        return solve_kkt_system(
            A=np.hstack((self.A, np.zeros((self.num_eq_constraints, 1)))),
            g=-self.gradient_barrier(x, t),
            hessian_solve=solve_arrow_sparsity_pattern_phase1,
            eta_inverse=self._hessian_ft_diagonal_inverse(x, t),
            one_over_psi_squared=self._hessian_one_over_psi_squared(x),
        )

    def btls_keep_feasible(
        self, x: npt.NDArray[np.float64], delta_x: npt.NDArray[np.float64]
    ) -> float:
        """Make sure x + btls_s * delta_x stays strictly feasible.

        In our case, we want a btls_s such that
           -(w + btls_s * delta_w) < s + btls_s * delta_s, and
           s + btls_s * delta_s < s0 + eps.

        The first criterion is equivalent to:
            (w + s) + btls_s * (delta_w + delta_s) > 0,
        which is automatically satisfied whenever delta_w + delta_s >= 0, since w and s
        are feasible (so w + s > 0). So we're only concerned with entries i having
        delta_w[i] + delta_s < 0, in which case we need
            btls_s < (w[i] + s) / -(delta_w[i] + delta_s),
        so we set btls_s as the minimum of these quantities. If all entries satisfy
        delta_w + delta_s >= 0, this constraint is not active.

        Similarly, if delta_s <= 0, there's no problem since s is strictly feasible. But
        if delta_s > 0, we want btls_s < ((s0 + eps) - s) / delta_s

        """
        M = self.dimension
        w = x[0:M]
        s = x[M]
        delta_w = delta_x[0:M]
        delta_s = delta_x[M]

        # Calculate an initial step size btls_s satisfying:
        #   btls_s <= 1
        #   w + btls_s * delta_w > -s - btls_s * delta_s
        #   s + btls_s * delta_s < s0 + eps
        btls_s = min(
            np.min(
                np.where(delta_w + delta_s < 0, (w + s) / -(delta_w + delta_s), np.inf)
            ),
            (self.s0_plus_eps - s) / delta_s if delta_s > 0 else np.inf,
        )

        return btls_s

    def evaluate_objective(self, x: npt.NDArray[np.float64]) -> float:
        """Calculate f0 at w."""
        return x[self.dimension]

    def evaluate_barrier_objective(self, x: npt.NDArray[np.float64], t: float) -> float:
        r"""Calculate ft at x.

        Our barrier objective is:
           ft(x, s) = t * s - \sum_i log(x_i + s) - M * log(s0 + eps - s).

        This differs from the nominal barrier objective in that we give additional
        weight to the last constraint to help it compete with the others. Otherwise,
        it's too easy to decrease the barrier objective by making s larger, rather than
        making x_i larger (which is what we want).

        """
        c = self.constraints(x)
        M = self.dimension
        return t * x[M] - np.sum(np.log(-c[0:M])) - M * np.log(-c[M])

    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Calculate gradient of f0 at x.

        f0(x, s) = s, so \partial f0 / \partial x_i = 0 and \partial f0 / \partial s =
        1.

        """
        grad = np.zeros_like(x)
        grad[self.dimension] = 1.0
        return grad

    def gradient_barrier(
        self, x: npt.NDArray[np.float64], t: float
    ) -> npt.NDArray[np.float64]:
        r"""Calculate gradient of ft at x.

        Our barrier objective is:
           ft(x, s) = t * s - \sum_i log(x_i + s) - M * log(s0 + eps - s),
        so:
           \partial ft / \partial x_i = -1 / (x_i + s), and
           \partial ft / \partial  s  = t - \sum_i (1 / (x_i + s)) + M / (s0 + eps - s)

        """
        M = self.dimension
        g = np.zeros((M + 1,))
        w = x[0:M]
        s = x[M]
        g[0:M] = -1.0 / (w + s)
        g[M] = t - np.sum(1.0 / (w + s)) + M / (self.s0_plus_eps - s)
        return g

    def hessian_multiply(
        self, x: npt.NDArray, t: float, y: npt.NDArray
    ) -> npt.NDArray[np.float64]:
        """Multiply H * y.

        Our Hessian has an arrow sparsity pattern,
                 _                 _
                |  diag(eta)  zeta  |
            H = |                   |,
                |_  zeta^T   theta _|
        so the ith entry of H * y is eta[i] * y[i] + zeta[i] * y[-1], for all but the
        last entry of H * y, and the last entry is np.dot(zeta, y[0:-1]) + theta * y[-1].


        """
        M = self.dimension
        eta = self._hessian_ft_diagonal(x, t)
        zeta = self._hessian_ft_edge(x)
        theta = self._hessian_ft_corner(x)
        Hy = np.zeros_like(y)
        Hy[0:M] = eta * y[0:M] + y[M] * zeta
        Hy[M] = np.dot(zeta, y[0:M]) + theta * y[M]
        return Hy

    def _hessian_ft_diagonal(
        self, x: npt.NDArray[np.float64], t: float
    ) -> npt.NDArray[np.float64]:
        r"""Calculate diagonal component of Hessian of ft at x.

        Our barrier objective is:
           ft(x, s) = t * s - \sum_i log(x_i + s) - M * log(s0 + eps - s),
        so:
          \partial ft / \partial x_i = -1 / (x_i + s), and
          \partial ft / \partial  s  = t - \sum_i (1 / (x_i + s)) + M / (s0 + eps - s),
        and:
          eta[i]  := \partial^2 ft / \partial x_i^2
                   = 1 / (x_i + s)^2,
          zeta[i] := \partial^2 ft / \partial s \partial x_i
                   = 1 / (x_i + s)^2,
          theta   := \partial^2 ft / \partial s^2
                   = \sum_i 1 / (x_i + s)^2 + M / (s0 + eps - s)^2.

        """
        M = self.dimension
        w = x[0:M]
        s = x[M]
        return np.square(1.0 / (w + np.full_like(w, s)))

    def _hessian_ft_diagonal_inverse(
        self, x: npt.NDArray[np.float64], t: float
    ) -> npt.NDArray[np.float64]:
        """Numerically stable version of eta^{-1}."""
        M = self.dimension
        w = x[0:M]
        s = x[M]
        return np.square(w + np.full_like(w, s))

    def _hessian_ft_edge(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate last row/column of Hessian of ft at x."""
        M = self.dimension
        w = x[0:M]
        s = x[M]
        return np.square(1.0 / (w + np.full_like(w, s)))

    def _hessian_ft_corner(self, x: npt.NDArray[np.float64]) -> float:
        """Calculate bottom right corner of Hessian of ft at x."""
        M = self.dimension
        w = x[0:M]
        s = x[M]
        return (M * ((1.0 / (self.s0_plus_eps - s)) ** 2)) + np.sum(
            np.square(1.0 / (w + np.full_like(w, s)))
        )

    def _hessian_one_over_psi_squared(self, x: npt.NDArray[np.float64]) -> float:
        M = self.dimension
        s0_plus_eps_minus_s = self.s0_plus_eps - x[M]
        return (s0_plus_eps_minus_s * s0_plus_eps_minus_s) / M

    def constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate the vector of constraints, fi(x) <= 0.

        Our constraints are:
            -x_i <= s,
              s  <= s0 + eps,
        or:
           -x_i - s              <= 0,
                  s - (s0 + eps) <= 0

        """
        M = self.dimension
        c = np.zeros((M + 1,))
        c[0:M] = -(x[0:M] + x[M])
        c[M] = x[M] - self.s0_plus_eps
        return c

    def grad_constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate the gradients of constraints, fi(x) <= 0.

        In this case, the matrix of gradients is:
           _       _
          | -I  -1 |
          |  0   1 |
           -       -


        """
        M = self.dimension
        G = np.zeros((M + 1, M + 1))
        G[0:M, 0:M] = -np.eye(M)
        G[0:M, M] = -1
        G[M, M] = 1
        return G

    def evaluate_dual(
        self,
        lmbda: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
        x_star: npt.NDArray[np.float64],
    ) -> float:
        r"""Evaluate dual function.

        Parameters
        ----------
         lmbda : vector
            Lagrange multipliers for inequality constraints.
         nu : vector
            Lagrange multipliers for equality constraints.

        Notes
        -----
        The Lagrangian is:
           s - lambda_1^T (x + s*1) + lambda_2 * (s - (s0 + eps)) + nu^T (A*x - b)
          = -b^T nu - (s0 + eps) * lambda_2 +  x^T (A^T nu - lambda_1)
            + s * (1 + lambda_2 - 1^T lambda_1).
        The Lagrangian dual function is:
          g(lambda_1, lambda_2, nu) = -b^T nu - (s0 + eps) * lambda_2
                                      + inf_x { x^T (A^T nu - lambda_1) }
                                      + inf_s { s * (1 + lambda_2 - 1^T lambda_1) },
        which is unbounded below unless:
             lambda_1 = A^T nu, and
             lambda_2 = 1^T lambda_1 - 1.
        So the dual function equals -b^T nu - (s0 + eps) * lambda_2 in that case, and is
        -infinity otherwise.

        """
        M = self.dimension
        if not np.allclose(lmbda[0:M], self.A.T @ nu, atol=1e-3):
            return -np.inf

        if abs(M * lmbda[M] - (np.sum(lmbda[0:M]) - 1)) > 1e-3:
            return -np.inf

        return -np.dot(self.b, nu) - M * lmbda[M] * self.s0_plus_eps


class EqualityWithBoundsAndNormConstraintSolver(PhaseIInteriorPointSolver):
    r"""Find x satisfying A * x = b, x > 0, and \| x \|_2^2 < phi.

    We do this by solving:
      minimize   \| x \|_2^2
      subject to A * x = b
                 x \succeq 0.

    If the resulting x^\star has \| x^\star \|_2^2 > phi, the original problem is
    infeasible.

    Parameters
    ----------
     x0 : vector, optional
        Initial guess for x. If A * x0 = b, x0 > 0, and \| x0 \|_2^2 < phi, we
        simply return it. If A * x0 = b and x0 > 0, we use it as a starting point.

    Returns
    -------
     res : OptimizationResult
        The solution.

    """

    def __init__(
        self,
        phi: float,
        phase1_solver: Optional[EqualityWithBoundsSolver] = None,
        settings: Optional[OptimizationSettings] = None,
    ) -> None:
        if phase1_solver is None:
            raise ValueError("phase1_solver is required.")

        super().__init__(phase1_solver=phase1_solver, settings=settings)
        self.phi = phi

    @property
    def dimension(self) -> int:
        """Problem dimension."""
        return self.A.shape[1]

    @property
    def num_eq_constraints(self) -> int:
        """Count equality constraints."""
        return self.A.shape[0]

    @property
    def num_ineq_constraints(self) -> int:
        """Count inequality constraints."""
        return self.A.shape[1]

    @property
    def A(self) -> npt.NDArray[np.float64]:
        """Wrap A."""
        if self.phase1_solver is None:
            raise ValueError("PhaseISolver not specified.")

        if not isinstance(self.phase1_solver, EqualityWithBoundsSolver):
            raise ValueError("PhaseISolver must be an EqualityWithBoundsSolver.")

        return self.phase1_solver.A

    @property
    def b(self) -> npt.NDArray[np.float64]:
        """Wrap b."""
        if self.phase1_solver is None:
            raise ValueError("PhaseISolver not specified.")

        if not isinstance(self.phase1_solver, EqualityWithBoundsSolver):
            raise ValueError("PhaseISolver must be an EqualityWithBoundsSolver.")

        return self.phase1_solver.b

    def is_feasible(self, x: npt.NDArray[np.float64]) -> bool:
        """Determine whether a feasible point has been found."""
        return np.dot(x, x) < self.phi

    def check_for_infeasibility(self, result: NewtonResult):
        """Check if infeasible."""
        if result.dual_value > self.phi:
            raise ProblemCertifiablyInfeasibleError(
                message=(
                    "Problem certifiably infeasible: dual value was "
                    f"{result.dual_value} > {self.phi}"
                ),
                result=result,
            )

    def initialize_barrier_parameter(self, x0: npt.NDArray[np.float64]) -> float:
        r"""Initialize barrier parameter.

        We try 2 different approaches to initializing the barrier parameter, based on
        the discussion in Boyd and Vandenberghe, section 11.3.1.
          1. Choose t0 so that m / t0 is of the same order as f0(x0) - p_star,
             where m is the number of inequality constraints. Since f0 = \| x \|^2,
             p_star is at least 0, so f0(x0) - p_star <= \| x0 \|_2^2. Thus, we want
             m / t0 = \| x0 \|_2^2, or t0 = m / \| x0 \|_2^2.
          2. Choose t0 to minimize the deviation of x0 from x^\star(t0). This ends up
             being a least squares equation (Eq 11.12), and is quite a bit more
             complicated.

        """
        t1 = self.num_ineq_constraints / max(
            self.settings.outer_tolerance, np.dot(x0, x0)
        )

        delta_phi = -(self.grad_constraints(x0).T @ (1.0 / self.constraints(x0)))
        delta_f0 = self.gradient(x0)
        A_delta_phi = self.A @ delta_phi
        A_delta_f0 = self.A @ delta_f0

        # z_phi = (A * A^T)^{-1} * A * delta_phi
        # Singular Value Decomposition: A = U * diag(s) * V^T,
        # where U is p-by-r (r is the rank of A), s is length r, V is M-by-r.
        # Since the columns of U and V are orthonormal, U^T U = V^T V = I_r,
        # but in general U * U^T does not equal I_p and V^T * V does not equal I_M.
        #
        # A * A^T = U * diag(s) * V^T * V * diag(s) * U^T
        #            = U * diag(s^2) * U^T
        # A * A^T * z_phi = A * delta_phi
        # -> U * diag(s^2) * U^T * z_phi = A * delta_phi
        # -> diag(s^2) * U^T * z_phi = U^T * A * delta_phi
        # -> U^T * z_phi = diag(s^{-2}) * U^T * A * delta_phi
        #
        # Now, in general U * U^T does not equal I_p, but if we *define*
        # z_phi = U * diag(s^{-2}) * U^T * A * delta_phi,
        # then U^T * z_phi = U^T * U * diag(s^{-2}) * U^T * A * delta_phi
        #                  = diag(s^{-2}) * U^T * A * delta_phi,
        # so this z_phi satisfies the equation, and is the minimum norm solution.
        #
        # When A has full row rank, z_phi is the *unique* solution to the equation,
        # but this strategy works even when A is rank deficient.
        U, s, Vh = linalg.svd(self.A, full_matrices=False)
        rank = int(np.sum(s > 1e-5))
        U = U[:, 0:rank]
        s = s[0:rank]
        z_phi = U @ ((U.T @ A_delta_phi) / np.square(s))
        z_f0 = U @ ((U.T @ A_delta_f0) / np.square(s))

        t2 = -(
            (np.dot(delta_f0, delta_phi) - np.dot(A_delta_f0, z_phi))
            / (np.dot(delta_f0, delta_f0) - np.dot(A_delta_f0, z_f0))
        )
        return max(1.0, t1, t2)

    def calculate_newton_step(
        self,
        x: npt.NDArray[np.float64],
        t: float,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Calculate Newton step.

        Calculates Newton step for the "inner" problem:
          minimize   ft(w) := t * \| w \|_2^2 - \sum_i log(w_i)
          subject to A * w = b.

        Parameters
        ----------
         x : vector
            Current estimate, [w; s].
         t : float
            Barrier parameter.

        Returns
        -------
         delta_x : vector
            Newton step.
         nu : vector
            Lagrange multiplier associated with equality constraints.

        Notes
        -----
        The Newton step, delta_x, is the solution of the system:
           _       _   _       _     _         _
          | H   A^T | | delta_x |   | - grad_ft |
          | A    0  | |   nu    | = |      0    |
           -       -   -       -     -         -
        where H is the Hessian of ft evaluated at x, grad_f is the gradient of ft
        evaluated at x, and nu is the Lagrange multiplier associated with the equality
        constraints. We use `solve_kkt_system` to solve this system in O(p^3 + p^2 * M)
        time.

        """
        return solve_kkt_system(
            A=self.A,
            g=-self.gradient_barrier(x, t),
            hessian_solve=solve_diagonal_eta_inverse,
            eta_inverse=self._hessian_ft_diagonal_inverse(x, t),
        )

    def btls_keep_feasible(
        self, x: npt.NDArray[np.float64], delta_x: npt.NDArray[np.float64]
    ) -> float:
        """Make sure x + btls_s * delta_x stays strictly feasible.

        In our case, we want a btls_s such that:
           x + btls_s * delta_x > 0.
        Since x is strictly feasible, and since btls_s > 0, this is only a concern for
        entries having delta_x[i] < 0, for which we want:
           btls_s < x[i] / -delta_x[i].
        We want this for all i having delta_x[i] < 0, so we set btls_s as the minimum of
        these quantities.


        """
        return np.min(np.where(delta_x < 0, x / -delta_x, np.inf))

    def evaluate_objective(self, x: npt.NDArray[np.float64]) -> float:
        """Calculate f0 at x."""
        return np.dot(x, x)

    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate gradient of ft at w."""
        return 2.0 * x

    def gradient_barrier(
        self, x: npt.NDArray[np.float64], t: float
    ) -> npt.NDArray[np.float64]:
        """Calculate gradient of ft at x."""
        return (2.0 * t) * x - 1.0 / x

    def hessian_multiply(
        self, x: npt.NDArray, t: float, y: npt.NDArray
    ) -> npt.NDArray[np.float64]:
        """Multiply H * y.

        Our Hessian is diagonal, H = diag(eta), H * y = eta * y

        """
        eta = self._hessian_ft_diagonal(x, t)
        return eta * y

    def _hessian_ft_diagonal(
        self, x: npt.NDArray[np.float64], t: float
    ) -> npt.NDArray[np.float64]:
        """Calculate diagonal component of Hessian of ft at x."""
        return 2.0 * t + np.square(1.0 / x)

    def _hessian_ft_diagonal_inverse(
        self, x: npt.NDArray[np.float64], t: float
    ) -> npt.NDArray[np.float64]:
        """Calculate diagonal component of Hessian of ft at x."""
        # eta[i]^{-1} = x[i]^2 / (2 * t * x[i]^2 + 1)
        #
        # We have numerical stability issues when t is large and x is close to 0. In
        # this case, calculating t * x^2 as (sqrt(t) * x)^2 involves multiplying a big
        # number times a small number, giving a reasonable number, then squaring that.
        #
        # Calculating den = (2 * t * x[i]^2 + 1) = 2 * tx2[i] + 1 offers no further
        # challenges.
        #
        # Calculating x[i]^2 / den may underflow, but calculating x[i] / sqrt(den), and
        # then squaring that, may improve stability.
        tx2 = np.square(np.sqrt(t) * x)
        return np.square(x / np.sqrt(2.0 * tx2 + 1.0))

    def constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate the vector of constraints, fi(x) <= 0.

        Our constraints are x >= 0, or -x <= 0

        """
        return -x

    def grad_constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate the gradients of constraints, fi(x) <= 0."""
        return -np.eye(len(x))

    def evaluate_dual(
        self,
        lmbda: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
        x_star: npt.NDArray[np.float64],
    ) -> float:
        """Evaluate dual function.

        Parameters
        ----------
         lmbda : vector
            Lagrange multipliers for inequality constraints.
         nu : vector
            Lagrange multipliers for equality constraints.

        Returns
        -------
         g : float
            The dual function evaluated at lmbda, nu.

        Notes
        -----
        Because the dual has a simple form, we can evaluate it at any lmbda, nu, not
        just the optimal lambda_star, nu_star.

        """
        lmbda_minus_AT_nu = lmbda - self.A.T @ nu
        return -np.dot(self.b, nu) - 0.25 * np.dot(lmbda_minus_AT_nu, lmbda_minus_AT_nu)
