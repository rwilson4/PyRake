"""Phase I solvers."""

import numpy as np
import numpy.typing as npt
from cvxium import (
    EqualityConstrainedInteriorPointMethodSolver,
    EqualityWithBoundsAndImbalanceConstraintSolver,
    EqualityWithBoundsSolver,
    FeasibilityInteriorPointSolver,
    NewtonResult,
    OptimizationSettings,
    ProblemCertifiablyInfeasibleError,
    solve_diagonal_eta_inverse,
    solve_kkt_system,
    solve_rank_p_update,
)


class EqualityWithBoundsAndNormConstraintSolver(
    EqualityConstrainedInteriorPointMethodSolver, FeasibilityInteriorPointSolver
):
    r"""Find x satisfying A * x = b, x > lb, and \| x \|_2^2 < phi.

    We do this by solving:
      minimize   \| x \|_2^2
      subject to A * x = b
                 x \succeq lb
                 \| B * x - c \|_\infty < psi (optional).

    If the resulting x^\star has \| x^\star \|_2^2 > phi, the original problem is
    infeasible.

    Parameters
    ----------
     phi : float
        Constraint on norm of x.
     phase1_solver : EqualityWithBoundsSolver | EqualityWithBoundsAndImbalanceConstraintSolver, optional
        This class requires a Phase I solver for the equality and inequality
        constraints, and there are two ways of generating this. The user can either pass
        an EqualityWithBoundsSolver (or EqualityWithBoundsAndImbalanceConstraintSolver),
        or pass A, b, and lb (and B, c and psi if applicable), and a PhaseISolver will
        be initialized.
     A, b : npt.NDArray
        Equality constraints: A * x = b. Required when phase1_solver is None.
     lb : float or list[float], optional
        Lower bound on elements of x. Defaults to 0.
     B, c : npt.NDArray, optional
        Parameters for imbalance constraints.
     psi : float, optional
        Parameters for imbalance constraints.
     settings : OptimizationSettings
        Optimization settings.

    """

    def __init__(
        self,
        phi: float,
        phase1_solver: (
            EqualityWithBoundsSolver
            | EqualityWithBoundsAndImbalanceConstraintSolver
            | None
        ) = None,
        A: npt.NDArray[np.float64] | None = None,
        b: npt.NDArray[np.float64] | None = None,
        lb: float | list[float] | npt.NDArray[np.float64] = 0.0,
        B: npt.NDArray[np.float64] | None = None,
        c: npt.NDArray[np.float64] | None = None,
        psi: float | list[float] | npt.NDArray[np.float64] | None = None,
        settings: OptimizationSettings | None = None,
    ) -> None:
        if phase1_solver is not None:
            super().__init__(
                phase1_solver=phase1_solver,
                settings=settings,
            )
        elif B is not None:
            if c is None or psi is None:
                raise ValueError("Must specify `c` and `psi`.")

            if A is None or b is None:
                raise ValueError("Must specify `A` and `b`.")

            super().__init__(
                phase1_solver=EqualityWithBoundsAndImbalanceConstraintSolver(
                    B=B, c=c, psi=psi, A=A, b=b, lb=lb, settings=settings
                ),
                settings=settings,
            )
        elif A is not None and b is not None:
            super().__init__(
                phase1_solver=EqualityWithBoundsSolver(
                    A=A,
                    b=b,
                    lb=lb,
                    settings=settings,
                ),
                settings=settings,
            )
        else:
            raise ValueError("Must specify `A` and `b`.")

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
        B = self.B
        if B is None:
            return self.A.shape[1]

        return self.A.shape[1] + 2 * B.shape[0]

    @property
    def A(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Wrap A."""
        if self.phase1_solver is None:
            raise ValueError("PhaseISolver not specified.")

        if isinstance(
            self.phase1_solver,
            EqualityWithBoundsSolver | EqualityWithBoundsAndImbalanceConstraintSolver,
        ):
            return self.phase1_solver.A

        raise ValueError("PhaseISolver must be an EqualityWithBoundsSolver.")

    @property
    def b(self) -> npt.NDArray[np.float64]:
        """Wrap b."""
        if self.phase1_solver is None:
            raise ValueError("PhaseISolver not specified.")

        if isinstance(
            self.phase1_solver,
            EqualityWithBoundsSolver | EqualityWithBoundsAndImbalanceConstraintSolver,
        ):
            return self.phase1_solver.b

        raise ValueError("PhaseISolver must be an EqualityWithBoundsSolver.")

    def svd_A(  # noqa: N802
        self,
    ) -> tuple[
        npt.NDArray[np.float32 | np.float64],
        npt.NDArray[np.float32 | np.float64],
        npt.NDArray[np.float32 | np.float64],
    ]:
        """Calculate and cache SVD of A."""
        if self.phase1_solver is None:
            raise ValueError("phase1_solver is required.")

        if isinstance(
            self.phase1_solver,
            EqualityWithBoundsSolver | EqualityWithBoundsAndImbalanceConstraintSolver,
        ):
            return self.phase1_solver.svd_A()

        raise ValueError("phase1_solver must be an EqualityWithBoundsSolver")

    @property
    def lb(self) -> float | list[float] | npt.NDArray[np.float64]:
        """Wrap lb."""
        if self.phase1_solver is None:
            raise ValueError("PhaseISolver not specified.")

        if isinstance(
            self.phase1_solver,
            EqualityWithBoundsSolver | EqualityWithBoundsAndImbalanceConstraintSolver,
        ):
            return self.phase1_solver.lb

        raise ValueError("PhaseISolver must be an EqualityWithBoundsSolver.")

    @property
    def B(self) -> npt.NDArray[np.float64] | None:  # noqa: N802
        """Wrap B."""
        if isinstance(
            self.phase1_solver, EqualityWithBoundsAndImbalanceConstraintSolver
        ):
            return self.phase1_solver.B

        return None

    @property
    def c(self) -> npt.NDArray[np.float64] | None:
        """Wrap c."""
        if isinstance(
            self.phase1_solver, EqualityWithBoundsAndImbalanceConstraintSolver
        ):
            return self.phase1_solver.c

        return None

    @property
    def psi(self) -> float | list[float] | npt.NDArray[np.float64] | None:
        """Wrap psi."""
        if isinstance(
            self.phase1_solver, EqualityWithBoundsAndImbalanceConstraintSolver
        ):
            return self.phase1_solver.psi

        return None

    def is_feasible(self, x: npt.NDArray[np.float64]) -> bool:
        """Determine whether a feasible point has been found."""
        return np.dot(x, x) < self.phi

    def check_for_infeasibility(self, result: NewtonResult) -> None:
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

        return max(t1, super().initialize_barrier_parameter(x0))

    def calculate_newton_step(
        self,
        x: npt.NDArray[np.float64],
        t: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Calculate Newton step.

        Calculates Newton step for the "inner" problem:
          minimize   ft(w) := t * \| w \|_2^2 - \sum_i log(w_i - lb)
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
        if self.B is None:
            return solve_kkt_system(
                A=self.A,
                g=-self.gradient_barrier(x, t),
                hessian_solve=solve_diagonal_eta_inverse,
                eta_inverse=self._hessian_ft_diagonal_inverse(x, t),
            )

        assert self.c is not None
        Bx_minus_c = self.B @ x - self.c  # shape (p2,)
        eta_inverse: npt.NDArray[np.float64] = self._hessian_ft_diagonal_inverse(x, t)
        kappa_pos: npt.NDArray[np.float64] = self._hessian_ft_kappa_pos(x, Bx_minus_c)
        kappa_neg: npt.NDArray[np.float64] = self._hessian_ft_kappa_neg(x, Bx_minus_c)

        def A_solve(  # noqa: N802
            b: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            return solve_rank_p_update(
                b,
                kappa=kappa_pos,
                A_solve=solve_diagonal_eta_inverse,
                eta_inverse=eta_inverse,
            )

        def A_solve_nested(  # noqa: N802
            b: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            return solve_rank_p_update(b, kappa=kappa_neg, A_solve=A_solve)

        return solve_kkt_system(
            A=self.A,
            g=-self.gradient_barrier(x, t),
            hessian_solve=A_solve_nested,
        )

    def btls_keep_feasible(
        self, x: npt.NDArray[np.float64], delta_x: npt.NDArray[np.float64]
    ) -> float:
        """Make sure x + btls_s * delta_x stays strictly feasible.

        In our case, we want a btls_s such that:
           x + btls_s * delta_x > lb.
        Since x is strictly feasible, and since btls_s > 0, this is only a concern for
        entries having delta_x[i] < 0, for which we want:
           btls_s < (x[i] - lb) / -delta_x[i].
        We want this for all i having delta_x[i] < 0, so we set btls_s as the minimum of
        these quantities.


        """
        mask1 = delta_x < 0
        theta1 = np.inf
        lb = self.lb
        if np.any(mask1):
            if isinstance(lb, list | tuple | np.ndarray):
                theta1 = np.min((x[mask1] - lb[mask1]) / -delta_x[mask1])
            else:
                theta1 = np.min((x[mask1] - lb) / -delta_x[mask1])

        if self.B is None:
            return float(theta1)

        assert self.c is not None
        assert self.psi is not None
        Bx_minus_c = self.B @ x - self.c
        B_delta_x = self.B @ delta_x

        mask2 = B_delta_x < 0
        theta2 = np.inf
        psi = self.psi
        if np.any(mask2):
            if isinstance(psi, list | tuple | np.ndarray):
                theta2 = np.min((Bx_minus_c[mask2] + psi[mask2]) / -B_delta_x[mask2])
            else:
                theta2 = np.min((Bx_minus_c[mask2] + self.psi) / -B_delta_x[mask2])

        mask3 = B_delta_x > 0
        theta3 = np.inf
        if np.any(mask3):
            if isinstance(psi, list | tuple | np.ndarray):
                theta3 = np.min(-(Bx_minus_c[mask3] - psi[mask3]) / B_delta_x[mask3])
            else:
                theta3 = np.min(-(Bx_minus_c[mask3] - psi) / B_delta_x[mask3])

        return min(theta1, theta2, theta3)

    def evaluate_objective(self, x: npt.NDArray[np.float64]) -> float:
        """Calculate f0 at x."""
        return np.dot(x, x)

    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate gradient of ft at w."""
        return 2.0 * x

    def hessian_multiply(
        self, x: npt.NDArray[np.float64], t: float, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Multiply H * y.

        Our Hessian is diagonal, H = diag(eta), H * y = eta * y

        """
        if self.B is None:
            eta = self._hessian_ft_diagonal(x, t)
            return eta * y

        assert self.c is not None
        Bx_minus_c = self.B @ x - self.c

        # Diagonal component
        eta = self._hessian_ft_diagonal(x, t)  # shape (M,)

        # Rank-q factors
        kappa_pos = self._hessian_ft_kappa_pos(x, Bx_minus_c)  # shape (M, p2)
        kappa_neg = self._hessian_ft_kappa_neg(x, Bx_minus_c)  # shape (M, p2)

        return (  # shape (M,)
            eta * y + kappa_pos @ (kappa_pos.T @ y) + kappa_neg @ (kappa_neg.T @ y)
        )

    def _hessian_ft_diagonal(
        self, x: npt.NDArray[np.float64], t: float
    ) -> npt.NDArray[np.float64]:
        """Calculate diagonal component of Hessian of ft at x."""
        return 2.0 * t + np.square(1.0 / (x - self.lb))

    def _hessian_ft_diagonal_inverse(
        self, x: npt.NDArray[np.float64], t: float
    ) -> npt.NDArray[np.float64]:
        """Calculate diagonal component of Hessian of ft at x."""
        # eta[i]^{-1} = (x[i] - lb)^2 / (2 * t * (x[i] - lb)^2 + 1)
        #
        # We have numerical stability issues when t is large and x is close to lb. In
        # this case, calculating t * (x - lb)^2 as (sqrt(t) * (x - lb))^2 involves
        # multiplying a big number times a small number, giving a reasonable number,
        # then squaring that.
        #
        # Calculating den = (2 * t * (x[i] - lb)^2 + 1) = 2 * tx2[i] + 1 offers no
        # further challenges.
        #
        # Calculating (x[i] - lb)^2 / den may underflow, but calculating (x[i] - lb) / sqrt(den), and
        # then squaring that, may improve stability.
        tx2 = np.square(np.sqrt(t) * (x - self.lb))
        return np.square((x - self.lb) / np.sqrt(2.0 * tx2 + 1.0))

    def _hessian_ft_kappa_pos(
        self, x: npt.NDArray[np.float64], Bx_minus_c: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate positive rank update to Hessian of ft at x."""
        psi = self.psi
        if psi is None:
            raise ValueError("psi not defined")

        denom_pos = psi + Bx_minus_c  # shape (p2,)
        if np.any(denom_pos <= 0):
            raise ValueError(
                "Barrier argument out of domain: denominators must be positive."
            )

        # Each column: B[j, :] / denom_pos[j]
        assert self.B is not None  # Satisfy type-checker
        return self.B.T / denom_pos

    def _hessian_ft_kappa_neg(
        self, x: npt.NDArray[np.float64], Bx_minus_c: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate negative rank update to Hessian of ft at x."""
        psi = self.psi
        if psi is None:
            raise ValueError("psi not defined")

        denom_neg = psi - Bx_minus_c  # shape (q,)
        if np.any(denom_neg <= 0):
            raise ValueError(
                "Barrier argument out of domain: denominators must be positive."
            )
        assert self.B is not None  # Satisfy type-checker
        return self.B.T / denom_neg

    def constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate the vector of constraints, fi(x) <= 0.

        Our constraints are x >= lb, or -x + lb <= 0

        """
        if self.B is None:
            return -x + self.lb

        c = self.c
        psi = self.psi

        if c is None or psi is None:
            raise ValueError("c and psi must be defined")

        Bx_minus_c = self.B @ x - c
        return np.concatenate(
            [
                -x + self.lb,
                -Bx_minus_c - psi,
                Bx_minus_c - psi,
            ]
        )

    def grad_constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate the gradients of constraints, fi(x) <= 0."""
        B = self.B
        if B is None:
            return -np.eye(len(x))

        return np.vstack(
            [
                -np.eye(len(x)),
                -B,
                B,
            ]
        )

    def grad_constraints_multiply(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate grad_constraints(x) @ y."""
        B = self.B
        if B is None:
            return -y

        By = B @ y
        return np.concatenate([-y, -By, By])

    def grad_constraints_transpose_multiply(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate grad_constraints(x).T @ y."""
        B = self.B
        if B is None:
            return -y

        M = self.dimension
        p2 = B.shape[0]

        y1 = y[:M]  # Top block
        y2 = y[M : M + p2]  # Middle block
        y3 = y[M + p2 :]  # Lower block

        return -y1 + B.T @ (y3 - y2)

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
        M = self.dimension
        B = self.B
        if isinstance(self.lb, list | tuple | np.ndarray):
            lb = np.asarray(self.lb)
        else:
            lb = np.full((M,), self.lb)

        if B is None:
            lmbda_minus_AT_nu = lmbda - self.A.T @ nu
            return (
                -0.25 * np.dot(lmbda_minus_AT_nu, lmbda_minus_AT_nu)
                - np.dot(self.b, nu)
                + np.dot(lb, lmbda)
            )

        c = self.c
        psi = self.psi

        if c is None or psi is None:
            raise ValueError("c and psi must be defined")

        p2 = B.shape[0]
        lmbda1 = lmbda[:M]
        lmbda2 = lmbda[M : (M + p2)]
        lmbda3 = lmbda[(M + p2) :]

        lmbda_minus_AT_nu = lmbda1 - self.A.T @ nu + B.T @ (lmbda2 - lmbda3)
        return (
            -0.25 * np.dot(lmbda_minus_AT_nu, lmbda_minus_AT_nu)
            - np.dot(self.b, nu)
            + np.dot(lb, lmbda1)
            + np.dot(c - psi, lmbda2)
            - np.dot(c + psi, lmbda3)
        )
