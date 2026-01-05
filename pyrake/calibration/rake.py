"""Solve optimization problem."""

import numpy as np
import numpy.typing as npt

from ..optimization import (
    EqualityConstrainedInteriorPointMethodSolver,
    InteriorPointMethodSolver,
    OptimizationSettings,
    PhaseISolver,
    solve_diagonal_eta_inverse,
    solve_kkt_system,
    solve_rank_one_update,
    solve_rank_p_update,
)
from .distance_metrics import Distance
from .phase1solvers import (
    EqualityWithBoundsAndImbalanceConstraintSolver,
    EqualityWithBoundsAndNormConstraintSolver,
    EqualityWithBoundsSolver,
)


class Rake(EqualityConstrainedInteriorPointMethodSolver, InteriorPointMethodSolver):
    r"""Solve optimization problem.

    Class for solving problems of the form:
           minimize    D(w, v)
           subject to  (1/M) * X^T * w = \mu
                       \| (1/M) * Z^T * w - \nu \|_\infty <= \psi (optional)
                       (1/M) * \| w \|_2^2 <= \phi (optional)
                        w >= min_weight,
    with optional constraints on max covariate imbalance and on the mean of the weights.

    Parameters
    ----------
     distance : Distance
         Distance object representing distance from baseline weights.
     X : npt.NDArray[np.float64]
         Matrix of covariates for which we desire exact balance.
     mu : npt.NDArray[np.float64]
         Vector of mean covariate values in the target population corresponding to X.
     Z : npt.NDArray[np.float64], optional
         Matrix of covariates for which we desire approximate balance.
     nu : npt.NDArray[np.float64], optional
         Vector of mean covariate values in the target population corresponding to Z.
         Required when Z is specified.
     psi : float or list[float], optional
         Bound on maximum covariate imbalance. See Notes. Defaults to 1.5 / sqrt(M),
         following guidance from (Cochran and Chambers, 1965, §3.1).
     phi : float
         Constraint on mean squared weight.
     min_weight : float or list[float], optional
         Lower bound on weights. Defaults to 0.
     constrain_mean_weight_to : float or None, default=1
         If not None, add a constraint that the average weight must equal
         `constrain_mean_weight_to`, which must be strictly positive. The default is an
         average weight of 1. Another common option is for the average weight to equal
         the average baseline weight. If no such constraint is desired, manually specify
         `constrain_mean_weight_to=None`.
     settings : OptimizationSettings
        Optimization settings.

    Notes
    -----
    Three levels of balance are supported here. The lowest level implicitly relies on
    the balancing property of propensity scores: inverse weighting approximately
    balances all covariates used to fit the models, all interactions, higher order
    moments, etc. But this is only true asymptotically; with finite sample sizes, some
    imbalance may remain.

    The next level is a constraint on the maximum imbalance. For covariates
    corresponding to the inputs Z and nu, the vector of covariate imbalances, after
    weighting, is (1/M) * Z^T * w - \nu. We want these imbalances to be close to zero,
    but not necessarily exactly zero. Even a true random sample would have some chance
    imbalances. The l∞ constraint allows us to achieve an adjustment strategy that
    mimics a random sample.

    When we assess covariate balance (using the tools in visualizations.py), we plot z
    statistics: adjusted sample mean minus population mean divided by sqrt(M) times the
    population standard deviation. If we whiten Z and nu (divide by the population
    standard deviation), which we typically recommend to put all covariates on a common
    scale, then setting psi to, say 1.96 / sqrt(M) will constrain covariate imbalances
    to be not-stat-sig at level 0.05. But (Cochran and Chambers, 1965, §3.1) emphasizes
    bias can occur even without stat sig differences in covariates. Their recommendation
    is equivalent to setting psi = 1.5 / sqrt(M), which is the default here.

    The highest level is to force the adjusted sample mean to match the population mean
    exactly. This can typically only be done for a few covariates without drastically
    inflating the variance.

    We thus recommend having a strong point of view on the covariates that are most
    important for predicting the response. Use propensity scores to balance all
    covariates approximately; balance the means of the more important covariates using
    the l∞ constraint; balance exactly the means only of the most important covariates.

    In a sense, the ideal weight to use is (M/N) / pi, where N is the population size
    and pi the true propensity score. That yields unbiased estimates, but of course
    requires known propensity scores. Weights that are less than (M/N) correspond to
    propensity scores > 1, which are impossible a priori. Thus, weights should generally
    never be less than M/N, which can be incorporated into the optimization problem
    using the `min_weight` argument.

    `min_weight` and `psi` can both be vector arguments. This is particularly helpful
    when jointly calibrating weights for treatment and control groups. In this case,
    min_weight for the control (treated) observations should be the control (treated)
    sample size divided by the population size, so different bounds should be specified
    for the two groups. For joint calibration, we may have up to 3 groups of imbalance
    constraints, comparing treatment and control (internal validity), comparing control
    to the target population (external validity), and comparing treatment to the target
    population (external validity). The two sets of external validity constraints would
    use, e.g. 1.5 / sqrt(control sample size) or 1.5 / sqrt(treated sample size),
    while the internal validity constraint would use
       1.5 * sqrt(1 / (control sample size) + 1 / (treated sample size)),
    corresponding to a 2 sample z test.

    References
    ----------
    - Cochran, William Gemmell and Chambers, S. Paul, The Planning of Observational
      Studies of Human Populations, Journal of the Royal Statistical Society, 1965.

    """

    def __init__(
        self,
        distance: Distance,
        X: npt.NDArray[np.float64],
        mu: npt.NDArray[np.float64],
        Z: npt.NDArray[np.float64] | None = None,
        nu: npt.NDArray[np.float64] | None = None,
        psi: float | list[float] | npt.NDArray[np.float64] | None = None,
        phi: float | None = None,
        min_weight: float | list[float] | npt.NDArray[np.float64] = 0.0,
        constrain_mean_weight_to: float | None = 1.0,
        settings: OptimizationSettings | None = None,
    ) -> None:
        """Create a Rake object."""
        M, p = X.shape
        assert mu.shape[0] == p

        if Z is not None:
            if Z.shape[0] != M:
                raise ValueError(
                    "Dimension mismatch: Z must have same number of rows as X."
                )

            if nu is None:
                raise ValueError(
                    "Must specify `nu` when approximately balancing covariates."
                )

            if nu.shape[0] != Z.shape[1]:
                raise ValueError(
                    "Dimension mismatch: nu must have length equal to the number of columns in Z."
                )

            if psi is None:
                psi = 1.5 / np.sqrt(M)

            self.B: npt.NDArray[np.float64] | None = (1 / M) * Z.T
            self.c: npt.NDArray[np.float64] | None = nu
            self.psi: float | list[float] | npt.NDArray[np.float64] | None = psi
        else:
            self.B = None
            self.c = None
            self.psi = None

        if constrain_mean_weight_to is not None:
            if constrain_mean_weight_to <= 0:
                raise ValueError("constrain_mean_weight_to must be positive or None.")

            # Add a column of ones to X
            self.X: npt.NDArray[np.float64] = np.hstack((X, np.ones((M, 1))))
            # Append the new mean constraint value to mu
            self.mu: npt.NDArray[np.float64] = np.append(mu, constrain_mean_weight_to)
            self.covariates_balanced: int = p + 1
        else:
            self.X = X
            self.mu = mu
            self.covariates_balanced = p

        self.distance = distance
        self.dimension: int = M
        self.phi = phi
        self.min_weight = min_weight
        if settings is None:
            self.settings: OptimizationSettings = OptimizationSettings()
        else:
            self.settings = settings

        if phi is not None:
            self.phase1_solver: PhaseISolver = (
                EqualityWithBoundsAndNormConstraintSolver(
                    phi=M * phi,
                    A=(1 / M) * self.X.T,
                    b=self.mu,
                    lb=min_weight,
                    B=self.B,
                    c=self.c,
                    psi=self.psi,
                    settings=self.settings,
                )
            )
        elif self.B is not None:
            assert self.c is not None
            assert self.psi is not None
            self.phase1_solver = EqualityWithBoundsAndImbalanceConstraintSolver(
                B=self.B,
                c=self.c,
                psi=self.psi,
                A=(1 / M) * self.X.T,
                b=self.mu,
                lb=min_weight,
                settings=self.settings,
            )
        else:
            self.phase1_solver = EqualityWithBoundsSolver(
                A=(1 / M) * self.X.T,
                b=self.mu,
                lb=min_weight,
                settings=self.settings,
            )

    @property
    def A(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Matrix representing equality constraints."""
        return (1 / self.dimension) * self.X.T

    @property
    def b(self) -> npt.NDArray[np.float64]:
        """Right-hand side of equality constraints."""
        return self.mu

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

        if not isinstance(
            self.phase1_solver,
            EqualityWithBoundsSolver
            | EqualityWithBoundsAndImbalanceConstraintSolver
            | EqualityWithBoundsAndNormConstraintSolver,
        ):
            raise ValueError(
                "phase1_solver must be an EqualityWithBoundsSolver or "
                "EqualityWithBoundsAndImbalanceConstraintSolver or"
                "EqualityWithBoundsAndNormConstraintSolver"
            )

        return self.phase1_solver.svd_A()

    def update_phi(self, phi: float | None = None) -> None:
        """Update phi, mostly for EfficientFrontier."""
        self.phi = phi
        if phi is not None:
            self.phase1_solver = EqualityWithBoundsAndNormConstraintSolver(
                phi=self.dimension * phi,
                A=(1 / self.dimension) * self.X.T,
                b=self.mu,
                lb=self.min_weight,
                B=self.B,
                c=self.c,
                psi=self.psi,
                settings=self.settings,
            )
        elif self.B is not None:
            assert self.c is not None
            assert self.psi is not None
            self.phase1_solver = EqualityWithBoundsAndImbalanceConstraintSolver(
                B=self.B,
                c=self.c,
                psi=self.psi,
                A=(1 / self.dimension) * self.X.T,
                b=self.mu,
                lb=self.min_weight,
                settings=self.settings,
            )
        else:
            self.phase1_solver = EqualityWithBoundsSolver(
                A=(1 / self.dimension) * self.X.T,
                b=self.mu,
                lb=self.min_weight,
                settings=self.settings,
            )

    @property
    def num_eq_constraints(self) -> int:
        """Count equality constraints."""
        return self.X.shape[1]

    @property
    def num_ineq_constraints(self) -> int:
        """Count inequality constraints."""
        n = self.dimension

        if self.B is not None:
            n += 2 * self.B.shape[0]

        if self.phi is not None:
            n += 1

        return n

    def initialize_barrier_parameter(self, x0: npt.NDArray[np.float64]) -> float:
        """Initialize barrier parameter."""
        # Initialize t as the number of inequality constraints divided by an estimate of
        # the suboptimality of w. Since the distance metric is always non-negative, the
        # sub-optimality is at most D(w, v). Always do at least 1 step so we get the
        # Lagrange multipliers.
        t1 = self.num_ineq_constraints / max(
            self.settings.outer_tolerance, self.distance.evaluate(x0)
        )

        return max(t1, super().initialize_barrier_parameter(x0))

    def calculate_newton_step(
        self, x: npt.NDArray[np.float64], t: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Calculate Newton step.

        Calculates Newton step for the "inner" problem:
          minimize   ft(w) := t * D(w, v)
                              - log(1 - (1 / (M * phi)) \| w \|_2^2)
                              - \sum_i log(w_i)
          subject to (1 / M) * X^T * w = \mu.

        Parameters
        ----------
         w : vector
            Current estimate.
         t : float
            Barrier parameter.

        Returns
        -------
         delta_w : vector
            Newton step.
         nu : vector
            Lagrange multiplier associated with equality constraints.

        Notes
        -----
        The Newton step, delta_w, is the solution of the system:
           _       _   _       _     _         _
          | H   A^T | | delta_w |   | - grad_ft |
          | A    0  | |   nu    | = |      0    |
           -       -   -       -     -         -
        where H is the Hessian of ft evaluated at w, A = (1/M) * X^T, grad_f is the
        gradient of ft evaluated at w, and nu is the Lagrange multiplier associated with
        the equality constraints. We use `solve_kkt_system` to solve this system in
        O(p^3 + p^2 * M) time.

        """
        eta_inverse: npt.NDArray[np.float64] = self._hessian_ft_diagonal_inverse(x, t)
        if self.B is not None and self.phi is not None:
            assert self.c is not None
            Bx_minus_c = self.B @ x - self.c
            kappa_B_pos: npt.NDArray[np.float64] = self._hessian_ft_kappa_pos(
                x, Bx_minus_c
            )
            kappa_B_neg: npt.NDArray[np.float64] = self._hessian_ft_kappa_neg(
                x, Bx_minus_c
            )
            kappa_phi: npt.NDArray[np.float64] = self._hessian_ft_rank_one(x)

            def A_solve_B_pos(  # noqa: N802
                b: npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
                return solve_rank_p_update(
                    b,
                    kappa=kappa_B_pos,
                    A_solve=solve_diagonal_eta_inverse,
                    eta_inverse=eta_inverse,
                )

            def A_solve_B_neg(  # noqa: N802
                b: npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
                return solve_rank_p_update(b, kappa=kappa_B_neg, A_solve=A_solve_B_pos)

            return solve_kkt_system(
                A=self.A,
                g=-self.gradient_barrier(x, t),
                hessian_solve=solve_rank_one_update,
                kappa=kappa_phi,
                A_solve=A_solve_B_neg,
            )

        if self.B is not None and self.phi is None:
            assert self.c is not None
            Bx_minus_c = self.B @ x - self.c
            kappa_B_pos = self._hessian_ft_kappa_pos(x, Bx_minus_c)
            kappa_B_neg = self._hessian_ft_kappa_neg(x, Bx_minus_c)

            def A_solve_nested(  # noqa: N802
                b: npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
                return solve_rank_p_update(
                    b,
                    kappa=kappa_B_pos,
                    A_solve=solve_diagonal_eta_inverse,
                    eta_inverse=eta_inverse,
                )

            return solve_kkt_system(
                A=self.A,
                g=-self.gradient_barrier(x, t),
                hessian_solve=solve_rank_p_update,
                kappa=kappa_B_neg,
                A_solve=A_solve_nested,
            )

        if self.B is None and self.phi is not None:
            kappa_phi = self._hessian_ft_rank_one(x)
            return solve_kkt_system(
                A=self.A,
                g=-self.gradient_barrier(x, t),
                hessian_solve=solve_rank_one_update,
                kappa=kappa_phi,
                A_solve=solve_diagonal_eta_inverse,
                eta_inverse=eta_inverse,
            )

        return solve_kkt_system(
            A=self.A,
            g=-self.gradient_barrier(x, t),
            hessian_solve=solve_diagonal_eta_inverse,
            eta_inverse=eta_inverse,
        )

    def btls_keep_feasible(
        self, x: npt.NDArray[np.float64], delta_x: npt.NDArray[np.float64]
    ) -> float:
        r"""Make sure x + btls_s * delta_x stays strictly feasible.

        In our case, we want a btls_s such that:
            x + btls_s * delta_x > min_weight, and
            \| x + btls_s * delta_x \|_2^2 <= M * phi

        For the first constraint, since x is strictly feasible, and since btls_s > 0,
        this is only a concern for entries having delta_x[i] < 0, for which we want:
            btls_s < (x[i] - min_weight) / -delta_x[i].
        We want this for all i having delta_x[i] < 0, so we set btls_s as the minimum of
        these quantities. If this minimum > 1, we just set btls_s = 1.0 If all entries
        of delta_x > 0, we just set btls_s = 1.

        The last constraint is convex quadratic in btls_s, so the acceptable btls_s lie
        between the two roots. It is straightforward to show that if
            \| x \|_2^2 <= M * phi,
        then the lower root is <= 0, and since the step size must be positive there
        is no need to enforce a lower bound.

        """
        M = self.dimension

        mask = delta_x < 0
        theta1 = np.inf
        min_weight = self.min_weight
        if np.any(mask):
            if isinstance(min_weight, list | tuple | np.ndarray):
                theta1 = np.min((x[mask] - min_weight[mask]) / -delta_x[mask])
            else:
                theta1 = np.min((x[mask] - min_weight) / -delta_x[mask])
        thetas = [theta1]

        B = self.B
        if B is not None:
            c = self.c
            psi = self.psi
            assert c is not None
            assert psi is not None
            Bx_minus_c = B @ x - c
            B_delta_x = B @ delta_x

            mask2 = B_delta_x < 0
            theta2 = np.inf
            if np.any(mask2):
                if isinstance(psi, list | tuple | np.ndarray):
                    theta2 = np.min(
                        (Bx_minus_c[mask2] + psi[mask2]) / -B_delta_x[mask2]
                    )
                else:
                    theta2 = np.min((Bx_minus_c[mask2] + psi) / -B_delta_x[mask2])

            mask3 = B_delta_x > 0
            theta3 = np.inf
            if np.any(mask3):
                if isinstance(psi, list | tuple | np.ndarray):
                    theta3 = np.min(
                        -(Bx_minus_c[mask3] - psi[mask3]) / B_delta_x[mask3]
                    )
                else:
                    theta3 = np.min(-(Bx_minus_c[mask3] - psi) / B_delta_x[mask3])

            thetas.append(theta2)
            thetas.append(theta3)

        phi = self.phi
        if phi is not None:
            quad_a = np.dot(delta_x, delta_x)
            quad_b = 2.0 * np.dot(x, delta_x)
            quad_c = np.dot(x, x) - M * phi
            theta4 = (-quad_b + np.sqrt(quad_b * quad_b - 4.0 * quad_a * quad_c)) / (
                2.0 * quad_a
            )

            thetas.append(theta4)

        return min(thetas)

    def evaluate_objective(self, x: npt.NDArray[np.float64]) -> float:
        """Calculate f0 at x."""
        return self.distance.evaluate(x)

    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate gradient of ft at w."""
        return self.distance.gradient(x)

    def hessian_multiply(
        self, x: npt.NDArray[np.float64], t: float, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Multiply H * y.

        Our Hessian is diag(eta) + zeta * zeta^T, so H * y is eta * y + (zeta^T * y) *
        zeta.

        """
        eta = self._hessian_ft_diagonal(x, t)
        zeta = self._hessian_ft_rank_one(x)
        Hy = eta * y + np.dot(zeta, y) * zeta

        if self.B is not None:
            assert self.c is not None
            Bx_minus_c = self.B @ x - self.c
            kappa_pos = self._hessian_ft_kappa_pos(x, Bx_minus_c)
            kappa_neg = self._hessian_ft_kappa_neg(x, Bx_minus_c)
            Hy += kappa_pos @ (kappa_pos.T @ y)
            Hy += kappa_neg @ (kappa_neg.T @ y)

        return Hy

    def _hessian_ft_diagonal(
        self, x: npt.NDArray[np.float64], t: float
    ) -> npt.NDArray[np.float64]:
        """Calculate diagonal component of Hessian of ft at w."""
        d = t * self.distance.hessian_diagonal(x) + np.square(
            1.0 / (x - self.min_weight)
        )
        if self.phi is None:
            return d

        M_phi_den = self.dimension * self.phi - np.dot(x, x)
        return d + np.full_like(x, 2.0 / M_phi_den)

    def _hessian_ft_diagonal_inverse(
        self, x: npt.NDArray[np.float64], t: float
    ) -> npt.NDArray[np.float64]:
        """Calculate diagonal component of Hessian of ft at w."""
        if self.phi is None:
            tx2 = self.distance.hessian_diagonal(x) * np.square(
                np.sqrt(t) * (x - self.min_weight)
            )
        else:
            M_phi_den = self.dimension * self.phi - np.dot(x, x)
            tx2 = np.square(
                np.sqrt(t * self.distance.hessian_diagonal(x) + 2.0 / M_phi_den)
                * (x - self.min_weight)
            )

        return np.square((x - self.min_weight) / np.sqrt(1.0 + tx2))

    def _hessian_ft_rank_one(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate rank one component of Hessian of ft at w."""
        if self.phi is None:
            return np.zeros_like(x)

        M_phi_den = self.dimension * self.phi - np.dot(x, x)
        return x * (2.0 / M_phi_den)

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
        r"""Calculate the vector of (inequality) constraints, fi(x) <= 0.

        Our constraints are x >= min_weight and 1 - (1 / M*\phi) * \| x \|_2^2 >= 0, or
           -x + min_weight <= 0
           -1 + (1 / M*\phi) * \| x \|_2^2 <= 0

        """
        cons = -x + self.min_weight

        if self.B is not None:
            assert self.c is not None
            assert self.psi is not None
            Bx_minus_c = self.B @ x - self.c
            cons = np.concatenate(
                [
                    cons,
                    -Bx_minus_c - self.psi,
                    Bx_minus_c - self.psi,
                ]
            )

        if self.phi is not None:
            cons = np.append(
                cons, (1.0 / (self.dimension * self.phi)) * np.dot(x, x) - 1.0
            )

        return cons

    def grad_constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate the gradients of constraints, fi(x) <= 0.

        If phi is None, G = -I. Otherwise, G is -I, with a row at the bottom of
        (2 / (M * phi)) * x

        """
        G = -np.eye(self.dimension)

        B = self.B
        if B is not None:
            G = np.vstack([G, -B, B])

        phi = self.phi
        if phi is not None:
            G = np.vstack([G, (2.0 / (self.dimension * phi)) * x])

        return G

    def grad_constraints_multiply(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate grad_constraints(x) @ y.

        If phi is None, this is just -y. Otherwise, the first entries are -y, and the
        last entry is (2 / (M * phi)) * <x, y>.

        """
        assert len(y) == self.dimension
        Gy = -y

        if self.B is not None:
            By = self.B @ y
            Gy = np.concatenate([Gy, -By, By])

        if self.phi is not None:
            Gy = np.append(Gy, (2.0 / (self.dimension * self.phi)) * np.dot(x, y))

        return Gy

    def grad_constraints_transpose_multiply(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate grad_constraints(x).T @ y."""
        assert len(y) == self.num_ineq_constraints
        M = self.dimension
        GTy = -y[0:M]

        B = self.B
        if B is not None:
            p2 = B.shape[0]
            GTy += B.T @ (y[(M + p2) : (M + 2 * p2)] - y[M : (M + p2)])

        phi = self.phi
        if phi is not None:
            GTy += ((2.0 * y[-1]) / (M * phi)) * x

        return GTy

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
        This implementation only works if lmbda and nu are optimal.

        """
        return (
            self.evaluate_objective(x_star)
            + np.dot(lmbda, self.constraints(x_star))
            + np.dot(nu, self.A @ x_star - self.b)
        )


class JointCalibrator(Rake):
    r"""Solve optimization problem.

    Class for solving problems of the form:
           minimize    D([w1, w2], [v1, v2])
           subject to  (1/M1) * X1^T * w1 = mu1                                               (external validity)
                       (1/M2) * X2^T * w2 = mu2                                               (external validity)
                       (1/M1) * X3^T * w1 = (1/M2) * X4^T * w2                                (internal validity)
                       \| (1/M1) * Z1^T * w1 - nu1 \|_\infty <= psi1                          (external validity)
                       \| (1/M2) * Z2^T * w2 - nu2 \|_\infty <= psi2                          (external validity)
                       \| (1/M2) * Z4^T * w2 - (1/M1) * Z3^T * w1 \|_\infty <= psi3           (internal validity)
                       (1/(M1 + M2)) * (\| w1 \|_2^2 + \| w2 \|_2^2) <= phi
                       w1 >= min_weight1
                       w2 >= min_weight2

    The use case is calibrating weights for treatment and control groups at the same
    time, to achieve internal and external validity.

    Parameters
    ----------
     X1, X2, X3, X4 : npt.NDArray[np.float64], optional
        Matrices for achieving exact covariate balance.
     mu1, mu2 : npt.NDArray[np.float64], optional
        Population means corresponding to X1, X2, resp.
     Z1, Z2, Z3, Z4 : npt.NDArray[np.float64], optional
        Matrices for constraining covariate imbalance.
     nu1, nu2 : npt.NDArray[np.float64], optional
        Population means corresponding to Z1, Z2, resp.
     psi1, psi2, psi3 : float or list[float], optional
        Constraints on covariate imbalances.
     phi : float
         Constraint on mean squared weight.
     min_weight1, min_weight2 : float or list[float], optional
         Lower bound on weights. Defaults to 0.
     constrain_mean_weight1_to, constrain_mean_weight2_to : float or None, default=1
         If not None, add a constraint that the average weight must equal
         `constrain_mean_weight_to`, which must be strictly positive. The default is an
         average weight of 1. Another common option is for the average weight to equal
         the average baseline weight. If no such constraint is desired, manually specify
         `constrain_mean_weight_to=None`.
     settings : OptimizationSettings
        Optimization settings.

    Notes
    -----
    This is really just a wrapper around Rake. Let:
    - w = [w1; w2]
    - v = [v1; v2],
    - M = M1 + M2
             -                                        -
            |   (M / M1) * X1^T             0          |
    - X^T = |          0               (M / M2) * X2^T |
            |  -(M / M1) * X3^T        (M / M2) * X4^T |
             -                                        -
    - mu = [mu1; mu2; 0]
             -                                        -
            |   (M / M1) * Z1^T             0          |
    - Z^T = |          0               (M / M2) * Z2^T |
            |  -(M / M1) * Z3^T        (M / M2) * Z4^T |
             -                                        -
    - nu = [nu1; nu2; 0]
    - psi = a vector with q1 entries of psi1, followed by q2 entries of psi2, followed
      by q3 entries of psi3, where q1, q2, and q3 are the numbers of columns in Z1, Z2,
      and Z3, respectively.
    - min_weight = a vector with M1 entries of min_weight1, followed by M2 entries of
      min_weight2, where M1 and M2 are the lengths of w1 and w2, respectively.

    Then the problem is equivalent to:
           minimize    D(w, v)
           subject to  (1/M) * X^T * w = mu                   (exact balance)
                       -psi <= (1/M) * Z^T * w - nu <= psi    (approximate balance)
                       (1/M) * \| w \|_2^2 <= phi             (variance)
                       w >= min_weight                        (positivity)

    Many of the constraints in the above optimization problem are optional. The user may
    wish to approximately balance some covariates for internal validity only, in which
    case they would specify Z3 and Z4 only, not X1, X2, X3, X4, Z1, or Z2. Then Z^T
    would just be:
          -                                        -
         |  -(M / M1) * Z3^T        (M / M2) * Z4^T |
          -                                        -

    Due to a technical limitation in Rake, we *require* at least one equality constraint
    in the optimization problem. Thankfully, we typically want to constrain the mean
    weight to be 1, which is represented as (1/M) 1^T w = 1, where 1^T is a vector of
    all ones. In our case, it makes sense to constrain w1 and w2 separately, e.g.
    (1 / M1) 1^T w1 = 1 and (1 / M2) 1^T w2 = 1. If the user does not specify any of the
    X's, we can just use these two constraints to satisfy Rake's requirement. Otherwise,
    corresponding constraints should be prepended as the first rows of X1 and X2,
    respectively, e.g. X1 = np.vstack([np.ones((M1,)), X1]) and
    mu1 = np.concatenate([[constrain_mean_weight1_to], mu1]).

    """

    def __init__(
        self,
        distance: Distance,
        X1: npt.NDArray[np.float64] | None = None,
        X2: npt.NDArray[np.float64] | None = None,
        X3: npt.NDArray[np.float64] | None = None,
        X4: npt.NDArray[np.float64] | None = None,
        mu1: npt.NDArray[np.float64] | None = None,
        mu2: npt.NDArray[np.float64] | None = None,
        Z1: npt.NDArray[np.float64] | None = None,
        Z2: npt.NDArray[np.float64] | None = None,
        Z3: npt.NDArray[np.float64] | None = None,
        Z4: npt.NDArray[np.float64] | None = None,
        nu1: npt.NDArray[np.float64] | None = None,
        nu2: npt.NDArray[np.float64] | None = None,
        psi1: float | list[float] | npt.NDArray[np.float64] | None = None,
        psi2: float | list[float] | npt.NDArray[np.float64] | None = None,
        psi3: float | list[float] | npt.NDArray[np.float64] | None = None,
        phi: float | None = None,
        min_weight1: float | list[float] | npt.NDArray[np.float64] = 0.0,
        min_weight2: float | list[float] | npt.NDArray[np.float64] = 0.0,
        constrain_mean_weight1_to: float | None = 1.0,
        constrain_mean_weight2_to: float | None = 1.0,
        settings: OptimizationSettings | None = None,
    ) -> None:
        M1 = M2 = None
        p1 = p2 = p3 = None
        q1 = q2 = q3 = None
        if X1 is not None:
            M1, p1 = X1.shape

            if mu1 is None:
                raise ValueError("Must specify both or neither of X1, mu1")

            if len(mu1) != p1:
                raise ValueError("Dimension mismatch")

        if X2 is not None:
            M2, p2 = X2.shape

            if mu2 is None:
                raise ValueError("Must specify both or neither of X2, mu2")

            if len(mu2) != p2:
                raise ValueError("Dimension mismatch")

        if X3 is not None:
            if M1 is not None and X3.shape[0] != M1:
                raise ValueError("Dimension mismatch")

            M1, p3 = X3.shape

        if X4 is not None:
            if M2 is not None and X4.shape[0] != M2:
                raise ValueError("Dimension mismatch")

            if p3 is not None and X4.shape[1] != p3:
                raise ValueError("Dimension mismatch")

            M2, p3 = X4.shape

        if not ((X3 is None and X4 is None) or (X3 is not None and X4 is not None)):
            raise ValueError("Must specify both or neither of X3, X4")

        if Z1 is not None:
            if M1 is not None and Z1.shape[0] != M1:
                raise ValueError("Dimension mismatch")

            M1, q1 = Z1.shape

            if nu1 is None:
                raise ValueError("Must specify both or neither of Z1, nu1")

            if len(nu1) != q1:
                raise ValueError("Dimension mismatch")

            if psi1 is None:
                psi1 = 1.5 / np.sqrt(M1)

            if isinstance(psi1, list | tuple | np.ndarray) and len(psi1) != q1:
                raise ValueError("Dimension mismatch")

        if Z2 is not None:
            if M2 is not None and Z2.shape[0] != M2:
                raise ValueError("Dimension mismatch")

            M2, q2 = Z2.shape

            if nu2 is None:
                raise ValueError("Must specify both or neither of Z2, nu2")

            if len(nu2) != q2:
                raise ValueError("Dimension mismatch")

            if psi2 is None:
                psi2 = 1.5 / np.sqrt(M2)

            if isinstance(psi2, list | tuple | np.ndarray) and len(psi2) != q2:
                raise ValueError("Dimension mismatch")

        if Z3 is not None:
            if M1 is not None and Z3.shape[0] != M1:
                raise ValueError("Dimension mismatch")

            M1, q3 = Z3.shape

            if Z4 is None:
                raise ValueError("Must specify both or neither of Z3, Z4")

            if M2 is not None and Z4.shape[0] != M2:
                raise ValueError("Dimension mismatch")

            if Z4.shape[1] != q3:
                raise ValueError("Dimension mismatch")

            M2, q3 = Z4.shape

            if psi3 is None:
                psi3 = 1.5 * np.sqrt(1 / M1 + 1 / M2)

            if isinstance(psi3, list | tuple | np.ndarray) and len(psi3) != q3:
                raise ValueError("Dimension mismatch")

        if M1 is None:
            if isinstance(min_weight1, list | tuple | np.ndarray):
                M1 = len(min_weight1)
            else:
                raise ValueError("Cannot infer weight dimension")
        elif (
            isinstance(min_weight1, list | tuple | np.ndarray)
            and len(min_weight1) != M1
        ):
            raise ValueError("Dimension mismatch")

        if M2 is None:
            if isinstance(min_weight2, list | tuple | np.ndarray):
                M2 = len(min_weight2)
            else:
                raise ValueError("Cannot infer weight dimension")
        elif (
            isinstance(min_weight2, list | tuple | np.ndarray)
            and len(min_weight2) != M2
        ):
            raise ValueError("Dimension mismatch")

        M = M1 + M2

        if not ((X1 is None and mu1 is None) or (X1 is not None and mu1 is not None)):
            raise ValueError("Must specify both or neither of X1, mu1")

        if not ((X2 is None and mu2 is None) or (X2 is not None and mu2 is not None)):
            raise ValueError("Must specify both or neither of X2, mu2")

        if not ((X3 is None and X4 is None) or (X3 is not None and X4 is not None)):
            raise ValueError("Must specify both or neither of X3, X4")

        if not (
            (Z1 is None and nu1 is None and psi1 is None)
            or (Z1 is not None and nu1 is not None and psi1 is not None)
        ):
            raise ValueError("Must specify all or none of Z1, nu1, psi1")

        if not (
            (Z2 is None and nu2 is None and psi2 is None)
            or (Z2 is not None and nu2 is not None and psi2 is not None)
        ):
            raise ValueError("Must specify all or none of Z2, nu2, psi2")

        if not (
            (Z3 is None and Z4 is None and psi3 is None)
            or (Z3 is not None and Z4 is not None and psi3 is not None)
        ):
            raise ValueError("Must specify all or none of Z3, Z4, psi3")

        if constrain_mean_weight1_to is not None:
            if X1 is None:
                X1 = np.ones((M1, 1))
                mu1 = np.array([constrain_mean_weight1_to])
                p1 = 1
            else:
                assert p1 is not None
                assert mu1 is not None
                X1 = np.hstack([np.ones((M1, 1)), X1])
                mu1 = np.concatenate([[constrain_mean_weight1_to], mu1])
                p1 += 1

        if constrain_mean_weight2_to is not None:
            if X2 is None:
                X2 = np.ones((M2, 1))
                mu2 = np.array([constrain_mean_weight2_to])
                p2 = 1
            else:
                assert p2 is not None
                assert mu2 is not None
                X2 = np.hstack([np.ones((M2, 1)), X2])
                mu2 = np.concatenate([[constrain_mean_weight2_to], mu2])
                p2 += 1

        if X1 is None and X2 is None and X3 is None and X4 is None:
            raise ValueError("We need at least one equality constraint.")

        # Build X and mu
        XT_blocks, mu_blocks = [], []
        if X1 is not None:
            assert p1 is not None
            assert mu1 is not None
            XT_blocks.append(np.hstack([(M / M1) * X1.T, np.zeros((p1, M2))]))
            mu_blocks.append(mu1)

        if X2 is not None:
            assert p2 is not None
            assert mu2 is not None
            XT_blocks.append(np.hstack([np.zeros((p2, M1)), (M / M2) * X2.T]))
            mu_blocks.append(mu2)

        if X3 is not None and X4 is not None:
            assert p3 is not None
            XT_blocks.append(np.hstack([-(M / M1) * X3.T, (M / M2) * X4.T]))
            mu_blocks.append(np.zeros(p3))

        X = np.vstack(XT_blocks).T
        mu = np.concatenate(mu_blocks)

        # Build Z, nu, psi
        ZT_blocks, nu_blocks, psi_blocks = [], [], []
        if Z1 is not None:
            assert q1 is not None
            assert nu1 is not None
            assert psi1 is not None
            ZT_blocks.append(np.hstack([(M / M1) * Z1.T, np.zeros((q1, M2))]))
            nu_blocks.append(nu1)

            if isinstance(psi1, list | tuple | np.ndarray):
                psi_blocks.append(np.asarray(psi1))
            else:
                psi_blocks.append(np.full(q1, psi1))

        if Z2 is not None:
            assert q2 is not None
            assert nu2 is not None
            ZT_blocks.append(np.hstack([np.zeros((q2, M1)), (M / M2) * Z2.T]))
            nu_blocks.append(nu2)

            if isinstance(psi2, list | tuple | np.ndarray):
                psi_blocks.append(np.asarray(psi2))
            else:
                psi_blocks.append(np.full(q2, psi2))

        if Z3 is not None and Z4 is not None:
            assert q3 is not None
            ZT_blocks.append(np.hstack([-(M / M1) * Z3.T, (M / M2) * Z4.T]))
            nu_blocks.append(np.zeros(q3))

            if isinstance(psi3, list | tuple | np.ndarray):
                psi_blocks.append(np.asarray(psi3))
            else:
                psi_blocks.append(np.full(q3, psi3))

        if len(ZT_blocks) > 0:
            Z = np.vstack(ZT_blocks).T
            nu = np.concatenate(nu_blocks)
            psi = np.concatenate(psi_blocks)
        else:
            Z = None
            nu = None
            psi = None

        # Build min_weight
        min_weight_blocks = []
        if isinstance(min_weight1, list | tuple | np.ndarray):
            min_weight_blocks.append(min_weight1)
        else:
            min_weight_blocks.append(np.full(M1, min_weight1))

        if isinstance(min_weight2, list | tuple | np.ndarray):
            min_weight_blocks.append(min_weight2)
        else:
            min_weight_blocks.append(np.full(M2, min_weight2))

        min_weight = np.concatenate(min_weight_blocks)

        # Call Rake
        super().__init__(
            distance=distance,
            X=X,
            mu=mu,
            Z=Z,
            nu=nu,
            psi=psi,
            phi=phi,
            min_weight=min_weight,
            constrain_mean_weight_to=None,  # Already handled above
            settings=settings,
        )
