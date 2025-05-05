"""Solve optimization problem."""

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from .distance_metrics import Distance
from .numerical_helpers import (
    solve_diagonal_plus_rank_one,
    solve_kkt_system,
)
from .optimization import (
    InteriorPointMethodSolver,
    OptimizationSettings,
)
from .phase1solvers import (
    EqualitySolver,
    EqualityWithBoundsAndNormConstraintSolver,
    EqualityWithBoundsSolver,
)


class Rake(InteriorPointMethodSolver):
    r"""Solve optimization problem.

    Class for solving problems of the form:
           minimize    D(w, v)
           subject to  (1/M) * X^T * w = \mu
                       (1/M) * \| w \|_2^2 <= \phi
                        w >= 0,
    with an optional constraint on the mean of weights.

    Parameters
    ----------
     distance : Distance
         Distance object representing distance from baseline weights.
     X : npt.NDArray[np.float64]
         Matrix of covariates for which we desire exact balance.
     mu : npt.NDArray[np.float64]
         Vector of mean covariate values in the target population.
     phi : float
         Constraint on mean squared weight.
     constrain_mean_weight_to : float or None, default=1
         If not None, add a constraint that the average weight must equal
         `constrain_mean_weight_to`, which must be strictly positive. The default is an
         average weight of 1. Another common option is for the average weight to equal
         the average baseline weight. If no such constraint is desired, manually specify
         `constrain_mean_weight_to=None`.

    """

    def __init__(
        self,
        distance: Distance,
        X: npt.NDArray[np.float64],
        mu: npt.NDArray[np.float64],
        phi: float,
        constrain_mean_weight_to: Optional[float] = 1.0,
        settings: Optional[OptimizationSettings] = None,
    ) -> None:
        """Create a Rake object."""
        M, p = X.shape
        assert mu.shape[0] == p

        if constrain_mean_weight_to is not None:
            if constrain_mean_weight_to <= 0:
                raise ValueError("constrain_mean_weight_to must be positive or None.")

            # Add a column of ones to X
            self.X = np.column_stack((X, np.ones(M)))
            # Append the new mean constraint value to mu
            self.mu = np.append(mu, constrain_mean_weight_to)
            self.covariates_balanced = p + 1
        else:
            self.X = X
            self.mu = mu
            self.covariates_balanced = p

        self.distance = distance
        self.dimension = M
        self.phi = phi
        if settings is None:
            self.settings: OptimizationSettings = OptimizationSettings()
        else:
            self.settings = settings

        self.phase1_solver = EqualityWithBoundsAndNormConstraintSolver(
            phi=M * phi,
            settings=self.settings,
            phase1_solver=EqualityWithBoundsSolver(
                settings=self.settings,
                phase1_solver=EqualitySolver(
                    A=(1 / M) * self.X.T,
                    b=self.mu,
                    settings=self.settings,
                ),
            ),
        )

    @property
    def num_eq_constraints(self) -> int:
        """Count equality constraints."""
        return self.X.shape[1]

    @property
    def num_ineq_constraints(self) -> int:
        """Count inequality constraints."""
        return self.dimension + 1

    def initialize_barrier_parameter(self, x0: npt.NDArray[np.float64]) -> float:
        """Initialize barrier parameter."""
        # Initialize t as the number of inequality constraints divided by an estimate of
        # the suboptimality of w. Since the distance metric is always non-negative, the
        # sub-optimality is at most D(w, v). Always do at least 1 step so we get the
        # Lagrange multipliers.
        return max(
            1.0,
            self.num_ineq_constraints
            / max(self.settings.outer_tolerance, self.distance.evaluate(x0)),
        )

    def calculate_newton_step(
        self, x: npt.NDArray[np.float64], t: float
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
        return solve_kkt_system(
            A=(1.0 / self.dimension) * self.X.T,
            g=-self.gradient_barrier(x, t),
            hessian_solve=solve_diagonal_plus_rank_one,
            eta=self._hessian_ft_diagonal(x, t),
            zeta=self._hessian_ft_rank_one(x),
        )

    def btls_keep_feasible(
        self, x: npt.NDArray[np.float64], delta_x: npt.NDArray[np.float64]
    ) -> float:
        r"""Make sure x + btls_s * delta_x stays strictly feasible.

        In our case, we want a btls_s such that:
            x + btls_s * delta_x > 0, and
            \| x + btls_s * delta_x \|_2^2 <= M * phi

        For the first constraint, since x is strictly feasible, and since btls_s > 0,
        this is only a concern for entries having delta_x[i] < 0, for which we want:
            btls_s < x[i] / -delta_x[i].
        We want this for all i having delta_x[i] < 0, so we set btls_s as the minimum of
        these quantities. If this minimum > 1, we just set btls_s = 1.0 If all entries
        of delta_x > 0, we just set btls_s = 1.

        The last constraint is convex quadratic in btls_s, so the acceptable btls_s lie
        between the two roots. It is straightforward to show that if
            \| x \|_2^2 <= M * phi,
        then the lower root is <= 0, and since the step size must be positive there
        is no need to enforce a lower bound.

        """
        beta = self.settings.backtracking_beta
        M = self.dimension

        quad_a = np.dot(delta_x, delta_x)
        quad_b = 2.0 * np.dot(x, delta_x)
        quad_c = np.dot(x, x) - M * self.phi
        btls_s_high = (-quad_b + np.sqrt(quad_b * quad_b - 4.0 * quad_a * quad_c)) / (
            2.0 * quad_a
        )

        btls_s = min(
            1.0,
            beta * -np.max(np.where(delta_x < 0, x / delta_x, -np.inf)),
            beta * btls_s_high,
        )
        return btls_s

    def evaluate_objective(self, x: npt.NDArray[np.float64]) -> float:
        """Calculate f0 at x."""
        return self.distance.evaluate(x)

    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate gradient of ft at w."""
        return self.distance.gradient(x)

    def gradient_barrier(
        self, x: npt.NDArray[np.float64], t: float
    ) -> npt.NDArray[np.float64]:
        """Calculate gradient of ft at x."""
        den = 1.0 - (1.0 / (self.dimension * self.phi)) * np.dot(x, x)
        return (
            t * self.distance.gradient(x)
            - (2.0 / (den * self.dimension * self.phi)) * x
            - 1.0 / x
        )

    def hessian_multiply(
        self, x: npt.NDArray, t: float, y: npt.NDArray
    ) -> npt.NDArray[np.float64]:
        """Multiply H * y.

        Our Hessian is diag(eta) + zeta * zeta^T, so H * y is eta * y + (zeta^T * y) *
        zeta.

        """
        eta = self._hessian_ft_diagonal(x, t)
        zeta = self._hessian_ft_rank_one(x)
        return eta * y + np.dot(zeta, y) * zeta

    def _hessian_ft_diagonal(
        self, x: npt.NDArray[np.float64], t: float
    ) -> npt.NDArray[np.float64]:
        """Calculate diagonal component of Hessian of ft at w."""
        M = self.dimension
        den = 1.0 - (1.0 / (M * self.phi)) * np.dot(x, x)
        return (
            t * self.distance.hessian_diagonal(x)
            + np.full_like(x, 2.0 / (M * self.phi * den))
            + np.square(1.0 / x)
        )

    def _hessian_ft_rank_one(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate rank one component of Hessian of ft at w."""
        M = self.dimension
        den = 1.0 - (1.0 / (M * self.phi)) * np.dot(x, x)
        return x * (2.0 / (M * self.phi * den))

    def constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Calculate the vector of constraints, fi(x) <= 0.

        Our constraints are x >= 0 and 1 - (1 / M*\phi) * \| x \|_2^2 >= 0, or
           -x <= 0
           -1 + (1 / M*\phi) * \| x \|_2^2 <= 0

        """
        c = np.zeros((self.dimension + 1,))
        c[0:-1] = -x
        c[-1] = -1.0 + (1.0 / (self.dimension * self.phi)) * np.dot(x, x)
        return c

    def grad_constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate the gradients of constraints, fi(x) <= 0."""
        G = np.zeros((self.dimension + 1, self.dimension))
        G[0:-1, :] = -np.eye(self.dimension)
        G[-1, :] = (2.0 / (self.dimension * self.phi)) * x
        return G

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
            + np.dot(nu, (1 / self.dimension) * (self.X.T @ x_star) - self.mu)
        )

    def predictor_corrector(
        self, x: npt.NDArray[np.float64], t: float
    ) -> npt.NDArray[np.float64]:
        """Predictor step of a predictor/corrector method.

        Given the current weights from the previous centering step and the
        barrier parameter, this method calculates a predicted solution
        for the next barrier parameter using a linear approximation.

        Parameters
        ----------
        w : npt.NDArray[np.float64]
            Weights obtained from the previous centering step.
        t : float
            The current barrier parameter.

        Returns
        -------
        npt.NDArray[np.float64]
            The predicted weights for the next barrier parameter.

        """
        # 1. Calculate the gradient of ft at w
        grad_ft = self.gradient_barrier(x, t)

        # 2. Compute the Hessian diagonal and rank one component
        eta = self._hessian_ft_diagonal(x, t)
        zeta = self._hessian_ft_rank_one(x)

        # 3. Solve for dx/dt
        dx_dt, _ = solve_kkt_system(
            A=(1.0 / self.dimension) * self.X.T,
            g=-grad_ft,
            hessian_solve=solve_diagonal_plus_rank_one,
            eta=eta,
            zeta=zeta,
        )
        # try:
        #     np.testing.assert_allclose(
        #         (1 / self.dimension) * (self.X.T @ dx_dt)),
        #         np.zeros_like(self.mu),
        #     )
        # except AssertionError as e:
        #     print(e)

        # 4. Calculate the barrier multiplier mu from settings
        mu = self.settings.barrier_multiplier

        # 5. Use the predictor formula to calculate the next weights
        # x_star(t) + (dx^star(t) / dt) * (mu * t - t)
        predicted_weights = x + dx_dt * (mu * t - t)
        try:
            np.testing.assert_allclose(
                (1 / self.dimension) * (self.X.T @ predicted_weights) - self.mu,
                np.zeros_like(self.mu),
            )
        except AssertionError as e:
            print(e)
        return predicted_weights
