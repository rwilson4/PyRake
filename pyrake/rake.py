"""Solve optimization problem."""

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, minimize

from .distance_metrics import Distance
from .exceptions import (
    BacktrackingLineSearchError,
    CenteringStepError,
    InteriorPointMethodError,
    ProblemInfeasibleError,
)
from .numerical_helpers import (
    solve_kkt_system_hessian_diagonal_plus_rank_one,
)


@dataclass
class OptimizationSettings:
    """Optimization settings."""

    outer_tolerance: float = 1e-6
    barrier_multiplier: float = 10.0
    inner_tolerance: float = 1e-6
    inner_tolerance_soft: float = 1e-4
    max_inner_iterations: int = 100
    backtracking_alpha: float = 0.01
    backtracking_beta: float = 0.5
    backtracking_min_step: float = 1e-3


@dataclass
class NewtonResult:
    """Wrapper for the results of Newton's method.

    Parameters
    ----------
     solution : vector
        The solution.
     equality_multipliers, inequality_multipliers: vectors
        Lagrange multipliers for constraints.
     suboptimality : float
        Suboptimality of solution.
     nits : int
        Number of iterations before convergence.
     status : [0, 1]
        Solution status:
          0 : method completed successfully
          1 : method failed to converge to the desired tolerance, but achieved an
              acceptable tolerance
     message : str
          Summary of result.

    """

    solution: npt.NDArray[np.float64]
    equality_multipliers: npt.NDArray[np.float64]
    inequality_multipliers: npt.NDArray[np.float64]
    suboptimality: float
    nits: int
    status: Literal[0, 1]
    message: str


@dataclass
class InteriorPointMethodResult:
    """Wrapper for the results of the Interior Point Method.

    Parameters
    ----------
    solution : vector
        The optimal weights.
    objective_value : float
        Distance between optimal weights and baseline.
    dual_value : float
        Dual function, evaluated at optimal weights and Lagrange multipliers.
    equality_multipliers, inequality_multipliers: vectors
        Optimal Lagrange multipliers for constraints.
    suboptimality : float
        Suboptimality of solution.
    nits : int
        The number of outer iterations performed during the optimization
        process, indicating how many times the centering step was executed.
    inner_nits : List[int]
        Number of inner iterations performed during each centering step.
    status : int
        Solution status:
          0 : method completed successfully

    """

    solution: npt.NDArray[np.float64]
    objective_value: float
    dual_value: float
    equality_multipliers: npt.NDArray[np.float64]
    inequality_multipliers: npt.NDArray[np.float64]
    suboptimality: float
    nits: int
    inner_nits: List[int]
    status: Literal[0]


class Rake:
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
            self.settings = OptimizationSettings()
        else:
            self.settings = settings

    def solve(
        self, w0: Optional[npt.NDArray[np.float64]] = None
    ) -> InteriorPointMethodResult:
        r"""Solve optimization problem.

        Uses an interior point method with a logarithmic barrier penalty to
        solve:
           minimize    D(w, v)
           subject to  (1/M) * X^T * w = \mu
                       (1/M) * \| w \|_2^2 <= \phi
                        w >= 0,
        where D is a distance metric than penalizes deviations from baseline
        weights v.

        Parameters
        ----------
         w0 : vector
            Initial guess. If infeasible, a Phase I method will be used to find a
            feasible point close to w0.

        Returns
        -------
         w : vector
            The optimal weights.

        """
        w = self.solve_phase1(w0=w0)

        M = self.dimension
        t = 1.0
        num_steps = (
            int(
                np.ceil(
                    np.log((M + 1) / (t * self.settings.outer_tolerance))
                    / np.log(self.settings.barrier_multiplier)
                )
            )
            + 1
        )
        inner_nits = []
        for ii in range(num_steps):
            try:
                result = self.centering_step(w, t, last_step=(ii + 1 == num_steps))
                w = result.solution
                inner_nits.append(result.nits)
            except CenteringStepError as e:
                raise InteriorPointMethodError(
                    message="Centering step failed",
                    suboptimality=(M + 1) * self.settings.barrier_multiplier / t,
                    last_iterate=e.last_iterate,
                ) from e

            t *= self.settings.barrier_multiplier

        lambda_star = result.inequality_multipliers
        nu_star = result.equality_multipliers
        f = self.distance.evaluate(w)
        g = (
            f
            - np.dot(lambda_star[0:M], w)
            - lambda_star[M] * (1.0 - (1.0 / (M * self.phi)) * np.dot(w, w))
            + np.dot(nu_star, (1 / M) * self.X.T.dot(w) - self.mu)
        )
        return InteriorPointMethodResult(
            solution=w,
            objective_value=f,
            dual_value=g,
            equality_multipliers=nu_star,
            inequality_multipliers=lambda_star,
            suboptimality=(M + 1) * self.settings.barrier_multiplier / t,
            nits=num_steps,
            inner_nits=inner_nits,
            status=0,
        )

    def centering_step(
        self, w0: npt.NDArray[np.float64], t: float, last_step: bool
    ) -> NewtonResult:
        r"""Solve centering step.

        The centering step solves:
          minimize   ft(w) := t * D(w, v)
                              - log(1 - (1 / (M * phi)) \| w \|_2^2)
                              - \sum_i log(w_i)
          subject to (1 / M) * X^T * w = \mu.

        Parameters
        ----------
         w0 : vector
            Initial guess. Must be strictly feasible.
         t : float
            Barrier parameter.
         last_step: bool
             Indicates whether this is the last centering step. See Notes.

        Returns
        -------
         w : vector
            Solution to centering step.
         lambda_star : vector
            Lagrange multipliers for inequality constraints.
         nu_star : vector
            Lagrange multipliers for equality constraints.

        Notes
        -----
        Uses Newton's method with a feasible starting point. It isn't always possible to
        solve this to high precision, so we use a "soft" threshold as a fallback. If we
        can solve the problem to high precision, great; otherwise we're content if we
        have solved it to medium precision.

        The exception is on the very last step, where the overall suboptimality of the
        interior point method relies on the last centering step being solved to high
        precision. So this "soft" threshold does not apply in this last step.

        """
        w = w0.copy()
        M = self.dimension
        for nit in range(self.settings.max_inner_iterations):
            delta_w, nu_hat = self.calculate_newton_step(w, t)

            # Check for convergence
            lmbda = self._newton_decrement(w, t, delta_w)
            suboptimality = 0.5 * lmbda * lmbda
            if suboptimality < self.settings.inner_tolerance:
                nu_star = nu_hat / t
                lambda_star = np.zeros((M + 1,))
                lambda_star[0:M] = (1.0 / t) / w
                lambda_star[M] = (1.0 / t) / (
                    1.0 - (1.0 / (M * self.phi)) * np.dot(w, w)
                )
                return NewtonResult(
                    solution=w,
                    equality_multipliers=nu_star,
                    inequality_multipliers=lambda_star,
                    suboptimality=suboptimality,
                    nits=nit + 1,
                    status=0,
                    message="Newton's method completed successfully to the desired tolerance",
                )

            # Update
            try:
                s = self.backtracking_line_search(w, delta_w, t)
            except BacktrackingLineSearchError:
                if not last_step and suboptimality < self.settings.inner_tolerance_soft:
                    nu_star = nu_hat / t
                    lambda_star = np.zeros((M + 1,))
                    lambda_star[0:M] = ((1.0 / t) / w) * (1.0 + delta_w / w)
                    lambda_star[M] = (1.0 / t) / (
                        1.0 - (1.0 / (M * self.phi)) * np.dot(w, w)
                    )
                    lambda_star[M] = lambda_star[M] * (
                        1.0
                        + (2.0 / (M * self.phi))
                        * np.dot(w, delta_w)
                        / (1 - (1 / (M * self.phi)) * np.dot(w, w))
                    )
                    return NewtonResult(
                        solution=w,
                        equality_multipliers=nu_star,
                        inequality_multipliers=lambda_star,
                        suboptimality=suboptimality,
                        nits=nit + 1,
                        status=1,
                        message=(
                            "Newton's method achieved an acceptable tolerance but then "
                            "ran into numerical issues"
                        ),
                    )

                raise CenteringStepError(
                    message="Backtracking line search failed",
                    suboptimality=suboptimality,
                    last_iterate=w,
                )

            w += s * delta_w

        raise CenteringStepError(
            "Centering step did not converge.",
            suboptimality=suboptimality,
            last_iterate=w,
        )

    def calculate_newton_step(
        self, w: npt.NDArray[np.float64], t: float
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
        the equality constraints. We use
        `solve_kkt_system_hessian_diagonal_plus_rank_one` to solve this system in
        O(p^3 + p^2 * M) time.

        """
        M = self.dimension
        return solve_kkt_system_hessian_diagonal_plus_rank_one(
            A=(1.0 / M) * self.X.T,
            g=-self._grad_ft(w, t),
            eta=self._hessian_ft_diagonal(w, t),
            zeta=self._hessian_ft_rank_one(w),
        )

    def backtracking_line_search(
        self,
        w: npt.NDArray[np.float64],
        delta_w: npt.NDArray[np.float64],
        t: float,
    ) -> float:
        """Perform backtracking line search.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current estimate.
         delta_w : npt.NDArray[np.float64]
            Descent direction.
         t : float
            Barrier parameter.

        Returns
        -------
         s : float
            Step modifier.

        """
        M = self.dimension
        alpha = self.settings.backtracking_alpha
        beta = self.settings.backtracking_beta
        min_step = self.settings.backtracking_min_step

        s = 1.0
        w_new = w + s * delta_w
        while np.any(w_new <= 0) or np.sum(w_new * w_new) >= M * self.phi:
            s *= beta
            if s < min_step:
                raise BacktrackingLineSearchError("Step size got too small.")

            w_new = w + s * delta_w

        fw = self._ft(w, t)
        grad_ft = self._grad_ft(w, t)
        grad_ft_dot_delta_w = np.dot(grad_ft, delta_w)

        while self._ft(w_new, t) > fw + alpha * s * grad_ft_dot_delta_w:
            s *= beta
            if s < min_step:
                raise BacktrackingLineSearchError("Step size got too small.")
            w_new = w + s * delta_w

        return s

    def solve_phase1(
        self, w0: Optional[npt.NDArray[np.float64]] = None
    ) -> npt.NDArray[np.float64]:
        r"""Find a feasible point.

        A point, w, is feasible if:
           (1/M) * X^T w = \mu
           (1/M) * \| w \|_2^2 <= \phi
           w >= 0

        We look for such a point by checking if an initial guess, w0, is
        feasible. If it is, we simply return w0. Otherwise we solve:
           minimize   \| w \|_2^2
           subject to (1/M) * X^T w = \mu
                      w >= 0.

        If the solution satisfies (1/M) * \| w_star \|_2^2 <= \phi, this point
        is feasible, otherwise the problem is infeasible.

        Parameters
        ----------
         w0 : npt.NDArray[np.float64], optional
            If specified, use this as the starting point. Otherwise use a
            vector of all 1s.

        Returns
        -------
         w : npt.NDArray[np.float64]
           Feasible point.

        """
        M = self.dimension
        if w0 is None:
            w0 = np.ones((M,))

        # Check whether w0 is feasible:
        if (
            np.all(w0 > 0)
            and np.allclose((1 / M) * np.dot(self.X.T, w0), self.mu)
            and np.dot(w0, w0) < M * self.phi
        ):
            return w0

        def fun(w: npt.NDArray[np.float64]) -> float:
            return np.dot(w, w)

        def jac(w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return 2.0 * w

        res = minimize(
            fun=fun,
            x0=w0,
            method="trust-constr",
            jac=jac,
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
        if np.dot(w, w) >= M * self.phi:
            raise ProblemInfeasibleError(
                "Minimum feasible norm exceeds variance budget (phi)."
            )

        return w

    def _ft(self, w: npt.NDArray[np.float64], t: float) -> float:
        """Calculate ft at w."""
        M = self.dimension
        return (
            t * self.distance.evaluate(w)
            - np.sum(np.log(w))
            - np.log(1.0 - (1.0 / (M * self.phi)) * np.dot(w, w))
        )

    def _grad_ft(self, w: npt.NDArray[np.float64], t: float) -> npt.NDArray[np.float64]:
        """Calculate gradient of ft at w."""
        M = self.dimension
        den = 1.0 - (1.0 / (M * self.phi)) * np.dot(w, w)
        grad_constraints = w * ((2.0 / (M * self.phi)) / den) - 1.0 / w
        return t * self.distance.gradient(w) + grad_constraints

    def _hessian_ft_diagonal(
        self, w: npt.NDArray[np.float64], t: float
    ) -> npt.NDArray[np.float64]:
        """Calculate diagonal component of Hessian of ft at w."""
        M = self.dimension
        den = 1.0 - (1.0 / (M * self.phi)) * np.dot(w, w)
        return (
            t * self.distance.hessian_diagonal(w)
            + ((2.0 / (M * self.phi)) / den) * np.ones_like(w)
            + 1.0 / (w * w)
        )

    def _hessian_ft_rank_one(
        self, w: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate rank one component of Hessian of ft at w."""
        M = self.dimension
        den = 1.0 - (1.0 / (M * self.phi)) * np.dot(w, w)
        return w * ((2.0 / (M * self.phi)) / den)

    def _newton_decrement(
        self,
        w: npt.NDArray[np.float64],
        t: float,
        delta_w: npt.NDArray[np.float64],
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
