"""Base optimization classes."""

import time
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .exceptions import (
    BacktrackingLineSearchError,
    CenteringStepError,
    ConstraintBoundaryError,
    InteriorPointMethodError,
    InvalidDescentDirectionError,
    NewtonStepError,
    ProblemInfeasibleError,
    SevereCurvatureError,
)


@dataclass
class OptimizationSettings:
    """Optimization settings.

    Parameters
    ----------
    outer_tolerance : float, default=1e-6
        The tolerance level for convergence of the overall optimization problem. It
        controls how close the solution must be to the actual optimal point in terms of
        the objective function value.
    barrier_multiplier : float, default=10.0
        The factor by which the barrier parameter is multiplied at each outer iteration
        to help guide the optimization process towards feasible solutions, balancing
        between the penalty for constraint violation and descent direction.
    inner_tolerance : float, default=1e-6
        The tolerance for convergence within inner iterations of the optimization
        method. This defines when the inner problem is considered solved to a
        satisfactory level.
    inner_tolerance_soft : float, default=1e-4
        A less strict tolerance for the inner iterations, which can be used as a
        fallback for allowing convergence when high precision is not achievable. This
        tolerance can be employed on steps that do not need strict adherence to
        constraints.
    max_inner_iterations : int, default=100
        The maximum number of iterations allowed for the inner optimization loop. This
        guards against infinite loops and over-computation in the case where convergence
        is slow.
    backtracking_alpha : float, default=0.01
        The coefficient used in backtracking line search to control the sufficient
        decrease condition. This value scales the gradient term to adjust the step size
        based on actual progress.
    backtracking_beta : float, default=0.5
        The factor used to reduce the step size in the backtracking line search. A value
        less than 1 encourages more conservative adjustments to the step size for
        stability.
    backtracking_min_step : float, default=1e-3
        The minimum allowable step size for backtracking line search. This prevents
        excessively small steps that could result in numerical issues or ineffective
        exploration of the optimization landscape.
    verbose : bool, default=False
        If True, print status along with how long it took to execute each step.

    """

    outer_tolerance: float = 1e-5
    barrier_multiplier: float = 10.0
    inner_tolerance: float = 1e-6
    inner_tolerance_soft: float = 1e-4
    max_inner_iterations: int = 100
    backtracking_alpha: float = 0.01
    backtracking_beta: float = 0.5
    backtracking_min_step: float = 1e-3
    verbose: bool = False


@dataclass
class OptimizationResult:
    """Wrapper for generic optimization result."""

    solution: npt.NDArray[np.float64]


@dataclass
class NewtonResult(OptimizationResult):
    """Wrapper for the results of Newton's method.

    Parameters
    ----------
     solution : vector
        The solution.
     objective_value : float
        Objective value.
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

    objective_value: float
    equality_multipliers: npt.NDArray[np.float64]
    inequality_multipliers: npt.NDArray[np.float64]
    suboptimality: float
    nits: int
    status: Literal[0, 1]
    message: str


@dataclass
class InteriorPointMethodResult(OptimizationResult):
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

    objective_value: float
    dual_value: float
    equality_multipliers: npt.NDArray[np.float64]
    inequality_multipliers: npt.NDArray[np.float64]
    suboptimality: float
    nits: int
    inner_nits: List[int]
    status: Literal[0]


class Optimizer(ABC):
    """Base class for an optimizer."""

    def __init__(
        self,
        phase1_solver: Optional["PhaseISolver"] = None,
        settings: Optional[OptimizationSettings] = None,
        **kwargs,
    ) -> None:
        """Initialize optimizer."""
        self.phase1_solver = phase1_solver
        if settings is None:
            self.settings: OptimizationSettings = OptimizationSettings()
        else:
            self.settings = settings

    @abstractproperty
    def num_eq_constraints(self) -> int:
        """Count equality constraints."""

    @abstractproperty
    def num_ineq_constraints(self) -> int:
        """Count inequality constraints."""

    @abstractmethod
    def solve(
        self, x0: Optional[npt.NDArray[np.float64]] = None, **kwargs
    ) -> OptimizationResult:
        """Solve optimization problem.

        Parameters
        ----------
         x0 : vector, optional
            Initial guess.

        Returns
        -------
         res : OptimizationResult
            The solution.

        """


class PhaseISolver(Optimizer):
    """Base class for a PhaseISolver."""

    @abstractmethod
    def solve(
        self,
        x0: Optional[npt.NDArray[np.float64]] = None,
        fully_optimize: bool = False,
        **kwargs,
    ) -> OptimizationResult:
        """Solve optimization problem.

        Parameters
        ----------
         x0 : vector, optional
            Initial guess.
         fully_optimize : bool, optional
            If True, solve the problem to the full optimal point. Otherwise, return as
            soon as we have a feasible point. Defaults to False.

        Returns
        -------
         res : OptimizationResult
            The solution.

        """


class InteriorPointMethodSolver(Optimizer):
    """Solve an optimization problem using an Interior Point Method."""

    def __init__(
        self,
        phase1_solver: Optional["PhaseISolver"] = None,
        settings: Optional[OptimizationSettings] = None,
        **kwargs,
    ) -> None:
        """Initialize optimizer."""
        self.phase1_solver = phase1_solver
        if settings is None:
            self.settings: OptimizationSettings = OptimizationSettings()
        else:
            self.settings = settings

    def solve(
        self,
        x0: Optional[npt.NDArray[np.float64]] = None,
        **kwargs,
    ) -> OptimizationResult:
        r"""Solve optimization problem.

        Uses an interior point method with a logarithmic barrier penalty to
        solve:
           minimize    f0(x)
           subject to  A * x = b
                       fi(x) <= 0, i=1, ..., M.

        Parameters
        ----------
         x0 : vector
            Initial guess. If infeasible, a Phase I method will be used to find a
            feasible point close to x0.

        Returns
        -------
         res : InteriorPointMethodResult
            The results are wrapped in a InteriorPointMethodResult class, which includes
            the solution and other helpful info.

        """
        if self.phase1_solver is None:
            raise ValueError("PhaseISolver not specified.")

        phase1_res = self.phase1_solver.solve(x0=x0, **kwargs)
        x = phase1_res.solution
        t = self.initialize_barrier_parameter(x0=x)
        num_steps = (
            int(
                np.ceil(
                    (
                        np.log(self.num_ineq_constraints)
                        - np.log(t * self.settings.outer_tolerance)
                    )
                    / np.log(self.settings.barrier_multiplier)
                )
            )
            + 1
        )
        inner_nits = []
        if self.settings.verbose:
            overall_start_time = time.time()
            print("  Starting IPM")

        for ii in range(num_steps):
            if self.settings.verbose:
                print(f"  {ii:02d} Beginning centering step with {t=:}")
                start_time = time.time()

            try:
                result = self.centering_step(x, t, last_step=(ii + 1 == num_steps))
            except CenteringStepError as e:
                raise InteriorPointMethodError(
                    message="Centering step failed",
                    remaining_steps=num_steps - ii - 1,
                    suboptimality=self.num_ineq_constraints
                    * self.settings.barrier_multiplier
                    / t,
                    last_iterate=e.last_iterate,
                ) from e

            if self.settings.verbose:
                end_time = time.time()
                print(
                    f"  {ii:02d} Centering step completed in "
                    f"{1000 * (end_time - start_time):.03f} ms"
                )

            x = result.solution
            inner_nits.append(result.nits)

            # w = self.predictor_corrector(w, t)
            t *= self.settings.barrier_multiplier

        if self.settings.verbose:
            overall_end_time = time.time()
            print(
                f"  IPM completed in "
                f"{1000 * (overall_end_time - overall_start_time):.03f} ms"
            )

        return InteriorPointMethodResult(
            solution=x,
            objective_value=self.evaluate_objective(x),
            dual_value=self.evaluate_dual(
                lmbda=result.inequality_multipliers,
                nu=result.equality_multipliers,
                x_star=x,
            ),
            equality_multipliers=result.equality_multipliers,
            inequality_multipliers=result.inequality_multipliers,
            suboptimality=self.num_ineq_constraints
            * self.settings.barrier_multiplier
            / t,
            nits=num_steps,
            inner_nits=inner_nits,
            status=0,
        )

    def initialize_barrier_parameter(self, x0: npt.NDArray[np.float64]) -> float:
        """Initialize barrier parameter.

        We can often skip early centering steps by initializing the barrier parameter to
        a higher value. See the discussion in Boyd and Vandenberghe, section 11.3.1.

        """
        return 1.0

    def centering_step(
        self, x0: npt.NDArray[np.float64], t: float, last_step: bool, **kwargs
    ) -> NewtonResult:
        r"""Solve centering step.

        The centering step solves:
          minimize   ft(x) := t * f0(x) - \sum_i log(-fi(x))
          subject to A * x = b.

        Parameters
        ----------
         x0 : vector
            Initial guess, [w0; s0]. Must be strictly feasible.
         t : float
            Barrier parameter.
         last_step: bool
             Indicates whether this is the last centering step. See Notes.

        Returns
        -------
         res : NewtonResult
            The results are wrapped in a NewtonResult class, which includes the optimal
            weights and other helpful info.

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
        x = x0.copy()
        suboptimality = np.inf
        for nit in range(self.settings.max_inner_iterations):
            if self.settings.verbose:
                start_time = time.time()

            try:
                delta_x, nu_hat = self.calculate_newton_step(x, t)
            except NewtonStepError as e:
                raise CenteringStepError(
                    message="Failed to calculate Newton step.",
                    suboptimality=suboptimality,
                    last_iterate=x,
                ) from e

            if self.settings.verbose:
                end_time = time.time()

            # Check for convergence
            lmbda = self.newton_decrement(x, t, delta_x)
            suboptimality = 0.5 * lmbda * lmbda
            if self.settings.verbose:
                print(
                    f"    {nit:02d} Newton step calculated in "
                    f"{1000 * (end_time - start_time):.03f} ms; "
                    f"Sub-optimality {suboptimality}"
                )

            if suboptimality < self.settings.inner_tolerance:
                nu_star = nu_hat / t
                lambda_star = self.inequality_multipliers(x, t, delta_x)
                return NewtonResult(
                    solution=x,
                    objective_value=self.evaluate_barrier_objective(x, t),
                    equality_multipliers=nu_star,
                    inequality_multipliers=lambda_star,
                    suboptimality=suboptimality,
                    nits=nit + 1,
                    status=0,
                    message=(
                        "Newton's method completed successfully to the desired "
                        "tolerance"
                    ),
                )

            # Update
            try:
                btls_s = self.backtracking_line_search(x, delta_x, t)
            except BacktrackingLineSearchError as e:
                if not last_step and suboptimality < self.settings.inner_tolerance_soft:
                    nu_star = nu_hat / t
                    lambda_star = self.inequality_multipliers(
                        x, t, delta_x, centering_step_solved_perfectly=False
                    )
                    return NewtonResult(
                        solution=x,
                        objective_value=self.evaluate_barrier_objective(x, t),
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
                    last_iterate=x,
                ) from e

            x += btls_s * delta_x

        raise CenteringStepError(
            "Centering step did not converge.",
            suboptimality=suboptimality,
            last_iterate=x,
        )

    def backtracking_line_search(
        self,
        x: npt.NDArray[np.float64],
        delta_x: npt.NDArray[np.float64],
        t: float,
    ) -> float:
        """Perform backtracking line search.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Current estimate.
         delta_x : npt.NDArray[np.float64]
            Descent direction.
         t : float
            Barrier parameter.

        Returns
        -------
         btls_s : float
            Step modifier.

        """
        alpha = self.settings.backtracking_alpha
        beta = self.settings.backtracking_beta
        min_step = self.settings.backtracking_min_step

        btls_s = self.btls_keep_feasible(x, delta_x)
        if btls_s < min_step:
            raise ConstraintBoundaryError(
                message="Descent step takes us too close to constraint boundaries.",
            )

        ft = self.evaluate_barrier_objective(x, t)
        grad_ft = self.gradient_barrier(x, t)
        grad_ft_dot_delta_x = np.dot(grad_ft, delta_x)
        if grad_ft_dot_delta_x > 0:
            raise InvalidDescentDirectionError(
                message="Newton step was not a descent direction.",
                grad_ft_dot_delta_x=grad_ft_dot_delta_x,
            )

        # alpha * -grad_ft_dot_delta_x is a positive number since delta_x is a descent
        # direction.
        neg_alpha_grad = alpha * -grad_ft_dot_delta_x

        x_new = x + btls_s * delta_x
        while self.evaluate_barrier_objective(x_new, t) + btls_s * neg_alpha_grad > ft:
            btls_s *= beta
            if btls_s < min_step:
                raise SevereCurvatureError(
                    message="Small step sizes did not adequately decrease objective.",
                    required_improvement=btls_s * neg_alpha_grad,
                    actual_improvement=ft - self.evaluate_barrier_objective(x_new, t),
                )
            x_new = x + btls_s * delta_x

        return btls_s

    def newton_decrement(
        self,
        x: npt.NDArray[np.float64],
        t: float,
        delta_x: npt.NDArray[np.float64],
    ) -> float:
        """Calculate Newton decrement.

        The Newton decrement is the square root of:
           delta_x * H * delta_x,
        where H is the Hessian.

        """
        return np.sqrt(np.dot(delta_x, self.hessian_multiply(x, t, delta_x)))

    def evaluate_barrier_objective(self, x: npt.NDArray[np.float64], t: float) -> float:
        """Calculate ft at x."""
        return t * self.evaluate_objective(x) - np.sum(np.log(-self.constraints(x)))

    def gradient_barrier(
        self, x: npt.NDArray[np.float64], t: float
    ) -> npt.NDArray[np.float64]:
        """Calculate gradient of ft at x.

        This is provided as a convenience, but this implementation involves a M-by-q
        matrix multiply, where M is the number of variables and q the number of
        constraints. As such, it ends up being one of the more expensive operations in
        the whole codebase. A custom implementation that takes into account the special
        structure of the problem can often get this down to O(M) time, rather than
        O(M*q).

        """
        return t * self.gradient(x) - np.dot(
            self.grad_constraints(x).T, (1.0 / self.constraints(x))
        )

    def inequality_multipliers(
        self,
        x: npt.NDArray[np.float64],
        t: float,
        delta_x: npt.NDArray[np.float64],
        centering_step_solved_perfectly: bool = True,
    ) -> npt.NDArray[np.float64]:
        """Calculate Lagrange multipliers for inequality constraints."""
        lmbda = -1.0 / (t * self.constraints(x))
        if not centering_step_solved_perfectly:
            lmbda *= 1.0 - np.dot(self.grad_constraints(x), delta_x) / self.constraints(
                x
            )
        return lmbda

    @abstractmethod
    def calculate_newton_step(
        self,
        x: npt.NDArray[np.float64],
        t: float,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Calculate Newton step.

        Calculates Newton step for the "inner" problem:
          minimize   ft(x) := t * f0(x) - \sum_i log(-fi(x))
          subject to A * x = b.

        Parameters
        ----------
         x : vector
            Current estimate.
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
        constraints.

        """

    @abstractmethod
    def btls_keep_feasible(
        self, x: npt.NDArray[np.float64], delta_x: npt.NDArray[np.float64]
    ) -> float:
        """Make sure x + btls_s * delta_x stays strictly feasible."""

    @abstractmethod
    def evaluate_objective(self, x: npt.NDArray[np.float64]) -> float:
        """Calculate f0 at x."""

    @abstractmethod
    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate gradient of f0 at x."""

    @abstractmethod
    def hessian_multiply(
        self, x: npt.NDArray, t: float, y: npt.NDArray
    ) -> npt.NDArray[np.float64]:
        """Multiply H * y."""

    @abstractmethod
    def constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate the vector of constraints, fi(x) <= 0."""

    @abstractmethod
    def grad_constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate the gradients of constraints, fi(x) <= 0."""

    @abstractmethod
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
         x_star : vector
            The solution.

        Returns
        -------
         g : float
            The dual function evaluated at lmbda, nu.

        Notes
        -----
        Ideally the implementation should support evaluating the dual at any points,
        lmbda and nu. However in practice evaluating the dual can be challenging, and we
        only really need to evaluate it at the optimal lambda_star and nu_star, in which
        case the dual is simply the Lagrangian evaluated at x_star, lambda_star, nu_star.

        """


class PhaseIInteriorPointSolver(InteriorPointMethodSolver, PhaseISolver):
    """Base class for a Phase I solver that uses an Interior Point Method.

    The main distinction is that a Phase I solver can terminate as soon as a feasible
    point is found.

    """

    def __init__(
        self,
        phase1_solver: Optional["PhaseISolver"] = None,
        settings: Optional[OptimizationSettings] = None,
        **kwargs,
    ) -> None:
        """Initialize optimizer."""
        super().__init__(phase1_solver=phase1_solver, settings=settings, **kwargs)

    def solve(
        self,
        x0: Optional[npt.NDArray[np.float64]] = None,
        fully_optimize: bool = False,
        **kwargs,
    ) -> OptimizationResult:
        r"""Solve optimization problem.

        Uses an interior point method with a logarithmic barrier penalty to
        solve:
           minimize    0
           subject to  A * x = b
                       fi(x) <= 0, i=1, ..., M.
                       fj(x) <= 0, j=1, ..., N.

        Parameters
        ----------
         x0 : vector
            Initial guess, intended to be feasible for some of the constraints, allowing
            the Phase I method to focus on a particular set of constraints. See Notes.
         fully_optimize : bool
            If True, solve the underlying problem to full precision. Otherwise, return a
            feasible point as soon as we find one. See Notes.

        Returns
        -------
         res : InteriorPointMethodResult
            The results are wrapped in a InteriorPointMethodResult class, which includes
            a feasible point and other helpful info.

        Raises
        ------
         ProblemInfeasibleError: if a feasible point does not exist.

        Notes
        -----
        Given a point x0 satisfying A * x0 = b and fi(x0) < 0 for i=1, ..., M, solves:
           minimize    s
           subject to  A * x = b
                       fi(x) <= 0, i=1, ..., M.
                       fj(x) <= s, j=1, ..., N.

        If the solution, s^\star, is > 0, the problem is infeasible, and the Lagrange
        multipliers provide a certificate of this infeasibility. Otherwise, the solution
        x^\star is feasible for all constraints.

        We do not have to solve this problem to high precision. If we find feasible (x,
        s) with s < 0, we can quit.

        """
        if self.phase1_solver is None:
            raise ValueError("PhaseISolver not specified.")

        # The nested Phase I Solver should return x such that A * x = b and fi(x) < 0,
        # i=1, ..., M.
        phase1_res = self.phase1_solver.solve(x0=x0, **kwargs)
        x = self.augment_previous_solution(phase1_res)
        if not fully_optimize and self.is_feasible(x):
            if self.settings.verbose:
                print("  Phase I solution was feasible so we're done")
            return phase1_res

        t = self.initialize_barrier_parameter(x0=x)
        num_steps = (
            int(
                np.ceil(
                    (
                        np.log(self.num_ineq_constraints)
                        - np.log(t * self.settings.outer_tolerance)
                    )
                    / np.log(self.settings.barrier_multiplier)
                )
            )
            + 1
        )
        inner_nits = []
        if self.settings.verbose:
            overall_start_time = time.time()
            print("  Starting IPM")

        for ii in range(num_steps):
            if self.settings.verbose:
                print(f"  {ii:02d} Beginning centering step with {t=:}")
                start_time = time.time()

            try:
                result = self.centering_step(
                    x, t, last_step=(ii + 1 == num_steps), fully_optimize=fully_optimize
                )
            except CenteringStepError as e:
                raise InteriorPointMethodError(
                    message="Centering step failed",
                    remaining_steps=num_steps - ii - 1,
                    suboptimality=self.num_ineq_constraints
                    * self.settings.barrier_multiplier
                    / t,
                    last_iterate=e.last_iterate,
                ) from e

            if self.settings.verbose:
                end_time = time.time()
                print(
                    f"  {ii:02d} Centering step completed in "
                    f"{1000 * (end_time - start_time):.03f} ms"
                )

            x = result.solution
            inner_nits.append(result.nits)
            if not fully_optimize and self.is_feasible(x):
                if self.settings.verbose:
                    print(
                        f"  {ii:02d} Result was strictly feasible so we're "
                        "early-stopping."
                    )
                break

            # w = self.predictor_corrector(w, t)
            t *= self.settings.barrier_multiplier

        if self.settings.verbose:
            overall_end_time = time.time()
            print(
                f"  IPM completed in "
                f"{1000 * (overall_end_time - overall_start_time):.03f} ms"
            )

        if not fully_optimize and not self.is_feasible(x):
            raise ProblemInfeasibleError("Problem was infeasible.")

        return InteriorPointMethodResult(
            solution=self.finalize_solution(x),
            objective_value=self.evaluate_objective(x),
            dual_value=self.evaluate_dual(
                lmbda=result.inequality_multipliers,
                nu=result.equality_multipliers,
                x_star=x,
            ),
            equality_multipliers=result.equality_multipliers,
            inequality_multipliers=result.inequality_multipliers,
            suboptimality=self.num_ineq_constraints
            * self.settings.barrier_multiplier
            / t,
            nits=num_steps,
            inner_nits=inner_nits,
            status=0,
        )

    @abstractmethod
    def is_feasible(self, x: npt.NDArray[np.float64]) -> bool:
        """Determine whether a feasible point has been found."""

    @abstractmethod
    def augment_previous_solution(
        self, phase1_res: OptimizationResult, **kwargs
    ) -> npt.NDArray[np.float64]:
        """Initialize variable based on Phase I result."""

    def finalize_solution(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """De-augment solution."""
        return x[0:-1]

    def centering_step(
        self,
        x0: npt.NDArray[np.float64],
        t: float,
        last_step: bool,
        fully_optimize: bool = False,
        **kwargs,
    ) -> NewtonResult:
        r"""Solve centering step.

        The centering step solves:
          minimize   ft(x) := t * f0(x) - \sum_i log(-fi(x))
          subject to A * x = b.

        Parameters
        ----------
         x0 : vector
            Initial guess, [w0; s0]. Must be strictly feasible.
         t : float
            Barrier parameter.
         last_step: bool
             Indicates whether this is the last centering step. See Notes.

        Returns
        -------
         res : NewtonResult
            The results are wrapped in a NewtonResult class, which includes the optimal
            weights and other helpful info.

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
        x = x0.copy()
        suboptimality = np.inf
        for nit in range(self.settings.max_inner_iterations):
            if self.settings.verbose:
                start_time = time.time()

            try:
                delta_x, nu_hat = self.calculate_newton_step(x, t)
            except NewtonStepError as e:
                raise CenteringStepError(
                    message="Failed to calculate Newton step.",
                    suboptimality=suboptimality,
                    last_iterate=x,
                ) from e

            if self.settings.verbose:
                end_time = time.time()

            # Check for convergence
            lmbda = self.newton_decrement(x, t, delta_x)
            suboptimality = 0.5 * lmbda * lmbda
            if self.settings.verbose:
                print(
                    f"    {nit:02d} Newton step calculated in "
                    f"{1000 * (end_time - start_time):.03f} ms; "
                    f"Sub-optimality {suboptimality}"
                )

            if suboptimality < self.settings.inner_tolerance:
                nu_star = nu_hat / t
                lambda_star = self.inequality_multipliers(x, t, delta_x)
                return NewtonResult(
                    solution=x,
                    objective_value=self.evaluate_barrier_objective(x, t),
                    equality_multipliers=nu_star,
                    inequality_multipliers=lambda_star,
                    suboptimality=suboptimality,
                    nits=nit + 1,
                    status=0,
                    message=(
                        "Newton's method completed successfully to the desired "
                        "tolerance"
                    ),
                )

            # Update
            try:
                btls_s = self.backtracking_line_search(x, delta_x, t)
            except BacktrackingLineSearchError as e:
                if not last_step and suboptimality < self.settings.inner_tolerance_soft:
                    nu_star = nu_hat / t
                    lambda_star = self.inequality_multipliers(
                        x, t, delta_x, centering_step_solved_perfectly=False
                    )
                    return NewtonResult(
                        solution=x,
                        objective_value=self.evaluate_barrier_objective(x, t),
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
                    last_iterate=x,
                ) from e

            if not fully_optimize and self.is_feasible(x + btls_s * delta_x):
                nu_star = nu_hat / t
                lambda_star = self.inequality_multipliers(
                    x, t, delta_x, centering_step_solved_perfectly=False
                )
                return NewtonResult(
                    solution=x + btls_s * delta_x,
                    objective_value=self.evaluate_barrier_objective(
                        x + btls_s * delta_x, t
                    ),
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

            x += btls_s * delta_x

        raise CenteringStepError(
            "Centering step did not converge.",
            suboptimality=suboptimality,
            last_iterate=x,
        )
