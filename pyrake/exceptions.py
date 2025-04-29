"""Custom exceptions."""

import numpy as np
import numpy.typing as npt


class ProblemInfeasibleError(Exception):
    """Raised when the balancing weight problem is infeasible."""


class BacktrackingLineSearchError(Exception):
    """Raised when BTLS fails."""

    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        """Pretty-print error."""
        return self.message


class ConstraintBoundaryError(BacktrackingLineSearchError):
    """Raised when BTLS fails because even small steps violated a constraint."""

    def __init__(
        self, message: str, positivity_step: float, variance_step: float
    ) -> None:
        self.message = message
        self.positivity_step = positivity_step
        self.variance_step = variance_step

    def __str__(self) -> str:
        """Pretty-print error."""
        msg = (
            f"{self.message} Max step size for positivity: {self.positivity_step:.03g};"
            f" for variance: {self.variance_step:.03g}."
        )
        return msg


class InvalidDescentDirectionError(BacktrackingLineSearchError):
    """Raised when BTLS fails because the Newton step wasn't a descent direction.

    Usually this is because the Hessian is nearly singular and there was some numerical
    issue.

    """

    def __init__(self, message: str, grad_ft_dot_delta_w: float) -> None:
        self.message = message
        self.grad_ft_dot_delta_w = grad_ft_dot_delta_w

    def __str__(self) -> str:
        """Pretty-print error."""
        msg = (
            f"{self.message} (∇f^T △w = {self.grad_ft_dot_delta_w} > 0, "
            "but should be <= 0)"
        )
        return msg


class SevereCurvatureError(BacktrackingLineSearchError):
    """Raised when BTLS fails because the backtracking condition was not met.

    Usually this happens because the linear approximation doesn't hold even for small
    step sizes, which indicates the Hessian has severe curvature.

    """

    def __init__(
        self,
        message: str,
        required_improvement: float,
        actual_improvement: float,
    ) -> None:
        self.message = message
        self.required_improvement = required_improvement
        self.actual_improvement = actual_improvement

    def __str__(self) -> str:
        """Pretty-print error."""
        msg = (
            f"{self.message} (required improvement >= {self.required_improvement:.03g}"
            f"; actual improvement = {self.actual_improvement:.03g})"
        )
        return msg


class OptimizationError(Exception):
    """Base class for optimization errors."""

    def __init__(
        self, message: str, suboptimality: float, last_iterate: npt.NDArray[np.float64]
    ) -> None:
        self.message = message
        self.suboptimality = suboptimality
        self.last_iterate = last_iterate

    def __str__(self) -> str:
        """Pretty-print error."""
        msg = f"{self.message} (suboptimality = {self.suboptimality:.03g})"
        return msg


class CenteringStepError(OptimizationError):
    """Raised when centering step fails."""


class InteriorPointMethodError(OptimizationError):
    """Raised when interior point method fails."""

    def __init__(
        self,
        message: str,
        remaining_steps: int,
        suboptimality: float,
        last_iterate: npt.NDArray[np.float64],
    ) -> None:
        self.message = message
        self.remaining_steps = remaining_steps
        self.suboptimality = suboptimality
        self.last_iterate = last_iterate

    def __str__(self) -> str:
        """Pretty-print error."""
        msg = (
            f"{self.message} ({self.remaining_steps} step(s) remaining; "
            f"suboptimality = {self.suboptimality:.03g})"
        )
        return msg
