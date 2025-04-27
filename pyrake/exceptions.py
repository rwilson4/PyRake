"""Custom exceptions."""

import numpy as np
import numpy.typing as npt


class ProblemInfeasibleError(Exception):
    """Raised when the balancing weight problem is infeasible."""


class BacktrackingLineSearchError(Exception):
    """Raised when BTLS fails."""


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
