"""Custom exceptions."""

import numpy as np
import numpy.typing as npt


class ProblemInfeasibleError(Exception):
    """Raised when the balancing weight problem is infeasible."""


class BacktrackingLineSearchError(Exception):
    """Raised when BTLS fails."""


class CenteringStepError(Exception):
    """Raised when centering step fails."""

    def __init__(self, message: str, last_iterate: npt.NDArray[np.float64]) -> None:
        self.message = message
        self.last_iterate = last_iterate


class InteriorPointMethodError(Exception):
    """Raised when interior point method fails."""

    def __init__(self, message: str, last_iterate: npt.NDArray[np.float64]) -> None:
        self.message = message
        self.last_iterate = last_iterate
