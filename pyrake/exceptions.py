"""Custom exceptions."""


class ProblemInfeasibleError(Exception):
    """Raised when the balancing weight problem is infeasible."""


class BacktrackingLineSearchError(Exception):
    """Raised when BTLS fails."""


class CenteringStepError(Exception):
    """Raised when centering step fails."""


class InteriorPointMethodError(Exception):
    """Raised when interior point method fails."""
