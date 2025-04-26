"""Custom exceptions."""


class ProblemInfeasibleError(Exception):
    """Raised when the balancing weight problem is infeasible."""

class BacktrackingLineSearchError(Exception):
    """Raised when BTLS fails."""
