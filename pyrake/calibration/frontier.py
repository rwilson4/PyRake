"""Bias/Variance Tradeoff."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from cvxium import (
    FrontierPoint,
    FrontierResults,
    InteriorPointMethodResult,
    MultiObjectiveOptimizer,
)

from .phase1solvers import (
    EqualityWithBoundsAndImbalanceConstraintSolver,
    EqualityWithBoundsAndNormConstraintSolver,
    EqualityWithBoundsSolver,
)
from .rake import Rake


class EfficientFrontierResults(FrontierResults):
    r"""Pareto frontier results with PyRake-specific convenience properties.

    Subclasses ``FrontierResults``; inherits ``knee()``.

    The ``.weights``, ``.distances``, and ``.variances`` properties all return
    lists sorted by variance ascending (low variance first), so that
    ``weights[i]``, ``distances[i]``, and ``variances[i]`` correspond to the
    same frontier point.

    """

    @property
    def _points_by_variance(self) -> list[FrontierPoint]:
        return sorted(self.points, key=lambda p: float(p.objectives[1]))

    @property
    def weights(self) -> list[npt.NDArray[np.float64]]:
        """Weights at each frontier point, sorted by variance ascending."""
        return [p.solution for p in self._points_by_variance]

    @property
    def distances(self) -> list[float]:
        r"""Distance D(w, v) at each frontier point, sorted by variance ascending."""
        return [float(p.objectives[0]) for p in self._points_by_variance]

    @property
    def variances(self) -> list[float]:
        r"""Variance (1/M)\|w\|_2^2 at each frontier point, sorted ascending."""
        return [float(p.objectives[1]) for p in self._points_by_variance]

    def plot(
        self,
        annotate_knee: bool = True,
        ax: plt.Axes | None = None,
        x_label: str = "Distance from Baseline Weights",
        y_label: str = "Variance Inflation Factor",
        title: str = "Bias-Variance Tradeoff",
    ) -> plt.Axes:
        """Plot the bias/variance tradeoff frontier.

        Parameters
        ----------
        annotate_knee : bool, default=True
            If True, mark the knee point and draw a tangent line.
        ax : matplotlib Axes, optional
            If provided, plot onto this Axes; otherwise create a new figure.
        x_label : str, default="Distance from Baseline Weights"
        y_label : str, default="Variance Inflation Factor"
        title : str, default="Bias-Variance Tradeoff"

        Returns
        -------
        matplotlib Axes

        """
        ax = super().plot(
            annotate_knee=annotate_knee, ax=ax, x_label=x_label, y_label=y_label
        )
        ax.set_title(title)
        return ax


class EfficientFrontier(MultiObjectiveOptimizer):
    r"""Trace the bias/variance Pareto frontier.

    Solves:
           minimize    D(w, v)
           subject to  (1/M) * X^T * w = \mu
                       (1/M) * \| w \|_2^2 <= \phi
                        w >= 0,
    for a range of \phi values spanning from the minimum achievable variance to the
    variance at the minimum-distance solution. Increasing \phi allows more variance but
    potentially less bias.

    Parameters
    ----------
    rake : Rake
        Configured calibration solver.

    """

    def __init__(self, rake: Rake) -> None:
        super().__init__(primary_objective=0)
        self.rake = rake

    def solve_with_bounds(
        self, bounds: npt.NDArray[np.float64]
    ) -> InteriorPointMethodResult:
        """Minimize D(w, v) subject to variance <= bounds[0].

        Parameters
        ----------
        bounds : array of shape (1,)
            ``bounds[0]`` is the upper bound on variance (1/M)||w||².

        Returns
        -------
        InteriorPointMethodResult

        """
        self.rake.update_phi(float(bounds[0]))
        return self.rake.solve()

    def minimize_objective(self, objective_index: int) -> InteriorPointMethodResult:
        """Minimize a single objective with no bounds on the other.

        Parameters
        ----------
        objective_index : int
            0 to minimize D(w, v); 1 to minimize (1/M)||w||².

        Returns
        -------
        InteriorPointMethodResult

        Raises
        ------
        ValueError
            If ``objective_index`` is not 0 or 1, or if the Phase I solver type is
            unrecognized when minimizing variance.

        """
        if objective_index == 0:
            # Corner 0: minimize D(w, v) with no variance constraint.
            self.rake.update_phi(None)
            return self.rake.solve()
        elif objective_index == 1:
            # Corner 1: minimize (1/M)||w||² subject to balance constraints.
            # Use the Phase I solver with phi=inf to find the minimum-norm
            # feasible point.
            if isinstance(
                self.rake.phase1_solver, EqualityWithBoundsAndNormConstraintSolver
            ):
                res = self.rake.phase1_solver.solve(fully_optimize=True)
            elif isinstance(
                self.rake.phase1_solver,
                EqualityWithBoundsSolver
                | EqualityWithBoundsAndImbalanceConstraintSolver,
            ):
                res = EqualityWithBoundsAndNormConstraintSolver(
                    phi=np.inf,
                    phase1_solver=self.rake.phase1_solver,
                    settings=self.rake.settings,
                ).solve(fully_optimize=True)
            else:
                raise ValueError("Unrecognized phase1_solver type.")
            assert isinstance(res, InteriorPointMethodResult)  # Satisfy type-checker.
            return res
        else:
            raise ValueError(f"Unknown objective_index: {objective_index}")

    def evaluate_objectives(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Return [D(w, v), (1/M)||w||²] at x.

        Parameters
        ----------
        x : array
            Weight vector.

        Returns
        -------
        array of shape (2,)

        """
        distance = self.rake.distance.evaluate(x)
        variance = float(np.dot(x, x)) / self.rake.dimension
        return np.array([distance, variance])

    def trace(self, num_points: int = 50) -> EfficientFrontierResults:
        r"""Trace the bias/variance tradeoff.

        Solves the optimization problem for ``num_points`` values of \phi between the
        two Pareto corners. Corner 0 minimizes D(w, v) (no variance constraint); corner
        1 minimizes (1/M)\|w\|_2^2 (no distance constraint).

        Parameters
        ----------
        num_points : int, default=50
            Number of grid points along the \phi dimension.

        Returns
        -------
        EfficientFrontierResults
            ``corners[0]`` minimizes D(w, v); ``corners[1]`` minimizes variance. The
            ``.weights``, ``.distances``, and ``.variances`` convenience properties are
            sorted by variance ascending.

        """
        fr = super().trace(num_points=num_points)
        return EfficientFrontierResults(
            points=fr.points,
            corners=fr.corners,
            primary_objective=fr.primary_objective,
            n_attempted=fr.n_attempted,
            n_skipped=fr.n_skipped,
        )
