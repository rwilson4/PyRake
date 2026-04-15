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


class ImbalanceVarianceFrontierResults(FrontierResults):
    r"""Pareto frontier results for the imbalance/variance tradeoff.

    Subclasses ``FrontierResults``; inherits ``knee()``.

    The ``.weights``, ``.variances``, and ``.imbalances`` properties all return
    lists sorted by imbalance ascending (tight balance first), so that
    ``weights[i]``, ``variances[i]``, and ``imbalances[i]`` correspond to the
    same frontier point.

    """

    @property
    def _points_by_imbalance(self) -> list[FrontierPoint]:
        return sorted(self.points, key=lambda p: float(p.objectives[1]))

    @property
    def weights(self) -> list[npt.NDArray[np.float64]]:
        """Weights at each frontier point, sorted by imbalance ascending."""
        return [p.solution for p in self._points_by_imbalance]

    @property
    def variances(self) -> list[float]:
        r"""Variance (1/M)\|w\|_2^2 at each frontier point, sorted by imbalance ascending."""
        return [float(p.objectives[0]) for p in self._points_by_imbalance]

    @property
    def imbalances(self) -> list[float]:
        r"""Max covariate imbalance \|(1/M)Z^T w - \nu\|_\infty at each frontier point, sorted ascending."""
        return [float(p.objectives[1]) for p in self._points_by_imbalance]

    def plot(
        self,
        annotate_knee: bool = True,
        ax: plt.Axes | None = None,
        x_label: str = "Variance (φ)",
        y_label: str = "Max Covariate Imbalance (ψ)",
        title: str = "Imbalance-Variance Tradeoff",
    ) -> plt.Axes:
        """Plot the imbalance/variance tradeoff frontier.

        Parameters
        ----------
        annotate_knee : bool, default=True
            If True, mark the knee point and draw a tangent line.
        ax : matplotlib Axes, optional
            If provided, plot onto this Axes; otherwise create a new figure.
        x_label : str, default="Variance (φ)"
        y_label : str, default="Max Covariate Imbalance (ψ)"
        title : str, default="Imbalance-Variance Tradeoff"

        Returns
        -------
        matplotlib Axes

        """
        ax = super().plot(
            annotate_knee=annotate_knee, ax=ax, x_label=x_label, y_label=y_label
        )
        ax.set_title(title)
        return ax


class ImbalanceVarianceFrontier(MultiObjectiveOptimizer):
    r"""Trace the imbalance/variance Pareto frontier.

    For a range of imbalance bounds (\psi), finds the minimum achievable variance
    (\phi). Shows how much variance must be accepted for a given covariate balance
    requirement.

    Solves, for each \psi value along the frontier:
           minimize    (1/M) \| w \|_2^2
           subject to  (1/M) * X^T * w = \mu
                       \| (1/M) * Z^T * w - \nu \|_\infty <= \psi
                        w >= 0.

    Parameters
    ----------
    rake : Rake
        Configured calibration solver. Must include approximate-balance covariates
        (i.e., ``Z`` and ``nu`` must have been supplied when constructing the Rake).

    Raises
    ------
    ValueError
        If ``rake`` does not have Z covariates set (``rake.B is None``).

    """

    def __init__(self, rake: Rake) -> None:
        super().__init__(primary_objective=0)
        if rake.B is None:
            raise ValueError(
                "rake must include approximate-balance covariates "
                "(Z, nu, psi) to use ImbalanceVarianceFrontier."
            )
        self.rake = rake

    def solve_with_bounds(
        self, bounds: npt.NDArray[np.float64]
    ) -> InteriorPointMethodResult:
        r"""Minimize (1/M)\|w\|_2^2 subject to max imbalance <= bounds[0].

        Parameters
        ----------
        bounds : array of shape (1,)
            ``bounds[0]`` is the upper bound on max covariate imbalance \psi.

        Returns
        -------
        InteriorPointMethodResult

        """
        assert self.rake.B is not None
        assert self.rake.c is not None
        res = EqualityWithBoundsAndNormConstraintSolver(
            phi=np.inf,
            A=self.rake.A,
            b=self.rake.b,
            lb=self.rake.min_weight,
            B=self.rake.B,
            c=self.rake.c,
            psi=float(bounds[0]),
            settings=self.rake.settings,
        ).solve(fully_optimize=True)
        assert isinstance(res, InteriorPointMethodResult)
        return res

    def minimize_objective(self, objective_index: int) -> InteriorPointMethodResult:
        r"""Minimize a single objective with no bounds on the other.

        Parameters
        ----------
        objective_index : int
            0 to minimize (1/M)\|w\|_2^2 (variance); 1 to minimize max imbalance.

        Returns
        -------
        InteriorPointMethodResult

        Raises
        ------
        ValueError
            If ``objective_index`` is not 0 or 1.

        """
        assert self.rake.B is not None
        assert self.rake.c is not None
        if objective_index == 0:
            # Corner 0: minimize variance with no imbalance constraint.
            res = EqualityWithBoundsAndNormConstraintSolver(
                phi=np.inf,
                A=self.rake.A,
                b=self.rake.b,
                lb=self.rake.min_weight,
                settings=self.rake.settings,
            ).solve(fully_optimize=True)
            assert isinstance(res, InteriorPointMethodResult)
            return res
        elif objective_index == 1:
            # Corner 1: minimize max imbalance with no variance constraint.
            # EqualityWithBoundsAndImbalanceConstraintSolver minimizes s = ||Bx - c||_inf
            # as its primary objective.  psi is used only for the post-hoc infeasibility
            # check (raises if s* > psi); the Rake always sets psi so that the problem is
            # feasible with that bound, hence s_min <= rake.psi is guaranteed.
            assert self.rake.psi is not None
            res = EqualityWithBoundsAndImbalanceConstraintSolver(
                B=self.rake.B,
                c=self.rake.c,
                psi=self.rake.psi,
                A=self.rake.A,
                b=self.rake.b,
                lb=self.rake.min_weight,
                settings=self.rake.settings,
            ).solve(fully_optimize=True)
            assert isinstance(res, InteriorPointMethodResult)
            return res
        else:
            raise ValueError(f"Unknown objective_index: {objective_index}")

    def evaluate_objectives(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Return [(1/M)\|w\|_2^2, \|(1/M)Z^T w - \nu\|_\infty] at x.

        Parameters
        ----------
        x : array
            Weight vector.

        Returns
        -------
        array of shape (2,)

        """
        assert self.rake.B is not None
        assert self.rake.c is not None
        variance = float(np.dot(x, x)) / self.rake.dimension
        imbalance = float(np.max(np.abs(self.rake.B @ x - self.rake.c)))
        return np.array([variance, imbalance])

    def trace(self, num_points: int = 50) -> ImbalanceVarianceFrontierResults:
        r"""Trace the imbalance/variance tradeoff.

        Solves the minimum-variance problem for ``num_points`` values of \psi between
        the two Pareto corners. Corner 0 minimizes variance (no imbalance constraint);
        corner 1 minimizes max imbalance (no variance constraint).

        Parameters
        ----------
        num_points : int, default=50
            Number of grid points along the \psi dimension.

        Returns
        -------
        ImbalanceVarianceFrontierResults
            ``corners[0]`` minimizes variance; ``corners[1]`` minimizes imbalance. The
            ``.weights``, ``.variances``, and ``.imbalances`` convenience properties are
            sorted by imbalance ascending.

        """
        fr = super().trace(num_points=num_points)
        return ImbalanceVarianceFrontierResults(
            points=fr.points,
            corners=fr.corners,
            primary_objective=fr.primary_objective,
            n_attempted=fr.n_attempted,
            n_skipped=fr.n_skipped,
        )
