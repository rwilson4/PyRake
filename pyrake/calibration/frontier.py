"""Bias/Variance, Imbalance/Variance and External-Validity/Variance Tradeoffs."""

from __future__ import annotations

import bisect
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from cvxium import (
    BacktrackingLineSearchError,
    BarrierSolver,
    BoundsConstraintLayer,
    ConstraintLayer,
    EqualityConstraintLayer,
    FrontierPoint,
    FrontierResults,
    ImbalanceConstraintLayer,
    LinearInequalityConstraintLayer,
    LinearObjective,
    MultiObjectiveOptimizer,
    NewtonStepError,
    OptimizationError,
    PrimalDualSolver,
    Problem,
    ProblemInfeasibleError,
    SolveResult,
)

from pyrake.calibration.distance_metrics import (
    SquaredL2,
)
from pyrake.calibration.joint_calibrator import (
    FallbackOptimizationResult,
    JointCalibrator,
)
from pyrake.calibration.rake import Rake

logger: logging.Logger = logging.getLogger(__name__)

# A labelled reference point for a frontier plot: a FrontierPoint, or an
# objective vector in the frontier's own (minimization) coordinates, such as
# ``frontier.evaluate_objectives(w)`` for a weighting solved some other way.
AnnotationPoint = FrontierPoint | Sequence[float] | npt.NDArray[np.float64]


def _annotation_objectives(
    label: str, point: AnnotationPoint
) -> npt.NDArray[np.float64]:
    """Return the objective vector behind an annotation.

    Checked for shape and finiteness.

    Parameters
    ----------
     label : str
        Annotation label, for the error when the shape is wrong.
     point : AnnotationPoint
        A ``FrontierPoint`` or a 2-entry objective vector.

    Returns
    -------
     objectives : npt.NDArray[np.float64]
        The raveled 2-entry finite objective vector.

    Raises
    ------
     ValueError
        When the vector is not 2-entry or not finite.

    """
    raw = point.objectives if isinstance(point, FrontierPoint) else point
    objectives = np.asarray(raw, dtype=np.float64).ravel()
    if objectives.shape != (2,):
        raise ValueError(
            f"Annotation {label!r} needs a FrontierPoint or a 2-entry objective "
            f"vector; got shape {objectives.shape}."
        )
    if not np.all(np.isfinite(objectives)):
        raise ValueError(f"Annotation {label!r} has a non-finite objective.")
    return objectives


def _draw_annotations(
    ax: plt.Axes,
    coordinates: dict[str, tuple[float, float]],
    xs: Sequence[float],
    marker: str = "*",
    marker_size: float = 10,
    marker_color: str = "orange",
    annotation_size: float | str | None = None,
) -> None:
    """Mark and label reference points, already in plot coordinates.

    Called before anything freezes the axis limits (the knee's crosshairs read and
    re-set them), so autoscaling still reaches a point that lies outside the traced
    range, such as a weighting that is not on the frontier.

    Labels sit beside their markers on whichever side faces the middle of the plot, so a
    point at either end of the curve keeps its label inside the axes. ``xs`` are the
    frontier's own abscissae, which together with the annotations decide where the
    middle is.

    Parameters
    ----------
     ax : plt.Axes
        Axes to draw onto.
     coordinates : dict[str, tuple[float, float]]
        Labelled ``(x, y)`` positions in plot coordinates.
     xs : Sequence[float]
        The frontier's own abscissae, to locate the middle of the plot.
     marker : str
        Marker style.
     marker_size : float
        Marker size.
     marker_color : str
        Marker color.
     annotation_size : float | str | None
        Label font size, or None for large.

    """
    if not coordinates:
        return

    every_x = list(xs) + [x for x, _ in coordinates.values()]
    middle = 0.5 * (min(every_x) + max(every_x))
    for label, (x, y) in coordinates.items():
        ax.plot(
            [x],
            [y],
            linestyle="none",
            marker=marker,
            markersize=marker_size,
            color=marker_color,
            zorder=3,
        )
        on_right = x > middle
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(-8, 0) if on_right else (8, 0),
            ha="right" if on_right else "left",
            va="center",
            fontsize="large" if annotation_size is None else annotation_size,
            zorder=4,
        )


def _primary_auxiliary_coordinates(
    results: FrontierResults,
    annotations: Mapping[str, AnnotationPoint] | None,
) -> dict[str, tuple[float, float]]:
    """Map annotations into the base plot's axes.

    Primary objective across, auxiliary up.

    Parameters
    ----------
     results : FrontierResults
        The traced frontier, carrying the primary-objective index.
     annotations : Mapping[str, AnnotationPoint] | None
        Labelled reference points, or None for none.

    Returns
    -------
     coordinates : dict[str, tuple[float, float]]
        Labelled ``(x, y)`` positions in plot coordinates.

    """
    pri = results.primary_objective
    aux = 1 - pri
    coordinates: dict[str, tuple[float, float]] = {}
    for label, point in (annotations or {}).items():
        objectives = _annotation_objectives(label, point)
        coordinates[label] = (float(objectives[pri]), float(objectives[aux]))
    return coordinates


def _plot_with_annotations(
    results: FrontierResults,
    annotations: Mapping[str, AnnotationPoint] | None,
    ax: plt.Axes | None,
    marker: str,
    marker_size: float,
    marker_color: str,
    annotation_size: float | str | None,
) -> plt.Axes:
    """Create the axes if needed and draw the annotations onto them.

    Shared by the two result classes whose curve comes from ``FrontierResults.plot``,
    which draws the knee's crosshairs and freezes the limits as it does. Drawing the
    annotations first, onto the axes that plot is then handed, is what keeps them inside
    those limits.

    Parameters
    ----------
     results : FrontierResults
        The traced frontier.
     annotations : Mapping[str, AnnotationPoint] | None
        Labelled reference points, or None for none.
     ax : plt.Axes | None
        Axes to draw onto, or None to create a figure.
     marker : str
        Marker style.
     marker_size : float
        Marker size.
     marker_color : str
        Marker color.
     annotation_size : float | str | None
        Label font size, or None for large.

    Returns
    -------
     ax : plt.Axes
        The axes with the annotations drawn.

    """
    coordinates = _primary_auxiliary_coordinates(results, annotations)
    if ax is None:
        _, ax = plt.subplots()
    pri = results.primary_objective
    _draw_annotations(
        ax,
        coordinates,
        [float(p.objectives[pri]) for p in results.points],
        marker=marker,
        marker_size=marker_size,
        marker_color=marker_color,
        annotation_size=annotation_size,
    )
    return ax


class EfficientFrontierResults(FrontierResults):
    """Pareto frontier results with PyRake-specific convenience properties.

    Subclasses ``FrontierResults``; inherits ``knee()``.

    The ``.weights``, ``.distances``, and ``.variances`` properties all return lists
    sorted by variance ascending (low variance first), so that ``weights[i]``,
    ``distances[i]``, and ``variances[i]`` correspond to the same frontier point.

    """

    @property
    def _points_by_variance(self) -> list[FrontierPoint]:
        """Return the points sorted by variance ascending."""
        return sorted(self.points, key=lambda p: float(p.objectives[1]))

    @property
    def weights(self) -> list[npt.NDArray[np.float64]]:
        """Return the weights at each frontier point, sorted by variance ascending.

        Returns
        -------
         weights : list[npt.NDArray[np.float64]]
            One weight vector per point.

        """
        return [p.solution for p in self._points_by_variance]

    @property
    def distances(self) -> list[float]:
        """Return the distance D(w, v) at each point, sorted by variance ascending.

        Returns
        -------
         distances : list[float]
            One distance per point.

        """
        return [float(p.objectives[0]) for p in self._points_by_variance]

    @property
    def variances(self) -> list[float]:
        """Return the variance (1/M) ‖w‖₂² at each point, sorted ascending.

        Returns
        -------
         variances : list[float]
            One variance per point.

        """
        return [float(p.objectives[1]) for p in self._points_by_variance]

    def plot(
        self,
        annotate_knee: bool = True,
        ax: plt.Axes | None = None,
        x_label: str = "Distance from Baseline Weights",
        y_label: str = "Variance Inflation Factor",
        title: str = "Bias-Variance Tradeoff",
        annotations: Mapping[str, AnnotationPoint] | None = None,
        marker: str = "*",
        marker_size: float = 10,
        marker_color: str = "orange",
        annotation_size: float | str | None = None,
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
         annotations : dict, optional
            Labelled reference points to mark, as ``{label: point}``. A point is a
            ``FrontierPoint`` (e.g. ``results.corners[0]``, the baseline weights, or
            ``results.corners[1]``, the minimum-variance weights) or an objective vector
            ``[distance, variance]``, such as ``frontier.evaluate_objectives(w)`` for
            weights solved some other way.
         marker : str, default="*"
            Marker style for the annotations.
         marker_size : float, default=10
            Marker size for the annotations.
         marker_color : str, default="orange"
            Marker color for the annotations.
         annotation_size : float | str | None, optional
            Label font size for the annotations, or None for large.

        Returns
        -------
         ax : matplotlib Axes
            The axes with the frontier drawn.

        Raises
        ------
         ValueError
            If an annotation is not a ``FrontierPoint`` or a finite 2-entry objective
            vector.

        """
        ax = _plot_with_annotations(
            self,
            annotations,
            ax,
            marker,
            marker_size,
            marker_color,
            annotation_size,
        )
        ax = super().plot(
            annotate_knee=annotate_knee,
            ax=ax,
            x_label=x_label,
            y_label=y_label,
        )
        ax.set_title(title)
        return ax


def _solve_min_variance(
    rake: Rake,
    imbalance_psi: float | list[float] | npt.NDArray[np.float64] | None,
    x0: npt.NDArray[np.float64] | None,
) -> SolveResult:
    """Minimize (1/M) ‖w‖₂² subject to Aw = b, bounds, and optional imbalance.

    Shared variance-minimization for the frontier corner / bounds solves. Uses
    ``SquaredL2(v=zeros(M))`` as the variance objective (``Distance`` natively
    implements ``Cvxium.Objective``), so it composes directly through either inner
    solver:

    Both engines share a ``ones``-clipped default warm start when ``x0`` is None
    (uniform weights, the natural start for the variance objective):
    - ``engine='pd'`` -> ``PrimalDualSolver`` (infeasible-start).
    - ``engine='ipm'`` -> ``BarrierSolver``, which projects a supplied
      ``x0`` onto the equality manifold itself and runs its structured Phase I
      to cascade the constraint layers for feasibility.

    Parameters
    ----------
     rake : Rake
        Source of dimension, weight bounds, equality solver, settings, engine, and
        (optionally) ``B`` / ``c`` for the imbalance layer.
     imbalance_psi : float | list[float] | npt.NDArray[np.float64] | None
        ``None`` -- omit the imbalance layer entirely. Otherwise add
        ``ImbalanceConstraintLayer(B=rake.B, c=rake.c, psi=imbalance_psi)``; ``rake.B``
        and ``rake.c`` must be set.
     x0 : npt.NDArray[np.float64] | None
        Warm-start point, or None for uniform weights clipped into the bounds.

    Returns
    -------
     res : SolveResult
        The minimum-variance weights.

    """
    M = rake.dimension
    constraints: list[ConstraintLayer] = [
        BoundsConstraintLayer(lb=rake.min_weight, ub=rake.max_weight, dimension=M),
    ]
    if imbalance_psi is not None:
        assert rake.B is not None
        assert rake.c is not None
        constraints.append(
            ImbalanceConstraintLayer(B=rake.B, c=rake.c, psi=imbalance_psi)
        )
    problem = Problem(
        obj=SquaredL2(v=np.zeros(M)),
        ineq=constraints,
        eq=rake.equality_layer,
    )
    # Domain-informed default warm start for both engines: uniform weights,
    # clipped into the bounds box. The barrier projects this onto the equality
    # manifold itself and Phase-I's it; PD drives the equality residual to zero.
    if x0 is None:
        ub = np.inf if rake.max_weight is None else rake.max_weight
        x0 = np.clip(np.full(M, 1.0), rake.min_weight, ub)
    if rake.engine == "pd":
        return PrimalDualSolver(rake.settings).solve(problem=problem, x0=x0)
    return BarrierSolver(rake.settings).solve(problem=problem, x0=x0)


class EfficientFrontier(MultiObjectiveOptimizer):
    """Trace the bias/variance Pareto frontier.

    Solves:
       minimize    D(w, v)
       subject to  (1/M) Xᵀw = μ,
                   (1/M) ‖w‖₂² ≤ φ,
                   w ≥ 0,
    for a range of φ values spanning from the minimum achievable variance to the
    variance at the minimum-distance solution. Increasing φ allows more variance but
    potentially less bias.

    Parameters
    ----------
     rake : Rake
        Configured calibration solver.
     engine : {"ipm", "pd"} or None, optional
        Inner solver engine. Defaults to ``"pd"`` because PD warm-starts naturally
        across phi sweep iterations (~2.3x speedup over IPM cold-restart on the bench
        shape; see 2026-05-21 re-benchmark). IPM cannot safely warm-start barrier
        methods across active-set changes. Pass ``"ipm"`` to opt out; pass ``None`` to
        inherit whatever engine ``rake`` was constructed with. Setting this argument
        mutates ``rake.engine``.

    """

    def __init__(
        self,
        rake: Rake,
        engine: Literal["ipm", "pd"] | None = "pd",
    ) -> None:
        super().__init__(primary_objective=0)
        if engine is not None:
            rake.engine = engine
        self.rake = rake

    def solve_with_bounds(
        self,
        bounds: npt.NDArray[np.float64],
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SolveResult:
        """Minimize D(w, v) subject to variance <= bounds[0].

        Parameters
        ----------
         bounds : npt.NDArray[np.float64]
            ``bounds[0]`` is the upper bound on variance (1/M) ‖w‖₂².
         x0 : npt.NDArray[np.float64] | None, optional
            Warm-start point. Forwarded only when ``self.rake.engine == 'pd'`` (PD's
            infeasible-start primal-dual method tolerates ``x0`` sitting on the active
            boundary of the previous phi). Dropped when ``engine == 'ipm'`` because the
            barrier method requires strictly-interior ``x0`` and a warm point from the
            previous sweep iteration typically lands on the boundary, triggering
            ``ProblemInfeasibleError`` from Phase I.

        Returns
        -------
         res : SolveResult
            The distance-minimizing weights within the variance bound.

        """
        self.rake.update_phi(float(bounds[0]))
        warm = x0 if self.rake.engine == "pd" else None
        return self.rake.solve(x0=warm)

    def minimize_objective(
        self,
        objective_index: int,
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SolveResult:
        """Minimize a single objective with no bounds on the other.

        Parameters
        ----------
         objective_index : int
            0 to minimize D(w, v); 1 to minimize (1/M) ‖w‖₂².
         x0 : npt.NDArray[np.float64] | None, optional
            Warm-start point forwarded to the inner solver.

        Returns
        -------
         res : SolveResult
            The single-objective minimizer.

        Raises
        ------
         ValueError
            If ``objective_index`` is not 0 or 1.

        """
        if objective_index == 0:
            # Corner 0: minimize D(w, v) with no variance constraint.
            # See ``solve_with_bounds`` for why ``x0`` is gated on
            # ``engine == 'pd'``.
            self.rake.update_phi(None)
            warm = x0 if self.rake.engine == "pd" else None
            return self.rake.solve(x0=warm)
        elif objective_index == 1:
            # Corner 1: minimize (1/M)||w||^2 subject to the balance constraints
            # (Bounds, and Imbalance when present). ``_solve_min_variance``
            # dispatches PD/IPM over a ``SquaredL2(v=zeros)`` Problem.
            psi = self.rake.psi if self.rake.B is not None else None
            return _solve_min_variance(self.rake, imbalance_psi=psi, x0=x0)
        else:
            raise ValueError(f"Unknown objective_index: {objective_index}")

    def evaluate_objectives(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Return [D(w, v), (1/M) ‖w‖₂²] at x.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Weight vector.

        Returns
        -------
         objectives : npt.NDArray[np.float64]
            Array of shape (2,).

        """
        distance = self.rake.distance.evaluate(x)
        variance = float(np.dot(x, x)) / self.rake.dimension
        return np.array([distance, variance])

    def trace(
        self,
        num_points: int = 50,
        callback: Callable[[FrontierPoint, int, int], bool] | None = None,
        n_jobs: int = 1,
    ) -> EfficientFrontierResults:
        """Trace the bias/variance tradeoff.

        Solves the optimization problem for ``num_points`` values of φ between the two
        Pareto corners. Corner 0 minimizes D(w, v) (no variance constraint); corner 1
        minimizes (1/M) ‖w‖₂² (no distance constraint).

        Parameters
        ----------
         num_points : int, default=50
            Number of grid points along the φ dimension.
         callback : callable or None, optional
            Forwarded to the parent ``trace``.
         n_jobs : int, default=1
            Forwarded to the parent ``trace``; ``n_jobs != 1`` sweeps the grid across
            worker processes.

        Returns
        -------
         EfficientFrontierResults
            ``corners[0]`` minimizes D(w, v); ``corners[1]`` minimizes variance. The
            ``.weights``, ``.distances``, and ``.variances`` convenience properties are
            sorted by variance ascending.

        """
        fr = super().trace(num_points=num_points, callback=callback, n_jobs=n_jobs)
        return EfficientFrontierResults(
            points=fr.points,
            corners=fr.corners,
            primary_objective=fr.primary_objective,
            n_attempted=fr.n_attempted,
            n_skipped=fr.n_skipped,
            # Not optional bookkeeping: drop it and every envelope
            # downstream quietly loses its lower bound.
            tangent_slopes=fr.tangent_slopes,
        )


class ExternalValidityVarianceFrontierResults(FrontierResults):
    """Pareto frontier results for the external-validity/variance tradeoff.

    Subclasses ``FrontierResults``; inherits ``knee()``.

    Reports **effective sample size** rather than the variance inflation factor the
    optimizer works in. The two are the same quantity: both experiment groups are pinned
    to mean weight one, so ``sum(w) = M`` and Kish's ``ESS = (sum w)^2 / sum w^2 = M /
    mean(w^2)``. ESS is often more interpratable than the variance inflation factor.
    ``plot`` will also draw it in the optimizer's own coordinates
    (``variance_space="vif"``) or in their square root (``"se"``), the two spaces where
    the frontier is provably convex; see ``plot``.

    The ``.weights``, ``.effective_sample_sizes``, and ``.imbalances`` properties all
    return lists sorted by imbalance ascending (tight balance first), so that entry
    ``i`` of each refers to the same frontier point.

    Parameters
    ----------
     knee_space : {"vif", "se"}, default="se"
        Which coordinate a band chosen off this frontier was chosen in, so ``plot`` can
        annotate the bend the calibration actually used instead of asking the caller to
        remember. A constructor argument rather than an attribute set afterwards, for
        the reason ``tangent_slopes`` is one: it is state not implied by the points, and
        a rebuild that drops it fails silently -- here by marking a band nobody used,
        which looks exactly like a correct plot.
     seam : float or None
        The smallest band every external row can meet, when the frontier was traced past
        it (see ``ExternalValidityVarianceFrontier``). Below it a point's imbalance
        coordinate is the band it was solved at, not its max imbalance, and ``plot``
        marks the seam. None when the frontier stops at the seam or the bands can be met
        all the way down.

    """

    def __init__(
        self,
        points: list[FrontierPoint],
        corners: list[FrontierPoint],
        primary_objective: int,
        n_attempted: int,
        n_skipped: int,
        tangent_slopes: npt.NDArray[np.float64] | None = None,
        knee_space: Literal["vif", "se"] = "se",
        seam: float | None = None,
    ) -> None:
        super().__init__(
            points=points,
            corners=corners,
            primary_objective=primary_objective,
            n_attempted=n_attempted,
            n_skipped=n_skipped,
            tangent_slopes=tangent_slopes,
        )
        self.knee_space: Literal["vif", "se"] = knee_space
        self.seam: float | None = seam

    @property
    def _points_by_imbalance(self) -> list[FrontierPoint]:
        """Return the points sorted by imbalance ascending."""
        return sorted(self.points, key=lambda p: float(p.objectives[1]))

    @staticmethod
    def _effective_sample_size(point: FrontierPoint) -> float:
        """Compute Kish's ESS at a frontier point.

        ``M`` is read off the solution rather than stored, so the results object stays
        constructible from a bare ``FrontierResults``.

        Parameters
        ----------
         point : FrontierPoint
            The frontier point.

        Returns
        -------
         ess : float
            ``len(point.solution) / objectives[0]``.

        """
        return len(point.solution) / float(point.objectives[0])

    @property
    def weights(self) -> list[npt.NDArray[np.float64]]:
        """Return the weights at each frontier point, sorted by imbalance ascending.

        Returns
        -------
         weights : list[npt.NDArray[np.float64]]
            One weight vector per point.

        """
        return [p.solution for p in self._points_by_imbalance]

    @property
    def effective_sample_sizes(self) -> list[float]:
        """Return Kish's ESS at each point, sorted by imbalance ascending.

        Returns
        -------
         ess : list[float]
            One effective sample size per point.

        """
        return [self._effective_sample_size(p) for p in self._points_by_imbalance]

    @property
    def imbalances(self) -> list[float]:
        """Return the max external covariate imbalance at each point, sorted ascending.

        Past the seam (see ``seam``), the band the point was solved at.

        Returns
        -------
         imb : list[float]
            One imbalance per point.

        """
        return [float(p.objectives[1]) for p in self._points_by_imbalance]

    def _variance_coordinate(
        self, point: FrontierPoint, variance_space: Literal["vif", "se", "ess"]
    ) -> float:
        """Return the abscissa of a frontier point in the requested variance space.

        Parameters
        ----------
         point : FrontierPoint
            The frontier point.
         variance_space : {"vif", "se", "ess"}
            Which variance coordinate to return.

        Returns
        -------
         x : float
            The point's abscissa in ``variance_space``.

        """
        if variance_space == "vif":
            return float(point.objectives[0])
        if variance_space == "se":
            return math.sqrt(float(point.objectives[0]))
        return self._effective_sample_size(point)

    def _variance_axis_value(
        self, vif: float, variance_space: Literal["vif", "se", "ess"]
    ) -> float:
        """Restate a variance inflation factor in the requested variance space.

        The objective-vector counterpart of ``_variance_coordinate``, for an annotation
        that has no weights of its own. ``M`` comes from the traced points, which all
        share it.

        Parameters
        ----------
         vif : float
            Variance inflation factor.
         variance_space : {"vif", "se", "ess"}
            Which variance coordinate to return.

        Returns
        -------
         x : float
            ``vif`` restated in ``variance_space``.

        Raises
        ------
         ValueError
            When ``vif`` is not positive and the space is not ``"vif"``.

        """
        if variance_space == "vif":
            return vif
        if vif <= 0.0:
            raise ValueError(
                f"A variance inflation factor must be positive, got {vif:g}."
            )
        if variance_space == "se":
            return math.sqrt(vif)
        return float(len(self.points[0].solution)) / vif

    @staticmethod
    def _variance_axis_name(variance_space: Literal["vif", "se", "ess"]) -> str:
        """Return what to call the x-axis, used for both the label and the title.

        Parameters
        ----------
         variance_space : {"vif", "se", "ess"}
            Which variance coordinate is drawn.

        Returns
        -------
         name : str
            The axis name.

        """
        if variance_space == "vif":
            return "Variance Inflation Factor"
        if variance_space == "se":
            return "Standard Error Inflation Factor"
        return "Effective Sample Size"

    @staticmethod
    def _root(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Take the square root of a variance ordinate, floored at zero.

        The interpolant's lower bound can dip below zero where the frontier is nearly
        flat, which is vacuous rather than wrong -- a variance inflation factor cannot
        be negative. Flooring tightens the bound to something true instead of handing
        ``np.sqrt`` a negative and drawing a gap in the band.

        Parameters
        ----------
         values : npt.NDArray[np.float64]
            Variance ordinates.

        Returns
        -------
         root : npt.NDArray[np.float64]
            The floored square roots.

        """
        return np.sqrt(np.maximum(values, 0.0))

    def plot(
        self,
        annotate_knee: bool = True,
        ax: plt.Axes | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        title: str | None = None,
        variance_space: Literal["vif", "se", "ess"] = "ess",
        knee_space: Literal["vif", "se"] | None = None,
        annotations: Mapping[str, AnnotationPoint] | None = None,
        marker: str = "*",
        marker_size: float = 10,
        marker_color: str = "orange",
        annotation_size: float | str | None = None,
    ) -> plt.Axes:
        """Plot the external-validity/variance frontier.

        Does not delegate to ``FrontierResults.plot``, which always plots the primary
        objective itself. Here the x-axis is selectable: ``"vif"`` is that objective,
        ``"se"`` is its square root, and ``"ess"`` is a reciprocal transform of it. The
        interpolant is carried across the transform rather than redrawn -- see
        ``_draw_frontier``.

        VIF and SE run the same way across the page, since the square root is
        increasing; ESS is their mirror image, since ``ESS = M / VIF`` is decreasing, so
        the curve that falls from top-left to bottom-right in VIF units rises from
        bottom-left to top-right in ESS units.

        The three are not interchangeable, and the difference is not only cosmetic. The
        frontier is convex in VIF and in SE -- the square root of a squared norm is a
        norm, and a value function under a convex objective is convex -- but not in ESS,
        where a reciprocal destroys it. So a bend read off an ESS panel is not the same
        quantity as the bend the knee machinery solves for; see ``knee_space``, and pass
        the two consistently if the marker is meant to be believed.

        Parameters
        ----------
         annotate_knee : bool, default=True
            If True, mark the knee with crosshairs and its dual-slope tangent.
         ax : matplotlib Axes, optional
            If provided, plot onto this Axes; otherwise create a new figure.
         variance_space : {"vif", "se", "ess"}, default="ess"
            Which variance coordinate to put on the x-axis. ``"vif"`` plots the variance
            inflation factor (the design effect) the optimizer minimizes; ``"se"`` plots
            its square root, the factor by which a standard error is inflated (Kish's
            DEFT), which is the version that enters a confidence interval; ``"ess"``
            plots Kish's effective sample size, which is the version a reader can price.
         x_label : str, optional
            Defaults to the name of the chosen variance space.
         y_label : str, optional
            Defaults to "Covariate Imbalance with Target Population", or, on a frontier
            traced past the seam, to a label saying that below the seam the coordinate
            is the band rather than the imbalance.
         title : str, optional
            Defaults to "External Validity vs <the chosen variance space>".
         knee_space : {"vif", "se"}, optional
            Which coordinate the annotated bend is measured in, independent of which one
            is drawn. Defaults to the space this frontier was traced for, so a band
            chosen in SE coordinates is annotated in SE coordinates without the caller
            having to say so twice. Override only to compare the two. See
            ``knee_external_scale``.

            Read only on a frontier whose knee has not been refined. Once
            ``refine_knee`` has run, the knee it solved for is the one marked, and it
            already carries the space it was solved in -- see ``_annotate_knee``.
         annotations : dict, optional
            Labelled reference points to mark, as ``{label: point}``. A point is a
            ``FrontierPoint`` or an objective vector ``[variance inflation factor, max
            external imbalance]``, carried into ``variance_space`` like the curve. The
            corners are ``results.corners[1]``, the tightest external balance attainable
            (exact balance when it is feasible), and ``results.corners[0]``, no external
            balance at all. For weights solved some other way, pass
            ``frontier.evaluate_objectives(w)``; for Cochran's band that is
            ``frontier.evaluate_objectives(calib.solve_with_fallback().solution)``.
         marker : str, default="*"
            Marker style for the annotations.
         marker_size : float, default=10
            Marker size for the annotations.
         marker_color : str, default="orange"
            Marker color for the annotations.
         annotation_size : float | str | None, optional
            Label font size for the annotations, or None for large.

        Returns
        -------
         ax : matplotlib Axes
            The axes with the frontier drawn.

        Raises
        ------
         ValueError
            If the frontier has no points, ``variance_space`` is not one of the three
            recognized values, or an annotation is not a ``FrontierPoint`` or a finite
            2-entry objective vector.

        """
        if variance_space not in ("vif", "se", "ess"):
            raise ValueError(
                f"variance_space must be 'vif', 'se' or 'ess', got {variance_space!r}."
            )
        if not self.points:
            raise ValueError("No frontier points to plot.")
        # Before any drawing, so a bad annotation leaves no stray figure.
        coordinates: dict[str, tuple[float, float]] = {}
        for label, point in (annotations or {}).items():
            objectives = _annotation_objectives(label, point)
            coordinates[label] = (
                self._variance_axis_value(float(objectives[0]), variance_space),
                float(objectives[1]),
            )

        if ax is None:
            _, ax = plt.subplots()

        pts = self._points_by_imbalance
        xs = [self._variance_coordinate(p, variance_space) for p in pts]
        ys = [float(p.objectives[1]) for p in pts]

        fx, fy = self._draw_frontier(ax, xs, ys, variance_space)
        # Shade under the drawn curve, not the raw points: filling from the
        # polyline leaves the region's boundary kinked at whichever sample sits
        # nearest the bend, and it coincides closely enough with the curve that
        # the artifact reads as the curve itself.
        ax.fill_between(fx, fy, max(fy), color="lightblue", alpha=0.5)
        if self.seam is not None and min(ys) < self.seam < max(ys):
            # Below this line the curve is the band, with the rows that cannot
            # meet it relaxed; above it, the band holds on every row.
            ax.axhline(self.seam, color="gray", linestyle=":", linewidth=1)
            ax.annotate(
                "band met on every row above",
                xy=(1.0, self.seam),
                xycoords=("axes fraction", "data"),
                xytext=(-4, 3),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize="x-small",
                color="gray",
            )
        # Before the knee: its crosshairs freeze the axis limits.
        _draw_annotations(
            ax,
            coordinates,
            xs,
            marker=marker,
            marker_size=marker_size,
            marker_color=marker_color,
            annotation_size=annotation_size,
        )

        if annotate_knee and len(pts) >= 2:
            self._annotate_knee(
                ax,
                pts,
                xs,
                ys,
                variance_space,
                self.knee_space if knee_space is None else knee_space,
            )

        axis_name = self._variance_axis_name(variance_space)
        ax.set_xlabel(axis_name if x_label is None else x_label)
        if y_label is None:
            y_label = "Covariate Imbalance with Target Population"
            if self.seam is not None:
                y_label = "Imbalance with Target (band \u03c6 below the dotted line)"
        ax.set_ylabel(y_label)
        ax.set_title(f"External Validity vs {axis_name}" if title is None else title)
        return ax

    def _draw_frontier(
        self,
        ax: plt.Axes,
        xs: list[float],
        ys: list[float],
        variance_space: Literal["vif", "se", "ess"] = "ess",
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Draw the certified interpolant in the requested variance space.

        ``FrontierResults.envelope`` works in the optimizer's own coordinates --
        imbalance on ``x``, variance on ``curve`` -- which is exactly the ``"vif"``
        space, so that branch draws it as-is. The other two carry the curve across their
        transform rather than replotting it.

        Whether the bounds swap is decided by the sign of the transform, not by the
        transform being nonlinear. ``SE = sqrt(VIF)`` is increasing, so the tangent
        still bounds below and the chord still above. ``ESS = M / VIF`` is decreasing,
        so it **swaps them**: an upper bound on variance is a LOWER bound on effective
        sample size. (The same swap ``robust_optimization`` documents for its negation
        into performance coordinates.)

        Without a certified band this still beats the polyline, which is biased rather
        than merely coarse: the chord lies above a convex frontier, so straight segments
        between sparse samples invent a vertex that the true curve does not have.

        Parameters
        ----------
         ax : plt.Axes
            Axes to draw onto.
         xs : list[float]
            Drawn abscissae, one per frontier point.
         ys : list[float]
            Drawn ordinates, one per frontier point.
         variance_space : {"vif", "se", "ess"}
            Which variance coordinate is drawn.

        Returns
        -------
         curve : npt.NDArray[np.float64]
            The drawn curve abscissae.
         ordinates : npt.NDArray[np.float64]
            The drawn curve ordinates.

        """
        try:
            # The envelope grids uniformly in imbalance, and ESS = M / VIF
            # stretches the low-imbalance end of that grid across a wide span of
            # sample size. A denser grid costs only spline evaluations, and
            # without it the transformed curve renders as visible facets exactly
            # where the frontier turns hardest.
            env = self.envelope(num=2000)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as e:
            logger.debug("Falling back to the polyline: envelope() raised %s", e)
            ax.plot(xs, ys)
            return np.asarray(xs), np.asarray(ys)

        if variance_space == "vif":
            ax.plot(env.curve, env.x)
            if np.isfinite(env.max_band_width):
                # tangent bounds variance below, chord above; no swap here.
                ax.fill_betweenx(
                    env.x, env.tangent, env.chord, color="grey", alpha=0.25
                )
            return env.curve, env.x

        if variance_space == "se":
            curve = self._root(env.curve)
            ax.plot(curve, env.x)
            if np.isfinite(env.max_band_width):
                # Increasing transform, so the bounds keep their roles.
                ax.fill_betweenx(
                    env.x,
                    self._root(env.tangent),
                    self._root(env.chord),
                    color="grey",
                    alpha=0.25,
                )
            return curve, env.x

        m = float(len(self.points[0].solution))
        with np.errstate(divide="ignore", invalid="ignore"):
            ax.plot(m / env.curve, env.x)
            if np.isfinite(env.max_band_width):
                # chord bounds variance above, so it bounds ESS below.
                ax.fill_betweenx(
                    env.x, m / env.chord, m / env.tangent, color="grey", alpha=0.25
                )

            return m / env.curve, env.x

    @staticmethod
    def _secant_bracket(ys: list[float], ky: float) -> tuple[int, int] | None:
        """Find two sample indices straddling ``ky`` whose ordinates actually differ.

        ``ys`` is the imbalance of each drawn point, ascending, which is what makes the
        knee's own imbalance enough to locate it among them -- bracketing rather than
        matching a point identically is what lets an off-grid knee be annotated at all.

        Two things the plain insertion point gets wrong.

        **A knee that lands on a sample needs a centred bracket.** An unrefined knee
        *is* one of the drawn points, so an insertion point alone yields ``(k-1, k)``: a
        backward difference where a central one is available, biased by half a grid step
        on a curve whose whole subject is curvature.

        **Ties must be stepped over.** Several sweep bands can saturate at one achieved
        imbalance, and a bracket whose two ordinates are equal reports a slope of zero
        -- a fact about the grid, not about the frontier. The caller's ``abs(dx) > 0``
        guard cannot catch that, since the abscissae still differ.

        Both fall out of asking for the last sample strictly *below* ``ky`` and the
        first strictly *above* it, which is what the two bisections return:
        ``bisect_left`` is the first index not below, so one before it is the last
        below; ``bisect_right`` is the first index above. A knee on a sample is then
        straddled rather than leaned on, and any run of ties at ``ky`` is skipped on
        both sides at once. Where no such pair exists -- at the two ends, or on a
        frontier flat throughout -- there is nothing true to draw, so None, and the
        caller draws no tangent.

        Parameters
        ----------
         ys : list[float]
            Ascending imbalance of each drawn point.
         ky : float
            The knee's imbalance.

        Returns
        -------
         bracket : tuple[int, int] | None
            The straddling ``(lo, hi)`` indices, or None where no such pair exists.

        """
        lo = bisect.bisect_left(ys, ky) - 1
        hi = bisect.bisect_right(ys, ky)
        return (lo, hi) if lo >= 0 and hi < len(ys) else None

    def _knee_tangent_slope(
        self,
        knee_pt: FrontierPoint,
        variance_space: Literal["vif", "se", "ess"] = "ess",
    ) -> float | None:
        """Compute the frontier's slope at the knee, from the knee's own duals.

        ``tangent_slopes`` holds ``s = d(VIF)/d(phi)`` at each point, read off the
        imbalance multipliers by the envelope theorem. A refined knee is folded into
        ``points`` together with its slope by ``absorb_probe``, and a sampled knee is
        one of the points already, so either is found by matching objective vectors
        under the tolerance the trace dedupes by.

        The plot puts phi on the y-axis and a variance coordinate x on the x-axis, so
        the slope it draws is d(phi)/dx = 1 / (dx/dphi), where

           vif:  dx/dphi = s,
           se:   dx/dphi = s / (2 √VIF),
           ess:  dx/dphi = -M s / VIF².

        Parameters
        ----------
         knee_pt : FrontierPoint
            The knee point.
         variance_space : {"vif", "se", "ess"}
            Which variance coordinate is drawn.

        Returns
        -------
         slope : float | None
            None when the knee carries no slope (no duals, a corner, or slopes
            misaligned with the points) or the slope is zero, which would be a vertical
            line. The caller then falls back to a secant.

        """
        slopes = self.tangent_slopes
        if slopes is None or len(slopes) != len(self.points):
            return None

        s = math.nan
        for point, slope in zip(self.points, slopes, strict=True):
            if np.allclose(point.objectives, knee_pt.objectives, atol=1e-8, rtol=0):
                s = float(slope)
                break
        if not math.isfinite(s) or s == 0.0:
            return None

        vif = float(knee_pt.objectives[0])
        if variance_space == "vif":
            dx_dphi = s
        elif variance_space == "se":
            if vif <= 0.0:
                return None
            dx_dphi = s / (2.0 * math.sqrt(vif))
        else:
            M = float(len(knee_pt.solution))
            dx_dphi = -M * s / (vif * vif)

        slope = 1.0 / dx_dphi
        return slope if math.isfinite(slope) else None

    def _annotate_knee(
        self,
        ax: plt.Axes,
        pts: list[FrontierPoint],
        xs: list[float],
        ys: list[float],
        variance_space: Literal["vif", "se", "ess"] = "ess",
        knee_space: Literal["vif", "se"] = "se",
    ) -> None:
        """Mark the knee with crosshairs and the tangent its duals report.

        **Which knee gets marked.** If ``refine_knee`` has been run on this frontier --
        which it has on every calibration that went through ``knee_external_scale`` --
        the marker goes on the knee it solved for, because that is the one the band came
        from. Only an unrefined frontier falls back to ``knee``, the sampled argmax.

        The distinction is not cosmetic. The refined knee is off-grid by construction:
        the sweep grids uniformly in imbalance and has no reason to put a sample at the
        bend, so the sampled argmax is the nearest grid point to an answer that lies
        between two of them. Marking that instead draws a crosshair at a band nobody
        used, off by up to half a grid step and by more where the bend is shallow.

        A refinement that ran and reported *no* knee leaves the plot unannotated. That
        case is a straight frontier, where the sampled argmax is fixed by differences
        near the noise floor -- precisely the number ``refine_knee`` declined to report
        -- and drawing it would dress up an arbitrary point as a decision. The
        calibration falls back to the conventional band there, which is not a point on
        this curve.

        The tangent is the knee's own: its slope comes from the imbalance duals of the
        solve that produced the knee (see ``_knee_tangent_slope``), restated in
        whichever variance space is drawn. Only a knee carrying no dual slope falls back
        to a secant across the two samples ``_secant_bracket`` picks out, computed from
        the ``xs`` the caller is drawing so that it matches the plot in every variance
        space. A degenerate frontier -- too few points, or corners the knee calculation
        rejects -- is left unannotated rather than raising, since the curve itself is
        still worth seeing.

        Parameters
        ----------
         ax : plt.Axes
            Axes to draw onto.
         pts : list[FrontierPoint]
            The drawn frontier points.
         xs : list[float]
            Drawn abscissae, one per point.
         ys : list[float]
            Drawn ordinates, one per point.
         variance_space : {"vif", "se", "ess"}
            Which variance coordinate is drawn.
         knee_space : {"vif", "se"}
            Which coordinate the bend is measured in.

        """
        if self.knee_refinement_ran:
            if self.refined_knee is None:
                return
            knee_pt = self.refined_knee
        else:
            try:
                knee_pt = self.knee(_cvxium_knee_space(knee_space))
            except ValueError:
                return

        kx = self._variance_coordinate(knee_pt, variance_space)
        ky = float(knee_pt.objectives[1])

        slope = self._knee_tangent_slope(knee_pt, variance_space)
        if slope is None:
            bracket = self._secant_bracket(ys, ky)
            if bracket is not None:
                lo, hi = bracket
                dx = xs[hi] - xs[lo]
                if abs(dx) > 0:
                    slope = (ys[hi] - ys[lo]) / dx

        if slope is not None:
            width = max(min(0.5 * (kx - min(xs)), 0.5 * (max(xs) - kx)), 0.0)
            x_vals = np.linspace(kx - width, kx + width)
            ax.plot(x_vals, ky + slope * (x_vals - kx), color="orange")

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.hlines(y=ky, xmin=xlim[0], xmax=kx, colors="orange", linestyles="dashed")
        ax.vlines(x=kx, ymin=ylim[0], ymax=ky, colors="orange", linestyles="dashed")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


class ExternalValidityVarianceFrontier(MultiObjectiveOptimizer):
    """Trace the external-validity/variance Pareto frontier for a joint calibration.

    The two-block analogue of the single-population frontier. Where that class trades a
    single population's covariate imbalance against variance, this one calibrates two
    experiment groups jointly and trades their *external* validity -- each group against
    the target population -- against variance, holding *internal* validity -- the two
    groups against each other -- fixed at ``psi3``.

    The two objectives are:
       minimize    ((1/M) ‖w‖₂², ‖Zᵀw - ν‖∞)
       subject to  mean(w1) = 1,
                   mean(w2) = 1,
                   w1 ≥ lb1,
                   w2 ≥ lb2,
                   ‖(1/M2) Z4ᵀw2 - (1/M1) Z3ᵀw1‖∞ ≤ ψ3,

    traced by moving the second objective into a constraint with threshold φ:
       minimize    (1/M) ‖w‖₂²
       subject to  mean(w1) = 1,
                   mean(w2) = 1,
                   w1 ≥ lb1,
                   w2 ≥ lb2,
                   ‖(1/M2) Z4ᵀw2 - (1/M1) Z3ᵀw1‖∞ ≤ ψ3,
                   ‖Zᵀw - ν‖∞ ≤ φ.

    In this notation Z = [Z1 / √M1, Z2 / √M2] and ν = [√M1 ν1; √M2 ν2], so ‖Zᵀw - ν‖∞ ≤
    φ is equivalent to the pair of per-group bands
       ‖(1/M1) Z1ᵀw1 - ν1‖∞ ≤ φ / √M1 =: ψ1, and
       ‖(1/M2) Z2ᵀw2 - ν2‖∞ ≤ φ / √M2 =: ψ2,
    which is why a single scalar φ suffices to sweep both groups at once.

    Parameters
    ----------
     Z1, Z2 : npt.NDArray[np.float64]
        External-validity covariates for group 1 (control/Holdout) and group 2
        (treated/Marketing), of shape ``(M1, q)`` and ``(M2, q)``. Whiten these by the
        population standard deviation, as elsewhere in PyRake, so that φ / √M reads
        directly as an allowed standardized mean difference.
     Z3, Z4 : npt.NDArray[np.float64]
        Internal-validity covariates for the two groups, same shapes. Usually the same
        matrices as ``Z1``, ``Z2``.
     nu1, nu2 : npt.NDArray[np.float64]
        Target population means for ``Z1`` and ``Z2``, of shape ``(q,)``, whitened the
        same way.
     min_weight1, min_weight2 : float or npt.NDArray[np.float64]
        Lower bound on the weights of each group. Weights below M_b / N imply a
        propensity score above one, so ``M1 / N`` and ``M2 / N`` are the principled
        floors.
     psi3 : float, optional
        Internal-validity tolerance, held fixed for the whole sweep. Defaults to ``1.5 *
        sqrt(1 / M1 + 1 / M2)``, the two-sample analogue of Rake's default and the same
        value ``JointCalibrator`` would choose.

    Raises
    ------
     ValueError
        If ``Z1``/``Z2`` disagree with ``Z3``/``Z4`` on the number of rows, or if the
        four do not share a column count. The external and internal blocks are reported
        against one shared covariate list, so they must describe the same covariates in
        the same order.

    Notes
    -----
    The motivation is ``JointCalibrator.solve_with_fallback``, which aims for phi = 1.5
    and, when that is infeasible, falls back to minimizing phi. That fallback silently
    selects the far end of this frontier, where the last few percent of covariate
    balance can cost most of the effective sample size. Tracing the frontier makes the
    choice explicit: ``results.knee()`` is typically a far better trade than the
    minimize-phi corner.

    The sweep uses ``SquaredL2(v=0)``, so it reports the minimum-variance weighting at
    each phi rather than the minimum-divergence one. Because both groups are pinned to
    mean weight one, minimizing ``mean(w^2)`` is exactly maximizing Kish's effective
    sample size: ``ESS = M / mean(w^2)``.

    **Past the seam.** The band cannot always be tightened to zero. Below some phi --
    the *seam*, ``seam()`` -- no weighting meets it on every row: a level with no
    respondents in one group, say, or a group whose shares must sum to one while its
    targets do not. With ``continue_past_seam`` (the default) the frontier does not stop
    there. Below the seam a point is what ``JointCalibrator.solve_with_fallback``
    returns at that band: the rows that cannot meet it are relaxed, holding their summed
    excess to the least achievable, and the rest are held to it. The frontier's
    imbalance coordinate is then the band, not the max imbalance, which stays pinned at
    the seam (see ``point_objectives``).

    It runs on down to the *floor*, ``floor()``: the tightest band at which the rows
    that can meet it can all do so at once. Below the floor, ``solve_with_fallback``
    would have to relax a row that could meet the band on its own and pin the rest at
    the band's edge, so it widens the band back to the floor instead. A group whose
    shares must sum to one has a floor whenever one of its levels is empty: the observed
    levels must take up the empty one's share, within their bands. The tight corner sits
    at the floor, and the knee may land on either side of the seam; a band between the
    two handed to ``solve_with_fallback`` reproduces the point the frontier traced.
    Where no row is unattainable -- the band simply cannot be met on every row at once
    -- the floor is the seam and there is nothing past it. Below the seam the curve is a
    two-stage solve, so it has no dual slope (those points report ``nan``) and need not
    be convex.

    """

    def __init__(
        self,
        Z1: npt.NDArray[np.float64],
        Z2: npt.NDArray[np.float64],
        Z3: npt.NDArray[np.float64],
        Z4: npt.NDArray[np.float64],
        nu1: npt.NDArray[np.float64],
        nu2: npt.NDArray[np.float64],
        min_weight1: float | npt.NDArray[np.float64],
        min_weight2: float | npt.NDArray[np.float64],
        psi3: float | npt.NDArray[np.float64] | None = None,
        engine: Literal["ipm", "pd"] = "pd",
        continue_past_seam: bool = True,
    ) -> None:
        """Create an ExternalValidityVarianceFrontier. See the class docstring.

        ``continue_past_seam=False`` restores the frontier that stops at the seam, with
        its tight corner at the smallest band every row can meet.

        """
        super().__init__(primary_objective=0)
        if Z1.shape[0] != Z3.shape[0] or Z2.shape[0] != Z4.shape[0]:
            raise ValueError("Z1/Z2 must have the same number of rows as Z3/Z4.")

        columns = {Z.shape[1] for Z in (Z1, Z2, Z3, Z4)}
        if len(columns) != 1:
            raise ValueError(
                "Z1, Z2, Z3 and Z4 must all have the same number of columns; got "
                f"{sorted(columns)}. The external and internal blocks are reported "
                "against one shared covariate list, so they must describe the same "
                "covariates in the same order."
            )

        self.Z1: npt.NDArray[np.float64] = Z1
        self.Z2: npt.NDArray[np.float64] = Z2
        self.Z3: npt.NDArray[np.float64] = Z3
        self.Z4: npt.NDArray[np.float64] = Z4
        self.nu1: npt.NDArray[np.float64] = nu1
        self.nu2: npt.NDArray[np.float64] = nu2
        self.min_weight1: float | npt.NDArray[np.float64] = min_weight1
        self.min_weight2: float | npt.NDArray[np.float64] = min_weight2
        self.M1: int = Z3.shape[0]
        self.M2: int = Z4.shape[0]
        self.psi3: float | npt.NDArray[np.float64] = (
            1.5 * float(np.sqrt(1 / self.M1 + 1 / self.M2)) if psi3 is None else psi3
        )
        self.engine: Literal["ipm", "pd"] = engine
        self.continue_past_seam: bool = continue_past_seam
        # Memo for _imbalance_multiplier_span; see that method for why one
        # answer serves the whole instance.
        self._span: tuple[int | None, int | None] | None = None
        # The seam and the linear program that found it, memoized for the same
        # reason: one problem, one answer.
        self._seam: float | None = None
        self._seam_corner: SolveResult | None = None
        # The floor and the calibration there, memoized. The tight corner past
        # the seam is that calibration, so point_objectives can report its band
        # rather than its max imbalance.
        self._floor: float | None = None
        self._floor_result: SolveResult | None = None
        self._tight_corner: SolveResult | None = None
        self._tight_band: float = 0.0

    @classmethod
    def from_calibrator(
        cls,
        rake: JointCalibrator,
        psi3: float | npt.NDArray[np.float64] | None = None,
        engine: Literal["ipm", "pd"] | None = "pd",
        continue_past_seam: bool = True,
    ) -> ExternalValidityVarianceFrontier:
        """Build the frontier for a configured joint calibration.

        Lives here rather than on ``JointCalibrator`` so the dependency stays one-way:
        this module imports ``rake``, and nothing in ``rake`` needs to know that
        frontiers exist.

        Parameters
        ----------
         rake : JointCalibrator
            Source of the covariate blocks and weight floors.
         psi3 : float or ndarray, optional
            Internal-validity band to hold fixed. Defaults to the calibrator's own, per
            row, so a heterogeneous internal band is carried through rather than
            collapsed to its first entry.
         engine : {"ipm", "pd"} or None, optional
            Inner solver for the sweep, matching ``EfficientFrontier``: ``None``
            inherits whatever ``rake`` was built with. Defaults to ``"pd"`` -- on this
            shape (pure joint calibration, active l-infinity bands, no exact-balance
            block) PD measured ~4x faster than the barrier cold, 21ms against 88ms, and
            it warm-starts across the sweep where the barrier cannot.
         continue_past_seam : bool, default=True
            See the class docstring.

        Returns
        -------
         frontier : ExternalValidityVarianceFrontier
            The frontier for the calibrator's blocks and floors.

        Raises
        ------
         ValueError
            If ``rake`` has no external-validity blocks, so there is no tradeoff to
            trace.

        """
        if (
            rake._Z1 is None
            or rake._Z2 is None
            or rake._Z3 is None
            or rake._Z4 is None
            or rake._nu1 is None
            or rake._nu2 is None
        ):
            raise ValueError(
                "from_calibrator() needs Z1/nu1, Z2/nu2 (external validity) and "
                "Z3/Z4 (internal validity); this calibrator was built without "
                "them, so there is no tradeoff to trace."
            )

        full = _full_psi(rake)
        if psi3 is None and full is not None:
            q_ext = (rake._q1 or 0) + (rake._q2 or 0)
            internal = full[q_ext:]
            psi3 = internal if internal.size else None

        min_weight = np.atleast_1d(np.asarray(rake.min_weight, dtype=np.float64))
        if min_weight.shape[0] == 1:
            min_weight = np.full(rake.dimension, float(min_weight[0]))

        return cls(
            Z1=rake._Z1,
            Z2=rake._Z2,
            Z3=rake._Z3,
            Z4=rake._Z4,
            nu1=rake._nu1,
            nu2=rake._nu2,
            min_weight1=min_weight[: rake._M1],
            min_weight2=min_weight[rake._M1 :],
            psi3=psi3,
            engine=rake.engine if engine is None else engine,
            continue_past_seam=continue_past_seam,
        )

    def solve_with_bounds(
        self, bounds: npt.NDArray[np.float64], x0: npt.NDArray[np.float64] | None = None
    ) -> SolveResult:
        """Minimize (1/M) ‖w‖₂² subject to max external imbalance <= bounds[0].

        Parameters
        ----------
         bounds : npt.NDArray[np.float64]
            Array of shape ``(1,)``; ``bounds[0]`` is φ, which splits into the per-group
            bands ``ψ1 = φ / √M1`` and ``ψ2 = φ / √M2``.
         x0 : npt.NDArray[np.float64] or None, optional
            Warm-start weights of length ``M1 + M2``. ``MultiObjectiveOptimizer``
            threads the previous grid point through ``trace``, which is where the
            primal-dual engine earns its keep.

        Returns
        -------
         res : SolveResult
            The minimum-variance weights within the band.

        Raises
        ------
         ProblemInfeasibleError
            If no weighting meets the band and the frontier does not continue past the
            seam, or internal validity cannot be met. ``trace`` catches this and drops
            the grid point, so an infeasible stretch shows up as a gap in the frontier
            rather than an exception.

        Notes
        -----
        Below the seam, and with ``continue_past_seam``, this is ``solve_with_fallback``
        at the band: see the class docstring. ``x0`` is not used there.

        """
        phi = float(bounds[0])
        calibrator = self._bounded_calibrator(phi)
        if not self.continue_past_seam:
            return calibrator.solve(x0=x0)
        seam = self.seam()
        if phi < (1.0 - _SEAM_TOLERANCE) * seam:
            return self._solve_past_the_seam(calibrator)
        # At or just under the seam, try the band as asked: the linear program
        # puts the seam a solver tolerance above the true minimum, and a band
        # in that gap is met on every row.
        try:
            return calibrator.solve(x0=x0)
        except ProblemInfeasibleError:
            if phi >= seam:
                raise
            return self._solve_past_the_seam(calibrator)

    def seam(self) -> float:
        """Return the smallest band every external row can meet.

        Found by the linear program behind the tight corner of a frontier that stops
        there. Memoized: it depends only on the covariates, the targets, the weight
        floors and the internal band.

        Returns
        -------
         s : float
            The seam band.

        """
        if self._seam is None:
            corner = self._minimum_imbalance_corner(None)
            self._seam_corner = corner
            self._seam = float(self.evaluate_objectives(np.asarray(corner.solution))[1])
        return self._seam

    def floor(self) -> float:
        """Return the tightest band past the seam at which nothing has to be forced.

        ``solve_with_fallback`` at a band near zero widens it to exactly this (see the
        class docstring); the band it reports is the floor, and the calibration there is
        the frontier's tight corner. Memoized, like the seam.

        Returns
        -------
         f : float
            The floor band.

        Raises
        ------
         ProblemInfeasibleError
            If internal validity, which the frontier holds fixed, had to be relaxed
            there as well.

        """
        if self._floor is None:
            band = _TIGHT_BAND_FRACTION * self.seam()
            result = self._bounded_calibrator(band).solve_with_fallback()
            if result.fallback_status not in (0, 2):
                raise ProblemInfeasibleError(
                    "Internal validity cannot be met past the seam "
                    f"(fallback status {result.fallback_status}): "
                    f"{result.fallback_message}"
                )
            self._floor_result = result.result
            self._floor = band * result.band_scales.get(_EXTERNAL_LABEL, 1.0)
        return self._floor

    def _continues(self) -> bool:
        """Return whether this frontier runs past its seam.

        Only when asked to and when there is somewhere to go: a seam within a hair of
        zero means the bands can be met all the way down, and a floor at the seam means
        nothing past it can be met without forcing a row that could meet the band on its
        own.

        Returns
        -------
         continues : bool
            True when the trace runs past the seam.

        """
        if not self.continue_past_seam:
            return False
        if self.seam() <= _SEAM_TOLERANCE * self._imbalance_scale():
            return False
        try:
            floor = self.floor()
        except ProblemInfeasibleError as e:
            logger.info("Not tracing past the seam: %s", e)
            return False
        return floor < (1.0 - _SEAM_TOLERANCE) * self.seam()

    def _solve_past_the_seam(self, calibrator: JointCalibrator) -> SolveResult:
        """Return the least-variance weighting at a band not every row can meet.

        Parameters
        ----------
         calibrator : JointCalibrator
            The joint calibration at the band.

        Returns
        -------
         res : SolveResult
            The fallback calibration's underlying solve.

        Raises
        ------
         ProblemInfeasibleError
            If internal validity, which the frontier holds fixed, had to be relaxed as
            well, or if the band is below the floor: ``solve_with_fallback`` then
            calibrates at a wider band than the one asked for, which is not this point.

        """
        result = calibrator.solve_with_fallback()
        if result.fallback_status not in (0, 2):
            raise ProblemInfeasibleError(
                "Internal validity cannot be met at this band "
                f"(fallback status {result.fallback_status}): "
                f"{result.fallback_message}"
            )
        scale = result.band_scales.get(_EXTERNAL_LABEL, 1.0)
        if scale > 1.0:
            raise ProblemInfeasibleError(
                f"The band is below the floor: solve_with_fallback widened it by "
                f"a factor of {scale:.3g}."
            )
        return result.result

    def point_objectives(
        self, result: SolveResult, bounds: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Record the objectives for a point: the band, past the seam.

        Above the seam the band binds, so it equals the max imbalance, or the weights
        balance better than the band and their own max imbalance is the coordinate --
        ``evaluate_objectives`` either way. Below it the max imbalance is pinned at the
        seam by the rows that cannot move, so the coordinate is the band the point was
        solved at. The tight corner is solved at a band too, and reports it.

        Parameters
        ----------
         result : SolveResult
            The point's solve.
         bounds : npt.NDArray[np.float64]
            The band the point was solved at, or non-finite for a corner.

        Returns
        -------
         objectives : npt.NDArray[np.float64]
            ``[variance, band-or-imbalance]``.

        """
        objectives = self.evaluate_objectives(np.asarray(result.solution))
        if np.all(np.isfinite(bounds)):
            objectives[1] = min(float(objectives[1]), float(bounds[0]))
        elif result is self._tight_corner:
            objectives[1] = self._tight_band
        return objectives

    def _bounded_calibrator(self, phi: float) -> JointCalibrator:
        """Build the calibration solved at external-validity scale ``phi``.

        Every grid point differs only in ``phi``, so the constraint layers keep the same
        shapes throughout -- which is what lets ``tangent_slopes`` locate the imbalance
        multipliers from one representative problem.

        Parameters
        ----------
         phi : float
            The external-validity scale; splits into per-group bands.

        Returns
        -------
         calibrator : JointCalibrator
            The joint calibration at ``phi``.

        """
        return JointCalibrator(
            distance=SquaredL2(v=np.zeros(self.M1 + self.M2)),
            Z1=self.Z1,
            Z2=self.Z2,
            Z3=self.Z3,
            Z4=self.Z4,
            nu1=self.nu1,
            nu2=self.nu2,
            psi1=phi / np.sqrt(self.M1),
            psi2=phi / np.sqrt(self.M2),
            psi3=self.psi3,
            min_weight1=self.min_weight1,
            min_weight2=self.min_weight2,
            engine=self.engine,
        )

    def tangent_slopes(
        self, results: FrontierResults
    ) -> npt.NDArray[np.float64] | None:
        """Compute the frontier slope d(variance) / d(φ) from the imbalance duals.

        By the envelope theorem the derivative of the optimum with respect to a
        constraint's right-hand side is minus that constraint's multiplier. Here the
        swept scalar ``phi`` is not one right-hand side: it is spread over the external
        rows as ``ψᵢ = φ / √M_b``, so the slopes have to be aggregated with those
        weights,

            d(variance)/d(phi)
                = -[(1/√M1) Σ_{i in block 1} λᵢ
                   + (1/√M2) Σ_{i in block 2} λᵢ],

        where ``λᵢ`` is the total multiplier on covariate ``i``'s band.
        ``ImbalanceConstraintLayer`` encodes each band as two one-sided rows, the
        positive side then the negative, both carrying ``-ψᵢ``, so ``λᵢ`` is their sum
        -- at most one is active at a time. The internal rows do not move with ``phi``
        and drop out.

        Corners are reported as ``nan``: corner 0 has no external-validity block and
        corner 1 is a linear program over an augmented variable, so neither shares this
        layer layout. A vertical tangent at corner 1 would be ``nan`` in any case.

        Parameters
        ----------
         results : FrontierResults
            The traced frontier, carrying the duals per point.

        Returns
        -------
         slopes : npt.NDArray[np.float64] | None
            None when duals are unavailable throughout, which leaves
            ``FrontierResults.envelope`` with the chord bound alone.

        """
        offset, q = self._imbalance_multiplier_span()
        if offset is None or q is None:
            return None

        q1 = self.Z1.shape[1]
        q2 = self.Z2.shape[1]
        slopes = np.full(len(results.points), np.nan)
        seam = self.seam() if self._continues() else 0.0
        for i, point in enumerate(results.points):
            if not np.all(np.isfinite(point.bounds)):
                continue  # a corner; different problem, different layout
            if float(point.bounds[0]) < seam:
                continue  # past the seam: a two-stage solve, no dual slope

            duals = point.ipm_result.duals
            lam = None if duals is None else duals.inequality_multipliers
            if lam is None or len(lam) < offset + 2 * q:
                continue

            band = np.asarray(lam[offset : offset + 2 * q], dtype=np.float64)
            per_covariate = band[:q] + band[q:]
            slopes[i] = -(
                per_covariate[:q1].sum() / np.sqrt(self.M1)
                + per_covariate[q1 : q1 + q2].sum() / np.sqrt(self.M2)
            )

        return None if np.all(np.isnan(slopes)) else slopes

    def _imbalance_multiplier_span(self) -> tuple[int | None, int | None]:
        """Locate the imbalance layer's multipliers in the stacked vector.

        Multipliers are concatenated in constraint-layer order, so the offset is the
        total size of the layers preceding the imbalance one. Read off a representative
        problem, since every grid point builds the same shapes.

        Memoized for exactly that reason. Reading it off costs a calibrator build and a
        problem assembly, and ``refine_knee`` asks for the slope at every bisection step
        -- so without the memo the layout is rediscovered once per probe, alongside the
        solve the probe is actually there to do. The shapes depend only on ``Z1..Z4``,
        which are fixed at construction.

        Returns
        -------
         span : tuple[int | None, int | None]
            ``(offset, q)`` of the imbalance multipliers, or ``(None, None)`` when there
            is no imbalance layer; a miss is cached too.

        """
        if self._span is not None:
            return self._span

        probe = self._bounded_calibrator(1.0)
        layers = probe._problem.ineq
        offset = 0
        span: tuple[int | None, int | None] = (None, None)
        for layer in layers:
            if isinstance(layer, ImbalanceConstraintLayer):
                span = (offset, layer.q)
                break

            offset += layer.num_constraints(probe.dimension)

        # A miss is cached too: no imbalance layer today means none on the next
        # probe either, and re-assembling to rediscover that is the cost this
        # memo exists to avoid.
        self._span = span
        return span

    def minimize_objective(
        self, objective_index: int, x0: npt.NDArray[np.float64] | None = None
    ) -> SolveResult:
        """Minimize a single objective with no bound on the other.

        These are the two corners of the frontier, and ``trace`` calls this for each of
        them first: they fix the φ range the grid then sweeps between.

        Parameters
        ----------
         objective_index : int
            0 to minimize (1/M) ‖w‖₂² with external validity unconstrained; 1 for the
            tight end: the least-variance weighting at the smallest band every row can
            meet (the seam) or, past the seam, at the floor.
         x0 : npt.NDArray[np.float64] or None, optional
            Warm-start weights of length ``M1 + M2``.

        Returns
        -------
         res : SolveResult
            Whose ``solution`` always has length ``M1 + M2``, including for corner 1,
            where the augmenting φ is stripped before returning.

        Raises
        ------
         ValueError
            If ``objective_index`` is not 0 or 1.

        """
        if objective_index == 0:
            # minimize    (1/M) \| w \|_2^2
            # subject to  mean(w1) = 1
            #             mean(w2) = 1
            #             w1 >= lb1
            #             w2 >= lb2
            #             \| (1/M2) * Z4^T * w2 - (1/M1) * Z3^T * w1 \|_\infty <= psi3
            rake = JointCalibrator(
                distance=SquaredL2(
                    v=np.zeros(
                        self.M1 + self.M2,
                    )
                ),
                Z3=self.Z3,
                Z4=self.Z4,
                psi3=self.psi3,
                min_weight1=self.min_weight1,
                min_weight2=self.min_weight2,
                engine=self.engine,
            )
            # Technically this minimizes \| w \|_2^2, not (1/M) \| w \|_2^2,
            # so the objective value and Lagrange duals are off by a factor of M,
            # but it doesn't matter for the frontier.
            return rake.solve(x0=x0)
        elif objective_index == 1:
            corner = self._minimum_imbalance_corner(x0)
            if self._seam is None:
                self._seam_corner = corner
                self._seam = float(
                    self.evaluate_objectives(np.asarray(corner.solution))[1]
                )
            if not self._continues():
                return self._min_variance_at(corner)
            # Past the seam the tight end is the floor.
            result = self._floor_result
            assert result is not None  # Satisfy type-checker; set by _continues
            self._tight_corner = result
            self._tight_band = self.floor()
            return result
        else:
            raise ValueError(f"Unknown objective_index: {objective_index}")

    def _minimum_imbalance_corner(
        self, x0: npt.NDArray[np.float64] | None
    ) -> SolveResult:
        """Solve the linear program for the smallest max external imbalance.

        Returns a vertex at the seam -- whichever one the solver lands on, which
        ``_min_variance_at`` then replaces with the least-variance point.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64] | None
            Warm-start weights, or None.

        Returns
        -------
         res : SolveResult
            The seam vertex and the minimal max imbalance.

        """
        # minimize    \| Z^T w - nu \|_infty
        # subject to  mean(w1) = 1
        #             mean(w2) = 1
        #             w1 >= lb1
        #             w2 >= lb2
        #             \| (1/M2) * Z4^T * w2 - (1/M1) * Z3^T * w1 \|_\infty <= psi3
        #
        # which is equivalent to
        #
        # minimize     phi
        # subject to   mean(w1) = 1
        #              mean(w2) = 1
        #              w1 >= lb1
        #              w2 >= lb2
        #              \| (1/M2) * Z4^T * w2 - (1/M1) * Z3^T * w1 \|_\infty <= psi3
        #              \| (1/M1) * Z1^T * w1 - nu1 \|_\infty <= phi / sqrt(M1)
        #              \| (1/M2) * Z2^T * w2 - nu2 \|_\infty <= phi / sqrt(M2),
        #
        # a linear program
        M1, M2 = self.M1, self.M2
        q1, q2, q3 = self.Z1.shape[1], self.Z2.shape[1], self.Z3.shape[1]
        Z1T, Z2T = self.Z1.T / M1, self.Z2.T / M2
        zz = np.zeros
        p1 = np.full((q1, 1), -1.0 / np.sqrt(M1))
        p2 = np.full((q2, 1), -1.0 / np.sqrt(M2))

        # phi enters both sides of a band with the SAME sign, so the pair is
        # not a two-sided row: each covariate needs its own two rows.
        A_ext = np.block(
            [
                [Z1T, zz((q1, M2)), p1],
                [-Z1T, zz((q1, M2)), p1],
                [zz((q2, M1)), Z2T, p2],
                [zz((q2, M1)), -Z2T, p2],
            ]
        )
        cu_ext = np.concatenate([self.nu1, -self.nu1, self.nu2, -self.nu2])

        # Internal validity has fixed width, so it IS genuinely two-sided.
        A_int = np.hstack([-self.Z3.T / M1, self.Z4.T / M2, zz((q3, 1))])

        A_eq = np.block(
            [
                [np.full((1, M1), 1.0 / M1), zz((1, M2)), zz((1, 1))],
                [zz((1, M1)), np.full((1, M2), 1.0 / M2), zz((1, 1))],
            ]
        )
        lb = np.concatenate(
            [
                np.broadcast_to(self.min_weight1, M1),
                np.broadcast_to(self.min_weight2, M2),
                [0.0],  # phi >= 0
            ]
        )
        c = np.zeros(M1 + M2 + 1)
        c[-1] = 1.0

        problem = Problem(
            obj=LinearObjective(c),
            eq=EqualityConstraintLayer(A=A_eq, b=np.ones(2)),
            ineq=[
                BoundsConstraintLayer(lb=lb),
                LinearInequalityConstraintLayer(
                    A=A_int,
                    cl=-np.broadcast_to(self.psi3, q3),
                    cu=np.broadcast_to(self.psi3, q3),
                ),
                LinearInequalityConstraintLayer(A=A_ext, cu=cu_ext),
            ],
        )
        try:
            res = problem.solve(PrimalDualSolver(), x0=self._augment(x0))
        except (NewtonStepError, BacktrackingLineSearchError) as e:
            # The primal-dual method's KKT system can lose definiteness near
            # this linear program's optimum on ordinary data (one in three
            # simulated surveys of a few thousand respondents); the barrier
            # method's centering steps do not. Primal-dual stays first because
            # it is the faster of the two where it succeeds.
            logger.info(
                "Primal-dual solve of the minimum-imbalance corner raised %s; "
                "retrying with the barrier method.",
                e,
            )
            # Cold, the barrier's own Phase I can stall on this linear program
            # too, so start it from the least-variance weights, which meet the
            # internal band with room to spare and leave the external bands
            # slack once ``_augment`` pads φ.
            start = (
                np.asarray(self.minimize_objective(0).solution) if x0 is None else x0
            )
            res = problem.solve(BarrierSolver(), x0=self._augment(start))
        # Not an ``assert``: stripped under ``-O``, a wrong-length solution
        # would make the slice below drop a real weight instead of phi, and
        # every objective downstream would be quietly wrong.
        if len(res.solution) != M1 + M2 + 1:
            raise ValueError(
                f"Expected an augmented solution of length {M1 + M2 + 1}, got "
                f"{len(res.solution)}; refusing to guess which entry is phi."
            )

        # Drop the trailing phi so every branch returns the same length.
        return replace(res, solution=np.asarray(res.solution)[:-1])

    def _imbalance_scale(self) -> float:
        """Roughly size the frontier's imbalance span, for an absolute slack.

        Uniform weights satisfy both mean-one equalities and are the unconstrained
        minimum-variance weighting, so their imbalance is the far corner's to within
        whatever the internal-validity band moves it. That is close enough to scale a
        tolerance by and costs no solve, which matters because this is consulted before
        the sweep has run.

        Returns
        -------
         scale : float
            The imbalance at uniform weights.

        """
        M = self.M1 + self.M2
        return float(self.evaluate_objectives(np.ones(M))[1])

    def _min_variance_at(self, corner: SolveResult) -> SolveResult:
        """Replace a minimum-φ weighting with the least-variance one at that φ.

        The linear program minimizes φ and nothing else, so among the many weightings
        achieving ``phi_min`` it returns whichever vertex it happens to land on. That
        vertex sits well off the frontier it is meant to anchor: the smallest possible
        imbalance at far more variance than necessary. On one real survey the LP vertex
        sat at an effective sample size of 165 while the least-variance weighting at the
        same band reached 432, and every interpolant faithfully drew a segment down to
        the stray point -- a "sharp segment" that looked like a numerical artifact and
        was not one.

        Re-solving at ``phi_min`` with a little slack recovers the genuine frontier
        point. The slack has to be **absolute**, not a fraction of ``phi_min``. A
        multiplicative ``phi_min * (1 + 1e-6)`` is self-defeating precisely when it is
        needed most: a calibration that can very nearly balance has a small ``phi_min``,
        so the slack shrinks with it and the re-solve is handed a feasible set with no
        usable interior. On one brand lift question ``phi_min`` was 0.234 against a span
        of 47, so the multiplicative slack came to 2e-7 -- about 5e-9 of the span -- the
        re-solve could not start, and the LP vertex survived at a variance inflation
        factor of 27.9 where the frontier was at 6.5. The corner anchors the chord that
        ``FrontierResults.knee`` and ``refine_knee`` both measure against, so that stray
        moved the chosen band, not merely the picture.

        The slack is therefore a fraction of ``_imbalance_scale``, tried smallest first
        so the corner is reported at the tightest band that actually solves. Preferring
        the re-solve whenever one succeeds is always right: on a genuine frontier a
        slightly looser band costs a hair of imbalance and buys a hair of variance,
        while on a stray it corrects the whole error.

        If every attempt fails the LP vertex is returned, since ``knee`` needs one
        corner per objective -- but loudly, because a corner off the frontier is not a
        cosmetic loss.

        Parameters
        ----------
         corner : SolveResult
            The linear program's minimum-φ vertex.

        Returns
        -------
         res : SolveResult
            The least-variance weighting at that band, or the vertex when every re-solve
            failed.

        """
        corner_x = np.asarray(corner.solution)
        phi = float(self.evaluate_objectives(corner_x)[1])
        scale = self._imbalance_scale()
        last: Exception | None = None
        for fraction in _CORNER_SLACK_FRACTIONS:
            slack = max(_KNEE_SLACK * phi, fraction * scale)
            try:
                return self.solve_with_bounds(np.array([phi + slack]))
            except (
                ValueError,
                ProblemInfeasibleError,
                OptimizationError,
                NewtonStepError,
                BacktrackingLineSearchError,
            ) as e:
                last = e
        logger.warning(
            "Keeping the minimum-imbalance corner as the linear program left "
            "it (imbalance %.4g, variance %.4g): every re-solve, out to a "
            "slack of %.3g, raised %s. That corner anchors the chord the knee "
            "is measured from, so if it is off the frontier the chosen band "
            "moves with it.",
            phi,
            float(self.evaluate_objectives(corner_x)[0]),
            _CORNER_SLACK_FRACTIONS[-1] * scale,
            last,
        )
        return corner

    def evaluate_objectives(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Return [(1/M) ‖w‖₂², ‖Zᵀw - ν‖∞] at x.

        This is the frontier's ground truth: ``MultiObjectiveOptimizer`` builds every
        ``FrontierPoint`` from this rather than from the inner solver's own objective
        value, so the two need not agree on scale (and do not -- ``SquaredL2(v=0)``
        minimizes ‖w‖₂², a factor of M larger).

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Weights of length ``M1 + M2``.

        Returns
        -------
         objectives : npt.NDArray[np.float64]
            Array of shape ``(2,)``. The first entry is ``M / ESS``, since both groups
            are pinned to mean weight one.

        """
        M1, M2 = self.M1, self.M2
        w1, w2 = x[:M1], x[M1:]

        return np.array(
            [
                float(np.mean(x * x)),
                max(
                    np.max(
                        np.abs(self.Z1.T @ w1 / np.sqrt(M1) - np.sqrt(M1) * self.nu1)
                    ),
                    np.max(
                        np.abs(self.Z2.T @ w2 / np.sqrt(M2) - np.sqrt(M2) * self.nu2)
                    ),
                ),
            ]
        )

    def diagnose(
        self,
        x: npt.NDArray[np.float64],
        covariate_names: Sequence[str] | None = None,
        phi: float | None = None,
    ) -> pd.DataFrame:
        """Per-covariate imbalance against its band, tightest slack first.

        Answers "what is stopping this from balancing better?". A single row at
        ``slack_frac`` near zero means one covariate is the bottleneck; a cluster of
        them means the constraint set is pinned on many faces at once, which is the
        common case once a categorical is entered as a full set of dummies -- those obey
        ``Σₖ σₖ dₖ = 0``, so their imbalances cannot all be driven to zero independently
        and the solver ends up electing one level to absorb the residual.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Weights of length ``M1 + M2``.
         covariate_names : sequence of str, optional
            Labels for the columns of Z, in order. Defaults to positional indices.
         phi : float, optional
            The external-validity level the weights were solved at. Defaults to the
            level they actually achieve, which is what you want at a corner or anywhere
            the band is tight.

        Returns
        -------
         pandas DataFrame
            One row per (kind, block, covariate), with ``imbalance``, ``band``,
            ``slack`` and ``slack_frac``, sorted tightest first. ``slack_frac`` is the
            comparable column: the three groups carry different bands (``phi/sqrt(M1)``,
            ``phi/sqrt(M2)``, ``psi3``), so absolute slack is not comparable across
            them.

        """
        M1, M2 = self.M1, self.M2
        w1, w2 = x[:M1], x[M1:]
        if phi is None:
            phi = float(self.evaluate_objectives(x)[1])

        names = (
            list(covariate_names)
            if covariate_names is not None
            else [str(j) for j in range(self.Z1.shape[1])]
        )
        blocks = {
            ("external", "holdout"): (
                self.Z1.T @ w1 / M1 - self.nu1,
                phi / np.sqrt(M1),
            ),
            ("external", "marketing"): (
                self.Z2.T @ w2 / M2 - self.nu2,
                phi / np.sqrt(M2),
            ),
            ("internal", "hldt vs mktg"): (
                self.Z4.T @ w2 / M2 - self.Z3.T @ w1 / M1,
                self.psi3,
            ),
        }

        rows: list[dict[str, str | float]] = []
        for (kind, block), (imbalance, band) in blocks.items():
            # The internal band may be per-row, so broadcast and report the band
            # each covariate was actually held to rather than a single number.
            bands = np.broadcast_to(np.asarray(band, dtype=np.float64), len(names))
            for name, value, row_band in zip(names, imbalance, bands, strict=True):
                rows.append(
                    {
                        "kind": kind,
                        "block": block,
                        "covariate": name,
                        "imbalance": abs(float(value)),
                        "band": float(row_band),
                        "slack": float(row_band) - abs(float(value)),
                    }
                )

        df = pd.DataFrame(rows)
        # Relative slack, so the three groups compare despite different bands.
        # A zero band is reachable -- ``phi=0`` asks for exact balance -- and
        # dividing by it would put inf/nan in the column and scramble the sort.
        # Such a row has no room at all: exactly pinned if it meets the band,
        # and maximally tight if it does not.
        band = df["band"].to_numpy()
        slack = df["slack"].to_numpy()
        df["slack_frac"] = np.divide(
            slack,
            band,
            out=np.where(slack < 0.0, -np.inf, 0.0),
            where=band > 0.0,
        )
        return df.sort_values("slack_frac").reset_index(drop=True)

    def _augment(
        self, x0: npt.NDArray[np.float64] | None
    ) -> npt.NDArray[np.float64] | None:
        """Extend a weight vector with a φ that leaves the bands slack.

        Corner 1 solves over ``[w1, w2, phi]``, so a caller's warm start -- which is
        just weights -- needs a φ appended. Padding above the imbalance ``x0`` actually
        achieves keeps the augmented point off the boundary; the additive term covers an
        ``x0`` that is already perfectly balanced.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64] | None
            Weight vector, or None.

        Returns
        -------
         augmented : npt.NDArray[np.float64] | None
            ``[x0, phi]``, or None when ``x0`` is None.

        """
        if x0 is None:
            return None
        return np.concatenate([x0, [1.05 * self.evaluate_objectives(x0)[1] + 1e-6]])

    def trace(
        self,
        num_points: int = 50,
        callback: Callable[[FrontierPoint, int, int], bool] | None = None,
        n_jobs: int = 1,
        knee_space: Literal["vif", "se"] = "se",
        refine_knee: bool = True,
    ) -> ExternalValidityVarianceFrontierResults:
        """Trace the external-validity/variance tradeoff.

        Solves for ``num_points`` values of φ between the two Pareto corners. Corner 0
        minimizes (1/M) ‖w‖₂² with external validity unconstrained; corner 1 minimizes
        the max external imbalance.

        Parameters
        ----------
         num_points : int, default=50
            Number of grid points along the φ dimension.
         callback : callable or None, optional
            Forwarded to the parent ``trace``.
         n_jobs : int, default=1
            Forwarded to the parent ``trace``. Leave at 1: the sequential path
            warm-starts each solve from the previous one, which is where the primal-dual
            engine wins, and these solves are individually cheap.
         knee_space : {"vif", "se"}, default="se"
            Recorded on the results, and the space the knee is refined in when
            ``refine_knee`` is True. The traced curve is the same either way; only the
            question asked of it differs. Set it to whatever the band will be chosen in
            so the results carry their own answer to "which bend was this?".
         refine_knee : bool, default=True
            Solve for the knee with ``refine_knee`` once the sweep is done, so ``plot``
            marks the knee a calibration would use rather than the nearest of the
            sampled points. Costs a handful of extra solves, which are folded into
            ``points``. A refinement that finds no knee (a straight frontier) is
            recorded as such, and ``plot`` then marks none. Pass False to get the bare
            sweep, e.g. to refine it yourself and act on the returned knee, as
            ``knee_external_scale`` does.

        Returns
        -------
         results : ExternalValidityVarianceFrontierResults
            Whose ``.effective_sample_sizes`` and ``.imbalances`` are sorted by
            imbalance ascending, and whose ``plot()`` is labelled in ESS units by
            default, or in the optimizer's own variance inflation factor with
            ``plot(variance_space="vif")``.

        Notes
        -----
        The grid is uniform in phi, so a frontier that turns sharply near its minimum
        phi is sampled least densely exactly where it bends most. Infeasible grid points
        are skipped silently, so a gap in the curve means no weighting met that band,
        not a solver failure; ``results.n_skipped`` against ``results.n_attempted`` says
        how much of the grid landed. (Knee refinement adds its probes to both counts and
        to ``points``, so ``len(results.points)`` can exceed ``num_points + 2``.)

        """
        fr = super().trace(num_points=num_points, callback=callback, n_jobs=n_jobs)
        continues = self._continues()
        results = ExternalValidityVarianceFrontierResults(
            points=fr.points,
            corners=fr.corners,
            primary_objective=fr.primary_objective,
            n_attempted=fr.n_attempted,
            n_skipped=fr.n_skipped,
            # Not optional bookkeeping: drop it and every envelope
            # downstream quietly loses its lower bound.
            tangent_slopes=fr.tangent_slopes,
            # Likewise not optional: this is what lets ``plot`` annotate the
            # bend the band was actually chosen from.
            knee_space=knee_space,
            seam=self.seam() if continues else None,
        )
        if continues and self._seam_corner is not None:
            # The seam is where the curve changes regime, and the uniform grid
            # need not land on it. Its least-variance point is one solve.
            seam_point = self._min_variance_at(self._seam_corner)
            phi = float(self.evaluate_objectives(np.asarray(seam_point.solution))[1])
            results.absorb_probe(
                FrontierPoint(
                    objectives=self.evaluate_objectives(
                        np.asarray(seam_point.solution)
                    ),
                    solution=np.asarray(seam_point.solution),
                    bounds=np.array([phi]),
                    ipm_result=seam_point,
                ),
                None,
            )
        if refine_knee:
            # Records its conclusion on ``results``, which is what ``plot``
            # reads; the returned knee is not needed here.
            self.refine_knee(results, knee_space=_cvxium_knee_space(knee_space))
        return results


# The conventional external-validity scale: a band of ``1.5 / sqrt(M)`` per
# group, following (Cochran and Chambers, 1965, section 3.1).
_COCHRAN_SCALE: float = 1.5

# Grid resolution for the knee sweep. Deliberately coarser than ``trace``'s own
# default of 50: this runs inside a calibration, not a diagnostic notebook, and
# the knee is a choice between broad regimes rather than a precise location.
# Public because callers that surface ``num_points`` have to name the default
# too, and a second literal 15 would drift from this one in silence.
KNEE_GRID_POINTS: int = 15

# Relative slack on the band chosen from the knee, so the chosen point is
# strictly inside it. The knee's own imbalance sits exactly on the band it was
# solved at, and adopting that verbatim leaves the subsequent solve with no
# strictly interior point to start from -- Phase I then reports the problem
# infeasible.
_KNEE_SLACK: float = 1e-6

# Slacks tried, smallest first, when re-solving the minimum-imbalance corner
# onto the frontier -- as fractions of the imbalance SPAN, not of phi_min.
# Scaling by phi_min fails exactly where the correction matters: a calibration
# that nearly balances has a tiny phi_min, so the slack vanishes with it and
# the re-solve gets no interior to start from. See
# ``ExternalValidityVarianceFrontier._min_variance_at``. The smallest entry is
# already several orders of magnitude of interior on a typical frontier; the
# larger ones exist so a badly scaled problem still gets a corner on the curve
# rather than the linear program's vertex.
_CORNER_SLACK_FRACTIONS: tuple[float, ...] = (1e-4, 1e-3, 1e-2)

# A seam below this fraction of the frontier's imbalance span counts as zero:
# the bands can be met all the way down, and there is nothing past the seam to
# trace.
_SEAM_TOLERANCE: float = 1e-3

# The band the floor is found from, as a fraction of the seam:
# ``solve_with_fallback`` widens it to the floor. Not zero, which leaves the
# widening factor nothing to multiply.
_TIGHT_BAND_FRACTION: float = 1e-3

# The key ``FallbackOptimizationResult.band_scales`` uses for the external
# validity band.
_EXTERNAL_LABEL: str = "external validity"

# Fallback status for "external validity deliberately relaxed to the knee".
# Distinct from 2 (external validity maximized but short of target): here the
# target was reachable and was declined, which a reader must not mistake for a
# shortfall -- nor for success.
_KNEE_RELAXED_STATUS: int = 4


def _cvxium_knee_space(
    knee_space: Literal["vif", "se"],
) -> Literal["primary", "root"]:
    """Translate this module's axis names into Cvxium's generic ones.

    Cvxium knows only that it is minimizing a primary objective; here that objective is
    the variance inflation factor, and its square root is the standard-error multiplier.
    The two vocabularies are kept apart rather than shared because ``"root"`` says
    nothing at a call site in a survey pipeline and ``"se"`` would be a promise Cvxium
    cannot keep for an arbitrary primary objective.

    Parameters
    ----------
     knee_space : {"vif", "se"}
        This module's axis name.

    Returns
    -------
     space : {"primary", "root"}
        Cvxium's generic axis name.

    Raises
    ------
     ValueError
        When ``knee_space`` is neither ``"vif"`` nor ``"se"``.

    """
    if knee_space == "se":
        return "root"
    if knee_space != "vif":
        raise ValueError(f"knee_space must be 'vif' or 'se', got {knee_space!r}.")
    return "primary"


def solve_with_knee_fallback(
    rake: JointCalibrator,
    num_points: int = KNEE_GRID_POINTS,
    n_jobs: int = 1,
    engine: Literal["ipm", "pd"] | None = "pd",
    knee_space: Literal["vif", "se"] = "se",
) -> FallbackOptimizationResult:
    """Calibrate, choosing the external band from the frontier's knee.

    ``JointCalibrator.solve_with_fallback`` minimizes the imbalance when the requested
    external band is infeasible, which silently selects the far end of the
    external-validity/variance frontier -- where the curve is near-flat and the last few
    percent of balance cost most of the sample. On one real survey (M = 1007, smallest
    achievable band 4.19) that landed at an effective sample size of 164 where the knee
    offered 448: 2.7x the variance, 284 respondents' worth of effective sample, for 7%
    better balance.

    This traces the frontier and calibrates at ``max(1.5, phi_knee)`` instead -- aim for
    the conventional Cochran scale, but never pay past the knee to reach it -- then
    hands back the usual cascade result with the frontier attached.

    Only the *band* comes from the frontier. The cascade still runs its own solve, so
    ``rake``'s configured distance is what gets minimized; the frontier minimizes
    variance, and for anything but ``SquaredL2(v=0)`` those are different objectives.

    Parameters
    ----------
     rake : JointCalibrator
        A configured joint calibration. Its ``psi`` is restored before this returns, so
        it is not left holding a band nobody asked for.
     num_points : int, optional
        Grid resolution for the sweep.
     n_jobs : int, optional
        Worker processes for the sweep, forwarded to ``trace``. The default runs
        sequentially and warm-starts each solve from the previous one, which is where
        the primal-dual engine earns its keep; ``n_jobs != 1`` fans the grid out and
        solves each point cold, so it pays off when a single solve is expensive rather
        than when there are many cheap ones. The calibration itself is unaffected either
        way.
     engine : {"ipm", "pd"} or None, optional
        Inner solver for the sweep only -- the final calibration always runs through
        ``rake``'s own engine. ``None`` inherits ``rake``'s.
     knee_space : {"vif", "se"}, optional
        Which coordinate the bend is located in; see ``knee_external_scale``.

    Returns
    -------
     result : FallbackOptimizationResult
        With ``frontier`` set when a sweep informed the band, and None when the sweep
        could not run and the requested band was used instead.

    Notes
    -----
    A sweep that cannot run -- a corner solve that raises, say -- degrades to the
    requested band and logs. That includes ``NewtonStepError`` and
    ``BacktrackingLineSearchError``, which derive from ``Exception`` rather than
    ``OptimizationError`` and so are not caught by the grid sweep's own swallow. A
    diagnostic sweep around a calibration should not turn a calibration that would
    otherwise succeed into an exception.

    A sweep that runs but finds no knee is a different case and no longer degrades to
    the requested band: ``refine_knee`` returns None for a missing corner or a frontier
    too straight to have a bend, and ``knee_external_scale`` then uses the conventional
    band floored at what the frontier reached. The traced frontier is still attached.

    """
    scale, results = knee_external_scale(rake, num_points, n_jobs, engine, knee_space)
    band = None if scale is None else external_band(rake, scale)
    if scale is None or band is None:
        return rake.solve_with_fallback()

    original_psi = rake.psi
    try:
        # Inside the try: update_psi assigns self.psi and then rebuilds the
        # problem, so a rebuild that raises leaves the band applied. Setting it
        # outside would skip the finally and strand the calibrator holding a
        # band nobody asked for -- the thing this block exists to prevent.
        rake.update_psi(band)
        result = rake.solve_with_fallback()
    finally:
        rake.update_psi(original_psi)

    return relabel_for_the_chosen_band(result, scale, results)


def relabel_for_the_chosen_band(
    result: FallbackOptimizationResult,
    scale: float,
    results: ExternalValidityVarianceFrontierResults | None,
) -> FallbackOptimizationResult:
    """Say which external band was used, and why it is not the conventional one.

    The cascade reports against the band it was handed, so calibrating at the knee and
    succeeding reads as "we achieved the desired covariate balance (both internal and
    external validity)". True of the band the cascade was given, false of the band the
    caller wanted -- and the second is the more misleading reading. A table of results
    would claim full success on every row while quietly sitting at several times the
    conventional imbalance.

    Three outcomes, distinguished because they call for different responses:

    - The knee is tighter than the conventional scale, so the floor decided and
      the conventional band was met. The cascade's own label is accurate.
    - The conventional scale is unattainable. Nothing could have done better:
      status 2, external validity not achieved.
    - The conventional scale was attainable and we declined it, buying variance
      with imbalance. A deliberate trade rather than a shortfall, and the one a
      reader must be told about explicitly, so it gets its own status.

    ``scale`` is the band actually used, which need not be this calibration's own knee:
    a caller holding several cells to one standard passes the loosest of them, so the
    wording speaks of the chosen band rather than the knee. ``results`` stays this
    calibration's own frontier, which is what ``phi_min`` has to come from --
    attainability is a property of the cell, not of the agreement.

    Parameters
    ----------
     result : FallbackOptimizationResult
        The cascade's result at the chosen band.
     scale : float
        The band actually used.
     results : ExternalValidityVarianceFrontierResults | None
        This calibration's own frontier, or None.

    Returns
    -------
     relabelled : FallbackOptimizationResult
        ``result`` with the message and status rewritten for the chosen band.

    """
    if results is None or scale <= _COCHRAN_SCALE:
        return replace(result, frontier=results)

    # The smallest band every row can meet. Past the seam the frontier's own
    # minimum is the floor, which only the relaxation reaches, so the seam is
    # the number that says whether the conventional band was there to be had.
    phi_min = (
        results.seam
        if results.seam is not None
        else min(float(p.objectives[1]) for p in results.points)
    )
    # The two groups do not share a band: each divides the scale by its own
    # sqrt(M_b), so the smaller group ends up with the looser one. Quote that
    # one, so the number is a bound on both rather than an average of neither.
    # Halving the solution length would assume M1 == M2 and report a band that
    # belongs to neither group.
    solution = np.asarray(result.solution)
    m1 = result.m1 or len(solution) // 2
    per_group = scale / np.sqrt(min(m1, len(solution) - m1))

    if phi_min <= _COCHRAN_SCALE:
        message = (
            f"External validity was relaxed on purpose: the conventional band "
            f"of {_COCHRAN_SCALE} was attainable, but the chosen band sits "
            f"at {scale:.3g} (at worst {per_group:.2g} standard deviations in "
            f"either group) and tightening past it costs sample size faster "
            f"than it buys balance. Internal validity and the design effect "
            f"were achieved at that band."
        )
        status = _KNEE_RELAXED_STATUS
    elif not result.relaxed_rows:
        message = (
            f"Desired external validity was unattainable: the smallest "
            f"achievable band is {phi_min:.3g}, against the conventional "
            f"{_COCHRAN_SCALE}. Calibrated at {scale:.3g} "
            f"(at worst {per_group:.2g} standard deviations in either "
            f"group), rather "
            f"than at the minimum, which would have cost most of the sample."
        )
        status = 2
    else:
        n_relaxed = len(result.relaxed_rows)
        message = (
            f"Desired external validity was unattainable on every row: the "
            f"smallest band every row can meet is {phi_min:.3g}, against the "
            f"conventional {_COCHRAN_SCALE}. Calibrated at {scale:.3g} "
            f"(at worst {per_group:.2g} standard deviations in either group), "
            f"where the frontier bends; {n_relaxed} constraint"
            f"{'' if n_relaxed == 1 else 's'} could not meet even that and "
            f"{'was' if n_relaxed == 1 else 'were'} relaxed, keeping the summed "
            f"excess to the least achievable, and the rest are within it."
        )
        status = 2

    return replace(
        result, frontier=results, fallback_message=message, fallback_status=status
    )


def knee_external_scale(
    rake: JointCalibrator,
    num_points: int = KNEE_GRID_POINTS,
    n_jobs: int = 1,
    engine: Literal["ipm", "pd"] | None = "pd",
    knee_space: Literal["vif", "se"] = "se",
) -> tuple[float | None, ExternalValidityVarianceFrontierResults | None]:
    """Compute the external-validity scale implied by this calibration's knee.

    Returns ``max(1.5, phi_knee)`` -- the conventional Cochran scale unless the knee
    says balance gets expensive before then -- as a *scale*, not a band. Callers
    converting it to a band divide by ``sqrt(M_b)`` per group, which is what
    ``external_band`` does.

    Split out from ``solve_with_knee_fallback`` so several calibrations can agree on one
    scale before any of them is solved. Cells of a lift study are compared against each
    other, so they should be held to a common standard; the caller takes the max across
    them and passes it back in.

    Parameters
    ----------
     rake : JointCalibrator
        A configured joint calibration.
     num_points : int, optional
        Grid resolution for the sweep.
     n_jobs : int, optional
        Worker processes for the sweep, forwarded to ``trace``.
     engine : {"ipm", "pd"} or None, optional
        Inner solver for the sweep. ``None`` inherits ``rake``'s.
     knee_space : {"vif", "se"}, default="se"
        Which coordinate to locate the bend in. ``"vif"`` uses the variance inflation
        factor the sweep minimizes. ``"se"`` uses its square root, the standard-error
        multiplier, which is the quantity that actually enters a confidence interval --
        so trading imbalance against it is closer to a bias/precision rule than trading
        against a variance. Both are legitimate: the frontier is convex in either (the
        square root of a squared norm is a norm, and a value function under a convex
        objective is convex), so the root-find is well posed in both. They need not
        agree, and where they differ ``"se"`` is the looser of the two, because the
        square root stretches the low-variance end of the axis and the maximum-distance
        point migrates into the stretch. They coincide where curvature is concentrated
        at a single point, since every monotone transform of the axis then finds that
        same point -- so the size of the disagreement measures how sharply the bend is
        defined, and the choice matters least where the two spaces differ least.

    Returns
    -------
     scale : float or None
        None only when the sweep could not run at all -- no imbalance rows, or a solve
        that raised -- in which case the caller should keep the requested band. A sweep
        that ran but yielded no determinable knee (no slopes, a missing corner, or a
        frontier straight to within tolerance) returns a scale rather than None: the
        conventional ``_COCHRAN_SCALE``, floored at the tightest band the frontier
        actually reached. Those are different failures. One leaves us knowing nothing
        about this calibration; the other leaves us with a traced frontier and only the
        *location of the bend* undetermined, which is enough to say the conventional
        band is attainable.
     frontier : ExternalValidityVarianceFrontierResults or None
        The traced frontier, present whenever the sweep ran -- including on the no-knee
        path, so callers can plot what was traced and see why no bend was found.

    """
    if rake.psi is None or (rake._q1 or 0) + (rake._q2 or 0) == 0:
        return None, None

    try:
        frontier = ExternalValidityVarianceFrontier.from_calibrator(rake, engine=engine)
        # Refined explicitly below, where the knee it returns sets the band.
        results = frontier.trace(
            num_points=num_points,
            n_jobs=n_jobs,
            knee_space=knee_space,
            refine_knee=False,
        )
        # Solve for the knee where the slopes allow it, rather than taking the
        # nearest sampled point: the sweep grids uniformly in phi and need not
        # put a point anywhere near the bend.
        # Read back off the results rather than from the argument, so the band
        # and the annotation this frontier will draw cannot disagree.
        knee = frontier.refine_knee(
            results, knee_space=_cvxium_knee_space(results.knee_space)
        )
        if knee is None:
            # Either no slopes, or a frontier too straight to have a knee. The
            # sampled argmax is no better in that case -- on a straight stretch
            # it is determined by differences near the noise floor -- so take
            # the conventional band rather than a number that looks precise.
            # Floored at the smallest band the frontier actually reached. On a
            # frontier that stops at the seam the conventional band can sit
            # below it, and asking for a band nobody can meet sends
            # solve_with_fallback down the relaxation -- the far end of that
            # frontier, at a fraction of the effective sample size. Past the
            # seam the frontier reaches its floor, usually below the
            # conventional band, which is then what is asked for: met on the
            # rows that can meet it and relaxed on the rest, a point this
            # frontier traced.
            phi_min = min(float(p.objectives[1]) for p in results.points)
            logger.info(
                "No determinable knee; using the conventional band of %.3g, "
                "floored at the attainable %.3g.",
                _COCHRAN_SCALE,
                phi_min,
            )
            return max(_COCHRAN_SCALE, phi_min), results
    except (
        ValueError,
        ProblemInfeasibleError,
        OptimizationError,
        NewtonStepError,
        BacktrackingLineSearchError,
    ) as e:
        logger.warning(
            "External-validity knee selection degraded to the requested band: "
            "the frontier sweep failed with %s: %s",
            type(e).__name__,
            e,
        )
        return None, None

    return max(_COCHRAN_SCALE, float(knee.objectives[1])), results


def _full_psi(rake: JointCalibrator) -> npt.NDArray[np.float64] | None:
    """Expand ``rake.psi`` to one entry per imbalance row.

    ``JointCalibrator`` expands a scalar band per block, so psi normally arrives already
    at full length. ``update_psi`` does not: a caller who sets a scalar there leaves a
    single entry standing for every row. Slicing that for the internal tail comes back
    empty, and the two callers then fail differently -- ``from_calibrator`` quietly
    substitutes the default internal band, while ``external_band`` builds a psi shorter
    than ``B`` has rows, so the rebuild errors or drops the internal block. Broadcasting
    once, here, is what stops the two from drifting apart again.

    Parameters
    ----------
     rake : JointCalibrator
        The joint calibration.

    Returns
    -------
     psi : npt.NDArray[np.float64] | None
        The broadcast band, or None when the calibrator has no band.

    """
    if rake.psi is None:
        return None

    psi = np.atleast_1d(np.asarray(rake.psi, dtype=np.float64))
    rows = (rake._q1 or 0) + (rake._q2 or 0) + (rake._q3 or 0)
    if psi.size == 1 and rows:
        return np.full(rows, float(psi[0]))
    return psi


def external_band(
    rake: JointCalibrator, scale: float
) -> npt.NDArray[np.float64] | None:
    """Compute the full psi vector for one external-validity scale.

    The vector holds this calibration's external validity at ``scale``.

    Each group's band is ``scale / √M_b``, so a shared scale gives groups of different
    size the bands that make them equally stringent -- the same convention as PyRake's
    ``1.5 / √M`` default. The internal rows are left as they are.

    A hair of slack is added because a scale read off a knee sits exactly on the band
    that knee was solved at, and adopting it verbatim leaves the subsequent solve with
    no strictly interior point to start from.

    Parameters
    ----------
     rake : JointCalibrator
        The joint calibration.
     scale : float
        The shared external-validity scale.

    Returns
    -------
     psi : npt.NDArray[np.float64] | None
        The full band vector, or None when there are no external rows.

    """
    psi = _full_psi(rake)
    if psi is None:
        return None

    q_ext = (rake._q1 or 0) + (rake._q2 or 0)
    if q_ext == 0:
        return None

    padded = scale * (1.0 + _KNEE_SLACK)
    return np.concatenate(
        [
            np.full(rake._q1 or 0, padded / np.sqrt(rake._M1)),
            np.full(rake._q2 or 0, padded / np.sqrt(rake._M2)),
            psi[q_ext:],
        ]
    )
