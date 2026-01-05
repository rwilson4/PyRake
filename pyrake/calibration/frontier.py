"""Bias/Variance Tradeoff."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from ..optimization import InteriorPointMethodResult
from .phase1solvers import (
    EqualityWithBoundsAndImbalanceConstraintSolver,
    EqualityWithBoundsAndNormConstraintSolver,
    EqualityWithBoundsSolver,
)
from .rake import Rake


class EfficientFrontierResults:
    """Wrapper for bias/variance tradeoff."""

    def __init__(
        self,
        weights: list[npt.NDArray[np.float64]],
        distances: list[float],
        variances: list[float],
        lagrange_multipliers: list[float],
        ipm_results: list[InteriorPointMethodResult],
    ) -> None:
        self.weights = weights
        self.distances = distances
        self.variances = variances
        self.lagrange_multipliers = lagrange_multipliers
        self.ipm_results = ipm_results

    def plot(
        self,
        annotate_knee: bool = True,
        annotate_index: int | None = None,
        ax: Axes | None = None,
    ) -> Axes:
        """Plot bias/variance tradeoff.

        Parameters
        ----------
         annotate_knee : bool, optional
            If True, indicate the "knee in the curve" on the plot. See Notes.
         annotate_index : int, optional
            Indicate the tangent to the frontier at the point corresponding to
            `annotate_index`. If specified, `annotate_knee` is ignored. By default, no
            annotation is shown.
         ax : matplotlib Axes, optional
            If specified, plot frontier on `ax`.

        Returns
        -------
         ax : matplotlib Axes

        Notes
        -----
        Programmatically identifying the knee in the curve is still experimental. We use
        the "max chord distance" method, which imagines a chord connecting the edges of
        the frontier: the minimum distance/max variance to maximum distance/min
        variance. Whichever point on the curve is farthest from this chord (as measured
        by perpendicular distance) is selected as the knee. Note that only points
        explicitly calculated during EfficientFrontier.trace() are considered, so the
        true knee may be between two of the points considered.

        """
        if ax is None:
            _, ax = plt.subplots()
        else:
            plt.sca(ax)

        ax.plot(self.distances, self.variances)
        plt.fill_between(
            self.distances,
            self.variances,
            max(self.variances),
            color="lightblue",
            alpha=0.5,
        )

        if annotate_index is None and annotate_knee:
            annotate_index = self.max_chord_distance()

        if annotate_index is not None:
            if not 0 <= annotate_index <= len(self.distances):
                raise ValueError(f"Invalid {annotate_index=:}")

            distance = self.distances[annotate_index]
            phi = self.variances[annotate_index]

            # The slope involves phi since under the hood, we're translating the
            # constraint (1/M) * \| w \|_2^2 <= \phi to:
            # (1 / (M * \phi)) * \| w \|_2^2 <= 1.
            slope = -phi / self.lagrange_multipliers[annotate_index]

            # This is an attempt to control the length of the tangent line when the
            # tangent point is close to the left or right of the plot.
            width = max(
                min(
                    0.5 * (distance - min(self.distances)),
                    0.5 * (max(self.distances) - distance),
                ),
                0,
            )
            x_vals = np.linspace(distance - width, distance + width)
            ax.plot(x_vals, phi + slope * (x_vals - distance), color="orange")

            # Note the current plot boundaries so we don't move them.
            xlim = plt.xlim()
            ylim = plt.ylim()

            # Draw dashed lines connecting point to axes.
            ax.hlines(
                y=phi,
                xmin=xlim[0],
                xmax=distance,
                colors="orange",
                linestyles="dashed",
            )

            ax.vlines(
                x=distance,
                ymin=ylim[0],
                ymax=phi,
                colors="orange",
                linestyles="dashed",
            )
            # Reset plot boundaries.
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])

        ax.set_xlabel("Distance from Baseline Weights")
        ax.set_ylabel("Variance Inflation Factor")
        ax.set_title("Bias-Variance Tradeoff")
        return ax

    def max_chord_distance(self) -> int:
        """Find the point of maximum chord distance."""
        x1 = self.distances[0]
        x2 = self.distances[-1]
        y1 = self.variances[0]
        y2 = self.variances[-1]
        dy = y2 - y1
        dx = x2 - x1
        x2y1_y2x1 = x2 * y1 - y2 * x1
        distances = np.array(
            [
                abs(dy * x - dx * y + x2y1_y2x1)
                for x, y in zip(self.distances, self.variances, strict=False)
            ]
        )
        return int(distances.argmax())


class EfficientFrontier:
    r"""Class for tracing out the bias/variance tradeoff.

    Solves:
           minimize    D(w, v)
           subject to  (1/M) * X^T * w = \mu
                       (1/M) * \| w \|_2^2 <= \phi
                        w >= 0,
    for a sequence of values for \phi. Increasing \phi involves more variance,
    but potentially less bias.

    """

    def __init__(self, rake: Rake) -> None:
        self.rake = rake

    def trace(
        self, phi_max: float | None = None, num_points: int = 50
    ) -> EfficientFrontierResults:
        """Trace bias/variance tradeoff."""
        if self.rake.phase1_solver is None:
            raise ValueError("Must specify a Phase I solver")

        # Find the minimum variance weights, regardless of distance from baseline
        if isinstance(
            self.rake.phase1_solver, EqualityWithBoundsAndNormConstraintSolver
        ):
            res_min = self.rake.phase1_solver.solve(fully_optimize=True)
        elif isinstance(
            self.rake.phase1_solver,
            EqualityWithBoundsSolver | EqualityWithBoundsAndImbalanceConstraintSolver,
        ):
            res_min = EqualityWithBoundsAndNormConstraintSolver(
                phi=np.inf,
                phase1_solver=self.rake.phase1_solver,
                settings=self.rake.settings,
            ).solve(fully_optimize=True)
        else:
            raise ValueError("Unrecognized Phase I Solver")

        w0 = res_min.solution
        phi_min = np.mean(w0 * w0)
        weights = [w0]
        variances = [float(phi_min)]
        lagrange_multipliers = [0.0]
        distances = [self.rake.distance.evaluate(w0)]

        # This is just a hack to tell the type checker that res_min has the desired type. By
        # construction, it always will.
        assert isinstance(res_min, InteriorPointMethodResult)

        ipm_results: list[InteriorPointMethodResult] = [res_min]

        # Find the feasible weights that most closely match baseline, regardless of
        # variance.
        self.rake.update_phi(phi_max)
        res_max = self.rake.solve()
        phi_max_nn = np.dot(res_max.solution, res_max.solution) / self.rake.dimension

        # Calculate a range of optimal weights between phi_min and phi_max
        phi_grid = np.linspace(phi_min, phi_max_nn, num=num_points)[1:-1]
        for phi in phi_grid:
            self.rake.update_phi(phi)
            ipm_res = self.rake.solve()

            weights.append(ipm_res.solution)
            variances.append(float(phi))
            lagrange_multipliers.append(ipm_res.inequality_multipliers[-1])
            distances.append(ipm_res.objective_value)
            ipm_results.append(ipm_res)

        weights.append(res_max.solution)
        variances.append(phi_max_nn)
        lagrange_multipliers.append(res_max.inequality_multipliers[-1])
        distances.append(res_max.objective_value)
        ipm_results.append(res_max)

        return EfficientFrontierResults(
            weights=weights,
            distances=distances,
            variances=variances,
            lagrange_multipliers=lagrange_multipliers,
            ipm_results=ipm_results,
        )
