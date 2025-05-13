"""Bias/Variance Tradeoff."""

from typing import List, Optional

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .rake import Rake


class EfficientFrontierResults:
    """Wrapper for bias/variance tradeoff."""

    def __init__(
        self,
        weights: List[npt.NDArray[np.float64]],
        phis: List[float],
        distances: List[float],
    ) -> None:
        self.weights = weights
        self.phis = phis
        self.distances = distances

    def plot(self, ax: Optional[Axes] = None) -> Axes:
        """Plot bias/variance tradeoff."""
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.distances, self.phis)
        plt.fill_between(
            self.distances, self.phis, max(self.phis), color="lightblue", alpha=0.5
        )
        ax.set_xlabel("Distance from Baseline Weights")
        ax.set_ylabel("Variance Inflation Factor")
        ax.set_title("Bias-Variance Tradeoff")
        return ax


class EfficientFrontier:
    r"""Class for tracing out the bias/variance tradeoff.

    Solves:
           minimize    D(w, v)
           subject to  (1/M) * X^T * w = \mu
                        \| w \|_2^2 \leq \phi
                        w >= 0,
    for a sequence of values for \phi. Increasing \phi involves more variance,
    but potentially less bias.

    """

    def __init__(self, rake: Rake) -> None:
        self.rake = rake

    def trace(
        self, phi_max: Optional[float] = None, num_points: int = 20
    ) -> EfficientFrontierResults:
        """Trace bias/variance tradeoff."""
        if self.rake.phase1_solver is None:
            raise ValueError("Must specify a Phase I solver")

        # Find the minimum variance weights, regardless of distance from baseline
        res_min = self.rake.phase1_solver.solve(fully_optimize=True)
        w0 = res_min.solution
        phi_min = np.dot(w0, w0) / self.rake.dimension
        weights = [w0]
        phis = [phi_min]
        distances = [self.rake.distance.evaluate(w0)]

        # Find the feasible weights that most closely match baseline, regardless of
        # variance. This is a little hacky, but we do this by multiplying phi_min by
        # 100, which assumes the unconstrained variance is less than this. Eventually
        # I'd like to make this constraint optional, at which point we can do this more
        # elegantly.
        self.rake.update_phi(phi_max or phi_min * 100)
        res_max = self.rake.solve(x0=w0)
        phi_max_nn = np.dot(res_max.solution, res_max.solution) / self.rake.dimension

        # Calculate a range of optimal weights between phi_min and phi_max
        phi_grid = np.geomspace(phi_min, phi_max_nn, num=num_points)[1:]
        w = w0
        for phi in phi_grid:
            self.rake.update_phi(phi)
            res = self.rake.solve(x0=w)
            weights.append(res.solution)
            phis.append(phi)
            distances.append(res.objective_value)
            w = res.solution

        weights.append(res_max.solution)
        phis.append(phi_max_nn)
        distances.append(res_max.objective_value)

        return EfficientFrontierResults(weights, phis, distances)
