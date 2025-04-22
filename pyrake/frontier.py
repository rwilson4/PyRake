"""Bias/Variance Tradeoff."""

import numpy as np
import matplotlib.pyplot as plt

from .core import Rake


class EfficientFrontierResults:
    """Wrapper for bias/variance tradeoff."""

    def __init__(self, weights, phis, distances):
        self.weights = weights
        self.phis = phis
        self.distances = distances

    def plot(self, ax=None):
        """Plot bias/variance tradeoff."""
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.phis, self.distances, marker="o")
        ax.set_xlabel("Variance")
        ax.set_ylabel("Bias")
        ax.set_title("Efficient Frontier: Bias-Variance Tradeoff")
        return ax


class EfficientFrontier:
    r"""Class for tracing out the bias/variance tradeoff.

    Solves:
           minimize    D(w, v)
           subject to  (1/M) * X^T * w = \mu
                        \| w \|_2^2 \leq \phi
                        w >= 0,
    for a sequence of values for \phi. Increasing \phi involves more variance,
    but potentially more bias.

    """

    def __init__(self, rake: Rake):
        self.rake = rake

    def trace(self, phi_max=None, num_points=20):
        """Trace bias/variance tradeoff."""
        w0 = self.rake.solve_phase1()
        min_phi = np.dot(w0, w0)

        weights = [w0]
        phis = [min_phi]
        distances = [self.rake.distance.evaluate(w0)]

        if phi_max is None:
            phi_max = min_phi * 100

        phi_grid = np.geomspace(min_phi, phi_max, num=num_points)[1:]
        w = w0
        for phi in phi_grid:
            self.rake.phi = phi
            w = self.rake.solve(w0=w)
            weights.append(w)
            phis.append(phi)
            distances.append(self.rake.distance.evaluate(w))

        return EfficientFrontierResults(weights, phis, distances)
