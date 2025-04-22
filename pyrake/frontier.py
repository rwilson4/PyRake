
import numpy as np
import matplotlib.pyplot as plt
from .phase1 import solve_phase1

class EfficientFrontierResults:
    def __init__(self, weights, phis, divergences):
        self.weights = weights
        self.phis = phis
        self.divergences = divergences

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.phis, self.divergences, marker='o')
        ax.set_xlabel(r"$\|w\|_2^2$ (variance proxy)")
        ax.set_ylabel(r"$D(w, v)$ (bias proxy)")
        ax.set_title("Efficient Frontier: Bias-Variance Tradeoff")
        return ax

class EfficientFrontier:
    def __init__(self, rake):
        self.rake = rake

    def trace(self, X, mu, v=None, phi_max=None, num_points=20):
        M, _ = X.shape
        if v is None:
            v = np.ones(M)

        w0 = solve_phase1(X, mu, phi=np.inf)
        min_phi = np.sum(w0**2)

        weights = [w0]
        phis = [min_phi]
        divergences = [self.rake._objective_D(w0, v)]

        if phi_max is None:
            phi_max = min_phi * 100

        phi_grid = np.geomspace(min_phi, phi_max, num=num_points)[1:]
        w = w0
        for phi in phi_grid:
            self.rake.phi = phi
            w = self.rake.interior_point(X, mu, phi, v=v)
            weights.append(w)
            phis.append(np.sum(w**2))
            divergences.append(self.rake._objective_D(w, v))

        return EfficientFrontierResults(weights, phis, divergences)
