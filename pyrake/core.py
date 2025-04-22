from dataclasses import dataclass

import numpy as np

from .phase1 import solve_phase1
from .exceptions import ProblemInfeasibleError


@dataclass
class OptimizationSettings:
    outer_tolerance: float = 1e-6
    barrier_multiplier: float = 10.0
    inner_tolerance: float = 1e-8
    max_inner_iterations: int = 100
    backtracking_alpha: float = 0.01
    backtracking_beta: float = 0.5


class Rake:
    def __init__(self, distance, phi, settings=None):
        self.distance = distance
        self.phi = phi
        self.settings = settings or OptimizationSettings()

    def calculate_newton_step(self, w, v, X, mu, t):
        M, p = X.shape
        S = 1 - np.sum(w**2) / self.phi
        grad = t * self._grad_D(w, v) + (2 * w) / (self.phi * S) - 1 / w
        Hinv = lambda b: self.distance.solve(w, v, b)
        A = X.T / M
        AHinv = np.stack([Hinv(a_i) for a_i in A])
        S_matrix = A @ AHinv.T
        rhs = A @ Hinv(grad)
        delta_lambda = np.linalg.solve(S_matrix, rhs)
        return -Hinv(grad) + Hinv(A.T @ delta_lambda)

    def backtracking_line_search(self, w, v, delta_w, t):
        alpha = self.settings.backtracking_alpha
        beta = self.settings.backtracking_beta
        f = self._barrier_objective
        s = 1.0
        fw = f(w, v, t)
        while True:
            w_new = w + s * delta_w
            if np.any(w_new <= 0) or np.sum(w_new**2) >= self.phi:
                s *= beta
                continue
            if f(w_new, v, t) > fw + alpha * s * np.dot(self._grad_f(w, v, t), delta_w):
                s *= beta
                continue
            break
        return s

    def centering_step(self, w0, v, X, mu, t):
        w = w0.copy()
        for _ in range(self.settings.max_inner_iterations):
            delta_w = self.calculate_newton_step(w, v, X, mu, t)
            if (
                0.5 * np.dot(delta_w, self._grad_f(w, v, t))
                < self.settings.inner_tolerance
            ):
                return w
            s = self.backtracking_line_search(w, v, delta_w, t)
            w += s * delta_w
        raise RuntimeError("Centering step did not converge")

    def interior_point(self, X, mu, phi, v=None):
        M, p = X.shape
        if v is None:
            v = np.ones(M)
        w = solve_phase1(X, mu, phi=np.inf)
        t = 1.0
        for _ in range(20):
            if p / t < self.settings.outer_tolerance:
                return w
            self.phi = phi
            w = self.centering_step(w, v, X, mu, t)
            t *= self.settings.barrier_multiplier
        raise RuntimeError("Interior point method did not converge")

    def _grad_D(self, w, v):
        if isinstance(self.distance, SquaredL2):
            return w - v
        elif isinstance(self.distance, KL):
            return np.log(w / v)
        elif isinstance(self.distance, L1):
            return np.sign(w - v)
        elif isinstance(self.distance, Huber):
            d = w - v
            delta = self.distance.delta
            return np.where(np.abs(d) <= delta, d, delta * np.sign(d))
        raise NotImplementedError()

    def _barrier_objective(self, w, v, t):
        S = 1 - np.sum(w**2) / self.phi
        if S <= 0 or np.any(w <= 0):
            return np.inf
        return t * self._objective_D(w, v) - np.sum(np.log(w)) - np.log(S)

    def _objective_D(self, w, v):
        if isinstance(self.distance, SquaredL2):
            return 0.5 * np.sum((w - v) ** 2)
        elif isinstance(self.distance, KL):
            return np.sum(w * np.log(w / v) - w + v)
        elif isinstance(self.distance, L1):
            return np.sum(np.abs(w - v))
        elif isinstance(self.distance, Huber):
            d = w - v
            delta = self.distance.delta
            return np.sum(
                np.where(
                    np.abs(d) <= delta, 0.5 * d**2, delta * (np.abs(d) - 0.5 * delta)
                )
            )
        raise NotImplementedError()

    def _grad_f(self, w, v, t):
        S = 1 - np.sum(w**2) / self.phi
        return t * self._grad_D(w, v) + (2 * w) / (self.phi * S) - 1 / w
