"""Distance metrics."""

from abc import ABC, abstractmethod

import numpy as np


class Distance(ABC):
    r"""Abstract base class for distance metrics.

    This class facilitates calculation of gradients, Hessians, and solutions of linear
    equations involving the Hessian, all for optimization problems of the form:
       minimize ft(w) := t * D(w, v) - log(1 - (1/phi) \| w \|_2^2) - \sum_i log(w_i),
    where D is a distance metric such as KL divergence or Euclidean distance.

    The ith component of the gradient of ft is:
      grad_ft_i = t * \partial D(w, v) / \partial w_i
               + (2 * w_i / phi) / (1 - (1/phi) \| w \|_2^2)
               - 1 / w_i.
    Only the first term depends on the chosen distance metric.

    The Hessian is:
       H_ft = t * H_D(w, v)
              + ((2 / phi) / (1 - (1 / phi) * \| w \|_2^2)) * I
              + diag(1 / w)
              + zeta * zeta^T,
    where zeta = (2 * w / phi) / (1 - (1 / phi) * \| w \|_2^2). The first three terms in
    this are diagonal; only the first term depends on the distance metric chosen.

    This class also facilitates the solution of equations of the form H_ft * x = b.

    """

    def __init__(self, v: np.ndarray, phi: float) -> None:
        self.v = v
        self.phi = phi

    @abstractmethod
    def gradient(self, w: np.ndarray, t: float) -> np.ndarray:
        """Calculate gradient."""

    def _grad_constraints(self, w: np.ndarray) -> np.ndarray:
        """Calculate gradient of inequality constraints."""
        den = 1.0 - (1.0 / self.phi) * np.dot(w, w)
        return w * ((2.0 / self.phi) / den) - 1.0 / w

    @abstractmethod
    def hessian_diagonal(self, w: np.ndarray, t: float) -> np.ndarray:
        r"""Calculate diagonal component of Hessian.

        Each class that implements this abstract based class must calculate the Hessian
        of ft. As noted in the docstring of the class, the Hessian consists of a
        diagonal component and a rank-one component. The diagonal component consists of
        3 terms, two of which are the same for all distance metrics. For convenience,
        these terms are calculated by Distance._hd_constraints. Classes that implement
        this function should calculate the Hessian of t * D(w, v) and then add
        self._hd_constraints(w).

        """

    def _hd_constraints(self, w: np.ndarray) -> np.ndarray:
        """Calculate Hessian of inequality constraints."""
        den = 1.0 - (1.0 / self.phi) * np.dot(w, w)
        return (1.0 / w) + ((2.0 / self.phi) / den)

    def hessian_rank_one(self, w: np.ndarray) -> np.ndarray:
        """Calculate rank one component of Hessian."""
        den = 1.0 - (1.0 / self.phi) * np.dot(w, w)
        return w * ((2.0 / self.phi) / den)

    def solve_hessian(self, w: np.ndarray, t: float, b: np.ndarray) -> np.ndarray:
        """Solve H * x = b.

        Solves a linear system of equations where H is diagonal plus a rank one matrix.
        Because of this structure, we can solve the system in linear time. See Notes for
        more details.

        Parameters
        ----------
         w

        Returns
        -------
         x : np.ndarray
            The solution.

        Notes
        -----
        Let H = diag(eta) + zeta * zeta^T, where eta and zeta are both vectors of length
        M. We can solve H * x = b in O(M) time as follows:
           1. Solve diag(eta) * x' = b. This is just x' = b / eta, elementwise (M divisions).
           2. Solve diag(eta) * xi = zeta, or xi = zeta / eta (M divisions).
           3. Calculate x as x' - ((zeta^T * x') / (1 + zeta^T * xi)) * xi. This is 2
              dot products (each involving M multiplies and M-1 adds) plus M multiplies
              and M subtractions, or O(3M) multiplies plus O(3M) adds, plus a single add
              and a division.
        In total that's 2M divisions, 3M multiplies, and 3M adds.

        """
        eta = self.hessian_diagonal(w, t)
        zeta = self.hessian_rank_one(w)
        assert np.all(eta > 0)

        x_prime = b / eta
        xi = zeta / eta
        x = x_prime - (np.dot(zeta, x_prime) / (1.0 + np.dot(zeta, xi))) * xi
        return x


class SquaredL2(Distance):
    r"""Distance metric with D(w, v) = \| w - v \|_2^2.

    In this case, the gradient of D(w, v) is 2 * (w - v), and the Hessian is 2 * I.

    """

    def gradient(self, w: np.ndarray, t: float) -> np.ndarray:
        """Calculate gradient of ft."""
        return (2.0 * t) * (w - self.v) + self._grad_constraints(w)

    def hessian_diagonal(self, w: np.ndarray, t: float):
        """Calculate diagonal component of Hessian of ft."""
        return (2.0 * t) * np.ones_like(w) + self._hd_constraints(w)


class KL(Distance):
    r"""Distance metric with D(w, v) = KL Divergence.

    In this case, D(w, v) = \sum_{i=1}^M w_i * log(w_i / v_i) - w_i + v_i, so the
    gradient is log(w / v) and the Hessian is diag(1 / w).

    """

    def gradient(self, w: np.ndarray, t: float) -> np.ndarray:
        """Calculate gradient of ft."""
        return t * np.log(w / self.v) + self._grad_constraints(w)

    def hessian_diagonal(self, w: np.ndarray, t: float):
        """Calculate diagonal component of Hessian of ft."""
        return t / w + self._hd_constraints(w)


class Huber(Distance):
    r"""Distance metric with Huber penalty.

    In this case, D(w, v) = \sum_{i=1}^M h( w_i - v_i), where
       h(x) =           x^2               if |x| \leq \delta
              \delta * (2 * |x| - \delta) if |x| > \delta.

    In this case, the ith element of the gradient of D(w, v) is
    2 * (w_i - v_i) if |w_i - v_i| \leq \delta and
    2 * \delta * sign(w_i - v_i) if |w_i - v_i| > \delta.

    The ith diagonal element of the Hessian of D(w, v) is 2 if |w_i - v_i| \leq \delta
    and 0 if |w_i - v_i| > \delta.

    """

    def __init__(self, v: np.ndarray, phi: float, delta=0.1):
        super().__init__(v=v, phi=phi)
        self.delta = delta

    def gradient(self, w: np.ndarray, t: float) -> np.ndarray:
        """Calculate gradient of ft."""
        d = w - self.v
        grad_dist = np.where(
            np.abs(d) <= self.delta, 2.0 * d, 2.0 * self.delta * np.sign(d)
        )
        return t * grad_dist + self._grad_constraints(w)

    def hessian_diagonal(self, w: np.ndarray, t: float):
        """Calculate diagonal component of Hessian of ft."""
        d = w - self.v
        hess_dist = np.where(np.abs(d) <= self.delta, 2.0, 0.0)
        return t * hess_dist + self._hd_constraints(w)
