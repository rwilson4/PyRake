"""Distance metrics."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import numpy.typing as npt


class Distance(ABC):
    r"""Abstract base class for distance metrics.

    A distance metric is a function D(w, v), such as EuclideanDistance, \| w - v \|_2^2.
    This function need not satisfy the mathematical definition of a distance metric
    (e.g. no need for the triangle inequality to apply), but it should be convex in w
    (so we can minimize over it) and additive in the components of w (so that the
    Hessian is diagonal).

    Parameters
    ----------
     v : npt.NDArray[np.float64], optional
        The baseline against which distance is measured. If not specified, we will
        assume that v is a vector of all ones.

    """

    def __init__(self, v: Optional[npt.NDArray[np.float64]] = None) -> None:
        self.v = v

    @abstractmethod
    def evaluate(self, w: npt.NDArray[np.float64]) -> float:
        """Evaluate distance metric."""

    @abstractmethod
    def gradient(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate gradient."""

    @abstractmethod
    def hessian_diagonal(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate diagonal component of Hessian."""


class SquaredL2(Distance):
    r"""Distance metric with D(w, v) = \| w - v \|_2^2.

    In this case, the gradient of D(w, v) is 2 * (w - v), and the Hessian is 2 * I.

    """

    def evaluate(self, w: npt.NDArray[np.float64]) -> float:
        """Evaluate distance metric."""
        if self.v is None:
            d = w - 1.0
            return np.dot(d, d)
        else:
            d = w - self.v
            return np.dot(d, d)

    def gradient(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate gradient of D(w, v)."""
        if self.v is None:
            return 2.0 * (w - 1.0)
        else:
            return 2.0 * (w - self.v)

    def hessian_diagonal(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate diagonal component of Hessian of D(w, v)."""
        return np.full_like(w, 2.0)


class KLDivergence(Distance):
    r"""Distance metric with D(w, v) = KL Divergence.

    In this case, D(w, v) = \sum_{i=1}^M w_i * log(w_i / v_i) - w_i + v_i, so the
    gradient is log(w / v) and the Hessian is diag(1 / w).

    """

    def evaluate(self, w: npt.NDArray[np.float64]) -> float:
        """Evaluate distance metric."""
        if self.v is None:
            return np.sum(w * np.log(w) - w + np.ones_like(w))
        else:
            return np.sum(w * (np.log(w) - np.log(self.v)) - w + self.v)

    def gradient(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate gradient of ft."""
        if self.v is None:
            return np.log(w)
        else:
            return np.log(w) - np.log(self.v)

    def hessian_diagonal(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate diagonal component of Hessian of D(w, v)."""
        return 1.0 / w


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

    def __init__(self, v: Optional[npt.NDArray[np.float64]] = None, delta=0.1) -> None:
        super().__init__(v=v)
        self.delta = delta

    def evaluate(self, w: npt.NDArray[np.float64]) -> float:
        """Evaluate distance metric."""
        if self.v is None:
            d = w - 1.0
        else:
            d = w - self.v

        return np.sum(
            np.where(
                np.abs(d) <= self.delta,
                d * d,
                self.delta * (2 * np.abs(d) - self.delta),
            )
        )

    def gradient(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate gradient of ft."""
        if self.v is None:
            d = w - 1.0
        else:
            d = w - self.v

        grad_dist = np.where(
            np.abs(d) <= self.delta, 2.0 * d, 2.0 * self.delta * np.sign(d)
        )
        return grad_dist

    def hessian_diagonal(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate diagonal component of Hessian of ft."""
        if self.v is None:
            d = w - 1.0
        else:
            d = w - self.v

        return np.where(np.abs(d) <= self.delta, 2.0, 0.0)
