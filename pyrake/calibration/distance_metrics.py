"""Distance penalties for PyRake's calibration objectives.

The calibration problem minimizes a distance D(w, v) between the weight vector w and a
baseline v, subject to balance constraints. ``Distance`` is the abstract base class,
itself a Cvxium ``Objective``; ``SquaredL2``, ``KLDivergence`` and ``Huber`` are the
concrete penalties. Every distance is convex in w and additive across components, so its
Hessian is diagonal and ``hessian_contribution`` wraps ``hessian_diagonal`` in a
``DiagonalSystem`` for the solver.

"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from cvxium import (
    DiagonalSystem,
    MatrixSystem,
    Objective,
)


class Distance(Objective, ABC):
    """Abstract distance penalty between weights and a baseline, as a Cvxium objective.

    A distance is a function D(w, v), for example the averaged squared error
       D(w, v) = (1 / M) ‖w - v‖₂².
    It need not satisfy the mathematical definition of a distance (no triangle
    inequality required), but it should be convex in w, so it can be minimized, and
    additive in the components of w, so that its Hessian is diagonal.

    A ``Distance`` subclasses Cvxium's ``Objective``, so it can be passed directly
    anywhere Cvxium expects one, for example as the objective of a ``Problem``. The full
    ``Objective`` interface comes from defaults built on the diagonal Hessian:
    ``hessian_contribution`` returns a ``DiagonalSystem``.

    Parameters
    ----------
     v : npt.NDArray[np.float64] | None
        Baseline the distance is measured against. When omitted, v is the vector of all
        ones. When given, its length pins the decision-variable dimension M.
     dimension : int | None
        Decision-variable dimension M. Used only when ``v`` is omitted; it may be given
        here or deferred and bound later with ``bind_dimension`` (as ``Rake`` does from
        its data).

    Raises
    ------
     ValueError
        When an explicit ``dimension`` disagrees with ``len(v)``.

    """

    def __init__(
        self,
        v: npt.NDArray[np.float64] | None = None,
        dimension: int | None = None,
    ) -> None:
        self.v = v
        if v is not None:
            M = len(v)
            if dimension is not None and dimension != M:
                raise ValueError(
                    f"Explicit `dimension` ({dimension}) disagrees with `len(v)` ({M})."
                )
            self._dimension: int | None = M
        else:
            # ``dimension`` may be deferred: a bare distance (no ``v``, no
            # ``dimension``) carries no length on its own, but a consumer that
            # owns the data (e.g. ``Rake``) binds ``M`` via ``bind_dimension``
            # before the distance is used as a ``Problem`` objective. Accessing
            # ``dimension`` while still unbound raises (see the property below).
            self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Return the decision-variable dimension M.

        Returns
        -------
         M : int
            The dimension set at construction or bound later with ``bind_dimension``.

        Raises
        ------
         ValueError
            When the dimension was deferred (bare construction) and not yet bound.

        """
        if self._dimension is None:
            raise ValueError(
                "Distance dimension is unset: construct with `v=` or `dimension=`, "
                "or bind it via `bind_dimension` (as `Rake` does) before using the "
                "distance as a `Problem` objective."
            )
        return self._dimension

    def bind_dimension(self, dimension: int) -> None:
        """Bind the decision-variable dimension M deferred at construction.

        A bare distance (no ``v``, no ``dimension``) carries no length until a consumer
        that owns the data binds M here, before the distance serves as a ``Problem``
        objective. Rebinding to the same value is a no-op.

        Parameters
        ----------
         dimension : int
            Decision-variable dimension M to bind.

        Raises
        ------
         ValueError
            When a different dimension was already set.

        """
        if self._dimension is not None and self._dimension != dimension:
            raise ValueError(
                f"Distance dimension already set to {self._dimension}; cannot "
                f"rebind to {dimension}."
            )
        self._dimension = dimension

    @abstractmethod
    def evaluate(self, w: npt.NDArray[np.float64]) -> float:
        """Evaluate the distance at w.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Weight vector, of length M.

        Returns
        -------
         d : float
            The distance D(w, v).

        """

    @abstractmethod
    def gradient(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the gradient of the distance at w.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Weight vector, of length M.

        Returns
        -------
         grad : npt.NDArray[np.float64]
            The gradient ∇D(w), of length M.

        """

    @abstractmethod
    def hessian_diagonal(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the diagonal of the Hessian at w.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Weight vector, of length M.

        Returns
        -------
         diag : npt.NDArray[np.float64]
            The Hessian diagonal, of length M.

        """

    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Return the Hessian as a diagonal system for Cvxium.

        The distance is additive in the components of ``w``, so its Hessian is diagonal;
        this wraps ``hessian_diagonal`` in a ``DiagonalSystem`` (a ``MatrixSystem``)
        that a ``Problem`` assembles via ``build_hessian_system``.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Weight vector, of length M.

        Returns
        -------
         system : MatrixSystem
            The Hessian of the distance at w, as a ``DiagonalSystem``.

        """
        return DiagonalSystem(self.hessian_diagonal(w))


class SquaredL2(Distance):
    """Squared L2 distance, averaged over components.

    The distance, gradient and Hessian diagonal are
       D(w, v) = (1 / M) ‖w - v‖₂²,
       ∇D(w) = (2 / M) (w - v),
       diag(∇²D) = (2 / M) 1.
    When ``v`` is omitted it defaults to the vector of all ones.

    Parameters
    ----------
     v : npt.NDArray[np.float64] | None
        Baseline; see ``Distance``.
     dimension : int | None
        Decision-variable dimension M; see ``Distance``.

    """

    def evaluate(self, w: npt.NDArray[np.float64]) -> float:
        """Evaluate the squared L2 distance at w.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Weight vector, of length M.

        Returns
        -------
         d : float
            The distance (1 / M) ‖w - v‖₂².

        """
        M = len(w)
        if self.v is None:
            d = w - 1.0
            return np.dot(d, d) / M
        else:
            d = w - self.v
            return np.dot(d, d) / M

    def gradient(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the gradient of the squared L2 distance at w.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Weight vector, of length M.

        Returns
        -------
         grad : npt.NDArray[np.float64]
            The gradient (2 / M) (w - v), of length M.

        """
        M = len(w)
        v = self.v
        if v is None:
            return np.asarray((2.0 / M) * (w - 1.0))
        else:
            return np.asarray((2.0 / M) * (w - v))

    def hessian_diagonal(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the Hessian diagonal of the squared L2 distance at w.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Weight vector, of length M.

        Returns
        -------
         diag : npt.NDArray[np.float64]
            The constant diagonal (2 / M) 1, of length M.

        """
        M = len(w)
        return np.full_like(w, 2.0 / M)


class KLDivergence(Distance):
    """Kullback-Leibler divergence, averaged over components.

    The distance, gradient and Hessian diagonal are
       D(w, v) = (1 / M) Σᵢ (wᵢ log(wᵢ / vᵢ) - wᵢ + vᵢ),
       ∇D(w) = (1 / M) log(w / v),
       diag(∇²D) = (1 / M) / w,
    with operations understood componentwise. When ``v`` is omitted it defaults to the
    vector of all ones.

    Parameters
    ----------
     v : npt.NDArray[np.float64] | None
        Baseline; see ``Distance``.
     dimension : int | None
        Decision-variable dimension M; see ``Distance``.

    Notes
    -----
    The domain is w > 0 (the baseline must be strictly positive too). ``evaluate`` is
    +inf outside it, which a line search reads as a rejected step, and
    ``btls_keep_in_domain`` bounds a step by the distance to the boundary, so a solver
    that minimizes over unconstrained w, such as the barrier method's dual evaluation,
    never steps out of it. ``gradient`` and ``hessian_diagonal`` assume w > 0.

    """

    def evaluate(self, w: npt.NDArray[np.float64]) -> float:
        """Evaluate the KL divergence at w.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Weight vector, of length M, strictly positive.

        Returns
        -------
         d : float
            The divergence (1 / M) Σᵢ (wᵢ log(wᵢ / vᵢ) - wᵢ + vᵢ), or +inf when any
            wᵢ ≤ 0.

        """
        if np.any(w <= 0.0):
            return np.inf
        M = len(w)
        v = self.v
        if v is None:
            return (1.0 / M) * np.sum(w * np.log(w) - w + np.ones_like(w))
        else:
            return (1.0 / M) * np.sum(w * (np.log(w) - np.log(v)) - w + v)

    def gradient(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the gradient of the KL divergence at w.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Weight vector, of length M, strictly positive.

        Returns
        -------
         grad : npt.NDArray[np.float64]
            The gradient (1 / M) log(w / v), of length M.

        """
        M = len(w)
        v = self.v
        if v is None:
            return np.asarray((1.0 / M) * np.log(w))
        else:
            return np.asarray((1.0 / M) * (np.log(w) - np.log(v)))

    def hessian_diagonal(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the Hessian diagonal of the KL divergence at w.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Weight vector, of length M, strictly positive.

        Returns
        -------
         diag : npt.NDArray[np.float64]
            The diagonal (1 / M) / w, of length M.

        """
        M = len(w)
        return (1.0 / M) / w

    def btls_keep_in_domain(
        self,
        w: npt.NDArray[np.float64],
        dw: npt.NDArray[np.float64],
    ) -> float:
        """Find the largest step along dw that keeps w + α dw > 0.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current point, strictly positive, of length M.
         dw : npt.NDArray[np.float64]
            Step direction, of length M.

        Returns
        -------
         alpha : float
            min over dwᵢ < 0 of -wᵢ / dwᵢ, the step at which the first weight reaches
            zero, or +inf when no component decreases.

        """
        decreasing = dw < 0.0
        if not np.any(decreasing):
            return np.inf
        return float(np.min(-w[decreasing] / dw[decreasing]))


class Huber(Distance):
    """Huber penalty, averaged over components.

    The distance applies the Huber function h to each residual:
       D(w, v) = (1 / M) Σᵢ h(wᵢ - vᵢ),
       h(x) = x², if |x| ≤ δ,
       h(x) = δ (2|x| - δ), if |x| > δ.
    The gradient is (2 / M) (wᵢ - vᵢ) within δ of the baseline and (2 / M) δ sign(wᵢ
    - vᵢ) outside it; the Hessian diagonal is 2 / M within δ and 0 outside it. When
    ``v`` is omitted it defaults to the vector of all ones.

    Parameters
    ----------
     v : npt.NDArray[np.float64] | None
        Baseline; see ``Distance``.
     delta : float
        Knee δ where the penalty switches from quadratic to linear.
     dimension : int | None
        Decision-variable dimension M; see ``Distance``.

    """

    def __init__(
        self,
        v: npt.NDArray[np.float64] | None = None,
        delta: float = 0.1,
        dimension: int | None = None,
    ) -> None:
        super().__init__(v=v, dimension=dimension)
        self.delta = delta

    def evaluate(self, w: npt.NDArray[np.float64]) -> float:
        """Evaluate the Huber penalty at w.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Weight vector, of length M.

        Returns
        -------
         d : float
            The penalty (1 / M) Σᵢ h(wᵢ - vᵢ).

        """
        M = len(w)
        if self.v is None:
            d = w - 1.0
        else:
            d = w - self.v

        return (1.0 / M) * np.sum(
            np.where(
                np.abs(d) <= self.delta,
                d * d,
                self.delta * (2 * np.abs(d) - self.delta),
            )
        )

    def gradient(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the gradient of the Huber penalty at w.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Weight vector, of length M.

        Returns
        -------
         grad : npt.NDArray[np.float64]
            The gradient, of length M: (2 / M) (wᵢ - vᵢ) within δ of the baseline, (2 /
            M) δ sign(wᵢ - vᵢ) outside it.

        """
        M = len(w)
        if self.v is None:
            d = w - 1.0
        else:
            d = w - self.v

        grad_dist = np.where(
            np.abs(d) <= self.delta, (2.0 / M) * d, (2.0 / M) * self.delta * np.sign(d)
        )
        return grad_dist

    def hessian_diagonal(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the Hessian diagonal of the Huber penalty at w.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Weight vector, of length M.

        Returns
        -------
         diag : npt.NDArray[np.float64]
            The diagonal, of length M: 2 / M within δ of the baseline, 0 outside.

        """
        M = len(w)
        if self.v is None:
            d = w - 1.0
        else:
            d = w - self.v

        return np.where(np.abs(d) <= self.delta, 2.0 / M, 0.0)
