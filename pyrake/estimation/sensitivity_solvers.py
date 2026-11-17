"""Cvxium solver for RatioEstimator sensitivity analysis."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from cvxium import (
    ArrowSystem,
    BarrierSolver,
    DiagonalSystem,
    EqualityConstraintLayer,
    LinearObjective,
    LineSearchSettings,
    MatrixSystem,
    OptimizationSettings,
    Problem,
    SlackConstraintLayer,
    SolveResult,
    ZeroSystem,
)


class CharnesCooperBoundsLayer(SlackConstraintLayer):
    """Charnes-Cooper bound constraints for the sensitivity LFP, as a Cvxium layer.

    Encodes the ``2n`` inequality constraints of the linear-fractional sensitivity
    program over the homogenized variable ``x = [w; s]`` (the ``n`` weights plus the
    Charnes-Cooper scaling variable ``s``):

       f1ᵢ(x) = wlᵢs − wᵢ ≤ 0, i = 1, ..., n (lower-bound block)
       f2ᵢ(x) = wᵢ − wuᵢs ≤ 0, i = 1, ..., n (upper-bound block)

    laid out as ``[lower-block, upper-block]``. These reformulate ``wlᵢ ≤ wᵢ/s ≤ wuᵢ``
    under the Charnes-Cooper transformation of ``min (cᵀw)/(yᵀw)``.

    Each constraint is affine in ``x`` (``wl``, ``wu`` are data), so the layer carries
    zero second-derivative curvature. Its barrier Hessian is NOT diagonal, however:
    because every bound couples ``wᵢ`` to the shared scaling variable ``s``, the
    Jacobian-outer term ``Gᵀ diag(1/c²)G`` has an arrow sparsity pattern -- a diagonal
    block over the weights plus a dense border row/column for ``s``.
    ``hessian_contribution`` therefore returns an ``ArrowSystem``, whose solve runs in
    O(n). A generic low-rank form would Woodbury over ``2n`` rank-1 updates and lose the
    structure, regressing the Newton step.

    Parameters
    ----------
     wl, wu : (n,) array
        Lower and upper bounds on the original weights from the sensitivity region.
        Pairs with ``wuᵢ − wlᵢ < 1e-10`` (degenerate; occurs when a weight is pinned to
        1.0) are widened symmetrically by ``1e-6`` so the log-barrier has an interior;
        the feasible band for ``wᵢ/s`` is then at most ``2e-6`` wide, so the optimum is
        unaffected to within tolerance.

    Notes
    -----
    Barrier Hessian, with the slacks ``s_loᵢ = wᵢ − wlᵢs + shift`` and ``s_upᵢ = wuᵢs −
    wᵢ + shift`` and per-constraint weights ``lmbda = [lmbda_lo; lmbda_up]``:

       g_loᵢ = lmbda_loᵢ/s_loᵢ, g_upᵢ = lmbda_upᵢ/s_upᵢ
       etaᵢ = g_loᵢ + g_upᵢ
       zetaᵢ = −wlᵢ × g_loᵢ − wuᵢ × g_upᵢ
       theta = Σᵢ(wlᵢ² × g_loᵢ + wuᵢ² × g_upᵢ)

       H = diag(eta)  zeta
           zetaᵀ      theta

    On the central path ``lmbdaᵢ = −1/fᵢ = 1/sᵢ`` gives ``g_loᵢ = 1/s_loᵢ²``, recovering
    the bespoke solver's barrier Hessian (eta, zeta, theta) exactly.

    """

    def __init__(
        self,
        wl: npt.NDArray[np.float64],
        wu: npt.NDArray[np.float64],
    ) -> None:
        """Store the (degeneracy-widened) sensitivity bounds.

        Raises
        ------
         ValueError
            If ``wl`` and ``wu`` are not 1-D arrays of equal length.

        """
        wl_arr = np.asarray(wl, dtype=np.float64).copy()
        wu_arr = np.asarray(wu, dtype=np.float64).copy()
        if wl_arr.ndim != 1 or wl_arr.shape != wu_arr.shape:
            raise ValueError(
                "wl and wu must be 1-D arrays of equal length; got "
                f"{wl_arr.shape} and {wu_arr.shape}."
            )
        # Widen degenerate bounds (wl_i == wu_i, e.g. when weight_i == 1.0) so the
        # log-barrier has an interior; the band for w_i/s is at most 2e-6 wide.
        degenerate = (wu_arr - wl_arr) < 1e-10
        if np.any(degenerate):
            wl_arr[degenerate] -= 1e-6
            wu_arr[degenerate] += 1e-6
        self.wl: npt.NDArray[np.float64] = wl_arr
        self.wu: npt.NDArray[np.float64] = wu_arr
        self.n: int = len(wl_arr)

    @property
    def dimension(self) -> int:
        """Homogenized decision-variable dimension ``M = n + 1`` (weights plus ``s``).

        Returns
        -------
         dimension : int
            Number of homogenized decision variables.

        """
        return self.n + 1

    def num_constraints(self, M: int) -> int:
        """Two bounds per weight: ``2 n`` (independent of ``M = n + 1``).

        Parameters
        ----------
         M : int
            Homogenized decision-variable dimension (unused; the count depends only on
            ``n``).

        Returns
        -------
         count : int
            Number of inequality constraints.

        """
        return 2 * self.n

    def constraints(
        self,
        w: npt.NDArray[np.float64],
        shift: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """Evaluate the stacked bound constraints.

        Parameters
        ----------
         w : (n+1,) ``npt.NDArray[np.float64]``
            Homogenized variable ``[w; s]``.
         shift : float, optional
            Barrier shift. Defaults to 0.0.

        Returns
        -------
         values : (2n,) ``npt.NDArray[np.float64]``
            ``[wl*s − w, w − wu*s]`` minus ``shift``.

        """
        weights, s = w[:-1], w[-1]
        return np.concatenate(
            [self.wl * s - weights - shift, weights - self.wu * s - shift]
        )

    def grad_constraints(
        self,
        w: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluate the constant constraint Jacobian.

        Parameters
        ----------
         w : (n+1,) ``npt.NDArray[np.float64]``
            Homogenized variable ``[w; s]`` (unused; the Jacobian is constant).

        Returns
        -------
         jacobian : (2n, n+1) ``npt.NDArray[np.float64]``
            ``[[-I, wl], [I, -wu]]``.

        """
        n = self.n
        top = np.hstack([-np.eye(n), self.wl.reshape(-1, 1)])
        bottom = np.hstack([np.eye(n), -self.wu.reshape(-1, 1)])
        return np.vstack([top, bottom])

    def grad_constraints_multiply(
        self,
        w: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute ``Gv`` without forming ``G``.

        Parameters
        ----------
         w : (n+1,) ``npt.NDArray[np.float64]``
            Homogenized variable ``[w; s]`` (unused).
         v : (n+1,) ``npt.NDArray[np.float64]``
            Vector ``[v_w; v_s]`` to multiply.

        Returns
        -------
         product : (2n,) ``npt.NDArray[np.float64]``
            Jacobian-vector product.

        """
        v_w, v_s = v[:-1], v[-1]
        return np.concatenate([-v_w + v_s * self.wl, v_w - v_s * self.wu])

    def grad_constraints_transpose_multiply(
        self,
        w: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute ``Gᵀu`` without forming ``G``.

        Parameters
        ----------
         w : (n+1,) ``npt.NDArray[np.float64]``
            Homogenized variable ``[w; s]`` (unused).
         u : (2n,) ``npt.NDArray[np.float64]``
            Vector ``[u_lo; u_up]`` to multiply.

        Returns
        -------
         product : (n+1,) ``npt.NDArray[np.float64]``
            Transposed-Jacobian-vector product.

        """
        n = self.n
        u_lo, u_up = u[:n], u[n:]
        return np.append(-u_lo + u_up, float(self.wl @ u_lo - self.wu @ u_up))

    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64] | None = None,
        shift: float = 0.0,
    ) -> MatrixSystem:
        """Return weighted barrier curvature as an ``ArrowSystem`` (see class Notes).

        ``lmbda`` defaults to ``None`` only for signature compatibility with
        ``Objective.hessian_contribution``; it is required here.

        Parameters
        ----------
         w : (n+1,) ``npt.NDArray[np.float64]``
            Homogenized variable ``[w; s]``.
         lmbda : (2n,) ``npt.NDArray[np.float64]``, optional
            Per-constraint weights. Required here despite the default.
         shift : float, optional
            Barrier shift. Defaults to 0.0.

        Returns
        -------
         system : ``MatrixSystem``
            Arrow-structured barrier Hessian contribution.

        Raises
        ------
         ValueError
            If ``lmbda`` is None.

        """
        if lmbda is None:
            raise ValueError(
                "CharnesCooperBoundsLayer.hessian_contribution requires `lmbda`."
            )
        n = self.n
        weights, s = w[:-1], w[-1]
        s_lo = weights - self.wl * s + shift  # = alpha + shift > 0, shape (n,)
        s_up = self.wu * s - weights + shift  # = beta  + shift > 0, shape (n,)
        g_lo = lmbda[:n] / s_lo
        g_up = lmbda[n:] / s_up
        eta = g_lo + g_up  # diagonal block, shape (n,)
        zeta = -self.wl * g_lo - self.wu * g_up  # border to s, shape (n,)
        theta = float(np.sum(self.wl * self.wl * g_lo + self.wu * self.wu * g_up))
        return ArrowSystem(DiagonalSystem(eta), zeta, theta)

    def hessian_curvature(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Return zero curvature: every constraint is affine in ``[w; s]``.

        Parameters
        ----------
         w : (n+1,) ``npt.NDArray[np.float64]``
            Homogenized variable ``[w; s]`` (unused).
         lmbda : (2n,) ``npt.NDArray[np.float64]``
            Per-constraint weights (unused).

        Returns
        -------
         system : ``MatrixSystem``
            Zero system of dimension ``len(w)``.

        """
        return ZeroSystem(len(w))

    def btls_max_step(
        self,
        w: npt.NDArray[np.float64],
        dw: npt.NDArray[np.float64],
        shift: float = 0.0,
        ds: float = 0.0,
        line_search: LineSearchSettings | None = None,
    ) -> float:
        """Take the largest step keeping the shifted constraints negative.

        Closed-form; ``line_search`` is ignored. ``ds_var = dw[-1]`` is the motion of
        the scaling variable ``s``, while ``ds`` is the Phase I slack motion. The lower
        constraint can bite when ``wlᵢ ds_var − dwᵢ − ds > 0``; the upper when ``dwᵢ −
        wuᵢ ds_var − ds > 0``.

        Parameters
        ----------
         w : (n+1,) ``npt.NDArray[np.float64]``
            Current homogenized variable ``[w; s]``.
         dw : (n+1,) ``npt.NDArray[np.float64]``
            Search direction.
         shift : float, optional
            Barrier shift. Defaults to 0.0.
         ds : float, optional
            Phase I slack motion. Defaults to 0.0.
         line_search : ``LineSearchSettings``, optional
            Ignored; present for signature compatibility. Defaults to None.

        Returns
        -------
         alpha : float
            Largest feasible step length (possibly infinite).

        """
        weights, s = w[:-1], w[-1]
        dw_w, dw_s = dw[:-1], dw[-1]
        alpha = np.inf

        s_lo = weights - self.wl * s + shift  # current -c_lo > 0
        coeff_lo = self.wl * dw_s - dw_w - ds
        mask_lo = coeff_lo > 0
        if np.any(mask_lo):
            alpha = min(alpha, float(np.min(s_lo[mask_lo] / coeff_lo[mask_lo])))

        s_up = self.wu * s - weights + shift  # current -c_up > 0
        coeff_up = dw_w - self.wu * dw_s - ds
        mask_up = coeff_up > 0
        if np.any(mask_up):
            alpha = min(alpha, float(np.min(s_up[mask_up] / coeff_up[mask_up])))

        return alpha


def solve_bounded_linear_fractional_program(
    c: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    wl: npt.NDArray[np.float64],
    wu: npt.NDArray[np.float64],
    settings: OptimizationSettings | None = None,
) -> SolveResult:
    """Solve the sensitivity linear fractional program via a Cvxium Problem.

    Minimizes the linear fractional objective

       (cᵀw)/(yᵀw) subject to wlᵢ ≤ wᵢ ≤ wuᵢ

    through its Charnes-Cooper transform into the linear program

       minimize cᵀ[w; s] subject to yᵀw = 1, wlᵢs − wᵢ ≤ 0, wᵢ − wuᵢs ≤ 0,

    assembled as a structured ``Problem`` -- a ``LinearObjective`` over ``[w; s]``, a
    ``CharnesCooperBoundsLayer`` for the ``2n`` bounds, and an
    ``EqualityConstraintLayer`` for ``yᵀw = 1`` -- and solved with ``BarrierSolver``.
    The bound layer emits an ``ArrowSystem`` Hessian, so the Newton step stays O(n).

    Parameters
    ----------
     c : (n,) array
        Numerator outcomes. Pass ``-c`` to maximize (the supremum bound).
     y : (n,) array
        Denominator outcomes; defines the equality constraint ``yᵀw = 1``.
     wl, wu : (n,) array
        Lower and upper bounds on the weights from the sensitivity region.
     settings : OptimizationSettings, optional
        Optimization settings.

    Returns
    -------
     result : SolveResult
        ``result.objective_value`` is the optimal fractional value (at the optimum ``yᵀw
        = 1``, so ``cᵀw = (cᵀw)/(yᵀw)``); ``result.solution`` is the optimal ``[w; s]``.

    Raises
    ------
     ValueError
        If the denominator outcomes have non-positive dot product with the midpoint of
        the sensitivity bounds.

    """
    c = np.asarray(c, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(c)

    layer = CharnesCooperBoundsLayer(wl, wu)
    problem = Problem.minimize(LinearObjective(np.append(c, 0.0))).subject_to(
        eq=EqualityConstraintLayer(
            A=np.append(y, 0.0).reshape(1, n + 1),  # [y^T, 0] @ [w; s] = 1
            b=np.array([1.0]),
        ),
        ineq=[layer],
    )

    # Analytic strictly-feasible start: the midpoint of the (widened) bounds, scaled
    # so y^T w = 1. Using the layer's widened bounds keeps the start strictly interior.
    w_mid = (layer.wl + layer.wu) / 2.0
    denom = float(y @ w_mid)
    if denom <= 0:
        raise ValueError(
            "Denominator outcomes y must have positive dot product with the "
            "midpoint of the sensitivity bounds."
        )
    x0 = np.append(w_mid / denom, 1.0 / denom)

    return problem.solve(BarrierSolver(settings=settings), x0=x0)
