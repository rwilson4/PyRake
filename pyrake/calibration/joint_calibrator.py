"""Joint calibration of treatment/control weights with internal validity.

``JointCalibrator`` extends ``Rake`` to calibrate two experiment groups at once, trading
each group's external validity against variance while holding the groups' internal
validity fixed. ``FallbackOptimizationResult`` labels which constraint the fallback
cascade had to relax.

"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
from cvxium import (
    BacktrackingLineSearchError,
    BarrierSolver,
    BoundsConstraintLayer,
    ConstraintLayer,
    EqualityConstraintLayer,
    ImbalanceConstraintLayer,
    LiftedConstraintLayer,
    LiftedObjective,
    LinearInequalityConstraintLayer,
    LinearObjective,
    NewtonStepError,
    OptimizationError,
    OptimizationSettings,
    PrimalDualSolver,
    Problem,
    ProblemCertifiablyInfeasibleError,
    ProblemInfeasibleError,
    ProblemMarginallyFeasibleError,
    Solver,
    SolveResult,
    SquaredNormConstraintLayer,
)

from pyrake.calibration.distance_metrics import (
    Distance,
)
from pyrake.calibration.rake import (
    Rake,
    _as_float_array,
    _as_float_scalar_or_array,
)

if TYPE_CHECKING:
    # Annotation only -- ``frontier`` imports this module, so a runtime import
    # would close the cycle. ``from __future__ import annotations`` keeps the
    # field's annotation a string, so nothing is evaluated at import time.
    from pyrake.calibration.frontier import (
        ExternalValidityVarianceFrontierResults,
    )

# The cascade logs under the historic "pyrake.rake" name, not the module name, so
# existing log parsers and the narration tests keep matching one stream.
logger: logging.Logger = logging.getLogger("pyrake.rake")


_RELAXATION_TOLERANCE: float = 1e-3

# A row counts as relaxed only if it exceeds its band by more than this fraction
# of the band (or by more than the budget's slack). Differences this small are
# within what the relaxation's solves leave each other, which depends on the
# caller's solver tolerances, and reading them as relaxations would report rows
# 1% over their band as relaxed and widen bands on noise.
_RELAXED_ROW_TOLERANCE: float = 0.02

# Relative precision of the factor a too-tight band is widened by (see
# ``JointCalibrator._relax_or_widen``).
_WIDEN_TOLERANCE: float = 1e-3

# How far above the least factor the widened band sits. At the least factor the
# rows that must fit it jointly have no interior, and 0.1% above it the final
# solve still crawls (1,000 primal-dual iterations on one near-boundary
# fixture); 1% gives it room, for a band 1% looser than the least.
_WIDEN_MARGIN: float = 1e-2

# Relative slack between a relaxed layer's per-row band and its budget on the
# summed excess: the per-row band each row's achieved excess is widened by this
# fraction of its band, and the budget allows twice that, summed, so the
# per-row band sits strictly inside the budget. Smaller than
# ``_RELAXATION_TOLERANCE`` because the final solve may move the whole budget
# slack onto one row.
_BUDGET_TOLERANCE: float = 1e-4

_DESIGN_EFFECT_MESSAGE: str = (
    "We achieved the desired covariate balance (both internal and external "
    "validity), and we have minimized the design effect."
)


def _deadzone_rows(
    B: npt.NDArray[np.float64],
    c: npt.NDArray[np.float64],
    psi: npt.NDArray[np.float64],
    offset: int,
    extra: int,
    budget: float | None = None,
) -> LinearInequalityConstraintLayer:
    """Build the deadzone rows for one imbalance layer over [w; t], plus a budget.

    Holds |Bⱼw - cⱼ| - ψⱼ ≤ tⱼ row by row, and optionally Σⱼ tⱼ ≤ budget. The excess
    block ``t`` for these rows is ``[w; t][M + offset : M + offset + q]`` where
    ``extra`` is the total number of auxiliary components; each band becomes the two
    one-sided rows Bw - t ≤ c + ψ and -Bw - t ≤ -c + ψ.

    Parameters
    ----------
     B : npt.NDArray[np.float64]
        Imbalance matrix, of shape (q, M).
     c : npt.NDArray[np.float64]
        Imbalance target, of length q.
     psi : npt.NDArray[np.float64]
        Imbalance band, of length q.
     offset : int
        Offset of this layer's excess block within the auxiliary components.
     extra : int
        Total number of auxiliary components.
     budget : float | None
        Cap on the summed excess, or None for no budget row.

    Returns
    -------
     layer : LinearInequalityConstraintLayer
        The two-sided band rows over ``[w; t]``, with the budget row appended when
        ``budget`` is given.

    """
    q, M = B.shape
    E = np.zeros((q, extra))
    E[:, offset : offset + q] = np.eye(q)
    A = np.vstack([np.hstack([B, -E]), np.hstack([-B, -E])])
    cu = np.concatenate([c + psi, psi - c])
    if budget is not None:
        total = np.zeros((1, M + extra))
        total[0, M + offset : M + offset + q] = 1.0
        A = np.vstack([A, total])
        cu = np.append(cu, budget)
    return LinearInequalityConstraintLayer(A=A, cu=cu)


@dataclass(frozen=True)
class _Relaxation:
    """An infeasible imbalance layer, held to a budget on its summed excess.

    The layer is |Bⱼw - cⱼ| ≤ ψⱼ for the assembled imbalance rows ``rows``. Relaxed, it
    becomes

       Σⱼ (|Bⱼw - cⱼ| - ψⱼ)₊ ≤ budget + 2 τ Σⱼ ψⱼ,
       |Bⱼw - cⱼ| - ψⱼ ≤ pin + 2 τ ψⱼ,

    with τ = ``_BUDGET_TOLERANCE``. The slack terms leave room for the per-row band
    (``per_row_layer``), which the rest of the cascade uses, to sit strictly inside the
    budget. A row counts as outside its band only if its excess exceeds what that slack
    could put there.

    """

    label: str
    rows: slice
    B: npt.NDArray[np.float64]
    c: npt.NDArray[np.float64]
    psi: npt.NDArray[np.float64]
    budget: float
    pin: float
    # Each row's excess at the relaxation's own solution, and whether the row can
    # meet its band at all (with the background held). Rows that can are held
    # to the excess they had there; only the others may absorb excess.
    w_excess: npt.NDArray[np.float64]
    attainable: npt.NDArray[np.bool_]

    @property
    def row_indices(self) -> npt.NDArray[np.int64]:
        """Return the indices of these rows among the assembled imbalance rows.

        Returns
        -------
         idx : npt.NDArray[np.int64]
            ``arange(rows.start, rows.stop)``.

        """
        return np.arange(self.rows.start, self.rows.stop)

    def excess(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute each row's excess over its band at w.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Weight vector, of length M.

        Returns
        -------
         e : npt.NDArray[np.float64]
            |Bⱼw - cⱼ| - ψⱼ, negative inside the band.

        """
        return np.abs(self.B @ w - self.c) - self.psi

    @property
    def slack(self) -> float:
        """Return how far the budget exceeds the least achievable summed excess.

        Returns
        -------
         s : float
            A fraction ``_RELAXATION_TOLERANCE`` of that excess, so the budget is
            "within 0.1% of the least achievable", and never less than the room the
            per-row band needs. A budget pinned much tighter than that leaves the final
            solve a sliver of a feasible set, which the barrier crosses only in damped
            steps.

        """
        return max(
            _RELAXATION_TOLERANCE * self.budget,
            2 * _BUDGET_TOLERANCE * float(np.sum(self.psi)),
        )

    def rows_outside(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Flag the rows ``w`` leaves outside their band beyond what slack explains.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Weight vector, of length M.

        Returns
        -------
         bad : npt.NDArray[np.bool_]
            True for rows whose excess exceeds ``_RELAXED_ROW_TOLERANCE`` times the band
            plus the budget slack.

        """
        return self.excess(w) > _RELAXED_ROW_TOLERANCE * self.psi + self.slack

    def per_row_layer(self) -> _PerRowBand:
        """Widen each row's band to what the relaxation achieved, strictly inside.

        Returns
        -------
         layer : _PerRowBand
            Each row's band widened by what it achieved plus ``_BUDGET_TOLERANCE`` times
            its band, strictly containing the achievement.

        """
        psi = self.psi + np.maximum(self.w_excess, 0.0) + _BUDGET_TOLERANCE * self.psi
        return _PerRowBand(B=self.B, c=self.c, psi=psi, relaxation=self)

    def t_max(self) -> npt.NDArray[np.float64]:
        """Bound each row's excess: the pin, or an attainable row's own excess.

        Returns
        -------
         cap : npt.NDArray[np.float64]
            The pin for unattainable rows, the row's own excess otherwise, plus
            ``_BUDGET_TOLERANCE`` slack.

        """
        own = np.maximum(self.w_excess, 0.0)
        return (
            np.where(self.attainable, own, self.pin) + 2 * _BUDGET_TOLERANCE * self.psi
        )

    def t_start(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Start the excess variables strictly feasible with ``w`` inside the band.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Weight vector, of length M, inside ``per_row_layer``.

        Returns
        -------
         t0 : npt.NDArray[np.float64]
            The excess at ``w`` floored at zero plus half the ``_BUDGET_TOLERANCE``
            slack.

        """
        return np.maximum(self.excess(w), 0.0) + 0.5 * _BUDGET_TOLERANCE * self.psi

    def budget_layer(self, offset: int, extra: int) -> LinearInequalityConstraintLayer:
        """Build the deadzone rows and the budget over ``[w; t]``.

        Parameters
        ----------
         offset : int
            Offset of this layer's excess block within the auxiliary components.
         extra : int
            Total number of auxiliary components.

        Returns
        -------
         layer : LinearInequalityConstraintLayer
            ``_deadzone_rows`` for this layer with the budget row appended.

        """
        return _deadzone_rows(
            self.B,
            self.c,
            self.psi,
            offset,
            extra,
            budget=self.budget + self.slack,
        )

    def note(self, n_outside: int, widened: bool) -> str:
        """Describe what was relaxed, for the result's message.

        Parameters
        ----------
         n_outside : int
            Number of rows left outside their band.
         widened : bool
            Whether the band was widened before relaxing.

        Returns
        -------
         msg : str
            The message fragment, or empty when nothing was relaxed.

        """
        q = len(self.psi)
        if widened:
            if n_outside == 0:
                return ""
            return (
                f"{n_outside} of {q} {self.label} constraints could not meet even "
                "that; we held the others to it and kept the summed excess of "
                "those to the least achievable."
            )
        if n_outside == 0:
            return (
                f"Desired {self.label} was only marginally infeasible: every "
                f"{self.label} constraint was met to within "
                f"{100 * _RELAXED_ROW_TOLERANCE:g}% of its band."
            )
        return (
            f"Desired {self.label} was infeasible: {n_outside} of {q} {self.label} "
            "constraints could not be met. We held the others to their band and "
            "kept the summed excess of those to the least achievable."
        )


class _PerRowBand(ImbalanceConstraintLayer):
    """A relaxed imbalance layer as a per-row band, remembering its relaxation.

    Parameters
    ----------
     B : npt.NDArray[np.float64]
        Imbalance matrix.
     c : npt.NDArray[np.float64]
        Imbalance target.
     psi : npt.NDArray[np.float64]
        Widened per-row band.
     relaxation : _Relaxation
        The relaxation this band came from.

    """

    def __init__(
        self,
        B: npt.NDArray[np.float64],
        c: npt.NDArray[np.float64],
        psi: npt.NDArray[np.float64],
        relaxation: _Relaxation,
    ) -> None:
        super().__init__(B=B, c=c, psi=psi)
        self.relaxation: _Relaxation = relaxation


@dataclass(frozen=True)
class FallbackOptimizationResult:
    """Result of JointCalibrator.solve_with_fallback.

    Composes the underlying ``SolveResult`` with the calibration-specific fallback
    label. ``fallback_status`` is a separate axis from the solver's own
    ``SolveResult.status``: it records which constraint the cascade had to relax, not
    whether a given solve converged.

    Fields
    ------
     result : SolveResult
        The underlying solve; ``result.solution`` is the weight vector (dimension M1 +
        M2).
     fallback_message : str
        What kind of solution was achieved.
     fallback_status : int
        Numeric status code: 0 = full solution, 1 = design effect minimized (but not to
        the level desired), 2 = external validity relaxed on the constraints that could
        not be met, 3 = internal validity relaxed on the constraints that could not be
        met, 4 = external validity deliberately relaxed to the frontier's knee (the
        desired band was attainable and was declined in exchange for effective sample
        size; see ``frontier.solve_with_knee_fallback``).
     m1 : int
        Length of block 1 (control), i.e. the split point of the combined weight vector:
        ``solution[:m1]`` are the control weights and ``solution[m1:]`` the treatment
        weights.
     psi : npt.NDArray[np.float64] or None
        The imbalance band the weights were calibrated to, one entry per row of the
        assembled imbalance constraints (external control, external treatment,
        internal). Equal to the calibrator's ``psi`` unless a layer was relaxed, in
        which case the rows that could not be met carry what they achieved. None when
        there are no imbalance constraints.
     relaxed_rows : tuple[int, ...]
        Indices, among the assembled imbalance rows, of those the weights leave outside
        their requested band -- the rows that absorbed what could not be balanced. Empty
        unless a layer was relaxed.
     band_scales : dict[str, float]
        The factor by which a layer's requested band was widened, keyed by the layer
        ("internal validity", "external validity"). A band is widened only when it is
        too tight for the constraints that could each meet it to all do so at once; see
        ``JointCalibrator.solve_with_fallback``. Empty when no band was widened.

    """

    result: SolveResult
    fallback_message: str
    fallback_status: int
    # Split point of the combined weight vector. Defaults to 0 (all
    # treatment) so existing constructions that predate this field keep
    # working; the real path (from_result via solve_with_fallback) always
    # supplies the control block size.
    m1: int = 0
    # The external-validity/variance frontier, when one was traced to choose
    # the external band (``frontier.solve_with_knee_fallback``).
    # None on every other path, so the cheap paths do not start carrying a
    # grid's worth of weight vectors. Plot it with ``.plot()`` to see the
    # tradeoff the chosen band sits on.
    frontier: ExternalValidityVarianceFrontierResults | None = None
    psi: npt.NDArray[np.float64] | None = None
    relaxed_rows: tuple[int, ...] = ()
    band_scales: dict[str, float] = field(default_factory=dict)

    @property
    def solution(self) -> npt.NDArray[np.float64]:
        """Return the weight vector (delegated to the underlying solve).

        Returns
        -------
         w : npt.NDArray[np.float64]
            The combined weight vector, of length M1 + M2.

        """
        return self.result.solution

    @staticmethod
    def _effective_sample_size(weights: npt.NDArray[np.float64]) -> float:
        """Compute Kish's effective sample size for a weight vector.

        nₑff = (Σᵢ wᵢ)² / Σᵢ wᵢ². Scale-invariant, so it does not depend on the
        normalization of the weights. Zero for an empty block or an all-zero weight
        vector.

        Parameters
        ----------
         weights : npt.NDArray[np.float64]
            Weight vector.

        Returns
        -------
         n_eff : float
            The effective sample size.

        """
        if len(weights) == 0:
            return 0.0
        sum_sq = np.sum(np.square(weights))
        if sum_sq == 0.0:
            return 0.0
        return float(np.square(np.sum(weights)) / sum_sq)

    @property
    def control_effective_sample_size(self) -> float:
        """Return Kish's effective sample size of the control (block 1) weights.

        Returns
        -------
         n_eff : float
            The effective sample size of ``solution[:m1]``.

        """
        return self._effective_sample_size(self.solution[: self.m1])

    @property
    def treatment_effective_sample_size(self) -> float:
        """Return Kish's effective sample size of the treatment (block 2) weights.

        Returns
        -------
         n_eff : float
            The effective sample size of ``solution[m1:]``.

        """
        return self._effective_sample_size(self.solution[self.m1 :])

    @property
    def combined_effective_sample_size(self) -> float:
        """Return Kish's effective sample size of the pooled weights.

        Returns
        -------
         n_eff : float
            The effective sample size of the control + treatment weights.

        """
        return self._effective_sample_size(self.solution)

    @staticmethod
    def from_result(
        result: SolveResult,
        fallback_message: str,
        fallback_status: int,
        m1: int,
        psi: npt.NDArray[np.float64] | None = None,
    ) -> FallbackOptimizationResult:
        """Wrap a ``SolveResult`` with a fallback-path label.

        Parameters
        ----------
         result : SolveResult
            The underlying solve.
         fallback_message : str
            What kind of solution was achieved.
         fallback_status : int
            Numeric status code; see the class docstring.
         m1 : int
            Length of block 1 (control).
         psi : npt.NDArray[np.float64] | None
            The imbalance band the weights were calibrated to, or None when there are no
            imbalance constraints.

        Returns
        -------
         out : FallbackOptimizationResult
            The labeled result.

        """
        return FallbackOptimizationResult(
            result=result,
            fallback_message=fallback_message,
            fallback_status=fallback_status,
            m1=m1,
            psi=psi,
        )


class JointCalibrator(Rake):
    """Jointly calibrated weights for treatment and control groups.

    Solves problems of the form:
       minimize    D([w1, w2], [v1, v2])
       subject to  (1/M1) X1ᵀw1 = μ1                                 (external validity)
                   (1/M2) X2ᵀw2 = μ2                                 (external validity)
                   (1/M1) X3ᵀw1 = (1/M2) X4ᵀw2                       (internal validity)
                   ‖(1/M1) Z1ᵀw1 - ν1‖∞ ≤ ψ1                         (external validity)
                   ‖(1/M2) Z2ᵀw2 - ν2‖∞ ≤ ψ2                         (external validity)
                   ‖(1/M2) Z4ᵀw2 - (1/M1) Z3ᵀw1‖∞ ≤ ψ3               (internal validity)
                   (1/(M1 + M2)) (‖w1‖₂² + ‖w2‖₂²) ≤ φ
                   w1 ≥ min_weight1
                   w2 ≥ min_weight2

    The use case is calibrating weights for treatment and control groups at the same
    time, to achieve internal and external validity.

    Parameters
    ----------
     X1, X2, X3, X4 : npt.NDArray[np.float64], optional
        Matrices for achieving exact covariate balance.
     mu1, mu2 : npt.NDArray[np.float64], optional
        Population means corresponding to X1, X2, resp.
     Z1, Z2, Z3, Z4 : npt.NDArray[np.float64], optional
        Matrices for constraining covariate imbalance.
     nu1, nu2 : npt.NDArray[np.float64], optional
        Population means corresponding to Z1, Z2, resp.
     psi1, psi2, psi3 : float or list[float], optional
        Constraints on covariate imbalances.
     phi : float
         Constraint on mean squared weight.
     min_weight1, min_weight2 : float or list[float], optional
         Lower bound on weights. Defaults to 0.
     max_weight1, max_weight2 : float or list[float] or None, optional
         Upper bound on weights for each block (a "weight cap"). ``None`` (the
         default) disables the cap on that block. May be set independently per block;
         if neither is set the wrapped ``Rake`` runs with no upper bound, otherwise the
         unbounded side is filled with ``+inf`` internally so the bound is in effect
         only on the requested block. See ``Rake.max_weight`` for scalar/vector
         broadcast semantics and validation.
     constrain_mean_weight1_to, constrain_mean_weight2_to : float or None, default=1
         If not None, add a constraint that the average weight must equal
         ``constrain_mean_weight_to``, which must be strictly positive. The default
         is an average weight of 1. Another common option is for the average
         weight to equal
         the average baseline weight. If no such constraint is desired, manually specify
         ``constrain_mean_weight_to=None``.
     settings : OptimizationSettings
        Optimization settings.
     engine : {"ipm", "pd"}, default="ipm"
        Inner solver engine, forwarded to ``Rake.__init__``. ``"ipm"`` uses the barrier
        interior-point method (current production behavior). ``"pd"`` dispatches the
        happy-path solve through ``PrimalDualSolver``. ``solve_with_fallback`` always
        runs its feasibility probes on the IPM Phase I machinery regardless of
        ``engine`` (PrimalDual has no progressive-relaxation probe); ``engine`` governs
        only its final distance-minimizing solve.

    Notes
    -----
    This is really just a wrapper around Rake. Let:
    - w = [w1; w2]
    - v = [v1; v2],
    - M = M1 + M2
            -                                        -
           |   (M / M1) * X1ᵀ             0           |
    - Xᵀ = |          0               (M / M2) * X2ᵀ  |
           |  -(M / M1) * X3ᵀ         (M / M2) * X4ᵀ  |
            -                                        -
    - mu = [mu1; mu2; 0]
             -                                        -
            |   (M / M1) * Z1ᵀ             0           |
    - Zᵀ =  |          0               (M / M2) * Z2ᵀ  |
            |  -(M / M1) * Z3ᵀ         (M / M2) * Z4ᵀ  |
             -                                        -
    - nu = [nu1; nu2; 0]
    - psi = a vector with q1 entries of psi1, followed by q2 entries of psi2, followed
      by q3 entries of psi3, where q1, q2, and q3 are the numbers of columns in Z1, Z2,
      and Z3, respectively.
    - min_weight = a vector with M1 entries of min_weight1, followed by M2 entries of
      min_weight2, where M1 and M2 are the lengths of w1 and w2, respectively.

    Then the problem is equivalent to:
       minimize    D(w, v)
       subject to  (1/M) Xᵀw = μ                    (exact balance)
                   -ψ ≤ (1/M) Zᵀw - ν ≤ ψ           (approximate balance)
                   (1/M) ‖w‖₂² ≤ φ                  (variance)
                   w ≥ min_weight                   (positivity)

    Many of the constraints in the above optimization problem are optional. The user may
    wish to approximately balance some covariates for internal validity only, in which
    case they would specify Z3 and Z4 only, not X1, X2, X3, X4, Z1, or Z2. Then Zᵀ would
    just be:
          -                                        -
         |  -(M / M1) * Z3ᵀ        (M / M2) * Z4ᵀ   |
          -                                        -

    Due to a technical limitation in Rake, we *require* at least one equality constraint
    in the optimization problem. Thankfully, we typically want to constrain the mean
    weight to be 1, which is represented as (1/M) 1ᵀw = 1, where 1ᵀ is a vector of all
    ones. In our case, it makes sense to constrain w1 and w2 separately, e.g. (1 / M1)
    1ᵀw1 = 1 and (1 / M2) 1ᵀw2 = 1. If the user does not specify any of the X's, we can
    just use these two constraints to satisfy Rake's requirement. Otherwise,
    corresponding constraints should be prepended as the first rows of X1 and X2,
    respectively, e.g. X1 = np.vstack([np.ones((M1,)), X1]) and mu1 =
    np.concatenate([[constrain_mean_weight1_to], mu1]).

    """

    def __init__(
        self,
        distance: Distance,
        X1: npt.NDArray[np.float64] | None = None,
        X2: npt.NDArray[np.float64] | None = None,
        X3: npt.NDArray[np.float64] | None = None,
        X4: npt.NDArray[np.float64] | None = None,
        mu1: npt.NDArray[np.float64] | None = None,
        mu2: npt.NDArray[np.float64] | None = None,
        Z1: npt.NDArray[np.float64] | None = None,
        Z2: npt.NDArray[np.float64] | None = None,
        Z3: npt.NDArray[np.float64] | None = None,
        Z4: npt.NDArray[np.float64] | None = None,
        nu1: npt.NDArray[np.float64] | None = None,
        nu2: npt.NDArray[np.float64] | None = None,
        psi1: float | list[float] | npt.NDArray[np.float64] | None = None,
        psi2: float | list[float] | npt.NDArray[np.float64] | None = None,
        psi3: float | list[float] | npt.NDArray[np.float64] | None = None,
        phi: float | None = None,
        min_weight1: float | list[float] | npt.NDArray[np.float64] = 0.0,
        min_weight2: float | list[float] | npt.NDArray[np.float64] = 0.0,
        max_weight1: float | list[float] | npt.NDArray[np.float64] | None = None,
        max_weight2: float | list[float] | npt.NDArray[np.float64] | None = None,
        constrain_mean_weight1_to: float | None = 1.0,
        constrain_mean_weight2_to: float | None = 1.0,
        settings: OptimizationSettings | None = None,
        engine: Literal["ipm", "pd"] = "ipm",
    ) -> None:
        # Coerce before any validation: `.shape` and the dimension checks
        # below succeed on object-dtype input, which then fails in the solver.
        X1 = _as_float_array(X1, "X1")
        X2 = _as_float_array(X2, "X2")
        X3 = _as_float_array(X3, "X3")
        X4 = _as_float_array(X4, "X4")
        mu1 = _as_float_array(mu1, "mu1")
        mu2 = _as_float_array(mu2, "mu2")
        Z1 = _as_float_array(Z1, "Z1")
        Z2 = _as_float_array(Z2, "Z2")
        Z3 = _as_float_array(Z3, "Z3")
        Z4 = _as_float_array(Z4, "Z4")
        nu1 = _as_float_array(nu1, "nu1")
        nu2 = _as_float_array(nu2, "nu2")
        psi1 = _as_float_scalar_or_array(psi1, "psi1")
        psi2 = _as_float_scalar_or_array(psi2, "psi2")
        psi3 = _as_float_scalar_or_array(psi3, "psi3")
        min_weight1 = _as_float_scalar_or_array(min_weight1, "min_weight1")
        min_weight2 = _as_float_scalar_or_array(min_weight2, "min_weight2")
        max_weight1 = _as_float_scalar_or_array(max_weight1, "max_weight1")
        max_weight2 = _as_float_scalar_or_array(max_weight2, "max_weight2")

        M1 = M2 = None
        p1 = p2 = p3 = None
        q1 = q2 = q3 = None
        if X1 is not None:
            M1, p1 = X1.shape

            if mu1 is None:
                raise ValueError("Must specify both or neither of X1, mu1")

            if len(mu1) != p1:
                raise ValueError("Dimension mismatch")

        if X2 is not None:
            M2, p2 = X2.shape

            if mu2 is None:
                raise ValueError("Must specify both or neither of X2, mu2")

            if len(mu2) != p2:
                raise ValueError("Dimension mismatch")

        if X3 is not None:
            if M1 is not None and X3.shape[0] != M1:
                raise ValueError("Dimension mismatch")

            M1, p3 = X3.shape

        if X4 is not None:
            if M2 is not None and X4.shape[0] != M2:
                raise ValueError("Dimension mismatch")

            if p3 is not None and X4.shape[1] != p3:
                raise ValueError("Dimension mismatch")

            M2, p3 = X4.shape

        if not ((X3 is None and X4 is None) or (X3 is not None and X4 is not None)):
            raise ValueError("Must specify both or neither of X3, X4")

        if Z1 is not None:
            if M1 is not None and Z1.shape[0] != M1:
                raise ValueError("Dimension mismatch")

            M1, q1 = Z1.shape

            if nu1 is None:
                raise ValueError("Must specify both or neither of Z1, nu1")

            if len(nu1) != q1:
                raise ValueError("Dimension mismatch")

            if psi1 is None:
                psi1 = 1.5 / np.sqrt(M1)

            if isinstance(psi1, (list, tuple, np.ndarray)) and len(psi1) != q1:
                raise ValueError("Dimension mismatch")

        if Z2 is not None:
            if M2 is not None and Z2.shape[0] != M2:
                raise ValueError("Dimension mismatch")

            M2, q2 = Z2.shape

            if nu2 is None:
                raise ValueError("Must specify both or neither of Z2, nu2")

            if len(nu2) != q2:
                raise ValueError("Dimension mismatch")

            if psi2 is None:
                psi2 = 1.5 / np.sqrt(M2)

            if isinstance(psi2, (list, tuple, np.ndarray)) and len(psi2) != q2:
                raise ValueError("Dimension mismatch")

        if Z3 is not None:
            if M1 is not None and Z3.shape[0] != M1:
                raise ValueError("Dimension mismatch")

            M1, q3 = Z3.shape

            if Z4 is None:
                raise ValueError("Must specify both or neither of Z3, Z4")

            if M2 is not None and Z4.shape[0] != M2:
                raise ValueError("Dimension mismatch")

            if Z4.shape[1] != q3:
                raise ValueError("Dimension mismatch")

            M2, q3 = Z4.shape

            if psi3 is None:
                psi3 = 1.5 * np.sqrt(1 / M1 + 1 / M2)

            if isinstance(psi3, (list, tuple, np.ndarray)) and len(psi3) != q3:
                raise ValueError("Dimension mismatch")

        if M1 is None:
            if isinstance(min_weight1, (list, tuple, np.ndarray)):
                M1 = len(min_weight1)
            else:
                raise ValueError("Cannot infer weight dimension")
        elif (
            isinstance(min_weight1, (list, tuple, np.ndarray))
            and len(min_weight1) != M1
        ):
            raise ValueError("Dimension mismatch")

        if M2 is None:
            if isinstance(min_weight2, (list, tuple, np.ndarray)):
                M2 = len(min_weight2)
            else:
                raise ValueError("Cannot infer weight dimension")
        elif (
            isinstance(min_weight2, (list, tuple, np.ndarray))
            and len(min_weight2) != M2
        ):
            raise ValueError("Dimension mismatch")

        if (
            isinstance(max_weight1, (list, tuple, np.ndarray))
            and len(max_weight1) != M1
        ):
            raise ValueError("Dimension mismatch")

        if (
            isinstance(max_weight2, (list, tuple, np.ndarray))
            and len(max_weight2) != M2
        ):
            raise ValueError("Dimension mismatch")

        M = M1 + M2

        if not ((X1 is None and mu1 is None) or (X1 is not None and mu1 is not None)):
            raise ValueError("Must specify both or neither of X1, mu1")

        if not ((X2 is None and mu2 is None) or (X2 is not None and mu2 is not None)):
            raise ValueError("Must specify both or neither of X2, mu2")

        if not ((X3 is None and X4 is None) or (X3 is not None and X4 is not None)):
            raise ValueError("Must specify both or neither of X3, X4")

        if not (
            (Z1 is None and nu1 is None and psi1 is None)
            or (Z1 is not None and nu1 is not None and psi1 is not None)
        ):
            raise ValueError("Must specify all or none of Z1, nu1, psi1")

        if not (
            (Z2 is None and nu2 is None and psi2 is None)
            or (Z2 is not None and nu2 is not None and psi2 is not None)
        ):
            raise ValueError("Must specify all or none of Z2, nu2, psi2")

        if not (
            (Z3 is None and Z4 is None and psi3 is None)
            or (Z3 is not None and Z4 is not None and psi3 is not None)
        ):
            raise ValueError("Must specify all or none of Z3, Z4, psi3")

        if constrain_mean_weight1_to is not None:
            if X1 is None:
                X1 = np.ones((M1, 1))
                mu1 = np.array([constrain_mean_weight1_to])
                p1 = 1
            else:
                assert p1 is not None
                assert mu1 is not None
                X1 = np.hstack([np.ones((M1, 1)), X1])
                mu1 = np.concatenate([[constrain_mean_weight1_to], mu1])
                p1 += 1

        if constrain_mean_weight2_to is not None:
            if X2 is None:
                X2 = np.ones((M2, 1))
                mu2 = np.array([constrain_mean_weight2_to])
                p2 = 1
            else:
                assert p2 is not None
                assert mu2 is not None
                X2 = np.hstack([np.ones((M2, 1)), X2])
                mu2 = np.concatenate([[constrain_mean_weight2_to], mu2])
                p2 += 1

        if X1 is None and X2 is None and X3 is None and X4 is None:
            raise ValueError("We need at least one equality constraint.")

        # Build X and mu
        XT_blocks, mu_blocks = [], []
        if X1 is not None:
            assert p1 is not None
            assert mu1 is not None
            XT_blocks.append(np.hstack([(M / M1) * X1.T, np.zeros((p1, M2))]))
            mu_blocks.append(mu1)

        if X2 is not None:
            assert p2 is not None
            assert mu2 is not None
            XT_blocks.append(np.hstack([np.zeros((p2, M1)), (M / M2) * X2.T]))
            mu_blocks.append(mu2)

        if X3 is not None and X4 is not None:
            assert p3 is not None
            XT_blocks.append(np.hstack([-(M / M1) * X3.T, (M / M2) * X4.T]))
            mu_blocks.append(np.zeros(p3))

        X = np.vstack(XT_blocks).T
        mu = np.concatenate(mu_blocks)

        # Build Z, nu, psi
        ZT_blocks, nu_blocks, psi_blocks = [], [], []
        if Z1 is not None:
            assert q1 is not None
            assert nu1 is not None
            assert psi1 is not None
            ZT_blocks.append(np.hstack([(M / M1) * Z1.T, np.zeros((q1, M2))]))
            nu_blocks.append(nu1)

            if isinstance(psi1, (list, tuple, np.ndarray)):
                psi_blocks.append(np.asarray(psi1))
            else:
                psi_blocks.append(np.full(q1, psi1))

        if Z2 is not None:
            assert q2 is not None
            assert nu2 is not None
            ZT_blocks.append(np.hstack([np.zeros((q2, M1)), (M / M2) * Z2.T]))
            nu_blocks.append(nu2)

            if isinstance(psi2, (list, tuple, np.ndarray)):
                psi_blocks.append(np.asarray(psi2))
            else:
                psi_blocks.append(np.full(q2, psi2))

        if Z3 is not None and Z4 is not None:
            assert q3 is not None
            ZT_blocks.append(np.hstack([-(M / M1) * Z3.T, (M / M2) * Z4.T]))
            nu_blocks.append(np.zeros(q3))

            if isinstance(psi3, (list, tuple, np.ndarray)):
                psi_blocks.append(np.asarray(psi3))
            else:
                psi_blocks.append(np.full(q3, psi3))

        if len(ZT_blocks) > 0:
            Z = np.vstack(ZT_blocks).T
            nu = np.concatenate(nu_blocks)
            psi = np.concatenate(psi_blocks)
        else:
            Z = None
            nu = None
            psi = None

        # Build min_weight
        min_weight_blocks = []
        if isinstance(min_weight1, (list, tuple, np.ndarray)):
            min_weight_blocks.append(min_weight1)
        else:
            min_weight_blocks.append(np.full(M1, min_weight1))

        if isinstance(min_weight2, (list, tuple, np.ndarray)):
            min_weight_blocks.append(min_weight2)
        else:
            min_weight_blocks.append(np.full(M2, min_weight2))

        min_weight = np.concatenate(min_weight_blocks)
        max_weight = self._build_max_weight(M1, M2, max_weight1, max_weight2)

        self._q1: int | None = q1
        self._q2: int | None = q2
        self._q3: int | None = q3
        # Length of block 1 (control), so the fallback result can split its
        # combined weight vector back into control and treatment blocks.
        self._M1: int = M1
        self._M2: int = M2

        # The raw covariate blocks, kept so the external-validity/variance
        # frontier can be rebuilt from a configured calibrator. ``Rake`` keeps
        # only the assembled ``B`` and ``c``; recovering the blocks from those
        # means undoing the ``M / M_b`` scaling and the sign carried on the
        # internal rows, which would couple every caller to the layout of the
        # assembly above. ``min_weight`` needs no such treatment -- the
        # concatenated vector splits at ``M1``.
        self._Z1: npt.NDArray[np.float64] | None = Z1
        self._Z2: npt.NDArray[np.float64] | None = Z2
        self._Z3: npt.NDArray[np.float64] | None = Z3
        self._Z4: npt.NDArray[np.float64] | None = Z4
        self._nu1: npt.NDArray[np.float64] | None = nu1
        self._nu2: npt.NDArray[np.float64] | None = nu2

        # Call Rake
        super().__init__(
            distance=distance,
            X=X,
            mu=mu,
            Z=Z,
            nu=nu,
            psi=psi,
            phi=phi,
            min_weight=min_weight,
            max_weight=max_weight,
            constrain_mean_weight_to=None,  # Already handled above
            settings=settings,
            engine=engine,
        )

    @staticmethod
    def _build_max_weight(
        M1: int,
        M2: int,
        max_weight1: float | list[float] | npt.NDArray[np.float64] | None,
        max_weight2: float | list[float] | npt.NDArray[np.float64] | None,
    ) -> npt.NDArray[np.float64] | None:
        """Concatenate per-block caps; ``+inf`` fills the unbounded side.

        Returns ``None`` when neither block was capped (so the wrapped ``Rake`` runs
        without an upper bound). Cvxium's ``BoundsConstraintLayer`` reads a ``+inf``
        entry as no bound and adds no constraint for it.

        Parameters
        ----------
         M1 : int
            Length of block 1 (control).
         M2 : int
            Length of block 2 (treatment).
         max_weight1 : float | list[float] | npt.NDArray[np.float64] | None
            Cap for block 1, or None for no cap on that block.
         max_weight2 : float | list[float] | npt.NDArray[np.float64] | None
            Cap for block 2, or None for no cap on that block.

        Returns
        -------
         cap : npt.NDArray[np.float64] | None
            The concatenated caps, or None when neither block was capped.

        """
        if max_weight1 is None and max_weight2 is None:
            return None

        def _block(
            cap: float | list[float] | npt.NDArray[np.float64] | None, M: int
        ) -> npt.NDArray[np.float64]:
            """Broadcast one block's cap, using ``+inf`` for an uncapped block."""
            if cap is None:
                return np.full(M, np.inf)
            if isinstance(cap, (list, tuple, np.ndarray)):
                return np.asarray(cap, dtype=np.float64)
            return np.full(M, cap)

        return np.concatenate([_block(max_weight1, M1), _block(max_weight2, M2)])

    def solve_with_fallback(self) -> FallbackOptimizationResult:
        r"""Solve with graceful fallback through progressively relaxed constraints.

        The fallback cascade is:
        1. Try the full problem (bounds + imbalance + phi + distance minimization).
        2. If phi is infeasible, return the imbalance-feasible min-norm solution
           ("phi_relaxed").
        3. If full imbalance is infeasible but internal validity is feasible,
           hold internal validity (and the bounds) as the feasible background and
           relax the external-validity constraints that cannot be met
           ("ambivalent").
        4. If internal validity is infeasible, relax the internal-validity
           constraints that cannot be met ("sad").
        5. If even bounds + equality are infeasible, re-raise ProblemInfeasibleError.

        A relaxed imbalance layer does not end the cascade. Only the rows that cannot be
        met are loosened, holding their summed excess to the least achievable (see
        ``_relax_imbalance``); the relaxed layer then joins the feasible background and
        the remaining steps run as usual, so the design effect and the distance are
        still minimized.

        When the band is too tight for the rows that *can* meet it to all do so at once
        -- relaxing would put excess on a row that could meet the band on its own -- the
        band is first widened, by the least factor that avoids that (see
        ``_relax_or_widen``). A band below what the data allow is therefore met where it
        can be at the tightest band that does not force anything, rather than failing.

        The status reports the most severe relaxation. The result's ``psi`` holds the
        band each row was held to, ``band_scales`` the factor any band was widened by,
        and ``relaxed_rows`` the rows that could not meet even that.

        Returns
        -------
         FallbackOptimizationResult
            Contains the solution and a status message.

        Raises
        ------
         ProblemInfeasibleError
            If even the basic bounds + equality constraints are infeasible.

        """
        # Step 1: bounds + equality feasibility. Raises
        # ProblemCertifiablyInfeasibleError (a ProblemInfeasibleError) if bounds +
        # equality are fundamentally infeasible.
        bounds_layer = BoundsConstraintLayer(
            lb=self.min_weight, ub=self.max_weight, dimension=self.dimension
        )
        solver = BarrierSolver(self.settings)
        ineq: list[ConstraintLayer] = [bounds_layer]

        started = time.monotonic()
        n_eq = self._equality_layer.num_eq_constraints
        logger.info(
            "Calibrating %d weights: %d exact-balance rows, %d imbalance rows%s",
            self.dimension,
            n_eq,
            0 if self.psi is None else len(np.atleast_1d(np.asarray(self.psi))),
            "" if self.phi is None else f", design effect <= {self.phi:g}",
        )

        logger.info("Stage: bounds and exact balance")
        stage_started = time.monotonic()
        x0 = Problem(eq=self._equality_layer, ineq=ineq).solve(solver).solution
        logger.info("  feasible (%.2fs)", time.monotonic() - stage_started)

        # Steps 2-4 share one shape: append the step's constraint layer as the new
        # *active* (last) layer over the strictly-feasible background built up so
        # far, then probe feasibility via ``_feasibility_step``. A feasible probe
        # seeds the next step. An infeasible imbalance layer is replaced by its
        # relaxation, which seeds the next step in the same way; an infeasible
        # design effect ends the cascade with the minimum-norm weights.
        q_ext = (self._q1 or 0) + (self._q2 or 0)
        # (layer, short label for the log, status if it cannot be met, the rows of
        # the assembled imbalance constraints it holds -- None for the design
        # effect)
        steps: list[tuple[ConstraintLayer, str, int, slice | None]] = []
        if (
            self.B is not None
            and self.c is not None
            and self.psi is not None
            and self._q3 is not None
        ):
            steps.append(
                (
                    ImbalanceConstraintLayer(
                        B=self.B[q_ext:, :],
                        c=self.c[q_ext:],
                        psi=np.asarray(self.psi)[q_ext:],
                    ),
                    "internal validity",
                    3,
                    slice(q_ext, q_ext + self._q3),
                )
            )

        if (
            self.B is not None
            and self.c is not None
            and self.psi is not None
            and q_ext > 0
        ):
            steps.append(
                (
                    ImbalanceConstraintLayer(
                        B=self.B[:q_ext, :],
                        c=self.c[:q_ext],
                        psi=np.asarray(self.psi)[:q_ext],
                    ),
                    "external validity",
                    2,
                    slice(0, q_ext),
                )
            )

        if self.phi is not None:
            steps.append(
                (
                    SquaredNormConstraintLayer(
                        phi=self.dimension * self.phi, dimension=self.dimension
                    ),
                    "design effect",
                    1,
                    None,
                )
            )

        psi_used = self._psi_vector()
        status = 0
        relaxations: list[_Relaxation] = []
        band_scales: dict[str, float] = {}
        for layer, label, fallback_status, rows in steps:
            ineq.append(layer)
            x0, best_effort = self._feasibility_step(solver, ineq, x0, label)
            if best_effort is None:
                continue

            status = max(status, fallback_status)
            if rows is None:
                return self._finish(
                    best_effort,
                    status,
                    relaxations,
                    band_scales,
                    psi_used,
                    started,
                    False,
                )

            # The layer that replaces the infeasible one keeps the rest of the
            # cascade in `w` alone: a widened band, or a relaxation's per-row
            # band, which is a subset of the budget the final solve holds the
            # layer to -- so a step that is feasible here is feasible there.
            used, relaxation, x0, scale = self._relax_or_widen(
                solver, ineq, best_effort.solution, label, rows
            )
            ineq[-1] = used
            if relaxation is not None:
                relaxations.append(relaxation)
            if scale > 1.0:
                band_scales[label] = scale
                assert psi_used is not None  # Satisfy type-checker
                psi_used[rows] = scale * np.asarray(psi_used[rows])

        # Step 5: Full Rake solve (minimize distance).
        logger.info("Stage: minimizing the distance")
        stage_started = time.monotonic()
        if relaxations:
            result = self._solve_within_budget(ineq, relaxations, x0)
        elif band_scales:
            result = self._solve_relaxed(ineq, x0)
        else:
            result = self.solve(x0=x0)
        logger.info("  solved (%.2fs)", time.monotonic() - stage_started)
        return self._finish(
            result, status, relaxations, band_scales, psi_used, started, True
        )

    def _psi_vector(self) -> npt.NDArray[np.float64] | None:
        """Return the requested band, one entry per assembled imbalance row.

        Returns
        -------
         band : npt.NDArray[np.float64] | None
            The broadcast band, or None when there is no imbalance layer.

        """
        if self.psi is None or self.c is None:
            return None
        q = self.c.shape[0]
        return np.broadcast_to(np.asarray(self.psi, dtype=np.float64), (q,)).copy()

    def _finish(
        self,
        result: SolveResult,
        status: int,
        relaxations: list[_Relaxation],
        band_scales: dict[str, float],
        psi: npt.NDArray[np.float64] | None,
        started: float,
        design_effect_met: bool,
    ) -> FallbackOptimizationResult:
        """Label the cascade's result with its status and what was relaxed.

        The rows reported as relaxed are those the final weights leave outside their
        band -- the requested one, or the widened one where a band was widened -- and
        ``psi`` is the band each row was held to, or what it achieved where it was
        relaxed.

        Parameters
        ----------
         result : SolveResult
            The cascade's final solve.
         status : int
            The most severe relaxation status; see ``fallback_status``.
         relaxations : list[_Relaxation]
            The relaxations the cascade used.
         band_scales : dict[str, float]
            The factor any band was widened by, keyed by layer.
         psi : npt.NDArray[np.float64] | None
            The band each row was held to, updated in place.
         started : float
            ``time.monotonic`` at the cascade start, for the log line.
         design_effect_met : bool
            Whether the design-effect bound was met.

        Returns
        -------
         out : FallbackOptimizationResult
            The labeled result with its message.

        """
        w = np.asarray(result.solution)
        relaxed_rows: list[int] = []
        parts: list[str] = []
        for label, scale in band_scales.items():
            parts.append(
                f"Desired {label} was infeasible, and too tight for the "
                f"{label} constraints that can each meet it to all do so at "
                f"once, so we widened the {label} band by a factor of "
                f"{scale:.3g}, the least that lets them."
            )
        for relaxation in relaxations:
            assert psi is not None  # Satisfy type-checker
            achieved = np.abs(relaxation.B @ w - relaxation.c)
            psi[relaxation.rows] = np.maximum(relaxation.psi, achieved)
            outside = relaxation.rows_outside(w)
            relaxed_rows.extend(int(j) for j in relaxation.row_indices[outside])
            note = relaxation.note(
                int(np.sum(outside)), widened=relaxation.label in band_scales
            )
            if note:
                parts.append(note)
            logger.info(
                "  %s: %d of %d rows outside the band %s",
                relaxation.label,
                int(np.sum(outside)),
                len(relaxation.psi),
                relaxation.row_indices[outside].tolist(),
            )

        if status == 0:
            message = "Full solution achieved."
        elif status == 1:
            message = _DESIGN_EFFECT_MESSAGE
        else:
            if status == 2 and self._q3 is not None:
                parts.insert(0, "We achieved the desired internal validity.")
            if self.phi is None:
                parts.append("Within the relaxed band, we minimized the distance.")
            elif not design_effect_met:
                parts.append(
                    "The design effect could not be met within the relaxed band, "
                    "so we minimized it."
                )
            else:
                parts.append(
                    "Within the relaxed band, we achieved the desired design "
                    "effect and minimized the distance."
                )
            message = " ".join(parts)

        logger.info(
            "Done: status %d after %.2fs -- %s",
            status,
            time.monotonic() - started,
            message,
        )
        return FallbackOptimizationResult(
            result=result,
            fallback_message=message,
            fallback_status=status,
            m1=self._M1,
            psi=psi,
            relaxed_rows=tuple(relaxed_rows),
            band_scales=dict(band_scales),
        )

    def _relax_imbalance(
        self,
        solver: BarrierSolver,
        ineq: list[ConstraintLayer],
        w0: npt.NDArray[np.float64],
        label: str,
        rows: slice,
        hard: ImbalanceConstraintLayer | None = None,
        strict: bool = False,
    ) -> tuple[_Relaxation, npt.NDArray[np.float64]]:
        """Relax only the rows of an infeasible imbalance layer that cannot be met.

        ``ineq[-1]`` is the infeasible layer, |Bw - c| ≤ ψ row by row, and ``w0`` the
        best-effort Phase I iterate, which minimizes the largest excess s⋆ = maxⱼ (|Bⱼw
        - cⱼ| - ψⱼ). That minimum is degenerate: the row that sets it is the only one
        with any pull, and every other row may sit anywhere up to ψⱼ + s⋆. One
        structurally unattainable row -- a level with no respondents in one group, say
        -- then loosens every row to its own excess.

        The rows are instead relaxed by the deadzone-linear penalty (Boyd and
        Vandenberghe, 2004, section 6.1.2; as a Phase I, the sum of infeasibilities of
        section 11.4.1), holding the worst row where Phase I left it:

            D⋆ = minimize  Σⱼ tⱼ
                 subject to  |Bⱼw - cⱼ| - ψⱼ ≤ tⱼ,
                             0 ≤ tⱼ ≤ s⋆ + ε,

        with the background layers, bounds and exact balance unchanged. A row met within
        its band costs nothing, so there is no reward for doing better than ψⱼ, and the
        ℓ₁ objective concentrates the excess on as few rows as it can.

        The layer is then held to that budget rather than to the rows the linear program
        happened to choose: the final solve minimizes the distance subject to Σⱼ (|Bⱼw -
        cⱼ| - ψⱼ)₊ ≤ D⋆ plus a small slack (see ``_Relaxation``). When several rows tie
        for absorbing the excess -- as when a group's shares must sum to one and some
        level has to take up the shortfall -- the vertex the linear program lands on is
        arbitrary, and the distance, not the vertex, should decide among them.

        A relaxation solve that fails degrades to the budget the Phase I iterate
        achieves.

        ``hard`` adds rows that must meet their band, to the background of the linear
        program only; with ``strict`` a solve that fails raises rather than degrading.
        Whether the relaxation is acceptable -- whether it relaxes only rows that cannot
        meet the band -- is for ``_relax_or_widen`` to decide.

        Parameters
        ----------
         solver : BarrierSolver
            The barrier solver for the relaxation linear program.
         ineq : list[ConstraintLayer]
            The cascade's layer stack; ``ineq[-1]`` is the infeasible layer.
         w0 : npt.NDArray[np.float64]
            The best-effort Phase I iterate.
         label : str
            Layer name, for log lines ("internal validity", ...).
         rows : slice
            The rows of the assembled imbalance constraints this layer holds.
         hard : ImbalanceConstraintLayer | None
            Extra rows that must meet their band, or None.
         strict : bool
            Whether a failed relaxation solve raises instead of degrading.

        Returns
        -------
         relaxation : _Relaxation
            The relaxed layer's budget.
         w : npt.NDArray[np.float64]
            The linear program's weights, to seed the next step.

        Raises
        ------
         ProblemInfeasibleError
            When the relaxation solve fails and ``strict`` is set.

        """
        active = ineq[-1]
        assert isinstance(active, ImbalanceConstraintLayer)  # Satisfy type-checker
        B, c, psi = active.B, active.c, active.psi
        M = self.dimension

        logger.info("Stage: relaxing %s on the constraints that cannot be met", label)
        started = time.monotonic()
        excess0 = np.abs(B @ w0 - c) - psi  # shape (q,)
        s_star = max(float(np.max(excess0)), 0.0)
        # The pin's slack is relative to the excess it pins. A row that cannot
        # move at all (one exact balance already fixes it) is held to a slab of
        # width eps, and a slab much thinner than the row's own scale stalls the
        # barrier.
        eps = _RELAXATION_TOLERANCE * max(s_star, float(np.max(psi)))
        lp_ineq = ineq[:-1] + ([hard] if hard is not None else []) + [ineq[-1]]
        problem = self._relaxation_problem(lp_ineq, s_star + eps)
        t0 = np.maximum(excess0, 0.0) + 0.5 * eps
        try:
            res = problem.solve(solver, x0=np.concatenate([w0, t0]))
            w = np.asarray(res.solution[:M])
        except (
            OptimizationError,
            ProblemInfeasibleError,
            NewtonStepError,
            BacktrackingLineSearchError,
        ) as e:
            if strict:
                raise ProblemInfeasibleError(str(e)) from e
            logger.warning(
                "  relaxation failed (%s); keeping the budget Phase I achieved", e
            )
            w = w0

        relaxation = _Relaxation(
            label=label,
            rows=rows,
            B=B,
            c=c,
            psi=psi,
            budget=float(np.sum(np.maximum(np.abs(B @ w - c) - psi, 0.0))),
            pin=s_star + eps,
            w_excess=np.abs(B @ w - c) - psi,
            attainable=np.ones(B.shape[0], dtype=bool),
        )
        logger.info(
            "  summed excess %.4g, largest %.4g (%.2fs)",
            relaxation.budget,
            s_star,
            time.monotonic() - started,
        )
        return relaxation, w

    def _relax_or_widen(
        self,
        solver: BarrierSolver,
        ineq: list[ConstraintLayer],
        w0: npt.NDArray[np.float64],
        label: str,
        rows: slice,
    ) -> tuple[
        ImbalanceConstraintLayer,
        _Relaxation | None,
        npt.NDArray[np.float64],
        float,
    ]:
        """Replace an infeasible imbalance layer: relax it, widening it first if needed.

        ``ineq[-1]`` is the infeasible layer and ``w0`` the Phase I iterate, which is
        strictly inside the background ``ineq[:-1]``.

        The relaxation (``_relax_imbalance``) is meant for rows that cannot meet the
        band at all, such as a level with no respondents in one group. Each row it
        leaves outside its band is tried on its own -- the band on that row alone, over
        the background -- and a row that can meet it there is *attainable*, as is every
        row it leaves inside. If the relaxation's least summed excess can also be had
        with every attainable row inside its band at once (a tie, which the linear
        program need not break that way), that relaxation is used. If it cannot, the
        band is too tight for the attainable rows to all meet it at once: relaxing would
        put excess on one and pin the rest at the edge of their band, wherever the
        variance would put them. The band is then widened instead, by the least factor t
        ≥ 1 at which that does not happen, found by bisection in log t to
        ``_WIDEN_TOLERANCE``, and the band used sits ``_WIDEN_MARGIN`` above it so the
        final solve has an interior. At t = 1 + (s⋆ + ε) / minⱼ ψⱼ the Phase I iterate
        meets every row, so the layer is feasible there and nothing is forced. At the
        widened band the layer is either feasible, or relaxed on the rows that still
        cannot meet it.

        The bisection assumes the band forces nothing at every factor above the least
        one; where that fails it finds a factor that forces nothing, not necessarily the
        least.

        Parameters
        ----------
         solver : BarrierSolver
            The barrier solver for the probe and relaxation solves.
         ineq : list[ConstraintLayer]
            The cascade's layer stack; ``ineq[-1]`` is the infeasible layer.
         w0 : npt.NDArray[np.float64]
            Weights strictly inside the background.
         label : str
            Layer name, for log lines.
         rows : slice
            The rows of the assembled imbalance constraints this layer holds.

        Returns
        -------
         layer : ImbalanceConstraintLayer
            What the cascade carries on with: the widened band, or a relaxation's
            per-row band.
         relaxation : _Relaxation or None
            The relaxation, or None when the widened band is feasible.
         w : npt.NDArray[np.float64]
            Weights strictly inside ``layer``, to seed the next step.
         scale : float
            The factor the band was widened by; 1.0 when it was not.

        """
        background = ineq[:-1]
        active = ineq[-1]
        assert isinstance(active, ImbalanceConstraintLayer)  # Satisfy type-checker
        B, c, psi = active.B, active.c, active.psi
        q = B.shape[0]
        # Attainability only grows with the band, so remember the widest band
        # each row is known to fail and the tightest it is known to meet.
        fails_at = np.zeros(q)
        meets_at = np.full(q, np.inf)

        def attainable(j: int, scale: float) -> bool:
            """Whether row ``j`` can meet its band alone at ``scale`` times width."""
            if meets_at[j] <= scale:
                return True
            if fails_at[j] >= scale:
                return False
            alone = ImbalanceConstraintLayer(
                B=B[j : j + 1], c=c[j : j + 1], psi=scale * psi[j : j + 1]
            )
            prob = Problem(eq=self._equality_layer, ineq=[*background, alone])
            try:
                prob.solve(solver, x0=w0, on_infeasible="continue")
            except (ProblemCertifiablyInfeasibleError, ProblemMarginallyFeasibleError):
                fails_at[j] = max(fails_at[j], scale)
                return False
            meets_at[j] = min(meets_at[j], scale)
            return True

        def attempt(
            scale: float,
        ) -> tuple[
            ImbalanceConstraintLayer,
            _Relaxation | None,
            npt.NDArray[np.float64],
            bool,
        ]:
            """Return the layer at ``scale`` times the band, plus whether it forces.

            Parameters
            ----------
             scale : float
                The factor to widen the band by.

            Returns
            -------
             layer : ImbalanceConstraintLayer
                The band at ``scale`` times the requested width.
             relaxation : _Relaxation or None
                The relaxation, or None when the widened band is feasible.
             w : npt.NDArray[np.float64]
                Weights strictly inside ``layer``.
             forced : bool
                Whether any attainable row is forced outside its band.

            """
            layer = active
            start = w0
            if scale != 1.0:
                layer = ImbalanceConstraintLayer(B=B, c=c, psi=scale * psi)
                x, best = self._feasibility_step(
                    solver, [*background, layer], w0, label
                )
                if best is None:
                    return layer, None, x, False
                start = np.asarray(best.solution)
            lay_ineq = [*background, layer]
            rel, w = self._relax_imbalance(solver, lay_ineq, start, label, rows)
            excess = np.abs(B @ w - c) - scale * psi
            outside = np.flatnonzero(excess > _RELAXATION_TOLERANCE * scale * psi)
            reached = [int(j) for j in outside if attainable(int(j), scale)]
            unreached = [int(j) for j in outside if int(j) not in reached]
            if reached:
                # Every attainable row at once, not only the ones the linear
                # program left outside: holding those alone lets the excess
                # move, one for one, onto an attainable row it had left inside.
                held = np.setdiff1d(np.arange(q), unreached)
                hard = ImbalanceConstraintLayer(
                    B=B[held], c=c[held], psi=scale * psi[held]
                )
                try:
                    rel2, w2 = self._relax_imbalance(
                        solver, lay_ineq, start, label, rows, hard=hard, strict=True
                    )
                except ProblemInfeasibleError:
                    return layer, None, w, True
                if rel2.budget > rel.budget * (1 + _RELAXATION_TOLERANCE) + 1e-12:
                    return layer, None, w, True
                rel, w = rel2, w2
            mask = np.ones(q, dtype=bool)
            mask[unreached] = False
            rel = replace(rel, attainable=mask)
            return rel.per_row_layer(), rel, w, False

        layer, rel, w, forced = attempt(1.0)
        if not forced:
            return layer, rel, w, 1.0

        s_star = max(float(np.max(np.abs(B @ w0 - c) - psi)), 0.0)
        eps = _RELAXATION_TOLERANCE * max(s_star, float(np.max(psi)))
        logger.info(
            "  the %s band forces constraints that could meet it alone; widening",
            label,
        )
        lo, hi = 1.0, 1.0 + (s_star + eps) / float(np.min(psi))
        while hi / lo > 1.0 + _WIDEN_TOLERANCE:
            mid = math.sqrt(lo * hi)
            if attempt(mid)[3]:
                lo = mid
            else:
                hi = mid
        scale = hi * (1.0 + _WIDEN_MARGIN)
        layer, rel, w, forced = attempt(scale)
        if forced:
            logger.warning(
                "  constraints are still forced at %.4g times the %s band; the "
                "bisection's monotonicity assumption failed here",
                scale,
                label,
            )
        logger.info("  widened the %s band by a factor of %.4g", label, scale)
        return layer, rel, w, scale

    def _bounds_over(
        self, extra: int, t_max: npt.NDArray[np.float64]
    ) -> BoundsConstraintLayer:
        """Build the weight bounds over ``[w; t]``, with 0 ≤ t ≤ ``t_max``.

        Parameters
        ----------
         extra : int
            Number of auxiliary components.
         t_max : npt.NDArray[np.float64]
            Upper bound per auxiliary component.

        Returns
        -------
         layer : BoundsConstraintLayer
            Bounds on ``[w; t]`` of dimension M + extra.

        """
        M = self.dimension
        lb_w = np.broadcast_to(np.asarray(self.min_weight, dtype=np.float64), (M,))
        if self.max_weight is None:
            ub_w = np.full(M, np.inf)
        else:
            ub_w = np.broadcast_to(np.asarray(self.max_weight, dtype=np.float64), (M,))
        return BoundsConstraintLayer(
            lb=np.concatenate([lb_w, np.zeros(extra)]),
            ub=np.concatenate([ub_w, t_max]),
            dimension=M + extra,
        )

    def _equality_over(self, extra: int) -> EqualityConstraintLayer:
        """Lift exact balance over ``[w; t]``; the auxiliary ``t`` does not enter it.

        Parameters
        ----------
         extra : int
            Number of auxiliary components.

        Returns
        -------
         layer : EqualityConstraintLayer
            The equality system with zero columns appended for ``t``.

        """
        A = self._equality_layer.A
        return EqualityConstraintLayer(
            A=np.hstack([A, np.zeros((A.shape[0], extra))]), b=self._equality_layer.b
        )

    def _relaxation_problem(self, ineq: list[ConstraintLayer], t_max: float) -> Problem:
        """Build the deadzone-linear relaxation of ``ineq[-1]`` over ``[w; t]``.

        ``ineq[0]`` is the bounds layer and ``ineq[1:-1]`` the imbalance layers already
        in the background, which are lifted to constrain ``w`` alone.

        Parameters
        ----------
         ineq : list[ConstraintLayer]
            The cascade's layer stack; ``ineq[-1]`` is the layer to relax.
         t_max : float
            Upper bound on each row's excess.

        Returns
        -------
         prob : Problem
            The linear program minimizing Σⱼ tⱼ over ``[w; t]``.

        """
        active = ineq[-1]
        assert isinstance(active, ImbalanceConstraintLayer)  # Satisfy type-checker
        q = active.q
        M = self.dimension
        layers: list[ConstraintLayer] = [self._bounds_over(q, np.full(q, t_max))]
        for layer in ineq[1:-1]:
            layers.append(
                LiftedConstraintLayer(base=layer, base_dimension=M, extra_dimension=q)
            )
        layers.append(_deadzone_rows(active.B, active.c, active.psi, 0, q))
        objective = LinearObjective(c=np.concatenate([np.zeros(M), np.ones(q)]))
        return Problem.minimize(objective).subject_to(
            eq=self._equality_over(q), ineq=layers
        )

    def _solve_within_budget(
        self,
        ineq: list[ConstraintLayer],
        relaxations: list[_Relaxation],
        x0: npt.NDArray[np.float64],
    ) -> SolveResult:
        """Minimize the distance, holding each relaxed layer to its budget.

        Solved over ``[w; t₁; t₂; ...]``, one block of ``t`` per relaxed layer:

            minimize    D(w, v)
            subject to  |Bⱼw - cⱼ| - ψⱼ ≤ tⱼ,  0 ≤ tⱼ ≤ pin,
                        Σⱼ tⱼ ≤ budget          (each relaxed layer)
                        the cascade's other layers, bounds and exact balance.

        ``ineq`` is the cascade's layer stack, in which each relaxed layer sits as its
        per-row band; those are replaced by the budget and the rest are lifted. ``x0``
        is strictly inside the per-row bands, hence strictly inside the budgets.

        If this solve fails, the distance is minimized within the per-row bands instead
        -- a subset of the budget, so feasible but possibly a worse weighting.

        Parameters
        ----------
         ineq : list[ConstraintLayer]
            The cascade's layer stack with per-row bands for relaxed layers.
         relaxations : list[_Relaxation]
            The relaxations whose budgets replace the per-row bands.
         x0 : npt.NDArray[np.float64]
            Weights strictly inside the per-row bands.

        Returns
        -------
         res : SolveResult
            The minimizing weights (first M components of the lifted solution).

        """
        M = self.dimension
        extra = sum(len(r.psi) for r in relaxations)

        layers: list[ConstraintLayer] = []
        t_max_blocks: list[npt.NDArray[np.float64]] = []
        t0_blocks: list[npt.NDArray[np.float64]] = []
        offset = 0
        for layer in ineq[1:]:
            if not isinstance(layer, _PerRowBand):
                layers.append(
                    LiftedConstraintLayer(
                        base=layer, base_dimension=M, extra_dimension=extra
                    )
                )
                continue
            relaxation = layer.relaxation
            layers.append(relaxation.budget_layer(offset, extra))
            t_max_blocks.append(relaxation.t_max())
            t0_blocks.append(relaxation.t_start(x0))
            offset += len(relaxation.psi)

        problem = Problem.minimize(LiftedObjective(self.distance, extra)).subject_to(
            eq=self._equality_over(extra),
            ineq=[self._bounds_over(extra, np.concatenate(t_max_blocks)), *layers],
        )
        # Always the barrier: the primal-dual engine's equality phase divides by
        # a zero barrier parameter on this lifted problem.
        try:
            res = problem.solve(
                BarrierSolver(self.settings), x0=np.concatenate([x0, *t0_blocks])
            )
        except (
            OptimizationError,
            ProblemInfeasibleError,
            NewtonStepError,
            BacktrackingLineSearchError,
        ) as e:
            logger.warning(
                "  minimizing within the budget failed (%s); minimizing within "
                "the per-row bands instead",
                e,
            )
            return self._solve_relaxed(ineq, x0)
        return replace(res, solution=np.asarray(res.solution)[:M])

    def _solve_relaxed(
        self, ineq: list[ConstraintLayer], x0: npt.NDArray[np.float64]
    ) -> SolveResult:
        """Minimize the distance subject to the cascade's (relaxed) layers.

        Parameters
        ----------
         ineq : list[ConstraintLayer]
            The cascade's layer stack.
         x0 : npt.NDArray[np.float64]
            Feasible starting weights.

        Returns
        -------
         res : SolveResult
            The minimizing weights.

        """
        solver: Solver
        if self.engine == "ipm":
            solver = BarrierSolver(self.settings)
        else:
            solver = PrimalDualSolver(self.settings)
        problem = Problem.minimize(self.distance).subject_to(
            eq=self._equality_layer, ineq=list(ineq)
        )
        return problem.solve(solver, x0=x0)

    def _feasibility_step(
        self,
        solver: BarrierSolver,
        ineq: list[ConstraintLayer],
        x0: npt.NDArray[np.float64],
        label: str,
    ) -> tuple[npt.NDArray[np.float64], SolveResult | None]:
        """Run one feasibility step of the fallback cascade.

        Solves ``{eq, ineq}`` from ``x0`` with ``on_infeasible="continue"``, where the
        last ``ineq`` layer is the one whose feasibility is probed and the earlier
        layers are the strictly-feasible background carried over from prior steps.

        Returns ``(next_x0, None)`` on success -- the feasible iterate that seeds the
        next step. On infeasibility returns ``(next_x0, best_effort)`` where
        ``best_effort`` is the minimum-violation result.

        Parameters
        ----------
         solver : BarrierSolver
            The barrier solver for the probe.
         ineq : list[ConstraintLayer]
            The background layers plus the probed layer last.
         x0 : npt.NDArray[np.float64]
            Feasible start for the background.
         label : str
            Stage name, for the log line.

        Returns
        -------
         next_x0 : npt.NDArray[np.float64]
            The iterate that seeds the next step.
         best_effort : SolveResult | None
            None on success; the minimum-violation result on infeasibility.

        """
        logger.info("Stage: %s", label)
        started = time.monotonic()
        prob = Problem(eq=self._equality_layer, ineq=ineq)
        try:
            res = prob.solve(solver, x0=x0, on_infeasible="continue")
        except (
            ProblemCertifiablyInfeasibleError,
            ProblemMarginallyFeasibleError,
        ) as e:
            # Cvxium has already de-augmented the best-effort result: the barrier
            # ``finalize_solution`` drops the slack-form Phase I's trailing slack,
            # so the recovered solution is at the original dimension.
            best_effort = e.solve_result
            # The expensive branch, and the one worth timing: minimizing a
            # violation from a deeply infeasible start is where a cascade that
            # looks stuck actually spends itself.
            logger.info(
                "  INFEASIBLE, minimized the violation instead (%.2fs)",
                time.monotonic() - started,
            )
            return best_effort.solution, best_effort
        logger.info("  feasible (%.2fs)", time.monotonic() - started)
        return res.solution, None
