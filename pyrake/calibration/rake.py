"""Weight calibration with exact and approximate balance constraints.

``Rake`` solves the single-group calibration problem, minimizing a ``Distance`` penalty
subject to equality, imbalance, weight-cap and design-effect constraints. Joint
treatment/control calibration lives in ``joint_calibrator``, which builds on this
module.

"""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
import numpy.typing as npt
from cvxium import (
    BarrierSolver,
    BoundsConstraintLayer,
    ConstraintLayer,
    EqualityConstraintLayer,
    ImbalanceConstraintLayer,
    OptimizationSettings,
    PrimalDualSolver,
    Problem,
    Solver,
    SolveResult,
    SquaredNormConstraintLayer,
)

from pyrake._typing import none_throws
from pyrake.calibration.distance_metrics import (
    Distance,
)


def _as_float_array(
    value: npt.ArrayLike | None, name: str
) -> npt.NDArray[np.float64] | None:
    """Coerce ``value`` to a float64 array, passing None through.

    Callers build these from pandas objects, and a frame that carries a single
    non-numeric column yields slices of dtype object. Numpy ufuncs cannot operate on
    those, so without this the solve fails inside the barrier loop with ``loop of ufunc
    does not support argument 0 of type float``, many frames from the input that caused
    it.

    Parameters
    ----------
     value : npt.ArrayLike | None
        Candidate array-like, or None.
     name : str
        Argument name, for the error when conversion fails.

    Returns
    -------
     arr : npt.NDArray[np.float64] | None
        ``value`` as float64, or None when ``value`` is None.

    Raises
    ------
     ValueError
        When ``value`` cannot be converted to float64.

    """
    if value is None:
        return None

    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"`{name}` must be numeric, but could not be converted to float64. "
            "If it came from a DataFrame, check for non-numeric columns, e.g. "
            "`df[covariates].astype(float)`."
        ) from e


@overload
def _as_float_scalar_or_array(
    value: float | list[float] | npt.NDArray[np.float64], name: str
) -> float | npt.NDArray[np.float64]: ...


@overload
def _as_float_scalar_or_array(
    value: float | list[float] | npt.NDArray[np.float64] | None, name: str
) -> float | npt.NDArray[np.float64] | None: ...


def _as_float_scalar_or_array(
    value: float | list[float] | npt.NDArray[np.float64] | None, name: str
) -> float | npt.NDArray[np.float64] | None:
    """Coerce ``value`` to a float or a float64 array, passing None through.

    Scalars stay scalars: callers downstream branch on whether these are sequences, and
    promoting a scalar to a 0-d array would change that.

    Parameters
    ----------
     value : float | list[float] | npt.NDArray[np.float64] | None
        Candidate scalar or array-like, or None.
     name : str
        Argument name, for the error when conversion fails.

    Returns
    -------
     out : float | npt.NDArray[np.float64] | None
        ``value`` as a float or float64 array, or None when ``value`` is None.

    Raises
    ------
     ValueError
        When ``value`` is neither a scalar nor convertible to float64.

    """
    if value is None:
        return None

    if isinstance(value, (list, tuple, np.ndarray)):
        return _as_float_array(value, name)

    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"`{name}` must be a number, but got {value!r}.") from e


class Rake:
    """Calibrated weights with exact and approximate covariate balance.

    Solves problems of the form:
       minimize    D(w, v)
       subject to  (1/M) Xᵀw = μ,
                   ‖(1/M) Zᵀw - ν‖∞ ≤ ψ (optional),
                   (1/M) ‖w‖₂² ≤ φ (optional),
                   w ≥ min_weight,
    with optional constraints on max covariate imbalance and on the mean of the weights.

    Parameters
    ----------
     distance : Distance
         Distance object representing distance from baseline weights.
     X : npt.NDArray[np.float64]
         Matrix of covariates for which we desire exact balance.
     mu : npt.NDArray[np.float64]
         Vector of mean covariate values in the target population corresponding to X.
     Z : npt.NDArray[np.float64], optional
         Matrix of covariates for which we desire approximate balance.
     nu : npt.NDArray[np.float64], optional
         Vector of mean covariate values in the target population corresponding to Z.
         Required when Z is specified.
     psi : float or list[float], optional
         Bound on maximum covariate imbalance. See Notes. Defaults to 1.5 / sqrt(M),
         following guidance from (Cochran and Chambers, 1965, §3.1).
     phi : float
         Constraint on mean squared weight.
     min_weight : float or list[float], optional
         Lower bound on weights. Defaults to 0.
     max_weight : float or list[float] or None, optional
         Upper bound on weights (a "weight cap"). A common use is to cap the influence
         of any single response: e.g. ``max_weight=20`` prevents one observation from
         driving the estimate. ``None`` (the default) disables the upper bound. Like
         ``min_weight``, may be scalar (broadcasts to all components) or a vector of
         per-component caps; vectors are particularly useful when jointly calibrating
         (the cap can vary by group).
     constrain_mean_weight_to : float or None, default=1
         If not None, add a constraint that the average weight must equal
         ``constrain_mean_weight_to``, which must be strictly positive. The default
         is an average weight of 1. Another common option is for the average
         weight to equal
         the average baseline weight. If no such constraint is desired, manually specify
         ``constrain_mean_weight_to=None``.
     settings : OptimizationSettings
        Optimization settings.

    Notes
    -----
    Three levels of balance are supported here. The lowest level implicitly relies on
    the balancing property of propensity scores: inverse weighting approximately
    balances all covariates used to fit the models, all interactions, higher order
    moments, etc. But this is only true asymptotically; with finite sample sizes, some
    imbalance may remain.

    The next level is a constraint on the maximum imbalance. For covariates
    corresponding to the inputs Z and nu, the vector of covariate imbalances, after
    weighting, is (1/M) Zᵀw - ν. We want these imbalances to be close to zero, but not
    necessarily exactly zero. Even a true random sample would have some chance
    imbalances. The ℓ∞ constraint allows us to achieve an adjustment strategy that
    mimics a random sample.

    When we assess covariate balance (using the tools in visualizations.py), we plot z
    statistics: adjusted sample mean minus population mean divided by √M times the
    population standard deviation. If we whiten Z and nu (divide by the population
    standard deviation), which we typically recommend to put all covariates on a common
    scale, then setting psi to, say 1.96 / √M will constrain covariate imbalances to be
    not-stat-sig at level 0.05. But (Cochran and Chambers, 1965, §3.1) emphasizes bias
    can occur even without stat sig differences in covariates. Their recommendation is
    equivalent to setting psi = 1.5 / √M, which is the default here.

    The highest level is to force the adjusted sample mean to match the population mean
    exactly. This can typically only be done for a few covariates without drastically
    inflating the variance.

    We thus recommend having a strong point of view on the covariates that are most
    important for predicting the response. Use propensity scores to balance all
    covariates approximately; balance the means of the more important covariates using
    the ℓ∞ constraint; balance exactly the means only of the most important covariates.

    In a sense, the ideal weight to use is (M/N) / π, where N is the population size and
    π the true propensity score. That yields unbiased estimates, but of course requires
    known propensity scores. Weights that are less than (M/N) correspond to propensity
    scores > 1, which are impossible a priori. Thus, weights should generally never be
    less than M/N, which can be incorporated into the optimization problem using the
    ``min_weight`` argument.

    ``min_weight`` and ``psi`` can both be vector arguments. This is particularly
    helpful when jointly calibrating weights for treatment and control groups. In this
    case, min_weight for the control (treated) observations should be the control
    (treated) sample size divided by the population size, so different bounds should be
    specified for the two groups. For joint calibration, we may have up to 3 groups of
    imbalance constraints, comparing treatment and control (internal validity),
    comparing control to the target population (external validity), and comparing
    treatment to the target population (external validity). The two sets of external
    validity constraints would use, e.g. 1.5 / sqrt(control sample size) or 1.5 /
    sqrt(treated sample size), while the internal validity constraint would use
       1.5 * sqrt(1 / (control sample size) + 1 / (treated sample size)),
    corresponding to a 2 sample z test.

    References
    ----------
    - Cochran, William Gemmell and Chambers, S. Paul, The Planning of Observational
      Studies of Human Populations, Journal of the Royal Statistical Society, 1965.

    """

    def __init__(
        self,
        distance: Distance,
        X: npt.NDArray[np.float64],
        mu: npt.NDArray[np.float64],
        Z: npt.NDArray[np.float64] | None = None,
        nu: npt.NDArray[np.float64] | None = None,
        psi: float | list[float] | npt.NDArray[np.float64] | None = None,
        phi: float | None = None,
        min_weight: float | list[float] | npt.NDArray[np.float64] = 0.0,
        max_weight: float | list[float] | npt.NDArray[np.float64] | None = None,
        constrain_mean_weight_to: float | None = 1.0,
        settings: OptimizationSettings | None = None,
        engine: Literal["ipm", "pd"] = "ipm",
    ) -> None:
        """Create a Rake calibrator.

        ``engine`` selects the inner solver used by ``Rake.solve()``: ``"ipm"``
        (default) routes through Cvxium's ``BarrierSolver`` over the cached ``Problem``;
        ``"pd"`` dispatches through ``PrimalDualSolver`` on an equivalent ``Problem``.
        Both engines converge to the same solution at high precision; on a single solve,
        IPM and PD are within ~20% of each other on Rake-shape problems (2026-05-21
        re-benchmark). PD's practical advantage shows up on *sweep* paths such as
        ``EfficientFrontier`` where it warm-starts naturally across Pareto-frontier
        iterations -- IPM cannot, because its barrier Phase I rejects boundary-feasible
        ``x0`` from the prior iteration. ``EfficientFrontier`` therefore defaults to PD;
        see its docstring.

        """
        if engine not in ("ipm", "pd"):
            raise ValueError(f"engine must be one of 'ipm' or 'pd' (got {engine!r}).")
        self.engine: Literal["ipm", "pd"] = engine

        # Coerce before any validation: `.shape` and the dimension checks
        # below succeed on object-dtype input, which then fails in the solver.
        X = none_throws(_as_float_array(X, "X"))
        mu = none_throws(_as_float_array(mu, "mu"))
        Z = _as_float_array(Z, "Z")
        nu = _as_float_array(nu, "nu")
        psi = _as_float_scalar_or_array(psi, "psi")
        min_weight = _as_float_scalar_or_array(min_weight, "min_weight")
        max_weight = _as_float_scalar_or_array(max_weight, "max_weight")

        M, p = X.shape
        if mu.shape[0] != p:
            raise ValueError(
                "Dimension mismatch: mu must have same number of rows as X has columns."
            )

        if Z is not None:
            if Z.shape[0] != M:
                raise ValueError(
                    "Dimension mismatch: Z must have same number of rows as X."
                )

            if nu is None:
                raise ValueError(
                    "Must specify `nu` when approximately balancing covariates."
                )

            if nu.shape[0] != Z.shape[1]:
                raise ValueError(
                    "Dimension mismatch: nu must have length equal to the number of columns in Z."
                )

            if psi is None:
                psi = 1.5 / np.sqrt(M)

            self.B: npt.NDArray[np.float64] | None = (1 / M) * Z.T
            self.c: npt.NDArray[np.float64] | None = nu
            self.psi: float | list[float] | npt.NDArray[np.float64] | None = psi
        else:
            self.B = None
            self.c = None
            self.psi = None

        if constrain_mean_weight_to is not None:
            if constrain_mean_weight_to <= 0:
                raise ValueError("constrain_mean_weight_to must be positive or None.")

            # Add a column of ones to X
            self.X: npt.NDArray[np.float64] = np.hstack((X, np.ones((M, 1))))
            # Append the new mean constraint value to mu
            self.mu: npt.NDArray[np.float64] = np.append(mu, constrain_mean_weight_to)
            self.covariates_balanced: int = p + 1
        else:
            self.X = X
            self.mu = mu
            self.covariates_balanced = p

        self.distance = distance
        self.dimension: int = M
        # A bare distance (e.g. ``KLDivergence()`` with no ``v``/``dimension``)
        # defers its dimension; bind this Rake's ``M`` so it can serve as the
        # ``Problem`` objective. Idempotent when already set to ``M``; raises on a
        # genuine mismatch (e.g. a ``v`` whose length disagrees with the data).
        self.distance.bind_dimension(M)
        self.phi = phi
        self.min_weight = min_weight
        self._validate_weight_bounds(min_weight, max_weight)
        self.max_weight = max_weight
        if settings is None:
            self.settings: OptimizationSettings = OptimizationSettings()
        else:
            self.settings = settings

        A_mat = (1 / M) * self.X.T
        self._equality_layer: EqualityConstraintLayer = EqualityConstraintLayer(
            A=A_mat, b=self.mu
        )
        # Cached structured `Problem` (objective + constraint layers + equality),
        # rebuilt whenever `phi` changes (see `update_phi`). Feasibility is handled
        # by the solver's structured Phase I, which cascades the constraint layers
        # itself; no pre-built per-layer Phase I solvers are needed.
        self._problem: Problem = self._build_problem()

    def solve(
        self,
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SolveResult:
        """Dispatch to the configured solver engine.

        ``engine='ipm'`` (default) routes through Cvxium's ``BarrierSolver``;
        ``engine='pd'`` dispatches through ``PrimalDualSolver``. Both consume the same
        cached ``Problem`` and handle equality feasibility themselves -- the barrier
        projects a supplied ``x0`` onto the manifold and runs its structured Phase I;
        PD's infeasible-start step drives the equality residual to zero.

        When ``x0`` is None it defaults -- for BOTH engines -- to a domain-informed warm
        start: ``self.distance.v`` when defined (the Distance's reference weights, e.g.
        ``KLDivergence(v=...)``) else ``np.full(M, 1.0)``, clipped into ``[min_weight,
        max_weight]`` (the bare default can fall outside the box, e.g. when ``max_weight
        < 1.0`` or when ``self.distance.v`` itself does). See ``_default_x0``. A
        user-supplied ``x0`` is passed through unchanged.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64] | None
            Starting weights, or None for the domain-informed default.

        Returns
        -------
         res : SolveResult
            The solver result; ``res.solution`` is the weight vector of length M.

        """
        solver: Solver
        if self.engine == "ipm":
            solver = BarrierSolver(self.settings)
        else:
            solver = PrimalDualSolver(self.settings)

        # Domain-informed default warm start for both engines: the barrier
        # converges faster from the Distance's reference weights than from a
        # bare min-norm seed, and PD already relied on this start. The barrier
        # projects it onto the equality manifold and Phase-I's it to strict
        # inequality feasibility; PD drives the equality residual to zero from it.
        if x0 is None:
            x0 = self._default_x0()

        return self._problem.solve(solver, x0=x0)

    def _default_x0(self) -> npt.NDArray[np.float64]:
        """Build the domain-informed default ``x0``, clipped into the bounds box.

        Used by both solver engines when no ``x0`` is supplied: ``self.distance.v`` when
        defined (e.g. ``KLDivergence(v=...)``) else ``np.full(M, 1.0)``, then
        ``np.clip`` into ``[min_weight, max_weight]`` so the start is bounds-feasible
        even when the bare default would otherwise violate the box.

        Returns
        -------
         x0 : npt.NDArray[np.float64]
            The clipped start, of length M.

        """
        distance_v = getattr(self.distance, "v", None)
        if distance_v is not None:
            x0 = np.asarray(distance_v, dtype=np.float64)
        else:
            x0 = np.full(self.dimension, 1.0)
        ub = np.inf if self.max_weight is None else self.max_weight
        return np.clip(x0, self.min_weight, ub)

    def _build_problem(self) -> Problem:
        """Assemble a Cvxium ``Problem`` mirroring this Rake's stack.

        Reuses the cached ``self._equality_layer`` (preserves its SVD cache) and
        assembles the inequality layers in the canonical order Bounds -> Imbalance? ->
        SquaredNorm?. The solver's structured Phase I cascades these layers for
        feasibility (bounds -> +imbalance -> +norm).

        The ``Distance`` objective declares a diagonal Hessian contribution, so the
        default ``BarrierSolver`` path routes this ``Problem`` through
        ``build_hessian_system`` / structured assembly: every component here contributes
        a diagonal-or-zero shape (Distance objective and BoundsConstraintLayer Diagonal,
        ImbalanceConstraintLayer Zero, SquaredNormConstraintLayer Diagonal), so the
        diagonal-plus-low-rank assembler absorbs them all. The PD path consumes the same
        ``Problem``, so the engine="pd" path is unchanged.

        Returns
        -------
         prob : Problem
            The assembled problem: distance objective plus equality and inequality
            layers.

        """
        constraints: list[ConstraintLayer] = [
            BoundsConstraintLayer(
                lb=self.min_weight, ub=self.max_weight, dimension=self.dimension
            ),
        ]
        if self.B is not None:
            assert self.c is not None
            assert self.psi is not None
            constraints.append(
                ImbalanceConstraintLayer(B=self.B, c=self.c, psi=self.psi)
            )
        if self.phi is not None:
            constraints.append(
                SquaredNormConstraintLayer(
                    phi=self.dimension * self.phi, dimension=self.dimension
                )
            )

        return Problem.minimize(self.distance).subject_to(
            eq=self._equality_layer, ineq=constraints
        )

    @staticmethod
    def _validate_weight_bounds(
        min_weight: float | list[float] | npt.NDArray[np.float64],
        max_weight: float | list[float] | npt.NDArray[np.float64] | None,
    ) -> None:
        """Reject inconsistent weight bounds at construction.

        Skipped when ``max_weight is None`` (the unconstrained case).

        Parameters
        ----------
         min_weight : float | list[float] | npt.NDArray[np.float64]
            Lower bound, scalar or per-component.
         max_weight : float | list[float] | npt.NDArray[np.float64] | None
            Upper bound, scalar or per-component, or None.

        Raises
        ------
         ValueError
            When both bounds are given but are not broadcast-compatible, or any
            component violates ``min_weight < max_weight`` strictly.

        """
        if max_weight is None:
            return
        lb_arr = np.atleast_1d(np.asarray(min_weight, dtype=np.float64))
        ub_arr = np.atleast_1d(np.asarray(max_weight, dtype=np.float64))
        try:
            lb_b, ub_b = np.broadcast_arrays(lb_arr, ub_arr)
        except ValueError as e:
            raise ValueError(
                f"`min_weight` (shape {lb_arr.shape}) and `max_weight` (shape "
                f"{ub_arr.shape}) are not broadcast-compatible."
            ) from e
        if not np.all(lb_b < ub_b):
            raise ValueError(
                "All components must satisfy `min_weight < max_weight` strictly; "
                f"got min(max_weight - min_weight) = {float(np.min(ub_b - lb_b))!r}."
            )

    @property
    def equality_layer(self) -> EqualityConstraintLayer:
        """Return the cached EqualityConstraintLayer for the (A, b) equality system.

        Use this when constructing Phase I solvers that share the same equality system
        (e.g. the Pareto-frontier corner solvers). Reusing the cached layer preserves
        the ``EqualityConstraintLayer.svd_A()`` cache across Phase I builds;
        constructing a fresh layer per call site forces an SVD recompute and defeats
        that cache.

        Returns
        -------
         layer : EqualityConstraintLayer
            The cached ``(A, b)`` equality layer.

        """
        return self._equality_layer

    def update_phi(self, phi: float | None = None) -> None:
        """Update phi, mostly for EfficientFrontier.

        Rebuilds the cached ``Problem`` so its ``SquaredNormConstraintLayer`` membership
        (present only when ``phi`` is set) matches the new ``phi``.

        Parameters
        ----------
         phi : float | None
            New design-effect bound, or None to drop the constraint.

        """
        self.phi = phi
        self._problem = self._build_problem()

    def update_psi(
        self, psi: float | list[float] | npt.NDArray[np.float64] | None = None
    ) -> None:
        """Update the imbalance tolerance, rebuilding the cached ``Problem``.

        The sibling of ``update_phi``, and necessary for the same reason: ``Problem`` is
        assembled once at construction, so assigning ``self.psi`` alone leaves ``solve``
        running against the band the calibrator was built with. Used by
        ``solve_with_fallback`` when the external band is chosen from the frontier's
        knee rather than requested up front (``frontier.solve_with_knee_fallback``).

        Parameters
        ----------
         psi : float | list[float] | npt.NDArray[np.float64] | None
            New imbalance band, or None to drop the imbalance layer.

        """
        self.psi = _as_float_scalar_or_array(psi, "psi")
        self._problem = self._build_problem()
