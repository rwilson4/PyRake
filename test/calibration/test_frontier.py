"""Tests for the bias/variance and external-validity/variance frontiers."""

import itertools
import math
import time
import unittest
from dataclasses import replace
from typing import Any, Literal, cast
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest
from cvxium import (
    FrontierPoint,
    NewtonStepError,
    OptimizationError,
    ProblemInfeasibleError,
    SolveResult,
    SolveStatus,
)
from parameterized import parameterized
from scipy.special import expit, logit

from pyrake._typing import none_throws
from pyrake.calibration.distance_metrics import (
    KLDivergence,
    SquaredL2,
)
from pyrake.calibration.frontier import (
    AnnotationPoint,
    EfficientFrontier,
    EfficientFrontierResults,
    ExternalValidityVarianceFrontier,
    ExternalValidityVarianceFrontierResults,
    _cvxium_knee_space,
)
from pyrake.calibration.joint_calibrator import (
    JointCalibrator,
)
from pyrake.calibration.rake import Rake


def _make_rake_for_default_engine_test(engine: Literal["ipm", "pd"] = "ipm") -> Rake:
    rng = np.random.default_rng(20260521)
    M, p, q = 80, 4, 3
    X = rng.standard_normal((M, p))
    Z = 0.1 * rng.standard_normal((M, q))
    w_target = rng.uniform(0.5, 2.0, size=M)
    mu = (1.0 / M) * X.T @ w_target
    nu = (1.0 / M) * Z.T @ w_target
    return Rake(
        distance=SquaredL2(dimension=M),
        X=X,
        mu=mu,
        Z=Z,
        nu=nu,
        min_weight=0.0,
        max_weight=10.0,
        constrain_mean_weight_to=None,
        engine=engine,
    )


class EfficientFrontierTest(unittest.TestCase):
    """Tests for ``EfficientFrontier``."""

    @parameterized.expand(
        [
            ("M100_p20", 101, 100, 20),
            ("M200_p30", 201, 200, 30),
            ("M50_p5", 301, 50, 5),
            ("M500_p100", 401, 500, 100),
            ("M13_p3", 501, 13, 3),
        ]
    )
    def test_trace(self, name: str, seed: int, M: int, p: int) -> None:
        """Test EfficientFrontier."""
        np.random.seed(seed)
        X = np.random.rand(M, p)

        # To generate population mean, simulate true propensity scores
        # with mean 0.1 and variance 0.0045
        q = 0.1
        sigma2 = 0.05 * q * (1 - q)
        s = q * (1 - q) / sigma2 - 1
        alpha = q * s
        beta = (1 - q) * s
        true_propensity = np.random.beta(alpha, beta, size=M)

        # Ideal weights are (M/N) / true_propensity
        w = 1.0 / true_propensity
        w /= np.mean(w)

        # Compute population mean
        mu = X.T @ w / M

        # Now forget we know w. We just know X for respondents and mu
        # for the target population.

        # Now test specifying baseline weights
        estimated_propensity = expit(logit(true_propensity) + 0.1 * np.random.randn(M))

        rake = Rake(
            distance=KLDivergence(v=np.reciprocal(estimated_propensity)),
            X=X,
            mu=mu,
            phi=2.0,
        )

        frontier = EfficientFrontier(rake)
        start_time = time.time()
        efr = frontier.trace()
        end_time = time.time()
        print(f"Complete in {1e3 * (end_time - start_time):.03f} ms")

        # Convenience properties are sorted by variance ascending.
        self.assertLessEqual(efr.variances[0], efr.variances[-1])
        self.assertLessEqual(efr.distances[-1], efr.distances[0])

        # corners[0] minimizes distance; corners[1] minimizes variance.
        self.assertLessEqual(efr.corners[1].objectives[1], efr.corners[0].objectives[1])
        self.assertGreaterEqual(
            efr.corners[1].objectives[0], efr.corners[0].objectives[0]
        )

        # knee() replaces max_chord_distance().
        knee = efr.knee()
        weights = knee.solution
        np.testing.assert_allclose((1 / M) * (X.T @ weights), mu)
        self.assertTrue(np.all(weights >= 0))

        variance_reduction = (efr.variances[-1] - knee.objectives[1]) / (
            efr.variances[-1] - efr.variances[0]
        )
        bias_increase = (knee.objectives[0] - efr.distances[-1]) / (
            efr.distances[0] - efr.distances[-1]
        )

        # 80/20 rule -- more like 70/30 in our case
        self.assertGreater(variance_reduction, 0.7)
        self.assertLess(bias_increase, 0.3)

    @parameterized.expand(
        [
            ("M200_p8_q5_imbalance", 5101, 200, 8, 5, True),
            ("M200_p8_q5_no_imbalance", 5102, 200, 8, 5, False),
            ("M100_p5_q0_no_imbalance", 5103, 100, 5, 0, False),
            ("M300_p10_q8_imbalance", 5104, 300, 10, 8, True),
        ]
    )
    def test_pd_engine_matches_ipm_engine(
        self,
        name: str,
        seed: int,
        M: int,
        p: int,
        q: int,
        with_imbalance: bool,
    ) -> None:
        """``EfficientFrontier`` Pareto frontier agrees across engines.

        Validates P5 D1's PD-routed corner-1 path (variance minimization via
        ``SquaredL2(v=zeros)``) against the existing MCVS / IPM corner.

        Compares scalar objective values (variance, distance) at both corners and the
        knee, since both solvers find the same optimum but IPM's log barrier holds
        boundary-active components ~1e-7 inside the feasible region while PD lands at
        ~1e-10. Componentwise weight comparison would fail despite identical objectives.
        The relevant invariants are: equality satisfied, bounds satisfied, objective
        values agree, and equivalent points found in the (distance, variance) plane.

        """
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((M, p))
        w_target = rng.uniform(0.5, 2.0, size=M)
        mu = (1.0 / M) * X.T @ w_target

        Z: npt.NDArray[np.float64] | None = None
        nu: npt.NDArray[np.float64] | None = None
        psi: float | None = None
        if with_imbalance:
            Z = 0.1 * rng.standard_normal((M, q))
            assert Z is not None  # Satisfy type-checker
            nu = (1.0 / M) * Z.T @ w_target
            base_resid = float(np.max(np.abs((1.0 / M) * Z.T @ np.ones(M) - nu)))
            psi = 1.5 * base_resid

        def make_frontier(engine: Literal["ipm", "pd"]) -> EfficientFrontier:
            rake = Rake(
                distance=SquaredL2(dimension=M),
                X=X,
                mu=mu,
                Z=Z,
                nu=nu,
                psi=psi,
                min_weight=0.0,
                max_weight=10.0,
                constrain_mean_weight_to=None,
                engine=engine,
            )
            # Pass engine=None so this test exercises the engine set on the
            # Rake rather than EfficientFrontier's PD default.
            return EfficientFrontier(rake, engine=None)

        ipm_frontier = make_frontier("ipm")
        pd_frontier = make_frontier("pd")

        ipm_results = ipm_frontier.trace(num_points=10)
        pd_results = pd_frontier.trace(num_points=10)

        # Corner 0 (distance-min) goes through Rake.solve in both engines;
        # objectives must agree.
        np.testing.assert_allclose(
            ipm_results.corners[0].objectives,
            pd_results.corners[0].objectives,
            rtol=1e-4,
            atol=1e-6,
            err_msg="corner 0 objectives",
        )

        # Corner 1 (variance-min) is PD-routed in PD engine, MCVS in IPM
        # engine.  When imbalance is active, MCVS may park at a feasible
        # point with `f_active = ||w||^2/M - phi = 0` instead of descending
        # further (a known MCVS conservatism — the active value is at zero
        # so the Newton search direction shrinks).  PD's natural objective
        # formulation has no such barrier-induced floor and reliably finds
        # the true min-variance point.  The right cross-engine invariant
        # is therefore "PD finds a corner at least as good as IPM" rather
        # than equality.
        ipm_var_corner1 = ipm_results.corners[1].objectives[1]
        pd_var_corner1 = pd_results.corners[1].objectives[1]
        self.assertLessEqual(
            pd_var_corner1,
            ipm_var_corner1 + 1e-6,
            f"PD corner-1 variance {pd_var_corner1:.4e} should not exceed "
            f"IPM corner-1 variance {ipm_var_corner1:.4e}",
        )

        # Both engines' corner solutions must be feasible.
        for corner_idx in (0, 1):
            for label, w in (
                ("ipm", ipm_results.corners[corner_idx].solution),
                ("pd", pd_results.corners[corner_idx].solution),
            ):
                np.testing.assert_allclose(
                    (1.0 / M) * X.T @ w,
                    mu,
                    atol=1e-6,
                    err_msg=f"{label} corner {corner_idx} equality",
                )
                self.assertTrue(
                    np.all(w >= -1e-9), f"{label} corner {corner_idx} bounds"
                )

    def test_pd_default_x0_clamped_to_max_weight(self) -> None:
        """``EfficientFrontier`` corner-1 PD path clips the default ``x0=ones``.

        ``EfficientFrontier.minimize_objective(1)`` with ``engine='pd'`` routes through
        ``_solve_min_variance_pd``. When the base-class
        ``MultiObjectiveOptimizer._compute_corners`` calls it with ``x0=None`` on the
        very first corner solve, the helper's auto-default ``ones(M)`` must be clipped
        into ``[min_weight, max_weight]`` so the PD solver's bounds-feasibility
        prerequisite is satisfied. Without the clip, a Rake with ``max_weight < 1.0``
        would fail.

        """
        rng = np.random.default_rng(20260513)
        M, p = 80, 4
        X = rng.standard_normal((M, p))
        w_target = rng.uniform(0.1, 0.4, size=M)
        mu = (1.0 / M) * X.T @ w_target
        rake = Rake(
            distance=SquaredL2(
                dimension=M
            ),  # SquaredL2 has no v -> default ones(M) needs clipping
            X=X,
            mu=mu,
            min_weight=0.0,
            max_weight=0.5,  # < 1.0 so unclipped ones(M) would be bounds-infeasible
            constrain_mean_weight_to=None,
            engine="pd",
        )
        frontier = EfficientFrontier(rake)
        # Call corner-1 directly with x0=None: the helper must clip the
        # default before passing to PD.  Without the clip, this raises in
        # PD's bounds-feasibility check.
        res = frontier.minimize_objective(objective_index=1, x0=None)
        self.assertTrue(np.all(res.solution <= 0.5 + 1e-6))
        self.assertTrue(np.all(res.solution >= -1e-6))

    def test_ipm_default_x0_clamped_to_max_weight(self) -> None:
        """Corner-1 IPM path also uses the clamped default ``x0=ones``.

        The default warm start now feeds both engines; with ``engine='ipm'`` the
        variance helper hands the clamped ``ones(M)`` to ``BarrierSolver``, which
        projects it onto the manifold and Phase-I's it -- returning an in-bounds
        solution even though the bare ``ones(M)`` is outside the box (``max_weight <
        1.0``).

        """
        rng = np.random.default_rng(20260513)
        M, p = 80, 4
        X = rng.standard_normal((M, p))
        w_target = rng.uniform(0.1, 0.4, size=M)
        mu = (1.0 / M) * X.T @ w_target
        rake = Rake(
            distance=SquaredL2(dimension=M),
            X=X,
            mu=mu,
            min_weight=0.0,
            max_weight=0.5,
            constrain_mean_weight_to=None,
            engine="ipm",
        )
        frontier = EfficientFrontier(rake, engine="ipm")
        res = frontier.minimize_objective(objective_index=1, x0=None)
        self.assertTrue(np.all(res.solution <= 0.5 + 1e-6))
        self.assertTrue(np.all(res.solution >= -1e-6))

    def test_defaults_to_pd_engine(self) -> None:
        """``EfficientFrontier(rake)`` overrides ``rake.engine`` to PD by default.

        PD warm-starts naturally across phi sweep iterations; IPM cannot (its barrier
        Phase I rejects boundary-feasible x0 from the previous iteration). Re-benchmark
        2026-05-21 measured ~2.3x speedup over the pre-change inherit-IPM behavior.

        """
        rake = _make_rake_for_default_engine_test(engine="ipm")
        self.assertEqual(rake.engine, "ipm")
        EfficientFrontier(rake)
        self.assertEqual(rake.engine, "pd", "EfficientFrontier should default to PD")

    def test_explicit_engine_overrides_default(self) -> None:
        """Passing ``engine='ipm'`` to ``EfficientFrontier`` opts out of PD default."""
        rake = _make_rake_for_default_engine_test(engine="pd")
        self.assertEqual(rake.engine, "pd")
        EfficientFrontier(rake, engine="ipm")
        self.assertEqual(rake.engine, "ipm")

    def test_engine_none_preserves_rake_engine(self) -> None:
        """Passing ``engine=None`` preserves whatever engine the Rake was constructed with."""
        rake = _make_rake_for_default_engine_test(engine="ipm")
        EfficientFrontier(rake, engine=None)
        self.assertEqual(rake.engine, "ipm")


@pytest.mark.slow
class ExternalValidityVarianceFrontierTest(unittest.TestCase):
    """Tests for the joint external-validity/variance frontier."""

    M1: int = 60
    M2: int = 60
    Q: int = 3

    @classmethod
    def _frontier(cls, psi3: float | None = None) -> ExternalValidityVarianceFrontier:
        """A joint problem whose respondents skew away from the population."""
        rng = np.random.default_rng(3607)
        Z1 = rng.standard_normal((cls.M1, cls.Q)) + 0.4
        Z2 = rng.standard_normal((cls.M2, cls.Q)) + 0.4
        return ExternalValidityVarianceFrontier(
            Z1=Z1,
            Z2=Z2,
            Z3=Z1,
            Z4=Z2,
            nu1=np.zeros(cls.Q),
            nu2=np.zeros(cls.Q),
            min_weight1=cls.M1 / 1e6,
            min_weight2=cls.M2 / 1e6,
            psi3=psi3,
        )

    def test_mismatched_rows_raise(self) -> None:
        """Z1/Z2 must agree with Z3/Z4 on the number of rows."""
        rng = np.random.default_rng(11)
        with self.assertRaisesRegex(ValueError, "same number of rows"):
            ExternalValidityVarianceFrontier(
                Z1=rng.standard_normal((self.M1 + 1, self.Q)),
                Z2=rng.standard_normal((self.M2, self.Q)),
                Z3=rng.standard_normal((self.M1, self.Q)),
                Z4=rng.standard_normal((self.M2, self.Q)),
                nu1=np.zeros(self.Q),
                nu2=np.zeros(self.Q),
                min_weight1=0.0,
                min_weight2=0.0,
            )

    def test_mismatched_columns_raise(self) -> None:
        """All four Z blocks share one covariate list, so column counts must agree."""
        rng = np.random.default_rng(12)
        Z1 = rng.standard_normal((self.M1, self.Q))
        with self.assertRaisesRegex(ValueError, "same number of columns"):
            ExternalValidityVarianceFrontier(
                Z1=Z1,
                Z2=rng.standard_normal((self.M2, self.Q)),
                Z3=Z1,
                Z4=rng.standard_normal((self.M2, self.Q + 1)),
                nu1=np.zeros(self.Q),
                nu2=np.zeros(self.Q),
                min_weight1=0.0,
                min_weight2=0.0,
            )

    def test_diagnose_survives_a_zero_band(self) -> None:
        """A zero band asks for exact balance; dividing by it must not produce nan."""
        frontier = self._frontier()
        uniform = np.ones(self.M1 + self.M2)

        df = frontier.diagnose(uniform, phi=0.0)
        external = df[df["kind"] == "external"]

        self.assertFalse(df["slack_frac"].isna().any())
        # Uniform weights do not balance this sample, so every zero-band row is
        # violated and sorts ahead of the internal rows, which have real room.
        self.assertTrue((external["slack_frac"] == -np.inf).all())
        self.assertEqual(df["kind"].iloc[0], "external")

    def test_diagnose_zero_band_exactly_met_is_pinned_not_violated(self) -> None:
        """A zero band that is actually met reads as pinned, not as maximally tight."""
        frontier = self._frontier()
        exact = ExternalValidityVarianceFrontier(
            Z1=frontier.Z1,
            Z2=frontier.Z2,
            Z3=frontier.Z3,
            Z4=frontier.Z4,
            # Computed exactly as ``diagnose`` computes the imbalance, so the band is
            # met to the bit; ``mean`` sums in a different order than the BLAS
            # matmul and leaves ~1e-16 on some platforms.
            nu1=frontier.Z1.T @ np.ones(self.M1) / self.M1,
            nu2=frontier.Z2.T @ np.ones(self.M2) / self.M2,
            min_weight1=frontier.min_weight1,
            min_weight2=frontier.min_weight2,
        )

        df = exact.diagnose(np.ones(self.M1 + self.M2), phi=0.0)
        external = df[df["kind"] == "external"]

        self.assertFalse(df["slack_frac"].isna().any())
        self.assertTrue((external["slack_frac"] == 0.0).all())

    def test_default_psi3_is_the_two_sample_value(self) -> None:
        """psi3 defaults to the same value JointCalibrator would choose."""
        self.assertAlmostEqual(
            self._frontier().psi3, 1.5 * np.sqrt(1 / self.M1 + 1 / self.M2)
        )
        self.assertEqual(self._frontier(psi3=0.25).psi3, 0.25)

    def test_variance_corner_is_near_uniform(self) -> None:
        """With external validity unconstrained, minimum variance is uniform."""
        frontier = self._frontier()
        res = frontier.minimize_objective(0)

        # mean(w^2) >= mean(w)^2 = 1, with equality only at uniform weights.
        self.assertAlmostEqual(
            float(frontier.evaluate_objectives(res.solution)[0]), 1.0, places=2
        )

    def test_min_variance_at_improves_a_dominated_corner(self) -> None:
        """The mechanism behind the corner fix, exercised directly.

        The LP minimizes phi and nothing else, so among the weightings that achieve
        ``phi_min`` it may return one with far more variance than necessary. Such a
        point is Pareto-dominated, and because it anchors the sweep, every interpolant
        draws a segment down to it -- which reads as a sharp artificial vertex. On one
        real survey that stray corner sat at an effective sample size of 165 where the
        frontier was at 432.

        A synthetic frontier need not have a degenerate LP optimum, so this feeds the
        mechanism a deliberately dominated point rather than hoping the fixture produces
        one: any mixture of two feasible weightings is feasible, and sits above the
        frontier at its own imbalance.

        """
        frontier = self._frontier()
        results = frontier.trace(num_points=6)
        by_imbalance = sorted(results.points, key=lambda p: float(p.objectives[1]))

        # A mixture of two frontier solutions: feasible, but off the frontier.
        mixed = 0.5 * (
            np.asarray(by_imbalance[1].solution) + np.asarray(by_imbalance[-1].solution)
        )
        dominated = replace(by_imbalance[1].ipm_result, solution=mixed)
        before = frontier.evaluate_objectives(mixed)

        improved = frontier._min_variance_at(dominated)
        after = frontier.evaluate_objectives(np.asarray(improved.solution))

        # Same imbalance to within the slack, strictly less variance.
        self.assertLessEqual(after[1], before[1] * (1 + 1e-3))
        self.assertLess(after[0], before[0])

    def test_imbalance_corner_drops_the_augmenting_phi(self) -> None:
        """Corner 1 solves over [w, phi] but must return weights only."""
        frontier = self._frontier()
        res = frontier.minimize_objective(1)

        self.assertEqual(len(np.asarray(res.solution)), self.M1 + self.M2)

    def test_imbalance_corner_beats_the_variance_corner_on_imbalance(self) -> None:
        """The two corners really are the ends of a tradeoff."""
        frontier = self._frontier()
        at_min_imbalance = frontier.evaluate_objectives(
            frontier.minimize_objective(1).solution
        )
        at_min_variance = frontier.evaluate_objectives(
            frontier.minimize_objective(0).solution
        )

        self.assertLess(at_min_imbalance[1], at_min_variance[1])
        self.assertGreater(at_min_imbalance[0], at_min_variance[0])

    def test_solve_with_bounds_respects_the_band(self) -> None:
        """A feasible phi yields weights inside the band it asked for."""
        frontier = self._frontier()
        phi = (
            0.9
            * frontier.evaluate_objectives(frontier.minimize_objective(0).solution)[1]
        )
        res = frontier.solve_with_bounds(np.array([phi]))

        self.assertLessEqual(
            float(frontier.evaluate_objectives(res.solution)[1]), phi + 1e-6
        )

    def test_diagnose_flags_the_binding_covariates(self) -> None:
        """At the imbalance corner, something is pinned to its band."""
        frontier = self._frontier()
        res = frontier.minimize_objective(1)
        df = frontier.diagnose(res.solution, covariate_names=["a", "b", "c"])

        self.assertEqual(len(df.index), 3 * self.Q)
        self.assertLess(float(df["slack_frac"].iloc[0]), 1e-6)
        # Sorted tightest first, and every group labelled on every row.
        self.assertTrue(df["slack_frac"].is_monotonic_increasing)
        self.assertEqual(set(df["kind"]), {"external", "internal"})

    def test_trace_returns_a_monotone_frontier(self) -> None:
        """Sweeping produces a genuine tradeoff curve."""
        results = self._frontier().trace(num_points=6)
        by_imbalance = sorted(results.points, key=lambda p: float(p.objectives[1]))
        variances = [float(p.objectives[0]) for p in by_imbalance]

        self.assertGreater(len(variances), 2)
        # Tighter imbalance costs more variance, monotonically.
        self.assertTrue(all(a >= b - 1e-6 for a, b in itertools.pairwise(variances)))

    def test_trace_returns_the_labelled_results_class(self) -> None:
        """trace() upgrades the bare FrontierResults to the ESS-aware subclass."""
        results = self._frontier().trace(num_points=6)

        self.assertIsInstance(results, ExternalValidityVarianceFrontierResults)
        self.assertEqual(len(results.effective_sample_sizes), len(results.imbalances))

    def test_effective_sample_size_matches_the_variance_objective(self) -> None:
        """ESS is M / mean(w^2), the same quantity the optimizer minimizes."""
        results = self._frontier().trace(num_points=6)
        M = self.M1 + self.M2
        by_imbalance = sorted(results.points, key=lambda p: float(p.objectives[1]))

        for point, ess in zip(
            by_imbalance, results.effective_sample_sizes, strict=True
        ):
            self.assertAlmostEqual(ess, M / float(point.objectives[0]))
            self.assertLessEqual(ess, M + 1e-6)

    def test_tighter_balance_costs_effective_sample_size(self) -> None:
        """Sorted by imbalance ascending, ESS increases."""
        results = self._frontier().trace(num_points=6)
        ess = results.effective_sample_sizes

        self.assertTrue(all(a <= b + 1e-6 for a, b in itertools.pairwise(ess)))

    def test_the_refined_knee_does_not_depend_on_the_grid(self) -> None:
        """The sampled knee moves with num_points; the solved one does not.

        The sweep grids uniformly in phi and need not place a point near the bend, so
        the argmax over samples is quantized to wherever the grid happened to fall.
        Root-finding on the slope removes that dependence.

        """
        frontier = self._frontier()
        coarse = frontier.trace(num_points=6, refine_knee=False)
        fine = frontier.trace(num_points=24, refine_knee=False)

        coarse_knee = none_throws(frontier.refine_knee(coarse))
        fine_knee = none_throws(frontier.refine_knee(fine))

        np.testing.assert_allclose(
            coarse_knee.objectives, fine_knee.objectives, rtol=1e-3, atol=1e-6
        )

    def test_the_refined_knee_is_a_real_solve(self) -> None:
        """The last bisection iterate is itself a frontier point.

        So there is no interpolated weight vector to confirm afterwards: the knee comes
        with weights that were actually solved for.

        """
        frontier = self._frontier()
        knee = none_throws(
            frontier.refine_knee(frontier.trace(num_points=6, refine_knee=False))
        )
        w = np.asarray(knee.solution)

        self.assertEqual(len(w), self.M1 + self.M2)
        np.testing.assert_allclose(
            frontier.evaluate_objectives(w), knee.objectives, rtol=1e-9
        )

    def test_plot_is_labelled_in_ess_and_inverted(self) -> None:
        """The x-axis reads in respondents and increases leftward."""
        results = self._frontier().trace(num_points=6)
        ax = results.plot()

        self.assertEqual(ax.get_xlabel(), "Effective Sample Size")
        self.assertEqual(ax.get_ylabel(), "Covariate Imbalance with Target Population")
        left, right = ax.get_xlim()
        self.assertLess(left, right)

    def test_plot_draws_the_interpolant_not_the_polyline(self) -> None:
        """The certified curve, carried across the reciprocal into ESS.

        Straight segments between sparse samples invent a vertex a convex frontier does
        not have, so this plot has to consume ``envelope`` and not just replot its own
        points.

        """
        results = self._frontier().trace(num_points=6)
        ax = results.plot(annotate_knee=False)
        drawn = np.concatenate([line.get_xdata() for line in ax.lines])

        # Far denser than the handful of solved points: it is the interpolant.
        self.assertGreater(len(drawn), 4 * len(results.points))

    def test_plot_maps_the_band_across_the_reciprocal(self) -> None:
        """ESS = M / VIF is decreasing, so the two bounds swap roles."""
        results = self._frontier().trace(num_points=6)
        env = results.envelope()
        self.assertTrue(np.isfinite(env.max_band_width))

        m = float(len(results.points[0].solution))
        # chord bounds variance ABOVE, hence effective sample size BELOW.
        np.testing.assert_array_less(m / env.chord, m / env.tangent + 1e-9)

    def test_the_shaded_region_follows_the_drawn_curve(self) -> None:
        """Filling from the raw points would re-introduce the vertex.

        The light-blue region and the interpolant have to share a boundary; when they do
        not, the polyline edge shows through and reads as a kink in a curve that does
        not have one.

        """
        results = self._frontier().trace(num_points=6)
        ax = results.plot(annotate_knee=False)

        curve = ax.lines[0].get_xdata()
        # The fill's path should span the same abscissa as the drawn curve,
        # not the handful of solved points.
        # Two filled collections when the band is certified: the grey lens
        # first, then the light-blue region. The region is the one drawn last.
        fills = [c for c in ax.collections if c.get_paths()]
        vertices = np.asarray(fills[-1].get_paths()[0].vertices)
        fill_x = vertices[:, 0]

        self.assertGreater(len(fill_x), 4 * len(results.points))
        self.assertAlmostEqual(float(np.max(fill_x)), float(np.max(curve)), delta=1e-6)

    def test_plot_in_vif_space_is_labelled_for_the_design_effect(self) -> None:
        """variance_space="vif" names the optimizer's own objective."""
        results = self._frontier().trace(num_points=6)
        ax = results.plot(variance_space="vif")

        self.assertEqual(ax.get_xlabel(), "Variance Inflation Factor")
        self.assertEqual(
            ax.get_title(), "External Validity vs Variance Inflation Factor"
        )
        self.assertEqual(ax.get_ylabel(), "Covariate Imbalance with Target Population")

    def test_ess_remains_the_default_variance_space(self) -> None:
        """The label and title both still read in respondents when unasked."""
        results = self._frontier().trace(num_points=6)
        ax = results.plot()

        self.assertEqual(ax.get_xlabel(), "Effective Sample Size")
        self.assertEqual(ax.get_title(), "External Validity vs Effective Sample Size")

    def test_an_explicit_label_still_wins_over_the_variance_space(self) -> None:
        """The space picks the default wording; the caller can override it."""
        results = self._frontier().trace(num_points=6)
        ax = results.plot(variance_space="vif", x_label="Design Effect", title="Q1")

        self.assertEqual(ax.get_xlabel(), "Design Effect")
        self.assertEqual(ax.get_title(), "Q1")

    def test_the_two_variance_spaces_are_reciprocal_mirrors(self) -> None:
        """Same curve, same interpolant: only the abscissa is transformed.

        The frontier is drawn from ``envelope()``, which lives in VIF, so the ESS panel
        has to be that same curve carried across ``ESS = M / VIF`` -- not a separate
        interpolation through transformed samples.

        """
        results = self._frontier().trace(num_points=6)
        m = float(len(results.points[0].solution))

        vif = results.plot(annotate_knee=False, variance_space="vif").lines[0]
        ess = results.plot(annotate_knee=False, variance_space="ess").lines[0]

        np.testing.assert_allclose(
            np.asarray(ess.get_xdata(), dtype=np.float64),
            m / np.asarray(vif.get_xdata(), dtype=np.float64),
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            np.asarray(ess.get_ydata(), dtype=np.float64),
            np.asarray(vif.get_ydata(), dtype=np.float64),
            rtol=1e-9,
        )

    def test_the_vif_panel_runs_the_other_way_across_the_page(self) -> None:
        """Buying balance costs variance, so VIF falls where ESS rises."""
        results = self._frontier().trace(num_points=6)

        spaces: tuple[tuple[Literal["vif", "ess"], bool], ...] = (
            ("vif", False),
            ("ess", True),
        )
        for space, tighter_on_the_right in spaces:
            with self.subTest(variance_space=space):
                line = results.plot(annotate_knee=False, variance_space=space).lines[0]
                x = np.asarray(line.get_xdata(), dtype=np.float64)
                y = np.asarray(line.get_ydata(), dtype=np.float64)
                left_to_right = y[np.argsort(x)]
                if tighter_on_the_right:
                    self.assertLess(left_to_right[0], left_to_right[-1])
                else:
                    self.assertGreater(left_to_right[0], left_to_right[-1])

    def test_plot_rejects_a_variance_space_it_does_not_know(self) -> None:
        """A typo should say so rather than silently fall through to ESS."""
        results = self._frontier().trace(num_points=6)

        with self.assertRaisesRegex(ValueError, "variance_space"):
            # why: deliberately outside the Literal; the test asserts rejection.
            results.plot(variance_space=cast(Any, "variance"))


class MinimumImbalanceCornerTest(unittest.TestCase):
    """The minimum-imbalance corner has to land on the frontier it anchors.

    The corner comes from a linear program that minimizes imbalance and nothing else, so
    it can sit at any vertex achieving that minimum -- including one at several times
    the necessary variance. Since both ``knee()`` and ``refine_knee`` measure distance
    from the chord through the corners, a stray corner moves the band that gets applied.

    """

    @staticmethod
    def _frontier() -> ExternalValidityVarianceFrontier:
        return ExternalValidityVarianceFrontierTest._frontier()

    def test_the_slack_is_absolute_not_a_fraction_of_the_minimum(self) -> None:
        """A near-balanced calibration must not get a near-zero slack.

        Scaling the slack by ``phi_min`` shrinks it to nothing exactly when the frontier
        is easiest to balance, which is when the re-solve is handed a feasible set with
        no interior and quietly gives up.

        """
        frontier = self._frontier()
        corner = frontier.minimize_objective(1)
        phi = float(frontier.evaluate_objectives(np.asarray(corner.solution))[1])

        bands: list[float] = []
        real = frontier.solve_with_bounds

        def record(
            bounds: npt.NDArray[np.float64],
            x0: npt.NDArray[np.float64] | None = None,
        ) -> SolveResult:
            bands.append(float(bounds[0]))
            return real(bounds, x0)

        with patch.object(frontier, "solve_with_bounds", side_effect=record):
            frontier._min_variance_at(corner)

        scale = frontier._imbalance_scale()
        # The band re-solved at, less the minimum it re-solved from, is the
        # slack that actually got used. A multiplicative 1e-6 would make this
        # ``phi * 1e-6``, which is nothing when phi is small.
        self.assertGreaterEqual(bands[0] - phi, 1e-4 * scale * 0.99)

    def test_the_corner_is_the_least_variance_point_at_its_own_band(self) -> None:
        """The corner must not be beaten by a point at the same imbalance.

        This is the invariant the stray vertex violates: on one real survey the corner
        sat at a variance inflation factor of 27.9 where the frontier at the same band
        was 6.5.

        """
        frontier = self._frontier()
        corner = frontier.minimize_objective(1)
        variance, phi = frontier.evaluate_objectives(np.asarray(corner.solution))

        honest = frontier.solve_with_bounds(np.array([phi]))
        best = float(frontier.evaluate_objectives(np.asarray(honest.solution))[0])

        # A little room for the two solves disagreeing in the last digits;
        # the failure this guards is a multiple, not a rounding difference.
        self.assertLessEqual(float(variance), best * 1.05)

    def test_the_corner_still_sits_at_the_tight_end(self) -> None:
        """Slack buys a corner on the curve; it must not buy a different corner."""
        frontier = self._frontier()
        results = frontier.trace(num_points=6)
        corner = frontier.minimize_objective(1)

        phi = float(frontier.evaluate_objectives(np.asarray(corner.solution))[1])
        scale = frontier._imbalance_scale()

        self.assertLessEqual(phi, min(results.imbalances) + 1e-2 * scale)

    def test_a_corner_that_cannot_be_resolved_is_kept_but_said_out_loud(self) -> None:
        """Silently anchoring the chord to a stray is the failure mode itself."""
        frontier = self._frontier()
        with (
            patch.object(
                frontier, "solve_with_bounds", side_effect=ProblemInfeasibleError("no")
            ),
            self.assertLogs("pyrake.calibration.frontier", level="WARNING") as logs,
        ):
            corner = frontier.minimize_objective(1)

        self.assertEqual(len(np.asarray(corner.solution)), frontier.M1 + frontier.M2)
        self.assertTrue(any("anchors the chord" in line for line in logs.output))

    def test_a_primal_dual_breakdown_falls_back_to_the_barrier(self) -> None:
        """A Newton step the primal-dual method cannot take must not end the trace."""
        frontier = self._frontier()
        expected = frontier.minimize_objective(1)
        frontier = self._frontier()
        broken = MagicMock()
        broken.return_value.solve.side_effect = NewtonStepError(
            "H is not positive definite"
        )
        with (
            patch("pyrake.calibration.frontier.PrimalDualSolver", broken),
            self.assertLogs("pyrake.calibration.frontier", level="INFO") as logs,
        ):
            corner = frontier.minimize_objective(1)

        self.assertTrue(broken.return_value.solve.called)
        self.assertTrue(any("barrier method" in line for line in logs.output))
        np.testing.assert_allclose(
            frontier.evaluate_objectives(np.asarray(corner.solution)),
            frontier.evaluate_objectives(np.asarray(expected.solution)),
            rtol=1e-2,
            atol=1e-3,
        )

    def test_a_breakdown_in_both_solvers_says_what_failed(self) -> None:
        """If the barrier fails too, the error names the program and both causes."""
        frontier = self._frontier()
        broken = MagicMock()
        broken.return_value.solve.side_effect = NewtonStepError(
            "H is not positive definite"
        )
        also_broken = MagicMock()
        also_broken.return_value.solve.side_effect = NewtonStepError(
            "Hessian is not strictly positive definite."
        )
        with (
            patch("pyrake.calibration.frontier.PrimalDualSolver", broken),
            patch("pyrake.calibration.frontier.BarrierSolver", also_broken),
            self.assertRaises(OptimizationError) as raised,
        ):
            frontier.minimize_objective(1, x0=np.ones(frontier.M1 + frontier.M2))

        message = str(raised.exception)
        self.assertIn("smallest external imbalance", message)
        self.assertIn("H is not positive definite", message)
        self.assertIn("Hessian is not strictly positive definite", message)
        self.assertIn("always feasible", message)

    def test_a_survey_whose_corner_broke_the_primal_dual_method(self) -> None:
        """A small simulated A/B-test survey on which the primal-dual corner raised.

        The responses depend on age, urbanicity and, in the treated arm only, heavy use
        of the platform, so each arm is unbalanced and differently so. Whether the
        primal-dual solve breaks down here depends on the BLAS's rounding; the trace
        must succeed either way.

        """
        rng = np.random.default_rng(1)
        N, n = 20_000, 300
        age = rng.uniform(18, 80, N)
        urban = (rng.random(N) < 0.6).astype(float)
        heavy = (rng.random(N) < expit(1.0 - 0.06 * (age - 30))).astype(float)
        female = (rng.random(N) < 0.5).astype(float)
        treated = rng.random(N) < 0.5
        X = np.column_stack([age, urban, heavy, female])
        mu, sigma = X.mean(axis=0), X.std(axis=0)

        def survey(arm: npt.NDArray[np.bool_], b: float) -> npt.NDArray[np.int64]:
            idx = rng.choice(np.flatnonzero(arm), n, replace=False)
            eta = -0.3 - 0.04 * (age[idx] - 45) + 0.5 * urban[idx] + b * heavy[idx]
            return idx[rng.random(n) < expit(eta)]

        control, treatment = survey(~treated, 0.0), survey(treated, 1.0)
        Z1, Z2 = X[control] / sigma, X[treatment] / sigma
        frontier = ExternalValidityVarianceFrontier(
            Z1=Z1,
            Z2=Z2,
            Z3=Z1,
            Z4=Z2,
            nu1=mu / sigma,
            nu2=mu / sigma,
            min_weight1=0.0,
            min_weight2=0.0,
        )
        results = frontier.trace(num_points=6, refine_knee=False)

        self.assertLess(min(results.imbalances), 1e-2 * max(results.imbalances))


class FromCalibratorTest(unittest.TestCase):
    """The frontier can be built from a configured joint calibration."""

    M1: int = 40
    M2: int = 50
    Q: int = 3

    @classmethod
    def _calibrator(cls, with_external: bool = True) -> JointCalibrator:
        """A joint calibration, optionally without external-validity blocks."""
        rng = np.random.default_rng(6017)
        Z1 = rng.standard_normal((cls.M1, cls.Q)) + 0.3
        Z2 = rng.standard_normal((cls.M2, cls.Q)) + 0.3
        external: dict[str, Any] = (
            {"Z1": Z1, "Z2": Z2, "nu1": np.zeros(cls.Q), "nu2": np.zeros(cls.Q)}
            if with_external
            else {}
        )
        return JointCalibrator(
            distance=SquaredL2(v=np.zeros(cls.M1 + cls.M2)),
            Z3=Z1,
            Z4=Z2,
            min_weight1=cls.M1 / 1e6,
            min_weight2=cls.M2 / 1e6,
            **external,
        )

    def test_frontier_round_trips_the_blocks(self) -> None:
        """The frontier sees the same covariates the calibrator was given."""
        rake = self._calibrator()

        frontier = ExternalValidityVarianceFrontier.from_calibrator(rake)

        np.testing.assert_allclose(frontier.Z1, none_throws(rake._Z1))
        np.testing.assert_allclose(frontier.Z2, none_throws(rake._Z2))
        np.testing.assert_allclose(frontier.Z3, none_throws(rake._Z3))
        np.testing.assert_allclose(frontier.Z4, none_throws(rake._Z4))
        self.assertEqual(frontier.M1, self.M1)
        self.assertEqual(frontier.M2, self.M2)

    def test_frontier_splits_min_weight_at_the_block_boundary(self) -> None:
        """Per-group floors survive the round trip through the packed vector."""
        frontier = ExternalValidityVarianceFrontier.from_calibrator(self._calibrator())

        np.testing.assert_allclose(
            np.broadcast_to(frontier.min_weight1, self.M1), self.M1 / 1e6
        )
        np.testing.assert_allclose(
            np.broadcast_to(frontier.min_weight2, self.M2), self.M2 / 1e6
        )

    def test_psi3_defaults_to_the_calibrators_own_band(self) -> None:
        """A cascade holds internal validity where the calibrator put it."""
        rake = self._calibrator()

        frontier = ExternalValidityVarianceFrontier.from_calibrator(rake)

        q_ext = (rake._q1 or 0) + (rake._q2 or 0)
        np.testing.assert_allclose(
            np.broadcast_to(frontier.psi3, self.Q),
            np.asarray(rake.psi)[q_ext:],
        )

    def test_psi3_carries_a_heterogeneous_internal_band(self) -> None:
        """Per-row internal bands survive rather than collapsing to the first."""
        rake = self._calibrator()
        band = np.asarray(rake.psi, dtype=float).copy()
        q_ext = (rake._q1 or 0) + (rake._q2 or 0)
        band[q_ext:] = [0.2, 0.3, 0.4]
        rake.update_psi(band)

        frontier = ExternalValidityVarianceFrontier.from_calibrator(rake)

        np.testing.assert_allclose(frontier.psi3, [0.2, 0.3, 0.4])

    def test_psi3_can_be_overridden(self) -> None:
        """An explicit band wins over the calibrator's."""
        frontier = ExternalValidityVarianceFrontier.from_calibrator(
            self._calibrator(), psi3=0.42
        )

        self.assertEqual(frontier.psi3, 0.42)

    def test_without_external_validity_blocks_it_raises(self) -> None:
        """No external blocks means there is no tradeoff to trace."""
        with self.assertRaisesRegex(ValueError, "external"):
            ExternalValidityVarianceFrontier.from_calibrator(
                self._calibrator(with_external=False)
            )

    def test_engine_defaults_to_pd_and_can_be_overridden(self) -> None:
        """The sweep's engine is selectable, like the sibling frontiers'."""
        rake = self._calibrator()

        self.assertEqual(
            ExternalValidityVarianceFrontier.from_calibrator(rake).engine, "pd"
        )
        self.assertEqual(
            ExternalValidityVarianceFrontier.from_calibrator(rake, engine="ipm").engine,
            "ipm",
        )

    def test_engine_none_inherits_the_calibrators(self) -> None:
        """``None`` defers to however the calibration was configured."""
        rake = self._calibrator()
        rake.engine = "ipm"

        frontier = ExternalValidityVarianceFrontier.from_calibrator(rake, engine=None)

        self.assertEqual(frontier.engine, "ipm")

    @pytest.mark.slow
    def test_both_engines_trace_the_same_frontier(self) -> None:
        """Engine is a performance choice, not a modelling one."""
        rake = self._calibrator()
        pd_imb = (
            ExternalValidityVarianceFrontier.from_calibrator(rake, engine="pd")
            .trace(num_points=5)
            .imbalances
        )
        ipm_imb = (
            ExternalValidityVarianceFrontier.from_calibrator(rake, engine="ipm")
            .trace(num_points=5)
            .imbalances
        )

        np.testing.assert_allclose(pd_imb, ipm_imb, rtol=1e-3, atol=1e-6)


class TangentSlopeTest(unittest.TestCase):
    """The frontier reports its own derivative, from the imbalance duals."""

    M1: int = 50
    M2: int = 50
    Q: int = 3

    @classmethod
    def _frontier(cls) -> ExternalValidityVarianceFrontier:
        rng = np.random.default_rng(4242)
        Z1 = rng.standard_normal((cls.M1, cls.Q)) + 0.6
        Z2 = rng.standard_normal((cls.M2, cls.Q)) + 0.6
        return ExternalValidityVarianceFrontier(
            Z1=Z1,
            Z2=Z2,
            Z3=Z1,
            Z4=Z2,
            nu1=np.zeros(cls.Q),
            nu2=np.zeros(cls.Q),
            min_weight1=cls.M1 / 1e6,
            min_weight2=cls.M2 / 1e6,
        )

    def test_slopes_are_negative_where_known(self) -> None:
        """Relaxing the band cannot increase the minimum variance."""
        results = self._frontier().trace(num_points=8)
        slopes = none_throws(results.tangent_slopes)
        known = slopes[np.isfinite(slopes)]

        self.assertGreater(len(known), 0)
        self.assertTrue(np.all(known <= 1e-9))

    def test_the_multiplier_layout_is_read_off_once(self) -> None:
        """``refine_knee`` asks for a slope per bisection step; each one used to

        rebuild a calibrator and reassemble its problem to rediscover a layout that
        cannot change. That is pure overhead sitting next to the solve the probe exists
        to do, so the layout is memoized -- and a memo nobody checks is a memo the next
        refactor drops.

        """
        frontier = self._frontier()

        with patch.object(
            frontier, "_bounded_calibrator", wraps=frontier._bounded_calibrator
        ) as built:
            frontier._imbalance_multiplier_span()
            first = built.call_count
            for _ in range(4):
                frontier._imbalance_multiplier_span()

        self.assertEqual(1, first)
        self.assertEqual(1, built.call_count)

    def test_corners_report_no_slope(self) -> None:
        """Their problems have a different layer layout entirely."""
        results = self._frontier().trace(num_points=8)
        slopes = none_throws(results.tangent_slopes)

        for i, point in enumerate(results.points):
            if not np.all(np.isfinite(point.bounds)):
                self.assertTrue(np.isnan(slopes[i]))

    def test_slopes_match_a_finite_difference(self) -> None:
        """The dual-derived derivative agrees with resolving either side."""
        frontier = self._frontier()
        results = frontier.trace(num_points=8)
        slopes = none_throws(results.tangent_slopes)

        interior = [
            i
            for i, p in enumerate(results.points)
            if np.all(np.isfinite(p.bounds)) and np.isfinite(slopes[i])
        ]
        self.assertGreater(len(interior), 0)

        i = interior[len(interior) // 2]
        phi = float(results.points[i].objectives[1])
        h = 0.02 * phi
        lo = frontier.solve_with_bounds(np.array([phi - h]))
        hi = frontier.solve_with_bounds(np.array([phi + h]))
        secant = (
            frontier.evaluate_objectives(np.asarray(hi.solution))[0]
            - frontier.evaluate_objectives(np.asarray(lo.solution))[0]
        ) / (2 * h)

        self.assertAlmostEqual(slopes[i], secant, delta=0.25 * abs(secant) + 1e-6)

    def test_slopes_unlock_the_tangent_bound(self) -> None:
        """With slopes the envelope is certified, not merely smoothed."""
        results = self._frontier().trace(num_points=8)

        env = results.envelope(num=60)

        self.assertTrue(np.isfinite(env.max_band_width))
        self.assertTrue(np.all(env.tangent <= env.chord + 1e-9))


class StandardErrorVarianceSpaceTest(unittest.TestCase):
    """The frontier can also be drawn against the standard-error multiplier."""

    @staticmethod
    def _results() -> ExternalValidityVarianceFrontierResults:
        return ExternalValidityVarianceFrontierTest._frontier().trace(num_points=6)

    def test_the_se_panel_is_labelled_for_the_standard_error(self) -> None:
        """Kish's DEFT, named in words a reader can act on."""
        ax = self._results().plot(variance_space="se")

        self.assertEqual(ax.get_xlabel(), "Standard Error Inflation Factor")
        self.assertEqual(
            ax.get_title(), "External Validity vs Standard Error Inflation Factor"
        )

    def test_the_se_curve_is_the_root_of_the_vif_curve(self) -> None:
        """One interpolant carried across the transform, not a second fit."""
        results = self._results()
        vif = results.plot(annotate_knee=False, variance_space="vif").lines[0]
        se = results.plot(annotate_knee=False, variance_space="se").lines[0]

        np.testing.assert_allclose(
            np.asarray(se.get_xdata(), dtype=np.float64),
            np.sqrt(np.asarray(vif.get_xdata(), dtype=np.float64)),
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            np.asarray(se.get_ydata(), dtype=np.float64),
            np.asarray(vif.get_ydata(), dtype=np.float64),
            rtol=1e-9,
        )

    def test_the_se_panel_runs_the_same_way_as_the_vif_panel(self) -> None:
        """The square root is increasing, so it does not mirror the page."""
        results = self._results()

        for space in ("vif", "se"):
            with self.subTest(variance_space=space):
                line = results.plot(annotate_knee=False, variance_space=space).lines[0]
                x = np.asarray(line.get_xdata(), dtype=np.float64)
                y = np.asarray(line.get_ydata(), dtype=np.float64)
                left_to_right = y[np.argsort(x)]
                self.assertGreater(left_to_right[0], left_to_right[-1])

    def test_the_certified_band_is_transformed_rather_than_refitted(self) -> None:
        r"""The lens is the VIF lens under the same square root as the curve.

        Note what is NOT asserted: which of the two bounds is the lower one.
        ``fill_betweenx(y, a, b)`` draws the same polygon as ``fill_betweenx(y, b, a)``,
        so the tangent/chord ordering is a naming question rather than a rendering one,
        and no assertion can see it. What is observable, and what would break if the
        band were re-derived in SE coordinates instead of carried across, is the edges
        themselves.

        """
        results = self._results()
        edges = []
        for space in ("vif", "se"):
            ax = results.plot(annotate_knee=False, variance_space=space)
            lens = next(c for c in ax.collections if c.get_paths())
            edges.append(np.sort(np.asarray(lens.get_paths()[0].vertices)[:, 0]))

        np.testing.assert_allclose(edges[1], np.sqrt(edges[0]), rtol=1e-9)

    def test_the_knee_crosshair_lands_at_the_root_of_its_vif(self) -> None:
        """The marker's abscissa is transformed too, not just the curve.

        The curve comes from ``_draw_frontier`` but the crosshair comes from
        ``_variance_coordinate``, so a space handled in one and not the other draws a
        correct curve with the marker parked somewhere else entirely.

        """
        results = self._results()
        knee_vif = float(
            results.knee(_cvxium_knee_space(results.knee_space)).objectives[0]
        )
        ax = results.plot(variance_space="se")

        # The vertical crosshair is an ``ax.vlines`` LineCollection: two
        # points sharing an abscissa.
        marked = [
            float(seg[0][0])
            for c in ax.collections
            for seg in getattr(c, "get_segments", list)()
            if len(seg) and seg[0][0] == seg[1][0]
        ]
        self.assertTrue(marked, "no vertical crosshair was drawn")
        self.assertAlmostEqual(marked[0], math.sqrt(knee_vif), places=9)

    def test_plot_rejects_a_variance_space_it_does_not_know(self) -> None:
        """The third option must not have widened the door to typos."""
        with self.assertRaisesRegex(ValueError, "variance_space"):
            # why: deliberately outside the Literal; the test asserts rejection.
            self._results().plot(variance_space=cast(Any, "stderr"))


class KneeSpaceIsRememberedTest(unittest.TestCase):
    r"""A frontier carries the coordinate its band was chosen in.

    Two places consult it -- the band selection in ``knee_external_scale`` and the
    annotation in ``plot`` -- and a disagreement between them is invisible: the plot
    simply marks a band nobody used, which looks exactly like a correct plot. Same
    failure mode as ``tangent_slopes`` before it became a constructor argument.

    """

    @staticmethod
    def _frontier() -> ExternalValidityVarianceFrontier:
        return ExternalValidityVarianceFrontierTest._frontier()

    def test_se_is_the_default(self) -> None:
        """The recommended rule is the default, not a footnote."""
        self.assertEqual(self._frontier().trace(num_points=4).knee_space, "se")

    def test_trace_records_what_it_was_asked_for(self) -> None:
        """Verify ``trace`` records the requested knee space."""
        results = self._frontier().trace(num_points=4, knee_space="vif")
        self.assertEqual(results.knee_space, "vif")

    def test_plot_annotates_the_space_the_frontier_remembers(self) -> None:
        """Without this the marker silently defaults away from the band.

        Both directions, because either alone is unfalsifiable: a ``plot`` that ignored
        the frontier and hard-coded one value would still pass the case that happens to
        match it.

        """
        knee_spaces: tuple[tuple[Literal["vif", "se"], str], ...] = (
            ("vif", "primary"),
            ("se", "root"),
        )
        for traced, expected in knee_spaces:
            with self.subTest(knee_space=traced):
                results = self._frontier().trace(
                    num_points=6, knee_space=traced, refine_knee=False
                )
                with patch.object(type(results), "knee", wraps=results.knee) as knee:
                    results.plot()
                self.assertEqual(knee.call_args.args[0], expected)

    def test_an_explicit_knee_space_still_wins(self) -> None:
        """Overriding is how the two are compared on one frontier."""
        results = self._frontier().trace(
            num_points=6, knee_space="vif", refine_knee=False
        )

        with patch.object(type(results), "knee", wraps=results.knee) as knee:
            results.plot(knee_space="se")

        self.assertEqual(knee.call_args.args[0], "root")

    def test_an_unknown_knee_space_is_rejected(self) -> None:
        """A typo must not quietly fall through to the VIF knee."""
        with self.assertRaisesRegex(ValueError, "knee_space"):
            # why: deliberately outside the Literal; the test asserts rejection.
            _cvxium_knee_space(cast(Any, "standard_error"))


class PlotMarksTheRefinedKneeTest(unittest.TestCase):
    r"""The crosshair goes on the band the calibration used.

    ``refine_knee`` solves for a knee that lies between two sweep samples;
    ``FrontierResults.knee`` returns the nearest sample to it. Marking the latter draws
    a crosshair at a band nobody applied, off by up to half a grid step, and the figure
    gives no hint that the two are different numbers.

    """

    @staticmethod
    def _frontier() -> ExternalValidityVarianceFrontier:
        return ExternalValidityVarianceFrontierTest._frontier()

    @staticmethod
    def _crosshair(ax: plt.Axes) -> list[float]:
        """Abscissae of the vertical crosshair, an ``ax.vlines`` collection."""
        return [
            float(seg[0][0])
            for c in ax.collections
            for seg in getattr(c, "get_segments", list)()
            if len(seg) and seg[0][0] == seg[1][0]
        ]

    def _refined(
        self,
    ) -> tuple[ExternalValidityVarianceFrontierResults, FrontierPoint]:
        frontier = self._frontier()
        results = frontier.trace(num_points=6, refine_knee=False)
        knee = frontier.refine_knee(
            results, knee_space=_cvxium_knee_space(results.knee_space)
        )
        assert knee is not None
        return results, knee

    @staticmethod
    def _grid_only_knee(
        results: ExternalValidityVarianceFrontierResults,
        grid: list[FrontierPoint],
    ) -> FrontierPoint:
        """The knee a plot would mark if it ignored the refinement entirely.

        Refinement folds its probes into ``results.points``, so by the time ``plot``
        runs, the sampled argmax is chosen from a set that contains the refined knee and
        generally returns it. The contrast that makes "the crosshair is on the refined
        knee" falsifiable is therefore against the argmax over the *pre-refinement*
        samples -- which is what the old code marked -- and that needs the grid captured
        beforehand.

        """
        return ExternalValidityVarianceFrontierResults(
            points=grid,
            corners=results.corners,
            primary_objective=results.primary_objective,
            n_attempted=len(grid),
            n_skipped=0,
            knee_space=results.knee_space,
        ).knee(_cvxium_knee_space(results.knee_space))

    def test_the_crosshair_sits_on_the_refined_knee(self) -> None:
        """And demonstrably not on the band the grid alone would have picked.

        The second assertion is what makes the first falsifiable: on a fixture where the
        two coincided, a plot that ignored the refinement entirely would pass.

        """
        frontier = self._frontier()
        results = frontier.trace(num_points=6, refine_knee=False)
        grid = list(results.points)
        knee = frontier.refine_knee(
            results, knee_space=_cvxium_knee_space(results.knee_space)
        )
        assert knee is not None

        refined_x = results._variance_coordinate(knee, "ess")
        sampled_x = results._variance_coordinate(
            self._grid_only_knee(results, grid), "ess"
        )
        self.assertNotAlmostEqual(
            refined_x,
            sampled_x,
            places=6,
            msg="refined and grid-only knees coincide on this fixture",
        )

        marked = self._crosshair(results.plot())

        self.assertTrue(marked, "no vertical crosshair was drawn")
        self.assertAlmostEqual(marked[0], refined_x, places=9)

    def test_the_crosshair_follows_the_variance_space(self) -> None:
        """The refined knee is transformed like everything else on the axis."""
        results, knee = self._refined()

        for space in ("vif", "se", "ess"):
            with self.subTest(variance_space=space):
                marked = self._crosshair(results.plot(variance_space=space))
                self.assertAlmostEqual(
                    marked[0],
                    results._variance_coordinate(knee, space),
                    places=9,
                )

    def test_an_off_grid_knee_still_gets_its_tangent(self) -> None:
        """The secant is bracketed, not looked up.

        The previous rule found the knee's index among the drawn points and gave up when
        there was none. A refined knee never has one, so that rule would silently drop
        the tangent from every refined plot.

        """
        results, _ = self._refined()
        ax = results.plot()

        self.assertTrue(
            any(line.get_color() == "orange" for line in ax.lines),
            "no tangent was drawn at the refined knee",
        )

    @staticmethod
    def _tangent_slope(ax: plt.Axes) -> float:
        """Slope of the orange tangent, read off the line actually drawn."""
        (line,) = [ln for ln in ax.lines if ln.get_color() == "orange"]
        x = np.asarray(line.get_xdata(), dtype=np.float64)
        y = np.asarray(line.get_ydata(), dtype=np.float64)
        return float((y[-1] - y[0]) / (x[-1] - x[0]))

    def test_the_tangent_is_the_refined_knees_own_slope(self) -> None:
        r"""Drawn from the knee's duals, restated in each variance space.

        With ``s = d(VIF)/d(phi)`` at the knee, the plotted d(phi)/dx is ``1/s`` against
        VIF, ``2 sqrt(VIF)/s`` against SE, and ``-VIF^2/(M s)`` against ESS. A secant
        between the neighbouring samples differs from these by its truncation error,
        which the tight tolerance here does not forgive.

        """
        results, knee = self._refined()
        slopes = none_throws(results.tangent_slopes)
        (s,) = [
            float(slope)
            for point, slope in zip(results.points, slopes, strict=True)
            if np.allclose(point.objectives, knee.objectives, atol=1e-8, rtol=0)
        ]
        self.assertTrue(math.isfinite(s) and s < 0.0)
        vif = float(knee.objectives[0])
        M = len(knee.solution)
        expected: dict[Literal["vif", "se", "ess"], float] = {
            "vif": 1.0 / s,
            "se": 2.0 * math.sqrt(vif) / s,
            "ess": -vif * vif / (M * s),
        }

        for space, slope in expected.items():
            with self.subTest(variance_space=space):
                ax = results.plot(variance_space=space)
                self.assertAlmostEqual(self._tangent_slope(ax) / slope, 1.0, places=9)

    def test_a_knee_without_duals_falls_back_to_the_secant(self) -> None:
        """No slope at the knee still draws a tangent, from the neighbours."""
        results, _ = self._refined()
        results.tangent_slopes = None

        slope = self._tangent_slope(results.plot())

        self.assertTrue(math.isfinite(slope) and slope > 0.0)

    def test_a_declined_refinement_draws_no_crosshair(self) -> None:
        """A straight frontier has no knee, and the plot must not invent one.

        Falling back to the sampled argmax here would mark the very number
        ``refine_knee`` refused to report -- one fixed by differences near the noise
        floor. The calibration falls back to the conventional band, which is not a point
        on this curve, so there is nothing honest to draw.

        """
        frontier = self._frontier()
        results = frontier.trace(num_points=6, refine_knee=False)
        results.tangent_slopes = np.full(len(results.points), -1.0)

        self.assertIsNone(frontier.refine_knee(results))
        self.assertEqual(self._crosshair(results.plot()), [])

    def test_an_unrefined_frontier_still_marks_the_sampled_knee(self) -> None:
        """Plotting a bare trace is unchanged; there is no better answer."""
        results = self._frontier().trace(num_points=6, refine_knee=False)
        expected = results._variance_coordinate(
            results.knee(_cvxium_knee_space(results.knee_space)), "ess"
        )

        marked = self._crosshair(results.plot())

        self.assertTrue(marked, "no vertical crosshair was drawn")
        self.assertAlmostEqual(marked[0], expected, places=9)

    def test_a_plain_trace_marks_the_refined_knee(self) -> None:
        """``trace()`` refines by default, so ``plot`` needs no extra call.

        Compared against an explicit refinement of a bare sweep, which is the knee
        ``knee_external_scale`` uses, and against the sampled knee of that same sweep,
        which is what a plain trace used to mark. The second is what makes the first
        falsifiable on this fixture.

        """
        frontier = self._frontier()
        bare = frontier.trace(num_points=6, refine_knee=False)
        grid = list(bare.points)
        explicit = none_throws(
            frontier.refine_knee(bare, knee_space=_cvxium_knee_space(bare.knee_space))
        )

        results = frontier.trace(num_points=6)

        self.assertTrue(results.knee_refinement_ran)
        knee = none_throws(results.refined_knee)
        np.testing.assert_allclose(
            knee.objectives, explicit.objectives, rtol=1e-6, atol=1e-9
        )
        marked = self._crosshair(results.plot())
        self.assertTrue(marked, "no vertical crosshair was drawn")
        self.assertAlmostEqual(
            marked[0], results._variance_coordinate(knee, "ess"), places=9
        )
        self.assertNotAlmostEqual(
            marked[0],
            results._variance_coordinate(self._grid_only_knee(bare, grid), "ess"),
            places=6,
        )

    def test_trace_refines_in_the_space_it_records(self) -> None:
        """The knee a plain trace solves for is the one its plot claims."""
        seen: list[str] = []
        real = ExternalValidityVarianceFrontier.refine_knee

        def record(
            self: ExternalValidityVarianceFrontier,
            results: ExternalValidityVarianceFrontierResults,
            max_iter: int = 24,
            slope_spread_tol: float = 1e-3,
            knee_space: Literal["primary", "root"] = "primary",
        ) -> FrontierPoint | None:
            seen.append(knee_space)
            return real(self, results, max_iter, slope_spread_tol, knee_space)

        with patch.object(ExternalValidityVarianceFrontier, "refine_knee", record):
            self._frontier().trace(num_points=4, knee_space="vif")
            self._frontier().trace(num_points=4, knee_space="se")
            self._frontier().trace(num_points=4, refine_knee=False)

        self.assertEqual(seen, ["primary", "root"])

    def test_refine_knee_false_returns_the_bare_sweep(self) -> None:
        """No probes folded in, and no refinement recorded."""
        results = self._frontier().trace(num_points=6, refine_knee=False)

        self.assertFalse(results.knee_refinement_ran)
        self.assertIsNone(results.refined_knee)
        self.assertLessEqual(len(results.points), 6 + 2)


class RefinementProbesReachTheReportTest(unittest.TestCase):
    r"""The knee probes are frontier points, so they show up like any other.

    Nothing here is special-cased for refinement: ``imbalances``,
    ``effective_sample_sizes``, ``weights`` and ``plot`` all read ``points``, and
    folding the probes into that list is the whole mechanism. These tests exist because
    that is easy to state and easy to break -- a rebuild that reconstructs the results
    object after refinement, or a property that caches, would silently drop the probes
    and leave the frontier looking exactly as coarse as before.

    The probes cluster at the bend, so their absence is worst precisely where the
    curve's shape is being read.

    """

    @staticmethod
    def _refined() -> (
        tuple[ExternalValidityVarianceFrontierResults, FrontierPoint, int]
    ):
        """A refined frontier, its knee, and how many points it had before."""
        frontier = PlotMarksTheRefinedKneeTest._frontier()
        results = frontier.trace(num_points=6, refine_knee=False)
        before = len(results.points)
        knee = frontier.refine_knee(
            results, knee_space=_cvxium_knee_space(results.knee_space)
        )
        assert knee is not None
        return results, knee, before

    def test_the_probes_are_in_the_reported_frontier(self) -> None:
        """Verify refinement probes appear in the reported frontier."""
        results, knee, before = self._refined()

        self.assertGreater(len(results.imbalances), before)
        self.assertIn(
            round(float(knee.objectives[1]), 12),
            [round(v, 12) for v in results.imbalances],
        )

    def test_every_property_reports_the_same_points(self) -> None:
        """Entry ``i`` of each is documented to be the same frontier point."""
        results, _, _ = self._refined()

        self.assertEqual(len(results.imbalances), len(results.points))
        self.assertEqual(len(results.effective_sample_sizes), len(results.points))
        self.assertEqual(len(results.weights), len(results.points))

    def test_the_probes_are_drawn(self) -> None:
        """Asserted on the data ``plot`` builds its curve from.

        Not on the rendered line: in ESS space the curve is an interpolated envelope, so
        the probe's own abscissa need not appear among the drawn vertices even when the
        probe is fully accounted for. What can be asserted is that the point reached the
        plotting path at all.

        """
        results, knee, before = self._refined()
        seen: list[list[float]] = []
        real = results._draw_frontier

        def capture(
            ax: plt.Axes,
            xs: list[float],
            ys: list[float],
            variance_space: Literal["vif", "se", "ess"] = "ess",
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            seen.append(list(xs))
            return real(ax, xs, ys, variance_space)

        with patch.object(results, "_draw_frontier", capture):
            results.plot()

        self.assertEqual(len(seen), 1)
        self.assertGreater(len(seen[0]), before)
        self.assertIn(
            round(results._variance_coordinate(knee, "ess"), 9),
            [round(x, 9) for x in seen[0]],
        )


class SecantBracketTest(unittest.TestCase):
    r"""The two samples the knee's tangent is estimated from.

    A unit test on the index arithmetic rather than on the drawn figure: the two defects
    this guards -- a one-sided difference where a central one is available, and a
    bracket straddling tied ordinates -- both produce a tangent that is drawn,
    plausible, and wrong, which a plot-level assertion can barely see.

    """

    bracket = staticmethod(ExternalValidityVarianceFrontierResults._secant_bracket)

    def test_an_off_grid_knee_straddles_its_neighbours(self) -> None:
        """Verify an off-grid knee straddles its neighbours."""
        self.assertEqual(self.bracket([0.0, 1.0, 2.0, 3.0], 1.5), (1, 2))

    def test_a_knee_on_a_sample_keeps_a_central_difference(self) -> None:
        """``bisect_left`` alone gives ``(k-1, k)``, a backward difference.

        Every unrefined plot takes this path, since an unrefined knee is one of the
        drawn points. Losing the centring biases the tangent by half a grid step on a
        curve whose entire subject is curvature.

        """
        self.assertEqual(self.bracket([0.0, 1.0, 2.0, 3.0], 2.0), (1, 3))

    def test_tied_ordinates_are_stepped_over(self) -> None:
        """A flat pair reports the grid's slope, not the frontier's.

        Sweep bands that saturate at one achieved imbalance put equal values in ``ys``.
        The caller's ``abs(dx) > 0`` guard does not catch it: the abscissae still
        differ, so the secant is drawn dead flat.

        """
        lo, hi = self.bracket([0.0, 1.0, 1.0, 3.0], 1.0) or (0, 0)

        self.assertNotEqual([0.0, 1.0, 1.0, 3.0][lo], [0.0, 1.0, 1.0, 3.0][hi])
        self.assertEqual((lo, hi), (0, 3))

    def test_a_wholly_flat_frontier_gets_no_tangent(self) -> None:
        """Nothing true to draw, so draw nothing."""
        self.assertIsNone(self.bracket([2.0, 2.0, 2.0], 2.0))

    def test_the_ends_get_no_tangent(self) -> None:
        """Unchanged: the old index rule also skipped the two endpoints."""
        self.assertIsNone(self.bracket([0.0, 1.0, 2.0], 0.0))
        self.assertIsNone(self.bracket([0.0, 1.0, 2.0], 2.0))


class FrontierAnnotationTest(unittest.TestCase):
    r"""Labelled reference points on all three PyRake frontier plots.

    Built from synthetic points rather than a trace, so the class runs in milliseconds:
    what is under test is where a point lands on the axes, not the solves behind it. The
    frontier is the convex curve ``primary = 1 + 2 exp(-auxiliary)``, which every plot
    can interpolate.

    """

    M: int = 10
    AUX: npt.NDArray[np.float64] = np.linspace(0.0, 5.0, 8)

    @classmethod
    def _points(cls) -> tuple[list[FrontierPoint], list[FrontierPoint]]:
        def point(aux: float, bound: float) -> FrontierPoint:
            solution = np.ones(cls.M)
            return FrontierPoint(
                objectives=np.array([1.0 + 2.0 * math.exp(-aux), aux]),
                solution=solution,
                bounds=np.array([bound]),
                ipm_result=SolveResult(
                    solution=solution,
                    status=SolveStatus.OPTIMAL,
                    message="",
                    solver="test",
                    n_iterations=0,
                ),
            )

        aux = cls.AUX
        points = [point(float(a), float(a)) for a in aux[1:-1]]
        corners = [point(float(aux[-1]), math.inf), point(float(aux[0]), math.inf)]
        return [corners[1], *points, corners[0]], corners

    @classmethod
    def _results(
        cls, kind: str
    ) -> EfficientFrontierResults | ExternalValidityVarianceFrontierResults:
        points, corners = cls._points()
        common: dict[str, Any] = {
            "points": points,
            "corners": corners,
            "primary_objective": 0,
            "n_attempted": len(points),
            "n_skipped": 0,
        }
        if kind == "efficient":
            return EfficientFrontierResults(**common)
        return ExternalValidityVarianceFrontierResults(**common)

    @staticmethod
    def _marked(ax: plt.Axes) -> list[tuple[float, float]]:
        """Coordinates of every star marker on the axes."""
        return [
            (
                float(np.asarray(line.get_xdata(), dtype=np.float64)[0]),
                float(np.asarray(line.get_ydata(), dtype=np.float64)[0]),
            )
            for line in ax.lines
            if line.get_marker() == "*"
        ]

    @staticmethod
    def _plot(
        results: EfficientFrontierResults | ExternalValidityVarianceFrontierResults,
        annotations: dict[str, AnnotationPoint],
        annotate_knee: bool = False,
    ) -> plt.Axes:
        """Plot in the primary objective's own units, whichever the class."""
        if isinstance(results, ExternalValidityVarianceFrontierResults):
            return results.plot(
                annotate_knee=annotate_knee,
                annotations=annotations,
                variance_space="vif",
            )
        return results.plot(annotate_knee=annotate_knee, annotations=annotations)

    @staticmethod
    def _labels(ax: plt.Axes) -> dict[str, str]:
        """Each annotation's text, mapped to its horizontal alignment."""
        return {t.get_text(): t.get_horizontalalignment() for t in ax.texts}

    def tearDown(self) -> None:
        plt.close("all")

    def test_each_frontier_marks_its_annotations(self) -> None:
        """Primary objective across, auxiliary up, in both classes."""
        for kind in ("efficient", "external"):
            with self.subTest(kind=kind):
                results = self._results(kind)
                ax = self._plot(results, {"Cochran": [1.6, 1.5]})
                self.assertEqual(self._marked(ax), [(1.6, 1.5)])
                self.assertIn("Cochran", self._labels(ax))

    def test_a_frontier_point_is_accepted(self) -> None:
        """A corner can be passed as it is, without unpacking its objectives."""
        for kind in ("efficient", "external"):
            with self.subTest(kind=kind):
                results = self._results(kind)
                ax = self._plot(results, {"Exact balance": results.corners[1]})
                vif, aux = results.corners[1].objectives
                self.assertEqual(self._marked(ax), [(float(vif), float(aux))])

    def test_annotations_follow_the_variance_space(self) -> None:
        """Carried into SE and ESS exactly as the curve is."""
        results = self._results("external")
        assert isinstance(
            results, ExternalValidityVarianceFrontierResults
        )  # Satisfy type-checker
        vif, aux = 1.6, 1.5
        expected: dict[Literal["vif", "se", "ess"], float] = {
            "vif": vif,
            "se": math.sqrt(vif),
            "ess": self.M / vif,
        }
        for space, x in expected.items():
            with self.subTest(variance_space=space):
                ax = results.plot(
                    annotate_knee=False,
                    annotations={"Cochran": [vif, aux]},
                    variance_space=space,
                )
                ((mx, my),) = self._marked(ax)
                self.assertAlmostEqual(mx, x, places=12)
                self.assertAlmostEqual(my, aux, places=12)

    def test_an_annotation_off_the_curve_stays_in_view(self) -> None:
        """Drawn before the knee's crosshairs freeze the axis limits.

        The point lies beyond every traced point on both axes. Drawn after the
        crosshairs, it would sit outside limits fitted to the curve alone.

        """
        far = (8.0, 9.0)
        for kind in ("efficient", "external"):
            with self.subTest(kind=kind):
                results = self._results(kind)
                ax = self._plot(results, {"far": far}, annotate_knee=True)
                lo_x, hi_x = ax.get_xlim()
                lo_y, hi_y = ax.get_ylim()
                self.assertTrue(lo_x <= far[0] <= hi_x, (lo_x, hi_x))
                self.assertTrue(lo_y <= far[1] <= hi_y, (lo_y, hi_y))

    def test_labels_face_the_middle_of_the_plot(self) -> None:
        """So a point at either end of the curve keeps its label in the axes."""
        results = self._results("efficient")
        ax = results.plot(
            annotate_knee=False,
            annotations={"left": [1.0, 5.0], "right": [3.0, 0.0]},
        )
        self.assertEqual(self._labels(ax), {"left": "left", "right": "right"})

    def test_a_malformed_annotation_raises_before_drawing(self) -> None:
        """Wrong length or non-finite, and no stray figure is left behind."""
        bad = {
            "three entries": [1.0, 2.0, 3.0],
            "nan": [float("nan"), 1.0],
        }
        for kind in ("efficient", "external"):
            for label, point in bad.items():
                with self.subTest(kind=kind, annotation=label):
                    results = self._results(kind)
                    before = len(plt.get_fignums())
                    with self.assertRaisesRegex(ValueError, label):
                        results.plot(annotations={label: point})
                    self.assertEqual(len(plt.get_fignums()), before)
