"""Test solve_with_knee_fallback, which picks the external band off the frontier."""

import unittest
from typing import Literal
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest
from cvxium import FrontierPoint, NewtonStepError

from pyrake._typing import none_throws
from pyrake.calibration.distance_metrics import (
    SquaredL2,
)
from pyrake.calibration.frontier import (
    ExternalValidityVarianceFrontier,
    ExternalValidityVarianceFrontierResults,
    external_band,
    knee_external_scale,
    relabel_for_the_chosen_band,
    solve_with_knee_fallback,
)
from pyrake.calibration.joint_calibrator import (
    JointCalibrator,
)


@pytest.mark.slow
class KneeFallbackTest(unittest.TestCase):
    """solve_with_knee_fallback picks the external band off the frontier."""

    M1: int = 60
    M2: int = 60
    Q: int = 3

    @classmethod
    def _calibrator(cls, offset: float) -> JointCalibrator:
        """A joint calibration whose respondents skew ``offset`` from the target.

        A large offset puts the conventional band out of reach, which is the regime
        where the minimize-phi fallback is expensive.

        """
        rng = np.random.default_rng(9109)
        Z1 = rng.standard_normal((cls.M1, cls.Q)) + offset
        Z2 = rng.standard_normal((cls.M2, cls.Q)) + offset
        return JointCalibrator(
            distance=SquaredL2(v=np.zeros(cls.M1 + cls.M2)),
            Z1=Z1,
            Z2=Z2,
            Z3=Z1,
            Z4=Z2,
            nu1=np.zeros(cls.Q),
            nu2=np.zeros(cls.Q),
            min_weight1=cls.M1 / 1e6,
            min_weight2=cls.M2 / 1e6,
            engine="pd",
        )

    def test_plain_fallback_traces_nothing(self) -> None:
        """The historical entry point is untouched and pays no sweep."""
        result = self._calibrator(offset=1.5).solve_with_fallback()

        self.assertIsNone(result.frontier)

    def test_knee_attaches_the_frontier(self) -> None:
        """The user can plot the tradeoff the chosen band sits on."""
        result = solve_with_knee_fallback(self._calibrator(offset=1.5))

        self.assertIsNotNone(result.frontier)
        self.assertGreater(len(none_throws(result.frontier).points), 2)

    def test_knee_buys_effective_sample_size_when_1_5_is_unreachable(self) -> None:
        """The point of the change: stop paying past the knee for balance."""
        requested = self._calibrator(offset=1.5).solve_with_fallback()
        knee = solve_with_knee_fallback(self._calibrator(offset=1.5))

        self.assertGreater(
            knee.combined_effective_sample_size,
            requested.combined_effective_sample_size,
        )

    def test_knee_never_tightens_past_the_conventional_band(self) -> None:
        """``max(1.5, phi_knee)`` floors the band at the Cochran scale.

        With a small offset the sample is nearly representative, so the conventional
        band is comfortably feasible and the rule must not chase a knee that sits
        tighter than it.

        """
        rake = self._calibrator(offset=0.05)
        result = solve_with_knee_fallback(rake)
        w = np.asarray(result.solution)
        frontier = none_throws(result.frontier)

        achieved = float(
            ExternalValidityVarianceFrontier.from_calibrator(rake).evaluate_objectives(
                w
            )[1]
        )
        knee_phi = float(frontier.knee().objectives[1])

        # The premise: this sample is representative enough that the knee sits
        # tighter than the conventional band. Without that the test proves
        # nothing about the floor.
        self.assertLess(knee_phi, 1.5)
        # So `max(1.5, phi_knee)` selects 1.5, and the solve sits on that rather
        # than chasing the knee to a tightness nobody asked for.
        self.assertAlmostEqual(achieved, 1.5, places=4)

    def test_a_failed_sweep_degrades_to_the_requested_band(self) -> None:
        """A broken frontier must not turn a good calibration into an error."""
        rake = self._calibrator(offset=1.5)
        with patch.object(
            ExternalValidityVarianceFrontier,
            "from_calibrator",
            side_effect=ValueError("no blocks"),
        ):
            result = solve_with_knee_fallback(rake)

        self.assertIsNone(result.frontier)
        self.assertEqual(len(np.asarray(result.solution)), self.M1 + self.M2)

    def test_an_unattainable_band_is_reported_as_such(self) -> None:
        """Succeeding at the knee must not read as meeting the target."""
        rake = self._calibrator(offset=1.5)

        result = solve_with_knee_fallback(rake)

        self.assertEqual(result.fallback_status, 2)
        self.assertIn("unattainable", result.fallback_message)
        self.assertNotIn(
            "achieved the desired covariate balance", result.fallback_message
        )

    def test_a_scalar_band_gives_external_band_every_row(self) -> None:
        """Both readers of psi must broadcast a scalar the same way.

        ``external_band`` keeps the internal tail by slicing ``psi[q_ext:]``. On a
        scalar that slice is empty, so the band it returns is shorter than ``B`` has
        rows and ``update_psi`` either errors or drops the internal block -- a different
        failure from ``from_calibrator``'s, out of the same omission.

        """
        rake = self._calibrator(offset=0.05)
        rake.update_psi(0.42)

        band = none_throws(external_band(rake, 3.0))

        self.assertEqual(len(band), 3 * self.Q)
        # The internal tail survives at the scalar it was set to.
        np.testing.assert_allclose(band[2 * self.Q :], np.full(self.Q, 0.42))
        rake.update_psi(band)  # must not raise: the band matches B's rows

    def test_no_knee_still_asks_for_an_attainable_band(self) -> None:
        """The conventional band can sit below anything the frontier reached.

        Asking for one nobody can meet sends solve_with_fallback down the
        minimize-imbalance branch, which is the far end of the frontier at a fraction of
        the effective sample size -- the outcome the knee exists to avoid, so a missing
        knee must not produce it.

        """
        rake = self._calibrator(offset=1.5)
        results = ExternalValidityVarianceFrontier.from_calibrator(rake).trace(
            num_points=6
        )
        phi_min = min(float(p.objectives[1]) for p in results.points)
        # The fixture is skewed hard enough that 1.5 is out of reach.
        self.assertGreater(phi_min, 1.5)

        with patch.object(
            ExternalValidityVarianceFrontier, "refine_knee", return_value=None
        ):
            scale, _ = knee_external_scale(rake)

        self.assertAlmostEqual(none_throws(scale), phi_min)

    def test_a_joint_infeasibility_has_nothing_past_the_seam(self) -> None:
        """Every row could meet a tighter band alone, but not all at once.

        Then the floor is the seam: below it ``solve_with_fallback`` would have to relax
        a row that could meet the band, so it widens the band back to the seam, and the
        frontier stops there as it always did.

        """
        rake = self._calibrator(offset=1.5)
        frontier = ExternalValidityVarianceFrontier.from_calibrator(rake)
        results = frontier.trace(num_points=6)
        self.assertIsNone(results.seam)
        self.assertAlmostEqual(
            frontier.floor(), frontier.seam(), delta=0.02 * frontier.seam()
        )
        phi_min = min(float(p.objectives[1]) for p in results.points)
        self.assertAlmostEqual(phi_min, frontier.seam(), delta=1e-3 * phi_min)

    def test_a_scalar_band_still_reaches_the_internal_block(self) -> None:
        """update_psi accepts a scalar; slicing one comes back empty.

        JointCalibrator expands a scalar band per block, so psi normally arrives at full
        length and the internal slice is well defined. A caller who sets a scalar
        through update_psi leaves one entry for every row, and slicing past it would
        silently substitute the default internal band -- tracing against a tighter rule
        than the calibration itself uses, with nothing to show for it.

        """
        rake = self._calibrator(offset=0.05)
        rake.update_psi(0.42)

        frontier = ExternalValidityVarianceFrontier.from_calibrator(rake)

        np.testing.assert_allclose(
            np.asarray(frontier.psi3, dtype=float), np.full(self.Q, 0.42)
        )

    def test_the_reported_band_does_not_assume_equal_groups(self) -> None:
        """Each group divides the scale by its own sqrt(M), so quote the looser.

        Halving the solution length assumes M1 == M2 and reports a band that belongs to
        neither group. The smaller group gets the looser band, and that is the one a
        reader needs, since it bounds both.

        """
        # The class fixture has equal groups, which cannot tell the two
        # formulas apart. 40 against 80 can: half the total is 60, and the
        # band that actually binds is the smaller group's, over sqrt(40).
        m1, m2, q = 40, 80, 3
        rng = np.random.default_rng(9109)
        Z1 = rng.standard_normal((m1, q)) + 0.05
        Z2 = rng.standard_normal((m2, q)) + 0.05
        rake = JointCalibrator(
            distance=SquaredL2(v=np.zeros(m1 + m2)),
            Z1=Z1,
            Z2=Z2,
            Z3=Z1,
            Z4=Z2,
            nu1=np.zeros(q),
            nu2=np.zeros(q),
            min_weight1=m1 / 1e6,
            min_weight2=m2 / 1e6,
            engine="pd",
        )
        results = ExternalValidityVarianceFrontier.from_calibrator(rake).trace(
            num_points=6
        )
        result = rake.solve_with_fallback()
        self.assertEqual(result.m1, m1)

        relabelled = relabel_for_the_chosen_band(result, 9.0, results)

        self.assertIn(f"{9.0 / np.sqrt(m1):.2g}", relabelled.fallback_message)
        self.assertNotIn(
            f"{9.0 / np.sqrt((m1 + m2) / 2):.2g}", relabelled.fallback_message
        )

    def test_a_deliberate_relaxation_says_so(self) -> None:
        """Declining an attainable band is a trade, not a shortfall."""
        rake = self._calibrator(offset=0.05)
        results = ExternalValidityVarianceFrontier.from_calibrator(rake).trace(
            num_points=6
        )
        phi_min = min(float(p.objectives[1]) for p in results.points)
        self.assertLess(phi_min, 1.5)  # the conventional band IS reachable

        with patch(
            "pyrake.calibration.frontier." "knee_external_scale",
            return_value=(9.0, results),
        ):
            result = solve_with_knee_fallback(rake)

        self.assertEqual(result.fallback_status, 4)
        self.assertIn("on purpose", result.fallback_message)
        self.assertIn("standard deviations", result.fallback_message)

    def test_the_conventional_band_keeps_the_cascades_own_label(self) -> None:
        """When the floor decides, nothing was relaxed and nothing is claimed."""
        rake = self._calibrator(offset=0.05)
        results = ExternalValidityVarianceFrontier.from_calibrator(rake).trace(
            num_points=6
        )

        with patch(
            "pyrake.calibration.frontier." "knee_external_scale",
            return_value=(1.5, results),
        ):
            result = solve_with_knee_fallback(rake)

        self.assertNotEqual(result.fallback_status, 4)
        self.assertNotIn("on purpose", result.fallback_message)

    def test_the_calibrator_keeps_its_own_band(self) -> None:
        """The chosen band is applied for the solve and then handed back."""
        rake = self._calibrator(offset=1.5)
        before = np.asarray(rake.psi, dtype=float).copy()

        result = solve_with_knee_fallback(rake)

        self.assertIsNotNone(result.frontier)
        np.testing.assert_allclose(np.asarray(rake.psi, dtype=float), before)

    def test_a_failed_rebuild_still_restores_the_band(self) -> None:
        """update_psi is two steps, and the first one already mutated psi.

        ``update_psi`` assigns ``self.psi`` and then rebuilds the problem. A rebuild
        that raises leaves the band applied, so the call has to sit inside the try:
        outside it, the finally never runs and the calibrator is stranded holding a band
        nobody asked for.

        """
        rake = self._calibrator(offset=1.5)
        before = np.asarray(rake.psi, dtype=float).copy()
        real = rake.update_psi
        calls = {"n": 0}

        def assign_then_fail(
            psi: float | list[float] | npt.NDArray[np.float64] | None,
        ) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                # Emulate the real sequence: psi lands, then the rebuild
                # raises. The restoring call must still work, or the test
                # cannot tell a restored band from one never applied.
                rake.psi = psi
                raise ValueError("rebuild failed")
            real(psi)

        # Patched on the instance, not the class: the sweep builds its own
        # bounded calibrators, and they must keep working.
        with (
            patch.object(rake, "update_psi", assign_then_fail),
            self.assertRaises(ValueError),
        ):
            solve_with_knee_fallback(rake)

        self.assertEqual(calls["n"], 2)  # applied, then restored
        np.testing.assert_allclose(np.asarray(rake.psi, dtype=float), before)

    def test_a_failed_sweep_still_restores_the_band(self) -> None:
        """Including when the sweep raises part-way through."""
        rake = self._calibrator(offset=1.5)
        before = np.asarray(rake.psi, dtype=float).copy()

        with patch.object(
            ExternalValidityVarianceFrontier, "trace", side_effect=ValueError("boom")
        ):
            solve_with_knee_fallback(rake)

        np.testing.assert_allclose(np.asarray(rake.psi, dtype=float), before)

    def test_a_newton_failure_degrades_rather_than_escaping(self) -> None:
        """NewtonStepError is not an OptimizationError, so it needs naming.

        The grid sweep's own swallow catches ``OptimizationError`` and
        ``ProblemInfeasibleError``; the corner solves can raise outside it, and
        ``NewtonStepError`` derives from ``Exception``. Without catching it by name a
        diagnostic sweep would break a calibration that is fine.

        """
        rake = self._calibrator(offset=1.5)

        with patch.object(
            ExternalValidityVarianceFrontier,
            "trace",
            side_effect=NewtonStepError("H is not positive definite"),
        ):
            scale, results = knee_external_scale(rake)
            result = solve_with_knee_fallback(rake)

        self.assertIsNone(scale)
        self.assertIsNone(results)
        self.assertIsNone(result.frontier)
        self.assertEqual(len(np.asarray(result.solution)), self.M1 + self.M2)

    def test_the_requested_space_reaches_the_root_find(self) -> None:
        """Translated into Cvxium's vocabulary on the way through.

        Deliberately not claiming to test the "read it back off the results" wiring: the
        argument and the recorded value agree by construction, so no assertion can tell
        which one ``knee_external_scale`` consulted. Reading it back is a structural
        guard against a future edit, not an observable behaviour. What is observable,
        and what this pins, is that the caller's choice arrives at ``refine_knee`` and
        arrives translated.

        """
        rake = self._calibrator(offset=1.5)
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
            knee_external_scale(rake, num_points=4, knee_space="vif")
            knee_external_scale(rake, num_points=4, knee_space="se")

        self.assertEqual(seen, ["primary", "root"])
