"""Test the external frontier past the smallest band every row can meet."""

import itertools
import math
import unittest

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest
from cvxium import ProblemInfeasibleError

from pyrake._typing import none_throws
from pyrake.calibration.distance_metrics import (
    SquaredL2,
)
from pyrake.calibration.frontier import (
    ExternalValidityVarianceFrontier,
    ExternalValidityVarianceFrontierResults,
    external_band,
    relabel_for_the_chosen_band,
)
from pyrake.calibration.joint_calibrator import (
    JointCalibrator,
)


def _l30_calibrator() -> JointCalibrator:
    """A BLS survey's five engagement levels, the second empty in the control group.

    No control weighting can give 8--14 days any share, so no band below the seam meets
    every external row; every other level can be balanced.

    """
    mu = np.array([0.072558, 0.033584, 0.043063, 0.245502, 0.605294])
    mu = mu / np.sum(mu)
    sigma = np.sqrt(mu * (1 - mu))

    def whitened_indicators(counts: list[int]) -> npt.NDArray[np.float64]:
        levels = np.repeat(np.arange(len(counts)), counts)
        return np.eye(len(counts))[levels] / sigma

    Z1 = whitened_indicators([3, 0, 5, 26, 472])
    Z2 = whitened_indicators([7, 6, 6, 34, 448])
    M = Z1.shape[0] + Z2.shape[0]
    return JointCalibrator(
        distance=SquaredL2(v=np.zeros(M)),
        Z1=Z1,
        Z2=Z2,
        Z3=Z1,
        Z4=Z2,
        nu1=mu / sigma,
        nu2=mu / sigma,
    )


@pytest.mark.slow
class PastTheSeamTest(unittest.TestCase):
    """The external frontier continues below the smallest band every row can meet."""

    _results: ExternalValidityVarianceFrontierResults | None = None
    _frontier: ExternalValidityVarianceFrontier | None = None

    @classmethod
    def _traced(
        cls,
    ) -> tuple[
        ExternalValidityVarianceFrontier, ExternalValidityVarianceFrontierResults
    ]:
        if cls._results is None or cls._frontier is None:
            cls._frontier = ExternalValidityVarianceFrontier.from_calibrator(
                _l30_calibrator()
            )
            cls._results = cls._frontier.trace(num_points=12)
        return cls._frontier, cls._results

    def test_the_seam_is_the_empty_level(self) -> None:
        """The 8--14 share's whole population share, in the group's own units."""
        frontier, results = self._traced()
        mu = 0.033584 / 1.000001
        expected = np.sqrt(506) * mu / np.sqrt(mu * (1 - mu))
        self.assertAlmostEqual(none_throws(results.seam), expected, delta=1e-3)
        self.assertEqual(results.seam, frontier.seam())

    def test_below_the_seam_the_coordinate_is_the_band(self) -> None:
        """Verify sub-seam points report the band as their coordinate."""
        frontier, results = self._traced()
        seam = none_throws(results.seam)
        below = [
            p
            for p in results.points
            if np.all(np.isfinite(p.bounds)) and float(p.bounds[0]) < seam
        ]
        self.assertGreaterEqual(len(below), 2)
        for point in below:
            self.assertAlmostEqual(float(point.objectives[1]), float(point.bounds[0]))
            # The weights' own max imbalance stays pinned at the seam.
            attained = float(frontier.evaluate_objectives(point.solution)[1])
            self.assertAlmostEqual(attained, seam, delta=1e-3 * seam)

    def test_tightening_past_the_seam_still_costs_sample(self) -> None:
        """Effective sample size falls monotonically all the way to the corner."""
        _, results = self._traced()
        ess = results.effective_sample_sizes  # sorted by imbalance ascending
        self.assertTrue(all(a <= b + 1e-6 for a, b in itertools.pairwise(ess)), ess)

    def test_the_tight_corner_sits_at_the_floor(self) -> None:
        """Holdout's observed levels can hold the empty one's share from about 0.55.

        Their shares must exceed their targets by 3.36 pp in total, and a band of phi
        standard deviations allows about 6.15 pp x phi.

        """
        frontier, results = self._traced()
        tight = min(results.corners, key=lambda p: float(p.objectives[1]))
        self.assertAlmostEqual(float(tight.objectives[1]), frontier.floor())
        self.assertGreater(frontier.floor(), 0.54)
        self.assertLess(frontier.floor(), 0.57)

    def test_a_band_below_the_floor_is_not_a_frontier_point(self) -> None:
        """Verify a band below the floor raises as infeasible."""
        frontier, _ = self._traced()
        with self.assertRaises(ProblemInfeasibleError):
            frontier.solve_with_bounds(np.array([0.5 * frontier.floor()]))

    def test_no_slopes_past_the_seam(self) -> None:
        """Verify points past the seam report no tangent slope."""
        _, results = self._traced()
        seam = none_throws(results.seam)
        slopes = none_throws(results.tangent_slopes)
        for point, slope in zip(results.points, slopes, strict=True):
            if np.all(np.isfinite(point.bounds)) and float(point.bounds[0]) < seam:
                self.assertTrue(math.isnan(slope))

    def test_plot_marks_the_seam(self) -> None:
        """Verify the plot marks the seam with a dotted line."""
        _, results = self._traced()
        ax = results.plot()
        try:
            seam = none_throws(results.seam)
            self.assertTrue(
                any(
                    np.allclose(line.get_ydata(), seam)
                    for line in ax.get_lines()
                    if line.get_linestyle() == ":"
                )
            )
        finally:
            plt.close("all")

    def test_a_frontier_that_stops_at_the_seam(self) -> None:
        """Verify disabling continuation stops the sweep at the seam."""
        frontier = ExternalValidityVarianceFrontier.from_calibrator(
            _l30_calibrator(), continue_past_seam=False
        )
        results = frontier.trace(num_points=4)
        self.assertIsNone(results.seam)
        lowest = min(float(p.objectives[1]) for p in results.points)
        self.assertAlmostEqual(lowest, frontier.seam(), delta=1e-3 * lowest)

    def test_a_band_past_the_seam_is_relabelled_with_its_relaxed_rows(self) -> None:
        """The knee can land between the conventional band and the seam."""
        _, results = self._traced()
        rake = _l30_calibrator()
        scale = 3.0
        rake.update_psi(none_throws(external_band(rake, scale)))
        result = rake.solve_with_fallback()
        # At this looser band the Marketing 8--14 row is met; only the empty
        # level is not.
        self.assertEqual(result.relaxed_rows, (1,))

        relabelled = relabel_for_the_chosen_band(result, scale, results)
        self.assertEqual(relabelled.fallback_status, 2)
        self.assertIn("unattainable on every row", relabelled.fallback_message)
        self.assertIn(f"{none_throws(results.seam):.3g}", relabelled.fallback_message)
        self.assertIn(
            "1 constraint could not meet even that", relabelled.fallback_message
        )
