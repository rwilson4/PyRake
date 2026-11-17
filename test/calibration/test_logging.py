"""The fallback cascade says which stage it is on, and how long it took."""

import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np

import pyrake
from pyrake.calibration.distance_metrics import (
    SquaredL2,
)
from pyrake.calibration.joint_calibrator import (
    JointCalibrator,
)


def _calibrator(offset: float, psi3: float | None = None) -> JointCalibrator:
    """A joint calibration whose respondents skew ``offset`` from the target.

    A large offset puts the requested band out of reach, which is the regime where the
    cascade falls through to minimizing the violation.

    """
    M1, M2, q = 40, 40, 3
    rng = np.random.default_rng(20260918)
    Z1 = rng.standard_normal((M1, q)) + offset
    Z2 = rng.standard_normal((M2, q)) + offset
    return JointCalibrator(
        distance=SquaredL2(v=np.zeros(M1 + M2)),
        Z1=Z1,
        Z2=Z2,
        Z3=Z1,
        Z4=Z2,
        nu1=np.zeros(q),
        nu2=np.zeros(q),
        psi3=psi3,
        min_weight1=M1 / 1e6,
        min_weight2=M2 / 1e6,
        engine="pd",
    )


class CascadeNarrationTest(unittest.TestCase):
    """What the log has to answer: which stage, and where did the time go."""

    def setUp(self) -> None:
        for name in ("pyrake", "cvxium"):
            log = logging.getLogger(name)
            self.addCleanup(log.setLevel, log.level)
            self.addCleanup(setattr, log, "propagate", log.propagate)
            self.addCleanup(setattr, log, "handlers", list(log.handlers))

    def _solve_and_read(self, offset: float) -> str:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "calibration.log"
            handler = pyrake.enable_logging(path)
            _calibrator(offset).solve_with_fallback()
            handler.flush()
            return path.read_text()

    def test_it_names_each_stage_it_reaches(self) -> None:
        """Without this the Cvxium output is thousands of uncontextualized lines."""
        text = self._solve_and_read(offset=0.05)

        self.assertIn("Stage: bounds and exact balance", text)
        self.assertIn("Stage: internal validity", text)
        self.assertIn("Stage: external validity", text)

    def test_it_reports_the_shape_of_the_problem(self) -> None:
        """So a log read later says what was being calibrated, not just how."""
        text = self._solve_and_read(offset=0.05)

        self.assertIn("Calibrating 80 weights", text)
        self.assertIn("imbalance rows", text)

    def test_an_infeasible_stage_says_so_rather_than_going_quiet(self) -> None:
        """The stage that falls back is the one that costs the time.

        It used to be indistinguishable from a feasible one: the cascade emitted nothing
        either way, so a run that spent all its time minimizing a violation looked
        exactly like one that sailed through.

        """
        text = self._solve_and_read(offset=3.0)

        self.assertIn("INFEASIBLE, minimized the violation instead", text)
        self.assertIn("Done: status", text)

    def test_cvxium_lands_in_the_same_file_under_its_stage(self) -> None:
        """One handler for both namespaces, which is what orders the file.

        Two handlers would buffer separately and the interleaving would be luck rather
        than a property.

        """
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "calibration.log"
            handler = pyrake.enable_logging(path, solver_level=logging.INFO)
            _calibrator(0.05).solve_with_fallback()
            handler.flush()
            lines = path.read_text().splitlines()

        first_stage = next(i for i, line in enumerate(lines) if "Stage:" in line)
        cvxium_after = [
            i for i, line in enumerate(lines) if "cvxium." in line and i > first_stage
        ]
        self.assertTrue(cvxium_after, "no Cvxium records followed the first stage")
        self.assertTrue(any("pyrake.rake" in line for line in lines))

    def test_the_default_is_narration_only(self) -> None:
        """Cvxium's INFO is per-centering-step and drowns the narration.

        The point of the default is that the file stays short enough to read top to
        bottom, so the solver is silent until asked for.

        """
        text = self._solve_and_read(offset=0.05)

        self.assertIn("pyrake.rake", text)
        self.assertNotIn("cvxium.", text)
        self.assertLess(len(text.splitlines()), 20)
