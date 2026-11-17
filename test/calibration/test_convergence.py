"""A calibration must take full Newton steps once it is entitled to.

Cvxium's own tests cover the step-cap arithmetic in isolation. These are the
consumer-level guards: they run a real `JointCalibrator` and assert on the convergence
the solver actually delivered, which is the thing that regressed silently for three
months while every unit test stayed green.

"""

from __future__ import annotations

import unittest

import numpy as np
from cvxium import BarrierDiagnostics, SolveResult

from pyrake.calibration.distance_metrics import (
    SquaredL2,
)
from pyrake.calibration.joint_calibrator import (
    JointCalibrator,
)


def _calibrator(offset: float) -> JointCalibrator:
    """A joint calibration whose respondents skew ``offset`` from the target."""
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
        min_weight1=M1 / 1e6,
        min_weight2=M2 / 1e6,
        engine="ipm",
    )


def _barrier_results(result: SolveResult) -> list[SolveResult]:
    """``result`` and every Phase I nested inside it.

    Phase I is where the step cap bit, and it hangs off ``SolveResult.phase1`` rather
    than appearing in the top-level diagnostics -- a guard that looked only at the outer
    result would have passed throughout the regression.

    """
    out: list[SolveResult] = []
    node: SolveResult | None = result
    while node is not None:
        out.append(node)
        node = node.phase1
    return out


def _quadratic_phase_steps(result: SolveResult) -> tuple[float, ...]:
    """Every step modifier taken past quadratic convergence, Phase I included."""
    steps: list[float] = []
    for node in _barrier_results(result):
        diagnostics = node.diagnostics
        if isinstance(diagnostics, BarrierDiagnostics):
            steps.extend(diagnostics.steps_past_quadratic_convergence())
    return tuple(steps)


def _newton_steps(result: SolveResult) -> int:
    """Total inner Newton iterations across the solve, Phase I included."""
    total = 0
    for node in _barrier_results(result):
        diagnostics = node.diagnostics
        if isinstance(diagnostics, BarrierDiagnostics):
            total += sum(diagnostics.inner_nits)
    return total


class FullStepsPastQuadraticConvergenceTest(unittest.TestCase):
    """Boyd and Vandenberghe 9.5.3: the unit step is accepted there."""

    def test_a_feasible_calibration_takes_full_steps_once_it_may(self) -> None:
        """Covers the main solve, which the 2026-06 regression never touched.

        Worth stating plainly: this one passes with that bug reinstated, so it is a
        guard against a future cap on the ordinary path rather than a reproduction of
        the known one. The infeasible case below is the test that catches it.

        """
        result = _calibrator(offset=0.05).solve_with_fallback().result
        steps = _quadratic_phase_steps(result)

        # Guard the guard: an assertion over an empty tuple passes forever.
        self.assertGreater(len(steps), 0, "no solve reached the quadratic phase")
        self.assertEqual(
            {1.0},
            set(steps),
            f"capped steps in the quadratic phase: {sorted(set(steps))}",
        )

    def test_an_infeasible_calibration_also_takes_full_steps(self) -> None:
        """The Phase-I-heavy path, and the one that reproduces the regression.

        Verified by reinstating the ``alpha = 1.0`` cap: this fails with "capped steps
        in the quadratic phase: [0.5]".

        """
        result = _calibrator(offset=3.0).solve_with_fallback().result
        steps = _quadratic_phase_steps(result)

        self.assertGreater(len(steps), 0, "no solve reached the quadratic phase")
        self.assertEqual(
            {1.0},
            set(steps),
            f"capped steps in the quadratic phase: {sorted(set(steps))}",
        )


class NewtonStepBudgetTest(unittest.TestCase):
    """A ceiling on iterations, so worse convergence cannot land quietly.

    The half-step regression cost 26% more Newton steps on Gandalf's real study and
    nothing failed. A budget is blunt, but it is the only thing that catches "still
    correct, just worse".

    The ceilings are set from measured counts with modest headroom, not generous
    headroom: a budget loose enough never to trip is also loose enough never to catch
    anything. Reinstating the ``alpha = 1.0`` cap takes the infeasible fixture from 86
    Newton steps to 123, so the ceiling sits between them. If a legitimate change pushes
    past one of these, raise it deliberately and record the new number here.

    """

    def test_a_feasible_calibration_stays_within_budget(self) -> None:
        """51 steps as measured; the ceiling allows ordinary drift."""
        result = _calibrator(offset=0.05).solve_with_fallback().result

        self.assertLess(_newton_steps(result), 60)

    def test_an_infeasible_calibration_stays_within_budget(self) -> None:
        """86 with full steps, 123 when the cap is reinstated."""
        result = _calibrator(offset=3.0).solve_with_fallback().result

        self.assertLess(_newton_steps(result), 100)
