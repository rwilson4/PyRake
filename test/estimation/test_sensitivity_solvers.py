"""Test the Cvxium sensitivity LFP solver and its constraint layer."""

import unittest

import numpy as np
import numpy.typing as npt
from cvxium import (
    ArrowSystem,
    SolveStatus,
    ZeroSystem,
)
from parameterized import parameterized
from scipy.optimize import linprog

from pyrake.estimation.sensitivity_solvers import (
    CharnesCooperBoundsLayer,
    solve_bounded_linear_fractional_program,
)


def _make_layer_and_point(
    n: int,
    seed: int,
    s: float = 1.0,
) -> tuple[
    CharnesCooperBoundsLayer,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Build a layer and a strictly feasible point ``x = [w; s]``.

    Returns the layer, the bounds ``wl``, ``wu``, and a strictly interior iterate ``x``
    (each ``w_i`` strictly between ``wl_i * s`` and ``wu_i * s``).

    """
    rng = np.random.default_rng(seed)
    wl = 0.2 + 0.3 * rng.random(n)  # in (0.2, 0.5)
    wu = wl + 0.5 + rng.random(n)  # strictly above wl
    layer = CharnesCooperBoundsLayer(wl, wu)
    fraction = 0.25 + 0.5 * rng.random(n)  # in (0.25, 0.75)
    weights = (wl + fraction * (wu - wl)) * s
    x = np.append(weights, s)
    return layer, wl, wu, x


class CharnesCooperBoundsLayerTest(unittest.TestCase):
    """Tests for ``CharnesCooperBoundsLayer``."""

    @parameterized.expand(
        [
            ("n3", 3, 1),
            ("n10", 10, 2),
            ("n50", 50, 3),
        ]
    )
    def test_num_constraints(self, _name: str, n: int, seed: int) -> None:
        """Verify the layer reports twice the dimension as constraints."""
        layer, _wl, _wu, x = _make_layer_and_point(n, seed)
        self.assertEqual(layer.num_constraints(len(x)), 2 * n)

    @parameterized.expand([("n3", 3, 1), ("n10", 10, 2)])
    def test_constraints_match_explicit(self, _name: str, n: int, seed: int) -> None:
        """Verify constraints match the explicit bound expressions."""
        layer, wl, wu, x = _make_layer_and_point(n, seed)
        weights, s = x[:-1], x[-1]
        expected = np.concatenate([wl * s - weights, weights - wu * s])
        np.testing.assert_allclose(layer.constraints(x), expected)
        # All strictly feasible at an interior point.
        self.assertTrue(np.all(layer.constraints(x) < 0))

    @parameterized.expand([("n3", 3, 4), ("n10", 10, 5)])
    def test_constraints_shift_loosens(self, _name: str, n: int, seed: int) -> None:
        """Verify a positive shift loosens all constraints."""
        layer, _wl, _wu, x = _make_layer_and_point(n, seed)
        shift = 0.05
        np.testing.assert_allclose(
            layer.constraints(x, shift=shift), layer.constraints(x) - shift
        )

    @parameterized.expand([("n3", 3, 6), ("n10", 10, 7)])
    def test_grad_constraints_and_multiplies(
        self, _name: str, n: int, seed: int
    ) -> None:
        """Verify constraint Jacobians and their multiplies match."""
        layer, wl, wu, x = _make_layer_and_point(n, seed)
        G = layer.grad_constraints(x)
        expected = np.vstack(
            [
                np.hstack([-np.eye(n), wl.reshape(-1, 1)]),
                np.hstack([np.eye(n), -wu.reshape(-1, 1)]),
            ]
        )
        np.testing.assert_allclose(G, expected)
        rng = np.random.default_rng(seed + 100)
        v = rng.standard_normal(n + 1)
        u = rng.standard_normal(2 * n)
        np.testing.assert_allclose(layer.grad_constraints_multiply(x, v), G @ v)
        np.testing.assert_allclose(
            layer.grad_constraints_transpose_multiply(x, u), G.T @ u
        )

    @parameterized.expand([("n3", 3, 8), ("n10", 10, 9), ("n50", 50, 10)])
    def test_hessian_contribution_matches_ground_truth(
        self, _name: str, n: int, seed: int
    ) -> None:
        """Verify the Hessian contribution matches ground truth."""
        layer, _wl, _wu, x = _make_layer_and_point(n, seed)
        rng = np.random.default_rng(seed + 200)
        lmbda = 0.5 + rng.random(2 * n)  # positive dual weights
        for shift in (0.0, 0.05):
            G = layer.grad_constraints(x)
            c = layer.constraints(x, shift=shift)
            # Curvature atom for affine constraints: sum_i (-lmbda_i / c_i) g_i g_i^T.
            d = -lmbda / c
            H_expected = G.T @ np.diag(d) @ G
            system = layer.hessian_contribution(x, lmbda=lmbda, shift=shift)
            self.assertIsInstance(system, ArrowSystem)
            v = rng.standard_normal(n + 1)
            np.testing.assert_allclose(
                system.multiply(v), H_expected @ v, rtol=1e-9, atol=1e-9
            )
            # Round-trip solve (H_expected is PD for positive lmbda at an interior x).
            rhs = H_expected @ v
            np.testing.assert_allclose(system.solve(rhs), v, rtol=1e-7, atol=1e-7)

    @parameterized.expand([("n3", 3, 15), ("n10", 10, 16)])
    def test_hessian_curvature_is_zero(self, _name: str, n: int, seed: int) -> None:
        """Verify the Hessian curvature vanishes for affine bounds."""
        layer, _wl, _wu, x = _make_layer_and_point(n, seed)
        lmbda = np.ones(2 * n)
        curvature = layer.hessian_curvature(x, lmbda)
        self.assertIsInstance(curvature, ZeroSystem)
        v = np.ones(n + 1)
        np.testing.assert_allclose(curvature.multiply(v), np.zeros(n + 1))

    def test_degenerate_bounds_are_widened(self) -> None:
        """Verify degenerate bounds widen symmetrically without mutation."""
        wl = np.array([1.0, 0.3, 1.0])
        wu = np.array([1.0, 0.9, 1.0])  # entries 0 and 2 are degenerate
        layer = CharnesCooperBoundsLayer(wl, wu)
        # Degenerate pairs widened symmetrically by 1e-6.
        np.testing.assert_allclose(layer.wl, [1.0 - 1e-6, 0.3, 1.0 - 1e-6])
        np.testing.assert_allclose(layer.wu, [1.0 + 1e-6, 0.9, 1.0 + 1e-6])
        # The caller's arrays are not mutated.
        np.testing.assert_allclose(wl, [1.0, 0.3, 1.0])
        np.testing.assert_allclose(wu, [1.0, 0.9, 1.0])

    @parameterized.expand([("n3", 3, 19), ("n10", 10, 20)])
    def test_btls_max_step_is_the_boundary(self, _name: str, n: int, seed: int) -> None:
        """Verify the backtracking step stops at the boundary."""
        layer, _wl, _wu, x = _make_layer_and_point(n, seed)
        rng = np.random.default_rng(seed + 500)
        dw = rng.standard_normal(n + 1)
        alpha = layer.btls_max_step(x, dw)
        if np.isfinite(alpha):
            # Just inside the step stays strictly feasible.
            self.assertTrue(np.all(layer.constraints(x + 0.999 * alpha * dw) < 0))
            # At the step the binding constraint touches zero.
            self.assertAlmostEqual(
                float(np.max(layer.constraints(x + alpha * dw))), 0.0
            )

    def test_invalid_inputs_raise(self) -> None:
        """Verify invalid inputs raise ``ValueError``."""
        with self.assertRaises(ValueError):
            CharnesCooperBoundsLayer(np.array([0.1, 0.2]), np.array([0.5]))
        layer = CharnesCooperBoundsLayer(np.array([0.1, 0.2]), np.array([0.5, 0.9]))
        x = np.array([0.3, 0.5, 1.0])
        with self.assertRaises(ValueError):
            layer.hessian_contribution(x, lmbda=None)


class SolveSensitivityLfpTest(unittest.TestCase):
    """Tests for ``solve_bounded_linear_fractional_program`` (the structured-Problem LFP path)."""

    @staticmethod
    def _random_instance(n: int, seed: int) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        rng = np.random.default_rng(seed)
        c = rng.standard_normal(n)
        y = 0.5 + rng.random(n)  # strictly positive denominator outcomes
        wl = 0.2 + 0.3 * rng.random(n)
        wu = wl + 0.5 + rng.random(n)
        return c, y, wl, wu

    @parameterized.expand([("n3", 3, 41), ("n8", 8, 42), ("n50", 50, 43)])
    def test_objective_matches_scipy_linprog(
        self, _name: str, n: int, seed: int
    ) -> None:
        """Verify the LFP objective matches ``scipy`` ``linprog``."""
        c, y, wl, wu = self._random_instance(n, seed)
        result = solve_bounded_linear_fractional_program(c=c, y=y, wl=wl, wu=wu)
        self.assertIn(result.status, (SolveStatus.OPTIMAL, SolveStatus.ACCEPTABLE))
        # Independent oracle: the Charnes-Cooper LP over [w; s].
        c_lp = np.append(c, 0.0)
        A_eq = np.append(y, 0.0).reshape(1, n + 1)
        b_eq = np.array([1.0])
        A_ub = np.vstack(
            [
                np.hstack([-np.eye(n), wl.reshape(-1, 1)]),  # wl s - w <= 0
                np.hstack([np.eye(n), -wu.reshape(-1, 1)]),  # w - wu s <= 0
            ]
        )
        b_ub = np.zeros(2 * n)
        bounds = [(None, None)] * n + [(0.0, None)]
        lp = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        self.assertEqual(lp.status, 0)
        # The IPM converges to its (soft) duality-gap tolerance, so it agrees with
        # the exact simplex optimum only to ~1e-5. The RatioEstimator golden
        # sensitivity tests (test_population.py) pin the end-to-end values tightly.
        self.assertAlmostEqual(result.objective_value, float(lp.fun), places=4)

    def test_nonpositive_denominator_raises(self) -> None:
        """Verify a nonpositive denominator raises ``ValueError``."""
        with self.assertRaises(ValueError):
            solve_bounded_linear_fractional_program(
                c=np.ones(3),
                y=-np.ones(3),
                wl=np.full(3, 0.5),
                wu=np.full(3, 1.5),
            )
