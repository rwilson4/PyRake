"""Test distance metrics."""

import unittest
import warnings

import numpy as np
from cvxium import DiagonalSystem
from scipy.special import expit

from pyrake.calibration.distance_metrics import (
    Huber,
    KLDivergence,
    SquaredL2,
)
from pyrake.calibration.joint_calibrator import JointCalibrator


class SquaredL2Test(unittest.TestCase):
    """Tests for ``SquaredL2``."""

    def test_squared_l2(self) -> None:
        """Verify ``SquaredL2`` value, gradient, and diagonal Hessian."""
        M = 100
        seed = 0
        np.random.seed(seed)
        w = np.random.rand(M)
        v = np.random.rand(M)
        metric = SquaredL2(v=v)

        val = metric.evaluate(w)
        expected_val = np.sum((w - v) ** 2) / M
        np.testing.assert_allclose(val, expected_val)

        grad = metric.gradient(w)
        expected_grad = (2.0 / M) * (w - v)
        np.testing.assert_allclose(grad, expected_grad)

        hess_diag = metric.hessian_diagonal(w)
        expected_hess = (2.0 / M) * np.ones_like(w)
        np.testing.assert_allclose(hess_diag, expected_hess)


class KLDivergenceTest(unittest.TestCase):
    """Tests for ``KLDivergence``."""

    def test_kl(self) -> None:
        """Verify ``KLDivergence`` value, gradient, and diagonal Hessian."""
        M = 100
        seed = 1
        np.random.seed(seed)
        w = np.random.rand(M) + 0.1  # ensure strictly positive
        v = np.random.rand(M) + 0.1
        metric = KLDivergence(v=v)

        val = metric.evaluate(w)
        expected_val = (1.0 / M) * np.sum(w * np.log(w / v) - w + v)
        np.testing.assert_allclose(val, expected_val)

        grad = metric.gradient(w)
        expected_grad = (1.0 / M) * np.log(w / v)
        np.testing.assert_allclose(grad, expected_grad)

        hess_diag = metric.hessian_diagonal(w)
        expected_hess = (1.0 / M) / w
        np.testing.assert_allclose(hess_diag, expected_hess)


class KLDivergenceDomainTest(unittest.TestCase):
    """``KLDivergence`` is defined only for w > 0, and says so to the solver."""

    def test_evaluate_is_infinite_outside_the_domain(self) -> None:
        """A nonpositive weight gives +inf, not nan, and no warning."""
        metric = KLDivergence(v=np.ones(3))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(metric.evaluate(np.array([1.0, -0.5, 2.0])), np.inf)
            self.assertEqual(metric.evaluate(np.array([1.0, 0.0, 2.0])), np.inf)

    def test_step_bound_reaches_the_boundary_exactly(self) -> None:
        """The bound is the step at which the first weight hits zero."""
        metric = KLDivergence()
        w = np.array([1.0, 2.0, 0.5])
        dw = np.array([-0.5, -4.0, 1.0])
        alpha = metric.btls_keep_in_domain(w, dw)
        self.assertAlmostEqual(alpha, 0.5)
        self.assertAlmostEqual(float(np.min(w + alpha * dw)), 0.0)

    def test_step_bound_is_infinite_when_nothing_decreases(self) -> None:
        """A direction that shrinks no weight never leaves the domain."""
        metric = KLDivergence()
        w = np.array([1.0, 2.0])
        self.assertEqual(metric.btls_keep_in_domain(w, np.array([0.0, 3.0])), np.inf)

    def test_joint_calibration_never_steps_outside_the_domain(self) -> None:
        """A simulated A/B-test survey whose dual solve once took log of w < 0.

        The barrier method evaluates its dual by minimizing the Lagrangian over
        unconstrained w, bounded only by the objective's domain. Without a domain bound
        it tried full steps past w = 0 and warned on the logarithm.

        """
        rng = np.random.default_rng(0)
        N, n = 500_000, 4_000
        age = rng.uniform(18, 80, N)
        urban = (rng.random(N) < 0.6).astype(float)
        heavy = (rng.random(N) < expit(1.5 - 0.05 * (age - 18))).astype(float)
        female = (rng.random(N) < 0.5).astype(float)
        treated = rng.random(N) < 0.5
        rng.normal(0, 1, N)  # The outcome noise, drawn when this case was found.
        X = np.column_stack([age, urban, heavy, female])
        mu, sigma = X.mean(axis=0), X.std(axis=0)

        def survey(group: np.ndarray, lift: float) -> np.ndarray:
            idx = rng.choice(np.flatnonzero(group), n, replace=False)
            eta = -0.3 - 0.04 * (age[idx] - 45) + 0.5 * urban[idx] + lift * heavy[idx]
            return idx[rng.random(n) < expit(eta)]

        control, treatment = survey(~treated, 0.2), survey(treated, 1.2)
        Z1, Z2 = X[control] / sigma, X[treatment] / sigma
        calibrator = JointCalibrator(
            distance=KLDivergence(),
            Z1=Z1,
            Z2=Z2,
            Z3=Z1,
            Z4=Z2,
            nu1=mu / sigma,
            nu2=mu / sigma,
            min_weight1=len(control) / int((~treated).sum()),
            min_weight2=len(treatment) / int(treated.sum()),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = calibrator.solve_with_fallback()

        self.assertEqual(result.fallback_status, 0)
        self.assertTrue(np.all(result.solution > 0))


class HuberTest(unittest.TestCase):
    """Tests for ``Huber``."""

    def test_huber(self) -> None:
        """Verify ``Huber`` value, gradient, and diagonal Hessian."""
        M = 100
        seed = 2
        np.random.seed(seed)
        w = np.random.rand(M)
        v = np.random.rand(M)
        delta = 0.1
        metric = Huber(v=v, delta=delta)

        d = w - v
        val = metric.evaluate(w)
        expected_val = (1.0 / M) * np.sum(
            np.where(np.abs(d) <= delta, d**2, delta * (2 * np.abs(d) - delta))
        )
        np.testing.assert_allclose(val, expected_val)

        grad = metric.gradient(w)
        expected_grad = np.where(
            np.abs(d) <= delta, (2.0 / M) * d, (2.0 / M) * delta * np.sign(d)
        )
        np.testing.assert_allclose(grad, expected_grad)

        hess_diag = metric.hessian_diagonal(w)
        expected_hess = np.where(np.abs(d) <= delta, (2.0 / M), 0.0)
        np.testing.assert_allclose(hess_diag, expected_hess)


class DistanceDimensionTest(unittest.TestCase):
    """``Distance`` resolves its dimension from ``v`` or an explicit ``dimension``."""

    def test_infers_dimension_from_v(self) -> None:
        """Verify the dimension infers from ``v`` for each metric."""
        v = np.arange(7, dtype=np.float64)
        for metric in (SquaredL2(v=v), KLDivergence(v=v), Huber(v=v)):
            with self.subTest(metric=type(metric).__name__):
                self.assertEqual(metric.dimension, 7)

    def test_explicit_dimension_when_no_v(self) -> None:
        """Verify an explicit dimension holds without ``v``."""
        for cls in (SquaredL2, KLDivergence, Huber):
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls(dimension=12).dimension, 12)

    def test_bare_distance_defers_dimension_until_bound(self) -> None:
        """Verify a bare distance defers dimension until bound."""
        # A bare distance (no ``v``, no ``dimension``) constructs successfully but
        # defers ``M``; accessing ``dimension`` before binding raises, and
        # ``bind_dimension`` resolves it (as ``Rake`` does from its data).
        for cls in (SquaredL2, KLDivergence, Huber):
            with self.subTest(cls=cls.__name__):
                metric = cls()
                with self.assertRaises(ValueError):
                    _ = metric.dimension
                metric.bind_dimension(9)
                self.assertEqual(metric.dimension, 9)

    def test_bind_dimension_conflict_raises(self) -> None:
        """Verify rebinding to a conflicting dimension raises."""
        for cls in (SquaredL2, KLDivergence, Huber):
            with self.subTest(cls=cls.__name__), self.assertRaises(ValueError):
                cls(dimension=5).bind_dimension(6)

    def test_explicit_dimension_mismatch_with_v_raises(self) -> None:
        """Verify a mismatched dimension with ``v`` raises."""
        v = np.arange(5, dtype=np.float64)
        for cls in (SquaredL2, KLDivergence, Huber):
            with self.subTest(cls=cls.__name__), self.assertRaises(ValueError):
                cls(v=v, dimension=4)


class DistanceContributionTest(unittest.TestCase):
    """Each Distance returns a diagonal Hessian system for Problem assembly."""

    def test_hessian_contribution_matches_hessian_diagonal(self) -> None:
        """Verify the Hessian contribution matches the diagonal."""
        np.random.seed(3)
        w = np.random.rand(20) + 0.5  # strictly positive (KLDivergence needs w > 0)
        for metric in (
            SquaredL2(dimension=20),
            KLDivergence(dimension=20),
            Huber(dimension=20),
        ):
            with self.subTest(metric=type(metric).__name__):
                contribution = metric.hessian_contribution(w)
                assert isinstance(contribution, DiagonalSystem)  # Narrows for mypy
                np.testing.assert_allclose(contribution.eta, metric.hessian_diagonal(w))
