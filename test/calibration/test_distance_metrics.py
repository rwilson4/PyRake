"""Test distance metrics."""

import unittest

import numpy as np
from cvxium import DiagonalSystem

from pyrake.calibration.distance_metrics import (
    Huber,
    KLDivergence,
    SquaredL2,
)


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
