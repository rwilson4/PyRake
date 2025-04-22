"""Test distance metrics."""

import numpy as np
from pyrake.distance_metrics import SquaredL2, KLDivergence, Huber


def test_squared_l2():
    """Test SquaredL2."""
    np.random.seed(0)
    w = np.random.rand(100)
    v = np.random.rand(100)
    metric = SquaredL2(v=v)

    val = metric.evaluate(w)
    expected_val = np.sum((w - v) ** 2)
    np.testing.assert_allclose(val, expected_val)

    grad = metric.gradient(w)
    expected_grad = 2 * (w - v)
    np.testing.assert_allclose(grad, expected_grad)

    hess_diag = metric.hessian_diagonal(w)
    expected_hess = 2 * np.ones_like(w)
    np.testing.assert_allclose(hess_diag, expected_hess)


def test_kl():
    """Test KLDivergence."""
    np.random.seed(1)
    w = np.random.rand(100) + 0.1  # ensure strictly positive
    v = np.random.rand(100) + 0.1
    metric = KLDivergence(v=v)

    val = metric.evaluate(w)
    expected_val = np.sum(w * np.log(w / v) - w + v)
    np.testing.assert_allclose(val, expected_val)

    grad = metric.gradient(w)
    expected_grad = np.log(w / v)
    np.testing.assert_allclose(grad, expected_grad)

    hess_diag = metric.hessian_diagonal(w)
    expected_hess = 1.0 / w
    np.testing.assert_allclose(hess_diag, expected_hess)


def test_huber():
    """Test Huber."""
    np.random.seed(2)
    w = np.random.rand(100)
    v = np.random.rand(100)
    delta = 0.1
    metric = Huber(v=v, delta=delta)

    d = w - v
    val = metric.evaluate(w)
    expected_val = np.sum(
        np.where(np.abs(d) <= delta, d**2, delta * (2 * np.abs(d) - delta))
    )
    np.testing.assert_allclose(val, expected_val)

    grad = metric.gradient(w)
    expected_grad = np.where(
        np.abs(d) <= delta, 2.0 * d, 2.0 * delta * np.sign(d)
    )
    np.testing.assert_allclose(grad, expected_grad)

    hess_diag = metric.hessian_diagonal(w)
    expected_hess = np.where(np.abs(d) <= delta, 2.0, 0.0)
    np.testing.assert_allclose(hess_diag, expected_hess)
