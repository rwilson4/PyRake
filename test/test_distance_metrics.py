"""Test distance metrics."""

import numpy as np
from pyrake.distance_metrics import SquaredL2, KLDivergence, Huber


def test_squared_l2() -> None:
    """Test SquaredL2."""
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


def test_kl() -> None:
    """Test KLDivergence."""
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


def test_huber() -> None:
    """Test Huber."""
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
