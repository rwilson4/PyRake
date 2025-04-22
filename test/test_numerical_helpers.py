"""Test numerical helpers."""

import time
import numpy as np
from pyrake.numerical_helpers import (
    solve_diagonal_plus_rank_one,
    solve_kkt_system_hessian_diagonal_plus_rank_one,
)


def test_solve_diagonal_plus_rank_one():
    """Test solving H*x = b by doing it the slow way."""
    np.random.seed(42)
    M = 100
    eta = np.random.rand(M) + 1.0
    zeta = np.random.randn(M)
    H = np.diag(eta) + np.outer(zeta, zeta)
    b = np.random.randn(M)

    st = time.time()
    x_expected = np.linalg.solve(H, b)
    mt = time.time()
    x = solve_diagonal_plus_rank_one(eta, zeta, b)
    et = time.time()

    print(f"Slow way completed in {1e6*(mt-st):.03f} us")
    print(f"Fast way completed in {1e6*(et-mt):.03f} us")

    np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)

    # Verify H*x = b
    np.testing.assert_allclose(H.dot(x), b, rtol=1e-8, atol=1e-8)


def test_solve_kkt_system_hessian_diagonal_plus_rank_one():
    """Test solving KKT system and checking answer."""
    np.random.seed(123)
    M = 80
    p = 5
    eta = np.random.rand(M) + 1.0
    zeta = np.random.randn(M)
    H = np.diag(eta) + np.outer(zeta, zeta)
    A = np.random.randn(p, M)
    g = np.random.randn(M)

    KKT = np.zeros((M + p, M + p))
    KKT[0:M, 0:M] = H
    KKT[0:M, M:] = A.T
    KKT[M:, 0:M] = A
    rhs = np.zeros((M + p,))
    rhs[0:M] = g

    st = time.time()
    x_expected = np.linalg.solve(KKT, rhs)
    mt = time.time()
    delta_w_expected = x_expected[0:M]
    xi_expected = x_expected[M:]

    delta_w, xi = solve_kkt_system_hessian_diagonal_plus_rank_one(
        A, g, eta, zeta
    )
    et = time.time()

    print(f"Slow way completed in {1e6*(mt-st):.03f} us")
    print(f"Fast way completed in {1e6*(et-mt):.03f} us")

    np.testing.assert_allclose(delta_w, delta_w_expected, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(xi, xi_expected, rtol=1e-8, atol=1e-8)

    lhs_top = H.dot(delta_w) + A.T.dot(xi)
    rhs_top = g
    lhs_bot = A.dot(delta_w)
    rhs_bot = np.zeros(p)

    # Verify H * delta_w + A^T * xi = g
    np.testing.assert_allclose(lhs_top, rhs_top, rtol=1e-8, atol=1e-8)

    # Verify A * delta_w = 0
    np.testing.assert_allclose(lhs_bot, rhs_bot, rtol=1e-8, atol=1e-8)
