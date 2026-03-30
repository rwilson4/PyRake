"""Tests for quadratic program solvers."""

import time

import numpy as np
import pytest
from scipy import linalg, optimize

from pyrake.optimization import NewtonResult
from pyrake.optimization.quadratic_programs import (
    QuadraticNewtonSolver,
    QuadraticProgramEqualityBoundsSolver,
)


@pytest.mark.parametrize(
    "seed,n",
    [
        (101, 5),
        (201, 10),
        (301, 20),
        (401, 50),
        (501, 100),
    ],
)
def test_unconstrained_newton_quadratic(seed: int, n: int) -> None:
    """UnconstrainedNewtonSolver finds the exact minimizer of a quadratic in one step."""
    rng = np.random.default_rng(seed)

    # Q = A^T A + I ensures strict positive definiteness.
    A = rng.standard_normal((n, n))
    Q = A.T @ A + np.eye(n)
    c = rng.standard_normal(n)
    x0 = rng.standard_normal(n)

    x_star_expected = linalg.solve(Q, -c)

    solver = QuadraticNewtonSolver(Q=Q, c=c)
    result = solver.solve(x0=x0)

    assert isinstance(result, NewtonResult)
    np.testing.assert_allclose(result.solution, x_star_expected, rtol=1e-6, atol=1e-8)
    # Newton's method is exact for quadratics: one step reaches the optimum, then a
    # second iteration detects the near-zero Newton decrement and returns.
    assert result.nits == 2


class TestQuadraticProgramEqualityBoundsSolver:
    r"""Tests for QuadraticProgramEqualityBoundsSolver.

    Solves: minimize  x^T Q x + c^T x
            subject to  A x = b
                        x >= xl

    Uses scipy.optimize.minimize (SLSQP) as ground truth.
    """

    @pytest.mark.parametrize(
        "seed,n,p",
        [
            (1001, 10, 3),
            (2001, 20, 5),
            (3001, 50, 10),
            (4001, 5, 2),
            (5001, 30, 7),
        ],
    )
    def test_solver_interior_solution(self, seed: int, n: int, p: int) -> None:
        """Solver matches scipy when optimal solution is strictly interior."""
        rng = np.random.default_rng(seed)

        # PD Q: B^T B + I ensures strict positive definiteness.
        B = rng.standard_normal((n, n))
        Q = B.T @ B + np.eye(n)

        # Choose c so that the unconstrained minimum -(1/2) Q^{-1} c is well inside
        # the feasible region (all positive). Use c = Q @ ones so x_unc = -0.5 * ones
        # which is negative; the equality constraints will force a valid solution.
        c = rng.standard_normal(n)

        # Lower bounds: slightly negative so they are rarely active.
        xl = -0.5 * np.ones(n)

        # Feasible starting point: x0 > xl satisfying A x0 = b.
        A = rng.standard_normal((p, n))
        x0_feas = xl + np.abs(rng.standard_normal(n)) + 1.0
        b = A @ x0_feas

        # --- scipy ground truth (trust-constr handles equality + bounds robustly) ---
        t0 = time.perf_counter()
        scipy_result = optimize.minimize(
            fun=lambda x: float(x @ Q @ x + c @ x),
            x0=x0_feas,
            method="trust-constr",
            jac=lambda x: 2.0 * Q @ x + c,
            hess=lambda x: 2.0 * Q,
            constraints=optimize.LinearConstraint(A, b, b),
            bounds=optimize.Bounds(lb=xl),
            options={"gtol": 1e-10, "maxiter": 2000},
        )
        scipy_ms = 1000.0 * (time.perf_counter() - t0)
        if not scipy_result.success:
            pytest.skip(f"scipy reference solver failed: {scipy_result.message}")

        # --- our solver ---
        t0 = time.perf_counter()
        solver = QuadraticProgramEqualityBoundsSolver(Q=Q, c=c, A=A, b=b, xl=xl)
        result = solver.solve()
        pyrake_ms = 1000.0 * (time.perf_counter() - t0)

        print(
            f"\n  n={n:3d}, p={p:2d} | "
            f"scipy: {scipy_ms:7.2f} ms ({scipy_result.nit} iters) | "
            f"PyRake: {pyrake_ms:7.2f} ms ({result.nits} outer, "
            f"{sum(result.inner_nits)} inner Newton) | "
            f"ratio: {pyrake_ms / scipy_ms:.2f}x"
        )

        np.testing.assert_allclose(
            result.solution,
            scipy_result.x,
            rtol=1e-3,
            atol=1e-3,
            err_msg="Solver solution does not match scipy ground truth.",
        )

    @pytest.mark.parametrize(
        "seed,n,p",
        [
            (1002, 10, 3),
            (2002, 15, 4),
            (3002, 8, 2),
        ],
    )
    def test_equality_constraints_satisfied(self, seed: int, n: int, p: int) -> None:
        """Solution satisfies equality constraints A x = b to high precision."""
        rng = np.random.default_rng(seed)

        B = rng.standard_normal((n, n))
        Q = B.T @ B + np.eye(n)
        c = rng.standard_normal(n)
        xl = np.zeros(n)

        A = rng.standard_normal((p, n))
        x0_feas = xl + np.abs(rng.standard_normal(n)) + 1.0
        b = A @ x0_feas

        solver = QuadraticProgramEqualityBoundsSolver(Q=Q, c=c, A=A, b=b, xl=xl)
        result = solver.solve()

        np.testing.assert_allclose(
            A @ result.solution,
            b,
            atol=1e-5,
            err_msg="Equality constraints not satisfied.",
        )

    @pytest.mark.parametrize(
        "seed,n,p",
        [
            (1003, 10, 3),
            (2003, 15, 4),
            (3003, 8, 2),
        ],
    )
    def test_bound_constraints_satisfied(self, seed: int, n: int, p: int) -> None:
        """Solution satisfies bound constraints x >= xl."""
        rng = np.random.default_rng(seed)

        B = rng.standard_normal((n, n))
        Q = B.T @ B + np.eye(n)
        c = rng.standard_normal(n)
        xl = -0.5 * np.ones(n)

        A = rng.standard_normal((p, n))
        x0_feas = xl + np.abs(rng.standard_normal(n)) + 1.0
        b = A @ x0_feas

        solver = QuadraticProgramEqualityBoundsSolver(Q=Q, c=c, A=A, b=b, xl=xl)
        result = solver.solve()

        assert np.all(
            result.solution >= xl - 1e-6
        ), "Bound constraints violated: x < xl for some components."

    @pytest.mark.parametrize(
        "seed,n,p",
        [
            (1004, 10, 3),
            (2004, 20, 5),
        ],
    )
    def test_scalar_lower_bound(self, seed: int, n: int, p: int) -> None:
        """Solver works when xl is a scalar (applied to all components)."""
        rng = np.random.default_rng(seed)

        B = rng.standard_normal((n, n))
        Q = B.T @ B + np.eye(n)
        c = rng.standard_normal(n)
        xl_scalar = 0.0  # scalar lower bound

        A = rng.standard_normal((p, n))
        x0_feas = np.abs(rng.standard_normal(n)) + 1.0
        b = A @ x0_feas

        solver = QuadraticProgramEqualityBoundsSolver(Q=Q, c=c, A=A, b=b, xl=xl_scalar)
        result = solver.solve()

        # Equality constraints satisfied.
        np.testing.assert_allclose(A @ result.solution, b, atol=1e-5)
        # Bound constraints satisfied.
        assert np.all(result.solution >= xl_scalar - 1e-6)

    @pytest.mark.parametrize(
        "seed,n,p",
        [
            (1005, 10, 3),
            (2005, 20, 5),
        ],
    )
    def test_dual_is_lower_bound(self, seed: int, n: int, p: int) -> None:
        """Dual value is a lower bound on the primal objective."""
        rng = np.random.default_rng(seed)

        B = rng.standard_normal((n, n))
        Q = B.T @ B + np.eye(n)
        c = rng.standard_normal(n)
        xl = np.zeros(n)

        A = rng.standard_normal((p, n))
        x0_feas = np.abs(rng.standard_normal(n)) + 1.0
        b = A @ x0_feas

        solver = QuadraticProgramEqualityBoundsSolver(Q=Q, c=c, A=A, b=b, xl=xl)
        result = solver.solve()

        # Dual value <= primal objective (weak duality).
        assert (
            result.dual_value <= result.objective_value + 1e-6
        ), f"Dual {result.dual_value} > primal {result.objective_value}: weak duality violated."

    @pytest.mark.parametrize(
        "seed,n,p,rank",
        [
            (1006, 10, 3, 7),
            (2006, 15, 4, 10),
        ],
    )
    def test_psd_q(self, seed: int, n: int, p: int, rank: int) -> None:
        """Solver works when Q is PSD (rank-deficient) rather than PD."""
        rng = np.random.default_rng(seed)

        # Build a rank-deficient PSD Q: Q = B^T B where B is rank x n.
        B = rng.standard_normal((rank, n))
        Q = B.T @ B  # PSD, rank = rank < n

        c = rng.standard_normal(n)
        xl = np.zeros(n)

        A = rng.standard_normal((p, n))
        x0_feas = np.abs(rng.standard_normal(n)) + 1.0
        b = A @ x0_feas

        solver = QuadraticProgramEqualityBoundsSolver(Q=Q, c=c, A=A, b=b, xl=xl)
        result = solver.solve()

        np.testing.assert_allclose(A @ result.solution, b, atol=1e-5)
        assert np.all(result.solution >= xl - 1e-6)
