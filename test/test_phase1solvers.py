"""Test Phase I Solvers."""

import numpy as np
import pytest
from scipy import linalg

from pyrake.exceptions import InteriorPointMethodError, ProblemInfeasibleError
from pyrake.optimization import (
    InteriorPointMethodResult,
    OptimizationSettings,
    ProblemCertifiablyInfeasibleError,
    ProblemMarginallyFeasibleError,
)
from pyrake.phase1solvers import (
    EqualitySolver,
    EqualityWithBoundsSolver,
    EqualityWithBoundsAndNormConstraintSolver,
)


class TestEqualitySolver:
    """Test EqualitySolver."""

    @staticmethod
    def test_solver() -> None:
        """Test solver."""
        np.random.seed(1)
        M, p = 100, 20
        A = np.random.randn(p, M)
        w = np.random.randn(M)
        b = A @ w

        solver = EqualitySolver(A=A, b=b)

        w = solver.solve().solution
        np.testing.assert_allclose(A @ w, b)

    @staticmethod
    def test_solver_not_full_rank() -> None:
        """Test solver."""
        np.random.seed(2)
        M, p = 100, 20
        A = np.random.randn(p, M)
        w = np.random.randn(M)

        U, s, Vh = linalg.svd(A, full_matrices=False)
        s[-1] = 0.0
        s[-2] = 0.0
        A = U @ np.diag(s) @ Vh

        b = A @ w

        solver = EqualitySolver(A=A, b=b)

        w = solver.solve().solution
        np.testing.assert_allclose(A @ w, b)

    @staticmethod
    def test_solver_infeasible() -> None:
        """Test solver."""
        np.random.seed(3)
        M, p = 100, 20
        A = np.random.randn(p, M)
        w = np.random.randn(M)

        U, s, Vh = linalg.svd(A, full_matrices=False)
        s[-1] = 0.0
        s[-2] = 0.0
        A = U @ np.diag(s) @ Vh

        b = A @ w + np.random.randn(p)

        solver = EqualitySolver(A=A, b=b)

        with pytest.raises(ProblemInfeasibleError):
            solver.solve().solution


class TestEqualityWithBoundsSolver:
    """Test EqualityWithBoundsSolver."""

    @staticmethod
    def test_solver_feasible_bounded_below() -> None:
        """Test solver when problem is feasible and the augmented problem bounded below.

        The dual function for the augmented problem gives a lower bound for the problem
        value. The dual function is finite if there exists a collection nu (length p),
        lambda1 (length M), lambda2 (scalar), such that:
           lambda1 = A^T nu
           lambda2 = sum(lambda1) - 1
           lambda1 >= 0
           lambda2 >= 0.

        And if there exists w such that A * w = b, then the problem is feasible.

        """
        np.random.seed(11)
        M, p = 100, 20

        nu = np.random.randn(p)
        # Construct A such that lambda1 := A^T nu >= 0
        A = np.random.randn(p, M)
        for ic in range(M):
            if np.dot(A[:, ic], nu) < 0:
                A[:, ic] = -A[:, ic]

        assert np.all(A.T @ nu >= 0)

        # Scale A such that lambda2 := sum(A^T nu) - 1 > 0
        A *= 1.1 / np.sum(A.T @ nu)
        assert np.sum(A.T @ nu) > 1

        # Construct feasible w
        w = 0.1 + np.random.rand(M)
        b = A @ w

        solver = EqualityWithBoundsSolver(
            phase1_solver=EqualitySolver(
                A=A, b=b, settings=OptimizationSettings(verbose=True)
            ),
            settings=OptimizationSettings(verbose=True, outer_tolerance=0.01),
        )

        # Verify we find a feasible point
        res = solver.solve()
        np.testing.assert_allclose(A @ res.solution, b)
        assert np.all(res.solution > 0)

        # Since augmented problem is bounded below, verify we can fully solve it (if for
        # whatever reason we wanted to).
        res = solver.solve(fully_optimize=True)
        np.testing.assert_allclose(A @ res.solution, b)
        assert np.all(res.solution > 0)

    @staticmethod
    def test_solver_feasible_unbounded_below() -> None:
        """Test solver when problem is feasible and the augmented problem is unbounded below.

        The augmented problem is unbounded below (yet is still feasible) if there does
        not exist a collection nu (length p), lambda1 (length M), lambda2 (scalar), such
        that:
           lambda1 = A^T nu
           lambda2 = sum(lambda1) - 1
           lambda1 >= 0
           lambda2 >= 0.

        One simple example is when A is a 1-by-M matrix with at least one strictly
        positive entry and at least one strictly negative entry. Then nu is just a
        scalar, and if nu is:
           positive: at least one entry in lambda1 is negative
           negative: at least one entry in lambda1 is negative
           0: lambda1 = 0 so lambda2 = -1.

        """
        np.random.seed(12)
        M, p = 100, 1

        A = np.random.randn(p, M)
        A[0, 0] = -abs(A[0, 0])
        A[0, 1] = abs(A[0, 1])

        # Generate feasible w
        w = np.random.rand(M)
        b = A @ w

        solver = EqualityWithBoundsSolver(
            phase1_solver=EqualitySolver(
                A=A, b=b, settings=OptimizationSettings(verbose=True)
            ),
            settings=OptimizationSettings(verbose=True),
        )

        # Verify we find a feasible point
        res = solver.solve()
        np.testing.assert_allclose(A @ res.solution, b)
        assert np.all(res.solution > 0)

        # See what happens if we try to fully optimize
        with pytest.raises(InteriorPointMethodError):
            solver.solve(fully_optimize=True)

    @staticmethod
    def test_solver_infeasible() -> None:
        """Test solver when problem is infeasible.

        We seek an x satisfying A * x = b and x > 0. No such x exists if there exists a
        nu satisfying:
            A^T * nu >= 0,
            c^T * nu = 1, where c = A * 1, and
            b^T nu < 0.
        This test constructs such an A, b and verifies the solver detects this
        infeasibility.

        """
        np.random.seed(13)
        M, p = 100, 20
        nu = np.random.randn(p)

        # Construct A such that A^T nu >= 0
        A = np.random.randn(p, M)
        for ic in range(M):
            if np.dot(A[:, ic], nu) < 0:
                A[:, ic] = -A[:, ic]

        assert np.all(A.T @ nu >= 0)

        # Scale A such that c^T * nu = 1
        A /= np.dot(A @ np.ones((M,)), nu)
        assert np.all(A.T @ nu >= 0)
        assert abs(np.dot(A @ np.ones((M,)), nu) - 1) <= 1e-6

        # Construct b such that b^T nu < 0.
        b = np.random.randn(p)
        alpha = np.dot(nu, b) / np.dot(nu, nu)
        b = b - (alpha + 0.1) * nu
        assert np.all(A.T @ nu >= 0)
        assert abs(np.dot(A @ np.ones((M,)), nu) - 1) <= 1e-6
        assert np.dot(b, nu) < 0

        solver = EqualityWithBoundsSolver(
            phase1_solver=EqualitySolver(
                A=A, b=b, settings=OptimizationSettings(verbose=True)
            ),
            settings=OptimizationSettings(verbose=True, outer_tolerance=2e-3),
        )

        with pytest.raises(ProblemCertifiablyInfeasibleError):
            solver.solve()

        # Since augmented problem is bounded below, verify we can fully solve it (if for
        # whatever reason we wanted to).
        res = solver.solve(fully_optimize=True)
        assert isinstance(res, InteriorPointMethodResult)
        assert res.dual_value > 0

    @staticmethod
    def test_solver_marginally_feasible() -> None:
        """Test solver when problem is marginally feasible.

        Here we test a scenario where there exists a technically feasible w, but where
        strict feasibility does not hold. In this case, the dual function will keep
        returning negative values, so we cannot certify the problem as infeasible, but
        we won't find a strictly feasible w (because it doesn't exist).

        """
        np.random.seed(14)
        M = 100

        w_star = np.random.rand(M)
        w_star[0] = 0
        # Generate a random orthonormal matrix
        A, _, _ = linalg.svd(np.random.randn(M, M))
        b = A @ w_star
        # w_star is the only solution to A * w = b, but w_star is only marginally feasible.

        solver = EqualityWithBoundsSolver(
            phase1_solver=EqualitySolver(
                A=A, b=b, settings=OptimizationSettings(verbose=True)
            ),
            settings=OptimizationSettings(verbose=True, outer_tolerance=1e-2),
        )

        with pytest.raises(ProblemMarginallyFeasibleError):
            solver.solve()


class TestEqualityWithBoundsAndNormConstraintSolver:
    """Test EqualityWithBoundsAndNormConstraintSolver."""

    @staticmethod
    def test_solver_feasible() -> None:
        """Test solver when problem is feasible."""
        np.random.seed(21)
        M, p = 100, 20
        phi = 1.0

        w = np.random.rand(M)
        # Constrain to norm < phi
        w /= np.sqrt(np.dot(w, w))
        w /= 1.1 * phi
        assert np.dot(w, w) < phi

        A = np.random.randn(p, M)
        b = A @ w

        solver = EqualityWithBoundsAndNormConstraintSolver(
            phase1_solver=EqualityWithBoundsSolver(
                phase1_solver=EqualitySolver(
                    A=A, b=b, settings=OptimizationSettings(verbose=True)
                ),
                settings=OptimizationSettings(verbose=True),
            ),
            phi=phi,
            settings=OptimizationSettings(verbose=True),
        )

        # Verify we find a feasible point
        res = solver.solve()
        np.testing.assert_allclose(A @ res.solution, b)
        assert np.all(res.solution > 0)
        assert np.dot(res.solution, res.solution) < phi

        # Verify we can fully optimize
        res = solver.solve(fully_optimize=True)
        np.testing.assert_allclose(A @ res.solution, b)
        assert np.all(res.solution > 0)
        assert np.dot(res.solution, res.solution) < phi

    @staticmethod
    def test_solver_infeasible() -> None:
        """Test solver when problem is infeasible."""
        np.random.seed(21)
        M = 100

        w = np.random.rand(M)
        A = np.random.randn(M, M)
        b = A @ w
        phi = int(0.9 * np.sqrt(np.dot(w, w)))

        solver = EqualityWithBoundsAndNormConstraintSolver(
            phase1_solver=EqualityWithBoundsSolver(
                phase1_solver=EqualitySolver(
                    A=A, b=b, settings=OptimizationSettings(verbose=True)
                ),
                settings=OptimizationSettings(verbose=True),
            ),
            phi=phi,
            settings=OptimizationSettings(verbose=True),
        )

        with pytest.raises(ProblemCertifiablyInfeasibleError):
            res = solver.solve()

        # Verify we can fully optimize
        res = solver.solve(fully_optimize=True)
        np.testing.assert_allclose(A @ res.solution, b)
        assert np.all(res.solution > 0)
        assert np.dot(w, w) > phi
