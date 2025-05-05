"""Test Phase I Solvers."""

import time

import numpy as np
import pytest
from scipy import linalg

from pyrake.exceptions import ProblemInfeasibleError
from pyrake.optimization import OptimizationSettings
from pyrake.phase1solvers import (
    EqualitySolver,
    EqualityWithBoundsSolver,
)


class TestEqualitySolver:
    """Test EqualitySolver."""

    @staticmethod
    def test_solver() -> None:
        """Test solver."""
        np.random.seed(42)
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
        np.random.seed(42)
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
        np.random.seed(42)
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
    def test_solver() -> None:
        """Test solver."""
        np.random.seed(42)
        M, p = 1000, 23
        A = np.random.randn(p, M)
        w = np.random.rand(M)
        b = A @ w

        solver = EqualityWithBoundsSolver(
            phase1_solver=EqualitySolver(
                A=A, b=b, settings=OptimizationSettings(verbose=True)
            ),
            settings=OptimizationSettings(verbose=True),
        )

        w = solver.solve().solution
        np.testing.assert_allclose(A @ w, b)
        assert np.all(w > 0)
