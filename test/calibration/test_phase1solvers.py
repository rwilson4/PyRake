"""Test Phase I solvers."""

import numpy as np
import pytest
from scipy import linalg

from pyrake.calibration.phase1solvers import EqualityWithBoundsAndNormConstraintSolver
from pyrake.optimization.optimization import (
    InteriorPointMethodResult,
    OptimizationSettings,
    ProblemCertifiablyInfeasibleError,
)

class TestEqualityWithBoundsAndNormConstraintSolver:
    """Test EqualityWithBoundsAndNormConstraintSolver."""

    @pytest.mark.parametrize(
        "seed,M,p",
        [
            (1401, 100, 20),
            (2401, 200, 30),
            (3401, 50, 5),
            (4401, 500, 100),
            (5401, 13, 3),
        ],
    )
    def test_solver_feasible(self, seed: int, M: int, p: int) -> None:
        """Test solver when problem is feasible."""
        np.random.seed(seed)
        phi = 1.0

        w = np.random.rand(M) + 0.5
        # Constrain to norm < phi
        w /= np.sqrt(np.dot(w, w))
        w /= 1.1 * phi
        assert np.dot(w, w) < phi
        lb = 0.01
        assert np.all(w > lb)

        A = np.random.randn(p, M)
        b = A @ w

        solver = EqualityWithBoundsAndNormConstraintSolver(
            A=A,
            b=b,
            lb=lb,
            phi=phi,
            settings=OptimizationSettings(verbose=True),
        )

        # Verify we find a feasible point
        res = solver.solve()
        np.testing.assert_allclose(A @ res.solution, b)
        assert np.all(res.solution > lb)
        assert np.dot(res.solution, res.solution) < phi

        # Verify we can fully optimize
        res = solver.solve(fully_optimize=True)
        np.testing.assert_allclose(A @ res.solution, b)
        assert np.all(res.solution > lb)
        assert np.dot(res.solution, res.solution) < phi

    @pytest.mark.parametrize(
        "seed,M,p",
        [
            (1402, 100, 20),
            (2402, 200, 30),
            (3402, 50, 5),
            (4402, 500, 100),
            (5402, 13, 3),
        ],
    )
    def test_solver_infeasible(self, seed: int, M: int, p: int) -> None:
        """Test solver when problem is infeasible."""
        np.random.seed(seed)

        w = np.random.rand(M)
        A = np.random.randn(M, M)
        b = A @ w
        phi = int(0.9 * np.sqrt(np.dot(w, w)))

        solver = EqualityWithBoundsAndNormConstraintSolver(
            A=A,
            b=b,
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

    @pytest.mark.parametrize(
        "seed,M,p",
        [
            (1403, 100, 20),
            (2403, 200, 30),
            (3403, 50, 5),
            (4403, 500, 100),
            (5403, 13, 3),
        ],
    )
    def test_solver_feasible_mean_constraint(self, seed: int, M: int, p: int) -> None:
        """Test solver when problem is feasible."""
        np.random.seed(seed)
        phi = 1.0
        A = (1 / M) * np.random.randn(M, p).T

        w = np.random.rand(M)
        # Constrain to norm < phi
        w /= np.sqrt(np.dot(w, w))
        w /= 1.1 * phi
        assert np.dot(w, w) < phi

        A = np.random.randn(p, M)
        # Add row of 1s to A
        A = np.vstack((A, np.ones((1, M))))
        b = A @ w

        solver = EqualityWithBoundsAndNormConstraintSolver(
            A=A,
            b=b,
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

    @pytest.mark.parametrize(
        "seed,M,p",
        [
            (1404, 100, 20),
            (2404, 200, 30),
            (3404, 50, 5),
            (4404, 500, 100),
            (5404, 13, 3),
        ],
    )
    def test_solver_feasible_rank_deficient(self, seed: int, M: int, p: int) -> None:
        """Test solver when problem is feasible."""
        np.random.seed(seed)
        phi = 1.0

        w = np.random.rand(M)
        # Constrain to norm < phi
        w /= np.sqrt(np.dot(w, w))
        w /= 1.1 * phi
        assert np.dot(w, w) < phi

        A = np.random.randn(p, M)
        U, s, Vh = linalg.svd(A, full_matrices=False)
        s[-1] = 0.0
        s[-2] = 0.0
        A = U @ np.diag(s) @ Vh

        b = A @ w

        solver = EqualityWithBoundsAndNormConstraintSolver(
            A=A,
            b=b,
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

    def test_solver_feasible_good_initial_guess(self) -> None:
        """Test solver when we pass a good initial guess."""
        seed = 1234
        M = 100
        p = 20
        np.random.seed(seed)
        phi = 1.0

        w = np.random.rand(M)
        # Constrain to norm < phi
        w /= np.sqrt(np.dot(w, w))
        w /= 1.1 * phi
        assert np.dot(w, w) < phi

        A = np.random.randn(p, M)
        b = A @ w

        # Solve it once, and then pass the solution as the initial guess on another
        # round.
        solver = EqualityWithBoundsAndNormConstraintSolver(
            A=A,
            b=b,
            phi=phi,
            settings=OptimizationSettings(verbose=True),
        )

        res = solver.solve(fully_optimize=True)
        w_star = res.solution
        np.testing.assert_allclose(A @ res.solution, b)
        assert np.all(res.solution > 0)
        assert np.dot(res.solution, res.solution) < phi

        solver = EqualityWithBoundsAndNormConstraintSolver(
            A=A,
            b=b,
            phi=phi,
            settings=OptimizationSettings(verbose=True),
        )
        res = solver.solve(x0=w_star, fully_optimize=True)
        assert isinstance(res, InteriorPointMethodResult)
        assert res.nits == 1

    @pytest.mark.parametrize(
        "seed,M,p1,p2",
        [
            (1405, 100, 10, 15),
            (2405, 200, 5, 30),
            (3405, 50, 5, 5),
            (4405, 500, 20, 30),
            (5405, 13, 3, 3),
        ],
    )
    def test_solver_max_imbalance_constraint(
        self, seed: int, M: int, p1: int, p2: int
    ) -> None:
        """Test solver with a constraint on max covariate imbalance."""
        np.random.seed(seed)
        A = np.random.randn(p1, M)
        B = np.random.randn(p2, M)

        # To generate population mean, simulate true propensity scores with mean 0.1 and
        # variance 0.0045
        q = 0.1
        sigma2 = 0.05 * q * (1 - q)
        s = q * (1 - q) / sigma2 - 1
        alpha = q * s
        beta = (1 - q) * s
        true_propensity = np.random.beta(alpha, beta, size=M)

        # Ideal weights are (M/N) / true_propensity, but to make it simple, do:
        w = 1.0 / true_propensity
        w /= np.mean(w)
        lb = 0.01
        assert np.all(w > lb)

        b = A @ w
        # Add a constraint on the mean weights
        A[0, :] = 1.0 / M
        b[0] = 1.0
        c = B @ w  # + 0.2 * np.random.rand(p2) - 0.1
        psi = 0.05
        phi = np.dot(w, w)

        solver = EqualityWithBoundsAndNormConstraintSolver(
            phi=phi,
            A=A,
            b=b,
            lb=lb,
            B=B,
            c=c,
            psi=psi,
            settings=OptimizationSettings(verbose=True),
        )

        # Verify we find a feasible point
        res = solver.solve()
        np.testing.assert_allclose(A @ res.solution, b)
        assert np.all(res.solution > lb)
        assert np.all(B @ res.solution - c > -psi)
        assert np.all(B @ res.solution - c < psi)
        assert np.dot(res.solution, res.solution) < phi

        # Verify we can fully optimize
        res = solver.solve(fully_optimize=True)
        np.testing.assert_allclose(A @ res.solution, b)
        assert np.all(res.solution > 0)
        assert np.all(B @ res.solution - c > -psi)
        assert np.all(B @ res.solution - c < psi)
        assert np.dot(res.solution, res.solution) < phi

        # Test with vector of lb and psi
        solver = EqualityWithBoundsAndNormConstraintSolver(
            phi=phi,
            A=A,
            b=b,
            lb=np.full((M,), lb),
            B=B,
            c=c,
            psi=np.full((p2,), psi),
            settings=OptimizationSettings(verbose=True),
        )

        # Verify we find a feasible point
        res = solver.solve()
        np.testing.assert_allclose(A @ res.solution, b)
        assert np.all(res.solution > lb)
        assert np.all(B @ res.solution - c > -psi)
        assert np.all(B @ res.solution - c < psi)
        assert np.dot(res.solution, res.solution) < phi
