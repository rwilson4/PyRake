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

    @pytest.mark.parametrize(
        "seed,M,p",
        [
            (1101, 100, 20),
            (2101, 200, 30),
            (3101, 50, 5),
            (4101, 500, 100),
            (5101, 13, 3),
        ],
    )
    def test_solver(self, seed: int, M: int, p: int) -> None:
        """Test solver."""
        np.random.seed(seed)
        A = np.random.randn(p, M)
        w = np.random.randn(M)
        b = A @ w

        solver = EqualitySolver(A=A, b=b)

        w = solver.solve().solution
        np.testing.assert_allclose(A @ w, b)

    @pytest.mark.parametrize(
        "seed,M,p",
        [
            (1102, 100, 20),
            (2102, 200, 30),
            (3102, 50, 5),
            (4102, 500, 100),
            (5102, 13, 3),
        ],
    )
    def test_solver_rank_deficient(self, seed: int, M: int, p: int) -> None:
        """Test solver."""
        np.random.seed(seed)
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

    @pytest.mark.parametrize(
        "seed,M,p",
        [
            (1103, 100, 20),
            (2103, 200, 30),
            (3103, 50, 5),
            (4103, 500, 100),
            (5103, 13, 3),
        ],
    )
    def test_solver_infeasible(self, seed: int, M: int, p: int) -> None:
        """Test solver."""
        np.random.seed(seed)
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

    @pytest.mark.parametrize(
        "seed,M,p",
        [
            (1104, 100, 20),
            (2104, 200, 30),
            (3104, 50, 5),
            (4104, 500, 100),
            (5104, 13, 3),
        ],
    )
    def test_solver_with_mean_constraint(self, seed: int, M: int, p: int) -> None:
        """Test solver with an extra row of all 1s in A."""
        np.random.seed(seed)
        A = np.random.randn(p, M)
        # Add a row to A of all ones.
        A = np.vstack((A, np.ones((1, M))))
        w = np.random.randn(M)
        b = A @ w

        solver = EqualitySolver(A=A, b=b)

        w = solver.solve().solution
        np.testing.assert_allclose(A @ w, b)

    @pytest.mark.parametrize(
        "seed,M,p",
        [
            (1105, 100, 20),
            (2105, 200, 30),
            (3105, 50, 5),
            (4105, 500, 100),
            (5105, 13, 3),
        ],
    )
    def test_solver_with_x0(self, seed: int, M: int, p: int) -> None:
        """Test solver with x0."""
        np.random.seed(seed)
        A = np.random.randn(p, M)
        w = np.random.randn(M)

        # Use a rank deficient A
        U, s, Vh = linalg.svd(A, full_matrices=False)
        s[-1] = 0.0
        s[-2] = 0.0
        A = U @ np.diag(s) @ Vh

        b = A @ w

        solver = EqualitySolver(A=A, b=b)
        # Give a set of weights close to the true w as a starting point.
        w0 = w + 0.01 * np.random.randn(M)
        w = solver.solve(x0=w0).solution

        # Verify w is feasible.
        np.testing.assert_allclose(A @ w, b)

        # Check that w is close to w0
        dist = np.dot(w - w0, w - w0)

        # Solve without passing w0
        w = solver.solve().solution

        # Verify the w we calculated when passing w0, is closer to w0 than the w we
        # calculate when not passing w0.
        assert dist < np.dot(w - w0, w - w0)


class TestEqualityWithBoundsSolver:
    """Test EqualityWithBoundsSolver."""

    @pytest.mark.parametrize(
        "seed,M,p,atol",
        [
            (1201, 100, 20, 1e-9),
            (2201, 200, 30, 1e-9),
            (3201, 50, 5, 1e-9),
            (4201, 500, 100, 1e-9),
            (5201, 13, 3, 1e-9),
        ],
    )
    def test_solver_feasible_bounded_below(
        self, seed: int, M: int, p: int, atol: float
    ) -> None:
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
        np.random.seed(seed)

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
            settings=OptimizationSettings(verbose=True),
        )

        # Verify we find a feasible point
        res = solver.solve()
        np.testing.assert_allclose(A @ res.solution, b)
        assert np.all(res.solution > 0)

        # Since augmented problem is bounded below, verify we can fully solve it (if for
        # whatever reason we wanted to).
        res = solver.solve(fully_optimize=True)
        np.testing.assert_allclose(A @ res.solution, b, atol=atol)
        assert np.all(res.solution > 0)

    @pytest.mark.parametrize(
        "seed,M,p",
        [
            (1202, 100, 20),
            (2202, 200, 30),
            (3202, 50, 5),
            (4202, 500, 100),
            (5202, 13, 3),
        ],
    )
    def test_solver_feasible_unbounded_below(self, seed: int, M: int, p: int) -> None:
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
        np.random.seed(seed)
        p = 1

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

    @pytest.mark.parametrize(
        "seed,M,p",
        [
            (1203, 100, 20),
            (2203, 200, 30),
            (3203, 50, 5),
            (4203, 500, 100),
            (5203, 13, 3),
        ],
    )
    def test_solver_infeasible(self, seed: int, M: int, p: int) -> None:
        """Test solver when problem is infeasible.

        We seek an x satisfying A * x = b and x > 0. No such x exists if there exists a
        nu satisfying:
            A^T * nu >= 0,
            c^T * nu = 1, where c = A * 1, and
            b^T nu < 0.
        This test constructs such an A, b and verifies the solver detects this
        infeasibility.

        """
        np.random.seed(seed)
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
            settings=OptimizationSettings(verbose=True),
        )

        with pytest.raises(ProblemCertifiablyInfeasibleError):
            solver.solve()

        # Since augmented problem is bounded below, verify we can fully solve it (if for
        # whatever reason we wanted to).
        res = solver.solve(fully_optimize=True)
        assert isinstance(res, InteriorPointMethodResult)
        assert res.dual_value > 0

    @pytest.mark.parametrize(
        "seed,M,p",
        [
            (1205, 100, 20),
            (2205, 200, 30),
            (3205, 50, 5),
            (4205, 500, 100),
            (5205, 13, 3),
        ],
    )
    def test_solver_marginally_feasible(self, seed: int, M: int, p: int) -> None:
        """Test solver when problem is marginally feasible.

        Here we test a scenario where there exists a technically feasible w, but where
        strict feasibility does not hold. In this case, the dual function will keep
        returning negative values, so we cannot certify the problem as infeasible, but
        we won't find a strictly feasible w (because it doesn't exist).

        """
        np.random.seed(seed)

        w_star = np.random.rand(M)

        # This is a little hacky. We're basically counting on the method to not certify
        # the problem as infeasible.
        w_star[0] = -1e-8

        # Generate a random orthonormal matrix
        A, _, _ = linalg.svd(np.random.randn(M, M))
        b = A @ w_star
        # w_star is the only solution to A * w = b, but w_star is only marginally feasible.

        solver = EqualityWithBoundsSolver(
            phase1_solver=EqualitySolver(
                A=A, b=b, settings=OptimizationSettings(verbose=True)
            ),
            settings=OptimizationSettings(verbose=True),
        )

        with pytest.raises(ProblemMarginallyFeasibleError):
            solver.solve()

    @pytest.mark.parametrize(
        "seed,M,p",
        [
            (1206, 100, 20),
            (2206, 200, 30),
            (3206, 50, 5),
            (4206, 500, 100),
            (5206, 13, 3),
        ],
    )
    def test_solver_feasible_mean_constraint(self, seed: int, M: int, p: int) -> None:
        """Test solver with an extra row of all 1s in A."""
        np.random.seed(seed)

        nu = np.random.randn(p + 1)
        # Construct A such that lambda1 := A^T nu[:p] >= 0
        A = np.random.randn(p, M)

        for ic in range(M):
            if np.dot(A[:, ic], nu[:p]) < 0:
                A[:, ic] = -A[:, ic]

        # Scale A such that lambda2 := sum(A^T nu) - 1 > 0
        A *= 1.1 / np.sum(A.T @ nu[:p])

        # Add a row to A of all ones. So long as the last element of nu is positive,
        # we'll still have have [A^T 1] * nu = A^T nu[:p] + nu[p] * 1 >= A^T nu[:p] >= 0
        # and sum([A^T 1] * nu) >= sum(A^T nu[:p]) > 1
        A = np.vstack((A, np.ones((1, M))))
        if nu[p] < 0:
            nu[p] = -nu[p]

        assert np.all(A.T @ nu >= 0)
        assert np.sum(A.T @ nu) > 1

        # Construct feasible w
        w = 0.1 + np.random.rand(M)
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

        # Since augmented problem is bounded below, verify we can fully solve it (if for
        # whatever reason we wanted to).
        res = solver.solve(fully_optimize=True)
        np.testing.assert_allclose(A @ res.solution, b, atol=1e-10)
        assert np.all(res.solution > 0)

    @pytest.mark.parametrize(
        "seed,M,p,atol",
        [
            (1207, 100, 20, 1e-9),
            (2207, 200, 30, 1e-9),
            (3207, 50, 5, 1e-7),
            (4207, 500, 100, 1e-4),
            (5207, 13, 3, 1e-9),
        ],
    )
    def test_solver_feasible_bounded_below_rank_deficient(
        self, seed: int, M: int, p: int, atol: float
    ) -> None:
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
        np.random.seed(seed)

        nu = np.random.randn(p)
        # Construct A such that lambda1 := A^T nu >= 0
        A = np.random.randn(p, M)
        U, s, Vh = linalg.svd(A, full_matrices=False)
        s[-1] = 0.0
        s[-2] = 0.0
        A = U @ np.diag(s) @ Vh
        for ic in range(M):
            if np.dot(A[:, ic], nu) < 0:
                A[:, ic] = -A[:, ic]

        assert np.all(A.T @ nu >= 0)

        # Scale A such that lambda2 := sum(A^T nu) - 1 > 0
        A *= 1.1 / np.sum(A.T @ nu)
        assert np.sum(A.T @ nu) > 1

        U, s, Vh = linalg.svd(A, full_matrices=False)
        assert s[-1] < 1e-10
        assert s[-2] < 1e-10

        # Construct feasible w
        w = 0.1 + np.random.rand(M)
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

        # Since augmented problem is bounded below, verify we can fully solve it (if for
        # whatever reason we wanted to).
        res = solver.solve(fully_optimize=True)
        np.testing.assert_allclose(A @ res.solution, b, atol=atol)
        assert np.all(res.solution > 0)


class TestEqualityWithBoundsAndNormConstraintSolver:
    """Test EqualityWithBoundsAndNormConstraintSolver."""

    @pytest.mark.parametrize(
        "seed,M,p",
        [
            (1301, 100, 20),
            (2301, 200, 30),
            (3301, 50, 5),
            (4301, 500, 100),
            (5301, 13, 3),
        ],
    )
    def test_solver_feasible(self, seed: int, M: int, p: int) -> None:
        """Test solver when problem is feasible."""
        np.random.seed(seed)
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

    @pytest.mark.parametrize(
        "seed,M,p",
        [
            (1302, 100, 20),
            (2302, 200, 30),
            (3302, 50, 5),
            (4302, 500, 100),
            (5302, 13, 3),
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

    @pytest.mark.parametrize(
        "seed,M,p",
        [
            (1303, 100, 20),
            (2303, 200, 30),
            (3303, 50, 5),
            (4303, 500, 100),
            (5303, 13, 3),
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

    @pytest.mark.parametrize(
        "seed,M,p",
        [
            (1304, 100, 20),
            (2304, 200, 30),
            (3304, 50, 5),
            (4304, 500, 100),
            (5304, 13, 3),
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
            phase1_solver=EqualityWithBoundsSolver(
                phase1_solver=EqualitySolver(
                    A=A, b=b, settings=OptimizationSettings(verbose=True)
                ),
                settings=OptimizationSettings(verbose=True),
            ),
            phi=phi,
            settings=OptimizationSettings(verbose=True),
        )

        res = solver.solve(fully_optimize=True)
        w_star = res.solution
        np.testing.assert_allclose(A @ res.solution, b)
        assert np.all(res.solution > 0)
        assert np.dot(res.solution, res.solution) < phi

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
        res = solver.solve(x0=w_star, fully_optimize=True)
        assert res.nits == 1
