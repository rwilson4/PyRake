"""Test Rake."""

import time
import unittest
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
from cvxium import (
    BarrierDiagnostics,
    ObservabilitySettings,
    OptimizationSettings,
    OuterSettings,
    Problem,
    ProblemCertifiablyInfeasibleError,
    ProblemInfeasibleError,
    SolveResult,
    SolveStatus,
)
from parameterized import parameterized
from scipy.special import expit, logit

from pyrake._typing import none_throws
from pyrake.calibration.distance_metrics import (
    Distance,
    Huber,
    KLDivergence,
    SquaredL2,
)
from pyrake.calibration.joint_calibrator import (
    FallbackOptimizationResult,
    JointCalibrator,
)
from pyrake.calibration.rake import Rake


def _barrier_diag(res: SolveResult) -> BarrierDiagnostics:
    """Narrow a barrier (ipm) solve's diagnostics for the convergence assertions."""
    assert isinstance(res.diagnostics, BarrierDiagnostics)
    return res.diagnostics


class RakeSolveTest(unittest.TestCase):
    """Tests for ``Rake.solve()`` across distance metrics and constraint shapes."""

    def test_bare_distance_binds_dimension_and_solves(self) -> None:
        """Verify a bare distance binds dimension and solves."""
        # Regression: a bare ``KLDivergence()`` (no ``v``/``dimension``) used as a
        # Rake objective must keep working -- Rake binds its own ``M`` so the
        # distance can serve as the ``Problem`` objective (the pattern used by
        # ``surveys.py`` JointCalibrator callers that build the distance bare).
        np.random.seed(11)
        M, p = 40, 5
        X = np.random.rand(M, p)
        mu = (1.0 / M) * (X.T @ np.ones(M))
        distance = KLDivergence()
        rake = Rake(distance=distance, X=X, mu=mu, min_weight=0.01)
        self.assertEqual(distance.dimension, M)
        res = rake.solve()
        np.testing.assert_allclose(X.T @ res.solution / M, mu)

    @parameterized.expand(
        [
            ("M100_p20_no_phi", 101, 100, 20, None),
            ("M200_p30_phi1.5", 201, 200, 30, 1.5),
            ("M50_p5_phi1.5", 301, 50, 5, 1.5),
            ("M500_p100_no_phi", 401, 500, 100, None),
            ("M13_p3_phi1.5", 501, 13, 3, 1.5),
        ]
    )
    def test_solve_kl_divergence(
        self, name: str, seed: int, M: int, p: int, phi: float
    ) -> None:
        """Test problems with KL Divergence as objective."""
        np.random.seed(seed)
        X = np.random.rand(M, p)

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
        min_weight = 0.01
        np.testing.assert_array_less(min_weight, w)

        # Compute population mean
        mu: npt.NDArray[np.float64] = (1 / M) * (X.T @ w)

        # Now forget we know w. We just know X for respondents and mu for the target
        # population.
        start_time = time.time()
        rake = Rake(
            distance=KLDivergence(dimension=M),
            X=X,
            mu=mu,
            phi=phi,
            min_weight=min_weight,
            settings=OptimizationSettings(
                observability=ObservabilitySettings(verbose=True)
            ),
        )
        res = rake.solve()
        end_time = time.time()
        print(f"Complete in {1e3 * (end_time - start_time):.03f} ms")

        # Verify solution is feasible
        np.testing.assert_allclose(X.T @ res.solution / M, mu)
        np.testing.assert_array_less(min_weight, res.solution)
        if phi is not None:
            self.assertLess(np.dot(res.solution, res.solution), M * phi)

        self.assertIsInstance(res, SolveResult)
        self.assertLessEqual(_barrier_diag(res).duality_gaps[-1], 1e-6)
        # Inner Newton step regression guard: after the cold-start first outer step,
        # each centering step should converge in ~10 inner iterations on this class of
        # problems. A blown-up step count (e.g. ~100 per outer) is the signature of a
        # backtracking line search regression.
        self.assertTrue(
            all(n <= 30 for n in _barrier_diag(res).inner_nits[1:]),
            f"Inner Newton step count regression: {_barrier_diag(res).inner_nits}",
        )

        # Now test specifying baseline weights
        estimated_propensity: npt.NDArray[np.float64] = expit(
            logit(true_propensity) + 0.1 * np.random.randn(M)
        )

        start_time = time.time()
        rake = Rake(
            distance=KLDivergence(v=1.0 / estimated_propensity),
            X=X,
            mu=mu,
            phi=phi,
            min_weight=min_weight,
            settings=OptimizationSettings(
                observability=ObservabilitySettings(verbose=True)
            ),
        )
        # Use v as the initial guess
        res = rake.solve(x0=1.0 / estimated_propensity)
        end_time = time.time()
        print(f"Complete in {1e3 * (end_time - start_time):.03f} ms")

        # Verify solution is feasible
        np.testing.assert_allclose(X.T @ res.solution / M, mu)
        np.testing.assert_array_less(min_weight, res.solution)
        if phi is not None:
            self.assertLess(np.dot(res.solution, res.solution), M * phi)

        self.assertIsInstance(res, SolveResult)
        self.assertLessEqual(_barrier_diag(res).duality_gaps[-1], 1e-6)
        self.assertTrue(
            all(n <= 30 for n in _barrier_diag(res).inner_nits[1:]),
            f"Inner Newton step count regression: {_barrier_diag(res).inner_nits}",
        )

    @parameterized.expand(
        [
            ("M100_p20_no_phi", 102, 100, 20, None),
            ("M200_p30_phi2.0", 202, 200, 30, 2.0),
            ("M50_p5_phi2.0", 302, 50, 5, 2.0),
            ("M500_p100_no_phi", 702, 500, 100, None),
            ("M13_p3_phi2.0", 502, 13, 3, 2.0),
        ]
    )
    def test_solve_squaredl2(
        self, name: str, seed: int, M: int, p: int, phi: float
    ) -> None:
        """Test problems with SquaredL2 as objective."""
        np.random.seed(seed)
        X = np.random.rand(M, p)

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

        # Compute population mean
        mu: npt.NDArray[np.float64] = (1 / M) * (X.T @ w)

        # Now forget we know w. We just know X for respondents and mu for the target
        # population.
        start_time = time.time()
        rake = Rake(
            distance=SquaredL2(dimension=M),
            X=X,
            mu=mu,
            phi=phi,
            settings=OptimizationSettings(
                observability=ObservabilitySettings(verbose=True)
            ),
        )
        res = rake.solve()
        end_time = time.time()
        print(f"Complete in {1e3 * (end_time - start_time):.03f} ms")

        # Verify solution is feasible
        np.testing.assert_allclose(X.T @ res.solution / M, mu)
        np.testing.assert_array_less(0, res.solution)
        if phi is not None:
            self.assertLess(np.dot(res.solution, res.solution), M * phi)

        self.assertIsInstance(res, SolveResult)
        self.assertLessEqual(_barrier_diag(res).duality_gaps[-1], 1e-6)
        # Inner Newton step regression guard: after the cold-start first outer step,
        # each centering step should converge in ~10 inner iterations on this class of
        # problems. A blown-up step count (e.g. ~100 per outer) is the signature of a
        # backtracking line search regression.
        self.assertTrue(
            all(n <= 30 for n in _barrier_diag(res).inner_nits[1:]),
            f"Inner Newton step count regression: {_barrier_diag(res).inner_nits}",
        )

        # Now test specifying baseline weights
        estimated_propensity: npt.NDArray[np.float64] = expit(
            logit(true_propensity) + 0.1 * np.random.randn(M)
        )

        start_time = time.time()
        rake = Rake(
            distance=SquaredL2(v=1.0 / estimated_propensity),
            X=X,
            mu=mu,
            phi=phi,
            settings=OptimizationSettings(
                observability=ObservabilitySettings(verbose=True)
            ),
        )
        # Use v as the initial guess
        res = rake.solve(x0=1.0 / estimated_propensity)
        end_time = time.time()
        print(f"Complete in {1e3 * (end_time - start_time):.03f} ms")

        # Verify solution is feasible
        np.testing.assert_allclose(X.T @ res.solution / M, mu)
        np.testing.assert_array_less(0, res.solution)
        if phi is not None:
            self.assertLess(np.dot(res.solution, res.solution), M * phi)

        self.assertIsInstance(res, SolveResult)
        # Warm-started IPM terminates after fewer outer iterations, so the Newton-
        # decrement correction term in the true `inf_x L` dual is non-trivial relative
        # to the m/t surrogate. Threshold is looser than the cold-start path above.
        self.assertLessEqual(_barrier_diag(res).duality_gaps[-1], 3e-6)
        self.assertTrue(
            all(n <= 30 for n in _barrier_diag(res).inner_nits[1:]),
            f"Inner Newton step count regression: {_barrier_diag(res).inner_nits}",
        )

    @parameterized.expand(
        [
            ("M100_p20_no_phi", 103, 100, 20, None),
            ("M200_p30_phi1.5", 203, 200, 30, 1.5),
            ("M50_p5_phi2.0", 303, 50, 5, 2.0),
            ("M500_p100_phi2.5", 403, 500, 100, 2.5),
            ("M13_p3_phi1.5", 503, 13, 3, 1.5),
        ]
    )
    def test_solve_huber(
        self, name: str, seed: int, M: int, p: int, phi: float
    ) -> None:
        """Test problems with Huber as objective."""
        np.random.seed(seed)
        X = np.random.rand(M, p)

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

        # Compute population mean
        mu: npt.NDArray[np.float64] = (1 / M) * (X.T @ w)

        # Now forget we know w. We just know X for respondents and mu for the target
        # population.
        start_time = time.time()
        rake = Rake(
            distance=Huber(dimension=M),
            X=X,
            mu=mu,
            phi=phi,
            settings=OptimizationSettings(
                observability=ObservabilitySettings(verbose=True)
            ),
        )
        res = rake.solve()
        end_time = time.time()
        print(f"Complete in {1e3 * (end_time - start_time):.03f} ms")

        # Verify solution is feasible
        np.testing.assert_allclose(X.T @ res.solution / M, mu)
        np.testing.assert_array_less(0, res.solution)
        if phi is not None:
            self.assertLess(np.dot(res.solution, res.solution), M * phi)

        self.assertIsInstance(res, SolveResult)
        self.assertLessEqual(_barrier_diag(res).duality_gaps[-1], 1e-6)
        # Inner Newton step regression guard: after the cold-start first outer step,
        # each centering step should converge in ~10 inner iterations on this class of
        # problems. A blown-up step count (e.g. ~100 per outer) is the signature of a
        # backtracking line search regression.
        self.assertTrue(
            all(n <= 30 for n in _barrier_diag(res).inner_nits[1:]),
            f"Inner Newton step count regression: {_barrier_diag(res).inner_nits}",
        )

        # Now test specifying baseline weights
        estimated_propensity: npt.NDArray[np.float64] = expit(
            logit(true_propensity) + 0.1 * np.random.randn(M)
        )

        start_time = time.time()
        rake = Rake(
            distance=Huber(v=1.0 / estimated_propensity),
            X=X,
            mu=mu,
            phi=phi,
            settings=OptimizationSettings(
                observability=ObservabilitySettings(verbose=True)
            ),
        )
        # Use v as the initial guess
        res = rake.solve(x0=1.0 / estimated_propensity)
        end_time = time.time()
        print(f"Complete in {1e3 * (end_time - start_time):.03f} ms")

        # Verify solution is feasible
        np.testing.assert_allclose(X.T @ res.solution / M, mu)
        np.testing.assert_array_less(0, res.solution)
        if phi is not None:
            self.assertLess(np.dot(res.solution, res.solution), M * phi)

        self.assertIsInstance(res, SolveResult)
        # Some CI machines reproduce a deterministic ~1.3e-12 absolute overshoot past
        # the requested 1e-6 (i.e. ~1.3e-6 relative) that others do not, consistent
        # with sub-ULP BLAS summation-order differences. A bare
        # equality against 1e-6 is not a meaningful assertion at that scale; assert
        # the solver met its requested tolerance to within floating point instead.
        self.assertLessEqual(_barrier_diag(res).duality_gaps[-1], 1e-6 * (1 + 1e-5))
        self.assertTrue(
            all(n <= 30 for n in _barrier_diag(res).inner_nits[1:]),
            f"Inner Newton step count regression: {_barrier_diag(res).inner_nits}",
        )

    @parameterized.expand(
        [
            ("M100_p20", 104, 100, 20),
            ("M200_p30", 204, 200, 30),
            ("M50_p5", 304, 50, 5),
            ("M500_p100", 404, 500, 100),
            ("M13_p3", 504, 13, 3),
        ]
    )
    def test_solve_infeasible(self, name: str, seed: int, M: int, p: int) -> None:
        """Test a scenario where the problem is infeasible."""
        np.random.seed(seed)
        nu = np.random.randn(p)

        # Construct A such that A^T nu >= 0
        A: npt.NDArray[np.float64] = np.random.randn(p, M)
        for ic in range(M):
            if np.dot(A[:, ic], nu) < 0:
                A[:, ic] = -A[:, ic]

        self.assertTrue(
            np.all(A.T @ nu >= 0), f"A.T @ nu must be >= 0; got {A.T @ nu!r}"
        )

        # Scale A such that c^T * nu = 1
        A /= np.dot(A @ np.ones((M,)), nu)
        self.assertTrue(
            np.all(A.T @ nu >= 0), f"A.T @ nu must be >= 0; got {A.T @ nu!r}"
        )
        self.assertLessEqual(abs(np.dot(A @ np.ones((M,)), nu) - 1), 1e-6)

        # Construct b such that b^T nu < 0.
        b = np.random.randn(p)
        alpha = np.dot(nu, b) / np.dot(nu, nu)
        b = b - (alpha + 0.1) * nu
        self.assertTrue(
            np.all(A.T @ nu >= 0), f"A.T @ nu must be >= 0; got {A.T @ nu!r}"
        )
        self.assertLessEqual(abs(np.dot(A @ np.ones((M,)), nu) - 1), 1e-6)
        self.assertLess(np.dot(b, nu), 0)

        phi = 1_000.0

        rake = Rake(
            distance=Huber(dimension=M),
            X=M * A.T,
            mu=b,
            phi=phi,
            settings=OptimizationSettings(
                observability=ObservabilitySettings(verbose=True)
            ),
        )

        with self.assertRaises(ProblemCertifiablyInfeasibleError):
            rake.solve()

    @parameterized.expand(
        [
            ("M100_p1_10_p2_15", 105, 100, 10, 15),
            ("M200_p1_5_p2_30", 205, 200, 5, 30),
            ("M50_p1_5_p2_5", 305, 50, 5, 5),
            ("M500_p1_20_p2_30", 405, 500, 20, 30),
            ("M13_p1_3_p2_3", 505, 13, 3, 3),
        ]
    )
    def test_max_imbalance(
        self, name: str, seed: int, M: int, p1: int, p2: int
    ) -> None:
        """Test a scenario with a constraint on the max covariate imbalance."""
        np.random.seed(seed)
        A: npt.NDArray[np.float64] = np.random.randn(p1, M)
        B: npt.NDArray[np.float64] = np.random.randn(p2, M)

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

        b = A @ w
        psi = 0.05
        c = B @ w + (2 * psi) * np.random.rand(p2) - psi

        phi = float(2.0 * np.mean(w * w))

        rake = Rake(
            distance=KLDivergence(dimension=M),
            X=M * A.T,
            mu=b,
            Z=M * B.T,
            nu=c,
            psi=psi,
            phi=phi,
            settings=OptimizationSettings(
                observability=ObservabilitySettings(verbose=True)
            ),
        )

        res = rake.solve()

        # Verify we find a feasible point
        np.testing.assert_allclose(A @ res.solution, b)
        np.testing.assert_array_less(0, res.solution)
        np.testing.assert_array_less(-psi, B @ res.solution - c)
        np.testing.assert_array_less(B @ res.solution - c, psi)
        self.assertLess(np.mean(res.solution * res.solution), phi)

        # Test with vector psi
        rake = Rake(
            distance=KLDivergence(dimension=M),
            X=M * A.T,
            mu=b,
            Z=M * B.T,
            nu=c,
            psi=np.full((p2,), psi),
            phi=phi,
            settings=OptimizationSettings(
                observability=ObservabilitySettings(verbose=True)
            ),
        )

        res = rake.solve()

        # Verify we find a feasible point
        np.testing.assert_allclose(A @ res.solution, b)
        np.testing.assert_array_less(0, res.solution)
        np.testing.assert_array_less(-psi, B @ res.solution - c)
        np.testing.assert_array_less(B @ res.solution - c, psi)
        self.assertLess(np.mean(res.solution * res.solution), phi)

    @parameterized.expand(
        [
            ("M100_p20", 106, 100, 20),
            ("M50_p5", 306, 50, 5),
            ("M13_p3", 506, 13, 3),
        ]
    )
    def test_max_weight_binding(self, name: str, seed: int, M: int, p: int) -> None:
        """A scalar ``max_weight`` tight enough to bind on the natural solution.

        Uses the clipped natural weights as the population reference so the box is
        guaranteed feasible (some clipped entries are at the cap, exercising the
        upper-bound machinery without straying into infeasibility).

        """
        np.random.seed(seed)
        X = np.random.rand(M, p)

        # Same propensity setup as the other tests; mean-1 weights have most mass
        # near 1 but a heavy tail (some weights ~5-15).
        q = 0.1
        sigma2 = 0.05 * q * (1 - q)
        s = q * (1 - q) / sigma2 - 1
        alpha = q * s
        beta = (1 - q) * s
        true_propensity = np.random.beta(alpha, beta, size=M)
        w = 1.0 / true_propensity
        w /= np.mean(w)

        min_weight = 0.01
        max_weight = 3.0
        # Construct mu from clipped weights so the box is provably feasible.
        w_clipped = np.clip(w, min_weight + 0.01, max_weight - 0.01)
        mu: npt.NDArray[np.float64] = (1 / M) * (X.T @ w_clipped)
        # Sanity: the natural weights would exceed the cap somewhere.
        self.assertTrue(np.any(w > max_weight))

        rake = Rake(
            distance=KLDivergence(dimension=M),
            X=X,
            mu=mu,
            min_weight=min_weight,
            max_weight=max_weight,
            constrain_mean_weight_to=None,
            settings=OptimizationSettings(
                observability=ObservabilitySettings(verbose=True)
            ),
        )
        res = rake.solve()

        # Verify feasibility: equality, lower bound, upper bound.
        np.testing.assert_allclose(X.T @ res.solution / M, mu)
        np.testing.assert_array_less(min_weight, res.solution)
        np.testing.assert_array_less(res.solution, max_weight)

        self.assertIsInstance(res, SolveResult)
        self.assertLessEqual(_barrier_diag(res).duality_gaps[-1], 1e-6)


class RakeWeightConstraintsTest(unittest.TestCase):
    """Tests for ``Rake`` weight-constraint shapes and validation."""

    def test_max_weight_with_phi(self) -> None:
        """``max_weight`` coexists with the squared-norm constraint when both fit."""
        np.random.seed(207)
        M, p = 80, 10
        X = np.random.rand(M, p)

        q = 0.1
        sigma2 = 0.05 * q * (1 - q)
        s = q * (1 - q) / sigma2 - 1
        alpha = q * s
        beta = (1 - q) * s
        true_propensity = np.random.beta(alpha, beta, size=M)
        w = 1.0 / true_propensity
        w /= np.mean(w)

        min_weight = 0.01
        max_weight = 5.0  # loose enough to coexist with phi
        w_clipped = np.clip(w, min_weight + 0.01, max_weight - 0.01)
        mu: npt.NDArray[np.float64] = (1 / M) * (X.T @ w_clipped)
        # Generous norm budget: at least the clipped solution's own norm.
        phi = float(2.0 * np.mean(np.square(w_clipped)))

        rake = Rake(
            distance=KLDivergence(dimension=M),
            X=X,
            mu=mu,
            phi=phi,
            min_weight=min_weight,
            max_weight=max_weight,
            constrain_mean_weight_to=None,
            settings=OptimizationSettings(
                observability=ObservabilitySettings(verbose=True)
            ),
        )
        res = rake.solve()

        np.testing.assert_allclose(X.T @ res.solution / M, mu)
        np.testing.assert_array_less(min_weight, res.solution)
        np.testing.assert_array_less(res.solution, max_weight)
        self.assertLess(np.dot(res.solution, res.solution), M * phi)
        self.assertIsInstance(res, SolveResult)
        self.assertLessEqual(_barrier_diag(res).duality_gaps[-1], 1e-6)

    def test_max_weight_vector_per_component(self) -> None:
        """A per-component vector ``max_weight`` is honoured at every coordinate."""
        np.random.seed(701)
        M, p = 80, 10
        X = np.random.rand(M, p)

        q = 0.1
        sigma2 = 0.05 * q * (1 - q)
        s = q * (1 - q) / sigma2 - 1
        alpha = q * s
        beta = (1 - q) * s
        true_propensity = np.random.beta(alpha, beta, size=M)
        w = 1.0 / true_propensity
        w /= np.mean(w)

        # Two different caps interleaved, both tight enough to bite somewhere.
        max_weight = np.where(np.arange(M) % 2 == 0, 2.5, 4.0)
        min_weight = 0.01
        w_clipped = np.clip(w, min_weight + 0.01, max_weight - 0.01)
        mu: npt.NDArray[np.float64] = (1 / M) * (X.T @ w_clipped)
        self.assertTrue(np.any(w >= max_weight))

        rake = Rake(
            distance=KLDivergence(dimension=M),
            X=X,
            mu=mu,
            min_weight=min_weight,
            max_weight=max_weight,
            constrain_mean_weight_to=None,
            settings=OptimizationSettings(
                observability=ObservabilitySettings(verbose=True)
            ),
        )
        res = rake.solve()

        np.testing.assert_allclose(X.T @ res.solution / M, mu)
        np.testing.assert_array_less(min_weight, res.solution)
        np.testing.assert_array_less(res.solution, max_weight)
        self.assertIsInstance(res, SolveResult)
        self.assertLessEqual(_barrier_diag(res).duality_gaps[-1], 1e-6)

    def test_min_weight_vector(self) -> None:
        """A per-component vector ``min_weight`` is honoured (no upper bound)."""
        np.random.seed(702)
        M, p = 60, 8
        X = np.random.rand(M, p)

        q = 0.1
        sigma2 = 0.05 * q * (1 - q)
        s = q * (1 - q) / sigma2 - 1
        alpha = q * s
        beta = (1 - q) * s
        true_propensity = np.random.beta(alpha, beta, size=M)
        w = 1.0 / true_propensity
        w /= np.mean(w)
        mu: npt.NDArray[np.float64] = (1 / M) * (X.T @ w)

        # Small staircase well below the natural weight distribution's minimum.
        min_weight = 0.005 + 0.0001 * np.arange(M, dtype=np.float64)
        np.testing.assert_array_less(min_weight, w)

        rake = Rake(
            distance=KLDivergence(dimension=M),
            X=X,
            mu=mu,
            min_weight=min_weight,
            settings=OptimizationSettings(
                observability=ObservabilitySettings(verbose=True)
            ),
        )
        res = rake.solve()

        np.testing.assert_allclose(X.T @ res.solution / M, mu)
        np.testing.assert_array_less(min_weight, res.solution)
        self.assertIsInstance(res, SolveResult)

    def test_unbounded_solution_no_max_weight(self) -> None:
        """``max_weight=None`` (default) reproduces the old behaviour exactly."""
        np.random.seed(703)
        M, p = 40, 6
        X = np.random.rand(M, p)
        w = np.full(M, 1.0)
        mu: npt.NDArray[np.float64] = (1 / M) * (X.T @ w)

        base = Rake(
            distance=KLDivergence(dimension=M),
            X=X,
            mu=mu,
            min_weight=0.01,
            settings=OptimizationSettings(
                observability=ObservabilitySettings(verbose=False)
            ),
        )
        explicit = Rake(
            distance=KLDivergence(dimension=M),
            X=X,
            mu=mu,
            min_weight=0.01,
            max_weight=None,
            settings=OptimizationSettings(
                observability=ObservabilitySettings(verbose=False)
            ),
        )
        base_res = base.solve()
        explicit_res = explicit.solve()
        # Both forms should produce the same trajectory (deterministic).
        np.testing.assert_allclose(base_res.solution, explicit_res.solution)

    def test_max_weight_validation_scalar_overlap(self) -> None:
        """Constructing with ``min_weight >= max_weight`` raises at __init__."""
        np.random.seed(704)
        X = np.random.rand(20, 4)
        mu = np.zeros(4)
        with self.assertRaisesRegex(ValueError, "min_weight < max_weight"):
            Rake(
                distance=KLDivergence(dimension=20),
                X=X,
                mu=mu,
                min_weight=0.5,
                max_weight=0.5,
            )
        with self.assertRaisesRegex(ValueError, "min_weight < max_weight"):
            Rake(
                distance=KLDivergence(dimension=20),
                X=X,
                mu=mu,
                min_weight=0.5,
                max_weight=0.4,
            )

    def test_max_weight_validation_vector_overlap(self) -> None:
        """A vector ``max_weight`` overlapping ``min_weight`` componentwise raises."""
        np.random.seed(705)
        X = np.random.rand(20, 4)
        mu = np.zeros(4)
        with self.assertRaisesRegex(ValueError, "min_weight < max_weight"):
            Rake(
                distance=KLDivergence(dimension=20),
                X=X,
                mu=mu,
                min_weight=np.array([0.0, 0.0, 5.0]),
                max_weight=4.0,
            )

    def test_max_weight_validation_broadcast_incompatible(self) -> None:
        """Vector bounds with non-broadcast-compatible shapes raise at __init__."""
        np.random.seed(706)
        X = np.random.rand(20, 4)
        mu = np.zeros(4)
        with self.assertRaisesRegex(ValueError, "broadcast-compatible"):
            Rake(
                distance=KLDivergence(dimension=20),
                X=X,
                mu=mu,
                min_weight=np.array([0.0, 1.0, 2.0]),
                max_weight=np.array([5.0, 6.0]),
            )


class InfeasibleExternalValidityTest(unittest.TestCase):
    """Over-tight external validity is reported as infeasible, not as a crash."""

    @staticmethod
    def _calibrator(engine: Literal["ipm", "pd"]) -> JointCalibrator:
        """A joint problem whose external-validity band cannot be met.

        ``nu1``/``nu2`` sit far outside the range any non-negative weighting of the
        sample can reach, with a band far too narrow to bridge it.

        """
        rng = np.random.default_rng(1117)
        M1, M2, q = 60, 60, 3
        Z = rng.standard_normal((M1 + M2, q))
        return JointCalibrator(
            distance=SquaredL2(v=np.zeros(M1 + M2)),
            Z1=Z[:M1],
            Z2=Z[M1:],
            Z3=Z[:M1],
            Z4=Z[M1:],
            nu1=np.full(q, 10.0),
            nu2=np.full(q, 10.0),
            psi1=1e-6,
            psi2=1e-6,
            engine=engine,
        )

    # Smallest achievable band for ``_near_boundary_calibrator``'s data, from
    #     minimize t  s.t.  |(1/M_b) Z_b^T w_b - nu_b| <= t / sqrt(M_b),
    #                       mean(w_b) = 1,  w_b >= M_b / N
    # -- the feasibility LP a caller runs to find where a psi sweep must stop.
    # Bands either side of it are infeasible and feasible respectively, by a
    # whisker, which is the regime that strains Phase I.
    MIN_PHI: float = 5.7665

    M1: int = 500
    M2: int = 500

    @classmethod
    def _near_boundary_data(
        cls,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Respondent covariates skewed well away from a target mean of zero."""
        rng = np.random.default_rng(7)
        q = 4
        return (
            rng.standard_normal((cls.M1, q)) + 1.5,
            rng.standard_normal((cls.M2, q)) + 1.5,
        )

    @classmethod
    def _near_boundary_calibrator(
        cls, engine: Literal["ipm", "pd"], phi_fraction: float
    ) -> JointCalibrator:
        """A joint problem whose band sits ``phi_fraction`` of the way to MIN_PHI.

        Below 1.0 the problem is infeasible by a hair rather than by a mile. Before the
        fix that case did not return a verdict: it surfaced a NewtonStepError about
        slack signs, several layers below anything the caller controls.

        """
        M1, M2 = cls.M1, cls.M2
        q = 4
        population_size = 2_000_000
        Z1, Z2 = cls._near_boundary_data()
        psi = phi_fraction * cls.MIN_PHI
        return JointCalibrator(
            distance=SquaredL2(v=np.zeros(M1 + M2)),
            Z1=Z1,
            Z2=Z2,
            Z3=Z1,
            Z4=Z2,
            nu1=np.zeros(q),
            nu2=np.zeros(q),
            psi1=psi / np.sqrt(M1),
            psi2=psi / np.sqrt(M2),
            # Keep internal validity slack so the external band is what binds.
            psi3=1e3,
            min_weight1=M1 / population_size,
            min_weight2=M2 / population_size,
            engine=engine,
        )

    @parameterized.expand([("pd",), ("ipm",)])
    def test_just_feasible_problem_still_solves(
        self, engine: Literal["ipm", "pd"]
    ) -> None:
        """The fix must not turn a feasible near-boundary problem into a verdict."""
        M1, M2 = self.M1, self.M2
        phi_fraction = 1.001
        psi = phi_fraction * self.MIN_PHI
        Z1, Z2 = self._near_boundary_data()

        result = self._near_boundary_calibrator(engine, phi_fraction).solve()
        w = np.asarray(result.solution)
        w1, w2 = w[:M1], w[M1:]

        self.assertEqual(len(w), M1 + M2)
        # The band it asked for is actually met, in both groups (nu is zero).
        self.assertLessEqual(
            float(np.max(np.abs(Z1.T @ w1 / M1))), psi / np.sqrt(M1) + 1e-6
        )
        self.assertLessEqual(
            float(np.max(np.abs(Z2.T @ w2 / M2))), psi / np.sqrt(M2) + 1e-6
        )
        # Mean weight one per group, so the solve respected the equalities.
        self.assertAlmostEqual(float(np.mean(w1)), 1.0, places=4)
        self.assertAlmostEqual(float(np.mean(w2)), 1.0, places=4)

    @parameterized.expand([("pd",), ("ipm",)])
    def test_infeasible_problem_raises_infeasible(
        self, engine: Literal["ipm", "pd"]
    ) -> None:
        """Both engines reach an infeasibility verdict rather than a ValueError."""
        with self.assertRaises(ProblemInfeasibleError):
            self._calibrator(engine).solve()

    @parameterized.expand([("pd",), ("ipm",)])
    def test_marginally_infeasible_problem_raises_infeasible(
        self, engine: Literal["ipm", "pd"]
    ) -> None:
        """A near-miss target is a verdict, not ``All entries of d must be nonzero``."""
        with self.assertRaises(ProblemInfeasibleError):
            self._near_boundary_calibrator(engine, phi_fraction=0.999).solve()

    def test_fallback_weights_are_a_real_weighting(self) -> None:
        """The fallback cascade never returns the untouched Phase I seed.

        ``solve_with_fallback`` runs the inner solves with ``on_infeasible="continue"``
        and hands back ``solve_result.solution`` as the calibrated weights. Converting a
        Phase I breakdown into a verdict there would present the seed -- for the direct
        form often zeros -- as a minimum-violation answer, so the conversion is scoped
        to the raise path. A weighting whose groups do not average one is the symptom.

        """
        M1, M2 = self.M1, self.M2
        result = self._near_boundary_calibrator(
            "pd", phi_fraction=0.999
        ).solve_with_fallback()
        w = np.asarray(result.solution)

        self.assertEqual(len(w), M1 + M2)
        self.assertAlmostEqual(float(np.mean(w[:M1])), 1.0, places=4)
        self.assertAlmostEqual(float(np.mean(w[M1:])), 1.0, places=4)
        self.assertGreater(result.combined_effective_sample_size, 0.0)


class FloatCoercionTest(unittest.TestCase):
    """Inputs are coerced to float64 at construction.

    Standardizing covariates against a population summary frame that carries any
    non-numeric column yields object-dtype arrays. Those reach the solver intact and
    fail inside the barrier loop on ``np.log``, far from the cause.

    """

    @staticmethod
    def _problem(dtype: npt.DTypeLike) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """A small internal-validity problem, in the requested dtype."""
        np.random.seed(1109)
        return (
            np.random.randn(30, 2).astype(dtype),
            np.random.randn(30, 2).astype(dtype),
        )

    def test_joint_calibrator_solves_with_object_dtype(self) -> None:
        """An object-dtype Z3/Z4 solves rather than raising from np.log."""
        Z3, Z4 = self._problem(object)
        self.assertEqual(Z3.dtype, np.dtype("O"))

        result = JointCalibrator(
            distance=KLDivergence(dimension=60), Z3=Z3, Z4=Z4, psi3=0.05
        ).solve()

        self.assertEqual(np.asarray(result.solution).dtype, np.float64)

    def _solve(self, dtype: npt.DTypeLike) -> npt.NDArray[np.float64]:
        """Solve the same problem with inputs of the given dtype."""
        Z3, Z4 = self._problem(dtype)
        result = JointCalibrator(
            distance=KLDivergence(dimension=60), Z3=Z3, Z4=Z4, psi3=0.05
        ).solve()
        return np.asarray(result.solution)

    def test_object_dtype_matches_float_dtype(self) -> None:
        """Coercion changes the dtype and nothing else."""
        np.testing.assert_allclose(self._solve(object), self._solve(np.float64))

    def test_rake_solves_with_object_dtype(self) -> None:
        """The same coercion applies to the single-population Rake."""
        np.random.seed(1110)
        X = np.random.randn(40, 3).astype(object)
        mu = np.zeros(3, dtype=object)

        result = Rake(distance=KLDivergence(dimension=40), X=X, mu=mu).solve()

        self.assertEqual(np.asarray(result.solution).dtype, np.float64)

    def test_non_numeric_input_raises_a_clear_error(self) -> None:
        """A genuinely non-numeric array names the offending argument."""
        Z3, Z4 = self._problem(np.float64)
        Z3 = Z3.astype(object)
        Z3[0, 0] = "not a number"

        with self.assertRaisesRegex(ValueError, "`Z3` must be numeric"):
            JointCalibrator(distance=KLDivergence(dimension=60), Z3=Z3, Z4=Z4)

    def test_non_numeric_scalar_raises_a_clear_error(self) -> None:
        """Scalar parameters are named too."""
        Z3, Z4 = self._problem(np.float64)

        with self.assertRaisesRegex(ValueError, "`psi3` must be a number"):
            JointCalibrator(
                distance=KLDivergence(dimension=60),
                Z3=Z3,
                Z4=Z4,
                # why: deliberately wrong, that is the test.
                psi3=cast(Any, "tight"),
            )


class JointCalibratorMaxWeightTest(unittest.TestCase):
    """Tests for ``JointCalibrator``'s per-block ``max_weight`` plumbing."""

    def test_both_blocks(self) -> None:
        """Both ``max_weight1`` and ``max_weight2`` are honoured."""
        np.random.seed(801)
        M1, M2 = 40, 40
        q = 3
        Z1 = np.random.randn(M1, q)
        Z2 = np.random.randn(M2, q)
        Z3 = np.random.randn(M1, q)
        Z4 = np.random.randn(M2, q)
        nu1 = np.mean(Z1, axis=0)
        nu2 = np.mean(Z2, axis=0)

        max_weight1 = 2.0
        max_weight2 = 1.5

        rake = JointCalibrator(
            distance=KLDivergence(dimension=M1 + M2),
            Z1=Z1,
            Z2=Z2,
            Z3=Z3,
            Z4=Z4,
            nu1=nu1,
            nu2=nu2,
            psi1=0.5,
            psi2=0.5,
            psi3=0.5,
            max_weight1=max_weight1,
            max_weight2=max_weight2,
            settings=OptimizationSettings(
                observability=ObservabilitySettings(verbose=True)
            ),
        )
        result = rake.solve_with_fallback()
        self.assertIsInstance(result, FallbackOptimizationResult)
        self.assertEqual(len(result.solution), M1 + M2)

        # Each block respects its own cap.
        np.testing.assert_array_less(result.solution[:M1], max_weight1)
        np.testing.assert_array_less(result.solution[M1:], max_weight2)
        np.testing.assert_array_less(0, result.solution)

    def test_only_first_block(self) -> None:
        """Setting only ``max_weight1`` leaves block 2 unbounded above."""
        np.random.seed(802)
        M1, M2 = 40, 40
        q = 3
        Z1 = np.random.randn(M1, q)
        Z2 = np.random.randn(M2, q)
        Z3 = np.random.randn(M1, q)
        Z4 = np.random.randn(M2, q)
        nu1 = np.mean(Z1, axis=0)
        nu2 = np.mean(Z2, axis=0)

        max_weight1 = 2.0

        rake = JointCalibrator(
            distance=KLDivergence(dimension=M1 + M2),
            Z1=Z1,
            Z2=Z2,
            Z3=Z3,
            Z4=Z4,
            nu1=nu1,
            nu2=nu2,
            psi1=0.5,
            psi2=0.5,
            psi3=0.5,
            max_weight1=max_weight1,
            # max_weight2 omitted: block 2 is unbounded above.
            settings=OptimizationSettings(
                observability=ObservabilitySettings(verbose=True)
            ),
        )
        # The combined `max_weight` vector should exist with +inf filler for block 2.
        max_weight = rake.max_weight
        assert isinstance(max_weight, np.ndarray)  # Narrows for mypy
        self.assertEqual(max_weight.shape, (M1 + M2,))
        np.testing.assert_array_equal(max_weight[:M1], max_weight1)
        self.assertTrue(
            np.all(np.isinf(max_weight[M1:])),
            f"max_weight[M1:] must be all inf; got {max_weight[M1:]!r}",
        )

        result = rake.solve_with_fallback()
        self.assertIsInstance(result, FallbackOptimizationResult)
        np.testing.assert_array_less(result.solution[:M1], max_weight1)
        np.testing.assert_array_less(0, result.solution)

    def test_an_uncapped_block_solves_as_if_its_cap_were_slack(self) -> None:
        """Leaving ``max_weight2`` unset matches a cap block 2 never reaches.

        The ``+inf`` filler for block 2 used to reach Cvxium as bound rows, whose
        barrier term made the barrier objective ``-inf`` and the line search accept any
        step.

        """
        np.random.seed(802)
        M1, M2 = 40, 40
        q = 3
        Z1 = np.random.randn(M1, q)
        Z2 = np.random.randn(M2, q)
        Z3 = np.random.randn(M1, q)
        Z4 = np.random.randn(M2, q)

        def solve(max_weight2: float | None) -> npt.NDArray[np.float64]:
            rake = JointCalibrator(
                distance=KLDivergence(dimension=M1 + M2),
                Z1=Z1,
                Z2=Z2,
                Z3=Z3,
                Z4=Z4,
                nu1=np.mean(Z1, axis=0),
                nu2=np.mean(Z2, axis=0),
                psi1=0.5,
                psi2=0.5,
                psi3=0.5,
                max_weight1=2.0,
                max_weight2=max_weight2,
            )
            return np.asarray(rake.solve_with_fallback().solution)

        uncapped = solve(None)
        slack_cap = solve(1e3)
        self.assertLess(float(np.max(uncapped[M1:])), 10.0)
        np.testing.assert_allclose(uncapped, slack_cap, rtol=1e-4, atol=1e-4)

    def test_neither_block_omits_ub(self) -> None:
        """Without per-block caps, ``Rake.max_weight is None``."""
        np.random.seed(803)
        M1, M2 = 30, 30
        q = 2
        Z1 = np.random.randn(M1, q)
        Z2 = np.random.randn(M2, q)
        nu1 = np.mean(Z1, axis=0)
        nu2 = np.mean(Z2, axis=0)

        rake = JointCalibrator(
            distance=KLDivergence(dimension=M1 + M2),
            Z1=Z1,
            Z2=Z2,
            nu1=nu1,
            nu2=nu2,
            psi1=0.5,
            psi2=0.5,
            settings=OptimizationSettings(
                observability=ObservabilitySettings(verbose=False)
            ),
        )
        self.assertIsNone(rake.max_weight)

    def test_per_block_dimension_mismatch(self) -> None:
        """A vector ``max_weight1`` whose length disagrees with M1 is rejected."""
        np.random.seed(804)
        M1, M2 = 30, 30
        q = 2
        Z1 = np.random.randn(M1, q)
        Z2 = np.random.randn(M2, q)
        nu1 = np.mean(Z1, axis=0)
        nu2 = np.mean(Z2, axis=0)

        with self.assertRaisesRegex(ValueError, "Dimension mismatch"):
            JointCalibrator(
                distance=KLDivergence(dimension=M1 + M2),
                Z1=Z1,
                Z2=Z2,
                nu1=nu1,
                nu2=nu2,
                psi1=0.5,
                psi2=0.5,
                max_weight1=np.full(M1 + 1, 2.0),  # wrong length
            )


class JointCalibratorFallbackTest(unittest.TestCase):
    """Tests for ``JointCalibrator.solve_with_fallback()`` cascade behavior."""

    def test_happy_path(self) -> None:
        """Test that solve_with_fallback returns 'happy' when all constraints are feasible."""
        np.random.seed(601)
        M1, M2 = 50, 50
        q = 3

        Z1 = np.random.randn(M1, q)
        Z2 = np.random.randn(M2, q)
        Z3 = np.random.randn(M1, q)
        Z4 = np.random.randn(M2, q)

        # Use uniform weights to compute population means, so problem is easy.
        nu1 = np.mean(Z1, axis=0)
        nu2 = np.mean(Z2, axis=0)

        rake = JointCalibrator(
            distance=KLDivergence(dimension=M1 + M2),
            Z1=Z1,
            Z2=Z2,
            Z3=Z3,
            Z4=Z4,
            nu1=nu1,
            nu2=nu2,
            psi1=0.5,
            psi2=0.5,
            psi3=0.5,
            phi=10.0,  # Very loose phi.
            settings=OptimizationSettings(
                observability=ObservabilitySettings(verbose=True)
            ),
        )

        result = rake.solve_with_fallback()
        self.assertIsInstance(result, FallbackOptimizationResult)
        self.assertEqual(len(result.solution), M1 + M2)

        # Check all constraints satisfied.
        np.testing.assert_array_less(0, result.solution)
        self.assertLess(np.dot(result.solution, result.solution), (M1 + M2) * 10.0)
        self.assertEqual(result.fallback_status, 0)

    def test_happy_path_pd_engine(self) -> None:
        """solve_with_fallback runs under engine='pd' (no longer rejected).

        The feasibility probes always use the barrier's Phase I machinery; only the
        final distance-minimizing solve dispatches through PrimalDual. The happy result
        must match the IPM engine (both minimize the same distance).

        """
        np.random.seed(601)
        M1, M2 = 50, 50
        q = 3

        Z1 = np.random.randn(M1, q)
        Z2 = np.random.randn(M2, q)
        Z3 = np.random.randn(M1, q)
        Z4 = np.random.randn(M2, q)

        nu1 = np.mean(Z1, axis=0)
        nu2 = np.mean(Z2, axis=0)

        kwargs = {
            "distance": KLDivergence(dimension=M1 + M2),
            "Z1": Z1,
            "Z2": Z2,
            "Z3": Z3,
            "Z4": Z4,
            "nu1": nu1,
            "nu2": nu2,
            "psi1": 0.5,
            "psi2": 0.5,
            "psi3": 0.5,
            "phi": 10.0,
        }

        result_pd = JointCalibrator(**kwargs, engine="pd").solve_with_fallback()
        self.assertIsInstance(result_pd, FallbackOptimizationResult)
        self.assertEqual(result_pd.fallback_status, 0)
        self.assertEqual(len(result_pd.solution), M1 + M2)
        np.testing.assert_array_less(0, result_pd.solution)
        self.assertLess(
            np.dot(result_pd.solution, result_pd.solution), (M1 + M2) * 10.0
        )

        result_ipm = JointCalibrator(**kwargs, engine="ipm").solve_with_fallback()
        np.testing.assert_allclose(
            result_pd.solution, result_ipm.solution, rtol=1e-4, atol=1e-4
        )

    def test_phi_relaxed(self) -> None:
        """Test that solve_with_fallback returns 'phi_relaxed' with very tight phi."""
        np.random.seed(602)
        M1, M2 = 50, 50
        q = 3

        Z1 = np.random.randn(M1, q)
        Z2 = np.random.randn(M2, q)
        Z3 = np.random.randn(M1, q)
        Z4 = np.random.randn(M2, q)

        nu1 = np.mean(Z1, axis=0)
        nu2 = np.mean(Z2, axis=0)

        rake = JointCalibrator(
            distance=KLDivergence(dimension=M1 + M2),
            Z1=Z1,
            Z2=Z2,
            Z3=Z3,
            Z4=Z4,
            nu1=nu1,
            nu2=nu2,
            psi1=0.5,
            psi2=0.5,
            psi3=0.5,
            phi=0.001,  # Extremely tight phi -- should be infeasible.
            settings=OptimizationSettings(
                observability=ObservabilitySettings(verbose=True)
            ),
        )

        result = rake.solve_with_fallback()
        self.assertIsInstance(result, FallbackOptimizationResult)
        self.assertEqual(len(result.solution), M1 + M2)

        # Imbalance constraints should be met.
        B = none_throws(rake.B)
        c = none_throws(rake.c)
        psi = none_throws(rake.psi)
        Bx = B @ result.solution - c
        np.testing.assert_array_less(np.abs(Bx), np.asarray(psi) + 1e-6)
        self.assertEqual(result.fallback_status, 1)

    def test_ambivalent(self) -> None:
        """Test that solve_with_fallback returns 'ambivalent' with tight external psi."""
        np.random.seed(603)
        M1, M2 = 50, 50
        q = 3

        Z1 = np.random.randn(M1, q)
        Z2 = np.random.randn(M2, q)
        Z3 = np.random.randn(M1, q)
        Z4 = np.random.randn(M2, q)

        # Set nu1, nu2 to wildly different values from the sample means so external
        # validity is infeasible.
        nu1 = np.mean(Z1, axis=0) + 10.0
        nu2 = np.mean(Z2, axis=0) + 10.0

        rake = JointCalibrator(
            distance=KLDivergence(dimension=M1 + M2),
            Z1=Z1,
            Z2=Z2,
            Z3=Z3,
            Z4=Z4,
            nu1=nu1,
            nu2=nu2,
            psi1=1e-10,  # Extremely tight external psi -- infeasible.
            psi2=1e-10,
            psi3=0.5,  # Loose internal psi.
            phi=10.0,
            settings=OptimizationSettings(
                observability=ObservabilitySettings(verbose=True),
                outer=OuterSettings(tolerance_soft=0.5),
            ),
        )

        result = rake.solve_with_fallback()
        self.assertIsInstance(result, FallbackOptimizationResult)
        self.assertEqual(len(result.solution), M1 + M2)

        # Internal validity should be met.
        q_ext = q + q  # q1 + q2
        B, c, psi = none_throws(rake.B), none_throws(rake.c), none_throws(rake.psi)
        B_int = B[q_ext:, :]
        c_int = c[q_ext:]
        psi_int = np.asarray(psi)[q_ext:]
        Bx_int = B_int @ result.solution - c_int
        np.testing.assert_array_less(np.abs(Bx_int), psi_int + 1e-6)
        self.assertEqual(result.fallback_status, 2)

    def test_step1_infeasible_raises(self) -> None:
        """Bounds + equality fundamentally infeasible -> ProblemInfeasibleError.

        ``min_weight=2.0`` per group with ``constrain_mean_weight=1.0`` per group forces
        ``mean(w_i) >= 2 > 1`` -- no ``w`` satisfies both bounds and the mean-weight
        equality. The cascade raises from Step 1 (``bounds_solver.solve()``).

        """
        rng = np.random.default_rng(605)
        M1, M2 = 50, 50
        q = 3
        Z3 = rng.standard_normal((M1, q))
        Z4 = rng.standard_normal((M2, q))

        rake = JointCalibrator(
            distance=KLDivergence(dimension=M1 + M2),
            Z3=Z3,
            Z4=Z4,
            psi3=0.5,
            min_weight1=2.0,
            min_weight2=2.0,
            constrain_mean_weight1_to=1.0,
            constrain_mean_weight2_to=1.0,
            phi=10.0,
        )
        with self.assertRaises(ProblemInfeasibleError):
            rake.solve_with_fallback()

    def test_sad(self) -> None:
        """Test that solve_with_fallback returns 'sad' with tight internal psi."""
        np.random.seed(604)
        M1, M2 = 50, 50
        q = 3

        # Create groups with very different distributions so internal validity is
        # infeasible with tight psi.
        Z3 = np.random.randn(M1, q) + 5.0
        Z4 = np.random.randn(M2, q) - 5.0

        rake = JointCalibrator(
            distance=KLDivergence(dimension=M1 + M2),
            Z3=Z3,
            Z4=Z4,
            psi3=1e-10,  # Extremely tight -- infeasible.
            phi=10.0,
            settings=OptimizationSettings(
                observability=ObservabilitySettings(verbose=True),
                outer=OuterSettings(tolerance_soft=0.6),
            ),
        )

        result = rake.solve_with_fallback()
        self.assertIsInstance(result, FallbackOptimizationResult)
        self.assertEqual(len(result.solution), M1 + M2)
        self.assertEqual(result.fallback_status, 3)

    def test_sad_drives_internal_imbalance_to_minimum(self) -> None:
        """status-3 fallback (``on_infeasible="continue"``) minimizes the internal

        imbalance, not just reports infeasibility.

        A single internal-validity covariate with control values in ``[0, 1]`` and
        treatment values in ``[2, 3]`` cannot be balanced: the weighted group means are
        at least ``min(Z4) - max(Z3) = 1`` apart for any nonnegative weights summing to
        the (fixed) mean. So the problem is infeasible for ``psi3 << 1``, and the best
        achievable max-imbalance is ~1.0. The ``continue`` policy must drive Phase I to
        that floor; a naive return of the bounds-feasible (uniform) start would instead
        leave the imbalance at ``mean(Z4) - mean(Z3) = 2.0``.

        """
        rng = np.random.default_rng(606)
        M1, M2 = 30, 30
        Z3 = rng.uniform(0.0, 1.0, size=(M1, 1))
        Z4 = rng.uniform(2.0, 3.0, size=(M2, 1))

        rake = JointCalibrator(
            distance=KLDivergence(dimension=M1 + M2),
            Z3=Z3,
            Z4=Z4,
            psi3=0.01,  # far below the irreducible gap of ~1.0 -> infeasible
            phi=10.0,
        )

        result = rake.solve_with_fallback()
        self.assertEqual(result.fallback_status, 3)

        B, c = none_throws(rake.B), none_throws(rake.c)
        max_imbalance = float(np.max(np.abs(B @ result.solution - c)))
        # Minimized toward the ~1.0 floor, and well clear of the 2.0 a
        # non-minimizing fallback (uniform weights) would leave.
        self.assertLess(max_imbalance, 1.6)
        self.assertGreater(max_imbalance, 0.9)

    def test_ambivalent_external_only_minimizes_external(self) -> None:
        """status-2 fallback with no internal-validity constraints (q3 absent).

        Exercises the two-layer Step 3 when only external validity is requested, so the
        internal ``ImbalanceConstraintLayer`` is omitted and the problem degrades to
        ``[bounds, external]``. The control covariate lies in ``[0, 10]`` but its target
        mean is ``11``, so the weighted mean can close the gap only to ~1.0 (driving it
        to the max). The treatment block carries a trivially-feasible external covariate
        (it pins ``M2`` and stays satisfied), so overall external validity is
        infeasible. The ``continue`` policy must minimize the external imbalance (~1.0,
        well below the ~6.0 uniform weights leave) and report status 2.

        """
        rng = np.random.default_rng(607)
        M1, M2 = 30, 30
        # Deterministic control covariate so the floor max(Z1) - nu1 = 10 - 11 = 1.
        Z1 = np.linspace(0.0, 10.0, M1).reshape(M1, 1)
        nu1 = np.array([11.0])  # above the reachable [0, 10] range -> infeasible
        # Treatment external covariate: pins M2 and is trivially feasible at its
        # own sample mean, so the binding infeasibility is the control side.
        Z2 = rng.uniform(0.0, 1.0, size=(M2, 1))
        nu2 = np.array([float(np.mean(Z2))])

        rake = JointCalibrator(
            distance=KLDivergence(dimension=M1 + M2),
            Z1=Z1,
            Z2=Z2,
            nu1=nu1,
            nu2=nu2,
            psi1=0.5,  # below the irreducible ~1.0 control gap -> infeasible
            psi2=0.5,
            phi=10.0,
        )

        result = rake.solve_with_fallback()
        self.assertEqual(result.fallback_status, 2)

        B, c = none_throws(rake.B), none_throws(rake.c)
        max_external = float(np.max(np.abs(B @ result.solution - c)))
        self.assertLess(max_external, 2.0)  # minimized toward the ~1.0 floor
        self.assertGreater(max_external, 0.5)  # cannot beat the irreducible gap


class JointCalibratorRelaxationTest(unittest.TestCase):
    """An infeasible imbalance layer is relaxed only on the rows that cannot be met.

    Phase I minimizes the largest excess over the band, which leaves every other row
    free to sit anywhere up to that excess. The relaxation concentrates the excess on
    the rows that need it and keeps the rest within their bands.

    """

    @staticmethod
    def _l30_calibrator() -> JointCalibrator:
        """A BLS survey's five engagement levels, with none of the control group

        in the second level (8--14 days of the last 30).

        No weighting can give that level any control share, so external validity is
        infeasible, and internal validity drags the treatment group's share of it down
        too. Every other level can be met. Covariates are whitened by the population
        standard deviation, as in a survey calibration.

        """
        mu = np.array([0.072558, 0.033584, 0.043063, 0.245502, 0.605294])
        mu = mu / np.sum(mu)
        counts_control = [3, 0, 5, 26, 472]
        counts_treatment = [7, 6, 6, 34, 448]
        sigma = np.sqrt(mu * (1 - mu))

        def whitened_indicators(counts: list[int]) -> npt.NDArray[np.float64]:
            levels = np.repeat(np.arange(len(counts)), counts)
            return np.eye(len(counts))[levels] / sigma

        Z_control = whitened_indicators(counts_control)
        Z_treatment = whitened_indicators(counts_treatment)
        M = Z_control.shape[0] + Z_treatment.shape[0]
        return JointCalibrator(
            distance=SquaredL2(v=np.zeros(M)),
            Z1=Z_control,
            Z2=Z_treatment,
            Z3=Z_control,
            Z4=Z_treatment,
            nu1=mu / sigma,
            nu2=mu / sigma,
        )

    def test_empty_level_relaxes_only_its_own_rows(self) -> None:
        """Verify the empty level relaxes only its own rows."""
        rake = self._l30_calibrator()
        result = rake.solve_with_fallback()
        self.assertEqual(result.fallback_status, 2)

        B, c, psi = rake.B, rake.c, rake.psi
        assert B is not None and c is not None and psi is not None
        psi = np.broadcast_to(np.asarray(psi), c.shape)
        imbalance = np.abs(B @ result.solution - c)
        # Within the band, to the 1% the relaxation's slack may leave.
        within_band = imbalance <= psi * 1.01

        # Rows 0-4 are the control group's external rows, 5-9 the treatment
        # group's, 10-14 internal. Only the empty level (row 1) and its
        # treatment counterpart (row 6) should be relaxed.
        relaxed = np.array([1, 6])
        met = np.setdiff1d(np.arange(len(c)), relaxed)
        self.assertTrue(np.all(within_band[met]), imbalance[met] - psi[met])
        self.assertFalse(np.any(within_band[relaxed]))

        # The empty level cannot do better than its whole population share.
        self.assertAlmostEqual(imbalance[1], c[1], places=6)

        # The rows that absorbed the shortfall are reported by index, and the
        # band is reported as used: requested where it was met, achieved where
        # it was not.
        self.assertEqual(result.relaxed_rows, (1, 6))
        result_psi = result.psi
        assert result_psi is not None
        np.testing.assert_allclose(result_psi[10:], psi[10:])
        np.testing.assert_allclose(result_psi[relaxed], imbalance[relaxed])
        external_met = np.setdiff1d(np.arange(10), relaxed)
        np.testing.assert_array_less(
            psi[external_met] - 1e-12, result_psi[external_met]
        )
        np.testing.assert_array_less(result_psi[external_met], psi[external_met] * 1.01)
        self.assertIn("2 of 10 external validity constraints", result.fallback_message)
        self.assertIn(
            "We achieved the desired internal validity", result.fallback_message
        )

    def test_relaxation_continues_to_the_design_effect(self) -> None:
        """A relaxed band joins the background and the cascade carries on."""
        rake = self._l30_calibrator()
        rake.update_phi(50.0)  # loose enough to be met within the relaxed band
        result = rake.solve_with_fallback()
        self.assertEqual(result.fallback_status, 2)
        self.assertLessEqual(
            float(np.mean(np.square(result.solution))), 50.0 * (1 + 1e-6)
        )
        self.assertIn("achieved the desired design effect", result.fallback_message)

    def test_internal_relaxation_keeps_the_attainable_row(self) -> None:
        """Status 3 relaxes only the internal row that cannot be balanced.

        The first covariate is 0 throughout the control group and 1 throughout the
        treatment group, so its imbalance is 1 whatever the weights. The second is drawn
        from one distribution for both groups and can be balanced. Phase I alone leaves
        the second free to sit anywhere up to the first row's excess.

        """
        rng = np.random.default_rng(611)
        M1, M2 = 40, 40
        Z3 = np.column_stack([np.zeros(M1), rng.standard_normal(M1)])
        Z4 = np.column_stack([np.ones(M2), rng.standard_normal(M2) + 0.5])
        psi3 = 0.01
        rake = JointCalibrator(
            distance=SquaredL2(v=np.zeros(M1 + M2)),
            Z3=Z3,
            Z4=Z4,
            psi3=psi3,
        )
        result = rake.solve_with_fallback()
        self.assertEqual(result.fallback_status, 3)

        B, c = rake.B, rake.c
        assert B is not None and c is not None
        imbalance = np.abs(B @ result.solution - c)
        self.assertAlmostEqual(imbalance[0], 1.0, places=9)
        # Within its band, up to the budget's slack: 0.1% of the summed excess,
        # which here is mostly the first row's irreducible 1.
        self.assertLessEqual(imbalance[1], psi3 + 1e-3 * imbalance[0] + 1e-6)
        self.assertIn("1 of 2 internal validity constraints", result.fallback_message)

    def test_the_distance_chooses_how_tied_rows_share_the_excess(self) -> None:
        """Rows that trade excess one for one are settled by the distance.

        Three control rows, none of which can meet its band: an empty level whose target
        is 5, which sets the largest excess, and two rows asking for incompatible means
        of one 0/1 covariate, ``mean(x) = 1.5`` and ``mean(1 - x) = 1.5``. Every
        weighting with ``mean(x)`` in ``[0, 1]`` gives those two the same summed excess,
        so the linear program is indifferent among them, and both stay below the empty
        level's excess, so the pin on the largest does not decide either. The variance
        does: uniform weights are the least-variance weighting, with ``mean(x)`` at its
        sample value of 0.3. Holding each row at the excess the linear program chose
        would fix ``mean(x)`` wherever its solver stopped, near the middle of the tie.

        """
        M1, M2 = 50, 50
        x = np.r_[np.ones(15), np.zeros(35)]
        Z1 = np.column_stack([np.zeros(M1), x, 1.0 - x])
        Z2 = np.random.default_rng(613).uniform(size=(M2, 1))
        rake = JointCalibrator(
            distance=SquaredL2(v=np.zeros(M1 + M2)),
            Z1=Z1,
            Z2=Z2,
            nu1=np.array([5.0, 1.5, 1.5]),
            nu2=np.array([float(np.mean(Z2))]),
            psi1=0.01,
            psi2=0.01,
        )
        result = rake.solve_with_fallback()
        self.assertEqual(result.fallback_status, 2)
        self.assertEqual(result.relaxed_rows, (0, 1, 2))
        w1 = result.solution[:M1]
        self.assertAlmostEqual(float(np.dot(w1, x) / M1), 0.3, delta=2e-3)
        np.testing.assert_allclose(w1, 1.0, atol=1e-2)

    def test_a_band_too_tight_for_the_attainable_rows_is_widened(self) -> None:
        """When the shortfall must be split among rows that can each meet the band.

        Three levels, the first empty in the control group. The control shares of levels
        1 and 2 must sum to one against targets summing to 0.9. Each could meet a band
        of 0.02 standard deviations on its own, but together they can hold the 0.1
        excess only if the band is at least ``0.05 / sigma = 0.1005``, five times wider.
        Relaxing one of them would pin the other at the edge of its band, so the band is
        widened to that factor instead and only the empty level is relaxed: both levels
        end at the edge of the widened band, half each.

        """
        mu = np.array([0.1, 0.45, 0.45])
        sigma = np.sqrt(mu * (1 - mu))

        def whitened_indicators(counts: list[int]) -> npt.NDArray[np.float64]:
            levels = np.repeat(np.arange(len(counts)), counts)
            return np.eye(len(counts))[levels] / sigma

        Z1 = whitened_indicators([0, 20, 200])
        Z2 = whitened_indicators([40, 100, 100])
        M = Z1.shape[0] + Z2.shape[0]
        rake = JointCalibrator(
            distance=SquaredL2(v=np.zeros(M)),
            Z1=Z1,
            Z2=Z2,
            nu1=mu / sigma,
            nu2=mu / sigma,
            psi1=0.02,
            psi2=0.02,
        )
        result = rake.solve_with_fallback()
        self.assertEqual(result.fallback_status, 2)
        scale = result.band_scales["external validity"]
        self.assertAlmostEqual(scale, 0.05 / sigma[1] / 0.02, delta=0.01 * scale)
        self.assertEqual(result.relaxed_rows, (0,))
        self.assertIn("widened the external validity band", result.fallback_message)
        w1 = result.solution[: Z1.shape[0]]
        share = np.array([np.sum(w1[:20]), np.sum(w1[20:])]) / len(w1)
        np.testing.assert_allclose(share, [0.5, 0.5], atol=2e-3)
        result_psi = result.psi
        assert result_psi is not None
        np.testing.assert_allclose(result_psi[1:3], scale * 0.02, rtol=1e-3)

    def test_a_band_below_the_floor_is_widened_to_it(self) -> None:
        """The BLS counts at a band the observed levels cannot jointly meet.

        Holdout's four observed levels must exceed their targets by the empty level's
        3.36 pp in total, and a band of phi standard deviations allows them about 6.15
        pp x phi, so below phi of about 0.55 they cannot all meet it. Asked for 0.3, the
        band is widened to that floor; asked for 0.7, it is used as is. Either way only
        the two 8--14 rows are relaxed.

        """
        rake = self._l30_calibrator()
        M1, M2 = 506, 501
        for phi, widened in ((0.3, True), (0.7, False)):
            psi = np.asarray(rake.psi, dtype=float).copy()
            psi[:5] = phi / np.sqrt(M1)
            psi[5:10] = phi / np.sqrt(M2)
            rake.update_psi(psi)
            result = rake.solve_with_fallback()
            self.assertEqual(result.relaxed_rows, (1, 6))
            if widened:
                floor = phi * result.band_scales["external validity"]
                self.assertGreater(floor, 0.54)
                self.assertLess(floor, 0.56)
            else:
                self.assertEqual(result.band_scales, {})

    def test_feasible_band_is_reported_as_requested(self) -> None:
        """Verify a feasible band is reported as requested."""
        rng = np.random.default_rng(612)
        M1, M2, q = 60, 60, 2
        Z3 = rng.standard_normal((M1, q))
        Z4 = rng.standard_normal((M2, q))
        rake = JointCalibrator(
            distance=SquaredL2(v=np.zeros(M1 + M2)), Z3=Z3, Z4=Z4, psi3=0.5
        )
        result = rake.solve_with_fallback()
        self.assertEqual(result.fallback_status, 0)
        self.assertEqual(result.fallback_message, "Full solution achieved.")
        result_psi = result.psi
        assert result_psi is not None
        np.testing.assert_allclose(result_psi, np.full(q, 0.5))


class RakeCrossEngineTest(unittest.TestCase):
    """Tests for ``Rake.solve(engine='pd')`` cross-engine equivalence (P4)."""

    @parameterized.expand(
        [
            ("KL_M100_p20", 4001, KLDivergence, 100, 20, None, False, None),
            ("KL_M200_p30_phi1.5", 4002, KLDivergence, 200, 30, 1.5, False, None),
            ("L2_M100_p20", 4003, SquaredL2, 100, 20, None, False, None),
            ("L2_M200_p30_phi2.0", 4004, SquaredL2, 200, 30, 2.0, False, None),
            ("Huber_M100_p20", 4005, Huber, 100, 20, None, False, None),
            ("L2_M200_p5_imbalance", 4006, SquaredL2, 200, 5, None, True, None),
            ("L2_M200_p5_imbalance_phi2.0", 4007, SquaredL2, 200, 5, 2.0, True, None),
            ("L2_M100_p20_maxweight5", 4008, SquaredL2, 100, 20, None, False, 5.0),
        ]
    )
    def test_pd_engine_matches_ipm_engine(
        self,
        name: str,
        seed: int,
        distance_factory: type[Distance],
        M: int,
        p: int,
        phi: float | None,
        with_imbalance: bool,
        max_weight: float | None,
    ) -> None:
        """``Rake(engine='pd').solve()`` agrees with ``Rake(engine='ipm').solve()``.

        Cross-engine equivalence test: same problem data dispatched through both engines
        must produce solutions that agree at rtol/atol=1e-4. Covers KL/SquaredL2/Huber
        distances, with/without phi, with/without imbalance, and bounded/unbounded
        weights.

        """
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((M, p))
        w_target = rng.uniform(0.5, 2.0, size=M)
        mu = (1.0 / M) * X.T @ w_target

        kwargs: dict[str, Any] = {
            "distance": distance_factory(dimension=M),
            "X": X,
            "mu": mu,
            "phi": phi,
            "min_weight": 0.0,
            "max_weight": max_weight,
            "settings": OptimizationSettings(),
        }
        if with_imbalance:
            Z = rng.standard_normal((M, 30)) * 0.1
            nu = (1.0 / M) * Z.T @ w_target
            kwargs["Z"] = Z
            kwargs["nu"] = nu
            # Pick psi that's loose enough to leave room but not trivially infinite.
            base_resid = float(np.max(np.abs((1.0 / M) * Z.T @ np.ones(M) - nu)))
            kwargs["psi"] = 5.0 * base_resid + 1e-6

        rake_ipm = Rake(**kwargs, engine="ipm")
        res_ipm = rake_ipm.solve()
        rake_pd = Rake(**kwargs, engine="pd")
        res_pd = rake_pd.solve()

        np.testing.assert_allclose(
            res_pd.solution, res_ipm.solution, rtol=1e-4, atol=1e-4
        )
        # Both should land feasible with respect to bounds.
        self.assertTrue(
            np.all(res_pd.solution >= -1e-6),
            f"PD solution violates lower bound; min={res_pd.solution.min()!r}",
        )
        if max_weight is not None:
            self.assertTrue(
                np.all(res_pd.solution <= max_weight + 1e-6),
                f"PD solution violates upper bound; max={res_pd.solution.max()!r}, bound={max_weight!r}",
            )
        # Both should satisfy the equality constraints.
        A_eq = (1.0 / M) * np.hstack([X, np.ones((M, 1))]).T
        mu_full = np.append(mu, 1.0)
        np.testing.assert_allclose(A_eq @ res_pd.solution, mu_full, atol=1e-4)

    def test_engine_invalid_raises(self) -> None:
        """Constructor rejects invalid engine values."""
        rng = np.random.default_rng(0)
        M, p = 10, 3
        X = rng.standard_normal((M, p))
        mu = (1.0 / M) * X.T @ np.ones(M)
        with self.assertRaisesRegex(ValueError, "engine must be one of"):
            Rake(
                distance=SquaredL2(dimension=M),
                X=X,
                mu=mu,
                engine=cast(Literal["ipm", "pd"], "bogus"),
            )

    def test_pd_default_x0_clamped_to_max_weight(self) -> None:
        """PD path clamps the auto-default ``x0=ones(M)`` to ``max_weight < 1.0``.

        Without clamping, the default ``x0=ones(M)`` violates the documented
        bounds-feasibility contract whenever ``max_weight < 1.0``, and the PD solve
        raises. The clamp lifts ``x0`` into the feasible box so the contract holds for
        the auto-default.

        """
        rng = np.random.default_rng(20260513)
        M, p = 80, 4
        X = rng.standard_normal((M, p))
        # Pick a target weight vector with mean below max_weight so the
        # equality constraint is satisfiable inside the box.
        w_target = rng.uniform(0.1, 0.4, size=M)
        mu = (1.0 / M) * X.T @ w_target
        rake = Rake(
            distance=SquaredL2(
                dimension=M
            ),  # SquaredL2 default has no v -> x0 = ones(M)
            X=X,
            mu=mu,
            min_weight=0.0,
            max_weight=0.5,  # < 1.0 so ones(M) is bounds-infeasible
            constrain_mean_weight_to=None,
            engine="pd",
        )
        res = rake.solve()  # x0 auto-default; should not raise
        self.assertTrue(
            np.all(res.solution <= 0.5 + 1e-6),
            f"Solution exceeds upper bound 0.5; max={res.solution.max()!r}",
        )
        self.assertTrue(
            np.all(res.solution >= -1e-6),
            f"Solution violates lower bound; min={res.solution.min()!r}",
        )

    def test_pd_default_x0_clamped_when_distance_v_out_of_bounds(self) -> None:
        """PD path clamps auto-default ``x0=distance.v`` when it exceeds the cap."""
        rng = np.random.default_rng(20260514)
        M, p = 80, 4
        X = rng.standard_normal((M, p))
        w_target = rng.uniform(0.5, 1.5, size=M)
        mu = (1.0 / M) * X.T @ w_target
        # `v` deliberately above max_weight so `distance.v` (the auto-default
        # for x0) would be bounds-infeasible without clamping.
        v_out_of_box = np.full(M, 5.0)
        rake = Rake(
            distance=KLDivergence(v=v_out_of_box),
            X=X,
            mu=mu,
            min_weight=0.0,
            max_weight=2.0,
            constrain_mean_weight_to=None,
            engine="pd",
        )
        res = rake.solve()  # x0 auto-default to v then clipped; should not raise
        self.assertTrue(
            np.all(res.solution <= 2.0 + 1e-6),
            f"Solution exceeds upper bound 2.0; max={res.solution.max()!r}",
        )
        self.assertTrue(
            np.all(res.solution >= -1e-6),
            f"Solution violates lower bound; min={res.solution.min()!r}",
        )

    def test_ipm_default_x0_applies_and_clamps_to_max_weight(self) -> None:
        """The domain-informed auto-default now feeds the IPM engine too.

        Previously only the PD path used ``_default_x0``; the IPM path passed ``None``
        and let the barrier seed a bare min-norm point. Now both engines share the
        clamped ``self.distance.v`` / ``ones`` warm start, so the IPM solve must run
        cleanly and return an in-bounds solution even when the bare default ``ones(M)``
        is outside the box (``max_weight < 1.0``).

        """
        rng = np.random.default_rng(20260513)
        M, p = 80, 4
        X = rng.standard_normal((M, p))
        w_target = rng.uniform(0.1, 0.4, size=M)
        mu = (1.0 / M) * X.T @ w_target
        rake = Rake(
            distance=SquaredL2(dimension=M),  # no v -> x0 = ones(M)
            X=X,
            mu=mu,
            min_weight=0.0,
            max_weight=0.5,  # < 1.0 so ones(M) is bounds-infeasible
            constrain_mean_weight_to=None,
            engine="ipm",
        )
        res = rake.solve()  # x0 auto-default; should not raise
        self.assertTrue(
            np.all(res.solution <= 0.5 + 1e-6),
            f"Solution exceeds upper bound 0.5; max={res.solution.max()!r}",
        )
        self.assertTrue(
            np.all(res.solution >= -1e-6),
            f"Solution violates lower bound; min={res.solution.min()!r}",
        )

    def test_ipm_default_x0_clamped_when_distance_v_out_of_bounds(self) -> None:
        """IPM path clamps the auto-default ``x0=distance.v`` when v exceeds max_weight."""
        rng = np.random.default_rng(20260514)
        M, p = 80, 4
        X = rng.standard_normal((M, p))
        w_target = rng.uniform(0.5, 1.5, size=M)
        mu = (1.0 / M) * X.T @ w_target
        v_out_of_box = np.full(M, 5.0)
        rake = Rake(
            distance=KLDivergence(v=v_out_of_box),
            X=X,
            mu=mu,
            min_weight=0.0,
            max_weight=2.0,
            constrain_mean_weight_to=None,
            engine="ipm",
        )
        res = rake.solve()
        self.assertTrue(
            np.all(res.solution <= 2.0 + 1e-6),
            f"Solution exceeds upper bound 2.0; max={res.solution.max()!r}",
        )
        self.assertTrue(
            np.all(res.solution >= -1e-6),
            f"Solution violates lower bound; min={res.solution.min()!r}",
        )


class JointCalibratorCrossEngineTest(unittest.TestCase):
    """Tests for ``JointCalibrator`` engine-parameter plumbing."""

    def test_engine_invalid_raises(self) -> None:
        """JointCalibrator constructor rejects invalid engine values."""
        rng = np.random.default_rng(602)
        M1, M2 = 20, 20
        q = 2
        Z1 = rng.standard_normal((M1, q))
        Z2 = rng.standard_normal((M2, q))
        nu1 = np.mean(Z1, axis=0)
        nu2 = np.mean(Z2, axis=0)
        with self.assertRaisesRegex(ValueError, "engine must be one of"):
            JointCalibrator(
                distance=KLDivergence(dimension=M1 + M2),
                Z1=Z1,
                Z2=Z2,
                nu1=nu1,
                nu2=nu2,
                psi1=0.5,
                psi2=0.5,
                engine=cast(Literal["ipm", "pd"], "bogus"),
            )


class RakeProblemAssemblyTest(unittest.TestCase):
    """The default IPM path assembles its Newton step via a structured ``Problem``.

    ``engine="ipm"`` caches the ``Problem`` built by ``Rake._build_problem``. Because
    its ``Distance`` objective (and layers) declare structured (diagonal/zero/low-rank)
    Hessians, ``BarrierSolver`` routes it through ``build_hessian_system`` / structured
    assembly. Every component a ``Rake`` stacks (Distance objective; Bounds, Imbalance,
    SquaredNorm layers) contributes a diagonal-or-zero shape, so the full stack
    assembles via the structured path.

    """

    def test_cached_problem_is_diagonal_with_all_layers(self) -> None:
        """Bounds + Imbalance + SquaredNorm all assemble via the structured path."""
        np.random.seed(411)
        M, p, q = 60, 8, 4
        X = np.random.rand(M, p)
        Z = np.random.rand(M, q)
        w = np.random.rand(M) + 0.5
        w /= np.mean(w)
        # why: scalar-times-array infers as float; reassert the array dtype.
        mu = np.asarray((1 / M) * (X.T @ w), dtype=np.float64)
        nu = np.asarray((1 / M) * (Z.T @ w), dtype=np.float64)
        phi = float(2.0 * np.mean(np.square(w)))

        rake = Rake(
            distance=KLDivergence(dimension=M),
            X=X,
            mu=mu,
            Z=Z,
            nu=nu,
            psi=0.5,
            phi=phi,
            min_weight=0.01,
            max_weight=5.0,
            constrain_mean_weight_to=None,
        )

        # The cached IPM Problem assembles via build_hessian_system because
        # its objective/layers declare structured (diagonal/zero/low-rank)
        # Hessians, so the full inequality stack (Bounds + Imbalance +
        # SquaredNorm) routes through the structured assembler.
        self.assertIsInstance(rake._problem, Problem)
        self.assertEqual(len(rake._problem.ineq), 3)

        # And it still reaches the analytic optimum through the v2 assembler.
        res = rake.solve()
        np.testing.assert_allclose(X.T @ res.solution / M, mu, atol=1e-6)
        np.testing.assert_array_less(0.01, res.solution)
        np.testing.assert_array_less(res.solution, 5.0)
        self.assertLessEqual(_barrier_diag(res).duality_gaps[-1], 1e-6)

    def test_cached_problem_is_diagonal_bounds_only(self) -> None:
        """The minimal stack (bounds only) is still a structured ``Problem``."""
        np.random.seed(412)
        M, p = 40, 5
        X = np.random.rand(M, p)
        w = np.random.rand(M) + 0.5
        w /= np.mean(w)
        # why: scalar-times-array infers as float; reassert the array dtype.
        mu = np.asarray((1 / M) * (X.T @ w), dtype=np.float64)

        rake = Rake(
            distance=SquaredL2(dimension=M),
            X=X,
            mu=mu,
            constrain_mean_weight_to=None,
        )

        self.assertIsInstance(rake._problem, Problem)
        # Bounds-only stack: exactly one constraint layer.
        self.assertEqual(len(rake._problem.ineq), 1)

        # And it still reaches the calibration optimum through the v2 assembler.
        res = rake.solve()
        np.testing.assert_allclose(X.T @ res.solution / M, mu, atol=1e-6)
        self.assertLessEqual(_barrier_diag(res).duality_gaps[-1], 1e-6)


class FallbackEffectiveSampleSizeTest(unittest.TestCase):
    """Effective-sample-size properties on ``FallbackOptimizationResult``."""

    @staticmethod
    def _result(
        solution: npt.NDArray[np.float64], m1: int
    ) -> FallbackOptimizationResult:
        """Wrap a weight vector in a minimal result split at ``m1``."""
        return FallbackOptimizationResult(
            result=SolveResult(
                solution=solution,
                status=SolveStatus.OPTIMAL,
                message="ok",
                solver="barrier",
                n_iterations=1,
            ),
            fallback_message="Full solution achieved.",
            fallback_status=0,
            m1=m1,
        )

    def test_kish_per_block(self) -> None:
        """Each block uses Kish's (sum w)^2 / sum w^2 on its own slice."""
        # control [1,1,1,1] -> 4; treated [1,3] -> 16/10; combined -> 64/14.
        res = self._result(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 3.0]), m1=4)
        self.assertAlmostEqual(res.control_effective_sample_size, 4.0)
        self.assertAlmostEqual(res.treatment_effective_sample_size, 1.6)
        self.assertAlmostEqual(res.combined_effective_sample_size, 64.0 / 14.0)

    def test_uniform_weights_equal_counts(self) -> None:
        """Uniform weights give n_eff equal to the raw block sizes."""
        res = self._result(np.ones(10), m1=6)
        self.assertAlmostEqual(res.control_effective_sample_size, 6.0)
        self.assertAlmostEqual(res.treatment_effective_sample_size, 4.0)
        self.assertAlmostEqual(res.combined_effective_sample_size, 10.0)

    def test_scale_invariance(self) -> None:
        """Kish's ESS does not depend on the weight normalization."""
        w = np.array([0.5, 2.0, 1.0, 4.0])
        a = self._result(w, m1=2)
        b = self._result(10.0 * w, m1=2)
        self.assertAlmostEqual(
            a.control_effective_sample_size, b.control_effective_sample_size
        )
        self.assertAlmostEqual(
            a.treatment_effective_sample_size, b.treatment_effective_sample_size
        )
        self.assertAlmostEqual(
            a.combined_effective_sample_size, b.combined_effective_sample_size
        )

    def test_empty_control_block(self) -> None:
        """An empty block has zero ESS; combined then equals the other block."""
        res = self._result(np.array([1.0, 2.0, 3.0]), m1=0)
        self.assertEqual(res.control_effective_sample_size, 0.0)
        self.assertAlmostEqual(
            res.treatment_effective_sample_size, res.combined_effective_sample_size
        )

    def test_all_zero_weights(self) -> None:
        """All-zero weights give zero ESS with no division by zero."""
        res = self._result(np.zeros(4), m1=2)
        self.assertEqual(res.control_effective_sample_size, 0.0)
        self.assertEqual(res.treatment_effective_sample_size, 0.0)
        self.assertEqual(res.combined_effective_sample_size, 0.0)

    def test_solve_with_fallback_sets_m1_and_splits(self) -> None:
        """solve_with_fallback records m1 and splits the solution correctly."""
        np.random.seed(601)
        M1, M2, q = 30, 20, 3
        Z1 = np.random.randn(M1, q)
        Z2 = np.random.randn(M2, q)
        Z3 = np.random.randn(M1, q)
        Z4 = np.random.randn(M2, q)
        nu1 = np.mean(Z1, axis=0)
        nu2 = np.mean(Z2, axis=0)

        rake = JointCalibrator(
            distance=KLDivergence(dimension=M1 + M2),
            Z1=Z1,
            Z2=Z2,
            Z3=Z3,
            Z4=Z4,
            nu1=nu1,
            nu2=nu2,
            psi1=0.5,
            psi2=0.5,
            psi3=0.5,
            phi=10.0,
        )
        result = rake.solve_with_fallback()

        self.assertEqual(result.m1, M1)
        sol = result.solution
        self.assertAlmostEqual(
            result.control_effective_sample_size,
            float(np.square(sol[:M1].sum()) / np.square(sol[:M1]).sum()),
        )
        self.assertAlmostEqual(
            result.treatment_effective_sample_size,
            float(np.square(sol[M1:].sum()) / np.square(sol[M1:]).sum()),
        )
        # An effective sample size never exceeds its raw block size.
        self.assertGreater(result.control_effective_sample_size, 0.0)
        self.assertLessEqual(result.control_effective_sample_size, M1 + 1e-6)
        self.assertLessEqual(result.treatment_effective_sample_size, M2 + 1e-6)
        self.assertLessEqual(result.combined_effective_sample_size, M1 + M2 + 1e-6)
