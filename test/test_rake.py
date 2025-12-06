"""Test Rake."""

import time

import numpy as np
import pytest
from scipy.special import expit, logit


from pyrake.distance_metrics import (
    Huber,
    KLDivergence,
    SquaredL2,
)
from pyrake.optimization import (
    InteriorPointMethodResult,
    OptimizationSettings,
    ProblemCertifiablyInfeasibleError,
)
from pyrake.rake import Rake


@pytest.mark.parametrize(
    "seed,M,p,phi",
    [
        (101, 100, 20, None),
        (201, 200, 30, 1.5),
        (301, 50, 5, 1.5),
        (401, 500, 100, None),
        (501, 13, 3, 1.5),
    ],
)
def test_rake_solve_kl_divergence(seed: int, M: int, p: int, phi: float) -> None:
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
    assert np.all(w > min_weight)

    # Compute population mean
    mu = (1 / M) * (X.T @ w)

    # Now forget we know w. We just know X for respondents and mu for the target
    # population.
    start_time = time.time()
    rake = Rake(
        distance=KLDivergence(),
        X=X,
        mu=mu,
        phi=phi,
        min_weight=min_weight,
        settings=OptimizationSettings(verbose=True),
    )
    res = rake.solve()
    end_time = time.time()
    print(f"Complete in {1e3 * (end_time - start_time):.03f} ms")

    # Verify solution is feasible
    np.testing.assert_allclose((1 / M) * X.T @ res.solution, mu)
    assert np.all(res.solution > min_weight)
    if phi is not None:
        assert np.dot(res.solution, res.solution) < M * phi

    assert isinstance(res, InteriorPointMethodResult)
    assert res.duality_gaps[-1] <= 1e-6

    # Now test specifying baseline weights
    estimated_propensity = expit(logit(true_propensity) + 0.1 * np.random.randn(M))

    start_time = time.time()
    rake = Rake(
        distance=KLDivergence(v=1.0 / estimated_propensity),
        X=X,
        mu=mu,
        phi=phi,
        min_weight=min_weight,
        settings=OptimizationSettings(verbose=True),
    )
    # Use v as the initial guess
    res = rake.solve(x0=1.0 / estimated_propensity)
    end_time = time.time()
    print(f"Complete in {1e3 * (end_time - start_time):.03f} ms")

    # Verify solution is feasible
    np.testing.assert_allclose((1 / M) * X.T @ res.solution, mu)
    assert np.all(res.solution > min_weight)
    if phi is not None:
        assert np.dot(res.solution, res.solution) < M * phi

    assert isinstance(res, InteriorPointMethodResult)
    assert res.duality_gaps[-1] <= 1e-6


@pytest.mark.parametrize(
    "seed,M,p,phi",
    [
        (102, 100, 20, None),
        (202, 200, 30, 2.0),
        (302, 50, 5, 2.0),
        (702, 500, 100, None),
        (502, 13, 3, 2.0),
    ],
)
def test_rake_solve_squaredl2(seed: int, M: int, p: int, phi: float) -> None:
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
    mu = (1 / M) * (X.T @ w)

    # Now forget we know w. We just know X for respondents and mu for the target
    # population.
    start_time = time.time()
    rake = Rake(
        distance=SquaredL2(),
        X=X,
        mu=mu,
        phi=phi,
        settings=OptimizationSettings(verbose=True),
    )
    res = rake.solve()
    end_time = time.time()
    print(f"Complete in {1e3 * (end_time - start_time):.03f} ms")

    # Verify solution is feasible
    np.testing.assert_allclose((1 / M) * X.T @ res.solution, mu)
    assert np.all(res.solution > 0)
    if phi is not None:
        assert np.dot(res.solution, res.solution) < M * phi

    assert isinstance(res, InteriorPointMethodResult)
    assert res.duality_gaps[-1] <= 1e-6

    # Now test specifying baseline weights
    estimated_propensity = expit(logit(true_propensity) + 0.1 * np.random.randn(M))

    start_time = time.time()
    rake = Rake(
        distance=SquaredL2(v=1.0 / estimated_propensity),
        X=X,
        mu=mu,
        phi=phi,
        settings=OptimizationSettings(verbose=True),
    )
    # Use v as the initial guess
    res = rake.solve(x0=1.0 / estimated_propensity)
    end_time = time.time()
    print(f"Complete in {1e3 * (end_time - start_time):.03f} ms")

    # Verify solution is feasible
    np.testing.assert_allclose((1 / M) * X.T @ res.solution, mu)
    assert np.all(res.solution > 0)
    if phi is not None:
        assert np.dot(res.solution, res.solution) < M * phi

    assert isinstance(res, InteriorPointMethodResult)
    assert res.duality_gaps[-1] <= 1.1e-6


@pytest.mark.parametrize(
    "seed,M,p,phi",
    [
        (103, 100, 20, None),
        (203, 200, 30, 1.5),
        (303, 50, 5, 2.0),
        (403, 500, 100, 2.5),
        (503, 13, 3, 1.5),
    ],
)
def test_rake_solve_huber(seed: int, M: int, p: int, phi: float) -> None:
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
    mu = (1 / M) * (X.T @ w)

    # Now forget we know w. We just know X for respondents and mu for the target
    # population.
    start_time = time.time()
    rake = Rake(
        distance=Huber(),
        X=X,
        mu=mu,
        phi=phi,
        settings=OptimizationSettings(verbose=True),
    )
    res = rake.solve()
    end_time = time.time()
    print(f"Complete in {1e3 * (end_time - start_time):.03f} ms")

    # Verify solution is feasible
    np.testing.assert_allclose((1 / M) * X.T @ res.solution, mu)
    assert np.all(res.solution > 0)
    if phi is not None:
        assert np.dot(res.solution, res.solution) < M * phi

    assert isinstance(res, InteriorPointMethodResult)
    assert res.duality_gaps[-1] <= 1e-6

    # Now test specifying baseline weights
    estimated_propensity = expit(logit(true_propensity) + 0.1 * np.random.randn(M))

    start_time = time.time()
    rake = Rake(
        distance=Huber(v=1.0 / estimated_propensity),
        X=X,
        mu=mu,
        phi=phi,
        settings=OptimizationSettings(verbose=True),
    )
    # Use v as the initial guess
    res = rake.solve(x0=1.0 / estimated_propensity)
    end_time = time.time()
    print(f"Complete in {1e3 * (end_time - start_time):.03f} ms")

    # Verify solution is feasible
    np.testing.assert_allclose((1 / M) * X.T @ res.solution, mu)
    assert np.all(res.solution > 0)
    if phi is not None:
        assert np.dot(res.solution, res.solution) < M * phi

    assert isinstance(res, InteriorPointMethodResult)
    assert res.duality_gaps[-1] <= 1e-6


@pytest.mark.parametrize(
    "seed,M,p",
    [
        (104, 100, 20),
        (204, 200, 30),
        (304, 50, 5),
        (404, 500, 100),
        (504, 13, 3),
    ],
)
def test_rake_solve_infeasible(seed: int, M: int, p: int) -> None:
    """Test a scenario where the problem is infeasible.

    See test_phase1solvers.py::TestEqualityWithBoundsSolver::test_solver_infeasible.

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

    phi = 1_000.0

    rake = Rake(
        distance=Huber(),
        X=M * A.T,
        mu=b,
        phi=phi,
        settings=OptimizationSettings(verbose=True),
    )

    with pytest.raises(ProblemCertifiablyInfeasibleError):
        rake.solve()


@pytest.mark.parametrize(
    "seed,M,p1,p2",
    [
        (105, 100, 10, 15),
        (205, 200, 5, 30),
        (305, 50, 5, 5),
        (405, 500, 20, 30),
        (505, 13, 3, 3),
    ],
)
def test_rake_max_imbalance(seed: int, M: int, p1: int, p2: int) -> None:
    """Test a scenario with a constraint on the max covariate imbalance."""
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

    b = A @ w
    psi = 0.05
    c = B @ w + (2 * psi) * np.random.rand(p2) - psi

    phi = float(2.0 * np.mean(w * w))

    rake = Rake(
        distance=KLDivergence(),
        X=M * A.T,
        mu=b,
        Z=M * B.T,
        nu=c,
        psi=psi,
        phi=phi,
        settings=OptimizationSettings(verbose=True),
    )

    res = rake.solve()

    # Verify we find a feasible point
    np.testing.assert_allclose(A @ res.solution, b)
    assert np.all(res.solution > 0)
    assert np.all(B @ res.solution - c > -psi)
    assert np.all(B @ res.solution - c < psi)
    assert np.mean(res.solution * res.solution) < phi

    # Test with vector psi
    rake = Rake(
        distance=KLDivergence(),
        X=M * A.T,
        mu=b,
        Z=M * B.T,
        nu=c,
        psi=np.full((p2,), psi),
        phi=phi,
        settings=OptimizationSettings(verbose=True),
    )

    res = rake.solve()

    # Verify we find a feasible point
    np.testing.assert_allclose(A @ res.solution, b)
    assert np.all(res.solution > 0)
    assert np.all(B @ res.solution - c > -psi)
    assert np.all(B @ res.solution - c < psi)
    assert np.mean(res.solution * res.solution) < phi
