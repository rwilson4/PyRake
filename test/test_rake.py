"""Test Rake."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyrake.distance_metrics import KLDivergence, SquaredL2, Huber
from pyrake.exceptions import ProblemInfeasibleError
from pyrake.optimization import InteriorPointMethodResult, OptimizationSettings
from pyrake.rake import Rake


def test_rake_solve() -> None:
    np.random.seed(23)
    M, p = 100, 20
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
    phi = 1.5

    rake = Rake(
        distance=KLDivergence(),
        X=X,
        mu=mu,
        phi=phi,
        settings=OptimizationSettings(verbose=True),
    )
    res = rake.solve()
    # if isinstance(res, InteriorPointMethodResult):
    #     res.plot_convergence()
    #     plt.show()


@pytest.mark.parametrize("dist", [KLDivergence(), SquaredL2(), Huber()])
def test_rake_solve_returns_feasible_weights(dist) -> None:
    np.random.seed(0)
    M, p = 200, 5
    X = np.random.rand(M, p)
    mu = X.mean(axis=0)
    phi = 1.5

    rake = Rake(distance=dist, X=X, mu=mu, phi=phi, constrain_mean_weight_to=None)
    res = rake.solve()
    w = res.solution

    # Feasibility checks
    assert np.all(w > 0)
    np.testing.assert_allclose((X.T @ w) / M, mu, atol=1e-6)
    assert np.sum(w**2) / M < phi + 1e-6


def test_phase1_infeasible() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    mu = np.array([100.0, 100.0])  # clearly infeasible
    phi = 1.0
    dist = SquaredL2()
    rake = Rake(distance=dist, X=X, mu=mu, phi=phi, constrain_mean_weight_to=None)

    with pytest.raises(ProblemInfeasibleError):
        rake.solve()
