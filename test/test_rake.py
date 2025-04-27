"""Test Rake."""

import numpy as np
import pytest

from pyrake.distance_metrics import KLDivergence, SquaredL2, Huber
from pyrake.exceptions import ProblemInfeasibleError
from pyrake.rake import Rake


@pytest.mark.parametrize("dist", [KLDivergence(), SquaredL2(), Huber()])
def test_rake_solve_returns_feasible_weights(dist) -> None:
    np.random.seed(0)
    M, p = 200, 5
    X = np.random.rand(M, p)
    mu = X.mean(axis=0)
    phi = 1.5

    rake = Rake(distance=dist, X=X, mu=mu, phi=phi)
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
    rake = Rake(distance=dist, X=X, mu=mu, phi=phi)

    with pytest.raises(ProblemInfeasibleError):
        rake.solve()
