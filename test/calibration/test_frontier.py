"""Test EfficientFrontier and ImbalanceVarianceFrontier."""

import time

import numpy as np
import pytest
from scipy.special import expit, logit

from pyrake.calibration.distance_metrics import KLDivergence
from pyrake.calibration.frontier import EfficientFrontier, ImbalanceVarianceFrontier
from pyrake.calibration.rake import Rake


@pytest.mark.parametrize(
    "seed,M,p",
    [
        (101, 100, 20),
        (201, 200, 30),
        (301, 50, 5),
        (401, 500, 100),
        (501, 13, 3),
    ],
)
def test_frontier(seed: int, M: int, p: int) -> None:
    """Test EfficientFrontier."""
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

    # Now test specifying baseline weights
    estimated_propensity = expit(logit(true_propensity) + 0.1 * np.random.randn(M))

    rake = Rake(
        distance=KLDivergence(v=1.0 / estimated_propensity),
        X=X,
        mu=mu,
        phi=2.0,
        # settings=OptimizationSettings(verbose=True),
    )

    frontier = EfficientFrontier(rake)
    start_time = time.time()
    efr = frontier.trace()
    end_time = time.time()
    print(f"Complete in {1e3 * (end_time - start_time):.03f} ms")

    # efr.plot()
    # plt.show()

    # Convenience properties are sorted by variance ascending.
    assert efr.variances[0] <= efr.variances[-1]
    assert efr.distances[-1] <= efr.distances[0]

    # corners[0] minimizes distance; corners[1] minimizes variance.
    assert efr.corners[1].objectives[1] <= efr.corners[0].objectives[1]
    assert efr.corners[1].objectives[0] >= efr.corners[0].objectives[0]

    # knee() replaces max_chord_distance().
    knee = efr.knee()
    weights = knee.solution
    np.testing.assert_allclose((1 / M) * (X.T @ weights), mu)
    assert np.all(weights >= 0)

    variance_reduction = (efr.variances[-1] - knee.objectives[1]) / (
        efr.variances[-1] - efr.variances[0]
    )
    bias_increase = (knee.objectives[0] - efr.distances[-1]) / (
        efr.distances[0] - efr.distances[-1]
    )

    # 80/20 rule -- more like 70/30 in our case
    assert variance_reduction > 0.7
    assert bias_increase < 0.3


@pytest.mark.parametrize(
    "seed,M,p,q",
    [
        (101, 100, 10, 5),
        (201, 200, 15, 8),
        (301, 50, 5, 3),
        (401, 300, 20, 10),
    ],
)
def test_imbalance_variance_frontier(seed: int, M: int, p: int, q: int) -> None:
    """Test ImbalanceVarianceFrontier."""
    np.random.seed(seed)
    X = np.random.rand(M, p)
    Z = np.random.rand(M, q)

    # Simulate propensity scores and compute population means
    s = 0.1 * (1 - 0.1) / (0.05 * 0.1 * (1 - 0.1)) - 1
    true_propensity = np.random.beta(0.1 * s, 0.9 * s, size=M)
    w_true = 1.0 / true_propensity
    w_true /= np.mean(w_true)

    mu = (1 / M) * (X.T @ w_true)
    nu = (1 / M) * (Z.T @ w_true)

    estimated_propensity = expit(logit(true_propensity) + 0.1 * np.random.randn(M))

    rake = Rake(
        distance=KLDivergence(v=1.0 / estimated_propensity),
        X=X,
        mu=mu,
        Z=Z,
        nu=nu,
    )

    frontier = ImbalanceVarianceFrontier(rake)
    start_time = time.time()
    ivfr = frontier.trace()
    end_time = time.time()
    print(f"Complete in {1e3 * (end_time - start_time):.03f} ms")

    # Convenience properties are sorted by imbalance ascending.
    assert ivfr.imbalances[0] <= ivfr.imbalances[-1]

    # Variance increases as imbalance bound tightens.
    assert ivfr.variances[0] >= ivfr.variances[-1]

    # corners[0] minimizes variance; corners[1] minimizes imbalance.
    assert ivfr.corners[0].objectives[0] <= ivfr.corners[1].objectives[0]
    assert ivfr.corners[1].objectives[1] <= ivfr.corners[0].objectives[1]

    # knee() satisfies equality constraints and positivity.
    knee = ivfr.knee()
    weights = knee.solution
    np.testing.assert_allclose((1 / M) * (X.T @ weights), mu, atol=1e-4)
    assert np.all(weights >= 0)
