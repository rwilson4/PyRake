"""Test EfficientFrontier."""

import time

import numpy as np
import pytest
from scipy.special import logit, expit

from pyrake.distance_metrics import KLDivergence
from pyrake.frontier import EfficientFrontier
from pyrake.rake import Rake


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

    assert efr.variances[0] <= efr.variances[-1]
    assert efr.distances[-1] <= efr.distances[0]

    variance_reduction = (
        efr.variances[-1] - efr.variances[efr.max_chord_distance()]
    ) / (efr.variances[-1] - efr.variances[0])

    bias_increase = (efr.distances[efr.max_chord_distance()] - efr.distances[-1]) / (
        efr.distances[0] - efr.distances[-1]
    )

    weights = efr.weights[efr.max_chord_distance()]
    np.testing.assert_allclose((1 / M) * (X.T @ weights), mu)
    assert np.all(weights >= 0)

    # 80/20 rule -- more like 70/30 in our case
    assert variance_reduction > 0.7
    assert bias_increase < 0.3
