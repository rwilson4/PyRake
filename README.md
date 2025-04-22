
# PyRake

PyRake is a Python library for computing balancing weights using
convex optimization. It supports multiple distance functions, exact
covariate balancing, and an interior-point method for tracing the
bias–variance tradeoff frontier.

## Installation

```bash
pip install .
```

## Example Usage

```python
from pyrake import Rake, KLDivergence, EfficientFrontier

# Inputs: X (M x p), mu (p,), v (M,)
rake = Rake(distance=KLDivergence(), phi=1.0)
ef = EfficientFrontier(rake)

results = ef.trace(X, mu, v=v, num_points=30)
results.plot()
```

## License

Apache
