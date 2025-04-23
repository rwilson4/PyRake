
# PyRake

PyRake is a Python library for calculating balancing weights to adjust
surveys for non-response bias. Balancing weights try to accomplish
three things:
1. Reduce bias
2. Balance important covariates
3. Reduce variance

If we knew the probability each person would respond to the survey, a
natural choice of weight is: <br/>
$` w_i = (M/N) / \pi_i, `$
where M people respond to the survey out of a total population size of
N, and $` \pi_i `$ is the response propensity score. Then
$` (1/M) \sum_i w_i \cdot Y_i `$ is unbiased for the population
average.
Typically the propensity scores are unknown and must be estimated, but
in that scenario the best we can hope for is the bias is small. Given
that an unbiased estimator is unattainable, we might as well see what
else weights can give us.

We might ask the weights exactly balance important covariates. If
$` X_i `$ is the covariate value for person i, and we know the mean of
this covariate is $` \mu `$ in the population, we might demand
$` (1 / M) \sum_i w_i \cdot X_i = \mu, `$ and we might demand this not
just for one covariate but perhaps several important ones believed to
be correlated with the response of interest. Enforcing this constraint
may force us to deviate from estimated propensity scores, but since
those are just estimates, forcing balance on important covariates may
actually /reduce/ the bias.

The variance of the estimator is proportional to $` \sum_i w_i^2, `$
so all else being equal, we would prefer small weights to larger ones.
This leads to a bias/variance tradeoff: staying close to the estimated
propensity scores keeps the bias in check, while pursuing smaller
weights reduces variance.

PyRake can be used for solving problems of the form:<br />
$`
\begin{array}{ll}
\textrm{minimize}    & D(w, v) \\
\textrm{subject to}  & (1/M) X^T w = \mu \\
           &  \| w \|_2^2 \leq \phi \\
           & w \succeq 0,
\end{array}
`$<br />
where D is a distance metric that keeps w close to v = (M/N)/estimated
propensity. We support multiple distance metrics, including an L2
error metric, KL Divergence, and a Huber penalty. If the user does not
have estimated propensity scores, the code defaults to v=1.

PyRake can also be used to solve a sequence of these problems, with
varying $` \phi `$. This allows the user to visualize the bias/variance
tradeoff.

## Connection with Other Methods
Popular methods for reweighting include Raking, Entropy Balancing, and
Stable Balancing Weights. These can be seen as special cases or
modifications of the problem family PyRake solves.

Raking (Deming and Stephan, 1940) solves:<br />
$`
\begin{array}{ll}
\textrm{minimize}    & D(w, v) \\
\textrm{subject to}  & (1/M) X^T w = \mu \\
           &  \| w \|_2^2 \leq \phi \\
           & w \succeq 0,
\end{array}
`$<br />
where D(w, v) is the KL divergence; v is typically chosen to be
uniform. Raking sets $` \phi `$ to be large enough so that the
variance constraint is not active; equivalently the constraint is
simply omitted. As the name implies, PyRake can be used to calculate
Raking weights.

Like Raking, Entropy Balancing (Hainmuller, 2012) uses KL divergence
as the distance metric and omits the variance constraint. An
additional constraint is applied: $` 1^T w = 1^T v`$. This constraint
can be accommodated by adding an extra column to X having all ones;
the corresponding entry in $`\mu`$ should be $` M \cdot 1^T v`$. This
could be handled more conveniently, but it's there.

Stable Balancing Weights (Zubizarreta, 2015) uses
$` D(w, v) = \| w - v \|_2^2, `$ which is supported by PyRake.

### References
W. Edwards Deming and Frederick F. Stephan, "On a least squares
adjustment of a sampled frequency table when the expected marginal
totals are known" (1940). Annals of Mathematical Statistics.

Jens Hainmuller, "Entropy balancing for causal effects: A multivariate
reweighting method to produce balanced samples in observational
studies" (2012). Political Analysis.

José R. Zubizarreta, "Stable weights that balance covariates for
estimation with incomplete outcome data." (2015). Journal of the
American Statistical Association.

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

## Development
### Architecture
We use [poetry](https://python-poetry.org/) to manage dependencies.
Test cases use [pytest](https://docs.pytest.org/en/latest/). We use
[black](https://github.com/python/black) and
[ruff](https://docs.astral.sh/ruff/) for formatting.

### Running the test cases
After cloning the repository, run `poetry shell`. That will create a
virtual environment. Then run `poetry install --no-root`. That will
install all the libraries needed to run the test cases (and the
package itself). Finally, run `python -m pyteset` to run the test
cases.

### Linting
We use both black and ruff as python linters. To check if the code is
properly formatted, use: `python -m ruff check pyrake test` and
`python -m black --check pyrake/ test/`.

## License

Apache
