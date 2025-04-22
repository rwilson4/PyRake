
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from .exceptions import ProblemInfeasibleError

def solve_phase1(X, mu, phi):
    M, p = X.shape

    def objective(w):
        return np.dot(w, w)

    def grad(w):
        return 2 * w

    A_eq = (X.T) / M
    b_eq = mu
    eq_constraint = LinearConstraint(A_eq, b_eq, b_eq)
    bounds = Bounds(0, np.inf)

    w0 = np.ones(M)

    res = minimize(
        objective,
        w0,
        method="trust-constr",
        jac=grad,
        constraints=[eq_constraint],
        bounds=bounds,
        options={"verbose": 0}
    )

    if not res.success:
        raise ProblemInfeasibleError("Optimization did not converge.")

    w = res.x
    norm_sq = np.sum(w**2)

    if norm_sq > phi:
        raise ProblemInfeasibleError("Minimum feasible norm exceeds variance budget (phi).")

    return w
