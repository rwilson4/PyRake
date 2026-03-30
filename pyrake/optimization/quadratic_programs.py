"""Quadratic program solvers."""

import numpy as np
import numpy.typing as npt
from scipy import linalg

from .linear_programs import EqualityWithBoundsSolver
from .numerical_helpers import (
    solve_diagonal_eta_inverse,
    solve_kkt_system,
    solve_rank_p_update,
)
from .optimization import (
    EqualityConstrainedInteriorPointMethodSolver,
    InteriorPointMethodResult,
    InteriorPointMethodSolver,
    OptimizationSettings,
    UnconstrainedNewtonSolver,
)


class QuadraticNewtonSolver(UnconstrainedNewtonSolver):
    r"""Solve min 0.5 * x^T Q x + c^T x, where Q is PSD.

    Optimal solution: x* = -Q^{-1} c.
    """

    def __init__(
        self,
        Q: npt.NDArray[np.float64],
        c: npt.NDArray[np.float64],
        settings: OptimizationSettings | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self.Q = Q
        self.c = c
        self._Q_factor = linalg.cho_factor(Q)

    def newton_step(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve Q * delta_x = -grad_f0 = -(Q*x + c)."""
        return linalg.cho_solve(self._Q_factor, -(self.Q @ x + self.c))

    def evaluate_objective(self, x: npt.NDArray[np.float64]) -> float:
        return float(0.5 * x @ self.Q @ x + self.c @ x)

    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.Q @ x + self.c

    def hessian_vector_product(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return self.Q @ y


class QuadraticProgramEqualityBoundsSolver(
    EqualityConstrainedInteriorPointMethodSolver, InteriorPointMethodSolver
):
    r"""Solve a quadratic program with equality and bound constraints.

    Solves:
      minimize   x^T Q x + c^T x
      subject to A x = b
                 x >= xl

    where Q is positive semi-definite (PSD).

    Parameters
    ----------
     Q : (n, n) array
        Positive semi-definite quadratic cost matrix.
     c : (n,) array
        Linear cost vector.
     A : (p, n) array
        Equality constraint matrix.
     b : (p,) array
        Equality constraint right-hand side.
     xl : float or (n,) array, optional
        Lower bounds on x. Defaults to 0.
     settings : OptimizationSettings, optional
        Optimization settings.

    Notes
    -----

    **Lagrangian**

    The Lagrangian (with lmbda >= 0 for bound constraints and nu free for
    equality constraints) is:

       L(x, lmbda, nu) = x^T Q x + c^T x + lmbda^T (xl - x) + nu^T (A x - b)

    **Lagrangian dual function**

    Taking the gradient wrt x and setting to zero:

       2 Q x + c - lmbda + A^T nu = 0
       => x* = -(1/2) Q^{-1} (c - lmbda + A^T nu)

    Let v = c - lmbda + A^T nu. Since Q is PSD, the infimum over x is
    finite only when v lies in the range of Q; otherwise g = -inf.
    When v is in range(Q) the minimizer is unique along the row space of Q
    and:

       g(lmbda, nu) = -(1/4) v^T Q^+ v + lmbda^T xl - nu^T b

    where Q^+ is the Moore-Penrose pseudoinverse of Q. The dual is also
    -inf when any lmbda_i < 0.

    **Barrier problem**

    The inequality constraints x_i >= xl_i are absorbed via a log barrier:

       ft(x) = t * (x^T Q x + c^T x) - sum_i log(x_i - xl_i)
       subject to A x = b

    Gradient of ft:
       grad_ft_j = t * (2(Qx)_j + c_j) - 1 / (x_j - xl_j)

    Hessian of ft:
       H_ft = 2t Q + D,   D = diag(1 / (x - xl)^2)

    The Hessian is strictly positive definite whenever x is strictly feasible
    (x > xl), regardless of whether Q is PD or only PSD, because D alone is
    already strictly positive definite.

    **Newton step**

    The equality-constrained Newton step solves the KKT system:

       | H_ft   A^T | | delta_x |   | -grad_ft |
       |  A      0  | |   nu    | = |    0     |

    Writing Q = U_r S_r U_r^T (precomputed rank-r SVD) and defining the
    cached factor kappa_cache = U_r * sqrt(S_r) (n-by-r, computed once),
    we have Q = kappa_cache @ kappa_cache^T.  Setting kappa = sqrt(2t) *
    kappa_cache (recomputed cheaply at each step, O(rn)) gives

       H_ft = D + kappa @ kappa^T.

    The Hessian system (D + kappa kappa^T) y = z is then solved via the
    Woodbury identity using `solve_rank_p_update`:

       H_ft^{-1} z = D^{-1} z
                     - D^{-1} kappa (I_r + kappa^T D^{-1} kappa)^{-1} kappa^T D^{-1} z

    Cost per Newton step: O(r^2 n + r^3).  For full-rank PD Q (r = n) this
    is the same O(n^3) as a direct Cholesky of H_ft.  For low-rank PSD Q
    (r << n) it is substantially cheaper.

    **Phase I**

    `EqualityWithBoundsSolver` finds the initial strictly feasible point
    satisfying A x = b and x > xl.

    """

    def __init__(
        self,
        Q: npt.NDArray[np.float64],
        c: npt.NDArray[np.float64],
        A: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        xl: float | list[float] | npt.NDArray[np.float64] = 0.0,
        settings: OptimizationSettings | None = None,
    ) -> None:
        phase1_solver = EqualityWithBoundsSolver(A=A, b=b, lb=xl, settings=settings)
        super().__init__(phase1_solver=phase1_solver, settings=settings)
        self.Q = Q
        self.c = c
        self._A = A
        self._b = b
        self.xl = xl
        self._n = Q.shape[0]

        # Precompute the economy SVD of Q (Q is PSD so U == V).
        # We retain only the rank-r subspace, which also handles PSD Q where
        # some singular values are numerically zero.
        U, s, _ = linalg.svd(Q, full_matrices=False)
        rank_tol = max(Q.shape) * np.finfo(float).eps * (s[0] if len(s) else 0.0)
        rank = int(np.sum(s > rank_tol))
        self._Q_svd_U_r: npt.NDArray[np.float64] = U[:, :rank]
        self._Q_svd_s_r: npt.NDArray[np.float64] = s[:rank]

        # Cached square-root factor: kappa_cache @ kappa_cache^T == Q.
        # Shape: n-by-r.  Used in hessian_multiply and calculate_newton_step
        # so that we never re-process Q at runtime; only a cheap O(r*n) scalar
        # multiply (to absorb sqrt(2t)) is needed per Newton step.
        self._kappa_cache: npt.NDArray[np.float64] = U[:, :rank] * np.sqrt(s[:rank])

    # ------------------------------------------------------------------
    # Properties required by EqualityConstrainedInteriorPointMethodSolver
    # ------------------------------------------------------------------

    @property
    def A(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Equality constraint matrix."""
        return self._A

    @property
    def b(self) -> npt.NDArray[np.float64]:
        """Equality constraint right-hand side."""
        return self._b

    # ------------------------------------------------------------------
    # Properties required by BaseInteriorPointMethodSolver
    # ------------------------------------------------------------------

    @property
    def num_eq_constraints(self) -> int:
        """Count equality constraints."""
        return self._A.shape[0]

    @property
    def num_ineq_constraints(self) -> int:
        """Count inequality (bound) constraints."""
        return self._n

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def evaluate_objective(self, x: npt.NDArray[np.float64]) -> float:
        """Evaluate f0(x) = x^T Q x + c^T x."""
        return float(x @ self.Q @ x + self.c @ x)

    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Gradient of f0: grad_f0 = 2 Q x + c."""
        return 2.0 * (self.Q @ x) + self.c

    # ------------------------------------------------------------------
    # Constraints: fi(x) = xl_i - x_i <= 0
    # ------------------------------------------------------------------

    def constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate fi(x) = xl_i - x_i (each <= 0 when x >= xl)."""
        return self.xl - x  # broadcasts for scalar or array xl

    def grad_constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Gradient matrix of constraints: row i is grad fi = -e_i, so G = -I."""
        return -np.eye(self._n)

    def grad_constraints_multiply(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Compute G @ y in O(n) time, where G is the constraint-gradient matrix.

        The n inequality constraints are fi(x) = xl_i - x_i, so

            grad fi(x) = -e_i   (i-th standard basis vector),

        giving the n-by-n gradient matrix G = -I. Thus G @ y = -y.
        This avoids forming the n-by-n identity and doing a full matrix-vector
        multiply (which would be O(n²)).

        """
        return -y

    def grad_constraints_transpose_multiply(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Compute G^T @ y in O(n) time.

        Since G = -I we have G^T = -I, so G^T @ y = -y.

        """
        return -y

    # ------------------------------------------------------------------
    # Hessian
    # ------------------------------------------------------------------

    def hessian_multiply(
        self, x: npt.NDArray[np.float64], t: float, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Multiply H_ft * y.

        H_ft = 2t Q + D = D + kappa @ kappa^T,
        where kappa = sqrt(2t) * kappa_cache and D = diag(1 / (x - xl)^2).

        Using Q = kappa_cache @ kappa_cache^T:

            H_ft y = 2t * kappa_cache @ (kappa_cache^T @ y) + d * y,

        which costs O(rn) instead of the O(n^2) dense matrix-vector product
        2t Q y for low-rank Q (r << n).
        """
        d = 1.0 / np.square(x - self.xl)
        return 2.0 * t * (self._kappa_cache @ (self._kappa_cache.T @ y)) + d * y

    # ------------------------------------------------------------------
    # Newton step
    # ------------------------------------------------------------------

    def calculate_newton_step(
        self,
        x: npt.NDArray[np.float64],
        t: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Calculate Newton step for the barrier sub-problem.

        Solves the KKT system:

           | H_ft   A^T | | delta_x |   | -grad_ft |
           |  A      0  | |   nu    | = |    0     |

        where H_ft = 2t Q + D, D = diag(1 / (x - xl)^2).

        We form H_ft explicitly and use Cholesky to solve H_ft * y = z.
        This is re-factored at every Newton step, which is correct but
        O(n^3) per step.

        """
        d = 1.0 / np.square(x - self.xl)
        r = self._kappa_cache.shape[1]

        if r < self._n:
            # Low-rank Q (r << n): use Woodbury identity.  O(r^2 n + r^3) per step.
            eta_inverse = 1.0 / d  # (x_j - xl_j)^2; diagonal D^{-1}
            kappa = np.sqrt(2.0 * t) * self._kappa_cache  # n-by-r, O(rn) per step

            def hessian_solve(rhs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                return solve_rank_p_update(
                    rhs, kappa, solve_diagonal_eta_inverse, eta_inverse=eta_inverse
                )

        else:
            # Full-rank Q: form H explicitly and Cholesky-factor.  O(n^3) per step.
            H = 2.0 * t * self.Q + np.diag(d)
            H_factor = linalg.cho_factor(H)

            def hessian_solve(rhs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                return linalg.cho_solve(H_factor, rhs)

        return solve_kkt_system(
            A=self.A,
            g=-self.gradient_barrier(x, t),
            hessian_solve=hessian_solve,
        )

    # ------------------------------------------------------------------
    # Backtracking line search feasibility
    # ------------------------------------------------------------------

    def btls_keep_feasible(
        self, x: npt.NDArray[np.float64], delta_x: npt.NDArray[np.float64]
    ) -> float:
        r"""Return the largest step s keeping x + s * delta_x strictly feasible.

        Strict feasibility requires fi(x + s * delta_x) < 0 for all i, i.e.

            xl_j - (x_j + s * delta_x_j) < 0
            ⟺  s * delta_x_j > xl_j - x_j    for all j.

        Three cases:

        * delta_x_j > 0: the left side grows with s, and since x_j > xl_j
          (strict feasibility of the current iterate) the inequality already
          holds at s = 0 and remains satisfied for all s > 0. No constraint.

        * delta_x_j = 0: reduces to 0 > xl_j - x_j, which holds by strict
          feasibility. No constraint.

        * delta_x_j < 0: dividing by delta_x_j flips the inequality:

              s < (xl_j - x_j) / delta_x_j
                = (x_j - xl_j) / (-delta_x_j).

          Both numerator and denominator are positive, giving a positive
          upper bound on s.

        The tightest bound is therefore

            s_max = min_{j : delta_x_j < 0}  (x_j - xl_j) / (-delta_x_j).

        This is computed in O(n) time with a single masked minimum.

        """
        mask = delta_x < 0
        if not np.any(mask):
            return 1.0

        gaps = x - self.xl  # (x_j - xl_j) > 0 for strictly feasible x
        ratios = gaps[mask] / (-delta_x[mask])
        return float(np.min(ratios))

    # ------------------------------------------------------------------
    # Dual function
    # ------------------------------------------------------------------

    def evaluate_dual(
        self,
        lmbda: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
        x_star: npt.NDArray[np.float64],
    ) -> float:
        r"""Evaluate the Lagrangian dual function.

        The dual function (derived by minimizing the Lagrangian over x) is:

            g(lmbda, nu) = -(1/4) v^T Q^+ v + lmbda^T xl - nu^T b

        where v = c - lmbda + A^T nu and Q^+ is the Moore-Penrose pseudoinverse
        of Q. The dual is -infinity when any lmbda_i < 0, or when v does not
        lie in the range of Q (so the infimum over x is -infinity).

        """
        if np.any(lmbda < 0):
            return -np.inf

        v = self.c - lmbda + self.A.T @ nu

        # Project v onto the range of Q.  If v has a non-trivial component in
        # the null space of Q, the Lagrangian is unbounded below and g = -inf.
        v_proj = self._Q_svd_U_r @ (self._Q_svd_U_r.T @ v)
        if not np.allclose(v_proj, v, atol=1e-6):
            return -np.inf

        # Q^+ v = U_r diag(1/s_r) U_r^T v  (Q is symmetric so U == V)
        Q_pinv_v = self._Q_svd_U_r @ ((self._Q_svd_U_r.T @ v) / self._Q_svd_s_r)

        return float(
            -0.25 * np.dot(v, Q_pinv_v) + np.sum(lmbda * self.xl) - np.dot(self.b, nu)
        )

    # ------------------------------------------------------------------
    # solve: force fully_optimize=True (inherited from InteriorPointMethodSolver)
    # ------------------------------------------------------------------

    def solve(
        self,
        x0: npt.NDArray[np.float64] | None = None,
        fully_optimize: bool = False,
        **kwargs: object,
    ) -> InteriorPointMethodResult:
        """Solve the QP to optimality."""
        result = super().solve(x0=x0, fully_optimize=True, **kwargs)
        assert isinstance(result, InteriorPointMethodResult)
        return result
