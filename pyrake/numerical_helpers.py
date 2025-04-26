"""Numerical linear algebra routines."""

from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy import linalg


def solve_diagonal_plus_rank_one(
    eta: npt.NDArray[np.float64],
    zeta: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solve H * x = b.

    Solves a linear system of equations where H is diagonal plus a rank one matrix,
       H = diag(eta) + zeta * zeta^T.
    Because of this structure, we can solve the system in linear time. See Notes for
    more details.

    Parameters
    ----------
     eta : npt.NDArray[np.float64]
        Diagonal component of H.
     zeta : npt.NDArray[np.float64]
        Rank-one component of H.
     b : npt.NDArray[np.float64]
        Right hand side.

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution.

    Notes
    -----
    Let H = diag(eta) + zeta * zeta^T, where eta and zeta are both vectors of length M.
    We can solve H * x = b in O(M) time as follows:
       1. Solve diag(eta) * x' = b. This is just x' = b / eta, elementwise (M
          divisions).
       2. Solve diag(eta) * xi = zeta, or xi = zeta / eta (M divisions).
       3. Calculate x as x' - ((zeta^T * x') / (1 + zeta^T * xi)) * xi. This is 2 dot
          products (each involving M multiplies and M-1 adds) plus M multiplies and M
          subtractions, or O(3M) multiplies plus O(3M) adds, plus a single add and a
          division.
    In total that's 2M divisions, 3M multiplies, and 3M adds.

    """
    assert np.all(eta > 0)
    assert len(eta) == len(zeta) == len(b)

    x_prime = b / eta
    xi = zeta / eta
    x = x_prime - (np.dot(zeta, x_prime) / (1.0 + np.dot(zeta, xi))) * xi
    return x


def solve_kkt_system_hessian_diagonal_plus_rank_one(
    A: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    eta: npt.NDArray[np.float64],
    zeta: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Solve a KKT system of equations.

    Parameters
    ----------
     A : p-by-M matrix.
        Parameter.
     g : vector of length M
        Right-hand-side.
     eta : vector of length M
        Diagonal component of H. See Notes.
     zeta : vector of length M
        Rank-one compoment of H. See Notes.

    Returns
    -------
     delta_w : vector of length M
        Solution to system. See Notes.
     nu : vector of length p
        Solution to system. See Notes.

    Notes
    -----
    Solves:
           _       _   _       _     _   _
          | H   A^T | | delta_w |   |  g  |
          | A    0  | |   nu    | = |  0  |
           -       -   -       -     -   -
    where H = diag(eta) + zeta * zeta^T.

    Since H is diagonal plus rank one, we can exploit the Schur complement and the
    matrix inversion lemma to calculate delta_w in O(p^3 + p^2*M) time, were p is the
    number of rows in A.

    Per the discussion in Boyd and Vandenberghe (2004), Algorithm C.4 (page
    673):
      1. Form B = H^{-1} * A^T and b = H^{-1} * g. This corresponds to p+1 solves. We
         use `solve_diagonal_plus_rank_one` to solve each system in O(M) time, for
         O(p*M) time total.
      2. Form S = -A * B and c = -A * b. Since A is p-by-M and B is M-by-p, forming S
         involves p^2 dot products of length M, which takes (p^2 * M) time. Forming c
         takes O(p * M) time.
      3. Solve S * nu = c via Cholesky decomposition. (S is negative definite, so we
         instead solve -S * nu = -c.) This takes O(p^3) time.
      4. Solve H * delta_w = g - A^T * nu. This takes O(M) time.

    """
    p, M = A.shape
    assert len(g) == M
    assert len(eta) == M
    assert len(zeta) == M

    # Step 1: form B = H^{-1} * A^T and b = H^{-1} * g
    B = np.zeros((M, p))
    for ip in range(p):
        B[:, ip] = solve_diagonal_plus_rank_one(eta, zeta, A.T[:, ip])
    b = solve_diagonal_plus_rank_one(eta, zeta, g)

    # Step 2: form -S = A * B and -c = A * b
    neg_S = np.dot(A, B)
    neg_c = np.dot(A, b)

    # Step 3: Solve -S * xi = -c
    nu = linalg.solve(neg_S, neg_c, assume_a="pos")

    # Step 4: Solve H * delta_w = -grad_ft - A^T * xi
    delta_w = solve_diagonal_plus_rank_one(eta, zeta, g - np.dot(A.T, nu))

    return delta_w, nu
