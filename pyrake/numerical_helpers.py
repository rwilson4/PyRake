"""Numerical linear algebra routines."""

from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt
from scipy import linalg

from .exceptions import NewtonStepError


def solve_diagonal(
    b: npt.NDArray[np.float64],
    eta: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solve H * x = b.

    Solves a linear system of equations where H is diagonal,
       H = diag(eta).
    Because of this structure, we can solve the system in linear time.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right hand side. Can be either a vector or a matrix, in which case we solve the
        system for each column of b.
     eta : npt.NDArray[np.float64]
        Diagonal elements of the upper left block of H.

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution.

    """
    if not np.all(eta > 0):
        raise NewtonStepError("Hessian is not strictly positive definite.")

    if b.ndim == 1:
        if b.shape != eta.shape:
            raise ValueError("b and eta must have the same length.")
        return b / eta
    elif b.ndim == 2:
        if b.shape[0] != eta.shape[0]:
            raise ValueError("Number of rows in beta must match length of eta.")
        return b / eta[:, np.newaxis]
    else:
        raise ValueError("b must be either a 1D or 2D NumPy array.")


def solve_diagonal_eta_inverse(
    b: npt.NDArray[np.float64],
    eta_inverse: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solve H * x = b.

    Solves a linear system of equations where H is diagonal,
       H = diag(eta).
    Because of this structure, we can solve the system in linear time.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right hand side. Can be either a vector or a matrix, in which case we solve the
        system for each column of b.
     eta_inverse : npt.NDArray[np.float64]
        One divided by diagonal elements of the upper left block of H.

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution.

    """
    if not np.all(eta_inverse > 0):
        raise NewtonStepError("Hessian is not strictly positive definite.")

    if b.ndim == 1:
        if b.shape != eta_inverse.shape:
            raise ValueError("b and eta_inverse must have the same length.")
        return b * eta_inverse
    elif b.ndim == 2:
        if b.shape[0] != eta_inverse.shape[0]:
            raise ValueError("Number of rows in beta must match length of eta_inverse.")
        return b * eta_inverse[:, np.newaxis]
    else:
        raise ValueError("b must be either a 1D or 2D NumPy array.")


def solve_diagonal_plus_rank_one(
    b: npt.NDArray[np.float64],
    eta: npt.NDArray[np.float64],
    zeta: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solve H * x = b.

    Solves a linear system of equations where H is diagonal plus a rank one matrix,
       H = diag(eta) + zeta * zeta^T.
    Because of this structure, we can solve the system in linear time. See Notes for
    more details.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right hand side. Can be either a vector or a matrix, in which case we solve the
        system for each column of b.
     eta : npt.NDArray[np.float64]
        Diagonal component of H.
     zeta : npt.NDArray[np.float64]
        Rank-one component of H.

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
    if not np.all(eta > 0):
        raise NewtonStepError("Hessian is not strictly positive definite.")

    if eta.shape != zeta.shape:
        raise ValueError("Dimension mismatch: eta and zeta had different dimensions.")

    xi = zeta / eta
    den = 1.0 / (1.0 + np.dot(zeta, xi))

    if b.ndim == 1:
        if b.shape != eta.shape:
            raise ValueError("b and eta must have the same length.")
        x_prime = b / eta
        return x_prime - den * (np.dot(zeta, x_prime) * xi)
    elif b.ndim == 2:
        if b.shape[0] != eta.shape[0]:
            raise ValueError("Number of rows in beta must match length of eta.")
        x_prime = b / eta[:, np.newaxis]
        return x_prime - den * np.outer(xi, zeta.T @ x_prime)
    else:
        raise ValueError("b must be either a 1D or 2D NumPy array.")


def solve_diagonal_plus_rank_one_eta_inverse(
    b: npt.NDArray[np.float64],
    eta_inverse: npt.NDArray[np.float64],
    zeta: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solve H * x = b.

    Solves a linear system of equations where H is diagonal plus a rank one matrix,
       H = diag(eta) + zeta * zeta^T.
    Because of this structure, we can solve the system in linear time. See Notes for
    more details.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right hand side. Can be either a vector or a matrix, in which case we solve the
        system for each column of b.
     eta_inverse : npt.NDArray[np.float64]
        One over the diagonal component of H.
     zeta : npt.NDArray[np.float64]
        Rank-one component of H.

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
    if not np.all(eta_inverse > 0):
        raise NewtonStepError("Hessian is not strictly positive definite.")

    if eta_inverse.shape != zeta.shape:
        raise ValueError("Dimension mismatch: eta and zeta had different dimensions.")

    xi = zeta * eta_inverse
    den = 1.0 / (1.0 + np.dot(zeta, xi))

    if b.ndim == 1:
        if b.shape != eta_inverse.shape:
            raise ValueError("b and eta_inverse must have the same length.")
        x_prime = b * eta_inverse
        return x_prime - den * (np.dot(zeta, x_prime) * xi)
    elif b.ndim == 2:
        if b.shape[0] != eta_inverse.shape[0]:
            raise ValueError("Number of rows in beta must match length of eta_inverse.")
        x_prime = b * eta_inverse[:, np.newaxis]
        return x_prime - den * np.outer(xi, zeta.T @ x_prime)
    else:
        raise ValueError("b must be either a 1D or 2D NumPy array.")


def solve_arrow_sparsity_pattern(
    b: npt.NDArray[np.float64],
    eta: npt.NDArray[np.float64],
    zeta: npt.NDArray[np.float64],
    theta: float,
) -> npt.NDArray[np.float64]:
    """Solve H * x = b.

    Solves a linear system of equations where H has an arrow sparsity pattern:
         _                 _
        |  diag(eta)  zeta  |
    H = |                   |
        |_  zeta^T   theta _|

    Because of this structure, we can solve the system in linear time. See Notes for
    more details.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right hand side. Can be either a vector or a matrix, in which case we solve the
        system for each column of b.
     eta : npt.NDArray[np.float64]
        Diagonal elements of the upper left block of H.
     zeta : npt.NDArray[np.float64]
        Last row/column of H, other than the bottom right element.
     theta : float
        The bottom right element of H.

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution.

    Notes
    -----
    In this case, we can calculate the Cholesky factorization of H analytically:
    H = L * L^T, where
         _                                                                          _
        |        diag(eta^{1/2})                         0                           |
    L = |                                                                            |
        |_  zeta^T * diag(eta^{-1/2})  sqrt(theta - zeta^T * diag(eta^{-1}) * zeta) _|.

    Call that bottom right element psi. Let y = L^T * x. THen H * x = b is equivalent to
    L * y = b. Let eta and zeta be of length M (and x, b, and y are therefore of length
    M + 1). Then the first M entries of y are simply b[0:M] / np.sqrt(eta). The last
    element of y satisfies:
        b[M] = zeta^T * diag(eta^{-1/2}) * y[0:M] + psi * y[M]
             = zeta^T * diag(eta^{-1}) * b[0:M] + psi * y[M], or
        y[M] = (1 / psi) * (b[M] - zeta^T * diag(eta^{-1}) * b[0:M]).

    Next we solve L^T * x = y. Starting from the last element, we have:
       psi * x[M] = y[M], or x[M] = y[M] / psi.
    The remaining equations are of the form:
       sqrt(eta[i]) * x[i] + (zeta[i] / sqrt(eta[i])) * x[M] = y[i],
    or x[i] = y[i] / sqrt(eta[i]) - (zeta[i] / eta[i]) * x[M].

    Since
        y[i] / sqrt(eta[i]) = b[i] / eta[i], and
                 y[M] / psi = (b[M] - zeta^T * diag(eta^{-1}) * b[0:M]) / psi_squared,
    we have:
          x[M] = (b[M] - zeta^T * diag(eta^{-1}) * b[0:M]) / psi_squared
               = (b[M] - (diag(eta^{-1}) * zeta)^T * b[0:M]) / psi_squared, and
        x[0:M] = b[0:M] / eta - x[M] * (diag(eta^{-1}) * zeta).

    It takes M divides to calculate diag(eta^{-1}) * zeta, then M multiplies plus M adds
    to calculate psi_squared. Then M multiplies, M adds, and one division to calculate
    x[M]. Then M divides, M multiplies, and M adds to calculate x[0:M]. In total, that's
    2*M + 1 divides, 3*M multiplies, and 3*M adds, or 8*M + 1 flops.

    """
    if not np.all(eta > 0):
        raise NewtonStepError("Hessian is not strictly positive definite.")

    if eta.shape != zeta.shape:
        raise ValueError("Dimension mismatch: eta and zeta had different dimensions.")

    M = eta.shape[0]

    # Calculate diag(eta)^{-1} * zeta and psi^2
    diag_eta_inverse_dot_zeta = zeta / eta
    psi_squared = theta - np.dot(diag_eta_inverse_dot_zeta, zeta)
    if psi_squared <= 0:
        raise NewtonStepError("Hessian is not strictly positive definite.")

    # Calculate x
    x = np.zeros_like(b)
    if b.ndim == 1:
        if b.shape[0] != M + 1:
            raise ValueError(
                "Dimension mismatch: b must have M + 1 entries, where M = len(eta)."
            )
        x[M] = (b[M] - np.dot(diag_eta_inverse_dot_zeta, b[0:M])) / psi_squared
        x[0:M] = b[0:M] / eta - x[M] * diag_eta_inverse_dot_zeta
    elif b.ndim == 2:
        if b.shape[0] != M + 1:
            raise ValueError(
                "Dimension mismatch: b must have M + 1 rows, where M = len(eta)."
            )
        x[M, :] = (b[M, :] - (diag_eta_inverse_dot_zeta.T @ b[0:M, :])) / psi_squared
        x[0:M, :] = b[0:M, :] / eta[:, np.newaxis] - np.outer(
            diag_eta_inverse_dot_zeta, x[M, :]
        )
    else:
        raise ValueError("Dimension mismatch: b must be either a 1D or 2D NumPy array.")

    return x


def solve_arrow_sparsity_pattern_phase1(
    b: npt.NDArray[np.float64],
    eta_inverse: npt.NDArray[np.float64],
    one_over_psi_squared: float,
) -> npt.NDArray[np.float64]:
    """Solve H * x = b.

    Solves a linear system of equations where H has an arrow sparsity pattern:
         _                 _
        |  diag(eta)   eta  |
    H = |                   |
        |_   eta^T   theta _|

    Because of this structure, we can solve the system in linear time. See Notes for
    more details.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right hand side. Can be either a vector or a matrix, in which case we solve the
        system for each column of b.
     eta_inverse : npt.NDArray[np.float64]
        One divided by the diagonal elements of the upper left block of H.
     one_over_psi_squared : float
        1.0 / (theta - np.sum(eta))

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution.

    Notes
    -----
    Like `solve_arrow_sparsity_pattern`, but for the specific instance used to solve:
      minimize    s
      subject to  A * x = b
                  -x <= s.

    In this case, eta[i]^{-1} = (x_i + s)^2, diag_eta_inverse_dot_zeta[i] = 1.0, and
    1 / psi_squared = (s0 + eps - s)^2 / M. Thus, we can solve the system both faster
    and with more numerical stability.

    `solve_arrow_sparsity_pattern` uses 2*M + 1 divides, 3*M multiplies, and 3*M adds,
    or 8*M + 1 flops. `solve_arrow_sparsity_pattern_phase1` uses 0 divides, M + 1
    multiplies, and 2 * M + 1 adds, or 3*M + 2 flops.

    """
    if not np.all(eta_inverse > 0) or one_over_psi_squared <= 0:
        raise NewtonStepError("Hessian is not strictly positive definite.")

    M = eta_inverse.shape[0]
    # Calculate x
    x = np.zeros_like(b)
    if b.ndim == 1:
        if b.shape[0] != M + 1:
            raise ValueError(
                "Dimension mismatch: b must have M + 1 entries, where M = len(eta)."
            )
        x[M] = (b[M] - np.sum(b[0:M])) * one_over_psi_squared
        x[0:M] = b[0:M] * eta_inverse - x[M]
    elif b.ndim == 2:
        if b.shape[0] != M + 1:
            raise ValueError(
                "Dimension mismatch: b must have M + 1 rows, where M = len(eta)."
            )
        x[M, :] = (b[M, :] - np.sum(b[0:M, :], axis=0)) * one_over_psi_squared
        x[0:M, :] = b[0:M, :] * eta_inverse[:, np.newaxis] - x[M, :]
    else:
        raise ValueError("Dimension mismatch: b must be either a 1D or 2D NumPy array.")

    return x


def solve_kkt_system(
    A: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    hessian_solve: Callable[..., npt.NDArray[np.float64]],
    **kwargs,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Solve a KKT system of equations.

    Parameters
    ----------
     A : p-by-M matrix.
        Parameter.
     g : vector of length M
        Right-hand-side.
     hessian_solve : Callable
        A function that solves H * x = y. The first argument to hessian_solve will be y.
        Additional arguments will be passed via **kwargs.
     kwargs
        Extra arguments to pass to hessian_solve.

    Returns
    -------
     delta_x : vector of length M
        Solution to system. See Notes.
     nu : vector of length p
        Solution to system. See Notes.

    Notes
    -----
    Solves:
           _       _   _       _     _   _
          | H   A^T | | delta_x |   |  g  |
          | A    0  | |   nu    | = |  0  |
           -       -   -       -     -   -
    where H is the Hessian.

    When we can solve systems H * x = y in O(M) time, we can exploit the Schur
    complement and the matrix inversion lemma to calculate delta_x in O(p^3 + p^2*M)
    time, were p is the number of rows in A.

    Per the discussion in Boyd and Vandenberghe (2004), Algorithm C.4 (page
    673):
      1. Form B = H^{-1} * A^T and b = H^{-1} * g. This corresponds to p+1 solves. We
         use `hessian_solve` to solve each system in O(M) time, for O(p*M) time total.
      2. Form S = -A * B and c = -A * b. Since A is p-by-M and B is M-by-p, forming S
         involves p^2 dot products of length M, which takes (p^2 * M) time. Forming c
         takes O(p * M) time.
      3. Solve S * nu = c via Cholesky decomposition. (S is negative definite, so we
         instead solve -S * nu = -c.) This takes O(p^3) time.
         a. If A is not full rank, S won't be, either. We can typically still solve the
            system using the Singular Value Decomposition (SVD) instead of the Cholesky
            decomposition. The SVD is slower, so we still at least *try* the Cholesky,
            and if that fails, we fall back to SVD.
      4. Solve H * delta_x= g - A^T * nu. This takes O(M) time.

    """
    p, M = A.shape
    if len(g) != M:
        raise ValueError(
            "Dimension mismatch: g should have one entry for each column of A."
        )

    # Step 1: form B = H^{-1} * A^T and b = H^{-1} * g
    B = hessian_solve(A.T, **kwargs)
    b = hessian_solve(g, **kwargs)

    # Step 2: form -S = A * B and -c = A * b
    neg_S = A @ B
    neg_c = A @ b

    # Step 3: Solve -S * xi = -c
    try:
        c, lower = linalg.cho_factor(neg_S, lower=True)
        nu = linalg.cho_solve((c, lower), neg_c)
    except np.linalg.LinAlgError:
        # This can happen when A is not full rank; fall back to SVD, which is slower but
        # more numerically stable. To be honest though, in my timing experiments, this
        # is really about the same speed as Cholesky, so consider just always doing SVD.
        U, s, Vh = linalg.svd(neg_S, full_matrices=False)
        rank = int(np.sum(s > 1e-10))
        U_r = U[:, 0:rank]
        if not np.allclose(U_r @ (U_r.T @ neg_c), neg_c):
            raise NewtonStepError(
                "KKT system did not have a solution, because A is not full rank."
            )

        s_inv = np.zeros_like(s)
        s_inv[0:rank] = 1.0 / s[0:rank]
        nu = Vh.T @ (s_inv * (U.T @ neg_c))

    # Step 4: Solve H * delta_x = -grad_ft - A^T * xi
    delta_x = hessian_solve(g - (A.T @ nu), **kwargs)

    return delta_x, nu
