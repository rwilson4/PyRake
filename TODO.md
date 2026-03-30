# TODO

## QuadraticProgramEqualityBoundsSolver

### ~~Exploit low-rank structure of Q in Newton step (O(n³) → O(r²n))~~ DONE

Currently `calculate_newton_step` forms `H_ft = 2t Q + D` explicitly and
Cholesky-factors it at every Newton step, costing O(n³) per step.

When Q is PSD with rank r ≪ n we can do much better. Write
`Q = U_r S_r U_r^T` (economy eigendecomposition) and let
`kappa = sqrt(2t) * U_r * sqrt(S_r)` be the n×r factor satisfying
`2t Q = kappa kappa^T`. Then

    H_ft = D + kappa kappa^T

where D = diag(1/(x-xl)²) is diagonal (trivial to invert). By the
Woodbury identity:

    H_ft^{-1} b = D^{-1} b - D^{-1} kappa (I_r + kappa^T D^{-1} kappa)^{-1} kappa^T D^{-1} b

The r×r Schur complement `I_r + kappa^T D^{-1} kappa` changes every Newton
step (because D changes), but computing it costs O(r² n) and factoring it
costs O(r³). Both are ≪ O(n³) when r ≪ n.

This is a direct application of `solve_rank_p_update` from numerical_helpers.py.
The n×r factor `kappa` can be recomputed cheaply at each Newton step since
`sqrt(2t) * U_r * sqrt(S_r)` involves only a scalar rescaling of a cached matrix.

**Estimated gain**: for a problem with n=500, r=10, the cost drops from
~125M flops (Cholesky of 500×500) to ~2.5M flops (rank-10 update).

When Q is full-rank PD the speedup is zero; the existing Cholesky path
should be kept as the fallback for that case.

**References**: Boyd & Vandenberghe (2004), Appendix C; `solve_rank_p_update`
in `pyrake/optimization/numerical_helpers.py`.
