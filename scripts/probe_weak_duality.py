"""Check Cvxium's best-bounds certificate for weak-duality violations.

Diagnostic only, for the macOS/Linux iteration-count split. Runs one unittest
case with Cvxium's barrier loop instrumented, and for every solve records each
centered step's objective, dual value, barrier parameter, and (for all-linear
Lagrangians) the gradient residual the dual was accepted at. A finite dual
above any primal objective in the same solve violates weak duality, so one of
the two bounds feeding the early-stop certificate is wrong.

Usage: PYTHONPATH=. python scripts/probe_weak_duality.py <module.Class.method>

Set PROBE_ALL=1 to print every solve's steps, not only the violating ones.

"""

import math
import os
import sys
import unittest

import numpy as np
from cvxium.solvers import barrier_solver as bs
from cvxium.solvers import optimization as opt

_orig_dual = bs._BarrierAdapter.evaluate_dual
_orig_update = opt.BaseInteriorPointMethodSolver._update_incumbent_bounds

# Each entry keeps its solver alive, so ids are never reused across solves.
solves: list[tuple[object, list[dict[str, float | bool | None]]]] = []


def _dual(self, lmbda, nu, x_star):  # type: ignore[no-untyped-def]
    g = _orig_dual(self, lmbda, nu, x_star)
    residual = math.nan
    if self._all_linear():
        atnu = self.A.T @ nu if self._equality is not None else None
        r = self._lagrangian_gradient(x_star, lmbda, atnu)
        residual = float(np.max(np.abs(r)))
    self._probe_residual = residual
    return g


def _update(self, result, t, best_primal, best_primal_subopt, best_dual):  # type: ignore[no-untyped-def]
    if best_primal is None:
        solves.append((self, []))
    linear = getattr(self, "_all_linear", lambda: None)()
    solves[-1][1].append(
        {
            "f": float(result.objective_value),
            "g": float(result.dual_value),
            "m_over_t": self.num_ineq_constraints / t,
            "linear": linear,
            "residual": getattr(self, "_probe_residual", math.nan),
        }
    )
    return _orig_update(self, result, t, best_primal, best_primal_subopt, best_dual)


def main() -> int:
    bs._BarrierAdapter.evaluate_dual = _dual
    opt.BaseInteriorPointMethodSolver._update_incumbent_bounds = _update
    module, cls, method = sys.argv[1].rsplit(".", 2)
    case = getattr(__import__(module, fromlist=[cls]), cls)(method)
    with open("/dev/null", "w") as devnull:
        unittest.TextTestRunner(stream=devnull).run(case)

    print(f"## {method}: {len(solves)} barrier solves")
    violations = 0
    for k, (_, steps) in enumerate(solves):
        pmin = min(s["f"] for s in steps)
        bad = [
            j
            for j, s in enumerate(steps)
            if math.isfinite(s["g"]) and s["g"] > pmin + 1e-12
        ]
        if not bad and os.environ.get("PROBE_ALL") != "1":
            continue
        violations += len(bad)
        print(f"\nsolve {k}: weak duality violated at step(s) {[j + 1 for j in bad]}")
        print("step  objective            dual                 m/t        linear  max|grad L|")
        for j, s in enumerate(steps):
            mark = " <-" if j in bad else ""
            print(
                f"{j + 1:>4}  {s['f']:<19.13g}  {s['g']:<19.13g}  "
                f"{s['m_over_t']:<9.3g}  {s['linear']!s:<6}  {s['residual']:.3g}{mark}"
            )
    print(f"\n{violations} violating steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
