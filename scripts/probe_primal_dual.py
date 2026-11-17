"""Record every Cvxium solve a test makes, for diffing macOS against Linux.

Diagnostic only, for the macOS/Linux iteration-count split that remains in the
frontier tests after rwilson4/Cvxium#21. Runs one unittest case with
``PrimalDualSolver.solve`` and ``BarrierSolver.solve`` wrapped, and writes one
JSON record per solve, in call order: the problem's shape, the outcome (or the
exception and any result it carries), and for primal-dual solves the
per-iteration surrogate gap, residual norms and step sizes. Aligning two runs by
index shows the first solve where the platforms part ways.

Usage: PYTHONPATH=. python scripts/probe_primal_dual.py <module.Class.method> <out.json>

"""

import json
import math
import sys
import unittest
from typing import Any

import numpy as np
from cvxium import BarrierSolver, PrimalDualSolver
from cvxium.problems.result import BarrierDiagnostics, PrimalDualDiagnostics

records: list[dict[str, Any]] = []


def _describe(problem: Any) -> dict[str, Any]:
    return {
        "M": int(problem.dimension),
        "m": int(problem.num_inequality_constraints()),
        "p": 0 if problem.eq is None else int(problem.eq.A.shape[0]),
        "objective": type(problem.obj).__name__,
        "layers": [type(layer).__name__ for layer in problem.ineq],
    }


def _floats(values: Any) -> list[float | None]:
    return [None if not math.isfinite(v) else float(v) for v in values]


def _outcome(result: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": str(result.status),
        "message": result.message,
        "n_iterations": result.n_iterations,
    }
    for name in ("objective_value", "suboptimality"):
        try:
            value = float(getattr(result, name))
            out[name] = value if math.isfinite(value) else None
        except Exception:  # noqa: BLE001 - absent on feasibility results
            out[name] = None
    diagnostics = result.diagnostics
    if isinstance(diagnostics, BarrierDiagnostics):
        out["inner_nits"] = list(diagnostics.inner_nits)
    if isinstance(diagnostics, PrimalDualDiagnostics):
        out["gaps"] = _floats(diagnostics.surrogate_gaps)
        out["r_pri"] = _floats(diagnostics.primal_residual_norms)
        out["r_dual"] = _floats(diagnostics.dual_residual_norms)
        out["steps"] = _floats(diagnostics.step_sizes)
    return out


def _wrap(cls: type, engine: str) -> None:
    original = cls.solve

    def solve(self, problem, x0=None, *args, **kwargs):  # type: ignore[no-untyped-def]
        rec: dict[str, Any] = {"engine": engine, **_describe(problem)}
        if x0 is not None:
            rec["x0_norm"] = float(np.linalg.norm(x0))
        records.append(rec)
        try:
            result = original(self, problem, x0, *args, **kwargs)
        except Exception as e:
            rec["raised"] = f"{type(e).__name__}: {str(e)[:300]}"
            attached = getattr(e, "result", None)
            if attached is not None:
                rec.update(_outcome(attached))
            raise
        rec.update(_outcome(result))
        return result

    cls.solve = solve  # type: ignore[method-assign]


def main() -> int:
    _wrap(PrimalDualSolver, "pd")
    _wrap(BarrierSolver, "ipm")
    module, cls, method = sys.argv[1].rsplit(".", 2)
    case = getattr(__import__(module, fromlist=[cls]), cls)(method)
    with open("/dev/null", "w") as devnull:
        outcome = unittest.TextTestRunner(stream=devnull).run(case)
    with open(sys.argv[2], "w") as f:
        json.dump({"test": method, "ok": outcome.wasSuccessful(), "solves": records}, f)
    n_pd = sum(r["engine"] == "pd" for r in records)
    iters = sum(r.get("n_iterations") or 0 for r in records)
    print(f"{method}: {len(records)} solves ({n_pd} pd), {iters} iterations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
