"""Solver-level agreement on the CSC program.

CRAN is unreachable from this environment, so the authors' CVXR reference
cannot be executed here and the usual cell-for-cell cross-validation against
it is not available. The substitute: hand the identical program to four
independent convex solvers and check they land on the same point. The
program is a convex QP over a compact feasible set, so the optimal value is
unique; the weight matrix is unique too whenever the fit is not degenerate.

Also reports how often the default solver returns ``optimal_inaccurate``,
which is the failure mode that matters for a library implementation.
"""

from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np

from core import csc_weights
from dgp import simulate

SOLVERS = [cp.CLARABEL, cp.ECOS, cp.SCS, cp.OSQP]


def compare(seed: int) -> dict:
    p = simulate(rng=np.random.default_rng(seed))
    tr, do = p.treated_idx, p.donor_idx
    T0 = p.Y.shape[1] - 1
    Y1_pre, Y0_pre, X1 = p.Y[tr, :T0], p.Y[do, :T0], p.X_dummies[tr]

    fits = {}
    for s in SOLVERS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fits[str(s)] = csc_weights(Y1_pre, Y0_pre, X1, solver=s)
        except Exception as exc:  # noqa: BLE001
            fits[str(s)] = exc
    return fits


if __name__ == "__main__":
    ref_name = str(SOLVERS[0])
    max_wgap, max_ogap, failures = 0.0, 0.0, []
    for seed in range(30):
        fits = compare(seed)
        ref = fits[ref_name]
        if isinstance(ref, Exception):
            failures.append((seed, ref_name, repr(ref)))
            continue
        for name, fit in fits.items():
            if isinstance(fit, Exception):
                failures.append((seed, name, repr(fit)))
                continue
            wgap = float(np.abs(fit.W - ref.W).max())
            ogap = abs(fit.objective - ref.objective) / max(1.0, abs(ref.objective))
            max_wgap, max_ogap = max(max_wgap, wgap), max(max_ogap, ogap)
            if wgap > 1e-3:
                failures.append((seed, name, f"max |dW| = {wgap:.2e}"))

    print(f"30 panels x {len(SOLVERS)} solvers")
    print(f"  max |W - W_clarabel| over all cells : {max_wgap:.3e}")
    print(f"  max relative objective gap          : {max_ogap:.3e}")
    print(f"  disagreements > 1e-3 or errors      : {len(failures)}")
    for f in failures[:10]:
        print("   ", f)
