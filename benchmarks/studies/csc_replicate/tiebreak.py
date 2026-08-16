"""Does a ridge tie-break restore a single answer, and at what cost in fit?

The CSC program as written has a face of minimisers, so the ATT a user reads
off depends on which optimum the solver returns. The standard remedy in this
library's vocabulary is the one RESCM's relaxation branch uses: keep the fit
constraint, then pick among the near-optimal weights by a strictly convex
criterion. Here the criterion is
``ridge * (||omega||^2 + ||alpha||_F^2)``.

Reported per ridge value: agreement of the weight matrix across four
solvers, the pre-treatment fit that was given up, and the ATT.
"""

from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np

from core import csc_counterfactual, csc_weights
from dgp import simulate

SOLVERS = [cp.CLARABEL, cp.ECOS, cp.SCS]
RIDGES = (0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0)


if __name__ == "__main__":
    rows = {r: {"dW": [], "att": [], "fit": [], "spread": []} for r in RIDGES}

    for seed in range(12):
        p = simulate(rng=np.random.default_rng(seed))
        tr, do = p.treated_idx, p.donor_idx
        if tr.size < 2:
            continue
        T0 = p.Y.shape[1] - 1
        Y1_pre, Y0_pre, X1 = p.Y[tr, :T0], p.Y[do, :T0], p.X_dummies[tr]
        Y1_post, Y0_post = p.Y[tr, T0:], p.Y[do, T0:]

        for r in RIDGES:
            fits, atts = [], []
            for s in SOLVERS:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        f = csc_weights(Y1_pre, Y0_pre, X1, ridge=r, solver=s)
                except Exception:  # noqa: BLE001
                    continue
                fits.append(f)
                atts.append(float(np.mean(Y1_post - csc_counterfactual(f, Y0_post))))
            if len(fits) < 2:
                continue
            rows[r]["dW"].append(max(np.abs(f.W - fits[0].W).max() for f in fits))
            rows[r]["spread"].append(max(atts) - min(atts))
            rows[r]["att"].append(atts[0])
            rows[r]["fit"].append(fits[0].objective)

    print(f"{'ridge':>8} {'max|dW| across solvers':>24} {'ATT spread':>12} {'ATT':>8} {'pre-fit SSR':>13}")
    for r in RIDGES:
        d = rows[r]
        if not d["dW"]:
            continue
        print(
            f"{r:>8.0e} {np.median(d['dW']):>24.2e} {np.median(d['spread']):>12.3f}"
            f" {np.median(d['att']):>8.2f} {np.median(d['fit']):>13.2f}"
        )
    print("\n(medians over 12 panels; true tau = 1.0)")
