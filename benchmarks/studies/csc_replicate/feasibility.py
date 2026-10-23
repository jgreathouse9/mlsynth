"""Does a continuous covariate really make the CSC program infeasible?

Section 3.f of the paper says it does: adding-up requires

    sum_k x_ik (sum_j alpha_jk) = 1 - sum_j omega_j                  (eq. 9)

whose right-hand side does not depend on i, so "if we have continuous x and
i >= 2, then (9) will not hold in general for all i and the optimisation
problem will be infeasible". The reference implementation works around this
by thresholding continuous predictors into 0/1 before fitting
(``merge_all_continous_predictors``).

The claim is checked here, not taken on faith. Write S for the
column sums of alpha and W0 for the sum of omega. Eq. (9) is ``X1 S =
(1 - W0) 1``. Taking ``S = 0`` and ``W0 = 1`` satisfies it for any X1
whatsoever, and the resulting weights ``w_ij = omega_j`` are the pooled
synthetic control -- feasible by construction. So the feasible set is never
empty. What a continuous covariate does is force ``S = 0`` exactly, where
mutually exclusive dummies only force the entries of S to be equal to each
other.

This script runs four covariate designs through the solver and reports
status, objective, and the realised column sums of alpha.
"""

from __future__ import annotations

import numpy as np

from core import csc_weights
from dgp import simulate


def report(name: str, Y1_pre, Y0_pre, X1) -> None:
    try:
        fit = csc_weights(Y1_pre, Y0_pre, X1)
    except RuntimeError as exc:
        print(f"{name:<34} SOLVE FAILED: {exc}")
        return
    colsums = fit.alpha.sum(axis=0)
    spread = float(colsums.max() - colsums.min())
    het = float(np.abs(fit.W - fit.W.mean(axis=0)).mean())
    print(
        f"{name:<34} obj={fit.objective:9.3f}  "
        f"sum_j alpha_jk in [{colsums.min():+.3f}, {colsums.max():+.3f}] "
        f"(spread {spread:.3f})  sum_j omega_j={fit.omega.sum():+.3f}  "
        f"mean |w_ij - wbar_j|={het:.4f}"
    )


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    p = simulate(rng=rng)
    tr, do = p.treated_idx, p.donor_idx
    T0 = p.Y.shape[1] - 1
    Y1_pre, Y0_pre = p.Y[tr, :T0], p.Y[do, :T0]
    n1 = tr.size

    dummies = p.X_dummies[tr]
    continuous = rng.normal(size=(n1, 1))
    binaries = (rng.uniform(size=(n1, 2)) < 0.5).astype(float)

    print(f"n1={n1}, n0={do.size}, T0={T0}\n")
    report("mutually exclusive dummies", Y1_pre, Y0_pre, dummies)
    report("one continuous covariate", Y1_pre, Y0_pre, continuous)
    report("dummies + one continuous", Y1_pre, Y0_pre, np.hstack([dummies, continuous]))
    report("two independent binaries", Y1_pre, Y0_pre, binaries)
    report("continuous, binned into 4", Y1_pre, Y0_pre, _bin := np.eye(4)[
        np.digitize(continuous.ravel(), np.quantile(continuous, [0.25, 0.5, 0.75]))
    ])
