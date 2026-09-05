"""The article's Section 4 Monte Carlo, with the baseline it omits.

Section 4.1 draws a single common factor ``F_t = 0.5 t + sin(t/3)`` over
``T0 = 20`` and sets ``y_jt = w_j F_t + e_jt``, with the treated unit's loading
fixed at ``w_1 = 1`` and the donors' loadings drawn as ``lambda_j / sum lambda``
from ``lambda_j ~ U[0.5, 1.5]``, so the donor loadings sum to one. The article
reads that normalisation as a courtesy to conventional SC.

It is the opposite. A convex combination of donors reaches factor loading
``sum_j v_j w_j``, which for ``v`` on the simplex is bounded above by
``max_j w_j``, of order ``1.5 / J``. The treated unit needs loading 1. So the
simplex arm cannot track the factor at all, and its deficit grows with ``J``
because the donor loadings shrink. The reported rise of SC's RMSPE in ``J`` is
that bound, not instability in the optimiser.

``run(...)`` therefore reports the reachable bound alongside SC, MSCb and MSCc,
so the gap can be read off directly. Two further properties of the design:

* Section 4.1 measures RMSPE on the pre-period only, so it scores in-sample fit
  with nothing held out;
* at ``J = 50`` with ``T0 = 20`` the relaxed arms carry more free parameters
  than observations and drive that in-sample error to zero, which makes the
  metric degenerate at the largest ``J`` the article reports.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear, nnls

T0 = 20
J_GRID = (5, 10, 20, 50)
N_REPS = 300


def factor(T0=T0):
    t = np.arange(1, T0 + 1)
    return 0.5 * t + np.sin(t / 3)


def simplex_fit(y, Y0):
    """SC weights: ``min ||y - Y0 w||`` over the simplex."""
    import cvxpy as cp

    w = cp.Variable(Y0.shape[1])
    cp.Problem(
        cp.Minimize(cp.sum_squares(y - Y0 @ w)), [w >= 0, cp.sum(w) == 1]
    ).solve(solver=cp.CLARABEL)
    return None if w.value is None else np.asarray(w.value)


def run(j_grid=J_GRID, n_reps=N_REPS, seed=0, T0=T0):
    """Return one record per ``J``: in-sample RMSPE by arm, plus the bound."""
    rng = np.random.default_rng(seed)
    F = factor(T0)
    out = []
    for J in j_grid:
        lam = rng.uniform(0.5, 1.5, J)
        w_true = lam / lam.sum()
        sc, mscb, mscc = [], [], []
        for _ in range(n_reps):
            eps = rng.standard_normal((T0, J + 1))
            y1 = 1.0 * F + eps[:, 0]
            Y0 = w_true * F[:, None] + eps[:, 1:]

            w_sc = simplex_fit(y1, Y0)
            if w_sc is not None:
                sc.append(np.mean((y1 - Y0 @ w_sc) ** 2))

            wb, _ = nnls(Y0, y1)
            mscb.append(np.mean((y1 - Y0 @ wb) ** 2))

            D = np.hstack([np.ones((T0, 1)), Y0])
            bc = lsq_linear(
                D, y1, bounds=(np.r_[-np.inf, np.zeros(J)], np.full(J + 1, np.inf))
            ).x
            mscc.append(np.mean((y1 - D @ bc) ** 2))

        out.append(
            dict(
                J=J,
                rmspe_sc=float(np.sqrt(np.mean(sc))) if sc else None,
                rmspe_mscb=float(np.sqrt(np.mean(mscb))),
                rmspe_mscc=float(np.sqrt(np.mean(mscc))),
                max_reachable_loading=float(w_true.max()),
                treated_loading=1.0,
                saturated=bool(J + 1 >= T0),
            )
        )
    return out
