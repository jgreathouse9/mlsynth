#!/usr/bin/env python3
"""Reference L2-relaxation solve for the ``pda_brexit`` cross-validation.

Faithful port of the authors' ``Fun/L2relax.R`` (Shi & Wang, "L2-Relaxation for
Economic Prediction"; https://github.com/ishwang1/L2relax-PDA), vendored
alongside this file under ``Fun/L2relax.R`` for provenance. This is the
*multiple-treated-units* application (Section 6.2, Brexit): each treated UK firm
gets its own L2-relaxation counterfactual against the shared control pool. The
per-firm program is exactly ``L2relax.R``:

    standardize y1 (treated firm pre, length T1) and X1 (shared control pre,
    T1 x N) by their mean/sd (intercept=TRUE, standardize=TRUE);
    Sigma = X1_tilde' X1_tilde / T1   (SHARED across firms);
    eta_j = X1_tilde' y1_tilde / T1   (per firm);
    minimize ||beta||_2^2  s.t.  |eta_j - Sigma beta| <= tau  (elementwise);
    de-standardize: beta_hat = sd1 * beta_tilde / Sd1 .

Solver note
-----------
The authors' default solver is CVXR (MOSEK in their notebooks, ECOS_BB in the
``L2relax.R`` signature). CVXR is NOT installable in this environment's R, so the
identical convex program is run via cvxpy with the authors' default open solver
family, ECOS. The L2-relaxation primal is strictly convex (objective
``||beta||^2``), so at a fixed ``tau`` its optimum is unique even though the
shared ``Sigma`` is rank-deficient here (N=300 controls > T1=253 pre-periods):
the regulariser ``P = I`` makes the QP strictly convex regardless. We verified
ECOS, OSQP, SCS and CLARABEL attain the same objective to solver precision.

This is a SOLVER cross-validation at a FIXED, matched ``tau`` (a single shared
``tau = 0.1 * max_j|eta_j|``, comfortably interior) -- not the CV-tuned ATE
(mlsynth tunes ``tau`` per firm by time-respecting validation; the paper uses a
future-leaking K-fold, so the tuned ATE differs by construction).

Emits the ``== REFERENCE VALUES ==`` block: per-firm de-standardised donor
weights as ``weight\\t<firm>::<donor>\\t<value>`` rows, plus summary scalars and
the cross-solver objective spread; then ``== SESSION INFO ==``.
"""
from __future__ import annotations

import platform
from pathlib import Path

import numpy as np
import cvxpy as cp
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "basedata" / "brexit_long.parquet"

TAU_FRAC = 0.1            # fixed interior tau = TAU_FRAC * max_j|eta_j|


def _layout():
    d = pd.read_parquet(DATA)
    w = d.pivot(index="time", columns="unit", values="y").sort_index()
    grp = d.drop_duplicates("unit").set_index("unit")["group"]
    uk = [u for u in w.columns if grp[u] == "UK"]
    ct = [u for u in w.columns if grp[u] == "control"]
    treated = d[d.unit == uk[0]].sort_values("time")
    T0 = int((treated["treat"] == 0).sum())
    Y = w[uk].to_numpy(float)
    X = w[ct].to_numpy(float)
    return Y, X, T0, list(uk), list(ct)


def _moments(Xpre, Ypre):
    """Shared standardised Sigma and per-firm eta (matches run_pda_multitreat)."""
    T1 = Xpre.shape[0]
    Mu = Xpre.mean(0)
    Sd = Xpre.std(0, ddof=1)
    Sd = np.where(Sd > 0, Sd, 1.0)
    Xt = (Xpre - Mu) / Sd
    Sigma = Xt.T @ Xt / T1
    muY = Ypre.mean(0)
    sdY = Ypre.std(0, ddof=1)
    sdY = np.where(sdY > 0, sdY, 1.0)
    Eta = Xt.T @ ((Ypre - muY) / sdY) / T1
    return Sigma, Eta, Mu, Sd, muY, sdY


def _solve_firm(Sigma, eta_j, tau, solver):
    N = Sigma.shape[0]
    b = cp.Variable(N)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(b)),
                      [eta_j - Sigma @ b <= tau, -eta_j + Sigma @ b <= tau])
    prob.solve(solver=getattr(cp, solver))
    return np.asarray(b.value, float).ravel(), float(prob.value)


def main() -> int:
    Y, X, T0, uk, ct = _layout()
    Xpre, Ypre = X[:T0], Y[:T0]
    Sigma, Eta, Mu, Sd, muY, sdY = _moments(Xpre, Ypre)
    J = Eta.shape[1]
    tau = TAU_FRAC * float(np.abs(Eta).max())

    # Cross-solver objective check on firm 0 (strictly convex => unique optimum).
    objs0 = {}
    for s in ("ECOS", "OSQP", "SCS", "CLARABEL"):
        try:
            _, o = _solve_firm(Sigma, Eta[:, 0], tau, s)
            objs0[s] = o
        except Exception:  # pragma: no cover - solver availability varies
            pass

    # Per-firm ECOS solve (authors' default family); de-standardise.
    rows = []
    beta_norms = []
    pre_rmses = []
    post_means = []
    for j in range(J):
        beta_tilde, _ = _solve_firm(Sigma, Eta[:, j], tau, "ECOS")
        beta_hat = sdY[j] * (beta_tilde / Sd)
        alpha = float(Ypre[:, j].mean() - Mu @ beta_hat)
        cf = X @ beta_hat + alpha
        beta_norms.append(float(np.sqrt(beta_hat @ beta_hat)))
        pre_rmses.append(float(np.sqrt(np.mean((Y[:T0, j] - cf[:T0]) ** 2))))
        post_means.append(float(np.mean(cf[T0:])))
        for d, val in zip(ct, beta_hat):
            rows.append((f"{uk[j]}::{d}", float(val)))

    print("== REFERENCE VALUES ==")
    print(f"tau\t{tau:.12g}")
    print(f"n_firms\t{J}")
    print(f"n_controls\t{Sigma.shape[0]}")
    print(f"T1\t{T0}")
    print(f"beta_norm_mean\t{float(np.mean(beta_norms)):.12g}")
    print(f"pre_rmse_mean\t{float(np.mean(pre_rmses)):.12g}")
    print(f"post_mean_pred_mean\t{float(np.mean(post_means)):.12g}")
    print(f"solver_obj_spread\t{max(objs0.values()) - min(objs0.values()):.12g}")
    for key, val in rows:
        print(f"weight\t{key}\t{val:.12g}")
    print()
    print("== SESSION INFO ==")
    print(f"python\t{platform.python_version()}")
    print(f"cvxpy\t{cp.__version__}")
    print(f"numpy\t{np.__version__}")
    print(f"pandas\t{pd.__version__}")
    print(f"solvers\t{','.join(objs0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
