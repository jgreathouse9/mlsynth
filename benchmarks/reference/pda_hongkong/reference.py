#!/usr/bin/env python3
"""Reference L2-relaxation solve for the ``pda_hongkong`` cross-validation.

Faithful port of the authors' ``Fun/L2relax.R`` (Shi & Wang, "L2-Relaxation for
Economic Prediction"; https://github.com/ishwang1/L2relax-PDA), vendored
alongside this file under ``Fun/L2relax.R`` for provenance. The R function

    standardize y1 (treated pre, length T1) and X1 (donor pre, T1 x N) by their
    mean/sd (intercept=TRUE, standardize=TRUE);
    Sigma = t(X1_tilde) %*% X1_tilde / T1 ;  eta = t(X1_tilde) %*% y1_tilde / T1 ;
    minimize ||beta||_2^2  s.t.  |eta - Sigma %*% beta| <= tau  (elementwise);
    de-standardize: beta_hat = sd1 * beta_tilde / Sd1 .

is reproduced here exactly with cvxpy.

Solver note
-----------
The authors' default solver is CVXR (MOSEK in their notebooks, ECOS_BB in the
``L2relax.R`` signature). CVXR is NOT installable in this environment's R
(no binary/source build of CVXR for R 4.3.3, and no MOSEK licence), so the
identical convex program is run via cvxpy with the authors' default open solver
family, ECOS. The L2-relaxation primal is strictly convex (objective
``||beta||^2``), so at a fixed ``tau`` its optimum is unique and solver-
invariant; we verified ECOS, OSQP, SCS and CLARABEL attain the same objective
and the same coefficients to solver precision.

This is a SOLVER cross-validation at a FIXED, matched ``tau`` -- not the
CV-tuned ATE (mlsynth and the paper use different cross-validation schemes, so
the tuned ATE differs by construction). ``tau`` is pinned deterministically to
``0.1 * max|eta|`` on the standardised pre-period moments, comfortably interior
so the QP is well conditioned; the Python case fixes mlsynth to the same value.

Emits the ``== REFERENCE VALUES ==`` block (de-standardised donor weights, the
fitted/predicted paths summary, and the objective) and ``== SESSION INFO ==``.
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np
import cvxpy as cp
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "basedata" / "HongKong.csv"

# Panel layout (matches mlsynth.utils.pda_helpers.setup.prepare_pda_inputs).
TREATED = "Hong Kong"
TAU_FRAC = 0.1            # fixed interior tau = TAU_FRAC * max|eta|


def _pivot():
    d = pd.read_csv(DATA)
    times = np.sort(pd.unique(d["Time"]))
    wide = d.pivot(index="Time", columns="Country", values="GDP").reindex(times)
    donors = [u for u in wide.columns if u != TREATED]
    y = wide[TREATED].to_numpy(float)
    X = wide[donors].to_numpy(float)
    intervention = d[d["Integration"] == 1]["Time"].min()
    T0 = int(np.sum(times < intervention))
    return y, X, T0, list(donors)


def _moments(y_pre, X_pre):
    """Standardised Sigma/eta and the (de)standardisation constants (L2relax.R)."""
    T1 = X_pre.shape[0]
    mu_y = float(y_pre.mean())
    Mu_X = X_pre.mean(0)
    sd_y = float(y_pre.std(ddof=1)) or 1.0
    Sd_X = X_pre.std(0, ddof=1)
    Sd_X = np.where(Sd_X > 0, Sd_X, 1.0)
    yt = (y_pre - mu_y) / sd_y
    Xt = (X_pre - Mu_X) / Sd_X
    Sigma = Xt.T @ Xt / T1
    eta = Xt.T @ yt / T1
    return Sigma, eta, mu_y, Mu_X, sd_y, Sd_X


def _solve(Sigma, eta, tau, solver):
    N = Sigma.shape[0]
    b = cp.Variable(N)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(b)),
                      [eta - Sigma @ b <= tau, -eta + Sigma @ b <= tau])
    prob.solve(solver=getattr(cp, solver))
    return np.asarray(b.value, float).ravel(), float(prob.value)


def main() -> int:
    y, X, T0, donors = _pivot()
    y_pre, X_pre = y[:T0], X[:T0]
    Sigma, eta, mu_y, Mu_X, sd_y, Sd_X = _moments(y_pre, X_pre)
    tau = TAU_FRAC * float(np.abs(eta).max())

    # Authors' default solver family is ECOS; verify the optimum is solver-
    # invariant (strictly convex QP => unique solution).
    beta_tilde, obj = _solve(Sigma, eta, tau, "ECOS")
    objs = {"ECOS": obj}
    for s in ("OSQP", "SCS", "CLARABEL"):
        try:
            _, o = _solve(Sigma, eta, tau, s)
            objs[s] = o
        except Exception:  # pragma: no cover - solver availability varies
            pass

    beta_hat = sd_y * (beta_tilde / Sd_X)
    alpha_hat = mu_y - float(Mu_X @ beta_hat)
    y1_hat = alpha_hat + X[:T0] @ beta_hat
    y2_hat = alpha_hat + X[T0:] @ beta_hat
    pre_rmse = float(np.sqrt(np.mean((y[:T0] - y1_hat) ** 2)))

    print("== REFERENCE VALUES ==")
    print(f"tau\t{tau:.12g}")
    print(f"obj_sqnorm\t{obj:.12g}")
    print(f"alpha\t{alpha_hat:.12g}")
    print(f"beta_l2_norm\t{float(np.sqrt(beta_hat @ beta_hat)):.12g}")
    print(f"pre_rmse\t{pre_rmse:.12g}")
    print(f"post_mean_pred\t{float(np.mean(y2_hat)):.12g}")
    print(f"solver_obj_spread\t{max(objs.values()) - min(objs.values()):.12g}")
    for lab, val in zip(donors, beta_hat):
        print(f"weight\t{lab}\t{float(val):.12g}")
    print()
    print("== SESSION INFO ==")
    print(f"python\t{platform.python_version()}")
    print(f"cvxpy\t{cp.__version__}")
    print(f"numpy\t{np.__version__}")
    print(f"pandas\t{pd.__version__}")
    print(f"solvers\t{','.join(objs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
