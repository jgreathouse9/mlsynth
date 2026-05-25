"""Core staggered-adoption engine for PPSCM, ported faithfully from
augsynth::multisynth (Ben-Michael, Feller & Rothstein 2022).

Pipeline (one call = one fit):
  1. ``fit_feff`` removes fixed effects (force=3 two-way: time effect from
     never-treated column means + per-cohort unit pre-mean) and balances the
     residuals.
  2. ``solve_cohort_qp`` solves the partially-pooled QP over donor weights,
     with the pooled imbalance aligned by **relative time** (front-padded) and
     the pooled/separate terms normalized by the separate fit's norms.
  3. ``run_multisynth`` chooses ``nu`` (triangle-inequality ratio when "auto"),
     refits, and produces the relative-time event study and ATT.

Validated to reproduce the multisynth vignette exactly (default nu=0.2607,
ATT=-0.011; time_cohort nu=0.3939, ATT=-0.017).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cvxpy as cp
import numpy as np

_EPS = 1e-12


def fit_feff(Xy: np.ndarray, trt: np.ndarray, adopt_indices, fixedeff: bool) -> Dict[int, np.ndarray]:
    """Residualize ``Xy`` per cohort.

    Returns ``{adoption_index: residual_matrix (n, T)}``. With ``fixedeff`` the
    time effect is the never-treated column mean and the unit effect is each
    unit's mean residual over its pre-adoption window ``[:tj]``; without it,
    only the time effect (control averages) is removed.
    """
    ever = np.isfinite(trt)
    time_eff = np.nanmean(Xy[~ever], axis=0)
    base = Xy - time_eff[None, :]
    out: Dict[int, np.ndarray] = {}
    for tj in adopt_indices:
        if fixedeff:
            unit_eff = np.nanmean(base[:, : int(tj)], axis=1)
            out[int(tj)] = base - unit_eff[:, None]
        else:
            out[int(tj)] = base
    return out


def _padded(bal: np.ndarray, members: List[int], donors: np.ndarray, tj: int, d: int):
    """Relative-time-aligned (front-padded) treated target sum and donor block."""
    xt = np.zeros(d)
    xt[d - tj:] = bal[members][:, :tj].sum(axis=0)        # colSums over cohort members
    Xc = np.zeros((len(donors), d))
    Xc[:, d - tj:] = bal[donors][:, :tj]
    return xt, Xc


def _imbalance_matrix(res, groups, adopt_of, members, donors, W, n1, d, n) -> np.ndarray:
    """Per-cohort pre-treatment imbalance vectors (d, J) on the sum scale."""
    M = np.zeros((d, len(groups)))
    for k, g in enumerate(groups):
        tj = adopt_of[g]
        bal = res[tj][:, :d]
        xt, _ = _padded(bal, members[g], donors[g], tj, d)
        Xc = np.zeros((n, d)); Xc[:, d - tj:] = bal[:, :tj]
        M[:, k] = xt - Xc.T @ (W[g] * n1[k])
    return M


def solve_cohort_qp(res, groups, adopt_of, members, donors, n1, d, n, n_lags,
                    nu, norm_pool, norm_sep, lam, solver) -> Dict[Any, np.ndarray]:
    """Partially-pooled QP: per-cohort simplex weights (summing to cohort size)."""
    J = len(groups)
    w = [cp.Variable(len(donors[g]), nonneg=True) for g in groups]
    imb = []
    for k, g in enumerate(groups):
        tj = adopt_of[g]
        bal = res[tj][:, :d]
        xt, Xc = _padded(bal, members[g], donors[g], tj, d)
        imb.append(xt - Xc.T @ w[k])
    sep = sum(cp.sum_squares(imb[k]) / max(adopt_of[g], 1) for k, g in enumerate(groups))
    pooled = cp.sum_squares(sum(imb)) / n_lags
    obj = nu / (norm_pool * J ** 2) * pooled + (1 - nu) / (norm_sep * J) * sep
    if lam:
        obj = obj + lam * sum(cp.sum_squares(wj) for wj in w)
    cons = [cp.sum(w[k]) == n1[k] for k in range(J)]
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=solver or cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, max_iter=400000)
    if any(wj.value is None for wj in w):
        prob.solve(solver=cp.SCS)
    W: Dict[Any, np.ndarray] = {}
    for k, g in enumerate(groups):
        wv = np.clip(np.asarray(w[k].value, dtype=float), 0.0, None)
        Wg = np.zeros(n); Wg[donors[g]] = wv / n1[k]
        W[g] = Wg
    return W


def run_multisynth(
    Xy: np.ndarray, trt: np.ndarray, d: int, n_leads: int, n_lags: int,
    *, fixedeff: bool = True, time_cohort: bool = False,
    nu: Optional[float] = None, lam: float = 0.0, solver: Any = None,
) -> Dict[str, Any]:
    """Run one multisynth fit; returns weights, event study, ATT, diagnostics."""
    n = Xy.shape[0]
    ever = np.where(np.isfinite(trt))[0]

    if time_cohort:
        groups = sorted({int(trt[i]) for i in ever})
        members = {g: [i for i in ever if int(trt[i]) == g] for g in groups}
        adopt_of = {g: g for g in groups}
    else:
        groups = list(range(len(ever)))
        members = {k: [int(ever[k])] for k in groups}
        adopt_of = {k: int(trt[ever[k]]) for k in groups}
    J = len(groups)
    n1 = np.array([len(members[g]) for g in groups], dtype=float)
    donors = {g: np.where(trt > adopt_of[g] + n_leads)[0] for g in groups}
    res = fit_feff(Xy, trt, set(adopt_of.values()), fixedeff)

    # separate fit (nu = 0) -> scaling norms + auto-nu
    W0 = solve_cohort_qp(res, groups, adopt_of, members, donors, n1, d, n, n_lags,
                         0.0, 1.0, 1.0, lam, solver)
    M0 = _imbalance_matrix(res, groups, adopt_of, members, donors, W0, n1, d, n)
    avg_imbal = M0.mean(axis=1)
    global_l2 = float(np.sqrt((avg_imbal ** 2).sum()) / np.sqrt(d))
    avg_l2 = float(np.mean([np.sqrt((M0[:, k] ** 2).sum()) for k in range(J)]))
    nnz = [max(int((np.abs(M0[:, k]) > 1e-12).sum()), 1) for k in range(J)]
    ind_l2 = float(np.sqrt(np.mean([(M0[:, k] ** 2).sum() / nnz[k] for k in range(J)])))
    nu_used = float(global_l2 * np.sqrt(d) / avg_l2) if nu is None else float(nu)

    W = solve_cohort_qp(res, groups, adopt_of, members, donors, n1, d, n, n_lags,
                        nu_used, global_l2 ** 2, ind_l2 ** 2, lam, solver)
    M = _imbalance_matrix(res, groups, adopt_of, members, donors, W, n1, d, n)

    # uniform-weight baseline for scaled imbalance
    Wunif = {g: (np.isin(np.arange(n), donors[g]).astype(float) / max(len(donors[g]), 1))
             for g in groups}
    Mu = _imbalance_matrix(res, groups, adopt_of, members, donors, Wunif, n1, d, n)
    unif_global = float(np.sqrt((Mu.mean(axis=1) ** 2).sum()) / np.sqrt(d)) + _EPS
    unif_ind = float(np.sqrt(np.mean([(Mu[:, k] ** 2).sum() / nnz[k] for k in range(J)]))) + _EPS
    fin_global = float(np.sqrt((M.mean(axis=1) ** 2).sum()) / np.sqrt(d))
    fin_ind = float(np.sqrt(np.mean([(M[:, k] ** 2).sum() / nnz[k] for k in range(J)])))

    # predict: relative-time tau per cohort, n1-weighted per-horizon average
    H = n_leads
    tau_rel = np.full((J, H), np.nan)
    for k, g in enumerate(groups):
        tj = adopt_of[g]; resid = res[tj]
        treated_resid = resid[members[g]].mean(axis=0)
        tt = treated_resid - W[g] @ resid
        hi = min(tj + H, Xy.shape[1])
        tau_rel[k, : hi - tj] = tt[tj:hi]
    denom = np.array([np.nansum(n1 * ~np.isnan(tau_rel[:, h])) for h in range(H)])
    per_time = np.array([np.nansum(n1 * tau_rel[:, h]) / denom[h] if denom[h] > 0 else np.nan
                         for h in range(H)])
    att = float(np.nanmean(per_time))

    return {
        "groups": groups, "members": members, "adopt_of": adopt_of, "donors": donors,
        "weights": W, "n1": n1, "tau_rel": tau_rel, "per_time": per_time, "att": att,
        "nu_used": nu_used, "global_l2": fin_global, "ind_l2": fin_ind,
        "scaled_global_l2": fin_global / unif_global, "scaled_ind_l2": fin_ind / unif_ind,
    }
