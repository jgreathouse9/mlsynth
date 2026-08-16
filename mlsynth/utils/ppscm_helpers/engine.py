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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cvxpy as cp
import numpy as np

from ...exceptions import MlsynthConfigError

_EPS = 1e-12


@dataclass(frozen=True)
class Conventions:
    """The three choices that decide which estimator a fit is.

    Carried as one object because they travel together and are forgotten
    separately: every place that refits the panel -- the jackknife, the
    conformal calibration's rolling origins -- has to run the estimator the
    caller configured, and three keyword arguments with defaults let a refit
    site keep an older signature and silently pin itself to augsynth's. A
    fourth convention added here reaches every refit without anyone editing
    them.

    The defaults are augsynth's ``multisynth``, so a fit that says nothing is
    the port it has always been.
    """

    donor_weights: str = "scm"
    base_period: str = "all_pre"
    donor_pool: str = "window"


def fit_feff(Xy: np.ndarray, trt: np.ndarray, adopt_indices, fixedeff: bool,
             base_period: str = "all_pre") -> Dict[int, np.ndarray]:
    """Residualize ``Xy`` per cohort.

    Returns ``{adoption_index: residual_matrix (n, T)}``. With ``fixedeff`` the
    time effect is the never-treated column mean and the unit effect is a
    pre-adoption baseline; without it, only the time effect (control averages)
    is removed.

    ``base_period`` selects the baseline, and the two are different estimators:

    * ``"all_pre"`` -- each unit's mean over its whole pre-adoption window
      ``[:tj]``. This is augsynth's ``fit_feff`` (``rowMeans(residuals[, 1:tj])``)
      and the default, verified against it to 0.0 over 300 random panels.
    * ``"pre_treatment"`` -- the single period ``tj - 1``, which is the base
      period Callaway-Sant'Anna and Sun-Abraham normalise against. Choosing it
      is one of the three conventions that make this estimator theirs (#465);
      on its own it shifts each cohort's level without moving the event-study
      shape.
    """
    if base_period not in ("all_pre", "pre_treatment"):
        raise ValueError(
            f"base_period must be 'all_pre' or 'pre_treatment', got {base_period!r}.")
    ever = np.isfinite(trt)
    time_eff = np.nanmean(Xy[~ever], axis=0)
    base = Xy - time_eff[None, :]
    out: Dict[int, np.ndarray] = {}
    for tj in adopt_indices:
        tj = int(tj)
        if fixedeff:
            unit_eff = (np.nanmean(base[:, :tj], axis=1) if base_period == "all_pre"
                        else base[:, tj - 1])
            out[tj] = base - unit_eff[:, None]
        else:
            out[tj] = base
    return out


def eligible_donors(trt: np.ndarray, adopt: int, n_leads: int, donor_pool: str) -> np.ndarray:
    """Indices admissible as donors for a cohort adopting at ``adopt``.

    Never-treated units carry a non-finite adoption time, so they satisfy every
    rule below. The three differ only in which *later-adopting* units they also
    admit, and that choice is what separates this estimator from
    Callaway-Sant'Anna when a later cohort outlives an earlier one's window:

    * ``"window"`` -- untreated through this cohort's whole estimation window,
      ``trt > adopt + n_leads``. augsynth's rule, and the default.
    * ``"never_treated"`` -- never treated at all, which is what CS and
      Sun-Abraham use by default.
    * ``"not_yet_treated"`` -- untreated as of this cohort's adoption.

    ``"window"`` and ``"never_treated"`` coincide exactly when every other
    cohort adopts inside the window, which is why the three estimators agree to
    machine precision there and diverge otherwise.
    """
    if donor_pool == "window":
        return np.where(trt > adopt + n_leads)[0]
    if donor_pool == "never_treated":
        return np.where(~np.isfinite(trt))[0]
    if donor_pool == "not_yet_treated":
        return np.where(trt > adopt)[0]
    raise ValueError(
        f"donor_pool must be 'window', 'never_treated' or 'not_yet_treated', "
        f"got {donor_pool!r}.")


def uniform_weights(donors, groups, n) -> Dict[Any, np.ndarray]:
    """Equal weight on every admissible donor -- the CS/Sun-Abraham comparison.

    Imposed, not approximated. Driving ``lam`` up does not reach it: the QP's
    weights saturate at ``max|w - 1/J| = 3.47e-03`` and do not move between
    ``lam = 1e9`` and ``1e18`` at any ``nu``.
    """
    W: Dict[Any, np.ndarray] = {}
    for g in groups:
        Wg = np.zeros(n)
        if len(donors[g]):
            Wg[donors[g]] = 1.0 / len(donors[g])
        W[g] = Wg
    return W


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
                    nu, norm_pool, norm_sep, lam, solver,
                    zt=None, Zc=None) -> Dict[Any, np.ndarray]:
    """Partially-pooled QP: per-cohort simplex weights (summing to cohort size).

    When ``zt``/``Zc`` (per-cohort scaled auxiliary-covariate target sums and
    donor blocks) are supplied, the covariate imbalance is stacked into the
    pooled and separate terms (normalized by the number of covariates),
    following augsynth::multisynth Sec 5.2.
    """
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
    if zt is not None:
        dz = zt[0].shape[0]
        imbz = [zt[k] - Zc[k].T @ w[k] for k in range(J)]
        sepz = sum(cp.sum_squares(imbz[k]) / dz for k in range(J))
        pooledz = cp.sum_squares(sum(imbz)) / dz
        obj = obj + nu / (norm_pool * J ** 2) * pooledz + (1 - nu) / (norm_sep * J) * sepz
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


def predict_tau(res, groups, adopt_of, members, donors, W, n1, H, n, bs_weight=None):
    """Relative-time ``tau`` per cohort, plus the n1-weighted event study and ATT.

    With ``bs_weight`` (per-unit multipliers, default all ones) this is augsynth's
    ``predict.multisynth(bs_weight=...)`` written in residual space: the
    fixed-effect terms cancel between the treated mean and the synthetic, leaving
    the treated residuals (scaled by ``bs_weight``, averaged over the cohort) minus
    the donor residuals weighted by ``W[g] * bs_weight``. ``bs_weight = ones``
    reproduces the point estimate exactly.
    """
    if bs_weight is None:
        bs_weight = np.ones(n)
    J = len(groups)
    tau_rel = np.full((J, H), np.nan)
    for k, g in enumerate(groups):
        tj = adopt_of[g]
        resid = res[tj]
        mem = members[g]
        mu1 = (bs_weight[mem, None] * resid[mem]).sum(axis=0) / n1[k]
        mu0 = (W[g] * bs_weight) @ resid
        tt = mu1 - mu0
        hi = min(tj + H, resid.shape[1])
        tau_rel[k, : hi - tj] = tt[tj:hi]
    denom = np.array([np.nansum(n1 * ~np.isnan(tau_rel[:, h])) for h in range(H)])
    per_time = np.array([np.nansum(n1 * tau_rel[:, h]) / denom[h] if denom[h] > 0 else np.nan
                         for h in range(H)])
    att = float(np.nanmean(per_time))
    return tau_rel, per_time, att


def _scale_covariates(Z: np.ndarray, trt: np.ndarray, res_first: np.ndarray, d: int):
    """augsynth Sec 5.2 scaling (multi_synth_qp.R line 98): z-score each
    covariate against the never-treated controls, then rescale by ``sdx`` so
    covariate and outcome imbalance share a scale.

    ``sdx`` is ``sd(X[[1]][is.finite(trt)])`` -- the standard deviation over the
    treated-ever rows of the *first cohort's* residualized pre-treatment block
    (``res_first``, shape ``(n, T)``; the first ``d`` columns are used)."""
    ever = np.isfinite(trt)
    ctrl = ~ever
    sdx = float(np.std(res_first[ever][:, :d], ddof=1))
    mu = Z[ctrl].mean(axis=0)
    sd = Z[ctrl].std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    return sdx * (Z - mu) / sd


def _uniform_fit(res, groups, adopt_of, members, donors, n1, d, n, n_leads,
                 nu, lam) -> Dict[str, Any]:
    """The fit with equal donor weights, where there is no QP to solve.

    Every diagnostic the partially-pooled path reports is a property of the
    solved weights relative to the uniform baseline, so with uniform weights the
    scaled imbalances are 1 by construction and ``nu`` and ``lam`` describe a
    program that was not posed. They are reported as such instead of being
    given values that would read as if a QP had run.
    """
    W = uniform_weights(donors, groups, n)
    M = _imbalance_matrix(res, groups, adopt_of, members, donors, W, n1, d, n)
    J = len(groups)
    nnz = [max(int((np.abs(M[:, k]) > 1e-12).sum()), 1) for k in range(J)]
    fin_global = float(np.sqrt((M.mean(axis=1) ** 2).sum()) / np.sqrt(d))
    fin_ind = float(np.sqrt(np.mean([(M[:, k] ** 2).sum() / nnz[k] for k in range(J)])))
    tau_rel, per_time, att = predict_tau(
        res, groups, adopt_of, members, donors, W, n1, n_leads, n)
    return {
        "groups": groups, "members": members, "adopt_of": adopt_of, "donors": donors,
        "weights": W, "n1": n1, "tau_rel": tau_rel, "per_time": per_time, "att": att,
        "nu_used": float("nan"), "global_l2": fin_global, "ind_l2": fin_ind,
        "scaled_global_l2": 1.0, "scaled_ind_l2": 1.0,
        "M": M, "nnz": np.asarray(nnz, dtype=float),
        "res": res, "n": n, "n_leads": n_leads,
    }


def run_multisynth(
    Xy: np.ndarray, trt: np.ndarray, d: int, n_leads: int, n_lags: int,
    *, fixedeff: bool = True, time_cohort: bool = False,
    nu: Optional[float] = None, lam: float = 0.0, solver: Any = None,
    Z: Optional[np.ndarray] = None,
    conventions: Conventions = Conventions(),
) -> Dict[str, Any]:
    """Run one multisynth fit; returns weights, event study, ATT, diagnostics."""
    donor_weights = conventions.donor_weights
    base_period = conventions.base_period
    donor_pool = conventions.donor_pool
    # A non-finite pooling level builds an objective CVXPY cannot recognise, and
    # the solver's complaint is the first anyone hears of it. The uniform fit
    # poses no program and records ``nu_used = NaN``, so a caller handing that
    # value back to a refit is the path that produced #467.
    if nu is not None and not np.isfinite(nu):
        raise MlsynthConfigError(
            f"nu must be finite or None (None selects augsynth's ratio); got {nu!r}. "
            "A uniform-weight fit poses no quadratic program and reports "
            "nu_used = NaN, which is not a pooling level a later fit can reuse.")
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
    donors = {g: eligible_donors(trt, adopt_of[g], n_leads, donor_pool)
              for g in groups}
    res = fit_feff(Xy, trt, set(adopt_of.values()), fixedeff, base_period)
    if donor_weights not in ("scm", "uniform"):
        raise ValueError(
            f"donor_weights must be 'scm' or 'uniform', got {donor_weights!r}.")

    # scaled auxiliary-covariate target sums (zt) and donor blocks (Zc) per cohort
    zt = Zc = None
    if Z is not None:
        # augsynth scales by the first cohort's residual sd (multi_synth_qp.R:98)
        res_first = res[adopt_of[groups[0]]]
        Zsc = _scale_covariates(Z, trt, res_first, d)
        zt = [Zsc[members[g]].sum(axis=0) for g in groups]
        Zc = [Zsc[donors[g]] for g in groups]

    if donor_weights == "uniform":
        return _uniform_fit(res, groups, adopt_of, members, donors, n1, d, n,
                            n_leads, nu, lam)

    # separate fit (nu = 0) -> scaling norms + auto-nu
    W0 = solve_cohort_qp(res, groups, adopt_of, members, donors, n1, d, n, n_lags,
                         0.0, 1.0, 1.0, lam, solver, zt=zt, Zc=Zc)
    M0 = _imbalance_matrix(res, groups, adopt_of, members, donors, W0, n1, d, n)
    avg_imbal = M0.mean(axis=1)
    global_l2 = float(np.sqrt((avg_imbal ** 2).sum()) / np.sqrt(d))
    avg_l2 = float(np.mean([np.sqrt((M0[:, k] ** 2).sum()) for k in range(J)]))
    nnz = [max(int((np.abs(M0[:, k]) > 1e-12).sum()), 1) for k in range(J)]
    ind_l2 = float(np.sqrt(np.mean([(M0[:, k] ** 2).sum() / nnz[k] for k in range(J)])))
    nu_used = float(global_l2 * np.sqrt(d) / avg_l2) if nu is None else float(nu)

    W = solve_cohort_qp(res, groups, adopt_of, members, donors, n1, d, n, n_lags,
                        nu_used, global_l2 ** 2, ind_l2 ** 2, lam, solver, zt=zt, Zc=Zc)
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
    tau_rel, per_time, att = predict_tau(
        res, groups, adopt_of, members, donors, W, n1, H, n)

    return {
        "groups": groups, "members": members, "adopt_of": adopt_of, "donors": donors,
        "weights": W, "n1": n1, "tau_rel": tau_rel, "per_time": per_time, "att": att,
        "nu_used": nu_used, "global_l2": fin_global, "ind_l2": fin_ind,
        "scaled_global_l2": fin_global / unif_global, "scaled_ind_l2": fin_ind / unif_ind,
        # Per-cohort final imbalance columns ``M`` (d, J) and the ``nnz`` counts
        # from the separate fit -- the components of ``ind_l2`` (each cohort's
        # in-sample error is ``sqrt((M[:,k]**2).sum() / nnz[k])``), returned so the
        # estimator can surface per-unit fit without recomputing the QP.
        "M": M, "nnz": np.asarray(nnz, dtype=float),
        "res": res, "n": n, "n_leads": n_leads,
    }
