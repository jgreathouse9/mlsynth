"""Estimation + inference for ROLLDID (clean-room from Lee & Wooldridge 2026).

The method collapses the panel to one cross-sectional observation per unit by a
pre-treatment **rolling transformation** (demean = Procedure 2.1; detrend =
Procedure 3.1), then reads the ATT off a cross-sectional regression of the
transformed post-average on a treatment indicator. Common timing and staggered
adoption share the same machinery (the staggered aggregate is eq. 7.18-7.19,
with never-treated units as the comparison). Inference is exact-t (CLM
normality), HC3, or randomization.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from mlsynth.exceptions import MlsynthConfigError


def _transform_post(y: pd.Series, g, mode: str) -> pd.Series:
    """Unit-level transformed post-period outcome (``ẏ_it`` / ``ÿ_it``).

    ``demean`` subtracts the pre-period (``t < g``) mean; ``detrend`` subtracts
    the pre-period linear-trend projection extrapolated into the post period.
    """
    pre = y[y.index < g]
    post = y[y.index >= g]
    if mode == "demean":
        return post - pre.mean()
    A, B = np.polyfit(pre.index.astype(float), pre.values, 1)[::-1]   # intercept, slope
    return pd.Series(post.values - (A + B * post.index.astype(float).values),
                     index=post.index)


def _transform_at(y: pd.Series, g, t, mode: str) -> float:
    """The transformed value at a single post period ``t`` (for per-period ATTs)."""
    pre = y[y.index < g]
    if mode == "demean":
        return float(y.loc[t] - pre.mean())
    A, B = np.polyfit(pre.index.astype(float), pre.values, 1)[::-1]
    return float(y.loc[t] - (A + B * float(t)))


def _infer(yv: np.ndarray, Dv: np.ndarray, *, alpha: float, inference: str,
           ri_reps: int, seed: int) -> Dict[str, Any]:
    """ATT (coef on D) from ``yv on 1, D`` with the requested inference."""
    X = np.column_stack([np.ones_like(Dv, dtype=float), Dv.astype(float)])
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    att = float(beta[1])
    resid = yv - X @ beta
    n = len(yv)
    df = n - 2
    XtXinv = np.linalg.inv(X.T @ X)

    if inference == "hc3":
        h = np.diag(X @ XtXinv @ X.T)
        if np.any(1.0 - h < 1e-10):
            raise MlsynthConfigError(
                "HC3 inference is undefined when a unit has leverage 1 (e.g. only "
                "one treated or one control unit); use inference='exact' for a "
                "single treated unit.")
        meat = X.T @ np.diag((resid / (1.0 - h)) ** 2) @ X
        se = float(np.sqrt((XtXinv @ meat @ XtXinv)[1, 1]))
        method = "hc3"
    else:
        s2 = float(resid @ resid) / df
        se = float(np.sqrt(s2 * XtXinv[1, 1]))
        method = "ri" if inference == "ri" else "exact-t"

    tstat = att / se if se > 0 else float("nan")

    if inference == "ri":
        rng = np.random.default_rng(seed)
        n1 = int(Dv.sum())
        idx = np.arange(n)
        count = 0
        for _ in range(ri_reps):
            perm = np.zeros(n)
            perm[rng.choice(idx, n1, replace=False)] = 1.0
            Xp = np.column_stack([np.ones(n), perm])
            bp, *_ = np.linalg.lstsq(Xp, yv, rcond=None)
            if abs(bp[1]) >= abs(att) - 1e-12:
                count += 1
        p_value = count / ri_reps
    else:
        p_value = float(2.0 * stats.t.cdf(-abs(tstat), df))

    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    return {"att": att, "se": se, "t": tstat, "p_value": p_value,
            "ci_lower": att - tcrit * se, "ci_upper": att + tcrit * se,
            "method": method, "df": df}


def _ybar_aggregated(prep: Dict[str, Any], mode: str) -> Dict[Any, float]:
    """Per-unit collapsed regressand (eq. 7.18): a treated unit's own-cohort
    transformed post-average; a never-treated unit's cohort-share-weighted
    average over all cohort definitions."""
    Ywide = prep["Ywide"]
    cohort_of = prep["cohort_of"]
    cohorts = prep["cohorts"]
    Ng = {g: sum(1 for u in cohort_of if cohort_of[u] == g) for g in cohorts}
    Ntreat = len(cohort_of)
    omega = {g: Ng[g] / Ntreat for g in cohorts}
    ybar: Dict[Any, float] = {}
    for u in Ywide.columns:
        y = Ywide[u]
        if u in cohort_of:
            ybar[u] = float(_transform_post(y, cohort_of[u], mode).mean())
        else:
            ybar[u] = float(sum(omega[g] * _transform_post(y, g, mode).mean()
                                for g in cohorts))
    return ybar


def _per_period(prep: Dict[str, Any], mode: str, *, alpha: float,
                inference: str, ri_reps: int, seed: int) -> pd.DataFrame:
    """Per-period ATTs (common timing): tau_t from the transformed value at t."""
    Ywide = prep["Ywide"]
    cohort_of = prep["cohort_of"]
    g = prep["cohorts"][0]
    units = list(Ywide.columns)
    Dv = np.array([1.0 if u in cohort_of else 0.0 for u in units])
    post_times = [t for t in Ywide.index if t >= g]
    rows: List[dict] = []
    for t in post_times:
        yv = np.array([_transform_at(Ywide[u], g, t, mode) for u in units])
        r = _infer(yv, Dv, alpha=alpha, inference=inference, ri_reps=ri_reps, seed=seed)
        rows.append({"time": t, "att": r["att"], "se": r["se"], "t": r["t"],
                     "p_value": r["p_value"], "ci_lower": r["ci_lower"],
                     "ci_upper": r["ci_upper"]})
    return pd.DataFrame(rows)


def _per_cohort(prep: Dict[str, Any], mode: str, *, alpha: float,
                inference: str, ri_reps: int, seed: int) -> pd.DataFrame:
    """Per-cohort ATTs (staggered): tau_g, cohort-g treated vs never-treated.

    The descriptive per-cohort SEs always use exact-t: a cohort can have a single
    treated unit (HC3 is then degenerate), and the requested ``inference`` mode
    governs the headline *aggregate* ATT, not this breakdown.
    """
    Ywide = prep["Ywide"]
    cohort_of = prep["cohort_of"]
    never = prep["never"]
    rows: List[dict] = []
    for g in prep["cohorts"]:
        treated_g = [u for u in cohort_of if cohort_of[u] == g]
        units = treated_g + never
        yv = np.array([float(_transform_post(Ywide[u], g, mode).mean()) for u in units])
        Dv = np.array([1.0] * len(treated_g) + [0.0] * len(never))
        r = _infer(yv, Dv, alpha=alpha, inference="exact", ri_reps=ri_reps, seed=seed)
        rows.append({"cohort": g, "n_treated": len(treated_g), "att": r["att"],
                     "se": r["se"], "p_value": r["p_value"],
                     "ci_lower": r["ci_lower"], "ci_upper": r["ci_upper"]})
    return pd.DataFrame(rows)


def estimate(prep: Dict[str, Any], *, mode: str, inference: str, alpha: float,
             ri_reps: int, seed: int) -> Dict[str, Any]:
    """Run the rolling-DiD estimate end to end and return a result dict."""
    Ywide = prep["Ywide"]
    cohort_of = prep["cohort_of"]
    units = list(Ywide.columns)

    ybar = _ybar_aggregated(prep, mode)
    yv = np.array([ybar[u] for u in units])
    Dv = np.array([1.0 if u in cohort_of else 0.0 for u in units])
    agg = _infer(yv, Dv, alpha=alpha, inference=inference, ri_reps=ri_reps, seed=seed)

    out: Dict[str, Any] = {
        "design": prep["design"], "transformation": mode,
        "n_treated": len(cohort_of), "n_control": len(prep["never"]),
        "aggregate": agg, "per_period": None, "per_cohort": None,
    }

    if prep["design"] == "common":
        out["per_period"] = _per_period(prep, mode, alpha=alpha, inference=inference,
                                        ri_reps=ri_reps, seed=seed)
        # event-study series: treated-unit (mean) observed path + per-period gap
        g = prep["cohorts"][0]
        treated_units = [u for u in units if u in cohort_of]
        observed = Ywide[treated_units].mean(axis=1)
        pp = out["per_period"].set_index("time")["att"]
        gap = pd.Series(0.0, index=Ywide.index)
        gap.loc[pp.index] = pp.values
        out["time_series"] = {
            "observed": observed.to_numpy(),
            "counterfactual": (observed - gap).to_numpy(),
            "gap": gap.to_numpy(),
            "time_periods": Ywide.index.to_numpy(),
            "intervention_time": g,
        }
    else:
        out["per_cohort"] = _per_cohort(prep, mode, alpha=alpha, inference=inference,
                                        ri_reps=ri_reps, seed=seed)
        out["time_series"] = None
    return out
