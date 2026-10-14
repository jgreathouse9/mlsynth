"""FEct, the fixed-effects counterfactual estimator, in the form the paper runs it.

Reference: Liu, Licheng, Ye Wang and Yiqing Xu (2022), "A Practical Guide to
Counterfactual Estimators for Causal Inference with Time-Series Cross-Sectional
Data", *American Journal of Political Science*, DOI 10.1111/ajps.12723.
Reference implementation: ``fect`` 0.4.1, shipped in the replication package.

The estimator is two-way fixed effects fitted on the UNTREATED cells alone,
extrapolated onto the treated ones, and the gaps averaged:

    Y_it(0)_hat = mu + alpha_i + xi_t    estimated on { (i,t) : D_it = 0 }
    tau_it      = Y_it - Y_it(0)_hat     for  { (i,t) : D_it = 1 }

That is the whole method. It differs from two-way-fixed-effects regression in
where the fixed effects come from: TWFE fits them on every cell, including the
treated ones whose contamination is the thing being measured.

Two details decide whether it reproduces the published numbers.

``min_T0`` drops treated units with too few pre-treatment observations;
``fect`` 0.4.1 defaults it to 5 (``R/default.R:61``) and the paper's second
example passes 4 explicitly. It is counted as the number of untreated
observations the unit has, NOT the length of its leading untreated run. The two
differ whenever treatment reverses, and on the paper's first example the
distinction moves the reported ATT by 0.7%.

``weighting`` chooses between the two ATTs ``fect`` prints: "Tr obs equally
weighted" averages over treated observations, "Tr units equally weighted"
averages the per-unit means. They answer different questions and on the paper's
examples they differ by a factor of 1.4 to 2.2.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def untreated_two_way_fit(y: np.ndarray, unit: np.ndarray, time: np.ndarray,
                          tol: float = 1e-13, max_iter: int = 5000
                          ) -> Tuple[float, pd.Series, pd.Series, int]:
    """Two-way fixed effects by alternating projections, on the cells given.

    Returns ``(grand_mean, unit_effects, time_effects, iterations)``. Alternating
    projections instead of a dummy-variable regression because the panels here
    carry 1200+ units, and the fit is wanted on an arbitrary (unbalanced) subset
    of cells, which is what the untreated set is.
    """
    y = np.asarray(y, dtype=float)
    mu = float(y.mean())
    a = pd.Series(0.0, index=np.unique(unit))
    b = pd.Series(0.0, index=np.unique(time))
    it = 0
    for it in range(1, max_iter + 1):
        r = y - mu - a.reindex(unit).to_numpy() - b.reindex(time).to_numpy()
        da = pd.Series(r).groupby(unit).mean()
        a = a.add(da, fill_value=0.0)
        r = y - mu - a.reindex(unit).to_numpy() - b.reindex(time).to_numpy()
        db = pd.Series(r).groupby(time).mean()
        b = b.add(db, fill_value=0.0)
        if max(abs(da).max(), abs(db).max()) < tol:
            break
    return mu, a, b, it


def dropped_units(df: pd.DataFrame, unit: str, treat: str, min_T0: int) -> np.ndarray:
    """Treated units with fewer than ``min_T0`` untreated observations.

    ``fect`` counts untreated OBSERVATIONS, so a unit whose treatment switches
    off again keeps contributing to its own T0. Counting the leading untreated
    run instead is the obvious reading and the wrong one.
    """
    ever = df.groupby(unit)[treat].max()
    n_untreated = (df[df[treat] == 0].groupby(unit).size()
                   .reindex(ever.index).fillna(0))
    return ever.index[(ever == 1) & (n_untreated < min_T0)].to_numpy()


def fect_fe(df: pd.DataFrame, unit: str, time: str, outcome: str, treat: str,
            *, min_T0: int = 5) -> Dict[str, object]:
    """FEct on a long panel. Returns both ATT weightings and the sample it used."""
    d = df.dropna(subset=[outcome, treat, unit, time]).sort_values([unit, time])
    drop = dropped_units(d, unit, treat, min_T0)
    d = d[~d[unit].isin(drop)]

    u = d[d[treat] == 0]
    mu, a, b, iters = untreated_two_way_fit(
        u[outcome].to_numpy(), u[unit].to_numpy(), u[time].to_numpy())

    tr = d[d[treat] == 1]
    yhat = (mu + a.reindex(tr[unit].to_numpy()).to_numpy()
            + b.reindex(tr[time].to_numpy()).to_numpy())
    gap = tr[outcome].to_numpy(dtype=float) - yhat
    per_unit = pd.Series(gap, index=tr[unit].to_numpy()).groupby(level=0).mean()
    return {
        "att_obs_weighted": float(np.mean(gap)),
        "att_unit_weighted": float(per_unit.mean()),
        "n_treated_obs": int(gap.size),
        "n_treated_units": int(per_unit.size),
        "n_units_dropped": int(drop.size),
        "n_units": int(d[unit].nunique()),
        "iterations": iters,
    }


def twfe(df: pd.DataFrame, unit: str, time: str, outcome: str, treat: str
         ) -> Dict[str, float]:
    """Two-way fixed effects with an HC1 standard error.

    ``fastplm``'s "Robust" variance is heteroskedasticity-robust, not clustered;
    clustering on the unit gives 0.185 on the first example against the 0.161
    the paper reports, which reads as a mismatch and is not one.
    """
    d = df.dropna(subset=[outcome, treat, unit, time])

    def demean(s: pd.Series) -> np.ndarray:
        v = s.to_numpy(dtype=float).copy()
        for _ in range(3000):
            v0 = v.copy()
            v -= pd.Series(v).groupby(d[unit].values).transform("mean").to_numpy()
            v -= pd.Series(v).groupby(d[time].values).transform("mean").to_numpy()
            if np.max(np.abs(v - v0)) < 1e-13:
                break
        return v

    y, x = demean(d[outcome]), demean(d[treat].astype(float))
    beta = float((x @ y) / (x @ x))
    resid = y - beta * x
    n = len(d)
    k = d[unit].nunique() + d[time].nunique() - 1 + 1     # absorbed params + slope
    dof = n - k
    bread = 1.0 / (x @ x)
    hc1 = float(np.sqrt(bread * np.sum((x ** 2) * (resid ** 2)) * bread * n / dof))
    return {"coef": beta, "se_hc1": hc1, "t": beta / hc1,
            "residual_sd": float(np.sqrt(np.sum(resid ** 2) / dof)),
            "dof": int(dof), "n": int(n)}
