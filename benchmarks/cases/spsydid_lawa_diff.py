"""Differential cross-validation: mlsynth SpSyDiD vs the authors' code on real data.

A tight, real-data companion to ``spsydid_state_mc`` (which is the paper's
Monte-Carlo simulation). Here the two implementations are run on the *same real
panel* -- the Arizona LAWA CPS extract aggregated to state means -- and the
mlsynth estimator is exercised end-to-end through :meth:`mlsynth.SpSyDiD.fit`
(never its internal helpers), then compared value-for-value against the authors'
own ``functions_ssdid`` (serenini/spatial_SDID, cloned on demand).

Why this is the exact test
--------------------------
The SDID machinery is identical across the two codebases -- the unit-weight QP,
the time-weight QP, and the regularisation ``zeta`` all agree to solver
tolerance. The only thing that ever differed between them is a handful of
*non-fitted* row-weight conventions that the paper leaves underspecified and the
authors' notebook fills with ``join_weights`` defaults (post-period time weight
``T_post/T``, affected-unit weight = the mean treated weight). Those depart from
canonical SDID; mlsynth uses the canonical choices (``1/T_post`` post weights,
``1/N_sp`` affected weights). Unifying the reference onto the canonical
convention, the two agree to ``~1e-8``:

* unit weights (``res.weights.donor_weights``)           -- to ~1e-9;
* direct ATT (``res.att``)                               -- to ~1e-7;
* spillover coefficient (``res.effects.additional_effects['aite']``) -- to ~1e-8.

Panel and W
-----------
* Panel: ``basedata/cps_lawa_arizona.parquet`` aggregated to survey-weighted mean
  log weekly earnings per state-month; Arizona treated from period 55.
* W: ``basedata/US_no_islands_matrix.gal`` (queen contiguity), parsed directly
  (no libpysal) and restricted/row-standardised to the 45 states common to the
  matrix and the CPS extract. Arizona (FIPS 4) has four present neighbours (CA,
  NV, CO, NM).

Reference: serenini/spatial_SDID (cloned on demand, pinned; no licence, not
vendored) -- its ``fit_unit_weights`` / ``fit_time_weights`` under the canonical
final regression.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import block_diag, lstsq

from benchmarks.compare import BenchmarkSkipped

_BASE = Path(__file__).resolve().parents[2] / "basedata"
_TREATED_FIPS = 4          # Arizona
_TSTAR = 55                # first treated period
_KAPPA_TOL = 1e-6


def _parse_gal(path: Path):
    lines = [ln.strip() for ln in open(path) if ln.strip()]
    order, adj, i = [], {}, 1
    while i < len(lines):
        uid, k = lines[i].split()[:2]
        uid, k = int(uid), int(k)
        i += 1
        adj[uid] = [int(x) for x in lines[i].split()] if k > 0 else []
        i += 1
        order.append(uid)
    return order, adj


def _panel_and_W():
    d = pd.read_parquet(_BASE / "cps_lawa_arizona.parquet")
    d["wy"] = d["wklyearn"] * d["weight"]
    p = ((d.groupby(["statefip", "period"])["wy"].sum()
          / d.groupby(["statefip", "period"])["weight"].sum())
         .rename("UR2").reset_index().rename(columns={"statefip": "ID", "period": "month"}))
    order, adj = _parse_gal(_BASE / "US_no_islands_matrix.gal")
    common = [f for f in order if f in set(p["ID"].unique())]
    idx = {f: i for i, f in enumerate(common)}
    A = np.zeros((len(common), len(common)))
    for f in common:
        for nb in adj[f]:
            if nb in idx:
                A[idx[f], idx[nb]] = 1.0
    W1 = A / np.where(A.sum(1, keepdims=True) == 0, 1, A.sum(1, keepdims=True))
    p = p[p["ID"].isin(common)].copy()
    p["ID"] = pd.Categorical(p["ID"], categories=common, ordered=True)
    p = p.sort_values(["month", "ID"]).reset_index(drop=True)
    T = int(p["month"].nunique())
    T0 = _TSTAR - 1
    p["treatment"] = (p["ID"] == _TREATED_FIPS)
    p["after_treatment"] = (p["month"] >= _TSTAR)
    p["interaction"] = (p["treatment"] & p["after_treatment"]).astype(int)
    p["spillover"] = block_diag(*[W1] * T).dot(p["interaction"].values)
    p["ID"] = p["ID"].astype(int)
    return p, W1, common, T, T0


def _reference_canonical(panel, common, T, T0):
    """Authors' fitted SDID weights + canonical convention -> (att, tau_s, unit_weights)."""
    from benchmarks.reference.clone_spsydid import import_functions_ssdid

    fns = import_functions_ssdid()
    affected = set(panel[(panel["spillover"] > 0) & (~panel["treatment"])]["ID"].unique())
    data1 = panel[~panel["ID"].isin(affected)].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uw = fns.fit_unit_weights(data1, "UR2", "month", "ID", "treatment", "after_treatment")
        tw = fns.fit_time_weights(data1, "UR2", "month", "ID", "treatment", "after_treatment")

    Tpost = T - T0
    N_sp = len(affected)
    omega = {s: (1.0 if s == _TREATED_FIPS
                 else (1.0 / N_sp if s in affected else float(uw.get(s, 0.0))))
             for s in common}
    lam = np.empty(T)
    lam[:T0] = tw.values
    lam[T0:] = 1.0 / Tpost

    wide = panel.pivot(index="ID", columns="month", values="UR2").loc[common].values
    Dm = panel.pivot(index="ID", columns="month", values="interaction").loc[common].values
    WDm = panel.pivot(index="ID", columns="month", values="spillover").loc[common].values
    N = len(common)
    wv = np.outer(np.array([omega[s] for s in common]), lam).flatten()
    sw = np.sqrt(np.clip(wv, 0, None))
    X = np.zeros((N * T, 1 + (N - 1) + (T - 1) + 2))
    X[:, 0] = 1.0
    for i in range(1, N):
        X[i * T:(i + 1) * T, i] = 1.0
    base = 1 + (N - 1)
    for t in range(1, T):
        X[np.arange(N) * T + t, base + t - 1] = 1.0
    X[:, -2] = Dm.flatten()
    X[:, -1] = WDm.flatten()
    beta = lstsq(X * sw[:, None], wide.flatten() * sw, lapack_driver="gelsy")[0]
    return float(beta[-2]), float(beta[-1]), uw


def _mlsynth_fit(panel, W1, common):
    from mlsynth import SpSyDiD

    panel = panel.copy()
    panel["treat_indicator"] = panel["interaction"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SpSyDiD({
            "df": panel[["ID", "month", "UR2", "treat_indicator"]],
            "outcome": "UR2", "treat": "treat_indicator", "unitid": "ID", "time": "month",
            "spatial_matrix": W1, "unit_order": list(common),
            "row_standardize_spatial": False, "display_graphs": False,
        }).fit()
    # Everything below is read from the fitted result object only.
    att = float(res.att)
    tau_s = float(res.effects.additional_effects["aite"])
    donor_w = {int(k): float(v) for k, v in (res.weights.donor_weights or {}).items()}
    return att, tau_s, donor_w


def run() -> dict:
    panel, W1, common, T, T0 = _panel_and_W()
    att_ref, taus_ref, uw = _reference_canonical(panel, common, T, T0)
    att_ml, taus_ml, donor_w = _mlsynth_fit(panel, W1, common)

    uw_diff = float(np.max(np.abs(
        uw.values - np.array([donor_w.get(int(s), np.nan) for s in uw.index]))))
    return {
        "lawa_unit_weight_max_abs_diff": uw_diff,
        "lawa_att_abs_diff_vs_authors": float(abs(att_ml - att_ref)),
        "lawa_taus_abs_diff_vs_authors": float(abs(taus_ml - taus_ref)),
    }


def comparison() -> dict:
    """mlsynth SpSyDiD (.fit) vs the authors' code, on the LAWA CPS earnings panel."""
    panel, W1, common, T, T0 = _panel_and_W()
    att_ref, taus_ref, _ = _reference_canonical(panel, common, T, T0)
    att_ml, taus_ml, _ = _mlsynth_fit(panel, W1, common)
    return {
        "rows": [
            {"quantity": "direct ATT (LAWA on log weekly earnings)",
             "mlsynth": round(att_ml, 8), "reference": round(att_ref, 8)},
            {"quantity": "spillover coefficient tau_s",
             "mlsynth": round(taus_ml, 8), "reference": round(taus_ref, 8)},
        ],
        "mlsynth_call": {"estimator": "SpSyDiD (.fit)",
                         "config": {"treated": "Arizona", "outcome": "mean log weekly earnings",
                                    "spatial_matrix": f"<queen contiguity, {len(common)} states>"}},
        "reference": {"impl": "authors' functions_ssdid fit_unit_weights / fit_time_weights "
                              "under the canonical SDID convention (1/T_post post weights, "
                              "1/N_sp affected weights)",
                      "version": "serenini/spatial_SDID (cloned on demand, pinned)"},
    }


# The two implementations share the SDID QPs exactly (unit weights ~1e-9), so once
# the reference is put on mlsynth's canonical row-weight convention the direct ATT
# and the spillover coefficient agree to solver tolerance. The mlsynth side is
# taken entirely from SpSyDiD.fit() (res.att / res.effects.additional_effects /
# res.weights.donor_weights).
EXPECTED = {
    "lawa_unit_weight_max_abs_diff": (0.0, 1e-6),
    "lawa_att_abs_diff_vs_authors": (0.0, 1e-6),
    "lawa_taus_abs_diff_vs_authors": (0.0, 1e-6),
}
