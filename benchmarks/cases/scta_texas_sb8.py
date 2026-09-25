"""SCTA vs augsynth on the authors' Texas SB8 panel: the knob, not the solver.

Path A plus cross-validation for Sun, Ben-Michael & Feller (2024), *"Temporal
Aggregation for the Synthetic Control Method"* (AEA P&P 114: 614-617), on their
own replication package: ``compileddata.csv`` vendored verbatim as
``basedata/texas_sb8_births.csv``, and ``time_aggregation.rmd`` transcribed into
``benchmarks/reference/scta_texas_sb8/reference.R``, run through the authors'
library ``augsynth`` 0.2.0 and captured.

The panel is monthly live births by state, 2016-2022, annualised by a factor of
12, with Texas treated from April 2022 (SB8). 51 states, 75 pre-treatment
months, six whole pre-treatment calendar years, nine post-treatment months.

The correspondence, which is the point of this case
---------------------------------------------------
mlsynth and augsynth spell the aggregation knob differently, and the difference
is a square. The paper writes the balancing objective as

    (a - B g)' V (a - B g),      V = diag(K nu, ..., K nu, 1, ..., 1),

and SCTA implements it by scaling the matching rows by ``sqrt(V)``, so a row's
weight in the objective is ``V``. augsynth reaches the same program by scaling
the matching *columns*, ``X_c <- X_c %*% V`` in ``fit_ridgeaug_formatted``, and
then solving with the weight matrix reset to the identity -- so a row's weight
in its objective is ``V squared``. Its ``year_wt`` and SCTA's ``nu`` are
therefore not the same number:

    K * nu  =  (K * year_wt)^2       ==>      nu = K * year_wt^2   (K = 12).

Under that mapping the two agree to about a tenth of a percent, and the residual
is a second, smaller convention difference: augsynth's ``fixedeff`` demeans by
``rowMeans`` over all 81 stacked pre-treatment columns (six aggregates and 75
months), while SCTA demeans by the 75 disaggregated months alone. The case pins
both, and pins the size of each.

What is not a difference
------------------------
The solver. augsynth 0.2.0 solves the simplex with OSQP at ``eps_abs =
eps_rel = 1e-8`` (``augsynth:::synth_qp``). Given the same design and the same
objective, its weights and mlsynth's active-set QP agree to the solver floor --
so the ``V``-squared reading is not an approximation that happens to fit, it is
what augsynth optimises exactly. The case asserts this in both directions: at
augsynth's own weights the ``V``-squared objective is at its minimum, and the
``V``-linear objective is not.

Provenance: the case reads augsynth's estimates, per-nu RMSPEs, donor weights
and objectives from the captured bundle; ``reference.R`` regenerates it and
needs R with ``augsynth``, ``dplyr``. No R runs here.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                     "texas_sb8_births.csv")
_REF = load_reference("scta_texas_sb8")
_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]
_TREATED = "Texas"
_TREAT_TIME = 2022.25                       # April 2022, the .rmd's cutoff
_K = 12
_GRID = (0.0, 0.5, 1.0, 2.0, 30.0)          # augsynth year_wt values captured
_TAG = {0.0: "0", 0.5: "0p5", 1.0: "1", 2.0: "2", 30.0: "30"}


def _long():
    """The authors' ``combined_dat``: annualised monthly births, Texas treated."""
    d = pd.read_csv(os.path.abspath(_DATA))
    d["m"] = d.month.map({m: i + 1 for i, m in enumerate(_MONTHS)})
    d = d.sort_values(["state", "year", "m"]).reset_index(drop=True)
    d["year_mth"] = d.year + (d.m - 1) / _K
    d["annualized"] = d.all_births * float(_K)
    d["treat"] = ((d.state == _TREATED) & (d.year_mth >= _TREAT_TIME)).astype(int)
    return d


def _augsynth_design(d):
    """augsynth's stacked matching design, rebuilt from the panel.

    Six yearly aggregate rows (the state-year means of the annualised series for
    2016-2021) on top of the 75 disaggregated pre-treatment months, demeaned the
    way ``augsynth:::demean_data`` does it: by the row mean over all 81 stacked
    columns. Returns the treated vector, the donor matrix, and the post-treatment
    outcomes on the same demeaning.
    """
    wide = d.pivot(index="year_mth", columns="state", values="annualized").sort_index()
    T0 = int((wide.index < _TREAT_TIME).sum())
    states = list(wide.columns)
    pre = wide.iloc[:T0].to_numpy(float)                 # (75, 51)
    post = wide.iloc[T0:].to_numpy(float)                # (9, 51)
    n_blocks = T0 // _K
    agg = pre[:n_blocks * _K].reshape(n_blocks, _K, len(states)).mean(axis=1)
    X = np.vstack([agg[::-1], pre]).T                    # (51, 81); -2021 first
    means = X.mean(axis=1)                               # augsynth's fixed effect
    Xd, Yd = X - means[:, None], post.T - means[:, None]
    ti = states.index(_TREATED)
    donors = [s for s in states if s != _TREATED]
    keep = [i for i in range(len(states)) if i != ti]
    return {"a": Xd[ti], "B": Xd[keep].T, "y1": Yd[ti], "Y0": Yd[keep].T,
            "donors": donors, "T0": T0, "n_blocks": n_blocks,
            "n_states": len(states)}


def _v(year_wt, n_blocks, T0, square):
    """The objective's row weights under each reading of augsynth's ``V``."""
    top = (_K * year_wt) ** 2 if square else _K * year_wt
    return np.concatenate([np.full(n_blocks, float(top)), np.ones(T0)])


def _solve(a, B, v):
    """mlsynth's exact active-set simplex QP on a ``v``-weighted design."""
    from mlsynth.utils.bilevel.ridge_augment import simplex_qp

    root = np.sqrt(v)
    return np.asarray(simplex_qp(root[:, None] * B, root * a), dtype=float)


def _ref_weights(year_wt, donors):
    """augsynth's captured donor weights at ``year_wt``, aligned to ``donors``."""
    tag = _TAG[year_wt]
    w = _REF["weights"]
    return np.array([w.get(f"{tag}@{d}", 0.0) for d in donors], dtype=float)


def _scta(d, nu):
    from mlsynth import SCTA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SCTA({"df": d, "outcome": "annualized", "treat": "treat",
                     "unitid": "state", "time": "year_mth", "block_length": _K,
                     "nu": nu, "display_graphs": False}).fit()


def _paired():
    """Every quantity both sides produce, at each captured ``year_wt``."""
    d = _long()
    g = _augsynth_design(d)
    a, B, y1, Y0 = g["a"], g["B"], g["y1"], g["Y0"]
    rows = []
    for wt in _GRID:
        tag = _TAG[wt]
        w_ref = _ref_weights(wt, g["donors"])
        out = {"year_wt": wt, "nu": _K * wt ** 2,
               "att_ref": _REF["values"][f"est_{tag}"],
               "rmse_mth_ref": _REF["values"][f"rmse_mth_{tag}"],
               "rmse_yr_ref": _REF["values"][f"rmse_yr_{tag}"],
               "w_ref": w_ref}
        # mlsynth's solver on augsynth's own design, under each reading of V
        for name, sq in (("v", False), ("v2", True)):
            v = _v(wt, g["n_blocks"], g["T0"], sq)
            w_ml = _solve(a, B, v)
            obj = lambda w, v=v: float(np.sum(v * (a - B @ w) ** 2))
            out[f"l1_{name}"] = float(np.abs(w_ref - w_ml).sum())
            out[f"excess_{name}"] = ((obj(w_ref) - obj(w_ml)) / obj(w_ml)
                                     if obj(w_ml) > 0 else 0.0)
            out[f"att_{name}"] = float(np.mean(y1 - Y0 @ w_ml))
        # and SCTA end to end, at the nu the mapping implies
        r = _scta(d, out["nu"])
        out["att_scta"] = float(r.effects.att)
        rows.append(out)
    return g, rows


def run() -> dict:
    g, rows = _paired()
    by_wt = {r["year_wt"]: r for r in rows}
    pos = [r for r in rows if r["year_wt"] > 0]      # V and V^2 coincide at 0

    return {
        # the design, against augsynth's own counts
        "n_states": float(g["n_states"] - _REF["values"]["n_states"]),
        "pre_months": float(g["T0"] - _REF["values"]["n_pre_months"]),
        "n_blocks": float(g["n_blocks"] - _REF["values"]["n_pre_yrs"]),
        # augsynth's weights ARE the exact optimum of the V-squared program:
        # same design, same objective, two solvers (OSQP vs active set).
        "max_l1_vs_augsynth_v2": max(r["l1_v2"] for r in rows),
        "max_excess_v2": max(abs(r["excess_v2"]) for r in rows),
        # ... and are NOT the optimum of the V-linear one, which is what the
        # knob mismatch looks like if the square is missed.
        "min_l1_vs_augsynth_v_at_positive_wt": min(r["l1_v"] for r in pos),
        "max_excess_v": max(r["excess_v"] for r in pos),
        # the mapping nu = K * year_wt^2, end to end through SCTA.fit()
        "max_rel_att_gap_scta_vs_augsynth": max(
            abs(r["att_scta"] - r["att_ref"]) / abs(r["att_ref"]) for r in rows),
        # the residual is the demeaning basis: on augsynth's basis the same
        # program gives augsynth's answer to the solver floor.
        "max_rel_att_gap_same_basis": max(
            abs(r["att_v2"] - r["att_ref"]) / abs(r["att_ref"]) for r in rows),
        "demean_basis_gap_tx": float(_REF["values"]["unit_mean_stacked_tx"]
                                     - _REF["values"]["unit_mean_monthly_tx"]),
        # Path A: the paper's headline cells, from augsynth, reproduced here
        "att_monthly_alone": by_wt[0.0]["att_ref"],
        "att_yearly_plus_monthly": by_wt[1.0]["att_ref"],
        "att_yearly_alone": by_wt[30.0]["att_ref"],
        # the frontier of Figure 1: monthly imbalance rises in the weight on the
        # yearly aggregates, yearly imbalance falls
        "rmse_mth_rises_in_wt": float(all(
            np.diff([r["rmse_mth_ref"] for r in rows]) > 0)),
        "rmse_yr_falls_in_wt": float(all(
            np.diff([r["rmse_yr_ref"] for r in rows]) < 0)),
    }


def comparison() -> dict:
    """SCTA against augsynth cell by cell, at the mapped knob."""
    _, rows = _paired()
    out = []
    for r in rows:
        out.append({"quantity": f"ATT (year_wt={r['year_wt']:g}, nu={r['nu']:g})",
                    "mlsynth": round(r["att_scta"], 3),
                    "reference": round(r["att_ref"], 3)})
    for r in rows:
        out.append({"quantity": f"donor-weight L1 vs augsynth "
                                f"(year_wt={r['year_wt']:g}, V^2 program)",
                    "mlsynth": round(r["l1_v2"], 8), "reference": 0.0})
    return {
        "rows": out,
        "mlsynth_call": {
            "estimator": "SCTA",
            "config": {"block_length": _K,
                       "nu": [r["nu"] for r in rows],
                       "note": "nu = K * year_wt^2 maps augsynth's knob to the "
                               "paper's V-linear one",
                       "panel": "Texas SB8 live births, TX treated 2022-04"}},
        "reference": {
            "impl": "augsynth 0.2.0 (the authors' R package) on their "
                    "time_aggregation.rmd construction, captured",
            "version": "R 4.3.3; augsynth 0.2.0; osqp 1.0.0"},
    }


# Tolerances.
#
# The three design cells are differences against augsynth's own counts and are
# exact or the panel is not the authors' panel.
#
# max_l1_vs_augsynth_v2 / max_excess_v2 are the cross-validation: OSQP at
# eps_abs = eps_rel = 1e-8 against the active-set QP on an identical design and
# objective. Observed 4.3e-7 in L1 over 50 donors and 6.7e-8 in absolute
# relative objective, the worst of it at year_wt = 30 where V squared reaches
# 360^2 and the design is worst-scaled. The excess is signed and lands on either
# side of zero across the grid, which is what two solvers meeting at the same
# optimum looks like; the band is an order of magnitude above OSQP's own stated
# tolerance and no tighter.
#
# min_l1_vs_augsynth_v / max_excess_v are the converse, and are asserted to be
# LARGE: if augsynth were solving the V-linear program these would be ~0 too,
# and the mapping this case pins would be wrong. The bands are wide because the
# claim is only that they are far from zero -- observed L1 >= 0.227 and an
# objective excess of 23.7 percent at year_wt = 2.
#
# max_rel_att_gap_scta_vs_augsynth is SCTA end to end vs augsynth: 0.11 percent
# at worst across the grid, which is the demeaning-basis difference and nothing
# else -- max_rel_att_gap_same_basis removes it and lands at 1e-7. Both are
# banded a little above the observed value so a solver or BLAS change does not
# red the case.
#
# The Path A cells are augsynth's own estimates and carry a band of 1.0
# annualised birth on values of 20-25 thousand; the frontier monotonicities are
# booleans.
EXPECTED = {
    "n_states": (0.0, 0.0),
    "pre_months": (0.0, 0.0),
    "n_blocks": (0.0, 0.0),
    "max_l1_vs_augsynth_v2": (4.5e-7, 1e-5),
    "max_excess_v2": (6.7e-8, 5e-7),
    "min_l1_vs_augsynth_v_at_positive_wt": (0.227, 0.15),
    "max_excess_v": (0.237, 0.05),
    "max_rel_att_gap_scta_vs_augsynth": (1.116e-3, 5e-4),
    "max_rel_att_gap_same_basis": (1.2e-7, 1e-5),
    "demean_basis_gap_tx": (28.705, 1.0),
    "att_monthly_alone": (20895.544, 1.0),
    "att_yearly_plus_monthly": (21735.248, 1.0),
    "att_yearly_alone": (24653.966, 1.0),
    "rmse_mth_rises_in_wt": (1.0, 0.0),
    "rmse_yr_falls_in_wt": (1.0, 0.0),
}
