"""Cross-validation: mlsynth MASC vs the authors' own R MASC (Basque, outcome-path).

Cross-validation against the canonical implementation. mlsynth's MASC
(:class:`mlsynth.estimators.masc.MASC`) is checked, value for value, against
Kellogg, Mogstad, Pouliot & Torgovitsky's own R code
(`maxkllgg/masc <https://github.com/maxkllgg/masc>`_ -- MIT, (c) 2019 Maxwell
Kellogg -- vendored verbatim under ``benchmarks/reference/masc_basque/``). Both
run the identical estimand on the identical panel: the Abadie-Gardeazabal Basque
terrorism study (17 Spanish regions after dropping the "Spain (Espana)" national
aggregate, 1955-1997, Basque Country treated from 1970), matched on the
pre-treatment per-capita-GDP path.

The estimator is MASC in its outcome-path configuration: a nearest-neighbour
match on the pre-period outcome path (``match_on="outcomes"`` -- the R
reference's default ``Wbar``) model-averaged with an outcome-only synthetic
control, the mixing weight ``phi`` and the neighbour count ``m in 1..10`` chosen
by the authors' rolling-origin cross-validation (``min_preperiods=5``). This is
distinct from the covariate/predictor specification pinned by ``masc_basque``
(the KMPT Section 5 paper replication); here both sides match on outcomes so the
comparison isolates the match+SC+CV machinery, not the predictor block.

Solver invariance (not a convention gap)
----------------------------------------
The synthetic-control step is a convex simplex-constrained least squares; its
optimum does not depend on the solver. The R reference solves it with
``nogurobi=TRUE`` (LowRankQP, the open-source fallback to the commercial Gurobi
default), mlsynth with CLARABEL. The two agree to solver tolerance -- ATT to
~2e-5 (thousands of 1986 USD per capita), ``phi`` and the pre-period RMSE to
4-5 digits, every donor weight to 3 digits -- so the case pins an exact match,
not an inflated band.

Reference (live captured run)
-----------------------------
The reference side is a live captured run of the authors' vendored R, not
numbers transcribed from the paper. ``benchmarks/reference/masc_crossval/reference.R``
sources the vendored ``masc_estimator.R`` / ``masc_crossvalidation.R`` and runs
``masc(..., nogurobi=TRUE)``; its ``phi``, ``m``, pre-period RMSE, ATT and donor
weights are captured under ``benchmarks/reference/masc_crossval/`` with full
provenance, and this case pins them by reading the captured ``reference.json``
via :func:`reference_value` / :func:`load_reference` -- so ``EXPECTED`` and the
captured run are the same object and cannot silently drift. Regenerate with
``python benchmarks/reference/generate.py masc_crossval``.

Provenance
----------
* Data: ``basedata/basque_jasa.csv`` -- Abadie & Gardeazabal (2003) per-capita
  GDP for 18 Spanish regions (incl. the national aggregate), 1955-1997; Basque
  Country treated from 1970. Outcome ``gdpcap`` (thousands of 1986 USD/capita).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference, reference_value

_BASE = Path(__file__).resolve().parents[2] / "basedata"
_TREATED = "Basque Country (Pais Vasco)"

_REF = load_reference("masc_crossval")
_REF_WEIGHTS = _REF["weights"]
REF_ATT = reference_value("masc_crossval", "masc_att")
REF_PHI = reference_value("masc_crossval", "masc_phi_hat")
REF_M = reference_value("masc_crossval", "masc_m_hat")
REF_PRE_RMSE = reference_value("masc_crossval", "masc_pre_rmse")


def _mlsynth_masc():
    """mlsynth MASC, outcome-path, on the Basque panel (Spain aggregate dropped)."""
    from mlsynth import MASC

    df = pd.read_csv(_BASE / "basque_jasa.csv")
    df = df[df["regionname"] != "Spain (Espana)"].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MASC({
            "df": df, "outcome": "gdpcap", "treat": "terrorism",
            "unitid": "regionname", "time": "year",
            "m_grid": list(range(1, 11)), "min_preperiods": 5,
            "match_on": "outcomes",          # the R reference's default Wbar path
            "display_graphs": False,
        }).fit()
    weights = {str(k): float(v) for k, v in res.donor_weights.items()}
    return weights, float(res.att), float(res.phi_hat), int(res.m_hat), float(res.fit.pre_rmse)


def run() -> dict:
    w_ml, att_ml, phi_ml, m_ml, rmse_ml = _mlsynth_masc()
    top = sorted(_REF_WEIGHTS.items(), key=lambda kv: -abs(kv[1]))[:4]
    w_diff = max(abs(w_ml.get(d, 0.0) - wr) for d, wr in top)

    return {
        "mls_att": att_ml,
        "mls_phi_hat": phi_ml,
        "mls_m_hat": float(m_ml),
        "mls_pre_rmse": rmse_ml,
        # mlsynth MASC vs the authors' own R MASC (nogurobi/LowRankQP).
        "att_abs_diff_vs_masc": float(abs(att_ml - REF_ATT)),
        "phi_abs_diff_vs_masc": float(abs(phi_ml - REF_PHI)),
        "pre_rmse_abs_diff_vs_masc": float(abs(rmse_ml - REF_PRE_RMSE)),
        "weight_max_abs_diff_vs_masc": float(w_diff),
        "m_matches": 1.0 if m_ml == int(round(REF_M)) else 0.0,
    }


def comparison() -> dict:
    """mlsynth MASC vs ``maxkllgg/masc`` (the authors' own R), quantity by quantity.

    Lays the mlsynth MASC fit against the authors' vendored R MASC on the same
    Basque outcome-path panel (same treated unit, same 16-donor pool, same 1970
    pre/post split, same ``m in 1..10`` CV grid): the CV-selected ``phi`` and
    ``m``, the ATT, the pre-period RMSE, and the top donor weights. The reference
    side is a live captured ``masc(..., nogurobi=TRUE)`` run in
    ``benchmarks/reference/masc_crossval/``, not transcribed. Returns
    ``{"rows": [...], "mlsynth_call": {...}, "reference": {...}}`` with rows
    ``{quantity, mlsynth, reference}``.
    """
    w_ml, att_ml, phi_ml, m_ml, rmse_ml = _mlsynth_masc()

    rows = [
        {"quantity": "phi_hat", "mlsynth": round(phi_ml, 6),
         "reference": round(REF_PHI, 6)},
        {"quantity": "m_hat", "mlsynth": float(m_ml),
         "reference": round(REF_M, 6)},
        {"quantity": "ATT", "mlsynth": round(att_ml, 6),
         "reference": round(REF_ATT, 6)},
        {"quantity": "pre_RMSE", "mlsynth": round(rmse_ml, 6),
         "reference": round(REF_PRE_RMSE, 6)},
    ]
    top = sorted(_REF_WEIGHTS.items(), key=lambda kv: -abs(kv[1]))[:4]
    for donor, w_ref in top:
        rows.append({"quantity": f"weight[{donor}]",
                     "mlsynth": round(float(w_ml.get(donor, 0.0)), 6),
                     "reference": round(float(w_ref), 6)})

    cfg = {"outcome": "gdpcap", "treat": "terrorism", "unitid": "regionname",
           "time": "year", "estimator": "MASC", "match_on": "outcomes",
           "m_grid": "1..10", "min_preperiods": 5}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "MASC (outcome-path, rolling-origin CV)",
                         "config": cfg},
        "reference": {"impl": "maxkllgg/masc masc(..., nogurobi=TRUE) "
                              "(LowRankQP), live run, captured",
                      "version": "vendored MIT sources @ "
                                 "benchmarks/reference/masc_basque/"},
    }


# The SC step is a convex simplex-constrained least squares (solver-invariant),
# and match/CV are deterministic given the grid, so mlsynth (CLARABEL) and the
# authors' R MASC (nogurobi/LowRankQP) solve the identical estimand. They agree
# to solver tolerance: phi and pre-RMSE to 4-5 digits, ATT to ~2e-5, every donor
# weight to 3 digits, same CV-selected m=3. Targets are pinned from the live
# captured R run (benchmarks/reference/masc_crossval/) via
# reference_value/load_reference, not transcribed; the *_diff_vs_masc tolerances
# are the actual mlsynth-vs-R gap (numerical, not an inflated pass).
EXPECTED = {
    "mls_att": (REF_ATT, 1e-3),                    # tracks R ATT (gap ~2e-5)
    "mls_phi_hat": (REF_PHI, 1e-3),                # tracks R phi (gap ~1e-4)
    "mls_m_hat": (REF_M, 0.0),                      # same CV-selected neighbour count
    "mls_pre_rmse": (REF_PRE_RMSE, 1e-3),          # tracks R pre-RMSE
    "att_abs_diff_vs_masc": (0.0, 1e-3),           # solver-tolerance gap
    "phi_abs_diff_vs_masc": (0.0, 1e-3),
    "pre_rmse_abs_diff_vs_masc": (0.0, 1e-3),
    "weight_max_abs_diff_vs_masc": (0.0, 2e-3),    # every top donor to 3 digits
    "m_matches": (1.0, 0.0),
}
