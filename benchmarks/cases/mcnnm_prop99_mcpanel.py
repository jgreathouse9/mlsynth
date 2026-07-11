"""Cross-validation benchmark: MC-NNM vs the authors' ``MCPanel`` R (Prop 99).

Cross-validates mlsynth's ``MCNNM`` against the reference implementation of the
method's own authors -- the ``susanathey/MCPanel`` R package (Athey, Bayati,
Doudchenko, Imbens & Khosravi 2021) -- on the Abadie-Diamond-Hainmueller
Proposition 99 smoking panel (``basedata/smoking_data.csv``: 39 states x 31
years, 1970-2000, California treated from 1989).

The reference is a captured live run of ``mcnnm_cv`` at the package defaults,
pinned under ``benchmarks/reference/mcnnm_prop99_mcpanel/`` (R 4.3.3, ``MCPanel``
commit 6b2706f, ``set.seed(1)``, data checksum).

Why the tolerances are loose. The two implementations are the same algorithm --
soft-impute with unregularised two-way fixed effects -- and at a *matched*
singular-value threshold they agree on the fit to observed cells to RMSE ~3e-3
with identical singular spectra. But each selects its own regulariser by
cross-validation (mlsynth: K-fold partition on a demeaned-spectrum grid;
MCPanel: Bernoulli 80/20 folds on a ``2*sigma_max(P_Omega)/|Omega|``-scaled grid
plus an explicit lambda=0 rung), so they land on different penalties, and the
estimand -- the treated-unit counterfactual -- is the *extrapolated* block, which
is intrinsically threshold-sensitive in nuclear-norm completion. Under each
side's own CV the ATT agrees to ~0.15 packs and the California post-treatment
counterfactual path to under one pack. That is the honest agreement for this
estimator; a tight cell match is available only at a shared threshold (see
``docs/replications/mcnnm.rst``). The existing ``mcnnm_prop99`` case cross-checks
the same estimand against the Python ``causaltensor`` port.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference

_BASE = Path(__file__).resolve().parents[2] / "basedata"
_REF = load_reference("mcnnm_prop99_mcpanel")["values"]
_R_ATT = float(_REF["mcnnm_att"])
_R_CF = np.asarray(_REF["ca_counterfactual_post"], float)
_YEARS_POST = np.asarray(_REF["years_post"], int)


def _load_panel() -> pd.DataFrame:
    df = pd.read_csv(_BASE / "smoking_data.csv")
    df["treat"] = df["Proposition 99"].astype(int)
    return df[["state", "year", "cigsale", "treat"]]


def _mlsynth_fit():
    from mlsynth import MCNNM

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MCNNM({"df": _load_panel(), "outcome": "cigsale", "treat": "treat",
                     "unitid": "state", "time": "year",
                     "display_graphs": False}).fit()
    tp = np.asarray(res.time_series.time_periods).astype(int)
    cf = np.asarray(res.time_series.counterfactual_outcome, float)
    post = np.isin(tp, _YEARS_POST)
    return float(res.att), cf[post]


def run() -> dict:
    ml_att, ml_cf = _mlsynth_fit()
    return {
        "mcnnm_att_vs_mcpanel_R": abs(ml_att - _R_ATT),
        "ca_counterfactual_rmse_vs_mcpanel_R":
            float(np.sqrt(np.mean((ml_cf - _R_CF) ** 2))),
    }


def comparison() -> dict:
    """mlsynth MC-NNM vs the authors' ``MCPanel`` R: ATT and the California
    post-treatment counterfactual path, side by side."""
    ml_att, ml_cf = _mlsynth_fit()
    rows = [{"quantity": "ATT", "mlsynth": round(ml_att, 6),
             "reference": round(_R_ATT, 6)}]
    for yr, m, r in zip(_YEARS_POST, ml_cf, _R_CF):
        rows.append({"quantity": f"counterfactual[{int(yr)}]",
                     "mlsynth": round(float(m), 4), "reference": round(float(r), 4)})
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "MCNNM",
                         "config": {"outcome": "cigsale", "treat": "treat",
                                    "unitid": "state", "time": "year"}},
        "reference": {"impl": "susanathey/MCPanel R (mcnnm_cv, defaults)",
                      "version": "0.0 (commit 6b2706f), R 4.3.3, seed 1"},
    }


# Loose by construction (see module docstring): the ATT and counterfactual are
# the extrapolated block, and the two implementations select different CV
# penalties. Observed under each side's own default CV: ATT diff 0.15, path RMSE
# 0.47. Tolerances leave head-room for that CV-selection gap without admitting a
# genuine port regression (the engines agree to ~3e-3 at a shared threshold).
EXPECTED = {
    "mcnnm_att_vs_mcpanel_R": (0.0, 0.4),
    "ca_counterfactual_rmse_vs_mcpanel_R": (0.0, 0.9),
}
