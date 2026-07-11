"""Cross-validation benchmark: SDID vs the authors' ``synthdid`` R (Prop 99).

Cross-validates mlsynth's ``SDID`` against the reference implementation of the
method's own authors -- the ``synth-inference/synthdid`` R package (Arkhangelsky,
Athey, Hirshberg, Imbens & Wager 2021) -- on the Abadie-Diamond-Hainmueller
Proposition 99 smoking panel (``basedata/smoking_data.csv``: 39 states x 31
years, 1970-2000, California treated from 1989).

The reference is a captured live run of ``synthdid_estimate`` on the identical
outcome matrix, pinned under ``benchmarks/reference/sdid_prop99_synthdid/`` with
its provenance (R 4.3.3, ``synthdid`` commit 70c1ce3, data checksum). mlsynth
reproduces the authors' Synthetic DiD point estimate to ~2e-3 packs -- the
residual is the unit-weight ridge (zeta) optimiser, not a methodological
difference. This complements the existing ``sdid_prop99`` case, which
cross-validates the same estimand against the Python ``causaltensor`` port.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from benchmarks.reference import load_reference

_BASE = Path(__file__).resolve().parents[2] / "basedata"
_REF = load_reference("sdid_prop99_synthdid")["values"]


def _load_panel() -> pd.DataFrame:
    df = pd.read_csv(_BASE / "smoking_data.csv")
    df["treat"] = df["Proposition 99"].astype(int)
    return df[["state", "year", "cigsale", "treat"]]


def _mlsynth_att() -> float:
    from mlsynth import SDID

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SDID({"df": _load_panel(), "outcome": "cigsale", "treat": "treat",
                    "unitid": "state", "time": "year",
                    "display_graphs": False}).fit()
    return float(res.att)


def run() -> dict:
    ml_att = _mlsynth_att()
    return {
        "sdid_att": ml_att,
        "sdid_att_vs_synthdid_R": abs(ml_att - _REF["sdid_att"]),
    }


def comparison() -> dict:
    """mlsynth SDID vs the authors' ``synthdid`` R, the Prop 99 ATT side by side.

    Also lists the DiD and pure-SC estimates the same R package produces on the
    identical matrix, for context (mlsynth's SDID targets the SDID column).
    """
    ml_att = _mlsynth_att()
    rows = [
        {"quantity": "SDID ATT", "mlsynth": round(ml_att, 6),
         "reference": round(float(_REF["sdid_att"]), 6)},
        {"quantity": "DID ATT (context)", "mlsynth": None,
         "reference": round(float(_REF["did_att"]), 6)},
        {"quantity": "SC ATT (context)", "mlsynth": None,
         "reference": round(float(_REF["sc_att"]), 6)},
    ]
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "SDID",
                         "config": {"outcome": "cigsale", "treat": "treat",
                                    "unitid": "state", "time": "year"}},
        "reference": {"impl": "synth-inference/synthdid R (synthdid_estimate)",
                      "version": "0.0.9 (commit 70c1ce3), R 4.3.3"},
    }


# mlsynth's SDID must land on the published -15.6 headline and match the authors'
# synthdid R tightly. 0.05 brackets display rounding of the AER headline; 0.02 is
# the unit-weight ridge (zeta) optimiser agreement (same estimand, different
# solver) -- observed 1.6e-3.
EXPECTED = {
    "sdid_att": (-15.604, 0.05),
    "sdid_att_vs_synthdid_R": (0.0, 0.02),
}
