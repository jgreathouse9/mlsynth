"""Path A benchmark: Mellace & Pasquini (2022) Proposition 99 mediation.

Reproduces the empirical result MEDSC is built to demonstrate -- the causal
decomposition of California's Proposition 99 tobacco-control effect into a
direct channel and an indirect channel that runs through the retail price of
cigarettes -- on the CDC / Orzechowski-Walker Tax Burden on Tobacco data
(``basedata/prop99_mediation.csv``, cigarette pack sales and the tax-inclusive
average cost per pack, 1970-2000).

The total effect is an ordinary synthetic control on the 38-state donor pool
(California's classic Abadie controls); the direct effect adds the seven high-
tax states back so the cross-world control can span California's high post-
treatment price, and matches that mediator path period by period (paper
Section 3.2). The paper's Table 1 reports (packs per capita):

  ====  ================  ================  ==================
  Year  Direct (paper)    MEDSC (mlsynth)   Indirect sign
  ====  ================  ================  ==================
  1995  -16.77            -16.8             ~ 0 -> negative
  2000  -17.28            -18.0             negative, growing
  ====  ================  ================  ==================

The novel cross-world direct effect reproduces the paper nearly cell-for-cell.
The indirect (price) channel reproduces its qualitative signature: essentially
zero at the 1989 intervention and growing negative thereafter. Its magnitude is
smaller than the paper's -14.31 because the outcome-path total (-26.8, tracking
a canonical Abadie SC) is smaller than the paper's predictor-tuned total
(-31.59); the decomposition mechanism -- not the tuned total -- is what this
benchmark pins.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_BASE = Path(__file__).resolve().parents[2] / "basedata"
_TREATED = "California"
_PROGRAM = ["Massachusetts", "Arizona", "Oregon", "Florida", "District of Columbia"]
_TAX = ["Alaska", "Hawaii", "Maryland", "Michigan", "New Jersey", "New York",
        "Washington"]


def run() -> dict:
    from mlsynth import MEDSC

    df = pd.read_csv(_BASE / "prop99_mediation.csv")
    allstates = sorted(df["state"].unique())
    direct_pool = [s for s in allstates if s not in [_TREATED] + _PROGRAM]
    total_pool = [s for s in direct_pool if s not in _TAX]
    df = df[df["state"].isin([_TREATED] + direct_pool)].copy()
    df["treated"] = ((df["state"] == _TREATED) & (df["year"] >= 1989)).astype(int)

    res = MEDSC({
        "df": df, "outcome": "cigsale", "mediator": "price", "treat": "treated",
        "unitid": "state", "time": "year",
        "total_donors": total_pool, "direct_donors": direct_pool,
        "inference": False, "display_graphs": False,
    }).fit()

    dec = res.decomposition
    years = list(res.inputs.time_labels)

    def direct(y):
        return float(dec.direct[years.index(y)])

    def indirect(y):
        return float(dec.indirect[years.index(y)])

    return {
        "n_total_donors": float(res.metadata["n_total_donors"]),
        "n_direct_donors": float(res.metadata["n_direct_donors"]),
        "pre_rmse_total": float(dec.pre_rmse_total),
        "direct_1995": direct(1995),
        "direct_2000": direct(2000),
        "indirect_1989": indirect(1989),
        "indirect_2000": indirect(2000),
    }


# Deterministic (outcome-path simplex QP on a fixed panel). Targets: the paper's
# reported cross-world direct effect (a near cell match) and the indirect
# channel's qualitative signature (~0 at intervention, negative and growing).
EXPECTED = {
    "n_total_donors": (38.0, 0.0),
    "n_direct_donors": (45.0, 0.0),
    "pre_rmse_total": (1.70, 0.5),         # well-fit pre-period
    "direct_1995": (-16.8, 1.0),           # Mellace-Pasquini Table 1: -16.77
    "direct_2000": (-18.0, 1.5),           # Mellace-Pasquini Table 1: -17.28
    "indirect_1989": (0.25, 0.75),         # channel opens at ~0
    "indirect_2000": (-8.8, 2.0),          # negative and grown by 2000
}
