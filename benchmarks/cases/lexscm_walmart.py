"""Path A benchmark: LEXSCM Walmart placebo design (Abadie & Zhao 2026, Sec. 4).

Reproduces the empirical illustration of the synthetic-experimental-design
framework (Abadie & Zhao 2026; the lexicographic solve is Vives-i-Bastida 2022)
on the Walmart store-sales panel: weekly sales for **45 stores over 143 weeks**.
Following the paper we design a *placebo* experiment with a fictitious
intervention at week 129 (``T0 = 128`` pre-experiment weeks, the first ~100 the
fitting window, the rest blank, leaving 15 experimental weeks) and ``m = 2``
treated stores.

Because the intervention is a placebo (no real effect), a correct design must
produce synthetic treated and control units that track closely (small pre-fit
RMSE) and an estimated experimental effect near zero whose permutation test
fails to reject the null. LEXSCM delivers exactly that, the same "no spurious
effect" result MAREX reports on this panel.

Provenance
----------
* Data: ``basedata/walmart_weekly_sales.csv`` (45 stores x 143 weeks).
* Headline: Abadie & Zhao (2026) Sec. 4 -- close pre-period tracking and a
  placebo effect indistinguishable from zero (their permutation p ~ 0.93).
  LEXSCM picks stores {1, 25}; pre-fit RMSE ~2.7% of mean sales, placebo
  effect ~0.9% of mean sales, p ~ 0.63, CI covers zero.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

_BASE = Path(__file__).resolve().parents[2] / "basedata"


def run() -> dict:
    from mlsynth import LEXSCM

    df = pd.read_csv(_BASE / "walmart_weekly_sales.csv")
    df["candidate"] = 1                       # every store eligible for treatment
    df["post"] = (df["week"] >= 129).astype(int)
    mean_sales = float(df["sales"].mean())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = LEXSCM({
            "df": df, "outcome": "sales", "unitid": "store", "time": "week",
            "candidate_col": "candidate", "m": 2, "post_col": "post",
            "frac_E": 100 / 128, "top_K": 5, "n_sims": 200,
            "n_post_grid": [5, 10, 15], "mde_horizon": "late", "alpha": 0.05,
            "verbose": False,
        }).fit()

    # The realized effect lives on the standardized report (the contract view).
    report = res.report
    ci_lo, ci_hi = report.att_ci
    return {
        "n_treated": float(len(res.selected_units)),
        "prefit_rmse_pct": float(report.fit_diagnostics.rmse_pre / mean_sales),
        "abs_ate_pct": abs(float(report.att / mean_sales)),
        "placebo_p_value": float(report.inference.p_value),
        # 1.0 iff the experimental CI covers zero (no spurious effect).
        "ci_covers_zero": float(ci_lo <= 0.0 <= ci_hi),
    }


# Deterministic (enumeration design + seeded permutation). The placebo design
# must: select exactly 2 stores; track to within a few % of mean sales pre-period
# (prefit_rmse_pct small); produce a near-zero placebo effect (abs_ate_pct small);
# and FAIL to reject the null (p well above 0.05, CI covering zero) -- the
# paper's "no spurious effect" result. Bands absorb solver/version drift.
EXPECTED = {
    "n_treated": (2.0, 0.0),
    "prefit_rmse_pct": (0.0275, 0.015),     # ~2.7% of mean sales (close tracking)
    "abs_ate_pct": (0.009, 0.04),           # near zero (<~5% of mean sales)
    "placebo_p_value": (0.63, 0.30),        # fails to reject (>> 0.05)
    "ci_covers_zero": (1.0, 0.0),
}
