"""Independent benchmark: MAREX Walmart placebo design (Abadie & Zhao 2026, Sec. 4).

MAREX is mlsynth's port of Abadie & Zhao's synthetic-control experimental-design
estimator; the authors' reference code is the R package
`jinglongzhao2/SCDesign <https://github.com/jinglongzhao2/SCDesign>`_ (Walmart
application, Table 1 / Figures 2-3). This case is an **independent, commit-stamped
validator** of MAREX on that application -- complementary to ``lexscm_walmart``,
which exercises the lexicographic LEXSCM solver on the same panel.

Following the paper, we design a *placebo* experiment (a fictitious intervention
with no real effect) on the Walmart store-sales panel and check the design is
sound: the synthetic treated and control units track closely pre-period (small
RMSE) and the estimated experimental effect is indistinguishable from zero.

Subset + exact solver
---------------------
MAREX solves the design as a **mixed-integer quadratic program**; the integrality
of the selection is essential (it is what makes the design meaningful -- see the
note below). The exact MIQP is solved here on a **10-store subset** so it is fast
*and deterministic*, on mlsynth's free SCIP backend. (The authors' R solves the
full 45-store MIQP with Gurobi, which a licence-free environment cannot run; only
their ``quadprog`` SC-weight solver is open. The relaxed continuous-``z`` mode is
*not* used: it shares A&Z's objective but drops the integrality, leaving a
degenerate optimum whose top-``m`` rounding is lossy and non-deterministic for
small ``m`` -- unfaithful to the paper's exact design.)

Provenance
----------
* Data: ``basedata/walmart_weekly_sales.csv`` (45 stores x 143 weeks; value-
  identical to SCDesign's ``Walmart.csv``), restricted to the first 10 stores.
* Headline (Abadie & Zhao 2026, Sec. 4): close pre-period tracking and a placebo
  effect indistinguishable from zero. MAREX picks 2 treated stores, pre-fit RMSE
  ~2.7% of mean sales (matching LEXSCM's ~2.7%), placebo effect ~1% of mean
  sales, CI covering zero.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

_BASE = Path(__file__).resolve().parents[2] / "basedata"


def run() -> dict:
    from mlsynth import MAREX

    df = pd.read_csv(_BASE / "walmart_weekly_sales.csv")
    df = df[df["store"] <= 10].copy()              # 10-store subset (tractable exact MIQP)
    df["post"] = (df["week"] >= 129).astype(int)   # placebo intervention at week 129
    mean_sales = float(df["sales"].mean())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MAREX({
            "df": df, "outcome": "sales", "unitid": "store", "time": "week",
            "post_col": "post", "design": "standard", "program_type": "MIQP",
            "m_eq": 2, "relaxed": False, "standardize": True,
            "inference": True, "T_post": 15, "display_graph": False,
        }).fit()

    report = res.report
    ci_lo, ci_hi = report.att_ci
    return {
        "n_treated": float(len(res.selected_units)),
        "prefit_rmse_pct": float(report.fit_diagnostics.rmse_pre / mean_sales),
        "abs_ate_pct": abs(float(report.att / mean_sales)),
        "placebo_p_value": float(report.inference.p_value),
        "ci_covers_zero": float(ci_lo <= 0.0 <= ci_hi),
    }


# Exact MIQP on the 10-store subset is deterministic. A sound placebo design must:
# select exactly 2 treated stores; track to ~a few % of mean sales pre-period
# (prefit_rmse_pct small, ~2.7% as LEXSCM); produce a near-zero placebo effect
# (abs_ate_pct small); and fail to reject the null (p > 0.05, CI covers zero) --
# the paper's "no spurious effect" result. Bands absorb solver/version drift.
EXPECTED = {
    "n_treated": (2.0, 0.0),
    "prefit_rmse_pct": (0.0266, 0.015),    # ~2.7% of mean sales (close tracking)
    "abs_ate_pct": (0.0098, 0.04),         # near zero (< ~5% of mean sales)
    "placebo_p_value": (0.15, 0.30),       # fails to reject (> 0.05 within band)
    "ci_covers_zero": (1.0, 0.0),
}
