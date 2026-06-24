"""Independent benchmark: MAREX Walmart placebo design (Abadie & Zhao 2026, Sec. 4).

MAREX is mlsynth's port of Abadie & Zhao's synthetic-control experimental-design
estimator; the authors' reference code is
`jinglongzhao2/SCDesign <https://github.com/jinglongzhao2/SCDesign>`_ (Walmart
application, Table 1 / Figures 2-3). This case cross-validates MAREX against a
**live captured run** of SCDesign's own design routine on the same 10-store
weekly panel: the reference numbers are read from the captured bundle in
``benchmarks/reference/marex_walmart/`` (regenerate with
``python benchmarks/reference/generate.py marex_walmart``; the fetch/install is
documented in ``benchmarks/R/install_scdesign.sh``), not transcribed Table-1
constants. Complementary to ``lexscm_walmart``, which exercises the lexicographic
LEXSCM solver on the same panel.

What SCDesign runs (live)
-------------------------
SCDesign's Gurobi non-convex MIQP is licence-gated, but its *constrained*
(cardinality-``K``) design -- ``Synthetic_Experiment_Cardinality_Constraint``,
which enumerates every partition of size ``<= K``, solves the treated and control
SC weights for each via the open ``quadprog`` path, and keeps the min-loss
partition -- is fully open and is the exact design MAREX's ``m_eq`` solves. With
``K = 2`` it selects exactly two treated stores. The reference bundle runs that
routine (plus SCDesign's permutation test and conformal interval) on identical
windows: weeks 1..143, fit 1..90, blank 91..128, experimental 129..143 (15).
SCDesign selects the same two stores (2, 9) MAREX does, and the design quantities
agree to ~1e-3 (placebo ATE to ~1e-5); tolerances below are the genuine gap.

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

from benchmarks.reference import reference_value

_BASE = Path(__file__).resolve().parents[2] / "basedata"

_CFG = {
    "outcome": "sales", "unitid": "store", "time": "week",
    "post_col": "post", "design": "standard", "program_type": "MIQP",
    "m_eq": 2, "relaxed": False, "standardize": True,
    "inference": True, "T_post": 15, "display_graph": False,
}


def _fit():
    """MAREX exact-MIQP placebo design on the 10-store weekly Walmart subset.

    Returns ``(quantities_dict, mean_sales)``."""
    from mlsynth import MAREX

    df = pd.read_csv(_BASE / "walmart_weekly_sales.csv")
    df = df[df["store"] <= 10].copy()              # 10-store subset (tractable exact MIQP)
    df["post"] = (df["week"] >= 129).astype(int)   # placebo intervention at week 129
    mean_sales = float(df["sales"].mean())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MAREX({"df": df, **_CFG}).fit()

    report = res.report
    ci_lo, ci_hi = report.att_ci
    return {
        "n_treated": float(len(res.selected_units)),
        "prefit_rmse_pct": float(report.fit_diagnostics.rmse_pre / mean_sales),
        "abs_ate_pct": abs(float(report.att / mean_sales)),
        "placebo_p_value": float(report.inference.p_value),
        "ci_covers_zero": float(ci_lo <= 0.0 <= ci_hi),
    }, mean_sales


def run() -> dict:
    return _fit()[0]


def comparison() -> dict:
    """mlsynth ``MAREX`` (exact MIQP, ``m_eq=2``) vs SCDesign's cardinality-``K=2``
    design, quantity by quantity.

    The reference side is a live ``SCDesign`` run captured in
    ``benchmarks/reference/marex_walmart/`` (Abadie & Zhao's own
    ``Synthetic_Experiment_Cardinality_Constraint`` plus their permutation test
    and conformal interval, on identical windows), read via
    :func:`reference_value` -- not transcribed Table-1 constants. Both pick the
    same two treated stores (2, 9)."""
    got, _ = _fit()
    rows = [{"quantity": k, "mlsynth": round(got[k], 6),
             "reference": round(reference_value("marex_walmart", k), 6)}
            for k in ("n_treated", "prefit_rmse_pct", "abs_ate_pct",
                      "placebo_p_value", "ci_covers_zero")]
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "MAREX", "config": _CFG},
        "reference": {"impl": "jinglongzhao2/SCDesign (cardinality-K design, live run, captured)",
                      "version": "live captured run (benchmarks/reference/marex_walmart/; "
                                 "Abadie & Zhao 2026, github.com/jinglongzhao2/SCDesign)"},
    }


# Exact MIQP on the 10-store subset is deterministic. A sound placebo design must:
# select exactly 2 treated stores; track to ~a few % of mean sales pre-period;
# produce a near-zero placebo effect; and fail to reject the null (CI covers
# zero) -- the paper's "no spurious effect" result. Targets are pinned from the
# live SCDesign run captured in benchmarks/reference/marex_walmart/ (not
# transcribed); MAREX selects the same two stores and agrees to ~1e-3 (placebo
# ATE to ~1e-5). Tolerances are the genuine mlsynth-vs-SCDesign gap with a small
# margin for solver/version drift.
_rv = lambda k: reference_value("marex_walmart", k)
EXPECTED = {
    "n_treated": (_rv("n_treated"), 0.0),               # exact: both select 2
    "prefit_rmse_pct": (_rv("prefit_rmse_pct"), 0.003), # gap ~7e-4
    "abs_ate_pct": (_rv("abs_ate_pct"), 0.001),         # gap ~2e-5
    "placebo_p_value": (_rv("placebo_p_value"), 0.03),  # gap ~7e-3
    "ci_covers_zero": (_rv("ci_covers_zero"), 0.0),     # exact: both cover zero
}
