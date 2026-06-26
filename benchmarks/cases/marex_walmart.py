"""Cross-validation: MAREX Walmart placebo design vs SCDesign (Abadie & Zhao 2026).

Reproduces the authors' empirical illustration (Abadie & Zhao 2026, Section 4) on
the **full 45-store** Walmart weekly panel, matching on the pre-period sales and
the **four store-level covariates** (Temperature, Fuel_Price, CPI, Unemployment)
the reference data carries -- the R code's "few covariates" configuration. MAREX
solves the cardinality-constrained design (formulation (7), ``m_eq = 2``) as a
mixed-integer quadratic program on its open SCIP backend; the reference is a
**live captured run** of SCDesign's own design routine, read from
``benchmarks/reference/marex_walmart/`` (regenerate with
``python benchmarks/reference/generate.py marex_walmart``).

Validated against the R code without Gurobi
-------------------------------------------
SCDesign's published design is a Gurobi non-convex MIQP, which is licence-gated.
Its *constrained* (cardinality-``K``) routine -- ``Synthetic_Experiment_
Cardinality_Constraint`` -- is fully open: it enumerates every partition of size
``<= K``, solves the treated and control SC weights for each via ``quadprog::
solve.QP``, and keeps the min-loss partition. That is the exact design MAREX's
``m_eq`` solves, and it runs with no commercial solver. The reference run
(``reference.R``) executes it on the full 45-store panel with the same four
covariates, the same windows (fit weeks 1..100, blank 101..128, experimental
129..143), uniform population weights, and per-predictor standardisation -- every
modelling choice identical to the MAREX call below (``T0 = 128`` is the end of the
pre-experiment period; with ``blank_periods = 28`` the fit window is weeks 1..100).

Both solvers select the **same two treated stores** (15 and 31) with the **same
treated weights** to ~2e-4 (0.461/0.539), the same pre-period fit (~6e-4), the
same placebo effect (~8e-5 of mean sales), and confidence intervals that both
cover zero -- the "no spurious effect" result. The placebo permutation p-value
agrees to ~0.02 (Monte-Carlo permutation sampling differs slightly across the two
implementations). Every quantity is pinned directly to the live reference.

Provenance
----------
* Data: ``basedata/walmart_weekly_sales_covariates.csv`` (45 stores x 143 weeks,
  with the four macro covariates; sales value-identical to SCDesign's ``Walmart.csv``).
* Reference: live ``SCDesign`` run captured in ``benchmarks/reference/marex_walmart/``
  (Abadie & Zhao's own cardinality-constrained design + permutation test +
  conformal interval, on the open quadprog backend; no Gurobi).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from benchmarks.reference import load_reference, reference_value

_BASE = Path(__file__).resolve().parents[2] / "basedata"
_COVS = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
_CFG = {
    "outcome": "sales", "unitid": "store", "time": "week",
    "T0": 128, "blank_periods": 28, "T_post": 15, "m_eq": 2,  # fit 1..100, blank 101..128, exp 129..143
    "covariates": _COVS, "standardize": True, "design": "standard",
    "program_type": "MIQP", "relaxed": False, "inference": True,
    "display_graph": False,
}


def _fit():
    """MAREX exact-MIQP placebo design on the full 45-store Walmart panel + covariates.

    Returns ``(quantities_dict, mean_sales)``."""
    from mlsynth import MAREX

    df = pd.read_csv(_BASE / "walmart_weekly_sales_covariates.csv")
    mean_sales = float(df["sales"].mean())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MAREX({"df": df, **_CFG}).fit()

    report = res.report
    ci_lo, ci_hi = report.att_ci
    w = res.globres.treated_weights_agg
    treated_w = sorted(float(x) for x in w[w > 1e-6])     # two treated weights, ascending
    return {
        "n_treated": float(len(res.selected_units)),
        "treated_weight_lo": treated_w[0],
        "treated_weight_hi": treated_w[1],
        "prefit_rmse_pct": float(report.fit_diagnostics.rmse_pre / mean_sales),
        "abs_ate_pct": abs(float(report.att / mean_sales)),
        "placebo_p_value": float(report.inference.p_value),
        "ci_covers_zero": float(ci_lo <= 0.0 <= ci_hi),
    }, mean_sales


def run() -> dict:
    return _fit()[0]


def comparison() -> dict:
    """mlsynth ``MAREX`` (exact MIQP, full panel + covariates) vs SCDesign's
    cardinality-``K=2`` design, quantity by quantity.

    The reference side is a live ``SCDesign`` run captured in
    ``benchmarks/reference/marex_walmart/`` (Abadie & Zhao's own
    ``Synthetic_Experiment_Cardinality_Constraint`` on the open quadprog backend,
    plus their permutation test and conformal interval), read via
    :func:`reference_value` -- not transcribed constants."""
    got, _ = _fit()
    rw = sorted(load_reference("marex_walmart")["weights"].values())
    ref = {
        "n_treated": reference_value("marex_walmart", "n_treated"),
        "treated_weight_lo": rw[0],
        "treated_weight_hi": rw[1],
        "prefit_rmse_pct": reference_value("marex_walmart", "prefit_rmse_pct"),
        "abs_ate_pct": reference_value("marex_walmart", "abs_ate_pct"),
        "placebo_p_value": reference_value("marex_walmart", "placebo_p_value"),
        "ci_covers_zero": reference_value("marex_walmart", "ci_covers_zero"),
    }
    rows = [{"quantity": k, "mlsynth": round(got[k], 6), "reference": round(ref[k], 6)}
            for k in ("n_treated", "treated_weight_lo", "treated_weight_hi",
                      "prefit_rmse_pct", "abs_ate_pct", "placebo_p_value",
                      "ci_covers_zero")]
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "MAREX", "config": _CFG},
        "reference": {"impl": "jinglongzhao2/SCDesign (cardinality-K design, open quadprog, live run)",
                      "version": "live captured run (benchmarks/reference/marex_walmart/; "
                                 "Abadie & Zhao 2026, github.com/jinglongzhao2/SCDesign)"},
    }


# Deterministic exact MIQP on the full 45-store panel with four covariates, pinned
# directly to the live SCDesign run (open quadprog cardinality design, no Gurobi).
# Both select exactly 2 treated stores (15, 31) with treated weights agreeing to
# ~2e-4, a pre-period fit to ~6e-4, a placebo effect to ~8e-5 of mean sales, and
# confidence intervals that both cover zero (the paper's "no spurious effect"
# result). The placebo p-value agrees to ~0.02 (Monte-Carlo permutation sampling
# differs slightly across implementations). Tolerances are the genuine gaps plus a
# small margin for solver/version drift.
_rw = sorted(load_reference("marex_walmart")["weights"].values())
EXPECTED = {
    "n_treated": (reference_value("marex_walmart", "n_treated"), 0.0),
    "treated_weight_lo": (_rw[0], 0.005),
    "treated_weight_hi": (_rw[1], 0.005),
    "prefit_rmse_pct": (reference_value("marex_walmart", "prefit_rmse_pct"), 0.002),
    "abs_ate_pct": (reference_value("marex_walmart", "abs_ate_pct"), 0.001),
    "placebo_p_value": (reference_value("marex_walmart", "placebo_p_value"), 0.03),
    "ci_covers_zero": (reference_value("marex_walmart", "ci_covers_zero"), 0.0),
}
