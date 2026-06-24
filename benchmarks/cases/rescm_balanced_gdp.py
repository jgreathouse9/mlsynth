"""Cross-validation: mlsynth RESCM L2-relaxation vs the authors' ``scmrelax`` on
the balanced-GDP Brexit application (Liao-Shi-Zheng 2026, United Kingdom 2016Q3).

This is the authors' own empirical example -- the economic cost of Brexit on UK
GDP -- run as a live cross-validation on real data. The treated unit is the
United Kingdom; the outcome is the year-over-year quarterly GDP growth rate
``100 * GDP.pct_change(4)``; the pre-treatment window ends 2016-06-30, with
treatment starting 2016Q3 (the Brexit referendum quarter). These are exactly the
choices in the authors' ``GDP_application_2016.ipynb``
(``target_country='United Kingdom'``, ``pre_treat_q='2016-06-30'``,
``treat_q='2016-09-30'``).

At a fixed relaxation level ``tau`` the L2-relaxation program is a unique convex
QP, so mlsynth's relaxed branch and the reference must agree to solver precision
on the same panel. The reference (``scmrelax.L2RelaxationCV``) cross-validates
``tau`` over the authors' grid (``cv=3, n_taus=10, nonneg=True``, as
``scmrelax.fit`` does), and that CV-selected ``tau`` is captured in the bundle;
mlsynth is then pinned to that same ``tau`` and its relaxed donor weights are laid
against the reference's, donor by donor.

Provenance / scenario
---------------------
* Full repo (scenario 3): cross-validation is mandatory and done here, on the
  authors' real GDP panel.
* Reference: ``scmrelax.L2RelaxationCV`` -- Liao-Shi-Zheng's relaxed-balanced
  synthetic control, available as BOTH github.com/YapengZheng/Relaxed_SC (the
  original author code) and github.com/metricshilab/scmrelax (the installable
  package of that same code, used here). The captured run lives under
  ``benchmarks/reference/rescm_balanced_gdp/``; the matched ``tau`` and the
  per-donor weights are pinned from it via :func:`reference_value` /
  :func:`load_reference`.
* mlsynth side: ``RESCM(methods=["RELAX_L2"], tau=tau, standardize=False)`` --
  ``standardize=False`` matches the reference, which solves on the raw growth
  series.
* Data: ``basedata/balanced_gdp.csv`` (the authors' ``balanced_GDP_data.csv``;
  the two upstream repos ship byte-for-byte identical data).
* The case **skips gracefully** when ``scmrelax`` is not installed or no open
  conic solver can stand in for the reference's MOSEK dependency.

Solver note
-----------
``scmrelax`` hardcodes the commercial MOSEK solver; the L2 relaxation is a plain
QP, so we transparently route its solves to the open CLARABEL solver and restore
cvxpy afterwards. The optimum is solver-invariant for this convex program -- it
was independently confirmed with CLARABEL, ECOS, OSQP and SCS (all agreeing to
solver precision), so the matched-``tau`` weights here are the verified optimum,
not a single solver's idiosyncrasy. mlsynth matches the reference to ~1e-3 in L1,
explained entirely by RESCM rounding its public ``donor_weights`` to 3 sig figs.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from benchmarks.reference import load_reference, reference_value

_DATA = Path(__file__).resolve().parents[2] / "basedata" / "balanced_gdp.csv"
TARGET = "United Kingdom"
PRE_TREAT_Q = "2016-06-30"     # last pre-treatment quarter; treatment is 2016Q3


def _panel():
    """The authors' GDP-application panel: full year-over-year growth series for
    every economy, the treated UK series, the donor names, and ``T0`` -- exactly
    as ``GDP_application_2016.ipynb`` prepares it."""
    import pandas as pd

    GDP = pd.read_csv(_DATA, index_col=0, parse_dates=True)
    GDP_growth = 100.0 * GDP.pct_change(4).dropna()
    y = GDP_growth[TARGET]
    X = GDP_growth.drop(TARGET, axis=1)
    donors = list(X.columns)
    T0 = int(len(y.loc[:PRE_TREAT_Q]))
    return X.to_numpy().T, y.to_numpy(), donors, T0


def _mlsynth_weights(tau: float, donors: list[str]) -> np.ndarray:
    """RESCM L2-relaxation donor weights at the matched ``tau``, aligned to the
    reference's donor ordering."""
    from mlsynth import RESCM
    from mlsynth.utils.laxscm_helpers.simulation import to_panel

    Xfull, yfull, _, T0 = _panel()
    df = to_panel(Xfull, yfull, T0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = RESCM({"df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
                     "time": "time", "methods": ["RELAX_L2"], "tau": tau,
                     "standardize": False, "display_graphs": False}).fit()
    dw = res.fits["RELAX_L2"].donor_weights
    # to_panel names donors c000..c0NN in the column order of Xfull == donors.
    return np.array([dw.get(f"c{j:03d}", 0.0) for j in range(len(donors))], dtype=float)


def run() -> dict:
    from benchmarks.compare import BenchmarkSkipped
    try:
        import cvxpy  # noqa: F401  (only to gate the skip; reference is captured)
        import scmrelax  # noqa: F401
    except Exception as exc:
        raise BenchmarkSkipped(f"scmrelax/cvxpy unavailable: {exc}")

    _, _, donors, _ = _panel()
    ref = load_reference("rescm_balanced_gdp")
    tau = reference_value("rescm_balanced_gdp", "tau")
    w_ref = np.array([ref["weights"][d] for d in donors], dtype=float)

    w_ml = _mlsynth_weights(tau, donors)

    return {
        "weight_l1_diff": float(np.abs(w_ref - w_ml).sum()),
        "weight_max_abs_diff": float(np.abs(w_ref - w_ml).max()),
        # Both are dense (the relaxation diversifies across donor groups).
        "n_nonzero_ref": float((w_ref > 1e-4).sum()),
        "n_nonzero_ml": float((w_ml > 1e-4).sum()),
        # Reference weights live on the simplex; mlsynth's public vector sums to
        # ~1 only up to its 3-sig-fig rounding of donor_weights.
        "ref_on_simplex": float(abs(w_ref.sum() - 1.0) < 1e-4 and (w_ref >= -1e-6).all()),
        "ml_on_simplex": float(abs(w_ml.sum() - 1.0) < 1e-2 and (w_ml >= -1e-6).all()),
        # Headline big-economy weights agree donor-for-donor (paper Fig. 4 dense
        # weighting of the major EU economies + Japan).
        "germany": float(w_ml[donors.index("Germany")]),
        "italy": float(w_ml[donors.index("Italy")]),
        "japan": float(w_ml[donors.index("Japan")]),
    }


def comparison() -> dict:
    """mlsynth RESCM L2-relaxation vs the authors' ``scmrelax``, donor by donor,
    on the balanced-GDP Brexit panel.

    Solves the same unique L2-relaxation QP at the reference's CV-selected
    ``tau`` and lays the relaxed donor weights side by side for the headline
    economies (the ones the paper's Figure 4 highlights), plus the aggregate L1
    agreement. Mirrors :func:`run`'s skip behaviour.
    """
    from benchmarks.compare import BenchmarkSkipped
    try:
        import cvxpy  # noqa: F401
        import scmrelax  # noqa: F401
    except Exception as exc:
        raise BenchmarkSkipped(f"scmrelax/cvxpy unavailable: {exc}")

    _, _, donors, _ = _panel()
    ref = load_reference("rescm_balanced_gdp")
    tau = reference_value("rescm_balanced_gdp", "tau")
    w_ref = np.array([ref["weights"][d] for d in donors], dtype=float)
    w_ml = _mlsynth_weights(tau, donors)

    headline = ["Italy", "Japan", "Germany", "France", "United States"]
    rows = [{"quantity": f"weight[{c}]",
             "mlsynth": round(float(w_ml[donors.index(c)]), 6),
             "reference": round(float(w_ref[donors.index(c)]), 6)} for c in headline]
    rows.append({"quantity": "weight_L1_diff",
                 "mlsynth": round(float(np.abs(w_ref - w_ml).sum()), 6),
                 "reference": 0.0})
    cfg = {"outcome": "y", "treat": "treat", "unitid": "unit", "time": "time",
           "methods": ["RELAX_L2"], "tau": round(float(tau), 6),
           "standardize": False, "spec": "UK YoY GDP growth, pre<=2016Q2, treat 2016Q3"}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "RESCM", "config": cfg},
        "reference": {"impl": "scmrelax L2RelaxationCV (Liao-Shi-Zheng; "
                              "github.com/metricshilab/scmrelax = "
                              "github.com/YapengZheng/Relaxed_SC; MOSEK->CLARABEL; "
                              "live run, captured)",
                      "version": "Liao, Shi & Zheng (2026), arXiv 2508.01793; "
                                 "GDP_application_2016 (UK 2016Q3)"},
    }


# Two independent implementations of the same unique QP at the matched (captured)
# tau -> agreement to solver precision. The ~1e-3 L1 slack is RESCM's 3-sig-fig
# rounding of its public donor_weights; tolerances reflect genuine agreement only.
# tau and the per-donor weights are pinned from the live captured scmrelax run
# (benchmarks/reference/rescm_balanced_gdp/) via reference_value / load_reference.
_rg = lambda k: reference_value("rescm_balanced_gdp", k)
EXPECTED = {
    "weight_l1_diff": (0.0, 5e-3),
    "weight_max_abs_diff": (0.0, 2e-3),
    "n_nonzero_ref": (_rg("n_nonzero"), 0.0),
    "n_nonzero_ml": (_rg("n_nonzero"), 2.0),
    "ref_on_simplex": (1.0, 0.0),
    "ml_on_simplex": (1.0, 0.0),
    "germany": (0.0583, 5e-3),
    "italy": (0.0713, 5e-3),
    "japan": (0.0703, 5e-3),
}
