"""SPILLSYNTH (inclusive SCM) cross-validation vs Melnychuk's reference.

Cross-validates mlsynth's ``SPILLSYNTH(method="iscm", iscm_intercept=True)``
against the reference implementation shipped with Melnychuk (2024)
(``Melnychuk-Andrii/Spillover-SCM``, pinned commit ``282b621``) on German
reunification, outcome-only, with Austria the single affected neighbour.

The case pins two comparisons, because the reference has two answers.

Against the reference's algorithm with its quadratic program solved to the
simplex, mlsynth agrees:

  ===============  ===============  ===============
  quantity         mlsynth          converged
  ===============  ===============  ===============
  Austria in WG    ~0.464           ~0.454
  WG in Austria    ~0.353           ~0.355
  inclusive ATT    ~-1279           ~-1276
  ===============  ===============  ===============

Against what the authors' code prints, it does not, and the reason is the
solver. ``scm_weights`` hands the demeaned Gram to ``kernlab::ipop`` and reads
its primal without checking feasibility. On this panel -- GDP in levels, so the
Gram carries entries of order 1e9 -- ipop reports ``how = "converged"`` on a
solution whose weights sum to 0.9666 fitting West Germany and 1.1933 fitting
Austria. Its fit residual (23,225) sits below the constrained optimum (88,602),
which is what solving a relaxed problem looks like. Austria's weight comes out
0.176 instead of 0.454, and the inclusive estimate, which divides by
:math:`1 - w_A \\lambda_{WG}`, comes out -1651.52 instead of -1275.53.

The naive SCM ATT is the quantity that hides this: -1472.60 against -1474.45,
0.1% apart, while the weights underneath disagree by a factor of two and a half.
That is why this case pins the weights and the inclusive estimate, not the
headline gap.

Cross-validation (the data and the algorithm are the reference's). The captured
R run lives in ``benchmarks/reference/spillsynth_iscm_german/`` with its
provenance; the converged comparison runs live and needs no R.
"""
from __future__ import annotations

import os
import warnings

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata", "repgermany.dta")


def _fit():
    import pandas as pd

    from mlsynth import SPILLSYNTH

    d = pd.read_stata(os.path.abspath(_DATA))
    d = d[["country", "year", "gdp", "trade", "infrate",
           "industry", "schooling", "invest80"]].copy()
    d["treat"] = ((d["country"] == "West Germany") & (d["year"] >= 1990)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SPILLSYNTH({
            "df": d, "outcome": "gdp", "treat": "treat",
            "unitid": "country", "time": "year", "method": "iscm",
            "affected_units": ["Austria"], "iscm_intercept": True,
            "display_graphs": False,
        }).fit()


def run() -> dict:
    from benchmarks.reference.clone_iscm import (
        converged_reference_german, shipped_reference_german)

    conv = converged_reference_german()
    ship = shipped_reference_german()
    res = _fit()
    cw = res.iscm.cross_weights
    w_A = float(cw["Austria in West Germany"])
    l_WG = float(cw["West Germany in Austria"])
    return {
        "w_A": w_A,
        "l_WG": l_WG,
        "inclusive_att": float(res.att),
        "naive_att": float(res.att_scm),
        # mlsynth reproduces the reference's algorithm, solved to the simplex
        "w_A_vs_converged": abs(w_A - conv["w_A"]),
        "l_WG_vs_converged": abs(l_WG - conv["l_WG"]),
        "inclusive_vs_converged": abs(float(res.att) - conv["inclusive_att"]),
        "naive_vs_converged": abs(float(res.att_scm) - conv["naive_att"]),
        # and it does not reproduce what the shipped code prints, because that
        # code's weights are not on the simplex
        "shipped_weight_sum_west_germany": ship["nocov_weight_sum_west_germany"],
        "shipped_weight_sum_austria": ship["nocov_weight_sum_austria"],
        "shipped_w_A": ship["nocov_w_austria_in_west_germany"],
        "shipped_inclusive_att": ship["nocov_inclusive_att"],
        "inclusive_vs_shipped": abs(float(res.att) - ship["nocov_inclusive_att"]),
        "naive_vs_shipped": abs(float(res.att_scm) - ship["nocov_regular_att"]),
    }


def comparison() -> dict:
    """mlsynth's inclusive SCM against both reference points, quantity by
    quantity: the two cross-weights and the naive / inclusive ATT, as the
    authors' code prints them and as their algorithm reads once its quadratic
    program is solved to the simplex."""
    from benchmarks.reference.clone_iscm import (
        _COMMIT, converged_reference_german, shipped_reference_german)

    conv = converged_reference_german()
    ship = shipped_reference_german()
    res = _fit()
    cw = res.iscm.cross_weights
    w_A = float(cw["Austria in West Germany"])
    l_WG = float(cw["West Germany in Austria"])
    cfg = {"outcome": "gdp", "treat": "treat", "unitid": "country", "time": "year",
           "method": "iscm", "affected_units": ["Austria"], "iscm_intercept": True}
    rows = [
        {"quantity": "weight[Austria in West Germany]",
         "mlsynth": round(w_A, 6), "reference": round(conv["w_A"], 6)},
        {"quantity": "weight[West Germany in Austria]",
         "mlsynth": round(l_WG, 6), "reference": round(conv["l_WG"], 6)},
        {"quantity": "naive_ATT", "mlsynth": round(float(res.att_scm), 6),
         "reference": round(conv["naive_att"], 6)},
        {"quantity": "inclusive_ATT", "mlsynth": round(float(res.att), 6),
         "reference": round(conv["inclusive_att"], 6)},
        # the same four against what the authors' code prints, whose weights are
        # not on the simplex
        {"quantity": "weight[Austria in West Germany], reference as shipped",
         "mlsynth": round(w_A, 6),
         "reference": round(ship["nocov_w_austria_in_west_germany"], 6)},
        {"quantity": "weight[West Germany in Austria], reference as shipped",
         "mlsynth": round(l_WG, 6),
         "reference": round(ship["nocov_w_west_germany_in_austria"], 6)},
        {"quantity": "naive_ATT, reference as shipped",
         "mlsynth": round(float(res.att_scm), 6),
         "reference": round(ship["nocov_regular_att"], 6)},
        {"quantity": "inclusive_ATT, reference as shipped",
         "mlsynth": round(float(res.att), 6),
         "reference": round(ship["nocov_inclusive_att"], 6)},
        {"quantity": "sum of donor weights, West Germany fit (1 = on the simplex)",
         "mlsynth": 1.0,
         "reference": round(ship["nocov_weight_sum_west_germany"], 6)},
        {"quantity": "sum of donor weights, Austria fit (1 = on the simplex)",
         "mlsynth": 1.0,
         "reference": round(ship["nocov_weight_sum_austria"], 6)},
    ]
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "SPILLSYNTH", "config": cfg},
        "reference": {"impl": "Melnychuk-Andrii/Spillover-SCM inclusive SCM "
                              "(scm_weights / runInclusiveSCM). The first four rows "
                              "solve the same program to the simplex; the rows "
                              "marked 'reference as shipped' are the authors' ipop "
                              "output, whose weights sum to 0.9666 and 1.1933",
                      "version": f"git {_COMMIT[:7]}"},
    }


# Deterministic (closed-form demeaned simplex SCM + Cramer's rule; the shipped
# cells come from a committed capture). The ``*_vs_converged`` cells pin mlsynth
# to the reference algorithm solved to the simplex -- residual is the solver,
# FISTA here against OSQP there, ~2%. The ``*_vs_shipped`` cells pin the distance
# to what the authors' code prints, so the ipop infeasibility stays on the record
# instead of being absorbed into a tolerance.
EXPECTED = {
    "w_A": (0.464, 0.03),                    # Austria in synthetic West Germany
    "l_WG": (0.353, 0.03),                   # West Germany in synthetic Austria
    "inclusive_att": (-1278.8, 20.0),
    "naive_att": (-1482.0, 20.0),
    "w_A_vs_converged": (0.010, 0.03),       # reproduces the reference weight
    "l_WG_vs_converged": (0.003, 0.03),
    "inclusive_vs_converged": (3.3, 20.0),   # reproduces the reference estimate
    "naive_vs_converged": (7.6, 20.0),
    "shipped_weight_sum_west_germany": (0.9666, 0.001),   # not 1: ipop is infeasible
    "shipped_weight_sum_austria": (1.1933, 0.001),        # not 1
    "shipped_w_A": (0.1762, 0.001),
    "shipped_inclusive_att": (-1651.52, 1.0),
    "inclusive_vs_shipped": (372.7, 25.0),   # the cost of the infeasible solve
    "naive_vs_shipped": (9.4, 25.0),         # which the headline gap hides
}
