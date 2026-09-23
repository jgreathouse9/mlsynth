"""NSC cross-validation: California Proposition 99 vs Tian's reference code.

Cross-validation (scenario: authors' reference implementation + canonical data).
Tian (2023), *"The Synthetic Control Method with Nonlinear Outcomes,"* revisits
Abadie-Diamond-Hainmueller's Proposition 99 study (Section 5.1, Table 2) with
the nonlinear synthetic-control estimator. The author's R implementation
(``benchmarks/R/nsc_tian2023_reference.R``) and the ADH smoking panel are public,
so we validate mlsynth's NSC against them directly.

The author's cross-validation of the penalty ``(a, b)`` is *stochastic* (it draws
a random held-in donor each fold, hence ``set.seed(123)`` in the application
script), so it does not port to Python. But the application is deterministic
given the *selected* penalty ``a* = 0.3, b* = 0.7`` reported in Table 2, so we
fix those and match:

  * the per-donor NSC weights against the author's code (all 38 donors,
    including the negative weights the method permits), and
  * the estimated effect path against the author's code -- the nonlinear SC
    reduces per-capita sales by ~9.5 packs in 1990, ~24.5 in 1995 and ~28.7 in
    2000, an average post-period effect near -20.

mlsynth's NSC is a faithful port of the reference QP (eigenvalue-scaled penalty,
the ``rbind(Z, -Z)`` negativity trick, distance-weighted L1), so the weights
match to a correlation of 1.000 and the effect path to ~1e-6 packs. What remains
is the author's 3-decimal rounding of the returned weights, which bounds any
single weight's deviation at 5e-4.

Until #463 this case carried tolerances an order of magnitude wider (ATT within
1.7 packs, correlation 0.989) and attributed them to "the standardisation
convention and the author's 3-decimal weight rounding". Both were wrong: the
standardisation already agreed, and the rounding accounted for 2 percent of the
observed gap. The cause was that mlsynth indexed its penalties on the spectrum
of ``Z_0 Z_0'`` where the reference indexes the stacked ``Z* Z*'``, so the
paper's ``(a*, b*)`` bought half the penalty they name. The tolerances here are
now what a faithful port reaches, so they fail if that regresses.

The reference side is a live captured run of Tian's own ``NSC()`` (vendored at
``benchmarks/R/nsc_tian2023_reference.R``, executed at ``a = 0.3, b = 0.7``),
captured under ``benchmarks/reference/nsc_prop99/`` with its provenance pinned
(R version, package versions, data checksum) -- not numbers transcribed from the
paper's Table 2. With the penalty fixed the estimator is deterministic, and the
captured run reproduces Table 2's weights and effect path to the digit.

Provenance: Tian (2023), arXiv:2306.01967v1, Section 5.1, Table 2; the author's
NSC.R / App_Cal.R and the ADH ``smoking_data.csv`` panel.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference, reference_value

_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "smoking_data.csv")

# Tian's NSC "NSC Weight" column (a* = 0.3, b* = 0.7), all 38 donors -- read from
# the live captured run in benchmarks/reference/nsc_prop99/ (which reproduces
# Table 2 exactly), not transcribed.
_REF_WEIGHTS = load_reference("nsc_prop99")["weights"]
# The author's NSC effect path (per-capita packs) at 1990/1995/2000, from the
# same captured run.
_REF_ITE = {yr: reference_value("nsc_prop99", f"ite_{yr}") for yr in (1990, 1995, 2000)}


def run() -> dict:
    from mlsynth import NSC

    d = pd.read_csv(os.path.abspath(_DATA))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = NSC({
            "df": d, "outcome": "cigsale", "treat": "Proposition 99",
            "unitid": "state", "time": "year",
            "a": 0.3, "b": 0.7,                  # Table-2 selected penalty
            "run_inference": True, "display_graphs": False,
        }).fit()

    w = {str(k): float(v) for k, v in res.design.donor_weights.items()}
    mv = np.array([w.get(s, 0.0) for s in _REF_WEIGHTS])
    rv = np.array(list(_REF_WEIGHTS.values()))

    years = sorted(d["year"].unique())
    gap = np.asarray(res.inference_detail.gap)
    ite = {yr: float(gap[years.index(yr)]) for yr in _REF_ITE}

    return {
        "n_donors": float(len(_REF_WEIGHTS)),
        "weight_max_abs_dev": float(np.max(np.abs(mv - rv))),
        "weight_mean_abs_dev": float(np.mean(np.abs(mv - rv))),
        "weight_correlation": float(np.corrcoef(mv, rv)[0, 1]),
        "att_mean_post": float(res.att),
        "ite_1990": ite[1990],
        "ite_1995": ite[1995],
        "ite_2000": ite[2000],
    }


def comparison() -> dict:
    """mlsynth NSC vs Tian's NSC.R reference, quantity by quantity.

    Pairs the mlsynth NSC fit against Tian's own ``NSC()``: every signed donor
    weight (those non-zero on either side) and the effect path at 1990, 1995 and
    2000. The reference side is a live ``NSC.R`` run captured in
    ``benchmarks/reference/nsc_prop99/`` (the penalty CV is stochastic and does
    not port, so the selected ``a* = 0.3, b* = 0.7`` are fixed, which makes the
    run deterministic). Returns ``{"rows": [...], "mlsynth_call": {...},
    "reference": {...}}`` with rows ``{quantity, mlsynth, reference}``.
    """
    from mlsynth import NSC

    d = pd.read_csv(os.path.abspath(_DATA))
    cfg = {"outcome": "cigsale", "treat": "Proposition 99", "unitid": "state",
           "time": "year", "a": 0.3, "b": 0.7, "run_inference": True}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = NSC({**cfg, "df": d, "display_graphs": False}).fit()

    w_ml = {str(k): float(v) for k, v in res.design.donor_weights.items()}
    years = sorted(d["year"].unique())
    gap = np.asarray(res.inference_detail.gap)

    keep = sorted((s for s in _REF_WEIGHTS
                   if abs(w_ml.get(s, 0.0)) > 1e-3 or abs(_REF_WEIGHTS[s]) > 1e-3),
                  key=lambda s: -max(abs(w_ml.get(s, 0.0)), abs(_REF_WEIGHTS[s])))
    rows = [{"quantity": f"weight[{s}]", "mlsynth": round(w_ml.get(s, 0.0), 6),
             "reference": round(_REF_WEIGHTS[s], 6)} for s in keep]
    for yr, ref in _REF_ITE.items():
        rows.append({"quantity": f"ITE[{yr}]",
                     "mlsynth": round(float(gap[years.index(yr)]), 6),
                     "reference": round(float(ref), 6)})
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "NSC", "config": cfg},
        "reference": {"impl": "Tian (2023) NSC.R (vendored, live run, captured), a*=0.3, b*=0.7",
                      "version": "NSC.R / Tian (2023) arXiv:2306.01967v1 "
                                 "(benchmarks/reference/nsc_prop99/)"},
    }


# Deterministic (penalty fixed at the Table-2 values; no stochastic CV) => exact
# re-runs. The reference values are pinned from the live captured NSC.R run
# (benchmarks/reference/nsc_prop99/) via reference_value. The tolerances are set
# to what a faithful port reaches and not to what mlsynth happened to produce:
# the only irreducible difference is NSC.R rounding its returned weights to three
# decimals, which bounds a single weight at 5e-4 and leaves the effect path,
# computed from unrounded weights on both sides, agreeing to ~1e-6 packs. The
# effect-path bands are 0.02 packs -- 0.1 percent of the estimate -- which is
# headroom for BLAS and solver-version drift, not for a methodological gap.
_ns = lambda k: reference_value("nsc_prop99", k)
EXPECTED = {
    "n_donors": (_ns("n_donors"), 0.0),
    "weight_max_abs_dev": (0.0, 1.5e-3),      # 3-decimal rounding floor: 5e-4
    "weight_mean_abs_dev": (0.0, 5.0e-4),
    "weight_correlation": (1.0, 1.0e-4),
    "att_mean_post": (_ns("att_mean_post"), 0.02),
    "ite_1990": (_ns("ite_1990"), 0.02),
    "ite_1995": (_ns("ite_1995"), 0.02),
    "ite_2000": (_ns("ite_2000"), 0.02),
}
