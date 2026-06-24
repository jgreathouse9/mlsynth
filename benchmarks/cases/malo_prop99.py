"""Malo et al. (2024) bilevel optimum: the California Prop 99 application.

Cross-validation (the reference is the SCM-Debug bilevel solver of Malo,
Eskelinen, Zhou & Kuosmanen 2024, ``scm.corner``: a constrained outcome QP for
the donor weights, then an LP for the predictor weights). Malo et al. show that
the seminal ADH (2010) Prop 99 synthetic control is a bilevel optimisation
problem whose global optimum is a corner solution: the predictor weights V
collapse onto a single predictor (cigarette sales per capita in 1980) and the
donor weights are the outcome-fit simplex. Their Table 1 reports this "Optimum",
which the Synth and MSCMT packages both miss.

Run on the same Prop-99 specification (outcome fit over the pre-period
1970-1988, treatment 1989), ``scm.corner`` returns the bilevel-optimum donor
weights

    Donor          scm.corner   Malo Table 1 "Optimum"
    Utah           0.3939       0.3939
    Montana        0.2318       0.2318
    Nevada         0.2049       0.2049
    Connecticut    0.1091       0.1091
    New Hampshire  0.0454       0.0454
    Colorado       0.0148       0.0148

These targets are a live captured run of ``scm.corner`` -- not transcribed
constants -- bundled in ``benchmarks/reference/malo_prop99/``; the case reads
them via :func:`load_reference`. The solver source ``scm.corner.R`` is reused
from the ``malo_basque`` bundle (vendored from SCM-Debug under MIT, see its
``NOTICE``); no separate install is needed (cf.
``benchmarks/R/install_scmcorner.sh``).

mlsynth's ``backend="malo"`` is the staged corner search of that paper. Through
``VanillaSC.fit()`` it reaches the Table 1 optimum:

    Donor          mlsynth malo   scm.corner
    Utah           0.3977         0.3939
    Montana        0.2270         0.2318
    Nevada         0.2039         0.2049
    Connecticut    0.1093         0.1091
    New Hampshire  0.0470         0.0454
    Colorado       0.0151         0.0148

mlsynth lands on the exact cigsale-1980-matched corner (L_W = 0); the captured
``scm.corner`` run reports the outcome-fit lower bound to which it rounds
(L_V = 2.74366). The two agree to ~5e-3 on every donor weight and on the
upper-level objective. The fix that makes the optimum reachable: the bilevel
stages solve the simplex least-squares with the exact active-set QP rather than
the FISTA primitive, which under-converges on the long (1970-1988) pre-period.

Provenance: Malo et al. (2024), Table 1, and their ``scm.corner`` solver
(github.com/Xun90/SCM-Debug). Data ship as
``basedata/augmented_cali_long.csv`` (ADH 2010 Prop 99 panel + predictors).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference, reference_value

_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "augmented_cali_long.csv")

_COVS = ["loginc", "p_cig", "pct15-24", "pc_beer", "cig1975", "cig1980", "cig1988"]
_WINDOWS = {"loginc": (1980, 1988), "p_cig": (1980, 1988), "pct15-24": (1980, 1988),
            "pc_beer": (1984, 1988), "cig1975": (1975, 1975),
            "cig1980": (1980, 1980), "cig1988": (1988, 1988)}


def _ref_weights() -> dict:
    """scm.corner's bilevel-optimum donor weights, from the live captured run in
    ``benchmarks/reference/malo_prop99/`` (parsed ``reference.json["weights"]``)."""
    return {str(k): float(v) for k, v in load_reference("malo_prop99")["weights"].items()}


def _load() -> pd.DataFrame:
    d = pd.read_csv(os.path.abspath(_DATA))
    d["treated"] = ((d.state == "California") & (d.year >= 1989)).astype(int)
    for L in (1975, 1980, 1988):
        m = d[d.year == L].set_index("state")["cigsale"]
        d[f"cig{L}"] = d["state"].map(m)
    return d


def _fit(d):
    from mlsynth import VanillaSC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC({
            "df": d, "outcome": "cigsale", "treat": "treated",
            "unitid": "state", "time": "year",
            "backend": "malo", "covariates": _COVS, "covariate_windows": _WINDOWS,
            "seed": 0, "display_graphs": False,
        }).fit()


def run() -> dict:
    res = _fit(_load())
    w = {str(k): float(v) for k, v in res.weights.donor_weights.items()}
    ref = _ref_weights()
    return {
        "weight_max_abs_dev": float(max(abs(w.get(k, 0.0) - v)
                                        for k, v in ref.items())),
        "weight_utah": w.get("Utah", 0.0),
        "weight_montana": w.get("Montana", 0.0),
        "weight_connecticut": w.get("Connecticut", 0.0),
        "L_V": float(res.fit_diagnostics.rmse_pre ** 2),
        "n_positive_donors": float(sum(1 for v in w.values() if v > 1e-3)),
    }


def comparison() -> dict:
    """mlsynth ``backend="malo"`` vs ``scm.corner``'s bilevel optimum, weight by
    weight. The reference is the six non-zero donor weights ``scm.corner``
    reports on this Prop-99 specification (Malo et al. Table 1 "Optimum"), read
    from the live captured run in ``benchmarks/reference/malo_prop99/``
    (``reference.json["weights"]``, the solver source vendored from SCM-Debug
    under MIT -- Malo et al. 2024) rather than transcribed constants. Returns
    ``{"rows": [...], "mlsynth_call": {...}, "reference": {...}}`` with
    ``{quantity, mlsynth, reference}`` rows for the six bilevel-optimum donors."""
    res = _fit(_load())
    w_ml = {str(k): float(v) for k, v in res.weights.donor_weights.items()}
    # scm.corner's bilevel optimum, from the live captured bundle.
    w_ref = _ref_weights()
    rows = [{"quantity": f"weight[{s}]", "mlsynth": round(w_ml.get(s, 0.0), 6),
             "reference": round(v, 6)} for s, v in w_ref.items()]
    cfg = {"outcome": "cigsale", "treat": "treated", "unitid": "state",
           "time": "year", "backend": "malo", "covariates": _COVS,
           "covariate_windows": _WINDOWS, "seed": 0}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "VanillaSC", "config": cfg},
        "reference": {"impl": "Malo et al. scm.corner (SCM-Debug, live run, captured)",
                      "version": "live captured run (benchmarks/reference/malo_prop99/; "
                                 "Malo et al. 2024, Table 1, github.com/Xun90/SCM-Debug)"},
    }


# Deterministic (exact QP, fixed seed). malo reproduces scm.corner's bilevel
# optimum -- the six donor weights are pinned from the live captured scm.corner
# run in benchmarks/reference/malo_prop99/ (not transcribed); malo agrees to
# ~5e-3 on every donor weight and on the upper-level objective L_V. Tolerances
# reflect the actual mlsynth-vs-scm.corner gap.
_rw = _ref_weights()
EXPECTED = {
    "weight_max_abs_dev": (0.0, 0.006),                 # all donors within ~5e-3
    "weight_utah": (_rw["Utah"], 0.006),                # scm.corner 0.3939
    "weight_montana": (_rw["Montana"], 0.006),          # scm.corner 0.2318
    "weight_connecticut": (_rw["Connecticut"], 0.006),  # scm.corner 0.1091
    "L_V": (reference_value("malo_prop99", "L_V"), 0.01),
    "n_positive_donors": (reference_value("malo_prop99", "n_positive_donors"), 1.0),
}
