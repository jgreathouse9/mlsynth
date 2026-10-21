"""Cross-validation: MAREX's exact backend vs brute-force enumeration in R.

MAREX's exact backend (``backend="exact"``) finds the design by a bounded
search: a closed-form lower bound on the treated fit discards candidates, an
incumbent cutoff stops the weakly-targeted fits early, and an active-set
solver fits the survivors. Each of those is an opportunity to lose the
optimum, and none of them would show in a comparison with a port of the same
search, which would share them.

The reference shares none of them. ``benchmarks/reference/marex_exact_bruteforce/
reference.R`` enumerates every treated set of the stated size and solves each
fixed-set program with ``quadprog::solve.QP``: two simplex fits for the
separable designs, one joint quadratic program over both weight vectors for
the weakly-targeted design. Its answer is the optimum by enumeration. The
captured run carries the panels themselves, so both sides score the identical
numbers.

Twelve configurations on six panels of eight markets cover every design the
backend solves and the options that change its objective: the standard design
with two and three treated, the penalized and unit-penalized designs, the
weakly-targeted design at beta = 0.01, 1 and 100, Abadie and L'Hour's control
penalty (Vives-i-Bastida 2022, program 1) at 0.1 and near its convexity bound,
and standardised covariates at one weight and at per-covariate weights.

Regenerate the reference with
``python benchmarks/reference/generate.py marex_exact_bruteforce``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference

_CASE = "marex_exact_bruteforce"

#: The R script's configurations, as MAREX configurations.
CONFIGS = {
    "standard": dict(design="standard", m_eq=2),
    "standard_m3": dict(design="standard", m_eq=3),
    "penalized": dict(design="penalized", m_eq=2, lambda1=0.05, lambda2=0.02),
    "unit_penalized": dict(design="unit_penalized", m_eq=2, xi=0.5,
                           lambda1_unit=0.05),
    "weakly_b001": dict(design="weakly_targeted", m_eq=2, beta=0.01),
    "weakly_b1": dict(design="weakly_targeted", m_eq=2, beta=1.0),
    "weakly_b100": dict(design="weakly_targeted", m_eq=2, beta=100.0),
    "penalty_b1": dict(design="weakly_targeted", m_eq=2, beta=1.0,
                       control_penalty=0.1),
    "penalty_b1_near_bound": dict(design="weakly_targeted", m_eq=2, beta=1.0,
                                  control_penalty=0.6),
    "covariates_std": dict(design="standard", m_eq=2, standardize=True,
                           covariates=["z1", "z2"], covariate_weight=0.1),
    "covariates_std_h": dict(design="standard", m_eq=2, standardize=True,
                             covariates=["z1", "z2"],
                             covariate_weight={"z1": 0.3, "z2": 2.0}),
    "covariates_weakly": dict(design="weakly_targeted", m_eq=2, beta=1.0,
                              control_penalty=0.1, standardize=True,
                              covariates=["z1", "z2"],
                              covariate_weight={"z1": 0.5, "z2": 0.1}),
}


def _frame(weights: dict, p: int, n: int, t_all: int) -> pd.DataFrame:
    rows = []
    for j in range(1, n + 1):
        z1, z2 = weights[f"z_p{p}_u{j}_k1"], weights[f"z_p{p}_u{j}_k2"]
        for t in range(1, t_all + 1):
            rows.append({"unit": j, "time": t, "y": weights[f"y_p{p}_u{j}_t{t}"],
                         "z1": z1, "z2": z2})
    return pd.DataFrame(rows)


def _fit(df: pd.DataFrame, t_fit: int, cfg: dict):
    from mlsynth import MAREX

    res = MAREX({
        "df": df, "outcome": "y", "unitid": "unit", "time": "time",
        "T0": t_fit, "blank_periods": 0, "inference": False,
        "display_graph": False, "backend": "exact", **cfg,
    }).fit()
    g = res.globres
    return (float(g.objective), np.asarray(g.treated_weights_agg, dtype=float),
            np.asarray(g.control_weights_agg, dtype=float))


def _reference(weights: dict, name: str, p: int, n: int):
    w = np.array([weights.get(f"w_{name}_p{p}_u{j}", 0.0) for j in range(1, n + 1)])
    v = np.array([weights.get(f"v_{name}_p{p}_u{j}", 0.0) for j in range(1, n + 1)])
    return w, v


def run() -> dict:
    ref = load_reference(_CASE)
    values, weights = ref["values"], ref["weights"]
    n_panels, n = int(values["n_panels"]), int(values["n_units"])
    t_fit = int(values["t_fit"])
    t_all = t_fit + int(values["t_post"])

    same_set, obj_gap, w_gap, v_gap, cells = [], [], [], [], 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(1, n_panels + 1):
            df = _frame(weights, p, n, t_all)
            for name, cfg in CONFIGS.items():
                obj, w, v = _fit(df, t_fit, cfg)
                rw, rv = _reference(weights, name, p, n)
                ref_obj = values[f"obj_{name}_p{p}"]
                same_set.append(float(set(np.flatnonzero(w > 1e-8))
                                      == set(np.flatnonzero(rw > 0))))
                obj_gap.append(abs(obj - ref_obj) / (1.0 + abs(ref_obj)))
                w_gap.append(float(np.max(np.abs(w - rw))))
                v_gap.append(float(np.max(np.abs(v - rv))))
                cells += 1

    return {
        "cells": float(cells),
        "treated_set_agreement": float(np.mean(same_set)),
        "max_objective_rel_gap": float(max(obj_gap)),
        "max_treated_weight_abs_gap": float(max(w_gap)),
        "max_control_weight_abs_gap": float(max(v_gap)),
    }


def comparison() -> dict:
    """Per configuration, the worst gaps over the panels, for the docs table."""
    ref = load_reference(_CASE)
    values, weights = ref["values"], ref["weights"]
    n_panels, n = int(values["n_panels"]), int(values["n_units"])
    t_fit = int(values["t_fit"])
    t_all = t_fit + int(values["t_post"])
    rows = {name: {"same_set": 0, "obj_gap": 0.0, "weight_gap": 0.0}
            for name in CONFIGS}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(1, n_panels + 1):
            df = _frame(weights, p, n, t_all)
            for name, cfg in CONFIGS.items():
                obj, w, v = _fit(df, t_fit, cfg)
                rw, rv = _reference(weights, name, p, n)
                ref_obj = values[f"obj_{name}_p{p}"]
                r = rows[name]
                r["same_set"] += int(set(np.flatnonzero(w > 1e-8))
                                     == set(np.flatnonzero(rw > 0)))
                r["obj_gap"] = max(r["obj_gap"],
                                   abs(obj - ref_obj) / (1.0 + abs(ref_obj)))
                r["weight_gap"] = max(r["weight_gap"],
                                      float(np.max(np.abs(np.r_[w - rw, v - rv]))))
    return rows


# Tolerances.
#
# The treated-set agreement is exact: both sides solve the same program on the
# same panel, the reference by enumeration, so a single disagreeing cell means
# the search discarded the optimum or solved a different objective, which is
# what the case exists to catch. ``cells`` guards the loop: 6 panels times 12
# configurations.
#
# The objective and weight gaps are the two solvers' numerical error. The R side
# adds a 1e-10 ridge to make quadprog's matrix positive definite and solves to
# quadprog's accuracy; MAREX's active-set solver is exact on the active face and
# the coupled fit stops on a 1e-10 relative Frank-Wolfe gap. Measured over the
# 72 cells: objectives agree to 2.2e-14 relative, treated weights to 1.1e-9 and
# control weights to 2.1e-8. The pins sit 50 to 5,000 times above those and far
# below the distance between distinct designs, whose weights differ by tenths.
#
# The case has power. With the search's bound inflated 50-fold, so that it
# discards candidates it should fit, treated-set agreement fell to 0.72 and the
# objective gap to 8.0.
EXPECTED = {
    "cells": (72.0, 0.0),
    "treated_set_agreement": (1.0, 0.0),
    "max_objective_rel_gap": (0.0, 1e-10),
    "max_treated_weight_abs_gap": (0.0, 1e-6),
    "max_control_weight_abs_gap": (0.0, 1e-6),
}
