"""Cross-validation: MAREX vs SCDesign's own design routine, on the paper's
Section 5 simulation panels.

The reference is a live run of the authors' R code, captured in
``benchmarks/reference/marex_scdesign_sim/`` (regenerate with
``python benchmarks/reference/generate.py marex_scdesign_sim``). Two pieces of
``github.com/jinglongzhao2/SCDesign`` are reproduced verbatim there:

* the Section 5 generation block (``SCdesign_LazyRun.R`` lines 19-176), which
  draws the panels; and
* ``Synthetic_Control`` + ``Synthetic_Experiment_Cardinality_Constraint``
  ("3. Walmart Data Simulations/Walmart_LazyRun.R"), the constrained design.

Validated without Gurobi
------------------------
SCDesign's headline design is a non-convex Gurobi MIQP and is licence-gated. Its
cardinality-constrained routine is fully open: it enumerates every partition of
size ``1..K``, solves the treated and control synthetic-control weights for each
by ``quadprog::solve.QP``, and keeps the min-loss partition. That is the design
MAREX solves with ``m_min = 1``, ``m_max = K``, and it needs no commercial
solver. The same route is what makes :mod:`benchmarks.cases.marex_walmart`
possible on the authors' empirical application; this case runs it on their
simulation instead.

What makes this a cross-validation and not a table comparison
--------------------------------------------------------------
The captured output carries the panels themselves -- every :math:`Y^N_{jt}` and
every observed covariate :math:`Z_{kj}` R drew -- so both sides score the
identical draw. Nothing here depends on reproducing R's random stream in numpy,
which cannot be done: the draws are read, not regenerated.

That also fixes what the paper's own Table 2 cannot settle.
``SCdesign_LazyRun.R`` is a single-simulation script, while Table 2 averages
1000 simulations from a driver that is not in the file, so even running the
authors' code under their seed does not reproduce their printed averages. Their
design routine on a named panel is exact, and that is what is compared here.
:mod:`benchmarks.cases.marex_section5_mc` covers the Table 2 numbers separately,
as Path B.

The design matches on the 20 fitting periods and the 7 observed covariates, with
per-predictor standardisation -- the routine's row rescaling, which is MAREX's
``standardize=True``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference

_CASE = "marex_scdesign_sim"
N_UNITS, T_TOTAL, N_COVARIATES = 15, 30, 7
T_NAUGHT, T_PRIME = 25, 20
_COVS = [f"z{k}" for k in range(1, N_COVARIATES + 1)]


def _panel(weights: dict, rep: int):
    """The outcome matrix ``(T, N)`` and covariates ``(R, N)`` R drew for ``rep``."""
    Y = np.array([[weights[f"y_r{rep}_u{j}_t{t}"] for j in range(1, N_UNITS + 1)]
                  for t in range(1, T_TOTAL + 1)])
    Z = np.array([[weights[f"z_r{rep}_k{k}_u{j}"] for j in range(1, N_UNITS + 1)]
                  for k in range(1, N_COVARIATES + 1)])
    return Y, Z


def _frame(Y: np.ndarray, Z: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({
        "unit": np.repeat(np.arange(1, N_UNITS + 1), T_TOTAL),
        "time": np.tile(np.arange(1, T_TOTAL + 1), N_UNITS),
        "y": Y.T.reshape(-1),
    })
    for k, name in enumerate(_COVS):
        df[name] = df["unit"].map({j + 1: Z[k, j] for j in range(N_UNITS)})
    return df


def _fit(df: pd.DataFrame, k_cardinality: int):
    from mlsynth import MAREX

    res = MAREX({
        "df": df, "outcome": "y", "unitid": "unit", "time": "time",
        "T0": T_NAUGHT, "blank_periods": T_NAUGHT - T_PRIME,
        "T_post": T_TOTAL - T_NAUGHT,
        "m_min": 1, "m_max": k_cardinality,
        "design": "standard", "program_type": "MIQP", "relaxed": False,
        "inference": False, "display_graph": False,
        "covariates": _COVS, "standardize": True,
    }).fit()
    selected = sorted(int(u) for u in res.selected_units)
    return selected, np.asarray(res.globres.treated_weights_agg, dtype=float)


def run() -> dict:
    ref = load_reference(_CASE)
    values, weights = ref["values"], ref["weights"]
    n_reps = int(values["m_reps"])
    k_card = int(values["k_cardinality"])

    same_set, weight_diffs, same_size = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rep in range(1, n_reps + 1):
            Y, Z = _panel(weights, rep)
            selected, w = _fit(_frame(Y, Z), k_card)

            ref_sel = sorted(int(k.rsplit("_u", 1)[1])
                             for k in weights if k.startswith(f"sel_r{rep}_u"))
            same_set.append(float(selected == ref_sel))
            same_size.append(float(len(selected) == int(values[f"n_treated_rep{rep}"])))
            if selected == ref_sel:
                weight_diffs.append(max(
                    abs(w[j - 1] - weights[f"sel_r{rep}_u{j}"]) for j in ref_sel))

    return {
        "n_reps": float(n_reps),
        "selected_set_agreement": float(np.mean(same_set)),
        "treated_count_agreement": float(np.mean(same_size)),
        "max_treated_weight_abs_diff": float(max(weight_diffs)) if weight_diffs else 1.0,
    }


def comparison() -> dict:
    """Per-panel view of the two designs, for the docs table."""
    ref = load_reference(_CASE)
    values, weights = ref["values"], ref["weights"]
    rows = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rep in range(1, int(values["m_reps"]) + 1):
            Y, Z = _panel(weights, rep)
            selected, w = _fit(_frame(Y, Z), int(values["k_cardinality"]))
            ref_sel = sorted(int(k.rsplit("_u", 1)[1])
                             for k in weights if k.startswith(f"sel_r{rep}_u"))
            rows[f"rep{rep}"] = {
                "mlsynth_selected": selected, "scdesign_selected": ref_sel,
                "mlsynth_weights": [round(float(w[j - 1]), 4) for j in selected],
                "scdesign_weights": [round(weights[f"sel_r{rep}_u{j}"], 4) for j in ref_sel],
            }
    return rows


# Tolerances.
#
# The two agreement figures are exact. Both sides solve the same program on the
# same panel -- the reference enumerates every partition of size 1..K and keeps
# the minimum, so its answer is the optimum by construction, and MAREX's
# mixed-integer solve must reach it. A single disagreeing panel means one of the
# two is not solving the design it claims to, which is what this case exists to
# catch.
#
# The weight gap is bounded by the two solvers' numerical tolerances. R's
# solve.QP works on a nearPD-repaired, mean-rescaled Hessian and rounds its
# output to six decimals; MAREX reaches the same optimum through SCIP. Measured
# 1.3e-04 on the first panel. 5e-03 covers that and a differently-built solver
# while staying far below the spacing between distinct designs -- the weights
# here are order 0.4 to 0.6, so a genuine disagreement is a swing of tenths.
#
# n_reps guards the loop: a case that silently stopped iterating panels would
# still report perfect agreement over whatever remained.
EXPECTED = {
    "n_reps": (6.0, 0.0),
    "selected_set_agreement": (1.0, 0.0),
    "treated_count_agreement": (1.0, 0.0),
    "max_treated_weight_abs_diff": (0.0, 5e-3),
}
