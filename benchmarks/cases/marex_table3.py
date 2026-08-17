"""Path-B benchmark: MAREX against Table 3 of Abadie & Zhao.

Table 3 sets the synthetic control design against five randomized alternatives
at every cardinality, over the authors' 1000 simulations:

===========  ====  ====  ====  ====  ====  ====
:math:`m`      SC   RND   STR   REG  1-NN  5-NN
===========  ====  ====  ====  ====  ====  ====
1            3.45  6.35  6.35  8.14  5.28  4.40
2            2.00  4.70  3.53  6.00  3.69  3.20
3            1.49  3.91  2.75  5.11  3.02  2.66
4            1.25  3.49  2.44  4.49  2.67  2.40
5            1.09  3.22  2.07  4.12  2.38  2.28
6            1.02  3.04  1.95  3.87  2.24  2.23
7            0.97  3.01  1.85  3.90  2.18  2.32
===========  ====  ====  ====  ====  ====  ====

RND is randomized assignment with difference in means; STR stratified
randomization with difference in means per stratum; REG randomized assignment
with regression adjustment on the covariates; 1-NN and 5-NN randomized
assignment with nearest-neighbour matching.

What is reproduced, and what is read off the table
--------------------------------------------------
The SC column is computed here, by MAREX, on panels drawn by the authors' own
DGP. Table 3's note defines SC as the Constrained formulation, which in
``Main_LazyRun.R`` is ``Synthetic_Experiment_Cardinality_Constraint`` -- the
quadprog routine, with no Gurobi call anywhere in that section -- and
:mod:`benchmarks.cases.marex_scdesign_sim` shows MAREX reaches the same design
as that routine, unit for unit. So the column is ours to compute, not to
copy, and the same numbers appear as Table 2's RMSE column.

The five comparator columns are taken from the published table. They are
properties of the authors' randomized designs, not of ``mlsynth``, so
re-implementing them would test a transcription. What matters for the library is
the ordering they establish, and the paper's headline is that SC wins at every
cardinality: the case asserts MAREX's SC beats the best published alternative at
each :math:`m`, which is the strongest of the five and therefore the whole row.

Panels come from ``benchmarks/reference/marex_scdesign_sim/``, captured from
``Generate_Model_Primitives`` under the authors' driver seeding (``set.seed(i)``
for simulation ``i``); see :mod:`benchmarks.cases.marex_section5_mc` for the
check that this reproduces their effect path.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference

_REF_CASE = "marex_scdesign_sim"
J, T_TOTAL, N_COVARIATES = 15, 30, 7
T_NAUGHT, T_PRIME = 25, 20
_COVS = [f"z{k}" for k in range(1, N_COVARIATES + 1)]

CARDINALITIES = (1, 2, 3, 4, 5, 6, 7)

#: Table 3, verbatim. SC is recomputed; the rest are the comparison.
TABLE3 = {
    1: {"SC": 3.45, "RND": 6.35, "STR": 6.35, "REG": 8.14, "1NN": 5.28, "5NN": 4.40},
    2: {"SC": 2.00, "RND": 4.70, "STR": 3.53, "REG": 6.00, "1NN": 3.69, "5NN": 3.20},
    3: {"SC": 1.49, "RND": 3.91, "STR": 2.75, "REG": 5.11, "1NN": 3.02, "5NN": 2.66},
    4: {"SC": 1.25, "RND": 3.49, "STR": 2.44, "REG": 4.49, "1NN": 2.67, "5NN": 2.40},
    5: {"SC": 1.09, "RND": 3.22, "STR": 2.07, "REG": 4.12, "1NN": 2.38, "5NN": 2.28},
    6: {"SC": 1.02, "RND": 3.04, "STR": 1.95, "REG": 3.87, "1NN": 2.24, "5NN": 2.23},
    7: {"SC": 0.97, "RND": 3.01, "STR": 1.85, "REG": 3.90, "1NN": 2.18, "5NN": 2.32},
}


def best_alternative(m: int) -> float:
    """The strongest published comparator at cardinality ``m``."""
    return min(v for k, v in TABLE3[m].items() if k != "SC")


def _panels(weights: dict, seed: int):
    Y_N = np.array([[weights[f"yn_s{seed}_u{j}_t{k}"] for j in range(1, J + 1)]
                    for k in range(1, T_TOTAL + 1)])
    Y_I = np.array([[weights[f"yi_s{seed}_u{j}_t{k}"] for j in range(1, J + 1)]
                    for k in range(T_NAUGHT + 1, T_TOTAL + 1)])
    Z = np.array([[weights[f"z_s{seed}_k{k}_u{j}"] for j in range(1, J + 1)]
                  for k in range(1, N_COVARIATES + 1)])
    return Y_N, Y_I, Z


def _frame(Y_N: np.ndarray, Z: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({
        "unit": np.repeat(np.arange(1, J + 1), T_TOTAL),
        "time": np.tile(np.arange(1, T_TOTAL + 1), J),
        "y": Y_N.T.reshape(-1),
    })
    for k, name in enumerate(_COVS):
        df[name] = df["unit"].map({j + 1: Z[k, j] for j in range(J)})
    return df


def _sc_rmse(weights: dict, m: int, n_panels: int) -> float:
    """RMSE of the design's effect estimate against the realized effect."""
    from mlsynth import MAREX

    errs = []
    for s in range(1, n_panels + 1):
        Y_N, Y_I, Z = _panels(weights, s)
        res = MAREX({
            "df": _frame(Y_N, Z), "outcome": "y", "unitid": "unit", "time": "time",
            "T0": T_NAUGHT, "blank_periods": T_NAUGHT - T_PRIME,
            "T_post": T_TOTAL - T_NAUGHT,
            "m_min": 1, "m_max": m,
            "design": "standard", "program_type": "MIQP", "relaxed": False,
            "inference": False, "display_graph": False,
            "covariates": _COVS, "standardize": True,
        }).fit()
        w = np.asarray(res.globres.treated_weights_agg, dtype=float)
        v = np.asarray(res.globres.control_weights_agg, dtype=float)
        tau_hat = Y_I @ w - Y_N[T_NAUGHT:] @ v            # equation (8)
        tau = (Y_I - Y_N[T_NAUGHT:]).mean(axis=1)         # uniform f
        errs.append(np.sqrt(((tau_hat - tau) ** 2).mean()))
    return float(np.mean(errs))


def run() -> dict:
    try:
        import pyscipopt  # noqa: F401
    except ImportError as exc:
        from benchmarks.compare import BenchmarkSkipped
        raise BenchmarkSkipped(
            "MAREX needs the SCIP MIP solver (pip install 'mlsynth[design]')."
        ) from exc
    ref = load_reference(_REF_CASE)
    weights = ref["weights"]
    n_panels = int(ref["values"]["n_panels"])

    out: dict = {"n_panels": float(n_panels)}
    beats = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in CARDINALITIES:
            rmse = _sc_rmse(weights, m, n_panels)
            out[f"sc_rmse_m{m}"] = rmse
            beats.append(float(rmse < best_alternative(m)))
    out["sc_beats_every_alternative"] = float(np.mean(beats))
    steep = (1, 2, 3)
    out["sc_rmse_falls_over_the_steep_range"] = float(
        all(out[f"sc_rmse_m{a}"] > out[f"sc_rmse_m{b}"]
            for a, b in zip(steep, steep[1:])))
    out["sc_rmse_m7_well_below_m1"] = float(
        out["sc_rmse_m7"] < 0.5 * out["sc_rmse_m1"])
    return out


# Tolerances.
#
# The SC cells run on 12 captured panels against the paper's 1000, so they carry
# Monte-Carlo error; the targets below are this run's measured values and each
# tolerance brackets the published one. The gap that matters is not between our
# cell and theirs but between SC and the alternatives, which is a factor of two
# to five at every cardinality -- so a tolerance wide enough to absorb 12-panel
# noise is still far too tight to let a broken design pass.
#
# sc_beats_every_alternative is the paper's headline for Table 3 and the reason
# the comparator columns are quoted instead of recomputed. It compares against
# the strongest published alternative at each m -- 4.40, 3.20, 2.66, 2.40, 2.07,
# 1.95, 1.85 -- so clearing it clears the whole row. It admits no near miss.
#
# Monotonicity is pinned only where 12 panels can resolve it. The table falls
# steeply at first -- 3.45, 2.00, 1.49 -- and then by 0.07 and 0.05 between
# m = 5, 6 and 7. Those last gaps are well inside the Monte-Carlo error of 12
# panels against the paper's 1000, and this run duly puts m = 6 (0.843) a hair
# above m = 5 (0.832). Asserting the full ordering would be asserting noise, so
# the case pins the steep range, where the gaps are an order of magnitude larger
# than the scatter, and separately pins that m = 7 lands below half of m = 1 --
# the overall trend, which no single cell carries.
EXPECTED = {
    "n_panels": (12.0, 0.0),
    "sc_rmse_m1": (3.46, 0.60),
    "sc_rmse_m2": (1.68, 0.55),
    "sc_rmse_m3": (1.30, 0.40),
    "sc_rmse_m4": (1.03, 0.40),
    "sc_rmse_m5": (0.83, 0.40),
    "sc_rmse_m6": (0.84, 0.40),
    "sc_rmse_m7": (0.82, 0.40),
    "sc_beats_every_alternative": (1.0, 0.0),
    "sc_rmse_falls_over_the_steep_range": (1.0, 0.0),
    "sc_rmse_m7_well_below_m1": (1.0, 0.0),
}
