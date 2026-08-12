"""Path-B benchmark: MAREX vs the Section 5 simulation of Abadie & Zhao.

Reproduces Table 2 of

    Abadie, A. and Zhao, J. "Synthetic Controls for Experimental Design."
    arXiv:2108.02196 [ABADIE2024].

Scenario 2 (code excerpt). The authors' ``SCDesign`` repository
(github.com/jinglongzhao2/SCDesign) ships the simulation as
``SCdesign_LazyRun.R``, whose generation block (lines 19-176) is the design
described in the paper's Section 5. Its optimization half calls a non-convex
Gurobi MIQP, which is licence-gated, so the design is solved here by MAREX on
the open SCIP backend and the reference is the paper's own published table.

The DGP, from Section 5 and reproduced by the script
-----------------------------------------------------
``J = 15`` units, ``R = 7`` observed and ``F = 11`` unobserved covariates,
``T = 30`` periods with ``T0 = 25``; weights are estimated on the first
``TE = 20`` periods, leaving 21-25 blank. Potential outcomes follow the linear
factor model of Assumption 1: ``delta_t`` and ``upsilon_t`` are small-to-large
rearrangements of Uniform(0, 20) draws, ``Z_j`` and ``mu_j`` are Uniform(0, 1),
the coefficient vectors are Uniform(0, 10), and the errors are ``N(0, 1)``.
Population weights are ``f_j = 1/J``.

Panels are read, not regenerated. ``benchmarks/reference/marex_scdesign_sim/``
captures a live run of the authors' own ``Generate_Model_Primitives`` under the
seeding their driver uses: ``R_job.qsub`` submits an SGE array job
(``#$ -t 1-1000``) and ``Main_LazyRun.R`` reads
``repetition.RANDOM.SEED = Sys.getenv("SGE_TASK_ID")``, so simulation ``i`` is
``set.seed(i)`` -- a thousand independent streams, not one advancing stream.
Under that convention their DGP reproduces Table 2's effect path to the printed
precision, and the capture carries both the per-panel outcomes and the 1000-simulation
average this case pins against the table.

What is pinned
--------------
1. Table 2's effect path. The authors' DGP, run under their seeds for their
   1000 simulations, must reach the ``tau_t`` the table prints. It does, to
   within 0.004 -- display rounding, not Monte-Carlo error. This involves no
   design solve, so it isolates the data-generating process.

2. Table 2. Mean absolute error and root mean square error of ``tau_hat_t``
   against the realized ``tau_t``, plus the number of treated units, for the
   Constrained design at ``m = 1`` and ``m = 3`` and for the Unconstrained
   design (``m = 1``, ``m_bar = J - 1``). The estimator is the paper's equation
   (8): ``tau_hat_t = w'Y_I,t - v'Y_N,t``.

3. The design family. ``design="weakly_targeted"`` carries the trade-off
   parameter ``beta`` of formulation (9): a small ``beta`` favours treated units
   that match the population predictor vector, a large one favours treated and
   control aggregates that match each other. The paper reports that a large
   ``beta`` gives high relative accuracy for the effect on the treated. Both
   designs' accuracy for that estimand is recorded; see EXPECTED for why the
   comparison is not asserted as an ordering.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# Section 5 dimensions and ranges.
J, T_TOTAL, T_NAUGHT, T_PRIME = 15, 30, 25, 20
R_OB, F_UNOB = 7, 11
INTERCEPT_MAX, COVAR_MAX, COEF_MAX, NOISE_SD = 20.0, 1.0, 10.0, 1.0

M_SIMS = 20          # the paper uses 1000; see EXPECTED for the tolerance argument
SEED = 0
BETA_LARGE = 20.0
M_WEAK = 3           # cardinality for the design-family comparison

_CFG = {
    "outcome": "y", "unitid": "unit", "time": "time",
    "T0": T_NAUGHT, "blank_periods": T_NAUGHT - T_PRIME,
    "T_post": T_TOTAL - T_NAUGHT,
    "design": "standard", "program_type": "MIQP", "relaxed": False,
    "inference": False, "display_graph": False,
}


_REF_CASE = "marex_scdesign_sim"
PAPER_TAU = {26: -13.58, 27: -10.99, 28: -8.35, 29: -5.00, 30: -2.50}


def _panels(weights: dict, seed: int):
    """The control and treated outcomes R drew for simulation ``seed``.

    ``(T, J)`` and ``(T - T0, J)``; treated outcomes exist only over the
    experimental periods, which is where the authors define them.
    """
    Y_N = np.array([[weights[f"yn_s{seed}_u{j}_t{k}"] for j in range(1, J + 1)]
                    for k in range(1, T_TOTAL + 1)])
    Y_I = np.array([[weights[f"yi_s{seed}_u{j}_t{k}"] for j in range(1, J + 1)]
                    for k in range(T_NAUGHT + 1, T_TOTAL + 1)])
    return Y_N, Y_I


def _long(Y: np.ndarray) -> pd.DataFrame:
    T, N = Y.shape
    return pd.DataFrame({
        "unit": np.repeat(np.arange(1, N + 1), T),
        "time": np.tile(np.arange(1, T + 1), N),
        "y": Y.T.reshape(-1),
    })


def _weights(Y_N: np.ndarray, setting, design="standard", beta=None):
    """Solve one design; return its aggregate treated and control weights."""
    from mlsynth import MAREX

    cfg = dict(_CFG)
    cfg["design"] = design
    if beta is not None:
        cfg["beta"] = beta
    if setting == "unconstrained":
        cfg["m_min"], cfg["m_max"] = 1, J - 1
    else:
        cfg["m_eq"] = int(setting)

    res = MAREX({"df": _long(Y_N), **cfg}).fit()
    g = res.globres
    return (np.asarray(g.treated_weights_agg, dtype=float),
            np.asarray(g.control_weights_agg, dtype=float))


def _cell(weights, setting, m_sims: int, design="standard", beta=None) -> dict:
    """One design, solved once per simulation, scored against both estimands.

    ``tau`` is the population effect under uniform ``f``; ``tau_treated`` is the
    effect on the ``w``-weighted treated, which is the estimand ``beta`` trades
    toward. Both come off the same solve, so the m = 3 standard design is not
    solved twice.
    """
    maes, rmses, sizes, treated = [], [], [], []
    for s in range(1, m_sims + 1):
        Y_N, Y_I = _panels(weights, s)
        w, v = _weights(Y_N, setting, design=design, beta=beta)
        tau_hat = Y_I @ w - Y_N[T_NAUGHT:] @ v             # equation (8)
        tau = (Y_I - Y_N[T_NAUGHT:]).mean(axis=1)          # uniform f
        err = tau_hat - tau
        maes.append(np.abs(err).mean())
        rmses.append(np.sqrt((err ** 2).mean()))
        sizes.append(int((w > 1e-6).sum()))
        treated.append(np.sqrt((((Y_I @ w - Y_N[T_NAUGHT:] @ w) - tau_hat) ** 2).mean()))
    return {"mae": float(np.mean(maes)), "rmse": float(np.mean(rmses)),
            "n_treated": float(np.mean(sizes)),
            "treated_rmse": float(np.mean(treated))}


def run() -> dict:
    from benchmarks.reference import load_reference

    ref = load_reference(_REF_CASE)
    values, weights = ref["values"], ref["weights"]
    n_panels = int(values["n_panels"])
    m_sims = min(M_SIMS, n_panels)

    out: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 1. the authors' DGP against Table 2's printed effect path
        out["tau_path_max_abs_dev"] = max(
            abs(values[f"tau_bar_t{t}"] - PAPER_TAU[t]) for t in PAPER_TAU)
        out["tau_sims"] = float(values["tau_sims"])

        # 2. Table 2, on the captured panels
        cells = {}
        for label, setting in (("m1", 1), ("m3", 3), ("unconstrained", "unconstrained")):
            cells[label] = _cell(weights, setting, m_sims)
            out[f"mae_{label}"] = cells[label]["mae"]
            out[f"rmse_{label}"] = cells[label]["rmse"]
            out[f"n_treated_{label}"] = cells[label]["n_treated"]
        out["mae_falls_with_m"] = float(
            out["mae_m1"] > out["mae_m3"] > out["mae_unconstrained"])

        # 3. the design family, at the cardinality the m = 3 cell already solved
        out["treated_rmse_standard"] = cells["m3"]["treated_rmse"]
        out["treated_rmse_weakly_targeted"] = _cell(
            weights, M_WEAK, m_sims, design="weakly_targeted",
            beta=BETA_LARGE)["treated_rmse"]
    return out


# Tolerances.
#
# tau_path_max_abs_dev compares the authors' own DGP, run under their seeds for
# their 1000 simulations, against the effect path Table 2 prints. Measured 0.004,
# which is the table's display rounding: the printed values carry two decimals.
# 0.01 admits that and nothing more. This is a deterministic capture, so there is
# no Monte-Carlo term -- the number moves only if the DGP or the seeding changes.
#
# The Table 2 cells run on 12 captured panels against the paper's 1000, so they
# carry real Monte-Carlo error. Measured at 12 and at 24 panels the cells move by
# 0.1 to 0.2, and the 24-panel values sit within 0.25 of the paper throughout.
# Each target is this run's measured value and each tolerance still brackets the
# paper's published one: 2.93/3.45 at m = 1, 1.26/1.49 at m = 3, 0.83/0.97
# unconstrained, and 6.76 treated units. 0.60 on the m = 1 cells, whose scale is
# largest, and 0.35 elsewhere. A regression in the design
# would move these by far more, since the spread across cardinalities is 2.9 down
# to 0.7.
#
# n_treated is exact for the constrained cells. The unconstrained cell averages an
# integer count and the paper reports 6.76; 0.8 covers the sampling of that
# average at 12 panels.
#
# mae_falls_with_m is the paper's headline for Table 2 -- performance improves as
# the cardinality constraint relaxes -- and the gaps are large enough that the
# indicator is not marginal.
#
# tau_sims pins that the captured path really is the authors' 1000-simulation
# average and not a shorter run that happened to land close.
#
# The two treated-effect numbers are recorded, not compared. The paper reports
# that a large beta favours the effect on the treated. The measurement does not
# settle it: standard leads on these 12 panels (1.46 against 1.66), weakly
# targeted led on the 20 drawn from an independent stream, and at 25 it led by
# about 0.9 combined standard errors. The ordering is inside the noise at any
# panel count this case can afford, so asserting it would be a coin flip. Pinning the levels catches a regression in either design; the
# ordering is left to the replication page to describe.
EXPECTED = {
    "tau_path_max_abs_dev": (0.0, 0.01),
    "tau_sims": (1000.0, 0.0),
    "mae_m1": (3.03, 0.60),
    "rmse_m1": (3.49, 0.60),
    "n_treated_m1": (1.0, 0.0),
    "mae_m3": (1.23, 0.35),
    "rmse_m3": (1.41, 0.35),
    "n_treated_m3": (3.0, 0.0),
    "mae_unconstrained": (0.57, 0.35),
    "rmse_unconstrained": (0.69, 0.35),
    "n_treated_unconstrained": (6.83, 0.80),
    "mae_falls_with_m": (1.0, 0.0),
    "treated_rmse_standard": (1.46, 0.50),
    "treated_rmse_weakly_targeted": (1.66, 0.50),
}
