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

R's Mersenne-Twister stream cannot be reproduced by numpy's PCG64, so these are
not the authors' draws. The structure, dimensions and ranges are theirs, and the
check below confirms the port reaches their treatment-effect path.

What is pinned
--------------
1. The DGP itself. Because both intercept series are order statistics of
   Uniform(0, 20) draws and the covariate terms share their distributions across
   the treated and control processes, ``tau_t`` has a closed form:
   ``20k/6 - 20(k+25)/31`` for the k-th experimental period. The port must reach
   it, which checks the port without involving MAREX at all. The paper's own
   Table 2 values sit within 1.3 standard errors of the same closed form at
   their ``M = 1000``.

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


def analytic_tau() -> np.ndarray:
    """``E[upsilon_k] - E[delta_{25+k}]`` for the five experimental periods.

    The k-th order statistic of ``n`` Uniform(0, b) draws has mean
    ``b k / (n + 1)``. ``upsilon`` sorts 5 draws and ``delta`` sorts 30, and the
    covariate terms have the same distribution on both sides, so they cancel in
    expectation and the difference of the intercepts is the whole effect.
    """
    k = np.arange(1, T_TOTAL - T_NAUGHT + 1)
    return INTERCEPT_MAX * k / 6.0 - INTERCEPT_MAX * (k + T_NAUGHT) / (T_TOTAL + 1)


def panel(seed: int):
    """Control and treated potential outcomes; ``(T, J)`` and ``(T - T0, J)``."""
    rng = np.random.default_rng(seed)
    n_post = T_TOTAL - T_NAUGHT

    delta = np.sort(np.concatenate([
        INTERCEPT_MAX * rng.random(T_NAUGHT), INTERCEPT_MAX * rng.random(n_post)]))
    upsilon = np.sort(INTERCEPT_MAX * rng.random(n_post))

    Z = COVAR_MAX * rng.random((R_OB, J))
    mu = COVAR_MAX * rng.random((F_UNOB, J))
    theta = COEF_MAX * rng.random((R_OB, T_TOTAL))
    gamma = COEF_MAX * rng.random((R_OB, n_post))
    lam = COEF_MAX * rng.random((F_UNOB, T_TOTAL))
    eta = COEF_MAX * rng.random((F_UNOB, n_post))

    eps = rng.normal(0.0, NOISE_SD, size=(T_TOTAL, J))
    xi = rng.normal(0.0, NOISE_SD, size=(n_post, J))

    Y_N = delta[:, None] + theta.T @ Z + lam.T @ mu + eps
    Y_I = upsilon[:, None] + gamma.T @ Z + eta.T @ mu + xi
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


def _cell(setting, m_sims: int, seed0: int, design="standard", beta=None) -> dict:
    """One design, solved once per simulation, scored against both estimands.

    ``tau`` is the population effect under uniform ``f``; ``tau_treated`` is the
    effect on the ``w``-weighted treated, which is the estimand ``beta`` trades
    toward. Both come off the same solve, so the m = 3 standard design is not
    solved twice.
    """
    maes, rmses, sizes, treated = [], [], [], []
    for s in range(m_sims):
        Y_N, Y_I = panel(seed0 + s)
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
    out: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 1. the port reaches the paper's treatment-effect path (no solver involved)
        acc = np.zeros(T_TOTAL - T_NAUGHT)
        for s in range(400):
            Y_N, Y_I = panel(SEED + s)
            acc += (Y_I - Y_N[T_NAUGHT:]).mean(axis=1)
        out["tau_path_max_abs_dev"] = float(np.abs(acc / 400 - analytic_tau()).max())

        # 2. Table 2
        cells = {}
        for label, setting in (("m1", 1), ("m3", 3), ("unconstrained", "unconstrained")):
            cells[label] = _cell(setting, M_SIMS, SEED)
            out[f"mae_{label}"] = cells[label]["mae"]
            out[f"rmse_{label}"] = cells[label]["rmse"]
            out[f"n_treated_{label}"] = cells[label]["n_treated"]
        out["mae_falls_with_m"] = float(
            out["mae_m1"] > out["mae_m3"] > out["mae_unconstrained"])

        # 3. the design family, at the cardinality the m = 3 cell already solved
        out["treated_rmse_standard"] = cells["m3"]["treated_rmse"]
        out["treated_rmse_weakly_targeted"] = _cell(
            M_WEAK, M_SIMS, SEED, design="weakly_targeted",
            beta=BETA_LARGE)["treated_rmse"]
    return out


# Tolerances.
#
# tau_path_max_abs_dev is measured against a closed form, so the only error is
# Monte-Carlo. The per-period standard deviation across simulations is about 9.4,
# so at 400 draws the standard error is about 0.47 per period; 1.2 admits the
# largest of five such deviations without admitting a mis-ported DGP, which would
# shift a period by whole units. The paper's own Table 2 sits within 1.3 standard
# errors of the same closed form at its M = 1000, which is what licenses using the
# closed form as the target instead of their printed values.
#
# The Table 2 cells run M = 20 against the paper's 1000, so they carry real
# Monte-Carlo error. Measured at M = 12 and M = 24 the cells move by 0.1 to 0.2,
# and the M = 24 values sit within 0.25 of the paper throughout. Tolerances are
# set to bracket both the paper's published value and that drift: 0.45 on the
# m = 1 cells, whose scale is largest, and 0.30 elsewhere. A regression in the
# design would move these by far more, since the spread across cardinalities is
# 2.9 down to 0.7.
#
# n_treated is exact for the constrained cells. The unconstrained cell averages
# an integer count and the paper reports 6.76; 0.6 covers the sampling of that
# average at M = 20.
#
# mae_falls_with_m is the paper's headline for Table 2 -- performance improves as
# the cardinality constraint relaxes -- and the gaps are large enough that the
# indicator is not marginal.
#
# The two treated-effect numbers are recorded, not compared. The paper reports
# that a large beta favours the effect on the treated, and the measurement agrees
# in direction (1.50 against 1.62 at M = 25), but the gap is about 0.9 combined
# standard errors, so an assertion that one beats the other would be a coin flip
# at any M this case can afford. Pinning the levels catches a regression in
# either design; the ordering is left to the replication page to describe.
EXPECTED = {
    "tau_path_max_abs_dev": (0.0, 1.2),
    "mae_m1": (3.00, 0.45),
    "rmse_m1": (3.60, 0.45),
    "n_treated_m1": (1.0, 0.0),
    "mae_m3": (1.31, 0.30),
    "rmse_m3": (1.56, 0.30),
    "n_treated_m3": (3.0, 0.0),
    "mae_unconstrained": (0.74, 0.30),
    "rmse_unconstrained": (0.88, 0.30),
    "n_treated_unconstrained": (6.75, 0.60),
    "mae_falls_with_m": (1.0, 0.0),
    "treated_rmse_standard": (1.62, 0.40),
    "treated_rmse_weakly_targeted": (1.50, 0.40),
}
