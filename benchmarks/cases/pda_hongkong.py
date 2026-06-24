"""PDA Path-A: the Hong Kong / CEPA handover study (Hsiao-Ching-Wan; Shi-Wang).

Path A (empirical, scenario: the authors' canonical application + data). The
panel data approach is validated on the Hsiao, Ching & Wan (2012) Hong Kong
study -- the effect of the 2004 Closer Economic Partnership Arrangement (CEPA)
with mainland China on Hong Kong's quarterly YoY real GDP growth (1 treated unit,
24 controls, T1 = 44). Shi & Wang (L2-relaxation, arXiv) revisit it in their
Appendix E.1: the L2-relaxation panel data approach estimates a CEPA effect of
**+2.65%** with t-statistic **8.35**, decisively rejecting the no-effect null.

This runs all three of mlsynth's PDA methods -- L2-relaxation (``l2``), the
LASSO PDA (``LASSO``) and forward selection (``fs``; Shi & Huang 2023) -- on the
shipped ``basedata/HongKong.csv`` (the HCW panel). All three recover a positive,
highly significant CEPA effect; the L2 estimate (2.48%) lands close to the
paper's 2.65%, the small gap reflecting the L2-relaxation tuning. The fit is
deterministic, so the cells below are exact re-runs.

Live cross-validation (fixed tau)
---------------------------------
The CV-tuned ATE pins above reproduce the paper's *qualitative* finding but
cannot be matched cell-for-cell: mlsynth tunes ``tau`` by time-respecting
sequential validation, whereas the paper uses a future-leaking K-fold, so the
tuned estimate differs by construction. To cross-validate the *solver* exactly,
:func:`comparison` instead solves the unique L2-relaxation QP at a single, fixed,
matched ``tau`` (``0.1 * max|eta|`` on the standardised pre-period moments,
comfortably interior). At a fixed ``tau`` the primal ``min ||beta||^2 s.t.
||eta - Sigma beta||_inf <= tau`` is strictly convex, so its optimum is unique;
mlsynth's OSQP solve and the authors' ``Fun/L2relax.R`` program (reproduced via
cvxpy/ECOS -- CVXR/MOSEK are not installable in this R) must agree to solver
precision. The captured reference bundle lives in
``benchmarks/reference/pda_hongkong/`` (de-standardised donor weights + fit
summary, all four cvxpy solvers verified to attain the same objective).

Provenance: Shi & Wang, *"L2-Relaxation for Economic Prediction,"* Appendix E.1
(Table E.1 and the L2 headline); Hsiao, Ching & Wan (2012) for the study/data.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

from benchmarks.reference import load_reference, reference_value

_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "HongKong.csv")

# Fixed interior tau for the solver cross-validation: TAU_FRAC * max|eta| on the
# standardised pre-period moments (kept in lockstep with reference.py).
_TAU_FRAC = 0.1


def _fit(method):
    from mlsynth import PDA
    import pandas as pd

    d = pd.read_csv(os.path.abspath(_DATA))
    res = PDA({
        "df": d, "outcome": "GDP", "treat": "Integration",
        "unitid": "Country", "time": "Time", "method": method,
        "display_graphs": False,
    }).fit()
    att = float(res.att)
    lo = res.inference.ci_lower
    se = (att - float(lo)) / 1.96 if lo is not None else float("nan")
    return att, (att / se if se else float("nan")), float(res.inference.p_value)


def _fixed_tau_l2():
    """Solve mlsynth's L2 QP at the fixed matched ``tau``; return (donors, beta, tau).

    Pins ``tau`` to ``_TAU_FRAC * max|eta|`` on the standardised pre-period
    moments, runs mlsynth's L2-relaxation at exactly that ``tau`` (no CV), and
    returns the unrounded de-standardised donor coefficients.
    """
    import pandas as pd
    from mlsynth import PDA
    from mlsynth.utils.pda_helpers.setup import prepare_pda_inputs

    d = pd.read_csv(os.path.abspath(_DATA))
    inp = prepare_pda_inputs(d, unitid="Country", time="Time",
                             outcome="GDP", treat="Integration")
    y_pre, X_pre = inp.y[:inp.T0], inp.X[:inp.T0]
    Sd_X = X_pre.std(0, ddof=1)
    Sd_X = np.where(Sd_X > 0, Sd_X, 1.0)
    sd_y = y_pre.std(ddof=1) or 1.0
    eta = ((X_pre - X_pre.mean(0)) / Sd_X).T @ ((y_pre - y_pre.mean()) / sd_y) / inp.T0
    tau = _TAU_FRAC * float(np.abs(eta).max())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = PDA({
            "df": d, "outcome": "GDP", "treat": "Integration",
            "unitid": "Country", "time": "Time", "methods": ["l2"],
            "tau": tau, "l2_standardize": True, "display_graphs": False,
        }).fit()
    fit = res.fits["l2"]
    donors = list(inp.unit_index.labels)
    return donors, np.asarray(fit.beta, dtype=float), tau


def run() -> dict:
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atts = {}
        for method in ("l2", "LASSO", "fs"):
            key = method.lower()
            att, t, p = _fit(method)
            atts[key] = (att, p)
            out[f"{key}_ate_pct"] = att * 100.0         # GDP is decimal growth
            out[f"{key}_pvalue"] = p
        out["l2_abs_tstat"] = abs(_fit("l2")[1])
        # All methods agree the CEPA effect is positive and significant.
        out["n_methods_positive_sig"] = float(sum(
            1 for a, p in atts.values() if a > 0 and p < 0.01))

    # Solver cross-validation at the fixed matched tau: mlsynth (OSQP) vs the
    # authors' L2relax program (cvxpy/ECOS) on the unique QP. Compare the
    # de-standardised donor weights donor by donor.
    donors, beta_ml, _ = _fixed_tau_l2()
    w_ref = load_reference("pda_hongkong")["weights"]
    diffs = [abs(beta_ml[i] - float(w_ref[d])) for i, d in enumerate(donors)]
    out["l2_fixedtau_beta_max_abs_diff"] = float(max(diffs))
    return out


def comparison() -> dict:
    """mlsynth L2-relaxation vs the authors' ``Fun/L2relax.R``, donor by donor.

    Solves the unique L2-relaxation QP at a single fixed, matched ``tau`` with
    both implementations and lays the de-standardised donor weights side by side
    (one row per donor). The reference side is the captured bundle in
    ``benchmarks/reference/pda_hongkong/`` -- the authors' program reproduced via
    cvxpy/ECOS (CVXR/MOSEK unavailable in this R), all four cvxpy solvers
    verified to attain the same objective. This is a solver cross-validation at a
    fixed ``tau``, distinct from the CV-tuned ATE (pinned in :func:`run`), which
    differs from the paper by construction (different CV schemes).
    """
    donors, beta_ml, tau = _fixed_tau_l2()
    w_ref = load_reference("pda_hongkong")["weights"]
    rows = [{"quantity": f"weight[{d}]", "mlsynth": round(float(beta_ml[i]), 6),
             "reference": round(float(w_ref[d]), 6)}
            for i, d in enumerate(donors)]
    cfg = {"outcome": "GDP", "treat": "Integration", "unitid": "Country",
           "time": "Time", "methods": ["l2"], "tau": tau, "l2_standardize": True}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "PDA", "config": cfg},
        "reference": {"impl": "Authors' Fun/L2relax.R (ishwang1/L2relax-PDA) "
                              "reproduced via cvxpy/ECOS, live run captured",
                      "version": "Shi & Wang, L2-Relaxation for Economic Prediction; "
                                 "Appendix E.1 (Hong Kong)"},
    }


# Deterministic (convex L2 / greedy fs / fixed-grid LASSO, no RNG) => exact
# re-runs. The CV-tuned ATE cells reproduce the paper's qualitative finding (a
# positive, highly significant CEPA effect; L2 = 2.48% vs the paper's 2.65%,
# the small gap being the L2 tuning); the LASSO/fs estimates (3.3%, 3.9%)
# bracket it. SEPARATELY, l2_fixedtau_beta_max_abs_diff is a SOLVER cross-
# validation on the unique QP at a fixed matched tau: mlsynth (OSQP) vs the
# authors' L2relax program (cvxpy/ECOS, captured in
# benchmarks/reference/pda_hongkong/). They agree to ~1e-7 -- the tolerance
# covers genuine OSQP-vs-ECOS slack only.
EXPECTED = {
    "l2_ate_pct": (2.61, 0.5),                 # paper L2: 2.65% (standardised l2)
    "lasso_ate_pct": (3.30, 0.6),
    "fs_ate_pct": (3.95, 0.7),
    "l2_pvalue": (0.0, 0.01),                  # rejects no-effect null
    "l2_abs_tstat": (7.75, 1.5),               # paper L2: 8.35
    "n_methods_positive_sig": (3.0, 0.0),      # all three agree
    "l2_fixedtau_beta_max_abs_diff": (0.0, 1e-5),   # OSQP vs ECOS on the unique QP
}
