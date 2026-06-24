"""PDA Path-A: Shi & Wang (2024) China PPI / real-estate regulation (L2-relaxation).

Reproduces the headline single-treated-unit application of the L2-relaxation PDA
paper (Section 6.1): the effect of China's 2020-21 real-estate tightening (the
"Three Red Lines" policy) on China's **monthly YoY PPI growth rate**, using
``l2`` against ``basedata/china_ppi_long.csv`` (treated ``China`` PPI + 64
country-control PPI series; pre-period before Aug 2020, T1=115; post-period from
Jun 2021, T2=43; the Aug-2020-May-2021 policy-rollout gap is excluded).

The series are **standardised** before solving (``l2_standardize=True``, the
default, matching the authors' released ``L2relax``). mlsynth recovers the
paper's finding -- a large, significant negative PPI effect:

  =====================  ===============  ========================
  Quantity               mlsynth l2       Shi & Wang (Sec. 6.1)
  =====================  ===============  ========================
  monthly ATE            -5.95%           -6.40%
  t-statistic            -4.48            -3.61  (p = 0.0003)
  =====================  ===============  ========================

Two deliberate method differences, both documented:

* **Cross-validation.** mlsynth tunes ``tau`` by **time-respecting** sequential
  out-of-sample validation (fit on the earlier window, validate on the recent
  tail). The authors' ``L2relax.CV`` uses a 5-block K-fold that trains on both
  past *and* future of each validation block -- it leaks future information into
  the counterfactual fit. The sequential CV selects slightly less regularisation,
  giving -5.95% rather than the paper's -6.40% (with their leaky CV mlsynth
  reproduces -6.40% exactly). The sign, magnitude, and 1%-significance agree.
* **Long-run variance.** mlsynth's Newey-West bandwidth (``newey_west_lag``)
  differs from R ``sandwich``'s automatic one, so the t-statistic (-4.48) differs
  from the paper's -3.61; both reject the zero-ATE null well below 1%.

Live cross-validation (fixed tau)
---------------------------------
Because the CV-tuned ATE differs from the paper by construction (the two CV
schemes differ), the *solver* is cross-validated separately on the unique QP at
a single fixed, matched ``tau`` (``0.1 * max|eta|`` on the standardised
pre-period moments, comfortably interior). At a fixed ``tau`` the primal
``min ||beta||^2 s.t. ||eta - Sigma beta||_inf <= tau`` is strictly convex, so
its optimum is unique; mlsynth's OSQP solve and the authors' ``Fun/L2relax.R``
program (reproduced via cvxpy/ECOS -- CVXR/MOSEK are not installable in this R)
agree to solver precision. The captured reference bundle is in
``benchmarks/reference/pda_ppi/`` (de-standardised donor weights + fit summary,
all four cvxpy solvers verified to attain the same objective).

Path A (scenario 3): the data and method are the authors'; the estimate is
deterministic (convex solve + deterministic CV).
"""
from __future__ import annotations

import os
import warnings

import numpy as np

from benchmarks.reference import load_reference

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "china_ppi_long.csv")

# Fixed interior tau for the solver cross-validation (in lockstep with reference.py).
_TAU_FRAC = 0.1


def _fixed_tau_l2():
    """Solve mlsynth's L2 QP at the fixed matched ``tau``; return (donors, beta, tau)."""
    import pandas as pd
    from mlsynth import PDA
    from mlsynth.utils.pda_helpers.setup import prepare_pda_inputs

    d = pd.read_csv(os.path.abspath(_DATA))
    inp = prepare_pda_inputs(d, unitid="unit", time="time",
                             outcome="y", treat="treat")
    y_pre, X_pre = inp.y[:inp.T0], inp.X[:inp.T0]
    Sd_X = X_pre.std(0, ddof=1)
    Sd_X = np.where(Sd_X > 0, Sd_X, 1.0)
    sd_y = y_pre.std(ddof=1) or 1.0
    eta = ((X_pre - X_pre.mean(0)) / Sd_X).T @ ((y_pre - y_pre.mean()) / sd_y) / inp.T0
    tau = _TAU_FRAC * float(np.abs(eta).max())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = PDA({
            "df": d, "outcome": "y", "treat": "treat", "unitid": "unit",
            "time": "time", "methods": ["l2"], "tau": tau,
            "l2_standardize": True, "display_graphs": False,
        }).fit()
    fit = res.fits["l2"]
    donors = list(inp.unit_index.labels)
    return donors, np.asarray(fit.beta, dtype=float), tau


def run() -> dict:
    import pandas as pd
    from mlsynth import PDA

    df = pd.read_csv(os.path.abspath(_DATA))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = PDA({
            "df": df, "outcome": "y", "treat": "treat",
            "unitid": "unit", "time": "time",
            "methods": ["l2"], "l2_standardize": True,
            "display_graphs": False,
        }).fit()

    fit = res.fits["l2"]

    # Solver cross-validation at the fixed matched tau: mlsynth (OSQP) vs the
    # authors' L2relax program (cvxpy/ECOS) on the unique QP, donor by donor.
    donors, beta_ml, _ = _fixed_tau_l2()
    w_ref = load_reference("pda_ppi")["weights"]
    diffs = [abs(beta_ml[i] - float(w_ref[d])) for i, d in enumerate(donors)]

    return {
        "l2_ate_pct": float(fit.att),
        "l2_abs_tstat": float(abs(fit.att / fit.att_se)),
        "l2_significant_5pct": 1.0 if fit.p_value < 0.05 else 0.0,
        "l2_fixedtau_beta_max_abs_diff": float(max(diffs)),
    }


def comparison() -> dict:
    """mlsynth L2-relaxation vs the authors' ``Fun/L2relax.R``, donor by donor.

    Solves the unique L2-relaxation QP at a single fixed, matched ``tau`` with
    both implementations and lays the de-standardised donor weights side by side
    (one row per donor). The reference side is the captured bundle in
    ``benchmarks/reference/pda_ppi/`` -- the authors' program reproduced via
    cvxpy/ECOS (CVXR/MOSEK unavailable in this R), all four cvxpy solvers
    verified to attain the same objective. Distinct from the CV-tuned ATE (pinned
    in :func:`run`), which differs from the paper by construction.
    """
    donors, beta_ml, tau = _fixed_tau_l2()
    w_ref = load_reference("pda_ppi")["weights"]
    rows = [{"quantity": f"weight[{d}]", "mlsynth": round(float(beta_ml[i]), 6),
             "reference": round(float(w_ref[d]), 6)}
            for i, d in enumerate(donors)]
    cfg = {"outcome": "y", "treat": "treat", "unitid": "unit", "time": "time",
           "methods": ["l2"], "tau": tau, "l2_standardize": True}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "PDA", "config": cfg},
        "reference": {"impl": "Authors' Fun/L2relax.R (ishwang1/L2relax-PDA) "
                              "reproduced via cvxpy/ECOS, live run captured",
                      "version": "Shi & Wang, L2-Relaxation for Economic Prediction; "
                                 "Section 6.1 (China PPI)"},
    }


# Deterministic (convex L2-relaxation solve + deterministic sequential CV). The
# ATE is pinned to mlsynth's time-respecting-CV value (-5.95%, vs the paper's
# -6.40% under its future-leaking K-fold); the tolerance keeps it close to the
# paper while guarding regressions. The t-stat is pinned loosely (the NW
# bandwidth differs from sandwich's) but must stay well past 5% significance.
# SEPARATELY, l2_fixedtau_beta_max_abs_diff is a SOLVER cross-validation on the
# unique QP at a fixed matched tau: mlsynth (OSQP) vs the authors' L2relax
# program (cvxpy/ECOS, captured in benchmarks/reference/pda_ppi/). They agree to
# ~1e-7 -- the tolerance covers genuine OSQP-vs-ECOS slack only.
EXPECTED = {
    "l2_ate_pct": (-5.948, 0.6),          # paper -6.40 (their leaky 5-block CV)
    "l2_abs_tstat": (4.48, 1.2),          # paper 3.61 (NW-bandwidth difference)
    "l2_significant_5pct": (1.0, 0.0),    # paper p = 0.0003
    "l2_fixedtau_beta_max_abs_diff": (0.0, 1e-5),   # OSQP vs ECOS on the unique QP
}
