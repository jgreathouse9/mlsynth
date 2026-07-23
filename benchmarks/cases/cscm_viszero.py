"""Cross-validation benchmark: CSCM (Bonander 2021) on the Vision Zero panel.

Reproduces the flexible count synthetic control on Sweden's Vision Zero road-
safety policy (``basedata/viszero.csv``: Sweden + 9 European donors, road-death
rate, treatment index 27) against a live captured run of the authors' own R
package (``CSCM_helper_functions.R`` from OSF osf.io/uvt5p), executed with only
two exact substitutions for sandbox-missing packages -- ``osqp`` -> quadprog
(identical simplex optimum) and ``Synth::dataprep`` -> a hand-built matrix
constructor.

Cross-validated cell by cell in the port review: the classic SCM warm-start and
the penalized relaxation reproduce R to ~1e-11 given identical V/lambda, and the
Poisson-ridge V matches glmnet to correlation ~0.98. On this panel glmnet's V
collapses to near-uniform, so ``v_method='uniform'`` (deterministic, fast) is
the faithful reproduction; the R reference numbers below use glmnet's V. The
tolerances absorb the glmnet-vs-uniform V difference (~1%).

R ground truth (captured run):
    classic SCM weights  = [0,0,0,0,1,0,0,0,0]  (100% Finland, ID 9)
    CSCM weights sum     = 0.486  (adding-up relaxed, extrapolates below simplex)
    full-sample RR       = 1.037  (sum obs_post / sum cf_post)
    cross-fitted RR (K=2) = 1.065,  95% CI [0.644, 1.760]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mlsynth import CSCM

_DATA = Path(__file__).resolve().parents[2] / "basedata" / "viszero.csv"
_DONORS = [2, 3, 5, 7, 9, 10, 13, 14, 16]

# Pinned from the captured R run (glmnet V); see module docstring.
_R = {
    "scm_finland": 1.0,
    "cscm_sum_weights": 0.486,
    "full_sample_rr": 1.037,
    "crossfit_rr": 1.065,
}


def _fit():
    df = pd.read_csv(_DATA)
    df = df[df["ID"].isin([25] + _DONORS)].copy()
    df["treated"] = ((df["ID"] == 25) & (df["TIME"] >= 1996)).astype(int)
    return CSCM({
        "df": df, "outcome": "deathrate_mln", "treat": "treated",
        "unitid": "ID", "time": "TIME", "display_graphs": False,
        "K": 2, "v_method": "uniform",
    }).fit()


def _quantities(res) -> dict:
    ts = res.time_series
    obs = np.asarray(ts.observed_outcome, dtype=float)
    cf = np.asarray(ts.counterfactual_outcome, dtype=float)
    T0 = 26                                        # 1970-1995 pre; 1996-2015 post (T1=20)
    full_rr = float(obs[T0:].sum() / cf[T0:].sum())
    return {
        "scm_finland": float(res.additional_outputs["scm_weights"]["9"]),
        "cscm_sum_weights": float(sum(res.weights.donor_weights.values())),
        "full_sample_rr": full_rr,
        "crossfit_rr": float(res.effects.additional_effects["rate_ratio"]),
    }


def run() -> dict:
    return _quantities(_fit())


def comparison() -> dict:
    q = _quantities(_fit())
    rows = [
        {"quantity": k, "mlsynth": round(q[k], 4), "reference": _R[k]}
        for k in ("scm_finland", "cscm_sum_weights", "full_sample_rr", "crossfit_rr")
    ]
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "CSCM",
                         "config": {"outcome": "deathrate_mln", "treat": "treated",
                                    "unitid": "ID", "time": "TIME", "K": 2,
                                    "v_method": "uniform"}},
        "reference": {"impl": "Bonander CSCM_helper_functions.R (OSF osf.io/uvt5p, live run, captured)",
                      "version": "Bonander (2021), Epidemiology 32(4):e18-e19"},
    }


EXPECTED = {
    "scm_finland": (_R["scm_finland"], 1e-3),        # SCM concentrates on Finland
    "cscm_sum_weights": (_R["cscm_sum_weights"], 0.03),  # relaxed below simplex (V-diff)
    "full_sample_rr": (_R["full_sample_rr"], 0.03),  # V + lambda-CV grid difference (~1.5%)
    "crossfit_rr": (_R["crossfit_rr"], 0.02),        # cross-fitted rate ratio (matches R to ~1%)
}
