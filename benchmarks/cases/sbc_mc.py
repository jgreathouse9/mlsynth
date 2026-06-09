"""Path B benchmark: SBC vs conventional SC on nonstationary data (Shi-Xi-Xie 2025).

Reproduces the headline of the paper's simulation study (Section 4, Table 1):
on nonstationary panels SBC -- which matches on the *cyclical* component after a
Hamilton trend projection -- has lower post-treatment MSE than a conventional
synthetic control on raw levels, except in the regime conventional SC is actually
built for (partial cointegration), where it is merely competitive. The reported
statistic is the post-MSE ratio ``MSE(SBC) / MSE(conventional SC)`` against the
treated unit's true (untreated) path; below 1 means SBC wins.

* Model 1 -- independent random walks (the spurious-regression regime): SBC wins
  big.
* Model 2 -- idiosyncratic unit roots + common stationary AR factors: SBC wins.
* Model 3 -- partial cointegration (half the units share RW factors in levels):
  conventional SC is closest, so the ratio is nearest 1.

Provenance
----------
* DGP: :func:`mlsynth.utils.sbc_helpers.simulation.simulate_shi_xi_xie`
  (the paper's Model 1/2/3, NU=12, h=2, p=2, phi=0.5).
* Headline: Shi-Xi-Xie (2025) Table 1 -- every post-MSE ratio < 1, lowest for
  the spurious-regression models and highest (near 1) under cointegration. The
  paper uses 10,000 reps; we use 60 (tolerances absorb the MC gap).
"""
from __future__ import annotations

import warnings

import numpy as np

NU, H, P, T0 = 12, 2, 2, 100
REPS = 60


def _ratio(model: int) -> float:
    import cvxpy as cp
    import pandas as pd

    from mlsynth import SBC
    from mlsynth.utils.sbc_helpers.simulation import simulate_shi_xi_xie

    def conv_sc(y_pre, X_pre, X_full):
        w = cp.Variable(X_pre.shape[1])
        cp.Problem(cp.Minimize(cp.sum_squares(y_pre - X_pre @ w)),
                   [w >= 0, cp.sum(w) == 1]).solve(solver="CLARABEL")
        return X_full @ w.value

    rng = np.random.default_rng(model)               # paired seed per model
    num = den = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(REPS):
            Y = simulate_shi_xi_xie(model, T0, H=H, NU=NU, rng=rng)
            T = T0 + H
            df = pd.DataFrame([{"unit": i, "time": t, "y": float(Y[i, t]),
                                "treat": int(i == 0 and t >= T0)}
                               for i in range(NU) for t in range(T)])
            cf = np.asarray(SBC({"df": df, "outcome": "y", "treat": "treat",
                                 "unitid": "unit", "time": "time", "h": H, "p": P,
                                 "weights_mode": "simplex",
                                 "display_graphs": False}).fit().counterfactual_full)
            y1, X = Y[0], Y[1:].T
            cf_sc = conv_sc(y1[:T0], X[:T0], X)
            po = slice(T0, T0 + H)
            num += float(np.sum((cf[po] - y1[po]) ** 2))
            den += float(np.sum((cf_sc[po] - y1[po]) ** 2))
    return num / den


def run() -> dict:
    r = {m: _ratio(m) for m in (1, 2, 3)}
    return {
        "ratio_model1": r[1],
        "ratio_model2": r[2],
        "ratio_model3": r[3],
        # 1.0 iff SBC beats conventional SC in every model (all ratios < 1).
        "sbc_beats_sc_all": float(max(r.values()) < 1.0),
        # 1.0 iff conventional SC is closest under cointegration (Model 3 highest).
        "model3_closest_to_sc": float(r[3] > r[1] and r[3] > r[2]),
    }


# Deterministic (paired seed per model). Binding facts: every post-MSE ratio < 1
# (SBC dominates conventional SC on nonstationary data) and Model 3's ratio is
# the highest (SC competitive under cointegration). Per-model ratios carry MC
# bands (60 reps vs the paper's 10,000).
EXPECTED = {
    "ratio_model1": (0.048, 0.08),
    "ratio_model2": (0.038, 0.08),
    "ratio_model3": (0.696, 0.22),
    "sbc_beats_sc_all": (1.0, 0.0),
    "model3_closest_to_sc": (1.0, 0.0),
}
