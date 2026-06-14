"""PDA Path-B: Jiang, Li, Shen & Zhou (2025) prediction-interval coverage (Tables 2-5).

Reproduces, through the public PDA API (``PDA(prediction_intervals=True)``), the
defining finding of the paper's Monte Carlo study: the bootstrap prediction
intervals (equal-tailed EQ and symmetric SY) of the per-period treatment effect
have coverage near the nominal 95%, while the normal-quantile intervals (NE) of
Fujiki & Hsiao under-cover -- markedly so under non-normal idiosyncratic errors.

The DGP is the paper's Setup 1 (their Eq. 4.1): a single treated unit, ``N0``
controls, a sparse three-factor model

    y_{i,t} = b_i' f_t + u_{i,t},   f_t ~ iid N(0, I_3),

with nonzero loadings on only the first ``max(N0/10, 5)`` controls (the rest are
pure-noise donors) and the treated unit; loadings are drawn once and held fixed
across replications (the paper's design). The idiosyncratic errors here are the
paper's hardest column -- centred exponential ``Exp(1) - 1`` -- so the NE
under-coverage is unambiguous. The treatment effect is the constant ``1`` (the
paper's value); each cell reports the share of replications whose 95% interval
for ``Delta_{T0+1}`` covers it.

We pin two estimators that differ in how they studentize:

* ``fs`` -- forward selection then OLS, exactly the post-selection-OLS structure
  the paper's Algorithm 2.1 assumes, so its HAC sandwich applies verbatim. EQ/SY
  coverage lands near nominal (mildly conservative at this sample size); NE
  under-covers heavily.
* ``LASSO`` -- the Li & Bell shrinkage estimator. Jiang's interval still applies
  via the sandwich on the selected support; EQ/SY coverage is near nominal
  (mildly liberal, the shrinkage bias the OLS sandwich does not model).

Path B (scenario 1): the DGP is re-implemented from the paper; the case asserts
the coverage geometry (EQ/SY near nominal, NE below, bootstrap beats normal), not
the exact table cells -- mlsynth uses a standard bootstrap rather than the
paper's warp-speed pooling, and its own LASSO/forward-selection tuning.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

_T0, _N0, _Q = 30, 40, 3


def _centered_exp(rng, n):
    """Centred exponential idiosyncratic error: ``Exp(1) - 1`` (mean 0, var 1)."""
    return rng.exponential(1.0, n) - 1.0


def _loadings(N0, seed=7):
    """Sparse factor loadings, drawn once: treated + first ``max(N0/10,5)`` controls."""
    rng = np.random.default_rng(seed)
    K = max(N0 // 10, 5)
    B = np.zeros((N0 + 1, _Q))
    B[0] = rng.standard_normal(_Q)                       # treated unit (index 0)
    for i in range(1, K + 1):
        B[i] = rng.standard_normal(_Q)                   # relevant controls
    return B                                             # rest are pure-noise donors


def _panel(seed, loadings, T0=_T0, N0=_N0, effect=1.0):
    rng = np.random.default_rng(seed)
    T = T0 + 5
    f = rng.standard_normal((T, _Q))
    rows = []
    for i in range(N0 + 1):
        y0 = loadings[i] @ f.T + _centered_exp(rng, T)
        for t in range(T):
            treated_post = (i == 0 and t >= T0)
            rows.append({"ID": i, "Period": t,
                         "Value": float(y0[t] + (effect if treated_post else 0.0)),
                         "IsTreated": int(treated_post)})
    return pd.DataFrame(rows)


def _coverage(method, reps, n_boot, *, effect=1.0):
    from mlsynth import PDA

    key = {"fs": "fs", "LASSO": "lasso"}[method]
    loadings = _loadings(_N0)
    z = float(norm.ppf(0.975))
    eq = sy = ne = 0
    for r in range(reps):
        df = _panel(2000 + r, loadings, effect=effect)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = PDA({"df": df, "outcome": "Value", "treat": "IsTreated",
                       "unitid": "ID", "time": "Period", "method": method,
                       "prediction_intervals": True, "pi_n_boot": n_boot,
                       "pi_seed": r, "display_graphs": False}).fit()
        pi = res.fits[key].prediction_intervals
        e, se = pi["effect"], pi["se"][0]
        eq += e["eq_lower"][0] <= effect <= e["eq_upper"][0]
        sy += e["sy_lower"][0] <= effect <= e["sy_upper"][0]
        pt = e["point"][0]
        ne += (pt - z * se) <= effect <= (pt + z * se)
    return eq / reps, sy / reps, ne / reps


def run() -> dict:
    fs_eq, fs_sy, fs_ne = _coverage("fs", 150, 149)
    la_eq, la_sy, la_ne = _coverage("LASSO", 200, 199)
    return {
        "fs_eq": fs_eq, "fs_sy": fs_sy, "fs_ne": fs_ne,
        "fs_boot_beats_normal": float(fs_sy - fs_ne),
        "lasso_eq": la_eq, "lasso_sy": la_sy,
    }


# Deterministic (fixed DGP and bootstrap seeds). Values track Jiang et al.
# Tables 2-5: EQ/SY near nominal 0.95, NE well below; tolerances absorb
# bootstrap/tuning differences and any solver/RNG drift.
EXPECTED = {
    "fs_eq": (0.980, 0.07),                 # near nominal (mildly conservative)
    "fs_sy": (0.980, 0.07),                 # near nominal
    "fs_ne": (0.773, 0.09),                 # normal-quantile under-covers
    "fs_boot_beats_normal": (0.207, 0.12),  # SY coverage exceeds NE by a wide margin
    "lasso_eq": (0.910, 0.07),              # near nominal
    "lasso_sy": (0.930, 0.06),              # near nominal
}
