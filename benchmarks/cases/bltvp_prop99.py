"""Path A: BL-TVP on Proposition 99 against Klinenberg (2023), Table 2.

Klinenberg (2023), "Synthetic Control with Time Varying Coefficients: A State
Space Approach with Bayesian Shrinkage", *Journal of Business & Economic
Statistics* 41(4), 1065-1076, proposes BL-TVP: a synthetic control whose donor
coefficients are decomposed into a time-invariant part and a random-walk part,
with independent Bayesian Lasso shrinkage on each, so the model chooses per
donor whether the relationship is time-varying, static, or absent.

Table 2 reports the average reduction in California's per-capita cigarette
sales for seven methods. This case pins the BL-TVP row against a port of the
paper's own sampler (``benchmarks/reference/bltvp_prop99/bltvp_reference.py``).

  =====================  =======  =========  ==========
  Quantity               paper    port       MCSE
  =====================  =======  =========  ==========
  Average decrease        17.7     17.78      0.11
  2.5th percentile       -16.1    -16.65      0.51
  97.5th percentile       51.7     52.28      0.48
  =====================  =======  =========  ==========

Measured over four chains of 10,000 draws (5000 burn-in), the settings the
author's ``3-BLTVP.R`` uses. All three agree within Monte Carlo error
(|z| <= 1.22), so the port reproduces the published row.

Two specification questions the paper alone does not settle
-----------------------------------------------------------
Both were resolved by running the alternatives, not by reading:

*The averaging window.* ``ca_results/ca_table.R`` averages over ``[t_0:T0]``
with ``t_0 = 19``, and period 19 is the last *pre-treatment* year (1988;
treatment lands in 1989, so the post window is 20 to 31). The published 17.7
therefore averages over one pre-treatment year plus the twelve post years.
Averaging the post window alone gives 19.25 (z = +13.3 against 17.7), so this
is not a rounding question -- the window is identifiable from the number, and
the paper's is the one including period 19.

*The intercept.* Model 1 carries an intercept as the ``(J+1)``-th regressor
(``x_{J+1,t} = 1``), but ``bitto_model_revised.R`` passes ``y ~ +-1+.``, which
suppresses it, and ``3-BLTVP.R`` adds no column of ones. Running both: the
point estimate does not care (17.78 without, 17.81 with), while the interval
bounds do, and the intercept-free fit -- the one the author's script actually
ran -- is the closer match. This case follows the script.

A caveat the fit statistic hides
--------------------------------
The pre-treatment RMSE is 0.036 against a California mean level of 116.2, which
is interpolation, not fit: 38 donors carry 76 free coefficients (a ``beta`` and
a ``sqrt(theta)`` each) against 19 pre-treatment observations. A near-zero
pre-period residual is what an overparameterized model does, so it is not on
its own evidence for the dynamics. The evidence for those is Table 1, where
BL-TVP is the only method reaching nominal coverage; that is a separate case.

The reported effect is not significant: the 95% band spans zero comfortably,
which is the paper's own reading (sec 6) and agrees with Arkhangelsky et al.
(2021) against ADH (2010).

Provenance
----------
* ``basedata/smoking_data.csv`` -- ADH (2010) Prop 99, 39 states x 31 years
  (1970-2000), California treated 1989, ``T0 = 19``, 38 donors. Ingested
  through ``datautils.dataprep``; matches the ``mixtape::smoking`` panel the
  author's script pivots by hand.
* Reference values are the paper's printed Table 2, p.1075.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parents[2] / "basedata"
_REF = Path(__file__).resolve().parents[1] / "reference" / "bltvp_prop99"

# Klinenberg (2023) Table 2, BL-TVP row (p.1075).
PAPER = {"avg": 17.7, "lo": -16.1, "hi": 51.7}

T0, T = 19, 31


def _panel():
    df = pd.read_csv(_BASE / "smoking_data.csv")
    df["treat"] = ((df["state"] == "California") & (df["year"] >= 1989)).astype(int)
    from mlsynth.utils.datautils import dataprep

    p = dataprep(df, "state", "year", "cigsale", "treat")
    return np.asarray(p["y"], float).ravel(), np.asarray(p["donor_matrix"], float)


def _fit(seed: int = 0):
    if str(_REF) not in sys.path:
        sys.path.insert(0, str(_REF))
    from bltvp_reference import bltvp

    y, Y0 = _panel()
    # No intercept, per the author's 3-BLTVP.R; 10000/5000 are its settings.
    pred = bltvp(y, Y0, T0, niter=10000, nburn=5000, seed=seed)
    return y, pred


def _table2(y, pred):
    """ca_table.R's statistic: per-period posterior mean / 2.5th / 97.5th of
    (counterfactual - observed), averaged over periods 19..31 inclusive."""
    d = pred - y[:, None]
    w = slice(18, 31)
    return (float(d.mean(axis=1)[w].mean()),
            float(np.quantile(d, 0.025, axis=1)[w].mean()),
            float(np.quantile(d, 0.975, axis=1)[w].mean()))


def run() -> dict:
    y, pred = _fit(seed=0)
    avg, lo, hi = _table2(y, pred)
    pre_rmse = float(np.sqrt(np.mean((pred[:T0].mean(axis=1) - y[:T0]) ** 2)))
    return {
        "avg_decrease_vs_paper": abs(avg - PAPER["avg"]),
        "lower_2p5_vs_paper": abs(lo - PAPER["lo"]),
        "upper_97p5_vs_paper": abs(hi - PAPER["hi"]),
        "effect_band_spans_zero": 1.0 if lo < 0.0 < hi else 0.0,
        "pre_period_rmse": pre_rmse,
    }


def comparison() -> dict:
    """mlsynth's BL-TVP port vs Klinenberg (2023) Table 2, BL-TVP row."""
    y, pred = _fit(seed=0)
    avg, lo, hi = _table2(y, pred)
    rows = [
        {"quantity": "Average decrease", "mlsynth": round(avg, 2), "reference": PAPER["avg"]},
        {"quantity": "2.5th percentile", "mlsynth": round(lo, 2), "reference": PAPER["lo"]},
        {"quantity": "97.5th percentile", "mlsynth": round(hi, 2), "reference": PAPER["hi"]},
    ]
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "BL-TVP (replication port)",
                         "config": {"niter": 10000, "nburn": 5000, "intercept": False,
                                    "window": "periods 19-31"}},
        "reference": {"impl": "Klinenberg (2023) Table 2, printed",
                      "version": "JBES 41(4), 1065-1076, p.1075"},
    }


# Single chain, so the tolerances carry one chain's Monte Carlo error: measured
# across-chain SDs are 0.22 (average), 1.01 and 0.95 (the two bounds). The
# tolerances are a little over 3 SD each, which leaves the case able to fail on
# the question it exists to settle -- averaging the post window alone moves the
# average to 19.25, a miss of 1.55.
EXPECTED = {
    "avg_decrease_vs_paper": (0.0, 1.0),
    "lower_2p5_vs_paper": (0.0, 3.5),
    "upper_97p5_vs_paper": (0.0, 3.5),
    "effect_band_spans_zero": (1.0, 0.0),
    "pre_period_rmse": (0.036, 0.05),
}
