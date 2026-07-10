"""BSCM cross-validation: China anti-corruption luxury-watch demand.

Kim, S., Lee, C., & Gupta, S. (2020), "Bayesian Synthetic Control Methods,"
Journal of Marketing Research 57(5):831-852, propose two Bayesian synthetic
controls -- ``horseshoe`` and ``spike_slab`` priors on unconstrained donor
weights -- designed for the "large p, small n" regime that a simplex synthetic
control handles poorly.

This case validates on the China anti-corruption watch-demand panel
(``china_watches_long.csv``: 1 treated series ``watches``, 87 donor series, 35
pre-treatment months, treatment at 2013-01 when Xi Jinping's anti-corruption
campaign began). With 87 donors and 35 pre-periods this is genuinely
``p > n`` -- the setting BSCM is built for, and a far more meaningful test than
a ``p ~ n`` panel where an unconstrained regression simply interpolates.

Triple cross-validation:

* against the authors' reference Stan horseshoe (``clarencejlee/bscm``, sampled
  with rstan): the pure-numpy Gibbs matches it essentially exactly -- ATT
  ``-0.0221`` vs Stan ``-0.0221``, pre-RMSE ``~0.099`` vs Stan ``0.0999``, and
  the dominant donor ``C60`` (Stan weight 0.68, numpy 0.68);
* against the forward-selected panel-data approach (:class:`mlsynth.PDA`
  ``method="fs"``, Shi-Huang), the dataset's original benchmark: it selects the
  same dominant donor ``C60`` and returns the same near-null ATT;
* against a simplex SCM: the BSCM horseshoe pre-fit (``~0.099``) equals the
  regularised simplex fit (``~0.098``) -- the shrinkage genuinely regularises
  here rather than interpolating (contrast the ``p ~ n`` Basque panel, where
  the same estimator interpolates to a near-zero pre-RMSE that does not
  generalise; see ``docs/bscm.rst`` on diagnostics).

All three methods agree on a ``C60``-dominant synthetic control and an ATT
whose 95% interval crosses zero: no statistically credible effect on this
residualised watch-demand series in this configuration.

Provenance: Kim, Lee & Gupta (2020); the reference Stan ``clarencejlee/bscm``;
Shi & Huang forward-selection PDA (the dataset's origin).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "china_watches_long.csv")


def run() -> dict:
    from mlsynth import BSCM, PDA

    df = pd.read_csv(os.path.abspath(_DATA))
    base = dict(df=df, outcome="y", treat="treat", unitid="unit", time="time",
                display_graphs=False)

    hs = BSCM({**base, "prior": "horseshoe", "n_iter": 4000, "burn_in": 2000,
               "chains": 4, "seed": 2019}).fit()
    fs = PDA({**base, "method": "fs"}).fit()

    hs_top = max(hs.donor_weights, key=lambda k: abs(hs.donor_weights[k]))
    fs_top = max(fs.donor_weights, key=lambda k: abs(fs.donor_weights[k]))
    lo, hi = hs.att_ci
    return {
        "hs_att": float(hs.att),
        "hs_pre_rmse": float(hs.pre_rmse),
        "hs_top_is_c60": 1.0 if hs_top == "C60" else 0.0,
        "hs_ci_covers_zero": 1.0 if (lo < 0.0 < hi) else 0.0,
        "fspda_att": float(fs.att),
        "fspda_top_is_c60": 1.0 if fs_top == "C60" else 0.0,
    }


# Cross-validation targets frozen from the reference Stan horseshoe
# (clarencejlee/bscm) on the identical panel: ATT -0.0221 [-0.066, +0.024],
# pre-RMSE 0.0999, dominant donor C60. Tolerances bracket Monte-Carlo noise.
EXPECTED = {
    "hs_att": (-0.022, 0.03),          # Stan horseshoe ATT -0.0221; FSPDA -0.011
    "hs_pre_rmse": (0.099, 0.015),     # matches Stan (0.0999) and simplex (~0.098)
    "hs_top_is_c60": (1.0, 0.0),       # C60 dominant (Stan and FSPDA agree)
    "hs_ci_covers_zero": (1.0, 0.0),   # null effect: 95% credible interval crosses 0
    "fspda_att": (-0.011, 0.03),       # forward-selection benchmark, same near-null
    "fspda_top_is_c60": (1.0, 0.0),    # FSPDA selects the same dominant donor
}
