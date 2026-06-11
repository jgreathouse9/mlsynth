"""MSQRT Path-B: Shen, Song & Abadie (2025) Section-6 simulation.

Validates mlsynth's ``MSQRT`` (multivariate square-root Lasso synthetic control)
against the data-generating process of Shen, Song & Abadie (2025), *"Efficiently
Learning Synthetic Control Models for High-dimensional Disaggregated Data"*
(arXiv 2510.22828), Section 6. With a true ATT of zero, the harness
(:func:`mlsynth.utils.msqrt_helpers.replication.run_msqrt_simulation`) reports, per
number of treated units ``m`` and DGP setting:

* the **ATT bias** (the first post-period treated gap; the truth is 0, so the gap
  *is* the bias), and
* the **RMSE** of the imputed counterfactual ``Y(0)`` over the post window.

The paper's headline reproduces qualitatively: the estimator is **unbiased**
(bias centred at zero) and the RMSE stays **near the noise floor** and roughly
flat across ``m`` -- the property that makes the high-dimensional, many-treated
regime work.

  ===========  ==================  ==================
  setting      bias (m=20 / 50)    RMSE (m=20 / 50)
  ===========  ==================  ==================
  1            -0.07 / +0.07       0.35 / 0.38
  2            +0.07 / +0.07       0.28 / 0.20
  ===========  ==================  ==================

This runs a **reduced regime** (``n = 80`` donors, ``T0 = 50``, 12 reps, the
penalty fixed once) so it finishes in ~2 min; the paper's exact ``n = 400``,
500-replication study pins the RMSE level at ~0.72. The case asserts the
*reproducible* facts -- the estimator is unbiased within Monte-Carlo noise and
the RMSE sits near the ``sigma = 0.5`` noise floor (not blowing up with ``m``) --
not the paper's exact large-``n`` cells. Deterministic (seeded).
"""
from __future__ import annotations

import warnings

import numpy as np


def run() -> dict:
    from mlsynth.utils.msqrt_helpers.replication import (
        SimConfig, run_msqrt_simulation)

    cfg = SimConfig(n=80, T0=50, T1=10, s=200, sigma=0.5,
                    m_grid=[20, 50], n_reps=12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = run_msqrt_simulation(cfg, settings=(1, 2), seed=0,
                                   lambda_override=0.5, verbose=False)

    biases, rmses = [], []
    for setting in (1, 2):
        biases += [r["bias_mean"] for r in out[setting]]
        rmses += [r["rmse_mean"] for r in out[setting]]
    biases = np.asarray(biases)
    rmses = np.asarray(rmses)
    return {
        "abs_bias_max": float(np.max(np.abs(biases))),
        "unbiased": float(np.all(np.abs(biases) < 0.2)),
        "rmse_min": float(rmses.min()),
        "rmse_max": float(rmses.max()),
        "rmse_near_noise_floor": float(np.all((rmses > 0.05) & (rmses < 0.8))),
    }


# Deterministic (seeded). Tolerances absorb the Monte-Carlo noise at 12 reps.
# Reproduces Shen-Song-Abadie's qualitative Section-6 finding: the square-root
# Lasso SC is unbiased (bias centred at zero, |bias| <= ~0.1 here vs the paper's
# ~0) and the counterfactual RMSE stays near the sigma=0.5 noise floor and
# roughly flat in m (the paper's exact n=400 level is ~0.72).
EXPECTED = {
    "abs_bias_max": (0.074, 0.12),
    "unbiased": (1.0, 0.0),
    "rmse_min": (0.20, 0.15),
    "rmse_max": (0.38, 0.20),
    "rmse_near_noise_floor": (1.0, 0.0),
}
