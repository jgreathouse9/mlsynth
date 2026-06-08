"""Path B benchmark: RSC training error approximates generalization error.

Reproduces the central finding of the synthetic study in Amjad, Shah & Shen
(2018), *Robust Synthetic Control* (JMLR 19:1-51), Section 5.3, Table 1: for the
PCR / RSC estimator on a low-rank latent-variable panel, the **pre-intervention
MSE (training error) closely tracks the post-intervention MSE (generalization
error)** across noise levels -- so the in-sample pre-fit is an honest predictor
of out-of-sample forecast accuracy. Both errors are measured against the *true*
(noise-free) mean, which the synthetic DGP makes available.

This exercises mlsynth's PCR-SC path through the public ``CLUSTERSC().fit()`` API
(``clustering=False`` -- the Amjad-Shah-Shen RSC: HSVT-denoise the donors, then
OLS), reading the counterfactual via the standardized ``res.counterfactual``
accessor. ``compute_shen_ci=False`` skips the per-period CIs this
prediction-only study does not need.

Provenance
----------
* DGP: :func:`mlsynth.utils.clustersc_helpers.simulation.simulate_rsc_panel`
  -- the latent-variable model of RSC Section 5.3 (``N=100``, ``T=2000``,
  treatment at ``t=1600``; seasonal + θ-scaled-trend mean, ≈ rank 3).
* Headline: RSC Table 1 reports training ≈ generalization error at every noise
  level (e.g. 0.48 / 0.53 at σ=3.1 down to 0.0005 / 0.0006 at σ=0.1). The
  *ratio* gen/train ≈ 1 is the paper's stated finding; absolute magnitudes
  depend on the (underspecified) truncation rank, so we pin the ratio and the
  noise-monotonicity rather than Table 1's exact cells.
"""
from __future__ import annotations

from typing import List

import numpy as np

from mlsynth.utils.clustersc_helpers.simulation import simulate_rsc_panel

RANK = 3            # true signal rank of the DGP
SEED = 0
N_TARGETS = 5       # placebo targets (units 0..4); DGP is deterministic given SEED
NOISE_LEVELS = [3.1, 1.3, 0.4]


def _train_gen_error(noise: float):
    """Mean (over placebo targets) pre/post MSE of the RSC estimate vs truth."""
    import pandas as pd

    from mlsynth import CLUSTERSC

    panel = simulate_rsc_panel(N=100, T=2000, T0=1600, noise=noise, seed=SEED)
    M, X, T0 = panel.means, panel.observed, panel.T0
    wide = pd.DataFrame(X.T)                          # (T, N); columns are unit ids
    wide.columns = range(X.shape[0])
    train, gen = [], []
    for ti in range(N_TARGETS):
        long = (wide.reset_index()
                .melt(id_vars="index", var_name="unit", value_name="y")
                .rename(columns={"index": "time"}))
        long["treat"] = ((long["unit"] == ti) & (long["time"] >= T0)).astype(int)
        res = CLUSTERSC({
            "df": long, "outcome": "y", "treat": "treat",
            "unitid": "unit", "time": "time",
            "method": "pcr", "clustering": False, "pcr_objective": "OLS",
            "rank": RANK, "rank_method": "fixed", "project_denoised": True,
            "compute_shen_ci": False, "display_graphs": False,
        }).fit()
        cf = np.asarray(res.counterfactual).ravel()
        truth = M[ti]
        train.append(float(np.mean((cf[:T0] - truth[:T0]) ** 2)))
        gen.append(float(np.mean((cf[T0:] - truth[T0:]) ** 2)))
    return float(np.mean(train)), float(np.mean(gen))


def run() -> dict:
    out: dict = {}
    ratios: List[float] = []
    gens: List[float] = []
    for noise in NOISE_LEVELS:
        train, gen = _train_gen_error(noise)
        tag = f"{int(noise * 100):03d}"
        out[f"gen_err_noise{tag}"] = gen
        out[f"ratio_noise{tag}"] = gen / train
        ratios.append(gen / train)
        gens.append(gen)
    # 1.0 iff training error approximates generalization error at every noise
    # level (gen/train within [0.8, 1.4] -- the paper's "train ≈ gen" claim).
    out["train_approximates_gen"] = float(all(0.8 <= r <= 1.4 for r in ratios))
    # generalization error must climb monotonically with the noise level.
    out["gen_monotone_in_noise"] = float(gens[0] > gens[1] > gens[2])
    return out


# Deterministic (fixed DGP seed, fixed rank). The binding assertions are
# `train_approximates_gen == 1` (the Table-1 headline) and monotonicity in
# noise; per-noise ratios and generalization errors are pinned with bands wide
# enough to absorb SVD/library drift.
EXPECTED = {
    "ratio_noise310": (1.148, 0.20),
    "ratio_noise130": (1.078, 0.20),
    "ratio_noise040": (1.030, 0.20),
    "gen_err_noise310": (0.202, 0.06),
    "gen_err_noise130": (0.044, 0.02),
    "gen_err_noise040": (0.0044, 0.003),
    "train_approximates_gen": (1.0, 0.0),
    "gen_monotone_in_noise": (1.0, 0.0),
}
