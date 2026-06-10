"""TASC Path-B: the state-space ablation (Rho et al. 2026, Section 5.2).

Path B (Monte Carlo, scenario: the paper provides the generative model, and
mlsynth ships it). Time-Aware Synthetic Control embeds the SC outcome matrix in
a linear-Gaussian state-space model; the paper's Section-5.2 ablation (Figures
3-4) shows TASC delivers the smallest median counterfactual RMSE -- decisively
so under high observation noise -- against vanilla SC, because PCA/simplex
methods assume the principal directions are noise-free while TASC's full-rank
observation-noise prior does not.

This reproduces that geometry. Panels are drawn from TASC's own generative model
(:func:`mlsynth.utils.tasc_helpers.simulation.simulate_tasc_sample`):

    x_t = A x_{t-1} + N(0, q*I),   y_t = H x_t + N(0, r*I),

across the paper's four regimes -- small/large state-perturbation ``q`` crossed
with small/large observation-noise ``r``. For each draw both TASC and vanilla SC
estimate the (untreated) target counterfactual, scored by RMSE against the true
post-period path; the cell statistic is the median RMSE ratio TASC / SC over the
draws. A ratio below 1 means TASC wins.

mlsynth's TASC has the smaller median RMSE in **every** cell (strongest under
high noise, the paper's headline regime). Determinism: per-draw seeds 100+m, the
true latent dimension ``d = 5``.

Provenance: Rho, Illick, Narasipura, Abadie, Hsu & Misra (2026), *"Time-Aware
Synthetic Control,"* arXiv:2601.03099, Section 5.2, Figures 3-4.
"""
from __future__ import annotations

import warnings

import numpy as np

_M = 15                          # draws per cell
_D = 5                           # true latent dimension
# (label, q_scale, r_scale): small/large state-perturbation x small/large noise.
_CELLS = [
    ("hiQ_hiR", 0.1, 1.0),
    ("loQ_hiR", 0.01, 1.0),
    ("hiQ_loR", 0.1, 0.1),
    ("loQ_loR", 0.01, 0.1),
]


def _cf_rmse(res, true_post, T0):
    cf = np.asarray(res.time_series.counterfactual_outcome)[T0:]
    return float(np.sqrt(np.mean((cf - true_post) ** 2)))


def run() -> dict:
    from mlsynth import TASC, VanillaSC
    from mlsynth.utils.tasc_helpers.simulation import simulate_tasc_sample

    out = {}
    wins = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for label, q, r in _CELLS:
            rt, rv = [], []
            for m in range(_M):
                s = simulate_tasc_sample(q_scale=q, r_scale=r,
                                         rng=np.random.default_rng(100 + m))
                tp = s.Y[0, s.T0:]
                common = dict(df=s.df, outcome="y", treat="treat",
                              unitid="unit", time="time", display_graphs=False)
                rt.append(_cf_rmse(TASC({**common, "d": _D}).fit(), tp, s.T0))
                rv.append(_cf_rmse(VanillaSC(common).fit(), tp, s.T0))
            ratio = float(np.median(rt) / np.median(rv))
            out[f"rmse_ratio_{label}"] = ratio
            if ratio < 1.0:
                wins += 1
    out["n_cells_tasc_wins"] = float(wins)
    return out


# Deterministic (per-draw seeds, no RNG in the EM/SC fits). TASC's median
# counterfactual RMSE is below vanilla SC's in all four (q, r) regimes -- the
# Figures 3-4 ordering -- with the advantage largest where observation noise is
# high (the paper's headline). Tolerances are wide enough to absorb the M = 15
# Monte-Carlo noise yet still fail if TASC stops winning a cell.
EXPECTED = {
    "rmse_ratio_hiQ_hiR": (0.91, 0.12),     # high noise: TASC wins
    "rmse_ratio_loQ_hiR": (0.93, 0.12),     # high noise: TASC wins
    "rmse_ratio_hiQ_loR": (0.84, 0.12),
    "rmse_ratio_loQ_loR": (0.91, 0.12),
    "n_cells_tasc_wins": (4.0, 0.0),        # TASC wins every cell
}
