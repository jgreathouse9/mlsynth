"""Path B benchmark: concatenated multi-outcome SCM (Tian-Lee-Panchenko 2026, Table 1).

Reproduces the simulation that motivates the concatenated SCMO: under a factor
model whose outcomes share the unit predictors, matching the synthetic control on
*more* related outcomes sharpens identification of the latent predictors and so
**reduces the post-treatment bias** of the single-outcome SC -- at the cost of a
larger (but truthful) pre-treatment fit error, which rises toward the noise floor
rather than overfitting to near-zero.

The reported cells are, for each ``T0`` in {1, 5, 10} and each outcome count
``K`` in {1, 5, 10}: the average pre-treatment RMSPE ("pre") and the average
absolute post-period gap on outcome 1 ("bias"), over ``M`` null (``tau = 0``)
draws. ``K = 1`` is the conventional single-outcome SC (the ``separate`` scheme);
``K = 5, 10`` are the ``concatenated`` multi-outcome SC.

Provenance
----------
* DGP: :func:`mlsynth.utils.scmo_helpers.simulation.simulate_tian` -- the
  Section-3 factor model of Tian-Lee-Panchenko (2026), identical to the
  ``Simulation1.R`` DGP of the Sun et al. (2025) replication package (N = 30,
  f = 4 predictors, 1 post-period, tau = 0).
* Headline: Tian-Lee-Panchenko (2026, Econometrics Journal) Table 1, reproduced
  cell-by-cell by the authors' code as ``Output/sim_tab1.txt``::

        T0   K=1 (pre/bias)   K=5            K=10
        1    0.04 / 1.23      0.38 / 1.21    0.62 / 1.12
        5    0.46 / 1.21      0.95 / 1.04    1.02 / 1.00
        10   0.77 / 1.13      1.05 / 1.01    1.09 / 0.98

  The paper uses 5,000 reps; we use M = 250 (tolerances absorb the MC gap:
  bias SE ~ 1.4/sqrt(250) ~ 0.09).
"""
from __future__ import annotations

import warnings

import numpy as np

M = 250
SEED = 321
T0_VALUES = (1, 5, 10)
K_VALUES = (1, 5, 10)


def _grid() -> dict:
    from mlsynth import SCMO
    from mlsynth.utils.scmo_helpers.simulation import simulate_tian, to_panel

    out: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for T0 in T0_VALUES:
            rng = np.random.default_rng(SEED)
            acc = {K: [] for K in K_VALUES}
            for _ in range(M):
                Ys, N, TT, tr = simulate_tian(rng, T0, max(K_VALUES))
                for K in K_VALUES:
                    df = to_panel(Ys[:K], N, TT, tr)
                    cfg = {"df": df, "outcome": "y0", "treat": "treat",
                           "unitid": "unit", "time": "time",
                           "schemes": ["separate" if K == 1 else "concatenated"],
                           "display_graphs": False}
                    if K > 1:
                        cfg["addout"] = [f"y{k}" for k in range(1, K)]
                    fit = SCMO(cfg).fit()._primary
                    acc[K].append((fit.pre_rmse, float(np.asarray(fit.gap)[-1])))
            for K in K_VALUES:
                a = np.asarray(acc[K])
                out[(T0, K)] = (float(a[:, 0].mean()), float(np.abs(a[:, 1]).mean()))
    return out


def run() -> dict:
    g = _grid()
    res: dict = {}
    for T0 in T0_VALUES:
        for K in K_VALUES:
            pre, bias = g[(T0, K)]
            res[f"pre_T{T0}_K{K}"] = pre
            res[f"bias_T{T0}_K{K}"] = bias
    # Headline geometry: bias falls and pre-fit rises as outcomes are added.
    res["bias_falls_with_K"] = float(all(
        g[(T0, 1)][1] >= g[(T0, 10)][1] for T0 in T0_VALUES))
    res["prefit_rises_with_K"] = float(all(
        g[(T0, 1)][0] <= g[(T0, 10)][0] for T0 in T0_VALUES))
    return res


# Stochastic (M=250 vs the paper's 5,000). Pre-fit reproduces tightly (it is the
# deterministic geometry of the SC fit) -> +-0.08; bias absorbs MC noise -> +-0.12.
EXPECTED = {
    "pre_T1_K1": (0.04, 0.08), "bias_T1_K1": (1.23, 0.12),
    "pre_T1_K5": (0.38, 0.08), "bias_T1_K5": (1.21, 0.12),
    "pre_T1_K10": (0.62, 0.08), "bias_T1_K10": (1.12, 0.12),
    "pre_T5_K1": (0.46, 0.08), "bias_T5_K1": (1.21, 0.12),
    "pre_T5_K5": (0.95, 0.08), "bias_T5_K5": (1.04, 0.12),
    "pre_T5_K10": (1.02, 0.08), "bias_T5_K10": (1.00, 0.12),
    "pre_T10_K1": (0.77, 0.08), "bias_T10_K1": (1.13, 0.12),
    "pre_T10_K5": (1.05, 0.08), "bias_T10_K5": (1.01, 0.12),
    "pre_T10_K10": (1.09, 0.08), "bias_T10_K10": (0.98, 0.12),
    "bias_falls_with_K": (1.0, 0.0),
    "prefit_rises_with_K": (1.0, 0.0),
}
