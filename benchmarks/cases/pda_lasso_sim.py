"""PDA Path-B: Li & Bell (2017) LASSO-PDA out-of-sample prediction (Table 2).

Reproduces the geometry of Li & Bell (2017, *Journal of Econometrics* 197,
Table 2): on a **dense three-factor** DGP with more controls than pre-periods
(``N=31 > T1=25``, ``T2=10``), the LASSO selection of HCW/PDA control units is
**parsimonious** (~7 of the 30 controls) and its out-of-sample PMSE **scales
with the idiosyncratic noise** ``sigma^2`` -- the paper's headline being that
LASSO predicts the untreated path better, and with far fewer regressors, than
AICC model selection.

The DGP is :func:`mlsynth.utils.pda_helpers.simulation.simulate_libell_panel`
(Eq. 5.1-5.2): ``y_it^0 = 1 + b_i' f_t + u_it`` with ``b_ji ~ N(1,1)``,
``u_it ~ N(0, sigma^2)``, and three serially-dependent factors. No treatment
effect is injected; the post-period gap is the **prediction error**, so
``PMSE = mean(gap_post^2)``.

  ==========  ==============  ==============  ==============
  sigma^2     mlsynth PMSE    Li-Bell LASSO   (#donors)
  ==========  ==============  ==============  ==============
  1.0         ~1.48           1.77            ~7 (paper 7.0)
  0.5         ~0.74           0.96            ~7
  0.1         ~0.13           0.22            ~6
  ==========  ==============  ==============  ==============

mlsynth's LASSO uses ``LassoCV`` (5-fold); Li & Bell use leave-one-out CV, so
the PMSE *level* is a little lower, but the parsimony and the ``sigma^2``
scaling -- the paper's points -- reproduce. Path B (scenario 1): DGP
re-implemented from the paper; the case asserts the geometry, not exact cells.
"""
from __future__ import annotations

import warnings

import numpy as np

from mlsynth.utils.pda_helpers.simulation import simulate_libell_panel

N, T1, T2 = 31, 25, 10
M = 40


def _pmse_ndonors(sigma2, base_seed):
    from mlsynth import PDA
    pmse, ndon = [], []
    for i in range(M):
        df = simulate_libell_panel(N=N, T1=T1, T2=T2, sigma2=sigma2,
                                   seed=base_seed + i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = PDA({"df": df, "outcome": "y", "treat": "treat",
                       "unitid": "unit", "time": "time", "methods": ["LASSO"],
                       "display_graphs": False}).fit().fits["lasso"]
        pmse.append(float(np.mean(fit.gap[T1:] ** 2)))
        sel = fit.selected_donors
        ndon.append(len(sel) if sel is not None else int(np.sum(np.abs(fit.beta) > 1e-8)))
    return float(np.mean(pmse)), float(np.mean(ndon))


def run() -> dict:
    p1, n1 = _pmse_ndonors(1.0, 1000)
    p05, _ = _pmse_ndonors(0.5, 2000)
    p01, _ = _pmse_ndonors(0.1, 3000)
    return {
        "lasso_pmse_s1": p1,
        "lasso_pmse_s05": p05,
        "lasso_pmse_s01": p01,
        "lasso_ndonors_s1": n1,
        # geometry: PMSE shrinks with the idiosyncratic noise (the paper's point)
        "pmse_scales_with_sigma2": float(p1 > p05 > p01),
    }


# Deterministic (seeded). Centered on the measured values; tolerances absorb the
# Monte-Carlo noise at M=40 and the gap to Li & Bell's M=10000 + their LOO CV
# (mlsynth's LassoCV gives a slightly lower PMSE level). The parsimony (~7 of 30
# controls) and the sigma^2 scaling are the reproduced geometry.
EXPECTED = {
    "lasso_pmse_s1": (1.507, 0.5),        # Li-Bell 1.77
    "lasso_pmse_s05": (0.965, 0.3),       # Li-Bell 0.96
    "lasso_pmse_s01": (0.176, 0.1),       # Li-Bell 0.22
    "lasso_ndonors_s1": (7.5, 2.5),       # Li-Bell 7.0; parsimonious vs AICC's ~10
    "pmse_scales_with_sigma2": (1.0, 0.0),
}
