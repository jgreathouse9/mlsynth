"""Path B benchmark: TSSC Figure-2 Monte Carlo MSE ratio (Li & Shankar 2024).

Reproduces the headline of the paper's simulation study (Figure 2): when the SC
restrictions (donor weights sum to one, no intercept) hold in population, the
more constrained SC estimator has *lower* MSE than the looser MSCc, and the
advantage has a characteristic geometry -- the MSE ratio
``MSE_SC / MSE_MSCc`` rises toward 1 as the pre-period ``T1`` grows (MSCc's slack
matters less with more data) and falls as the post-period ``T2`` grows (SC's
bias advantage compounds over a longer post-mean).

The DGP (three latent factors, homogeneous unit loadings b = [1,1,1]) is
:func:`mlsynth.utils.tssc_helpers.simulation.simulate_tssc_sample`; the SC / MSCc
weight solves are the package's own ``_solve`` / ``_features``.

Provenance
----------
* Headline: Li & Shankar (2024) Figure 2 -- every MSE-ratio cell lies below 1;
  the paper uses M = 10,000. We use M = 200 (tolerances absorb the MC gap), at
  the grid corners that show both monotonicities.
"""
from __future__ import annotations

import numpy as np

M = 200
T1_LO, T1_HI = 30, 200
T2_LO, T2_HI = 5, 30


def _ratio(T1: int, T2: int) -> float:
    from mlsynth.utils.tssc_helpers.simulation import simulate_tssc_sample
    from mlsynth.utils.tssc_helpers.estimation import _solve, _features

    def att(method, s):
        w = _solve(method, s.donors[:s.T1], s.y_treated[:s.T1], s.T1, s.N_co)
        cf = _features(method, s.donors) @ w
        return float(np.mean(s.y_treated[s.T1:] - cf[s.T1:]))

    sc, mscc = [], []
    for j in range(M):
        s = simulate_tssc_sample(T1=T1, T2=T2, N_co=10, rng=np.random.default_rng(j))
        sc.append(att("SC", s))
        mscc.append(att("MSCc", s))
    return float(np.mean(np.asarray(sc) ** 2) / np.mean(np.asarray(mscc) ** 2))


def run() -> dict:
    r = {(t1, t2): _ratio(t1, t2)
         for t1 in (T1_LO, T1_HI) for t2 in (T2_LO, T2_HI)}
    ratios = list(r.values())
    return {
        "ratio_t1_30_t2_5": r[(30, 5)],
        "ratio_t1_30_t2_30": r[(30, 30)],
        "ratio_t1_200_t2_5": r[(200, 5)],
        "ratio_t1_200_t2_30": r[(200, 30)],
        # 1.0 iff every cell is below 1 (SC dominates MSCc in MSE).
        "all_below_one": float(max(ratios) < 1.0),
        # 1.0 iff the ratio rises with T1 (at both T2) ...
        "rises_with_T1": float(r[(200, 5)] > r[(30, 5)] and r[(200, 30)] > r[(30, 30)]),
        # ... and falls with T2 (at both T1).
        "falls_with_T2": float(r[(30, 5)] > r[(30, 30)] and r[(200, 5)] > r[(200, 30)]),
    }


# Deterministic (rng=default_rng(j) per draw). Binding facts: every ratio < 1
# and the two monotonicities; per-cell ratios pinned with bands for the M=200
# vs paper-M=10,000 Monte-Carlo gap.
EXPECTED = {
    "ratio_t1_30_t2_5": (0.433, 0.10),
    "ratio_t1_30_t2_30": (0.037, 0.05),
    "ratio_t1_200_t2_5": (0.918, 0.10),
    "ratio_t1_200_t2_30": (0.684, 0.12),
    "all_below_one": (1.0, 0.0),
    "rises_with_T1": (1.0, 0.0),
    "falls_with_T2": (1.0, 0.0),
}
