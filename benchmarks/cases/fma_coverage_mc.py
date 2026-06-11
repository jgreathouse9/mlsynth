"""FMA Path-B: Li & Sonnier (2023) asymptotic-CI coverage (Web Appendix E).

Validates mlsynth's ``FMA`` (factor-model approach) against the paper's own Monte
Carlo. The methodological contribution is that the asymptotic confidence interval
of Theorem 3.1 attains **nominal coverage regardless of whether the treated and
control idiosyncratic variances are equal** -- the regime where the Xu (2017)
interval breaks. The DGP
(:func:`mlsynth.utils.fma_helpers.simulation.simulate_fma_sample`) has a zero true
effect, so coverage is whether the CI contains 0.

Across the three variance regimes (``equal`` / ``treated_smaller`` /
``treated_larger``) FMA's 95% CI covers at ~0.95-0.97 -- near nominal in every
cell:

  =================  ===============
  variance regime    FMA coverage
  =================  ===============
  equal              ~0.95
  treated_smaller    ~0.95
  treated_larger     ~0.95
  =================  ===============

Path B (the paper's simulation): the case asserts every cell covers near the
nominal 95% -- the paper's headline that coverage is robust to variance
inequality -- not exact cells (fewer reps than the paper's 100,000).
Deterministic (seeded).
"""
from __future__ import annotations

import warnings

import numpy as np

M = 40


def _coverage(dgp: str, variance_case: str) -> float:
    from mlsynth import FMA
    from mlsynth.utils.fma_helpers.simulation import simulate_fma_sample

    covers = np.empty(M)
    for j in range(M):
        s = simulate_fma_sample(dgp=dgp, N_co=30, T1=30, T2=20,
                                variance_case=variance_case,
                                rng=np.random.default_rng(j))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = FMA({"df": s.df, "outcome": "y", "treat": "D",
                       "unitid": "unit", "time": "time",
                       "display_graphs": False}).fit()
        d = res.inference_detail
        covers[j] = 1.0 if d.asymptotic_att_lower <= 0.0 <= d.asymptotic_att_upper else 0.0
    return float(covers.mean())


def run() -> dict:
    cells = {
        "cov_equal": _coverage("dgp1", "equal"),
        "cov_treated_smaller": _coverage("dgp1", "treated_smaller"),
        "cov_treated_larger": _coverage("dgp1", "treated_larger"),
        "cov_dgp2_equal": _coverage("dgp2", "equal"),
    }
    cells["min_coverage"] = float(min(cells.values()))
    cells["all_near_nominal"] = float(all(c >= 0.88 for c in
                                          list(cells.values())[:4]))
    return cells


# Deterministic (seeded). Tolerances absorb the Monte Carlo noise at M=40 (the
# paper uses 100,000). Reproduces Li & Sonnier's headline: the asymptotic CI
# covers near the nominal 95% in every variance regime, including when the treated
# and control variances differ (where Xu's interval fails).
EXPECTED = {
    "cov_equal": (0.95, 0.10),
    "cov_treated_smaller": (0.95, 0.10),
    "cov_treated_larger": (0.95, 0.10),
    "cov_dgp2_equal": (0.95, 0.12),
    "min_coverage": (0.92, 0.10),
    "all_near_nominal": (1.0, 0.0),
}
