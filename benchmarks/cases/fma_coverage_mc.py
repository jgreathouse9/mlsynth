"""FMA Path-B: Li & Sonnier (2023) asymptotic-CI coverage (Web Appendix E).

Validates mlsynth's ``FMA`` (factor-model approach) against the paper's own Monte
Carlo. The methodological contribution is that the asymptotic confidence interval
of Theorem 3.1 attains **nominal coverage regardless of whether the treated and
control idiosyncratic variances are equal** -- the regime where the Xu (2017)
interval breaks. The DGP
(:func:`mlsynth.utils.fma_helpers.simulation.simulate_fma_sample`) has a zero true
effect, so coverage is whether the CI contains 0.

Across the three variance regimes (``equal`` / ``treated_smaller`` /
``treated_larger``) FMA's 95% CI covers near nominal in every cell:

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

What this case does not do is pin a value. Coverage is a property both a
correct and an incorrect variance estimator can satisfy, and this design has
T1 = 30 against r + 1 = 4 regressors, where a degrees-of-freedom correction
moves the standard error by only sqrt(30/26) = 1.07. FMA shipped a variance
that was wrong by a factor of 1.35 on the authors' own panel and every cell
here stayed inside its tolerance throughout. The check with the power to catch
that is a reference pin, and it lives in
``mlsynth/tests/test_fma.py::TestAsymptoticMatchesTheAuthorsImplementation``,
which compares the standard error against the authors' Web Appendix I MATLAB
on HCW's Hong Kong panel to six significant figures.

Wang, Racine & Wang (2025) Tables 1-2 measure this interval under-covering at
short pre-periods -- 0.91-0.93 at T1 = 30, falling to 0.80 at T1 = 10 -- so the
cells below sit slightly under 0.95 by construction, not by defect. The
studentized bootstrap that corrects it is ``fma_percentile_t_mc``.
"""
from __future__ import annotations

import warnings

import numpy as np

M = 120


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
    # 0.85 is the floor the per-cell targets already imply (0.95 with a
    # tolerance of 0.10). It was 0.88, which was stricter than the cells it
    # summarised and was calibrated while the variance estimator was inflated
    # by a degrees-of-freedom correction the paper does not apply. With that
    # corrected the interval covers where Wang, Racine & Wang measure it --
    # 0.91-0.93 at T1 = 30 -- so a floor above that asserts something the
    # source says is false.
    cells["all_near_nominal"] = float(all(c >= 0.85 for c in
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
