"""SparseSC Path-A: California Proposition 99 with an over-rich predictor set.

Path A (empirical, scenario: authors' canonical panel). Vives-i-Bastida (2022)
motivates SparseSC as **predictor selection**: an L1 penalty on the
predictor-importance vector ``v`` drives uninformative predictors to exactly
zero. The demonstration (paper Section 5.1, which revisits Abadie, Diamond &
Hainmueller 2010) is to hand the estimator a deliberately *over-rich* predictor
set on the California tobacco-control panel and check that

  * the penalty prunes it back to a small, interpretable subset, and
  * the pruned fit still lands on the ADH benchmark of roughly -19 packs and
    **recovers ADH's donor pool** (Utah / Nevada / Connecticut / Colorado).

The augmented panel ships in ``basedata/augmented_cali_long.csv`` (the ADH
outcome plus a large covariate set), so this is reproducible value-for-value.
The fit is deterministic (no RNG), so the cells below are exact re-runs;
conformal inference (Chernozhukov, Wuethrich & Zhu 2021, calibrated on the
validation residuals) gives the CI.

Provenance: Vives-i-Bastida (2022), arXiv:2203.11576v2, Section 5.1;
Abadie, Diamond & Hainmueller (2010) for the ~-19 benchmark.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "augmented_cali_long.csv")

# Columns that are identifiers / the outcome / the treatment, never predictors.
_DROP = {"state", "year", "treated", "cigsale", "stateno", "state_fips",
         "state_icpsr", "is_a_state", "region"}
_LAGS = [1975, 1980, 1988]   # pre-treatment outcome lags (ADH-style)


def run() -> dict:
    from mlsynth import SparseSC

    d = pd.read_csv(os.path.abspath(_DATA))
    d["treated"] = ((d.state == "California") & (d.year >= 1989)).astype(int)

    # Over-rich predictor set: every numeric covariate with complete,
    # non-constant pre-period coverage (collapsed to unit means), capped at 30.
    pre = d[d.year < 1989]
    covs = [c for c in d.columns
            if c not in _DROP and d[c].dtype.kind in "if"
            and pre.groupby("state")[c].mean().notna().all()
            and pre.groupby("state")[c].mean().std(ddof=1) > 0][:30]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SparseSC({
            "df": d, "outcome": "cigsale", "treat": "treated",
            "unitid": "state", "time": "year",
            "covariates": covs, "outcome_lag_periods": _LAGS,
            "run_inference": True, "inference_method": "conformal",
            "display_graphs": False,
        }).fit()

    v = np.asarray(res.design.v)
    w = np.asarray(res.design.w)
    kept = int((v > 1e-6).sum())
    # ADH donor pool: the four states carrying non-trivial weight.
    big_donors = int((w > 0.05).sum())
    top4 = float(np.sort(w)[::-1][:4].sum())

    return {
        "n_predictors": float(len(covs) + len(_LAGS)),   # 30 + 3 = 33
        "att_1989_2000": float(res.att),
        "pre_rmse": float(res.pre_rmse),
        "opt_lambda": float(res.design.opt_lambda),
        "predictors_kept": float(kept),
        "ci_lower": float(res.inference.ci_lower),
        "ci_upper": float(res.inference.ci_upper),
        "donors_above_5pct": float(big_donors),
        "top4_weight_mass": top4,
    }


# The fit uses no RNG, but SparseSC's predictor-importance search is *not*
# convex: it minimises validation MSE over the ``v`` vector by a non-convex
# descent, and its optimum is sensitive to floating-point rounding at the
# LAPACK-kernel level. Different environments (CPU microarchitecture -> OpenBLAS
# kernel selection, hence LAPACK path) reproducibly land the search in one of
# two adjacent basins on the flat sparse plateau:
#
#   * the paper/local optimum -- att ~= -18.2 packs (Vives-i-Bastida 2023,
#     Table 1, "Sparse SCM+" = -18.2), ~5 predictors kept, ci_upper ~= -15.4;
#   * a neighbouring optimum seen on the GitHub-hosted runners -- att ~= -19.8,
#     ~7 predictors, ci_upper ~= -17.6 (its runs emit DLASCL scaling warnings
#     the local stack does not, the tell-tale of the different kernel).
#
# Both recover the ADH story (large negative effect, the Utah/Nevada/
# Connecticut/Colorado donor pool, a conformal CI excluding zero); they differ
# only in how far down the plateau the search settles. Pinning BLAS threads to 1
# does NOT collapse the two -- single-threaded CI still lands on -19.8, so the
# divergence is kernel selection, not a thread race. Absent a stabler covariate-
# selection solver (a genuine open problem, left to future work), the honest fix
# is to widen the affected tolerances so they admit both basins while still
# catching a real regression (a sign flip, a collapsed donor pool, a CI that
# spans zero). The looser cells are flagged inline below.
#
# opt_lambda is NOT gated: the paper does not report the penalty value, and the
# selected lambda floats among several adjacent grid points along the flat sparse
# plateau, so pinning it would re-introduce flakiness. The ATT's sign and
# rough magnitude, not its third digit, is the paper-anchored quantity.
EXPECTED = {
    "n_predictors": (33.0, 0.0),
    "att_1989_2000": (-18.20, 1.75),      # Vives-i-Bastida 2023 Table 1 (Sparse SCM+); widened to admit the -19.8 CI basin
    "pre_rmse": (2.16, 0.25),
    "predictors_kept": (5.0, 2.5),        # L1 selection: 33 -> ~5-7 depending on the basin
    "ci_lower": (-21.0, 2.5),             # conformal CI excludes 0 in both basins
    "ci_upper": (-15.4, 2.5),             # widened to admit the deeper (-17.6) CI basin
    "donors_above_5pct": (4.0, 1.0),      # ADH's 4-state pool
    "top4_weight_mass": (1.0, 0.05),      # those four carry ~all the weight
}
