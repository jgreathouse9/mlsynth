"""Path A: Sun et al. (2025, AJAE), Tennessee's 2016 wine sale reform.

Tennessee let grocery stores sell wine from July 2016. The paper asks whether
the intended gain -- wine excise tax revenue -- came with unintended costs in
liquor store employment, and estimates each with both SCM and SDID. This pins
both against the authors' own replication package (OSF ``rbt4n``).

Study 1 (liquor store establishments) is not here and cannot be: its outcome
comes from proprietary NielsenIQ data the authors are not free to redistribute,
which also rules out the first column of Table 6.

What is pinned, and at what tolerance, follows from what the data determine.

The *oracle* rows feed the paper's own published donor weights back through the
counterfactual arithmetic and ask for the paper's own per-period effects. They
agree to 0.01 -- three decimals on numbers of order 40 -- and because the
weights are held fixed, nothing in the predictor-weight search can perturb them.
They isolate the data handling. If a fitted row below ever drifts while these
hold, the drift is in the weight search and nowhere else.

The *fitted* rows run the estimators. Study 2's SCM lands at -40.50 against a
published -42.50, and the gap is recorded rather than tuned away, because
mlsynth's weights achieve a *lower* pre-treatment RMSE than the published ones
(14.52 against 14.74). Stata's ``synth ... nested allopt`` and mlsynth's
differential-evolution predictor-weight search are two optimisers on a
non-convex nested problem; landing on a better-fitting local optimum is not an
error, and pinning the published number tightly would amount to asserting that
Stata's optimum is the correct one. The RMSE comparison is pinned instead, so
the excuse has to keep being true.

Study 3's donor weights are deliberately *not* pinned. With three pre-treatment
periods and five donors the weight vector is not identified: the best attainable
pre-treatment RMSE over all donor subsets is 0.0413 against the published
weights' 0.0439, a gap of well under a tenth of the outcome's own standard
deviation. Several weight vectors fit about equally well and the published one
is not the best of them, so a row demanding it would be demanding one arbitrary
point from a flat region. Its ATT *is* identified and is pinned, to 0.05.

The SDID rows are the reason #322 and #323 existed. The paper's Stata call is
``sdid ..., covariates(...)`` with no type named, so it takes Stata's default,
which is ``optimized`` rather than ``projected``. Study 2 also carries the
regression guard for #325: ``incomepc`` has roughly seventy times the outcome's
within-unit dispersion, and before the covariates were scaled this panel
returned 1.7e+11.

Reference values are the published tables; see
``benchmarks/reference/wine_tennessee/provenance.txt``. No Stata runs in CI.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

_DIR = Path(__file__).resolve().parents[1] / "reference" / "wine_tennessee"

PAPER_W2 = {"KY": 0.231, "NY": 0.551, "UT": 0.218}
PAPER_W3 = {"AK": 0.104, "MN": 0.636, "NY": 0.259}
PAPER_T4_AVG = -42.500
PAPER_T5_AVG = 0.589
PAPER_T6B_AVG = -53.154
PAPER_T6C_AVG = 0.610

COVARIATES = ["popdense2010", "pop21per", "incomepc", "wineper", "college",
              "unemploy", "nonwhite"]


def _load(name):
    d = pd.read_csv(_DIR / name)
    for c in d.columns:
        if d[c].dtype == "float32":
            d[c] = d[c].astype("float64")
    return d


def _synthetic(df, time, outcome, weights, periods):
    wide = df.pivot_table(index=time, columns="state", values=outcome)
    actual = wide.loc[periods, "TN"].to_numpy(float)
    synthetic = np.zeros_like(actual)
    for unit, weight in weights.items():
        synthetic = synthetic + weight * wide.loc[periods, unit].to_numpy(float)
    return actual - synthetic


def _split(df, time, last_pre):
    idx = sorted(df[time].unique())
    return ([t for t in idx if t <= last_pre], [t for t in idx if t > last_pre])


def _pre_rmse(df, time, outcome, weights, last_pre):
    pre, _ = _split(df, time, last_pre)
    gap = _synthetic(df, time, outcome, weights, pre)
    return float(np.sqrt(np.mean(gap ** 2)))


def _scm(df, outcome, time, windows, lag_windows):
    from mlsynth import VanillaSC

    d = df.copy()
    for name, window in lag_windows.items():
        d[name] = d[outcome]
        windows = {**windows, name: window}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC({
            "df": d, "outcome": outcome, "treat": "treat", "unitid": "state",
            "time": time, "covariates": list(lag_windows) + COVARIATES,
            "covariate_windows": windows, "backend": "mscmt",
            "display_graphs": False}).fit()


def _sdid(df, outcome, time, covariates=None):
    from mlsynth import SDID

    cfg = {"df": df, "outcome": outcome, "treat": "treat", "unitid": "state",
           "time": time, "vce": "noinference", "display_graphs": False}
    if covariates is not None:
        cfg["covariates"] = {"optimized": covariates}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(SDID(cfg).fit().effects.att)


def _best_attainable_rmse(df, time, outcome, last_pre):
    """Lowest pre-treatment RMSE over every donor subset, on the simplex."""
    from scipy.optimize import nnls

    pre, _ = _split(df, time, last_pre)
    wide = df.pivot_table(index=time, columns="state", values=outcome)
    donors = [c for c in wide.columns if c != "TN"]
    target = wide.loc[pre, "TN"].to_numpy(float)
    best = np.inf
    for size in range(1, len(donors) + 1):
        for combo in combinations(donors, size):
            design = wide.loc[pre, list(combo)].to_numpy(float)
            big = 1e6                                  # enforce sum-to-one
            sol, _ = nnls(np.vstack([design, big * np.ones(len(combo))]),
                          np.append(target, big))
            best = min(best, float(np.sqrt(
                np.mean((target - design @ sol) ** 2))))
    return best


def run() -> dict:
    study2 = _load("study2_employment.csv")
    study3 = _load("study3_taxrevenue.csv")
    out: dict = {}

    # --- panel shape: a truncated panel must fail here, not silently later ---
    out["study2_rows"] = float(len(study2))
    out["study3_rows"] = float(len(study3))

    # --- oracle: the paper's weights through the counterfactual arithmetic ---
    _, post2 = _split(study2, "time", 30)
    _, post3 = _split(study3, "year", 2016)
    gap2 = _synthetic(study2, "time", "employper", PAPER_W2, post2)
    gap3 = _synthetic(study3, "year", "taxper", PAPER_W3, post3)
    out["oracle_table4_avg_diff"] = float(abs(np.mean(gap2) - PAPER_T4_AVG))
    out["oracle_table4_max_diff"] = float(np.max(np.abs(gap2 - np.array([
        -13.081, -30.672, -31.033, -52.340, -62.457,
        -61.404, -54.817, -49.229, -31.310, -38.665]))))
    out["oracle_table5_avg_diff"] = float(abs(np.mean(gap3) - PAPER_T5_AVG))
    out["oracle_table5_max_diff"] = float(
        np.max(np.abs(gap3 - np.array([0.700, 0.480, 0.587]))))

    # --- fitted SCM ---
    fit2 = _scm(study2, "employper", "time",
                {c: (1, 30) for c in ("pop21per", "incomepc", "wineper",
                                      "unemploy")},
                {"employper_lag": (1, 30)})
    w2 = {k: float(v) for k, v in (fit2.weights.donor_weights or {}).items()
          if abs(v) > 1e-3}
    out["scm2_donor_set_matches"] = float(set(w2) == set(PAPER_W2))
    out["scm2_max_weight_diff"] = float(max(
        abs(w2.get(u, 0.0) - v) for u, v in PAPER_W2.items()))
    out["scm2_att_diff"] = float(abs(float(fit2.effects.att) - PAPER_T4_AVG))
    # The excuse for the loose ATT tolerance, pinned so it must stay true.
    out["scm2_rmse_advantage"] = float(
        _pre_rmse(study2, "time", "employper", PAPER_W2, 30)
        - _pre_rmse(study2, "time", "employper", w2, 30))

    fit3 = _scm(study3, "taxper", "year",
                {c: (2014, 2016) for c in ("pop21per", "incomepc", "wineper",
                                           "unemploy", "nonwhite")},
                {"taxper_2014": (2014, 2014), "taxper_2016": (2016, 2016)})
    out["scm3_att_diff"] = float(abs(float(fit3.effects.att) - PAPER_T5_AVG))

    # --- what the data do not determine ---
    published3 = _pre_rmse(study3, "year", "taxper", PAPER_W3, 2016)
    best3 = _best_attainable_rmse(study3, "year", "taxper", 2016)
    out["study3_weights_unidentified"] = float(best3 < published3)
    out["study3_rmse_gap_in_sd"] = float(
        (published3 - best3) / float(study3["taxper"].std()))

    # --- fitted SDID (Table 6) ---
    sdid2 = _sdid(study2, "employper", "time", COVARIATES)
    sdid3 = _sdid(study3, "taxper", "year", COVARIATES)
    out["sdid2_att_diff"] = float(abs(sdid2 - PAPER_T6B_AVG))
    out["sdid3_att_diff"] = float(abs(sdid3 - PAPER_T6C_AVG))
    out["sdid2_finite"] = float(np.isfinite(sdid2) and abs(sdid2) < 1e3)
    # the covariates must do work, or the rows above would pass vacuously
    plain2 = _sdid(study2, "employper", "time")
    out["sdid2_covariates_bind"] = float(abs(sdid2 - plain2))
    out["sdid2_covariates_help"] = float(
        abs(sdid2 - PAPER_T6B_AVG) < abs(plain2 - PAPER_T6B_AVG))
    return out


#: Distances from the published values, so each row reads as "how far apart are
#: we". The tolerances are not uniform and the differences are the finding --
#: see the module docstring for why ``scm2_att_diff`` is 2.5 while the oracle
#: rows are 0.01.
EXPECTED = {
    "study2_rows": (440.0, 0.5),
    "study3_rows": (36.0, 0.5),

    # Fixed weights: the arithmetic either reproduces the paper or it does not.
    "oracle_table4_avg_diff": (0.0, 0.01),
    "oracle_table4_max_diff": (0.0, 0.01),
    "oracle_table5_avg_diff": (0.0, 0.01),
    "oracle_table5_max_diff": (0.0, 0.01),

    # Fitted: two optimisers on a non-convex nested problem.
    "scm2_donor_set_matches": (1.0, 0.5),
    "scm2_max_weight_diff": (0.0, 0.02),
    "scm2_att_diff": (0.0, 2.5),
    # Positive means mlsynth fits the pre-period better than the published
    # weights, which is the whole justification for the loose tolerance above.
    # The band is deliberately clear of zero: a version of this row that
    # admitted a negative value would let the justification fail silently.
    "scm2_rmse_advantage": (0.22, 0.15),
    "scm3_att_diff": (0.0, 0.05),

    # Recorded, not pinned: Study 3's weights are not identified.
    "study3_weights_unidentified": (1.0, 0.5),
    "study3_rmse_gap_in_sd": (0.0007, 0.01),

    # Table 6, via Stata's default covariate method.
    "sdid2_att_diff": (0.0, 0.5),
    "sdid3_att_diff": (0.0, 0.02),
    "sdid2_finite": (1.0, 0.5),
    "sdid2_covariates_bind": (6.2, 3.0),
    "sdid2_covariates_help": (1.0, 0.5),
}
