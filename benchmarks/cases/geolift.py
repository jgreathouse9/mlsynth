"""GEOLIFT cross-validation vs GeoLift/augsynth: the full GeoLift_Walkthrough.

Cross-validation (scenario: match an authoritative reference implementation).
Meta's GeoLift package walkthrough runs Ben-Michael, Feller & Rothstein's
Augmented SCM through augsynth with ``fixed_effects=TRUE`` (the package default)
and Chernozhukov-Wuthrich-Zhu conformal inference. The vignette's public call is

    GeoTest <- GeoLift(Y_id = "Y", data = GeoTestData_Test,
                       locations = c("chicago", "portland"),
                       treatment_start_time = 91, treatment_end_time = 105)
    summary(GeoTest)                       # the unaugmented "base" model
    GeoTestBest <- GeoLift(..., model = "best")   # ridge-augmented "best" model
    summary(GeoTestBest)

over the last 15 of 105 days (40 markets, the other 38 as donors). The walkthrough
prints two summaries -- the base (unaugmented simplex) model and the ridge-augmented
"best" model -- and this case pins mlsynth's **public** estimator, driven exactly as
a user would (the two markets forced via ``to_be_treated`` + ``treatment_size``, the
post window marked by ``post_col``), against every published number in both:

base (``augment=None``, the unaugmented model):
    Average ATT 155.556, Percent Lift 5.4, Incremental 4667, conformal p 0.01,
    L2 Imbalance 909.489, Scaled L2 0.1636, % Improvement from Naive 83.64.
ridge-augmented (``augment="ridge"``, the "best" model):
    Average ATT 156.805, Percent Lift 5.5, Incremental 4704, conformal p 0.01,
    L2 Imbalance 903.525, Scaled L2 0.1626, % Improvement from Naive 83.74,
    Average Estimated Bias Removed -1.249 (= base ATT - augmented ATT).

It also pins the augmented model's 13 donor weights (cincinnati 0.2272, miami
0.2028, baton rouge 0.1335, ...) against the vignette's printed weight table.

The point estimate, L2 imbalance, scaled L2, percent improvement and weights are
deterministic augsynth quantities, so they match to the published digits; the
conformal p-value is deterministic at ``seed=0`` / ``ns=2000`` and lands at the
vignette's reported ~1.4% / ~1.3% (the summary rounds both to 0.01). The live
augsynth weights/lambda/ATT are separately pinned to a fresh Rscript run by
``geolift_augsynth_ref``. See ``docs/replications/geolift.rst``.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from mlsynth import GEOLIFT

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "geolift_test_data.csv")
_TREATED = ["chicago", "portland"]
_PRE = 90
_NS = 2000

# The augmented ("best") model's donor weights as printed in the walkthrough.
_PUBLISHED_WEIGHTS = {
    "cincinnati": 0.2272, "miami": 0.2028, "baton rouge": 0.1335,
    "minneapolis": 0.0900, "dallas": 0.0739, "nashville": 0.0685,
    "honolulu": 0.0673, "austin": 0.0465, "san diego": 0.0451,
    "reno": 0.0306, "san antonio": 0.0054, "new york": 0.0046,
    "houston": 0.0046,
}


def _fit(df: pd.DataFrame, n_post: int, augment):
    config = {
        "df": df, "outcome": "Y", "unitid": "location", "time": "date",
        "treatment_size": len(_TREATED), "to_be_treated": _TREATED,
        "durations": [n_post], "effect_sizes": [0.0, 0.10],
        "lookback_window": 1, "post_col": "post",
        "how": "mean", "augment": augment, "fixed_effects": True,
        "power_threshold": 0.8, "alpha": 0.1, "ns": _NS, "seed": 0,
        "conformal_type": "iid", "display_graphs": False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GEOLIFT(config).fit()


def _naive_l2(df: pd.DataFrame, dates) -> float:
    """augsynth's naive (uniform-weight) pre-period imbalance: the L2 norm of the
    pre-period gap between the demeaned treated mean and the demeaned simple mean
    of all donors. Scaled L2 = model L2 / this; % improvement = 100 * (1 - scaled).
    """
    wide = df.pivot(index="date", columns="location", values="Y").loc[dates]
    donors = [c for c in wide.columns if c not in _TREATED]
    treated = wide[_TREATED].mean(axis=1).to_numpy(float)
    uniform = wide[donors].mean(axis=1).to_numpy(float)
    dm = lambda x: x - x[:_PRE].mean()  # noqa: E731 - fixed-effect demeaning
    gap = dm(treated)[:_PRE] - dm(uniform)[:_PRE]
    return float(np.sqrt(np.sum(gap ** 2)))


def run() -> dict:
    df = pd.read_csv(os.path.abspath(_DATA))
    dates = sorted(df["date"].unique())
    n_post = len(dates) - _PRE
    # Mark the 15 post-treatment periods (days 91-105), as the walkthrough does
    # with treatment_start_time = 91.
    df["post"] = df["date"].isin(set(dates[_PRE:])).astype(int)
    l2_naive = _naive_l2(df, dates)

    def summarize(res):
        rep = res.report
        gap = np.asarray(rep.time_series.estimated_gap, dtype=float)
        cf_post = float(np.mean(rep.time_series.counterfactual_outcome[_PRE:]))
        att = float(rep.effects.att)                          # per-unit (how="mean")
        l2 = float(np.sqrt(np.sum(gap[:_PRE] ** 2)))          # augsynth L2 imbalance
        return {
            "att": att,
            "pct_lift": 100.0 * att / cf_post,
            "incremental": float(np.sum(gap[_PRE:])) * len(_TREATED),
            "conformal_p": float(rep.inference.p_value),
            "l2": l2,
            "scaled_l2": l2 / l2_naive,
            "pct_improve": 100.0 * (1.0 - l2 / l2_naive),
            "weights": rep.weights.donor_weights,
        }

    base_res = _fit(df, n_post, None)
    aug = summarize(_fit(df, n_post, "ridge"))
    base = summarize(base_res)

    # augmented donor weights vs the printed vignette table (max abs gap over 13)
    wmax = max(abs(aug["weights"].get(k, 0.0) - v)
               for k, v in _PUBLISHED_WEIGHTS.items())

    return {
        # the design forces the walkthrough's two markets
        "selected_chicago_portland": float(
            set(base_res.selected_units) == set(_TREATED)),
        # base (unaugmented) summary
        "base_att": base["att"],
        "base_pct_lift": base["pct_lift"],
        "base_incremental": base["incremental"],
        "base_conformal_p": base["conformal_p"],
        "base_l2": base["l2"],
        "base_scaled_l2": base["scaled_l2"],
        "base_pct_improve": base["pct_improve"],
        # ridge-augmented ("best") summary
        "aug_att": aug["att"],
        "aug_pct_lift": aug["pct_lift"],
        "aug_incremental": aug["incremental"],
        "aug_conformal_p": aug["conformal_p"],
        "aug_l2": aug["l2"],
        "aug_scaled_l2": aug["scaled_l2"],
        "aug_pct_improve": aug["pct_improve"],
        "bias_removed": base["att"] - aug["att"],   # GeoLift -1.249
        "weight_max_abs_diff": wmax,
    }


# Deterministic (fixed CV lambda, fixed seed/ns). Targets are the GeoLift
# walkthrough's printed summaries; tolerances accept the small numerical gap.
EXPECTED = {
    "selected_chicago_portland": (1.0, 0.5),
    # base (unaugmented) model
    "base_att": (155.556, 0.5),         # GeoLift 155.556
    "base_pct_lift": (5.4, 0.1),        # GeoLift 5.4%
    "base_incremental": (4667.0, 5.0),  # GeoLift 4667
    "base_conformal_p": (0.01, 0.02),   # GeoLift 0.01 (deterministic at seed=0)
    "base_l2": (909.489, 1.0),          # GeoLift 909.489
    "base_scaled_l2": (0.1636, 0.001),  # GeoLift 0.1636
    "base_pct_improve": (83.64, 0.1),   # GeoLift 83.64%
    # ridge-augmented ("best") model
    "aug_att": (156.805, 0.5),          # GeoLift 156.805
    "aug_pct_lift": (5.5, 0.1),         # GeoLift 5.5%
    "aug_incremental": (4704.0, 5.0),   # GeoLift 4704
    "aug_conformal_p": (0.01, 0.02),    # GeoLift 0.01 (deterministic at seed=0)
    "aug_l2": (903.525, 1.0),           # GeoLift 903.525
    "aug_scaled_l2": (0.1626, 0.001),   # GeoLift 0.1626
    "aug_pct_improve": (83.74, 0.1),    # GeoLift 83.74%
    "bias_removed": (-1.249, 0.05),     # GeoLift -1.249
    "weight_max_abs_diff": (0.0, 0.0005),  # 13 published donor weights
}
