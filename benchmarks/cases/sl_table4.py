"""Path A benchmark: SL against Viviano and Bradic's published Table 4.

Their application is the 2005 TennCare disenrollment: Tennessee against the six
southern states that did not expand Medicaid, quarterly from 1993Q1, outcome the
share reporting they could not see a doctor because of cost. Experts fit on
periods 1 to 30, weights on 31 to 50, treatment at 51.

Three of their settings are settings and not tolerances, and each is passed here
through the public API:

* the design. Their forest reads the six donor outcomes plus all 51 columns of
  ``employment_BFRSS.txt``, employment for 50 states and Tennessee, which the
  panel-derived ``covariates`` block cannot carry -- that block is one column per
  panel unit. ``external_covariates`` is what carries it.
* the learning rate. Their ``analyze_main_text.R:403`` block runs
  ``1/(sqrt(88) var(med_ts))`` on a series of length 100 and gets 51.4307, where
  the paper's own ``1/(sqrt(T) var(y))`` at T = 100 is 48.25. Passed as ``eta``.
* the measured window. They report over periods 52 to 88 and drop 89 to 100
  everywhere -- from the statistic, the effect and the bootstrap pool alike. SL
  measures to the end of the panel and has no setting that moves the end of the
  window, so the four horizons are read off ``estimated_gap`` here.

Their table is four of their blocks rbound and multiplied by 100 on the way out
(``analyze_main_text.R:465``), so every figure in it is a proportion times a
hundred. The rows pinned here are its second half, from the block at line 403,
whose split and bias definition are mlsynth's exactly.

What is left between the two is the forest's seed. Over 20 seeds the effect's
relative difference to their published value has a standard deviation of 0.0024
and a largest absolute value of 0.0070, and the statistic's 0.0116 and 0.0350;
the tolerances below are sized from those, since a different scikit-learn build
moves the forest the way a different seed does.

``benchmarks/studies/sl_forest_languages`` is where the three settings were
measured apart, and where R's ``randomForest`` and scikit-learn's
``RandomForestRegressor`` are shown to be exchangeable draws once the design and
the hyperparameters match.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PANEL = "basedata/sl_tennessee_medcost.csv"
EMPLOYMENT = (Path(__file__).resolve().parents[1] / "reference" / "sl_table4"
              / "employment_BFRSS.txt")
TRAIN, ETA = 30, 51.4307

#: Table 4, second half, already times 100: horizon to (statistic, effect).
THEIRS = {
    "m0":   (0.690993973668991, 5.22265086402609),
    "m1yr": (0.622535806616939, 5.36235247515936),
    "m2yr": (0.624658251401803, 5.51667748005122),
    "m3yr": (0.610809309405541, 5.58704759037530),
}
#: The windows those rows measure, ``analyze_main_text.R`` lines 426 to 455,
#: as half-open Python slices over the 100 quarters.
WINDOWS = {"m0": (51, 88), "m1yr": (55, 88), "m2yr": (59, 88), "m3yr": (63, 88)}


def _external() -> pd.DataFrame:
    """Their employment matrix, monthly to quarterly as their script averages it.

    ``analyze_main_text.R`` lines 27 to 34 take the mean of each consecutive
    three rows, turning 300 months into 100 quarters.
    """
    monthly = pd.read_csv(EMPLOYMENT, sep=r"\s+").to_numpy(float)
    q = np.stack([monthly[3 * i:3 * i + 3].mean(axis=0) for i in range(100)])
    ext = pd.DataFrame(q, columns=[f"E{j}" for j in range(q.shape[1])])
    ext.insert(0, "quarter", np.arange(1, 101))
    return ext


def run() -> dict:
    from mlsynth import SL

    res = SL(dict(df=pd.read_csv(PANEL), outcome="medcost", treat="expansion",
                  unitid="state", time="quarter",
                  external_covariates=_external(), eta=ETA,
                  train_periods=TRAIN, n_boot=500, block=3, seed=0,
                  display_graphs=False)).fit()
    f = res.fit
    gap = np.asarray(res.time_series.estimated_gap, dtype=float)

    eff, stat = {}, {}
    for m, (a, b) in WINDOWS.items():
        g = gap[a:b]
        their_stat, their_eff = THEIRS[m]
        eff[m] = abs((float(np.mean(g)) - f.bias) * 100 - their_eff) / their_eff
        stat[m] = abs(float(np.sum(g ** 2) / np.sqrt(len(g))) * 100
                      - their_stat) / their_stat

    a, b = WINDOWS["m0"]
    return {
        # the replication itself, worst horizon of the four
        "effect_max_rel_diff_over_horizons": max(eff.values()),
        "statistic_max_rel_diff_over_horizons": max(stat.values()),
        # and the headline cell in their own units
        "effect_m0": (float(np.mean(gap[a:b])) - f.bias) * 100,
        # that the fit is the one their script ran
        "forest_predictors": float(f.details["forest"]["n_features"]),
        "experts_built": float(len(f.experts)),
        "eta": float(f.eta),
        "forest_weight": float(f.weights["forest"]),
        # their verdict: no rejection at 10 or 20 percent
        "p_value_above_10pct": float(f.p_value > 0.10),
    }


EXPECTED = {
    # 0.03 percent at seed 0; tolerance is five seed standard deviations and
    # sits above the 0.70 percent worst seed of twenty.
    "effect_max_rel_diff_over_horizons": (0.00029, 0.012),
    # 0.56 percent at seed 0, worst seed 3.5 percent.
    "statistic_max_rel_diff_over_horizons": (0.00557, 0.055),
    # their 5.2227, in their units.
    "effect_m0": (5.2240, 0.06),
    # six donor outcomes and their 51 employment columns.
    "forest_predictors": (57.0, 0.0),
    "experts_built": (4.0, 0.0),
    "eta": (51.4307, 1e-9),
    "forest_weight": (0.1454, 0.02),
    "p_value_above_10pct": (1.0, 0.0),
}
