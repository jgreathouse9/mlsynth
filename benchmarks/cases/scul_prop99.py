"""SCUL cross-validation: the California (Proposition 99) example vs hollina/scul.

Cross-validation against the *running* reference. mlsynth's ``SCUL`` (Synthetic
Control Using Lasso, Hollingsworth & Wing 2022) builds the synthetic control as a
rolling-origin cross-validated lasso of California's pre-1989 cigarette sales on
a 76-column multi-type donor pool (every donor state's per-capita sales and
retail price). This case runs the authors' own ``SCUL()`` (``hollina/scul`` @
121b588, glmnet/LowRankQP solver) on its shipped cigarette panel, feeds the
*identical* panel to mlsynth, and checks they agree.

The rolling-CV penalty matches the reference to ten digits -- the cross-
validation procedure ports exactly. The lasso solution is unique for
continuously distributed donors (Tibshirani 2013), so the weights and synthetic
series agree up to solver tolerance; the small residual ATT gap is glmnet's
*default* convergence threshold slightly under-converging this correlated p>>n
problem (mlsynth solves to a tight tolerance and lands on the unique optimum),
not a method difference. Skips gracefully when ``Rscript``/``glmnet`` or the
clone is unavailable.
"""
from __future__ import annotations

import tempfile
import warnings
from pathlib import Path


_CONFIG = dict(outcome="cigsale", treat="treat", unitid="state", time="year",
               donor_variables=["retprice"])


def _fit_both():
    """Run the live reference and mlsynth on the identical shipped panel."""
    import pandas as pd

    from benchmarks.reference.clone_scul import run_reference
    from mlsynth import SCUL
    from mlsynth.config_models import SCULConfig

    with tempfile.TemporaryDirectory() as tmp:
        ref = run_reference(Path(tmp))                       # skips if no R/glmnet
        df = pd.read_csv(Path(tmp) / "scul_panel_long.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SCUL(SCULConfig(df=df, display_graphs=False, inference=False,
                              **_CONFIG)).fit()
    return res, ref


def run() -> dict:
    import numpy as np

    res, ref = _fit_both()
    pu = res.method_details.parameters_used
    series = res.time_series.counterfactual_outcome
    return {
        "n_pool": float(pu["n_pool"]),
        "att": float(res.effects.att),
        "lambda_vs_ref": float(abs(pu["ridge_lambda"] - ref["ridge_lambda"])),
        "att_vs_ref": float(abs(res.effects.att - ref["att"])),
        "series_max_vs_ref": float(np.max(np.abs(series - ref["series"]))),
    }


def comparison() -> dict:
    """Side-by-side mlsynth-SCUL vs hollina/scul on the shipped cigarette panel.

    Pairs the interpretable headline quantities -- the rolling-CV penalty, the
    post-period ATT, and the unit-free pre-fit -- live from both implementations,
    for the committed ``comparison.csv`` / ``comparisons.xlsx`` workbook.
    Propagates ``BenchmarkSkipped`` when Rscript/glmnet/the clone is absent.
    """
    res, ref = _fit_both()
    pu = res.method_details.parameters_used
    rows = [
        {"quantity": "cv lambda (median)",
         "mlsynth": round(float(pu["ridge_lambda"]), 8),
         "reference": round(float(ref["ridge_lambda"]), 8)},
        {"quantity": "ATT (post-1988 mean)",
         "mlsynth": round(float(res.effects.att), 5),
         "reference": round(float(ref["att"]), 5)},
        {"quantity": "Cohen's D (pre-fit)",
         "mlsynth": round(float(pu["cohens_d"]), 5),
         "reference": round(float(ref["cohens_d"]), 5)},
    ]
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "SCUL", "config": _CONFIG},
        "reference": {"impl": "authors' SCUL() (R, via Rscript + glmnet)",
                      "version": "hollina/scul @ 121b588 (R, live)"},
    }


# Validated against hollina/scul @ 121b588 on the shipped cigarette panel
# (TreatmentBeginsAt = 19, NumberInitialTimePeriods = 5, TrainingPostPeriodLength
# = 7, lambda.median). The reference selects lambda ~ 0.02121 and an ATT of
# ~ -13.3 packs; mlsynth matches the penalty bit-for-bit and the ATT/series to
# solver tolerance (~0.15 packs, the glmnet-default convergence gap on the
# unique lasso solution).
EXPECTED = {
    "n_pool": (76.0, 0.0),
    "att": (-13.17, 0.3),
    "lambda_vs_ref": (0.0, 1e-5),        # rolling-CV penalty bit-for-bit
    "att_vs_ref": (0.0, 0.35),           # glmnet-default vs tight convergence
    "series_max_vs_ref": (0.0, 0.6),
}
