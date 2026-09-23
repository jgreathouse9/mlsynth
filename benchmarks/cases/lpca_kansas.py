"""Cross-validation benchmark: LPCA vs Feng's own R (2012 Kansas tax cut).

Cross-validates mlsynth's ``LPCA`` against the method's own author -- the
``knn.index`` / ``lpca`` / ``sel.r`` functions in
``yingjieum/Replication_NonlinearFactorModel_2023`` -- on the panel behind
Section 6.1 of Feng (2024). Both sides read ``basedata/kansas_taxcut.csv``, and
the reference run asserts that file equals the ``kansas.rda`` Feng ships before
using it, so the check cannot drift onto a different vintage of the data.

The reference is a captured live run pinned under
``benchmarks/reference/lpca_kansas/`` (R 4.3.3, replication repo commit
``ca34fba``, Rfast and irlba at the SHAs in ``benchmarks/R/install_feng_lpca.sh``,
data checksum recorded).

Why the tolerances are tight. This is the same algorithm on the same bytes, and
the tuning is fully determined by the reference script -- ``K = round(n^(2/3))``
= 14, 40 differenced quarters for matching, at most three local components -- so
nothing is left for either implementation to choose. The one deliberate
difference is the linear algebra: the reference calls ``irlba`` for a truncated
SVD of the neighbourhood, mlsynth takes the eigendecomposition of that
neighbourhood's Gram matrix. Only one row of the reconstruction is ever read and
the right singular vectors cancel from it, so the two are the same computation;
they agree to about 5e-14, and ``irlba``'s iterative tolerance is what sets the
floor, not the port.

A note on the paper's other arm. Feng's script also fits a simplex synthetic
control for comparison. That is not run here: mlsynth's ``LPCA`` implements
local PCA, the SC comparison belongs to a different estimator, and the paper's
November 2023 version reported it with a centring defect the author corrected in
July 2024 (recorded in ``benchmarks/reference/lpca_kansas/README.md``).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference import load_reference, reference_value

_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "basedata")
_CASE = "lpca_kansas"

# Feng_LPCA_app.R fixes all three.
_MATCH_PERIODS = 40
_MAX_COMPONENTS = 3


def _panel() -> pd.DataFrame:
    """The Kansas panel in growth rates, as Section 6.1 analyses it.

    The paper works with first differences of log GDP per capita in percent.
    ``LPCA`` takes the outcome as given, so the transformation happens here --
    the same place the reference script does it.
    """
    df = pd.read_csv(os.path.join(_BASE, "kansas_taxcut.csv"))
    df = df.sort_values(["state", "year_qtr"])
    df["growth"] = df.groupby("state")["lngdpcapita"].diff() * 100.0
    return df.dropna(subset=["growth"]).reset_index(drop=True)


def run() -> dict:
    try:
        ref = load_reference(_CASE)["values"]
    except FileNotFoundError as exc:
        raise BenchmarkSkipped(str(exc)) from exc
    if not int(ref.get("csv_matches_rda", 0)):
        raise BenchmarkSkipped(
            "the reference run could not confirm basedata/kansas_taxcut.csv "
            "against Feng's kansas.rda; regenerate the bundle with the "
            "replication repo present"
        )

    from mlsynth import LPCA

    cfg = {"df": _panel(), "outcome": "growth", "treat": "treated",
           "unitid": "state", "time": "year_qtr",
           "match_periods": _MATCH_PERIODS, "max_components": _MAX_COMPONENTS,
           "display_graphs": False}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = LPCA(cfg).fit()

    counterfactual = np.asarray(res.counterfactual, float)
    n_pre = int(res.metadata["pre_periods_reported"])
    post = counterfactual[n_pre:]
    ref_post = np.array([load_reference(_CASE)["weights"][f"cf_post_{i + 1:02d}"]
                         for i in range(post.size)], dtype=float)

    got = {
        "lpca_att": float(res.att),
        "counterfactual_post_mean": float(post.mean()),
        "observed_post_mean": float(
            np.asarray(res.time_series.observed_outcome, float)[n_pre:].mean()),
        "pre_period_rmse": float(res.pre_rmse),
        "K": float(res.n_neighbours),
        "neighbourhood_size": float(res.neighbourhood_size),
        "local_rank": float(res.local_rank),
        "post_periods_observed_below": float((post > np.asarray(
            res.time_series.observed_outcome, float)[n_pre:]).sum()),
        # The whole counterfactual path, not just its mean: a cell-by-cell claim.
        "counterfactual_max_abs_dev": float(np.abs(post - ref_post).max()),
    }

    rows = [{"period": i + 1, "mlsynth": float(post[i]), "reference": float(ref_post[i])}
            for i in range(post.size)]
    return {
        **got,
        "rows": rows,
        "mlsynth_call": {"estimator": "LPCA",
                         "config": {k: v for k, v in cfg.items() if k != "df"}},
        "reference": {"impl": "Feng's replication scripts (live run, captured)",
                      "version": "Replication_NonlinearFactorModel_2023 @ ca34fba"},
    }


_r = lambda k: reference_value(_CASE, k)

# Same algorithm, same bytes, no free tuning: the only slack is irlba's
# iterative tolerance against an exact eigendecomposition.
EXPECTED = {
    "lpca_att": (_r("lpca_att"), 1e-8),
    "counterfactual_post_mean": (_r("counterfactual_post_mean"), 1e-8),
    "observed_post_mean": (_r("observed_post_mean"), 1e-10),
    "pre_period_rmse": (_r("pre_period_rmse"), 1e-8),
    "K": (_r("K"), 0),
    "neighbourhood_size": (_r("neighbourhood_size"), 0),
    "local_rank": (_r("local_rank"), 0),
    "post_periods_observed_below": (_r("post_periods_observed_below"), 0),
    "counterfactual_max_abs_dev": (0.0, 1e-8),
}
