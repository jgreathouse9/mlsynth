"""VanillaSC reproduces Schulte et al. (2026)'s lost-autonomy SCM finding.

Path A: on the authors' 13-region autonomy panel, ``VanillaSC(backend=
"outcome-only")`` recovers a large post-trigger secessionist surge for both
Catalonia (2010) and the Faroe Islands (1994), tracking the authors' published
``SyntheticControlMethods`` synthetic (Catalonia correlation ~0.92) with a
comparable-or-tighter pre-treatment fit. The durable, dashboard-adjacent version
lives in ``benchmarks/cases/secession_scm.py``; this pins the reproducible
headline in the fast unit suite, offline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC

_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / "basedata" / "secession_autonomy.csv"
_BUNDLE = _ROOT / "benchmarks" / "reference" / "secession_scm"
_REF = json.loads((_BUNDLE / "reference.json").read_text())["values"]


def _fit(region, trigger):
    df = pd.read_csv(_DATA)[["region_name", "year", "av_sec1_all"]].copy()
    df["treat"] = ((df.region_name == region) & (df.year >= trigger)).astype(int)
    return VanillaSC({"df": df, "outcome": "av_sec1_all", "treat": "treat",
                      "unitid": "region_name", "time": "year",
                      "backend": "outcome-only", "display_graphs": False,
                      "seed": 123}).fit()


def _pre_rmse(res, trigger):
    ts = res.time_series
    yr = np.asarray(ts.time_periods).ravel()
    obs = np.asarray(ts.observed_outcome, float).ravel()
    cf = np.asarray(ts.counterfactual_outcome, float).ravel()
    pre = yr < trigger
    return float(np.sqrt(np.mean((obs[pre] - cf[pre]) ** 2))), yr, cf


@pytest.mark.parametrize("region,trigger", [("Catalonia", 2010), ("Faroe Islands", 1994)])
def test_large_positive_surge(region, trigger):
    """Both triggers produce a large positive post-treatment gap (the paper's finding)."""
    res = _fit(region, trigger)
    assert res.att > 10.0                              # a pronounced secessionist surge


@pytest.mark.parametrize("region,trigger,key", [
    ("Catalonia", 2010, "cat_paper_pre_rmse"),
    ("Faroe Islands", 1994, "far_paper_pre_rmse"),
])
def test_pre_fit_ties_or_beats_paper(region, trigger, key):
    """mlsynth's outcome-only pre-treatment fit is at least as tight as the
    authors' penalized ``SyntheticControlMethods`` fit."""
    res = _fit(region, trigger)
    pre_rmse, _, _ = _pre_rmse(res, trigger)
    assert pre_rmse <= _REF[key] + 1e-6


def test_tracks_authors_published_synthetic():
    """The mlsynth synthetic tracks the authors' published series for Catalonia."""
    res = _fit("Catalonia", 2010)
    _, yr, cf = _pre_rmse(res, 2010)
    paper = pd.read_csv(_BUNDLE / "reference_synth_catalonia.csv")
    ml = pd.Series(cf, index=yr).reindex(paper.year.values).to_numpy()
    corr = float(np.corrcoef(paper.synthetic.to_numpy(), ml)[0, 1])
    assert corr > 0.85
