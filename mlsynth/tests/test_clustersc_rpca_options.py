"""CLUSTERSC public-config surface for the RPCA-SC options: fGRC clustering,
the HSVT/SVHT denoiser, and simplex (convex-hull) weights.

Defaults must reproduce Bayani (2021); the new config fields must reach the
``run_rpca`` pipeline and drive the validated behaviour end-to-end.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import CLUSTERSC
from mlsynth.exceptions import MlsynthConfigError

_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def germany_df():
    df = pd.read_csv(_ROOT / "basedata" / "german_reunification.csv")
    df["treat"] = df["Reunification"].astype(int)
    return df


def _cfg(df, **kw):
    return dict(df=df, outcome="gdp", treat="treat", unitid="country", time="year",
                method="rpca", display_graphs=False, **kw)


def test_default_config_reproduces_bayani(germany_df):
    r = CLUSTERSC(_cfg(germany_df, rpca_method="PCP")).fit()
    w = r.donor_weights
    assert w["Norway"] == pytest.approx(0.485, abs=5e-3)
    assert w["France"] == pytest.approx(0.354, abs=5e-3)
    assert w["New Zealand"] == pytest.approx(0.296, abs=5e-3)


@pytest.mark.parametrize("kw", [
    dict(rpca_method="HSVT"),
    dict(cluster_method="fgrc"),
    dict(weight_objective="simplex"),
    dict(rpca_method="HSVT", cluster_method="fgrc", weight_objective="simplex", fgrc_k=2),
])
def test_new_options_run_end_to_end(germany_df, kw):
    r = CLUSTERSC(_cfg(germany_df, **kw)).fit()
    assert np.isfinite(r.att)
    assert np.isfinite(r.pre_rmse)
    assert len(r.donor_weights) >= 1


def test_full_pipeline_matches_prototype(germany_df):
    """fGRC -> HSVT/SVHT -> simplex through the public estimator reproduces the
    validated West-Germany prototype (ATT ~ -1543, weights on the simplex)."""
    r = CLUSTERSC(_cfg(germany_df, rpca_method="HSVT", cluster_method="fgrc",
                       weight_objective="simplex", fgrc_k=2)).fit()
    assert r.att == pytest.approx(-1543.5, abs=30.0)
    assert np.isclose(sum(r.donor_weights.values()), 1.0, atol=1e-6)


@pytest.mark.parametrize("kw", [
    dict(cluster_method="bogus"),
    dict(weight_objective="bogus"),
    dict(rpca_method="bogus"),
])
def test_invalid_config_raises_translated(germany_df, kw):
    with pytest.raises(MlsynthConfigError):
        CLUSTERSC(_cfg(germany_df, **kw))
