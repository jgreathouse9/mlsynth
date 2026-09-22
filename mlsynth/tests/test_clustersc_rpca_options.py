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


# --------------------------------------------------------------------------
# fGRC restart counts
#
# Yamamoto & Hwang (2017) report that the number of local optima of the fGRC
# objective varies with the data and recommend "many random initial starts"
# (Section 5). The port honours that with 40 loading restarts and 40 k-means
# starts, but the counts were not reachable from the config, so a user meeting
# an unstable panel had no way to spend more restarts on it.
# --------------------------------------------------------------------------
def test_restart_counts_default_to_the_ported_values(germany_df):
    from mlsynth.config_models import CLUSTERSCConfig
    cfg = CLUSTERSCConfig(**_cfg(germany_df))
    assert cfg.fgrc_n_random == 40
    assert cfg.fgrc_nstart == 40


def test_restart_counts_reach_the_pipeline(germany_df):
    r = CLUSTERSC(_cfg(germany_df, cluster_method="fgrc",
                       fgrc_n_random=3, fgrc_nstart=5)).fit()
    meta = r.rpca.metadata
    assert meta["fgrc_n_random"] == 3
    assert meta["fgrc_nstart"] == 5


def test_more_restarts_never_worsen_the_objective(germany_df):
    """The restarts are a minimisation over random starts, so spending more of
    them cannot raise the retained loss. That is what the knob buys: a better
    optimum, not a different criterion."""
    lo = CLUSTERSC(_cfg(germany_df, cluster_method="fgrc",
                        fgrc_n_random=1, fgrc_nstart=1)).fit().rpca.metadata["fgrc_loss"]
    hi = CLUSTERSC(_cfg(germany_df, cluster_method="fgrc",
                        fgrc_n_random=25, fgrc_nstart=25)).fit().rpca.metadata["fgrc_loss"]
    assert hi <= lo + 1e-8


@pytest.mark.parametrize("kw", [
    dict(cluster_method="fgrc", fgrc_n_random=0),
    dict(cluster_method="fgrc", fgrc_nstart=0),
    dict(cluster_method="fgrc", fgrc_n_random=-1),
])
def test_invalid_restart_counts_raise_translated(germany_df, kw):
    with pytest.raises(MlsynthConfigError):
        CLUSTERSC(_cfg(germany_df, **kw))


def test_defaults_are_unchanged_by_the_new_fields(germany_df):
    """Passing the defaults explicitly must reproduce not passing them."""
    a = CLUSTERSC(_cfg(germany_df, cluster_method="fgrc")).fit().rpca
    b = CLUSTERSC(_cfg(germany_df, cluster_method="fgrc",
                       fgrc_n_random=40, fgrc_nstart=40)).fit().rpca
    assert a.att == pytest.approx(b.att, rel=1e-12)
    assert a.metadata["cluster_labels"] == b.metadata["cluster_labels"]


# --------------------------------------------------------------------------
# The Gap-statistic selector is not a configuration option
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kw", [
    dict(fgrc_k_selection="gap"),
    dict(fgrc_k_selection="fixed"),
    dict(fgrc_k_candidates=[2, 3]),
    dict(fgrc_gap_n_ref=5),
])
def test_the_gap_selector_is_not_reachable_from_the_config(germany_df, kw):
    """A root-cause analysis (see ``rpca/selection.py``) found the selection
    rule's acceptance rate under the null uncontrolled in the regime this
    library runs in, so no configuration wires it to ``fgrc_k``. The keys are
    refused by ``extra="forbid"`` and the selector stays a diagnostic the
    caller invokes deliberately."""
    with pytest.raises(MlsynthConfigError):
        CLUSTERSC(_cfg(germany_df, cluster_method="fgrc", **kw))


def test_fgrc_k_comes_only_from_the_configured_value(germany_df):
    a = CLUSTERSC(_cfg(germany_df, cluster_method="fgrc")).fit().rpca
    b = CLUSTERSC(_cfg(germany_df, cluster_method="fgrc", fgrc_k=2)).fit().rpca
    assert a.metadata["fgrc_k"] == 2
    assert a.att == pytest.approx(b.att, rel=1e-12)
    assert "fgrc_k_selection" not in a.metadata
    assert "fgrc_gap_curves" not in a.metadata
