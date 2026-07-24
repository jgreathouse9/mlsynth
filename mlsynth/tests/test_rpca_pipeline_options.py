"""Opt-in options on the RPCA-SC pipeline (``run_rpca``): fGRC clustering, the
HSVT/SVHT denoiser, and simplex (convex-hull) weights.

The load-bearing contract here is *backward compatibility*: the default
``PCP`` + FPCA-clustering + NNLS path must stay byte-identical to Bayani (2021),
so the ``clustersc_rpca_germany`` value-for-value benchmark keeps passing. The
new pieces are additive and default-off.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
from mlsynth.utils.clustersc_helpers.rpca.pipeline import run_rpca

_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def germany():
    """West-Germany reunification panel as run_rpca arrays (C < n, identified)."""
    df = pd.read_csv(_ROOT / "basedata" / "german_reunification.csv")
    wide = df.pivot(index="country", columns="year", values="gdp")
    years = sorted(wide.columns)
    T0 = sum(y < 1990 for y in years)
    y = wide.loc["West Germany", years].to_numpy(float)
    donors = [u for u in wide.index if u != "West Germany"]
    D = wide.loc[donors, years].to_numpy(float).T
    return y, D, donors, T0


def _fit(germany, **kw):
    y, D, dn, T0 = germany
    return run_rpca(treated_outcome=y, donor_outcomes=D, donor_names=dn, T0=T0, **kw)


# --------------------------------------------------------------------------
# Backward compatibility: the default path reproduces Bayani value-for-value
# --------------------------------------------------------------------------
def test_default_path_reproduces_bayani(germany):
    f = _fit(germany)                                  # PCP + fpca + nnls
    w = f.donor_weights
    assert w["Norway"] == pytest.approx(0.485, abs=5e-3)
    assert w["France"] == pytest.approx(0.354, abs=5e-3)
    assert w["New Zealand"] == pytest.approx(0.296, abs=5e-3)
    assert w["Austria"] == pytest.approx(0.023, abs=5e-3)
    assert f.metadata["cluster_method"] == "fpca"
    assert f.metadata["weight_objective"] == "nnls"
    assert f.metadata["rpca_method"] == "PCP"
    # FPCA metadata keys preserved for existing consumers
    assert {"fpca_rank", "k_clusters", "cluster_labels"} <= set(f.metadata)


# --------------------------------------------------------------------------
# Each new option runs and returns finite, well-formed output
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kw", [
    dict(rpca_method="HSVT"),
    dict(rpca_method="HSVT", hsvt_rank_method="cumvar"),
    dict(rpca_method="HSVT", hsvt_rank_method="fixed", hsvt_rank=3),
    dict(cluster_method="fgrc"),
    dict(cluster_method="fgrc", fgrc_k=3),
    dict(weight_objective="simplex"),
    dict(rpca_method="HSVT", cluster_method="fgrc", weight_objective="simplex"),
])
def test_options_run_and_are_finite(germany, kw):
    f = _fit(germany, **kw)
    assert np.isfinite(f.att)
    assert np.isfinite(f.pre_rmse)
    assert len(f.donor_weights) >= 1


def test_simplex_weights_are_on_the_simplex(germany):
    f = _fit(germany, weight_objective="simplex")
    assert np.isclose(sum(f.donor_weights.values()), 1.0, atol=1e-6)
    assert all(v >= -1e-8 for v in f.donor_weights.values())


def test_cft_refit_threads_new_options(germany):
    """CFT prediction-interval bootstrap recurses through run_rpca; the new
    options must thread through the refit consistently (exercises the recursive
    call with cluster_method/weight_objective/hsvt kwargs)."""
    f = _fit(germany, cluster_method="fgrc", weight_objective="simplex",
             compute_cft_pi=True, cft_sims=10)
    assert np.isfinite(f.att)
    assert f.metadata.get("cft_inference") is not None


def test_hsvt_reports_rank_metadata(germany):
    f = _fit(germany, rpca_method="HSVT")
    assert f.metadata["rpca_method"] == "HSVT"
    assert f.metadata["hsvt_rank_method"] == "usvt"
    assert f.metadata["hsvt_rank"] >= 1


def test_fgrc_reports_cluster_metadata(germany):
    f = _fit(germany, cluster_method="fgrc")
    assert f.metadata["cluster_method"] == "fgrc"
    assert {"fgrc_c1", "fgrc_c2", "fgrc_k", "fgrc_loss", "cluster_labels"} <= set(f.metadata)


# --------------------------------------------------------------------------
# The recommended full pipeline reproduces the validated prototype
# --------------------------------------------------------------------------
def test_full_pipeline_matches_prototype(germany):
    """fGRC cluster -> HSVT/SVHT denoise -> simplex weights recovers the
    West-Germany prototype: a coherent core, France/USA lead, ATT ~ -1543."""
    f = _fit(germany, rpca_method="HSVT", cluster_method="fgrc",
             weight_objective="simplex", fgrc_k=2)
    assert np.isclose(sum(f.donor_weights.values()), 1.0, atol=1e-6)
    lead = max(f.donor_weights.items(), key=lambda kv: kv[1])[0]
    assert lead in {"France", "USA"}
    assert f.att == pytest.approx(-1543.5, abs=30.0)     # regression pin vs prototype
    assert -1700.0 < f.att < -1300.0                     # and inside the Bayani ballpark


# --------------------------------------------------------------------------
# Failure — invalid options raise translated exceptions
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kw,exc", [
    (dict(cluster_method="bogus"), MlsynthConfigError),
    (dict(weight_objective="bogus"), MlsynthConfigError),
    (dict(rpca_method="bogus"), MlsynthEstimationError),
])
def test_invalid_option_raises_translated(germany, kw, exc):
    with pytest.raises(exc):
        _fit(germany, **kw)
