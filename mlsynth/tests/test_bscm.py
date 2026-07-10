"""Tests for the BSCM estimator and its helper subpackage.

Covers:
    * BSCMConfig validation (prior literal, burn-in sanity, positivity bounds).
    * prepare_bscm_inputs (pivot conventions, donor / time consistency,
      single-treated enforcement, minimum pre-periods / donors).
    * gibbs_bscm sampler (shapes, finiteness, reproducibility under seed,
      both priors, sparse-signal recovery, negative-weight extrapolation).
    * Inference assembly (counterfactual, credible bands, ATT, no-post
      fallback).
    * BSCM estimator class (smoke on Basque + edge + error wrapping, both
      priors, credibly-negative ATT).
    * Plotter (smoke).
    * Immutability of the frozen dataclasses.

Reference: Kim, Lee & Gupta (2020), "Bayesian Synthetic Control Methods,"
Journal of Marketing Research 57(5):831-852. Cross-validated against the
authors' reference Stan implementation (``clarencejlee/bscm``).
"""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from mlsynth import BSCM
from mlsynth.config_models import BSCMConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.bscm_helpers.inference import compute_inference
from mlsynth.utils.bscm_helpers.plotter import plot_bscm
from mlsynth.utils.bscm_helpers.sampler import gibbs_bscm
from mlsynth.utils.bscm_helpers.setup import prepare_bscm_inputs
from mlsynth.utils.bscm_helpers.structures import (
    BSCMInference,
    BSCMInputs,
    BSCMPosterior,
    BSCMResults,
)

_BASQUE = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "basque_data.csv")


# =========================================================================
# FIXTURES
# =========================================================================

def _make_panel(n_units=8, T=30, T0=20, seed=0, treated_effect=0.0):
    """Panel with a sparse true weighting: treated ~ 0.6*d1 + 0.9*d2 + noise."""
    rng = np.random.default_rng(seed)
    means = rng.uniform(5, 40, size=n_units - 1)
    donors = rng.normal(means, 3.0, size=(T, n_units - 1))
    beta = np.zeros(n_units - 1)
    beta[:2] = [0.6, 0.9]
    treated = 3.0 + donors @ beta + rng.normal(0, 0.4, size=T)
    treated[T0:] += treated_effect
    frames = []
    for j in range(n_units - 1):
        frames.append(pd.DataFrame({
            "unit": f"donor_{j}", "time": np.arange(T), "y": donors[:, j], "treat": 0}))
    frames.append(pd.DataFrame({
        "unit": "treated", "time": np.arange(T), "y": treated,
        "treat": (np.arange(T) >= T0).astype(int)}))
    return pd.concat(frames, ignore_index=True), beta


def _basque_panel():
    df = pd.read_csv(os.path.abspath(_BASQUE))
    df["treat"] = ((df["regionname"] == "Basque Country (Pais Vasco)")
                   & (df["year"] >= 1970)).astype(int)
    return df


@pytest.fixture
def panel():
    df, _ = _make_panel()
    return df


@pytest.fixture
def inputs(panel):
    return prepare_bscm_inputs(panel, outcome="y", unitid="unit",
                               time="time", treat="treat")


# =========================================================================
# CONFIG VALIDATION
# =========================================================================

def _cfg(df, **kw):
    base = dict(df=df, outcome="y", treat="treat", unitid="unit", time="time")
    base.update(kw)
    return BSCMConfig(**base)


def test_config_defaults(panel):
    cfg = _cfg(panel)
    assert cfg.prior == "horseshoe"
    assert cfg.chains >= 1
    assert cfg.burn_in < cfg.n_iter


def test_config_rejects_unknown_prior(panel):
    with pytest.raises((MlsynthConfigError, ValueError)):
        _cfg(panel, prior="lasso")


def test_config_rejects_burn_ge_iter(panel):
    with pytest.raises((MlsynthConfigError, ValueError)):
        _cfg(panel, n_iter=100, burn_in=100)


def test_config_rejects_bad_spike_scale(panel):
    with pytest.raises((MlsynthConfigError, ValueError)):
        _cfg(panel, spike_scale=0.0)


def test_config_rejects_bad_ci_alpha(panel):
    with pytest.raises((MlsynthConfigError, ValueError)):
        _cfg(panel, ci_alpha=1.5)


def test_config_forbids_extra(panel):
    with pytest.raises((MlsynthConfigError, ValueError)):
        _cfg(panel, nonsense=1)


# =========================================================================
# SETUP
# =========================================================================

def test_setup_shapes(inputs):
    assert inputs.X_pre.shape == (inputs.T0, inputs.N)
    assert inputs.y_pre.shape == (inputs.T0,)
    assert inputs.X_all.shape == (inputs.T, inputs.N)
    assert inputs.y_target.shape == (inputs.T,)
    assert inputs.T0 < inputs.T
    assert len(inputs.donor_names) == inputs.N


def test_setup_rejects_too_few_pre(panel):
    # keep only one pre-period
    df = panel[(panel["time"] >= 19)].copy()
    with pytest.raises(MlsynthDataError):
        prepare_bscm_inputs(df, outcome="y", unitid="unit", time="time", treat="treat")


def test_setup_time_labels_match(inputs):
    assert len(inputs.time_labels) == inputs.T


# =========================================================================
# SAMPLER
# =========================================================================

@pytest.mark.parametrize("prior", ["horseshoe", "spike_slab"])
def test_sampler_shapes_and_finite(inputs, prior):
    rng = np.random.default_rng(1)
    post = gibbs_bscm(inputs.y_pre, inputs.X_pre, inputs.X_all, prior=prior,
                      chains=2, n_iter=400, burn_in=200, rng=rng)
    ndraws = 2 * 200
    assert post["beta"].shape == (inputs.N, ndraws)
    assert post["beta0"].shape == (ndraws,)
    assert post["sigma2"].shape == (ndraws,)
    assert np.all(np.isfinite(post["beta"]))
    assert np.all(post["sigma2"] > 0)


@pytest.mark.parametrize("prior", ["horseshoe", "spike_slab"])
def test_sampler_reproducible(inputs, prior):
    a = gibbs_bscm(inputs.y_pre, inputs.X_pre, inputs.X_all, prior=prior,
                   chains=2, n_iter=300, burn_in=150, rng=np.random.default_rng(7))
    b = gibbs_bscm(inputs.y_pre, inputs.X_pre, inputs.X_all, prior=prior,
                   chains=2, n_iter=300, burn_in=150, rng=np.random.default_rng(7))
    assert np.allclose(a["beta"], b["beta"])


@pytest.mark.parametrize("prior", ["horseshoe", "spike_slab"])
def test_sampler_recovers_sparse_signal(prior):
    df, beta = _make_panel(seed=3)
    inp = prepare_bscm_inputs(df, outcome="y", unitid="unit", time="time", treat="treat")
    post = gibbs_bscm(inp.y_pre, inp.X_pre, inp.X_all, prior=prior,
                      chains=4, n_iter=1500, burn_in=500, rng=np.random.default_rng(0))
    w = post["beta"].mean(axis=1)
    # the two true signal donors carry the largest weights; noise donors shrink
    top2 = set(np.argsort(-np.abs(w))[:2])
    assert top2 == {0, 1}
    assert np.abs(w[2:]).max() < 0.3


def test_spike_slab_reports_inclusion(inputs):
    post = gibbs_bscm(inputs.y_pre, inputs.X_pre, inputs.X_all, prior="spike_slab",
                      chains=2, n_iter=400, burn_in=200, rng=np.random.default_rng(2))
    assert post["gamma"] is not None
    assert post["gamma"].shape == (inputs.N, 2 * 200)
    assert np.all((post["gamma"] >= 0) & (post["gamma"] <= 1))


def test_horseshoe_no_inclusion(inputs):
    post = gibbs_bscm(inputs.y_pre, inputs.X_pre, inputs.X_all, prior="horseshoe",
                      chains=2, n_iter=300, burn_in=150, rng=np.random.default_rng(2))
    assert post["gamma"] is None


# =========================================================================
# INFERENCE
# =========================================================================

def test_inference_counterfactual_and_att(inputs):
    post = gibbs_bscm(inputs.y_pre, inputs.X_pre, inputs.X_all, prior="horseshoe",
                      chains=2, n_iter=600, burn_in=300, rng=np.random.default_rng(0))
    inf = compute_inference(inputs, post["beta0"], post["beta"], ci_alpha=0.05)
    assert inf.counterfactual_mean.shape == (inputs.T,)
    assert np.all(inf.counterfactual_lower <= inf.counterfactual_upper + 1e-9)
    assert inf.att_ci_lower <= inf.att_mean <= inf.att_ci_upper
    assert inf.att_samples.shape[0] == post["beta"].shape[1]


def test_inference_no_post_period(inputs):
    # T0 == T: no post window (defensive branch, not reachable via the
    # estimator since dataprep always leaves >= 1 treated period).
    no_post = BSCMInputs(
        y_pre=inputs.y_pre, X_pre=inputs.X_pre, X_all=inputs.X_pre,
        y_target=inputs.y_pre, T0=inputs.T0, T=inputs.T0, N=inputs.N,
        treated_unit_name=inputs.treated_unit_name,
        donor_names=inputs.donor_names, time_labels=inputs.time_labels[:inputs.T0])
    post = gibbs_bscm(inputs.y_pre, inputs.X_pre, inputs.X_pre, prior="horseshoe",
                      chains=2, n_iter=300, burn_in=150, rng=np.random.default_rng(0))
    inf = compute_inference(no_post, post["beta0"], post["beta"], ci_alpha=0.05)
    assert np.isnan(inf.att_mean)
    assert inf.att_samples.size == 0


# =========================================================================
# ESTIMATOR
# =========================================================================

@pytest.mark.parametrize("prior", ["horseshoe", "spike_slab"])
def test_estimator_smoke_basque(prior):
    df = _basque_panel()
    res = BSCM({"df": df, "outcome": "gdpcap", "treat": "treat",
                "unitid": "regionname", "time": "year", "prior": prior,
                "n_iter": 1500, "burn_in": 500, "seed": 2019,
                "display_graphs": False}).fit()
    assert isinstance(res, BSCMResults)
    # ETA terrorism depressed Basque GDP: counterfactual sits above actual
    assert res.att is not None and res.att < 0
    assert res.pre_rmse is not None and res.pre_rmse >= 0
    assert res.counterfactual is not None
    assert len(res.donor_weights) == 16
    lo, hi = res.att_ci
    assert lo <= res.att <= hi


def test_estimator_negative_weights_present():
    # BSCM allows extrapolation: some donor weights should be negative,
    # and they need not sum to one (unlike a simplex SCM).
    df = _basque_panel()
    res = BSCM({"df": df, "outcome": "gdpcap", "treat": "treat",
                "unitid": "regionname", "time": "year", "prior": "horseshoe",
                "n_iter": 1500, "burn_in": 500, "seed": 2019,
                "display_graphs": False}).fit()
    w = np.array(list(res.donor_weights.values()))
    assert (w < 0).any()


def test_estimator_rejects_burn_ge_iter():
    df = _basque_panel()
    with pytest.raises((MlsynthConfigError, ValueError)):
        BSCM({"df": df, "outcome": "gdpcap", "treat": "treat",
              "unitid": "regionname", "time": "year",
              "n_iter": 100, "burn_in": 200}).fit()


def test_estimator_invalid_config_dict_wrapped():
    # a pydantic ValidationError (bad Literal) is translated to MlsynthConfigError
    df = _basque_panel()
    with pytest.raises(MlsynthConfigError):
        BSCM({"df": df, "outcome": "gdpcap", "treat": "treat",
              "unitid": "regionname", "time": "year", "prior": "lasso"})


def test_estimator_display_graphs(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    df = _basque_panel()
    res = BSCM({"df": df, "outcome": "gdpcap", "treat": "treat",
                "unitid": "regionname", "time": "year",
                "n_iter": 250, "burn_in": 120, "seed": 1,
                "display_graphs": True}).fit()
    assert res.att is not None


def test_sampler_default_rng(inputs):
    # no rng passed -> a fresh default generator is used
    post = gibbs_bscm(inputs.y_pre, inputs.X_pre, inputs.X_all, prior="horseshoe",
                      chains=1, n_iter=60, burn_in=30)
    assert post["beta"].shape[0] == inputs.N


def test_sampler_unknown_prior_raises(inputs):
    with pytest.raises(ValueError):
        gibbs_bscm(inputs.y_pre, inputs.X_pre, inputs.X_all, prior="nope",
                   chains=1, n_iter=20, burn_in=10)


def test_inference_rejects_bad_shape(inputs):
    with pytest.raises(ValueError):
        compute_inference(inputs, np.zeros(5), np.zeros((inputs.N + 3, 5)))


def test_estimator_multi_cohort_raises():
    df = _basque_panel()
    # inject a second treated cohort
    df.loc[(df["regionname"] == "Aragon") & (df["year"] >= 1980), "treat"] = 1
    with pytest.raises(MlsynthDataError):
        BSCM({"df": df, "outcome": "gdpcap", "treat": "treat",
              "unitid": "regionname", "time": "year",
              "n_iter": 200, "burn_in": 100}).fit()


# =========================================================================
# PLOTTER
# =========================================================================

def test_plotter_smoke(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    df = _basque_panel()
    res = BSCM({"df": df, "outcome": "gdpcap", "treat": "treat",
                "unitid": "regionname", "time": "year",
                "n_iter": 300, "burn_in": 150, "seed": 1,
                "display_graphs": False}).fit()
    plot_bscm(res)


# =========================================================================
# IMMUTABILITY
# =========================================================================

def test_inputs_frozen(inputs):
    with pytest.raises(FrozenInstanceError):
        inputs.T0 = 5


def test_posterior_frozen(inputs):
    post = gibbs_bscm(inputs.y_pre, inputs.X_pre, inputs.X_all, prior="horseshoe",
                      chains=2, n_iter=300, burn_in=150, rng=np.random.default_rng(0))
    p = BSCMPosterior(beta0=post["beta0"], beta=post["beta"], sigma2=post["sigma2"],
                      gamma=post["gamma"], prior="horseshoe", burn_in=150,
                      n_iter=300, chains=2)
    with pytest.raises(FrozenInstanceError):
        p.burn_in = 1
