"""Tests for the BLTVP estimator and its helper subpackage.

Covers:
    * BLTVPConfig validation (burn-in sanity, positivity bounds, chain count).
    * prepare_bltvp_inputs (pivot conventions, single-treated enforcement,
      minimum pre-periods).
    * The Gibbs sampler (shapes, finiteness, reproducibility under seed,
      recovery of which donor carries the time-varying relationship, and the
      static nesting when the dynamics are switched off).
    * Inference assembly (counterfactual, credible bands, ATT, no-post
      fallback).
    * BLTVP estimator class (smoke, error wrapping, edge panels).
    * Plotter (smoke).
    * Immutability of the frozen dataclasses.

Reference: Klinenberg, D. (2023), "Synthetic Control with Time Varying
Coefficients: A State Space Approach with Bayesian Shrinkage," Journal of
Business & Economic Statistics 41(4):1065-1076. Replicated against the paper's
Table 2 in ``benchmarks/cases/bltvp_prop99.py``.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from mlsynth import BLTVP
from mlsynth.config_models import BLTVPConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.bltvp_helpers.inference import compute_inference
from mlsynth.utils.bltvp_helpers.plotter import plot_bltvp
from mlsynth.utils.bltvp_helpers.sampler import gibbs_bltvp
from mlsynth.utils.bltvp_helpers.setup import prepare_bltvp_inputs
from mlsynth.utils.bltvp_helpers.structures import (
    BLTVPInference,
    BLTVPInputs,
    BLTVPPosterior,
    BLTVPResults,
)

FAST = dict(n_iter=300, burn_in=150, seed=0)


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------

def _panel(T=24, T0=16, n_donors=4, seed=0, tv_donor=None, effect=0.0):
    """Long panel. ``tv_donor`` gives one donor a random-walk coefficient."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        np.sin(np.pi * np.arange(1, T + 1) / 12),
        np.cos(np.pi * np.arange(1, T + 1) / 6),
        3 + np.sin(np.arange(1, T + 1)),
        np.cos(np.arange(1, T + 1) / 2),
    ][:n_donors]) * 10.0 + 100.0
    beta = np.zeros(n_donors); beta[0] = 1.0
    y = X @ beta
    if tv_donor is not None:
        y = y + X[:, tv_donor] * np.cumsum(rng.normal(scale=0.05, size=T))
    y = y + rng.normal(scale=0.1, size=T)
    y[T0:] += effect
    rows = []
    for t in range(T):
        rows.append({"unit": "treated", "time": t, "y": y[t],
                     "treat": int(t >= T0)})
        for j in range(n_donors):
            rows.append({"unit": f"d{j}", "time": t, "y": X[t, j], "treat": 0})
    return pd.DataFrame(rows)


def _cfg(df, **kw):
    base = dict(df=df, outcome="y", unitid="unit", time="time", treat="treat",
                display_graphs=False)
    base.update(FAST); base.update(kw)
    return base


# --------------------------------------------------------------------------
# config validation
# --------------------------------------------------------------------------

def test_burn_in_must_be_below_n_iter():
    with pytest.raises(MlsynthConfigError, match="burn_in"):
        BLTVPConfig(df=_panel(), outcome="y", unitid="unit", time="time",
                    treat="treat", n_iter=100, burn_in=100)


@pytest.mark.parametrize("bad", [{"n_iter": 5}, {"chains": 0},
                                 {"ci_alpha": 0.0}, {"ci_alpha": 1.0}])
def test_out_of_range_config_is_refused(bad):
    with pytest.raises((MlsynthConfigError, Exception)):
        BLTVPConfig(df=_panel(), outcome="y", unitid="unit", time="time",
                    treat="treat", **bad)


def test_config_forbids_unknown_fields():
    with pytest.raises(Exception):
        BLTVPConfig(df=_panel(), outcome="y", unitid="unit", time="time",
                    treat="treat", not_a_real_option=1)


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------

def test_prepare_inputs_shapes_and_labels():
    inp = prepare_bltvp_inputs(_panel(), "y", "unit", "time", "treat")
    assert inp.X_pre.shape == (inp.T0, inp.N)
    assert inp.X_all.shape == (inp.T, inp.N)
    assert inp.y_pre.shape == (inp.T0,)
    assert inp.y_target.shape == (inp.T,)
    assert inp.T0 == 16 and inp.T == 24 and inp.N == 4
    assert len(inp.donor_names) == inp.N


def test_too_few_pre_periods_is_refused():
    df = _panel(T=6, T0=1)
    with pytest.raises(MlsynthDataError, match="pre-treatment"):
        prepare_bltvp_inputs(df, "y", "unit", "time", "treat")


# --------------------------------------------------------------------------
# sampler
# --------------------------------------------------------------------------

def test_sampler_shapes_and_finiteness():
    inp = prepare_bltvp_inputs(_panel(), "y", "unit", "time", "treat")
    s = gibbs_bltvp(inp.y_target, inp.X_all, inp.T0, **FAST)
    keep = FAST["n_iter"] - FAST["burn_in"]
    assert s["predictive"].shape == (inp.T, keep)
    assert s["beta"].shape == (inp.N, keep)
    assert s["sqrt_theta"].shape == (inp.N, keep)
    assert s["sigma2"].shape == (keep,)
    assert np.all(np.isfinite(s["predictive"]))
    assert np.all(s["sigma2"] > 0)


def test_sampler_is_reproducible_under_seed():
    inp = prepare_bltvp_inputs(_panel(), "y", "unit", "time", "treat")
    a = gibbs_bltvp(inp.y_target, inp.X_all, inp.T0, **FAST)
    b = gibbs_bltvp(inp.y_target, inp.X_all, inp.T0, **FAST)
    assert np.array_equal(a["predictive"], b["predictive"])


def test_different_seeds_give_different_draws():
    inp = prepare_bltvp_inputs(_panel(), "y", "unit", "time", "treat")
    a = gibbs_bltvp(inp.y_target, inp.X_all, inp.T0, n_iter=300, burn_in=150, seed=0)
    b = gibbs_bltvp(inp.y_target, inp.X_all, inp.T0, n_iter=300, burn_in=150, seed=1)
    assert not np.array_equal(a["predictive"], b["predictive"])


def test_time_varying_donor_gets_the_larger_dynamic_scale():
    """The decomposition's whole point: |sqrt(theta)_j| should single out the
    donor whose relationship actually drifts."""
    inp = prepare_bltvp_inputs(
        _panel(T=40, T0=32, seed=3, tv_donor=1), "y", "unit", "time", "treat")
    s = gibbs_bltvp(inp.y_target, inp.X_all, inp.T0, n_iter=1500, burn_in=750, seed=0)
    scale = np.abs(s["sqrt_theta"]).mean(axis=1)
    assert scale.argmax() == 1, f"dynamic scale by donor: {scale}"


def test_static_mode_zeroes_the_dynamics():
    """``time_varying=False`` is the nesting the paper names: the model
    collapses to a Bayesian Lasso synthetic control."""
    inp = prepare_bltvp_inputs(_panel(), "y", "unit", "time", "treat")
    s = gibbs_bltvp(inp.y_target, inp.X_all, inp.T0, time_varying=False, **FAST)
    assert np.allclose(s["sqrt_theta"], 0.0)
    assert np.all(np.isfinite(s["predictive"]))


def test_intercept_adds_a_column():
    inp = prepare_bltvp_inputs(_panel(), "y", "unit", "time", "treat")
    s = gibbs_bltvp(inp.y_target, inp.X_all, inp.T0, intercept=True, **FAST)
    assert s["beta"].shape[0] == inp.N + 1


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------

def test_inference_bands_bracket_the_mean():
    inp = prepare_bltvp_inputs(_panel(), "y", "unit", "time", "treat")
    s = gibbs_bltvp(inp.y_target, inp.X_all, inp.T0, **FAST)
    inf = compute_inference(inp, s["predictive"], ci_alpha=0.05)
    assert np.all(inf.counterfactual_lower <= inf.counterfactual_mean + 1e-8)
    assert np.all(inf.counterfactual_mean <= inf.counterfactual_upper + 1e-8)
    assert inf.att_ci_lower <= inf.att_mean <= inf.att_ci_upper
    assert inf.att_samples.size == s["predictive"].shape[1]


def test_inference_without_a_post_window_reports_nan():
    """T0 == T is a helper-level guard, not reachable through the estimator:
    a panel with no post window has no treated observation for dataprep to
    find, so it is refused before inference is ever called."""
    full = prepare_bltvp_inputs(_panel(), "y", "unit", "time", "treat")
    no_post = BLTVPInputs(
        y_pre=full.y_pre, X_pre=full.X_pre, X_all=full.X_pre,
        y_target=full.y_pre, T0=full.T0, T=full.T0, N=full.N,
        treated_unit_name=full.treated_unit_name,
        donor_names=full.donor_names, time_labels=full.time_labels[:full.T0])
    s = gibbs_bltvp(full.y_pre, full.X_pre, full.T0, **FAST)
    inf = compute_inference(no_post, s["predictive"], ci_alpha=0.05)
    assert np.isnan(inf.att_mean) and inf.att_samples.size == 0


def test_a_panel_with_no_post_window_is_refused_at_ingestion():
    with pytest.raises(MlsynthDataError, match="treated"):
        prepare_bltvp_inputs(_panel(T=18, T0=18), "y", "unit", "time", "treat")


def test_inference_rejects_a_mismatched_draw_matrix():
    inp = prepare_bltvp_inputs(_panel(), "y", "unit", "time", "treat")
    with pytest.raises(ValueError, match="T"):
        compute_inference(inp, np.zeros((inp.T + 3, 10)), ci_alpha=0.05)


# --------------------------------------------------------------------------
# estimator
# --------------------------------------------------------------------------

def test_fit_smoke_and_contract():
    res = BLTVP(_cfg(_panel(effect=-5.0))).fit()
    assert isinstance(res, BLTVPResults)
    assert np.isfinite(res.effects.att)
    assert res.time_series.counterfactual_outcome.shape == (res.inputs.T,)
    assert res.time_series.estimated_gap.shape == (res.inputs.T,)
    assert set(res.weights.donor_weights) == {str(d) for d in res.inputs.donor_names}
    assert res.fit_diagnostics.rmse_pre >= 0.0
    assert res.method_details.method_name == "BLTVP"
    assert res.inference.ci_lower <= res.inference.ci_upper


def test_fit_recovers_the_sign_of_a_large_effect():
    res = BLTVP(_cfg(_panel(T=40, T0=32, seed=5, effect=-25.0),
                     n_iter=1500, burn_in=750)).fit()
    assert res.effects.att < 0.0


def test_dynamic_scales_are_reported_per_donor():
    res = BLTVP(_cfg(_panel())).fit()
    assert set(res.dynamic_scales) == {str(d) for d in res.inputs.donor_names}
    assert all(v >= 0.0 for v in res.dynamic_scales.values())


def test_accepts_a_config_object_as_well_as_a_dict():
    cfg = BLTVPConfig(**_cfg(_panel()))
    assert np.isfinite(BLTVP(cfg).fit().effects.att)


def test_invalid_config_dict_raises_config_error():
    with pytest.raises(MlsynthConfigError):
        BLTVP({"df": _panel(), "outcome": "y", "unitid": "unit",
               "time": "time", "treat": "treat", "n_iter": -1})


def test_unbalanced_panel_is_refused():
    df = _panel()
    with pytest.raises(MlsynthDataError):
        BLTVP(_cfg(df.iloc[:-1])).fit()


# --------------------------------------------------------------------------
# plotter and immutability
# --------------------------------------------------------------------------

def test_plotter_smoke():
    import matplotlib
    matplotlib.use("Agg")
    res = BLTVP(_cfg(_panel())).fit()
    plot_bltvp(res)


def test_display_graphs_path_runs():
    import matplotlib
    matplotlib.use("Agg")
    assert BLTVP(_cfg(_panel(), display_graphs=True)).fit() is not None


@pytest.mark.parametrize("cls,field", [(BLTVPInputs, "T0"),
                                       (BLTVPPosterior, "n_iter"),
                                       (BLTVPInference, "ci_alpha")])
def test_containers_are_frozen(cls, field):
    res = BLTVP(_cfg(_panel())).fit()
    obj = {BLTVPInputs: res.inputs, BLTVPPosterior: res.posterior,
           BLTVPInference: res.inference_detail}[cls]
    with pytest.raises(FrozenInstanceError):
        setattr(obj, field, 1)


# --------------------------------------------------------------------------
# failure paths
# --------------------------------------------------------------------------

def _multi_cohort_panel(T=12):
    rows = []
    for t in range(T):
        for unit, start in (("a", 6), ("b", 9)):
            rows.append({"unit": unit, "time": t, "y": float(t),
                         "treat": int(t >= start)})
        for j in range(3):
            rows.append({"unit": f"d{j}", "time": t, "y": float(t + j), "treat": 0})
    return pd.DataFrame(rows)


def test_staggered_adoption_is_refused():
    """BLTVP models one treated series; a staggered panel is a different
    estimand and is rejected instead of silently pooling cohorts."""
    with pytest.raises(MlsynthDataError, match="single treated unit"):
        prepare_bltvp_inputs(_multi_cohort_panel(), "y", "unit", "time", "treat")


def test_estimator_passes_through_an_ingestion_failure():
    with pytest.raises(MlsynthDataError, match="pre-treatment"):
        BLTVP(_cfg(_panel(T=6, T0=1))).fit()


@pytest.mark.parametrize("kwargs,match", [
    (dict(X=np.zeros(5), T0=3), "2-D"),
    (dict(X=np.zeros((7, 2)), T0=3), "must equal"),
    (dict(T0=1), r"must lie in"),
    (dict(T0=99), r"must lie in"),
    (dict(n_iter=100, burn_in=100), "burn_in"),
])
def test_sampler_refuses_malformed_input(kwargs, match):
    args = dict(y=np.arange(10, dtype=float), X=np.ones((10, 2)), T0=5,
                n_iter=50, burn_in=25, seed=0)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        gibbs_bltvp(args.pop("y"), args.pop("X"), args.pop("T0"), **args)
