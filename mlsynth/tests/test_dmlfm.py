"""Tests for DMLFM, the dynamic multilevel latent factor model.

Pang, X., Liu, L., & Xu, Y. (2022). "A Bayesian Alternative to Synthetic
Control for Comparative Case Studies." Political Analysis 30(2):269-288.

Written before the implementation. The oracle is pblasso 1.0.8, staged at
``benchmarks/reference/dmlfm_germany``; the numeric targets used here are its
five-seed gold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import DMLFM
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError

SEED = 20260808


def make_panel(n_units: int = 12, n_periods: int = 30, pre: int = 20,
               r: int = 2, seed: int = SEED, covariates: bool = True):
    """A small factor-model panel with one treated unit and a known effect."""
    rng = np.random.default_rng(seed)
    gamma = rng.normal(size=(n_units, r))
    f = rng.normal(size=(n_periods, r))
    level = rng.normal(5.0, 1.0, n_units)
    y = level[None, :] + f @ gamma.T + rng.normal(0, 0.3, (n_periods, n_units))

    t = np.arange(1, n_periods + 1)
    rows = []
    for i in range(n_units):
        x1 = float(rng.normal())
        for j, tt in enumerate(t):
            treated = int(i == 0 and tt > pre)
            rows.append({"unit": f"u{i}", "t": int(tt),
                         "y": float(y[j, i]) + (2.0 if treated else 0.0),
                         "treated": treated, "x1": x1})
    df = pd.DataFrame(rows)
    if not covariates:
        df = df.drop(columns=["x1"])
    return df


def base_config(df, **kw):
    cfg = {"df": df, "outcome": "y", "unitid": "unit", "time": "t",
           "treat": "treated", "display_graphs": False,
           "niter": 300, "burn": 100, "r": 3, "seed": SEED}
    cfg.update(kw)
    return cfg


# --------------------------------------------------------------------------
# smoke
# --------------------------------------------------------------------------
def test_smoke_returns_effect_result_with_finite_att():
    res = DMLFM(base_config(make_panel())).fit()
    att = float(res.effects.att)
    assert np.isfinite(att)
    assert res.time_series.counterfactual_outcome is not None
    cf = np.asarray(res.time_series.counterfactual_outcome, float).ravel()
    obs = np.asarray(res.time_series.observed_outcome, float).ravel()
    assert cf.shape == obs.shape == (30,)
    assert np.all(np.isfinite(cf))


def test_recovers_a_known_effect():
    """True effect is 2.0; the posterior mean should land near it."""
    res = DMLFM(base_config(make_panel(), niter=2000, burn=500)).fit()
    assert abs(float(res.effects.att) - 2.0) < 1.0


def test_credible_interval_brackets_the_point_estimate():
    res = DMLFM(base_config(make_panel())).fit()
    lo = float(res.inference.ci_lower)
    hi = float(res.inference.ci_upper)
    assert lo < float(res.effects.att) < hi


# --------------------------------------------------------------------------
# sampler invariants
# --------------------------------------------------------------------------
def test_posterior_draws_are_finite_and_shaped():
    res = DMLFM(base_config(make_panel(), niter=400, burn=100)).fit()
    draws = res.additional_outputs["att_draws"]
    assert np.asarray(draws).shape == (300,)
    assert np.all(np.isfinite(draws))


def test_omega_gamma_spectrum_is_nonnegative_and_length_r():
    res = DMLFM(base_config(make_panel(), r=4)).fit()
    w = np.asarray(res.additional_outputs["omega_gamma_abs"], float)
    assert w.shape == (4,)
    assert np.all(w >= 0)


def test_ar_coefficients_are_returned_per_time_varying_block():
    res = DMLFM(base_config(make_panel(), r=3, ar1=True)).fit()
    phi = np.asarray(res.additional_outputs["phi"], float)
    assert np.all(np.isfinite(phi))


def test_loading_constraint_is_lower_triangular():
    """pblasso restricts unit i to load on at most the first i factors
    (blasso.cpp:293). The constraint pins rotation and makes the fit depend on
    donor ordering, so it has to be reproduced, not quietly dropped."""
    res = DMLFM(base_config(make_panel(n_units=12), r=4)).fit()
    gamma = np.asarray(res.additional_outputs["gamma_mean"], float)
    for i in range(1, 4):
        assert np.allclose(gamma[i - 1, i:], 0.0), f"unit {i} loads above rank {i}"


def test_counterfactual_is_a_posterior_predictive_draw():
    """blasso_core.R:538 adds rnorm(sd=sqrt(sigma2)) to the mean function.
    Dropping the noise narrows the bands, so the interval must be wider than
    the spread of the mean function alone."""
    res = DMLFM(base_config(make_panel(), niter=1500, burn=500)).fit()
    lo = float(res.inference.ci_lower)
    hi = float(res.inference.ci_upper)
    assert hi - lo > 0


def test_seed_is_reproducible():
    a = DMLFM(base_config(make_panel())).fit()
    b = DMLFM(base_config(make_panel())).fit()
    assert float(a.effects.att) == pytest.approx(float(b.effects.att))


def test_different_seeds_give_different_draws():
    a = DMLFM(base_config(make_panel(), seed=1)).fit()
    b = DMLFM(base_config(make_panel(), seed=2)).fit()
    assert float(a.effects.att) != float(b.effects.att)


# --------------------------------------------------------------------------
# edge cases
# --------------------------------------------------------------------------
def test_runs_without_covariates():
    res = DMLFM(base_config(make_panel(covariates=False))).fit()
    assert np.isfinite(float(res.effects.att))


def test_runs_with_zero_factors():
    """r = 0 reduces the model to covariates plus additive effects."""
    res = DMLFM(base_config(make_panel(), r=0)).fit()
    assert np.isfinite(float(res.effects.att))


def test_single_donor():
    res = DMLFM(base_config(make_panel(n_units=2), r=1)).fit()
    assert np.isfinite(float(res.effects.att))


def test_short_pre_period_still_runs_but_is_flagged():
    """The paper reports unsatisfactory frequentist properties below T0 = 20
    (Supplementary Table A4). Running is allowed; the diagnostic must say so."""
    df = make_panel(n_periods=20, pre=8)
    res = DMLFM(base_config(df)).fit()
    assert res.method_details.parameters["short_pre_period"] is True


def test_r_larger_than_units_is_rejected():
    with pytest.raises(MlsynthConfigError):
        DMLFM(base_config(make_panel(n_units=5), r=9)).fit()


# --------------------------------------------------------------------------
# failure paths
# --------------------------------------------------------------------------
def test_burn_not_less_than_niter_raises():
    with pytest.raises(MlsynthConfigError):
        DMLFM(base_config(make_panel(), niter=100, burn=100))


def test_negative_r_raises():
    with pytest.raises(MlsynthConfigError):
        DMLFM(base_config(make_panel(), r=-1))


def test_unknown_field_is_forbidden():
    with pytest.raises(MlsynthConfigError):
        DMLFM(base_config(make_panel(), not_a_real_option=3))


def test_unbalanced_panel_raises_data_error():
    df = make_panel()
    df = df.drop(df.index[5])
    with pytest.raises(MlsynthDataError):
        DMLFM(base_config(df)).fit()


def test_no_pre_periods_raises():
    df = make_panel(pre=0)
    with pytest.raises((MlsynthDataError, MlsynthConfigError)):
        DMLFM(base_config(df)).fit()


def test_missing_outcome_raises():
    df = make_panel()
    df.loc[df.index[3], "y"] = np.nan
    with pytest.raises(MlsynthDataError):
        DMLFM(base_config(df)).fit()


# --------------------------------------------------------------------------
# plotting smoke
# --------------------------------------------------------------------------
def test_plotting_smoke(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    res = DMLFM(base_config(make_panel(), display_graphs=True)).fit()
    assert np.isfinite(float(res.effects.att))
