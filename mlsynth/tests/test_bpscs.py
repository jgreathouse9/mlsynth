"""Tests for BPSCS -- Penalized Synthetic Control under Spillovers with
utility-based shrinkage priors (Fernandez-Morales, Oganisian & Lee 2026).

Fast NUTS (tiny warmup/samples, one chain, small panel) throughout so the suite
stays quick; the durable Philadelphia cross-check vs the authors' GPL-3 Stan
lives in ``benchmarks/cases/bpscs_beverage.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import BPSCS
from mlsynth.config_models import BPSCSConfig
from mlsynth.config_models import EffectResult
from mlsynth.exceptions import MlsynthEstimationError


# --------------------------------------------------------------------------- #
# panel: units on a 2-D map, a smooth common factor, unit 0 treated
# --------------------------------------------------------------------------- #
def _spatial_panel(n_units=10, T=16, T0=11, effect=-3.0, seed=0, spillover=0.0):
    """Units placed in [0,1]^2 (treated at the origin); covariates track
    location; a smooth common factor drives outcomes. Optional post-period
    spillover contaminates donors near the treated unit."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1, size=(n_units, 2))
    coords[0] = [0.0, 0.0]                                  # treated at origin
    f = np.cumsum(rng.standard_normal(T))                    # smooth common factor
    rows = []
    dist0 = np.linalg.norm(coords - coords[0], axis=1)
    for i in range(n_units):
        load = 1.0 + rng.standard_normal() * 0.4
        cov = coords[i] + rng.standard_normal(2) * 0.05      # covariates track location
        y = 20.0 + load * f + rng.standard_normal(T) * 0.3
        if i == 0:
            y[T0:] += effect
        elif spillover > 0 and dist0[i] < 0.35:              # nearby donors contaminated
            y[T0:] += spillover
        for t in range(T):
            rows.append({"unit": f"u{i:02d}", "t": t, "y": float(y[t]),
                         "treat": int(i == 0 and t >= T0),
                         "cov1": float(cov[0]), "cov2": float(cov[1]),
                         "lat": float(coords[i, 0]), "lon": float(coords[i, 1])})
    return pd.DataFrame(rows)


_COLS = dict(outcome="y", treat="treat", unitid="unit", time="t",
             covariates=["cov1", "cov2"], coords=["lat", "lon"])
_FAST = dict(n_warmup=150, n_samples=150, n_chains=1, target_accept=0.9,
             max_tree_depth=10, seed=0, display_graphs=False)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_config_defaults():
    cfg = BPSCSConfig(df=_spatial_panel(), **_COLS)
    assert cfg.prior == "dhs"
    assert 0.0 <= cfg.kappa_d <= 1.0 and 0.0 < cfg.inclusion_quantile < 1.0
    assert cfg.covariates == ["cov1", "cov2"] and cfg.coords == ["lat", "lon"]


@pytest.mark.parametrize("bad", [
    {"prior": "nope"}, {"kappa_d": -0.1}, {"kappa_d": 1.1},
    {"inclusion_quantile": 0.0}, {"inclusion_quantile": 1.0},
    {"covariates": []}, {"coords": []},
    {"n_samples": 0}, {"n_chains": 0}, {"target_accept": 1.0},
    {"n_samples": 1, "n_chains": 1},               # < 2 total posterior draws
])
def test_config_rejects_bad(bad):
    kw = {**_COLS, **bad}
    with pytest.raises(Exception):
        BPSCSConfig(df=_spatial_panel(), **kw)


def test_dict_config_error_is_translated():
    """A bad dict config raises a translated MlsynthConfigError, not ValidationError."""
    from mlsynth.exceptions import MlsynthConfigError
    with pytest.raises(MlsynthConfigError):
        BPSCS({"df": _spatial_panel(), **_COLS, "kappa_d": 5.0})


# --------------------------------------------------------------------------- #
# smoke + contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prior", ["dhs", "ds2"])
def test_smoke_returns_effect_result(prior):
    pytest.importorskip("numpyro")
    res = BPSCS({"df": _spatial_panel(), **_COLS, "prior": prior, **_FAST}).fit()
    assert isinstance(res, EffectResult)
    T = res.time_series.time_periods.shape[0]
    assert res.time_series.counterfactual_outcome.shape == (T,)
    assert res.time_series.estimated_gap.shape == (T,)
    assert np.isfinite(res.att)
    lo, hi = res.inference.ci_lower, res.inference.ci_upper
    assert lo is not None and hi is not None and lo <= hi


def test_recovers_effect_sign():
    pytest.importorskip("numpyro")
    res = BPSCS({"df": _spatial_panel(effect=-3.0), **_COLS,
                **{**_FAST, "n_warmup": 300, "n_samples": 300}}).fit()
    assert res.att < 0.0
    null = BPSCS({"df": _spatial_panel(effect=0.0), **_COLS,
                 **{**_FAST, "n_warmup": 300, "n_samples": 300}}).fit()
    assert abs(null.att) < abs(res.att)


def test_pre_period_fit_is_reasonable():
    pytest.importorskip("numpyro")
    res = BPSCS({"df": _spatial_panel(), **_COLS, **_FAST}).fit()
    assert res.fit_diagnostics.rmse_pre < 3.0


def test_donor_coefficients_exposed():
    """Beta posterior medians are surfaced as (signed) donor weights."""
    pytest.importorskip("numpyro")
    res = BPSCS({"df": _spatial_panel(), **_COLS, **_FAST}).fit()
    w = res.weights.donor_weights
    assert isinstance(w, dict) and len(w) == 9          # 10 units -> 9 donors
    assert all(np.isfinite(v) for v in w.values())


def test_ds2_reports_inclusion_and_rho():
    pytest.importorskip("numpyro")
    res = BPSCS({"df": _spatial_panel(), **_COLS, "prior": "ds2", **_FAST}).fit()
    ss = res.weights.summary_stats
    assert ss["prior"] == "ds2"
    assert "inclusion_radius" in ss and np.isfinite(ss["inclusion_radius"])


def test_kappa_changes_the_fit():
    """kappa_d=0 (spatial) and kappa_d=1 (covariate) weight donors differently."""
    pytest.importorskip("numpyro")
    common = {**_FAST, "n_warmup": 250, "n_samples": 250}
    r0 = BPSCS({"df": _spatial_panel(seed=1), **_COLS, "kappa_d": 0.0, **common}).fit()
    r1 = BPSCS({"df": _spatial_panel(seed=1), **_COLS, "kappa_d": 1.0, **common}).fit()
    w0 = np.array([r0.weights.donor_weights[k] for k in sorted(r0.weights.donor_weights)])
    w1 = np.array([r1.weights.donor_weights[k] for k in sorted(r1.weights.donor_weights)])
    assert not np.allclose(w0, w1, atol=1e-3)


# --------------------------------------------------------------------------- #
# plotting + failure paths
# --------------------------------------------------------------------------- #
def test_plotting_smoke(monkeypatch):
    pytest.importorskip("numpyro")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    res = BPSCS({"df": _spatial_panel(), **_COLS,
                **{**_FAST, "display_graphs": True}}).fit()
    assert res is not None


def test_missing_covariate_column_raises():
    with pytest.raises(Exception):
        BPSCS({"df": _spatial_panel(), **{**_COLS, "covariates": ["nope"]},
              **_FAST}).fit()


def test_nonfinite_covariate_raises():
    from mlsynth.exceptions import MlsynthDataError
    df = _spatial_panel()
    df.loc[df["unit"] == "u03", "cov1"] = np.nan
    with pytest.raises(MlsynthDataError):
        BPSCS({"df": df, **_COLS, **_FAST}).fit()


def test_constant_preperiod_series_raises():
    """A donor that is constant over the pre-period cannot be standardized."""
    from mlsynth.exceptions import MlsynthDataError
    df = _spatial_panel(T0=11)
    df.loc[(df["unit"] == "u04") & (df["t"] < 11), "y"] = 5.0   # flat pre-period
    with pytest.raises(MlsynthDataError):
        BPSCS({"df": df, **_COLS, **_FAST}).fit()


def test_missing_numpyro_raises(monkeypatch):
    from mlsynth.utils.bpscs_helpers import model as bpscs_model

    def _boom():
        raise ImportError("no numpyro")
    monkeypatch.setattr(bpscs_model, "_import_backend", _boom)
    with pytest.raises(MlsynthEstimationError):
        BPSCS({"df": _spatial_panel(), **_COLS, **_FAST}).fit()
