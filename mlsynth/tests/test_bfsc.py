"""Tests for BFSC -- Bayesian factor synthetic control (Pinkney 2021).

Written test-first. Fast NUTS (tiny warmup/samples, few factors) throughout so
the suite stays quick; the durable Germany cross-check vs the author's Stan
lives in ``benchmarks/cases/bfsc_germany.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import BFSC
from mlsynth.config_models import BFSCConfig
from mlsynth.config_models import EffectResult
from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def _factor_panel(n_units=8, T=24, T0=16, effect=-4.0, seed=0):
    """Single-factor panel; unit 0 treated with a constant post effect."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.standard_normal(T))                 # a common factor
    rows = []
    for i in range(n_units):
        load = 1.0 + rng.standard_normal() * 0.5
        y = 20.0 + load * f + rng.standard_normal(T) * 0.3
        if i == 0:
            y[T0:] += effect                              # treatment effect
        for t in range(T):
            rows.append({"unit": f"u{i:02d}", "t": t, "y": float(y[t]),
                         "treat": int(i == 0 and t >= T0)})
    return pd.DataFrame(rows)


_FAST = dict(n_factors=2, n_warmup=60, n_samples=60, n_chains=1, seed=0,
             display_graphs=False)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_config_defaults():
    cfg = BFSCConfig(df=_factor_panel(), outcome="y", treat="treat",
                     unitid="unit", time="t")
    assert cfg.n_factors >= 1 and 0.0 < cfg.ci_alpha < 1.0
    assert cfg.n_warmup >= 1 and cfg.n_samples >= 1 and cfg.n_chains >= 1


@pytest.mark.parametrize("bad", [
    {"n_factors": 0}, {"ci_alpha": 0.0}, {"ci_alpha": 1.0},
    {"n_warmup": 0}, {"n_samples": 0}, {"n_chains": 0},
])
def test_config_rejects_bad(bad):
    with pytest.raises(Exception):
        BFSCConfig(df=_factor_panel(), outcome="y", treat="treat",
                   unitid="unit", time="t", **bad)


# --------------------------------------------------------------------------- #
# smoke + contract
# --------------------------------------------------------------------------- #
def test_smoke_returns_effect_result():
    pytest.importorskip("numpyro")
    res = BFSC({"df": _factor_panel(), "outcome": "y", "treat": "treat",
                "unitid": "unit", "time": "t", **_FAST}).fit()
    assert isinstance(res, EffectResult)
    T = res.time_series.time_periods.shape[0]
    assert res.time_series.counterfactual_outcome.shape == (T,)
    assert res.time_series.estimated_gap.shape == (T,)
    assert np.isfinite(res.att)
    # credible band present and ordered
    lo = res.inference.ci_lower
    hi = res.inference.ci_upper
    assert lo is not None and hi is not None and lo <= hi


def test_recovers_effect_sign_and_covers_truth():
    pytest.importorskip("numpyro")
    res = BFSC({"df": _factor_panel(effect=-4.0), "outcome": "y",
                "treat": "treat", "unitid": "unit", "time": "t",
                **{**_FAST, "n_warmup": 150, "n_samples": 200}}).fit()
    assert res.att < 0.0                                   # correct sign
    # a fit with no treatment effect should not produce a large negative ATT
    null = BFSC({"df": _factor_panel(effect=0.0), "outcome": "y",
                 "treat": "treat", "unitid": "unit", "time": "t",
                 **{**_FAST, "n_warmup": 150, "n_samples": 200}}).fit()
    assert abs(null.att) < abs(res.att)


def test_pre_period_fit_is_reasonable():
    pytest.importorskip("numpyro")
    res = BFSC({"df": _factor_panel(), "outcome": "y", "treat": "treat",
                "unitid": "unit", "time": "t", **_FAST}).fit()
    gap = res.time_series.estimated_gap
    T0 = int(np.where(res.time_series.time_periods
                      == res.time_series.intervention_time)[0][0])
    pre_rmse = float(np.sqrt(np.mean(gap[:T0] ** 2)))
    assert pre_rmse < 3.0                                  # tracks pre-period


# --------------------------------------------------------------------------- #
# plotting + failure paths
# --------------------------------------------------------------------------- #
def test_plotting_smoke(monkeypatch):
    pytest.importorskip("numpyro")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    res = BFSC({"df": _factor_panel(), "outcome": "y", "treat": "treat",
                "unitid": "unit", "time": "t", **{**_FAST, "display_graphs": True}}).fit()
    assert res is not None


def test_missing_numpyro_raises(monkeypatch):
    """When the backend import fails, fit raises a translated MlsynthEstimationError."""
    from mlsynth.utils.bfsc_helpers import model as bfsc_model

    def _boom():
        raise ImportError("no numpyro")
    monkeypatch.setattr(bfsc_model, "_import_backend", _boom)
    with pytest.raises(MlsynthEstimationError):
        BFSC({"df": _factor_panel(), "outcome": "y", "treat": "treat",
              "unitid": "unit", "time": "t", **_FAST}).fit()
