"""Tests for MTGP -- Multitask Gaussian Process synthetic control (Ben-Michael
et al. 2023).

Fast NUTS (tiny warmup/samples, few factors, small panel) throughout so the
suite stays quick; the durable California cross-check vs the author's Stan lives
in ``benchmarks/cases/mtgp_california.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import MTGP
from mlsynth.config_models import MTGPConfig
from mlsynth.config_models import EffectResult
from mlsynth.exceptions import MlsynthEstimationError


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def _gp_panel(n_units=8, T=16, T0=11, effect=-4.0, seed=0, with_pop=False):
    """Smooth-common-factor panel; unit 0 treated with a constant post effect."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.standard_normal(T))                 # a smooth common factor
    rows = []
    for i in range(n_units):
        load = 1.0 + rng.standard_normal() * 0.5
        y = 20.0 + load * f + rng.standard_normal(T) * 0.3
        if i == 0:
            y[T0:] += effect
        for t in range(T):
            row = {"unit": f"u{i:02d}", "t": t, "y": float(y[t]),
                   "treat": int(i == 0 and t >= T0)}
            if with_pop:
                row["pop"] = float(1e6 * (1 + i))         # positive, varies by unit
            rows.append(row)
    return pd.DataFrame(rows)


_FAST = dict(n_factors=2, n_warmup=120, n_samples=120, n_chains=1, seed=0,
             display_graphs=False)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_config_defaults():
    cfg = MTGPConfig(df=_gp_panel(), outcome="y", treat="treat",
                     unitid="unit", time="t")
    assert cfg.n_factors >= 1 and 0.0 < cfg.ci_alpha < 1.0
    assert cfg.n_warmup >= 1 and cfg.n_samples >= 1 and cfg.n_chains >= 1
    assert cfg.max_tree_depth >= 1 and cfg.population is None


@pytest.mark.parametrize("bad", [
    {"n_factors": 0}, {"ci_alpha": 0.0}, {"ci_alpha": 1.0}, {"n_warmup": 0},
    {"n_samples": 0}, {"n_chains": 0}, {"max_tree_depth": 0}, {"target_accept": 1.0},
])
def test_config_rejects_bad(bad):
    with pytest.raises(Exception):
        MTGPConfig(df=_gp_panel(), outcome="y", treat="treat",
                   unitid="unit", time="t", **bad)


# --------------------------------------------------------------------------- #
# smoke + contract
# --------------------------------------------------------------------------- #
def test_smoke_returns_effect_result():
    pytest.importorskip("numpyro")
    res = MTGP({"df": _gp_panel(), "outcome": "y", "treat": "treat",
                "unitid": "unit", "time": "t", **_FAST}).fit()
    assert isinstance(res, EffectResult)
    T = res.time_series.time_periods.shape[0]
    assert res.time_series.counterfactual_outcome.shape == (T,)
    assert res.time_series.estimated_gap.shape == (T,)
    assert np.isfinite(res.att)
    lo, hi = res.inference.ci_lower, res.inference.ci_upper
    assert lo is not None and hi is not None and lo <= hi


def test_recovers_effect_sign_and_covers_truth():
    pytest.importorskip("numpyro")
    res = MTGP({"df": _gp_panel(effect=-4.0), "outcome": "y", "treat": "treat",
                "unitid": "unit", "time": "t",
                **{**_FAST, "n_warmup": 250, "n_samples": 250}}).fit()
    assert res.att < 0.0                                   # correct sign
    null = MTGP({"df": _gp_panel(effect=0.0), "outcome": "y", "treat": "treat",
                 "unitid": "unit", "time": "t",
                 **{**_FAST, "n_warmup": 250, "n_samples": 250}}).fit()
    assert abs(null.att) < abs(res.att)


def test_pre_period_fit_is_reasonable():
    pytest.importorskip("numpyro")
    res = MTGP({"df": _gp_panel(), "outcome": "y", "treat": "treat",
                "unitid": "unit", "time": "t", **_FAST}).fit()
    assert res.fit_diagnostics.rmse_pre < 3.0             # tracks pre-period


def test_credible_band_widens_post_treatment():
    """The signature MTGP feature: the counterfactual band grows post-treatment."""
    pytest.importorskip("numpyro")
    res = MTGP({"df": _gp_panel(), "outcome": "y", "treat": "treat",
                "unitid": "unit", "time": "t", **_FAST}).fit()
    inf = res.inference_detail
    width = np.asarray(inf.counterfactual_upper) - np.asarray(inf.counterfactual_lower)
    T0 = res.inputs.T0
    assert width[T0:].mean() > width[:T0].mean()


def test_heteroskedastic_population_branch():
    pytest.importorskip("numpyro")
    res = MTGP({"df": _gp_panel(with_pop=True), "outcome": "y", "treat": "treat",
                "unitid": "unit", "time": "t", "population": "pop", **_FAST}).fit()
    assert np.isfinite(res.att)
    assert res.inputs.inv_pop.shape == (res.inputs.T, res.inputs.D)


# --------------------------------------------------------------------------- #
# plotting + failure paths
# --------------------------------------------------------------------------- #
def test_plotting_smoke(monkeypatch):
    pytest.importorskip("numpyro")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    res = MTGP({"df": _gp_panel(), "outcome": "y", "treat": "treat", "unitid": "unit",
                "time": "t", **{**_FAST, "display_graphs": True}}).fit()
    assert res is not None


def test_missing_numpyro_raises(monkeypatch):
    """When the backend import fails, fit raises a translated MlsynthEstimationError."""
    from mlsynth.utils.mtgp_helpers import model as mtgp_model

    def _boom():
        raise ImportError("no numpyro")
    monkeypatch.setattr(mtgp_model, "_import_backend", _boom)
    with pytest.raises(MlsynthEstimationError):
        MTGP({"df": _gp_panel(), "outcome": "y", "treat": "treat",
              "unitid": "unit", "time": "t", **_FAST}).fit()
