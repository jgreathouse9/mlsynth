"""Tests for MVBBSC -- Martinez & Vives-i-Bastida (2024) Bayesian synthetic control.

Written test-first. The model is the outcome-only Bayesian SC of Martinez &
Vives-i-Bastida (arXiv:2206.01779): uniform-Dirichlet simplex weights, a
``HalfNormal`` idiosyncratic scale, a Gaussian likelihood on the pre-period,
and pre-period standardization of the treated and donor series. NUTS is run
tiny (few warmup/samples) so the suite stays fast; the durable Germany
cross-check vs the ``bsynth`` reference package lives in
``benchmarks/cases/mvbbsc_germany.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import MVBBSC
from mlsynth.config_models import EffectResult, MVBBSCConfig
from mlsynth.exceptions import MlsynthEstimationError


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def _hull_panel(n_units=8, T=24, T0=16, effect=-4.0, seed=0):
    """Panel where the treated unit is a convex combination of two donors.

    Because the treated series lives inside the donor convex hull, the
    simplex-constrained B-MV weights can track it well in the pre-period.
    Unit 0 is treated with a constant post-treatment effect.
    """
    rng = np.random.default_rng(seed)
    f1 = 100.0 + np.cumsum(rng.standard_normal(T))
    f2 = 120.0 + np.cumsum(rng.standard_normal(T))
    donors = []
    for i in range(n_units - 1):
        a = rng.uniform(0.2, 1.0)
        donors.append(a * f1 + (1 - a) * f2 + rng.standard_normal(T) * 0.4)
    # treated = 0.6 * donor0 + 0.4 * donor1 (inside the hull) + tiny noise
    treated = 0.6 * donors[0] + 0.4 * donors[1] + rng.standard_normal(T) * 0.3
    treated = treated.copy()
    treated[T0:] += effect

    rows = []
    for t in range(T):
        rows.append({"unit": "u00", "t": t, "y": float(treated[t]),
                     "treat": int(t >= T0)})
    for i, d in enumerate(donors):
        for t in range(T):
            rows.append({"unit": f"d{i:02d}", "t": t, "y": float(d[t]),
                         "treat": 0})
    return pd.DataFrame(rows)


_FAST = dict(n_warmup=80, n_samples=80, n_chains=1, seed=0, display_graphs=False)


def _cfg(df=None, **kw):
    if df is None:
        df = _hull_panel()
    base = dict(df=df, outcome="y", treat="treat", unitid="unit", time="t")
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_config_defaults():
    cfg = MVBBSCConfig(**_cfg())
    assert cfg.n_warmup >= 1 and cfg.n_samples >= 1 and cfg.n_chains >= 1
    assert 0.0 < cfg.ci_alpha < 1.0


@pytest.mark.parametrize("bad", [
    {"n_warmup": 0}, {"n_samples": 0}, {"n_chains": 0},
    {"ci_alpha": 0.0}, {"ci_alpha": 1.0},
])
def test_config_rejects_bad(bad):
    with pytest.raises(Exception):
        MVBBSCConfig(**_cfg(**bad))


def test_config_rejects_extra_field():
    with pytest.raises(Exception):
        MVBBSCConfig(**_cfg(not_a_field=1))


def test_config_rejects_too_few_draws():
    """A single posterior draw (n_samples * n_chains < 2) is rejected."""
    from mlsynth.exceptions import MlsynthConfigError
    with pytest.raises(MlsynthConfigError):
        MVBBSCConfig(**_cfg(n_samples=1, n_chains=1))


# --------------------------------------------------------------------------- #
# smoke + contract
# --------------------------------------------------------------------------- #
def test_smoke_returns_effect_result():
    pytest.importorskip("numpyro")
    res = MVBBSC(_cfg(**_FAST)).fit()
    assert isinstance(res, EffectResult)
    T = res.time_series.time_periods.shape[0]
    assert res.time_series.counterfactual_outcome.shape == (T,)
    assert res.time_series.estimated_gap.shape == (T,)
    assert np.isfinite(res.att)
    lo, hi = res.inference.ci_lower, res.inference.ci_upper
    assert lo is not None and hi is not None and lo <= hi


def test_weights_are_a_simplex():
    pytest.importorskip("numpyro")
    res = MVBBSC(_cfg(**_FAST)).fit()
    w = np.array(list(res.weights.donor_weights.values()))
    assert np.all(w >= -1e-8)                       # non-negative
    assert abs(w.sum() - 1.0) < 1e-6                # sums to one


def test_recovers_effect_sign_and_covers_truth():
    pytest.importorskip("numpyro")
    slow = {**_FAST, "n_warmup": 200, "n_samples": 300}
    res = MVBBSC(_cfg(df=_hull_panel(effect=-4.0), **slow)).fit()
    assert res.att < 0.0                            # correct sign
    null = MVBBSC(_cfg(df=_hull_panel(effect=0.0), **slow)).fit()
    assert abs(null.att) < abs(res.att)


def test_multichain_reports_rhat():
    """Two chains populate the split-R-hat convergence diagnostic."""
    pytest.importorskip("numpyro")
    import os
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
    res = MVBBSC(_cfg(**{**_FAST, "n_chains": 2})).fit()
    max_rhat = res.weights.summary_stats["max_rhat"]
    assert np.isfinite(max_rhat) and max_rhat > 0.0


def test_multi_cohort_raises():
    """A panel with a second treated cohort is rejected in setup."""
    from mlsynth.exceptions import MlsynthDataError
    df = _hull_panel()
    # inject a second treated cohort adopting at a different time
    df.loc[(df["unit"] == "d00") & (df["t"] >= 10), "treat"] = 1
    with pytest.raises(MlsynthDataError):
        MVBBSC(_cfg(df=df, **_FAST)).fit()


def test_pre_period_fit_is_reasonable():
    pytest.importorskip("numpyro")
    res = MVBBSC(_cfg(**_FAST)).fit()
    gap = res.time_series.estimated_gap
    T0 = int(np.where(res.time_series.time_periods
                      == res.time_series.intervention_time)[0][0])
    pre_rmse = float(np.sqrt(np.mean(gap[:T0] ** 2)))
    assert pre_rmse < 5.0                           # tracks the pre-period


def test_scale_invariance_of_weights():
    """B-MV standardizes internally, so rescaling every series leaves the
    posterior weights (approximately) unchanged and scales the ATT by the same
    factor.

    The invariance is exact in real arithmetic (the pre-period standardization
    removes the scale), but NUTS is a stochastic, finite-precision sampler: the
    float32 standardized inputs for ``y`` and ``1000*y`` are only bit-identical
    on some JAX/NumPyro builds. On others the short ``_FAST`` chains amplify a
    ~1e-6 float difference to ~1e-2 in the posterior-mean weights. Assert
    invariance to MCMC / finite-precision error, not to machine epsilon, so the
    check is robust across sampler builds rather than passing only where the
    inputs happen to round identically.
    """
    pytest.importorskip("numpyro")
    df = _hull_panel()
    res1 = MVBBSC(_cfg(df=df, **_FAST)).fit()
    df2 = df.copy()
    df2["y"] = df2["y"] * 1000.0
    res2 = MVBBSC(_cfg(df=df2, **_FAST)).fit()
    w1 = np.array(list(res1.weights.donor_weights.values()))
    w2 = np.array(list(res2.weights.donor_weights.values()))
    assert np.allclose(w1, w2, atol=0.05)                 # simplex weights, MCMC-close
    assert res2.att == pytest.approx(res1.att * 1000.0, rel=0.05)


# --------------------------------------------------------------------------- #
# plotting + failure paths
# --------------------------------------------------------------------------- #
def test_plotting_smoke(monkeypatch):
    pytest.importorskip("numpyro")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    res = MVBBSC(_cfg(**{**_FAST, "display_graphs": True})).fit()
    assert res is not None


def test_invalid_config_dict_raises():
    """A bad dict config is translated to MlsynthConfigError at construction."""
    from mlsynth.exceptions import MlsynthConfigError
    with pytest.raises(MlsynthConfigError):
        MVBBSC(_cfg(ci_alpha=2.0))


def test_too_few_pre_periods_raises():
    """A panel with a single pre-period fails in setup with MlsynthDataError."""
    from mlsynth.exceptions import MlsynthDataError
    # treatment starts at t=1, leaving only one pre-period.
    df = _hull_panel(T=6, T0=1)
    with pytest.raises(MlsynthDataError):
        MVBBSC(_cfg(df=df, **_FAST)).fit()


def test_missing_numpyro_raises(monkeypatch):
    """When the backend import fails, fit raises a translated error."""
    from mlsynth.utils.mvbbsc_helpers import model as mvbbsc_model

    def _boom():
        raise ImportError("no numpyro")
    monkeypatch.setattr(mvbbsc_model, "_import_backend", _boom)
    with pytest.raises(MlsynthEstimationError):
        MVBBSC(_cfg(**_FAST)).fit()
