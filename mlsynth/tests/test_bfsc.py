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


# --------------------------------------------------------------------------- #
# the draw-count contract: n_samples is per chain
# --------------------------------------------------------------------------- #
# The config documents n_samples as "retained NUTS samples per chain", and the
# replication page compares 4 chains x 500 draws with the author's Stan. The
# sampler asked NumPyro for ceil(n_samples / n_chains) draws per chain, a
# quarter of that at the defaults: 125 per chain. Split R-hat on 125
# autocorrelated draws exceeds 1.1 from noise alone, which is how bfsc_germany
# came to fail its R-hat band on some CI runners and pass on others. A fake
# backend records what the sampler is asked for, so these tests need no NumPyro.
from hypothesis import HealthCheck, given, settings, strategies as st  # noqa: E402


class _FakeBackend:
    """Stands in for (jax, jnp, numpyro, dist, MCMC, NUTS, diagnostics): records the
    arguments the sampler receives and returns constant draws of the shape asked for.
    Its split R-hat asserts at least 4 draws per chain, as NumPyro's does."""

    def __init__(self, draws_override=None):
        self.calls = {}
        self.draws_override = draws_override
        outer = self

        class _Key:
            @staticmethod
            def PRNGKey(seed):
                outer.calls["seed"] = seed
                return seed

        class _Jax:
            random = _Key

        class _NUTS:
            def __init__(self, model, target_accept_prob):
                outer.calls["target_accept"] = target_accept_prob

        class _MCMC:
            def __init__(self, kernel, num_warmup, num_samples, num_chains, progress_bar):
                outer.calls.update(num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)

            def run(self, key, extra_fields=()):
                pass

            def get_samples(self, group_by_chain=True):
                c, s = outer.calls["num_chains"], outer.calls["num_samples"]
                if outer.draws_override is not None:
                    s = outer.draws_override
                return {"cf": np.zeros((c, s, outer.T)), "sigma": np.ones((c, s))}

            def get_extra_fields(self):
                return {"accept_prob": np.full(3, 0.9), "diverging": np.zeros(3, bool)}

        class _Diagnostics:
            @staticmethod
            def split_gelman_rubin(x):
                x = np.asarray(x)
                assert x.ndim >= 2 and x.shape[1] >= 4
                return np.ones(x.shape[2:])

        self.modules = (_Jax, np, None, None, _MCMC, _NUTS, _Diagnostics)

    def __call__(self):
        return self.modules


def _run(monkeypatch, n_samples, n_chains, n_warmup=7, seed=3, target_accept=0.9, draws_override=None, T=12):
    from mlsynth.utils.bfsc_helpers import model as bfsc_model
    fake = _FakeBackend(draws_override)
    fake.T = T
    monkeypatch.setattr(bfsc_model, "_import_backend", fake)
    Y = np.random.default_rng(0).normal(size=(5, T))
    out = bfsc_model.run_bfsc(Y, 8, n_factors=2, n_warmup=n_warmup, n_samples=n_samples, n_chains=n_chains,
                              target_accept=target_accept, seed=seed)
    return fake.calls, out


def test_the_reproduction_four_chains_of_500_draws_each(monkeypatch):
    """bfsc_germany's settings (500 warm-up, 500 samples, 4 chains): the sampler is asked
    for 500 draws per chain and returns 2,000 counterfactual draws, not 125 per chain."""
    calls, out = _run(monkeypatch, n_samples=500, n_chains=4, n_warmup=500)
    assert calls["num_samples"] == 500 and calls["num_chains"] == 4
    assert out["counterfactual"].shape == (2000, 12)


@pytest.mark.parametrize("n_samples, n_chains", [(1, 2), (2, 2), (3, 4), (1, 7)])
def test_too_few_draws_per_chain_for_r_hat_report_nan(monkeypatch, n_samples, n_chains):
    """Edge: split R-hat needs at least 4 draws per chain. With fewer the draws are still
    returned (n_samples per chain) and max_rhat is NaN, not a crash."""
    calls, out = _run(monkeypatch, n_samples=n_samples, n_chains=n_chains)
    assert calls["num_samples"] == n_samples
    assert out["counterfactual"].shape[0] == n_samples * n_chains and np.isnan(out["max_rhat"])


def test_one_chain_has_no_r_hat(monkeypatch):
    calls, out = _run(monkeypatch, n_samples=10, n_chains=1)
    assert out["counterfactual"].shape[0] == 10 and np.isnan(out["max_rhat"])


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(n_samples=st.integers(1, 60), n_chains=st.integers(1, 6), n_warmup=st.integers(1, 50),
       seed=st.integers(0, 2**31 - 1), target_accept=st.floats(0.05, 0.99))
def test_every_setting_reaches_the_sampler(monkeypatch, n_samples, n_chains, n_warmup, seed, target_accept):
    """Property, over every sampler argument: NumPyro receives n_samples draws per chain,
    n_chains chains, n_warmup warm-up steps, the seed and the target acceptance as given;
    the result holds n_samples * n_chains draws; R-hat is reported exactly when it is
    defined (two or more chains of four or more draws)."""
    calls, out = _run(monkeypatch, n_samples=n_samples, n_chains=n_chains, n_warmup=n_warmup, seed=seed,
                      target_accept=target_accept)
    assert calls == dict(seed=seed, target_accept=target_accept, num_warmup=n_warmup,
                         num_samples=n_samples, num_chains=n_chains)
    assert out["counterfactual"].shape[0] == n_samples * n_chains == out["sigma"].shape[0]
    assert np.isnan(out["max_rhat"]) == (n_chains < 2 or n_samples < 4)


def test_a_short_draw_from_the_sampler_is_reported(monkeypatch):
    """Failure: the result checks its own draw count. A backend that returns fewer draws
    per chain than asked raises a translated MlsynthEstimationError."""
    with pytest.raises(MlsynthEstimationError, match="draws"):
        _run(monkeypatch, n_samples=20, n_chains=2, draws_override=19)
