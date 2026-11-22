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


# --------------------------------------------------------------------------- #
# the draw-count contract: n_samples is per chain
# --------------------------------------------------------------------------- #
# The config described n_samples as "per chain (total, across chains)", and the
# sampler asked NumPyro for ceil(n_samples / n_chains) per chain: 250 at the
# defaults, against the Stan reference's 1000 per chain (iter = warmup +
# n_samples). BFSC had the same defect (jgreathouse9/mlsynth#1109), where split
# R-hat on a quarter of the draws exceeded 1.1 on some CI runners. A fake backend
# records what the sampler is asked for, so these tests need no NumPyro.
from hypothesis import HealthCheck, given, settings, strategies as st  # noqa: E402


class _FakeBackend:
    """Stands in for (jax, jnp, numpyro, dist, MCMC, NUTS, diagnostics): records the
    arguments the sampler receives and returns constant draws of the shape asked for.
    Its split R-hat asserts at least 4 draws per chain, as NumPyro's does."""

    def __init__(self, T, draws_override=None):
        self.calls, self.T, self.draws_override = {}, T, draws_override
        outer = self

        class _Key:
            @staticmethod
            def PRNGKey(seed):
                outer.calls["seed"] = seed
                return seed

        class _Jax:
            random = _Key

        class _NUTS:
            def __init__(self, model, target_accept_prob, max_tree_depth):
                outer.calls.update(target_accept=target_accept_prob, max_tree_depth=max_tree_depth)

        class _MCMC:
            def __init__(self, kernel, num_warmup, num_samples, num_chains, progress_bar):
                outer.calls.update(num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)

            def run(self, key, extra_fields=()):
                pass

            def _shape(self):
                c, s = outer.calls["num_chains"], outer.calls["num_samples"]
                return c, (outer.draws_override if outer.draws_override is not None else s)

            def get_samples(self, group_by_chain=False):
                c, s = self._shape()
                if group_by_chain:
                    return {"cf": np.zeros((c, s, outer.T)), "sigman": np.ones((c, s))}
                return {"lengthscale_f": np.ones(c * s), "lengthscale_global": np.ones(c * s)}

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


def _run(monkeypatch, n_samples, n_chains, n_warmup=7, seed=3, target_accept=0.9, max_tree_depth=10,
         draws_override=None, T=12, D=5):
    from mlsynth.utils.mtgp_helpers import model as mtgp_model
    fake = _FakeBackend(T, draws_override)
    monkeypatch.setattr(mtgp_model, "_import_backend", fake)
    Y = np.random.default_rng(0).normal(size=(T, D))
    out = mtgp_model.run_mtgp(Y, np.ones((T, D)), 8, 0, n_factors=2, n_warmup=n_warmup, n_samples=n_samples,
                              n_chains=n_chains, target_accept=target_accept, max_tree_depth=max_tree_depth,
                              seed=seed)
    return fake.calls, out


def test_the_reproduction_four_chains_of_1000_draws_each(monkeypatch):
    """The defaults and mtgp_california's settings (1000 warm-up, 1000 samples, 4 chains): the
    sampler is asked for 1000 draws per chain, as the Stan reference draws, and returns 4,000
    counterfactual draws, not 250 per chain."""
    calls, out = _run(monkeypatch, n_samples=1000, n_chains=4, n_warmup=1000)
    assert calls["num_samples"] == 1000 and calls["num_chains"] == 4
    assert out["counterfactual"].shape == (4000, 12)


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
       seed=st.integers(0, 2**31 - 1), target_accept=st.floats(0.05, 0.99), max_tree_depth=st.integers(1, 15))
def test_every_setting_reaches_the_sampler(monkeypatch, n_samples, n_chains, n_warmup, seed, target_accept,
                                           max_tree_depth):
    """Property, over every sampler argument: NumPyro receives n_samples draws per chain,
    n_chains chains, n_warmup warm-up steps, the seed, the target acceptance and the maximum
    tree depth as given; the result holds n_samples * n_chains draws; R-hat is reported
    exactly when it is defined (two or more chains of four or more draws)."""
    calls, out = _run(monkeypatch, n_samples=n_samples, n_chains=n_chains, n_warmup=n_warmup, seed=seed,
                      target_accept=target_accept, max_tree_depth=max_tree_depth)
    assert calls == dict(seed=seed, target_accept=target_accept, max_tree_depth=max_tree_depth,
                         num_warmup=n_warmup, num_samples=n_samples, num_chains=n_chains)
    assert out["counterfactual"].shape[0] == n_samples * n_chains == out["sigma"].shape[0]
    assert np.isnan(out["max_rhat"]) == (n_chains < 2 or n_samples < 4)


def test_a_short_draw_from_the_sampler_is_reported(monkeypatch):
    """Failure: the result checks its own draw count. A backend that returns fewer draws
    per chain than asked raises a translated MlsynthEstimationError."""
    with pytest.raises(MlsynthEstimationError, match="draws"):
        _run(monkeypatch, n_samples=20, n_chains=2, draws_override=19)
