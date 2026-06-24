"""Tests for the CMBSTS estimator (Causal Multivariate Bayesian Structural
Time Series; Menchetti and Bojinov 2022).

Layered per ``agents/agents_tests.md``: numerical engine (Layer 1), data
ingestion (Layer 2), estimator orchestration (Layer 3), and public API /
result-contract conformance (Layer 4). Synthetic panels are tiny, seeded, and
interpretable; assertions check invariants and translated exceptions, not
brittle floats.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from mlsynth import CMBSTS  # noqa: E402
from mlsynth.config_models import (  # noqa: E402
    BaseEstimatorResults,
    EffectResult,
    MlsynthResult,
)
from mlsynth.exceptions import (  # noqa: E402
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.cmbsts_helpers import engine as E  # noqa: E402
from mlsynth.utils.cmbsts_helpers.plotter import plot_cmbsts  # noqa: E402
from mlsynth.utils.cmbsts_helpers.setup import prepare_cmbsts_inputs  # noqa: E402
from mlsynth.utils.cmbsts_helpers.structures import CMBSTSResults  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures: a tiny, deterministic multivariate panel with a known effect.
# --------------------------------------------------------------------------- #
def make_panel(T=80, T0=60, effect_store=8.0, effect_comp=3.0, seed=0,
               controls=("wineA", "wineB", "wineC"), drift=0.0, noise=1.0):
    """Long-format panel: store (treated) + comp (group) + control wines.

    Outcomes share a common stationary (or lightly drifting) level so the
    controls track the treated series; a step effect turns on at ``T0``.
    """
    rng = np.random.default_rng(seed)
    base = 20.0 + drift * np.arange(T) + np.cumsum(rng.normal(0, 0.2, T))
    offsets = {"store": 0.0, "comp": -3.0}
    for i, c in enumerate(controls):
        offsets[c] = 2.0 * (i + 1) - 3.0
    rows = []
    for u in ["store", "comp", *controls]:
        series = base + offsets[u] + rng.normal(0, noise, T)
        for t in range(T):
            e = (effect_store if (u == "store" and t >= T0)
                 else effect_comp if (u == "comp" and t >= T0) else 0.0)
            rows.append({
                "item": u, "week": t, "sales": series[t] + e,
                "treated": 1 if (u == "store" and t >= T0) else 0,
                "sat": 1.0 if t % 7 == 5 else 0.0,
                "drop": 1.0 if t in (T - 1, T - 2) else 0.0,
            })
    return pd.DataFrame(rows)


def base_cfg(df, **over):
    cfg = {
        "df": df, "outcome": "sales", "unitid": "item", "time": "week",
        "treat": "treated", "group_units": ["comp"],
        "control_units": ["wineA", "wineB", "wineC"],
        "components": ["trend"], "niter": 500, "burn": 100, "seed": 0,
        "display_graphs": False,
    }
    cfg.update(over)
    return cfg


# --------------------------------------------------------------------------- #
# Layer 1 -- numerical engine
# --------------------------------------------------------------------------- #
class TestEngine:
    @pytest.mark.parametrize("comps,sp,cp,expected", [
        (["trend"], None, None, 2),                 # d level states (d=2)
        (["trend", "slope"], None, None, 4),         # level + slope
        (["trend", "seasonal"], 7, None, 2 + 6 * 2),  # + (S-1)*d seasonal
        (["trend", "cycle"], None, 75, 2 + 2 * 2),    # + 2d cycle
        (["seasonal"], 4, None, 3 * 2),               # seasonal only
    ])
    def test_build_ssm_dimensions(self, comps, sp, cp, expected):
        T, Z, dist, M = E.build_ssm(2, comps, seas_period=sp, cycle_period=cp)
        assert M == expected
        assert T.shape == (M, M)
        assert Z.shape == (2, M)
        # Z loads exactly the observed states with unit coefficients.
        assert np.all(np.isin(Z, [0.0, 1.0]))
        for sl, key in dist:
            assert key in ("level", "slope", "seasonal", "cycle1", "cycle2")

    def test_build_Q_places_blocks(self):
        T, Z, dist, M = E.build_ssm(2, ["trend", "seasonal"], seas_period=7)
        sig = {k: np.array([[1.0, 0.3], [0.3, 2.0]]) for _, k in dist}
        Q = E.build_Q(M, dist, sig)
        assert Q.shape == (M, M)
        assert np.allclose(Q[:2, :2], sig["level"])      # level block
        # seasonal lags >1 carry no disturbance
        assert np.allclose(Q[4:, 4:], 0.0)

    def test_ffbs_finite_shape(self):
        rng = np.random.default_rng(0)
        y = np.cumsum(rng.normal(0, 1, (40, 2)), axis=0)
        T, Z, dist, M = E.build_ssm(2, ["trend"])
        Q = E.build_Q(M, dist, {"level": np.eye(2)})
        alpha = E.ffbs(y, T, Z, Q, np.eye(2), rng)
        assert alpha.shape == (40, M)
        assert np.all(np.isfinite(alpha))

    def test_mvn_handles_non_psd(self):
        rng = np.random.default_rng(0)
        cov = np.array([[1.0, 1.0], [1.0, 1.0 - 1e-9]])  # numerically non-PSD
        draws = np.array([E._mvn(np.zeros(2), cov, rng) for _ in range(50)])
        assert np.all(np.isfinite(draws))

    def test_lpy_X_finite(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=(30, 2))
        X = rng.normal(size=(30, 3))
        val = E._lpy_X(y, X, E._inv(X.T @ X), nu0=4, s0=np.eye(2))
        assert np.isfinite(val)
        # empty regressor set is also finite
        assert np.isfinite(E._lpy_X(y, X[:, :0], np.zeros((0, 0)), 4, np.eye(2)))

    def test_run_gibbs_outputs(self):
        rng = np.random.default_rng(0)
        y = np.cumsum(rng.normal(0, 1, (50, 2)), axis=0) + 10
        T, Z, dist, M = E.build_ssm(2, ["trend"])
        fit = E.run_gibbs(y, T, Z, dist, M, np.eye(2), np.eye(2), 4, 200, 50, rng)
        assert fit["n_kept"] == 150
        assert len(fit["draws"]) == 150
        assert fit["prefit_mean"].shape == (50, 2)
        assert "inclusion" not in fit                    # no regressors

    def test_run_gibbs_with_regressors_has_inclusion(self):
        rng = np.random.default_rng(0)
        y = np.cumsum(rng.normal(0, 1, (50, 2)), axis=0) + 10
        X = rng.normal(size=(50, 3))
        T, Z, dist, M = E.build_ssm(2, ["trend"])
        fit = E.run_gibbs(y, T, Z, dist, M, np.eye(2), np.eye(2), 4, 200, 50, rng, X_pre=X)
        assert fit["inclusion"].shape == (3,)
        assert np.all((fit["inclusion"] >= 0) & (fit["inclusion"] <= 1))

    def test_causal_effect_recovers_step(self):
        # Clean stationary DGP: flat level + a +8 post step -> att ~ 8.
        rng = np.random.default_rng(1)
        T0, h, d = 60, 25, 2
        pre = 20 + rng.normal(0, 0.5, (T0, d))
        post = 20 + rng.normal(0, 0.5, (h, d)) + np.array([8.0, 3.0])
        Tm, Z, dist, M = E.build_ssm(d, ["trend"])
        fit = E.run_gibbs(pre, Tm, Z, dist, M, 0.01 * np.eye(d), 0.5 * np.eye(d),
                          4, 800, 200, rng)
        eff = E.causal_effect(post, fit, Tm, Z, rng)
        assert eff["att_mean"].shape == (d,)
        assert eff["effect_path"].shape == (h, d)
        assert eff["att_samples"].shape == (600, d)
        assert eff["att_mean"][0] == pytest.approx(8.0, abs=2.0)
        assert eff["att_mean"][1] == pytest.approx(3.0, abs=2.0)

    def test_horizon_restricts_summary(self):
        rng = np.random.default_rng(2)
        T0, h, d = 50, 30, 2
        pre = 20 + rng.normal(0, 0.5, (T0, d))
        post = 20 + rng.normal(0, 0.5, (h, d)) + np.array([8.0, 3.0])
        Tm, Z, dist, M = E.build_ssm(d, ["trend"])
        fit = E.run_gibbs(pre, Tm, Z, dist, M, 0.01 * np.eye(d), 0.5 * np.eye(d),
                          4, 400, 100, rng)
        full = E.causal_effect(post, fit, Tm, Z, np.random.default_rng(5))
        short = E.causal_effect(post, fit, Tm, Z, np.random.default_rng(5), horizon=5)
        # cumulative over 5 periods is far smaller than over all 30
        assert abs(short["cum_mean"][0]) < abs(full["cum_mean"][0])
        # the average effect still recovers the step
        assert short["att_mean"][0] == pytest.approx(8.0, abs=2.5)


# --------------------------------------------------------------------------- #
# Layer 2 -- data ingestion
# --------------------------------------------------------------------------- #
class TestSetup:
    def test_inputs_shapes_and_order(self):
        from mlsynth.config_models import CMBSTSConfig
        inp = prepare_cmbsts_inputs(CMBSTSConfig(**base_cfg(make_panel(), covariates=["sat"])))
        assert inp.Y.shape == (80, 2)
        assert inp.series_names == ["store", "comp"]      # treated first
        assert inp.control_names == ["wineA", "wineB", "wineC"]
        assert inp.covariate_names == ["sat"]
        assert inp.X.shape == (80, 1 + 3)                 # covariate + 3 controls
        assert inp.T0 == 60 and inp.T == 80

    def test_univariate_no_regressors(self):
        from mlsynth.config_models import CMBSTSConfig
        inp = prepare_cmbsts_inputs(CMBSTSConfig(**base_cfg(
            make_panel(), group_units=None, control_units=None, covariates=None)))
        assert inp.Y.shape == (80, 1)
        assert inp.X is None

    def test_missing_group_unit_raises(self):
        from mlsynth.config_models import CMBSTSConfig
        with pytest.raises(MlsynthDataError):
            prepare_cmbsts_inputs(CMBSTSConfig(**base_cfg(make_panel(), group_units=["ghost"])))

    def test_control_is_treated_raises(self):
        # control overlapping the *treated* unit is caught in setup (group vs
        # control overlap is caught earlier by the config validator).
        from mlsynth.config_models import CMBSTSConfig
        with pytest.raises(MlsynthDataError):
            prepare_cmbsts_inputs(CMBSTSConfig(**base_cfg(
                make_panel(), control_units=["store", "wineA"])))

    def test_missing_covariate_raises(self):
        from mlsynth.config_models import CMBSTSConfig
        with pytest.raises(MlsynthDataError):
            prepare_cmbsts_inputs(CMBSTSConfig(**base_cfg(make_panel(), covariates=["nope"])))

    def test_excl_dates_mask(self):
        from mlsynth.config_models import CMBSTSConfig
        inp = prepare_cmbsts_inputs(CMBSTSConfig(**base_cfg(make_panel(), excl_dates="drop")))
        assert inp.excl_post is not None
        assert inp.excl_post.sum() == 2                   # last two periods flagged

    def test_too_few_pre_periods_raises(self):
        from mlsynth.config_models import CMBSTSConfig
        df = make_panel(T=10, T0=1)                       # only one pre period
        with pytest.raises(MlsynthDataError):
            prepare_cmbsts_inputs(CMBSTSConfig(**base_cfg(df)))

    def test_dtw_without_fastdtw_raises_config_error(self):
        # fastdtw is optional; absent here, so the dtw screen must report cleanly.
        from mlsynth.config_models import CMBSTSConfig
        pytest.importorskip  # noqa: B018 - keep import side effect explicit below
        try:
            import fastdtw  # noqa: F401
            pytest.skip("fastdtw installed; guarded-error path not exercised")
        except ImportError:
            pass
        with pytest.raises(MlsynthConfigError):
            prepare_cmbsts_inputs(CMBSTSConfig(**base_cfg(
                make_panel(), control_units=None, control_selection="dtw", n_controls=2)))


# --------------------------------------------------------------------------- #
# Layer 1/2 -- config validation (failure paths)
# --------------------------------------------------------------------------- #
class TestConfigValidation:
    @pytest.mark.parametrize("over", [
        {"components": ["bogus"]},
        {"components": ["slope"]},                        # slope needs trend
        {"components": []},
        {"components": ["seasonal"]},                     # seasonal needs seas_period
        {"components": ["cycle"]},                        # cycle needs cycle_period
        {"niter": 100, "burn": 100},                      # burn >= niter
        {"control_selection": "dtw"},                     # dtw needs n_controls
        {"group_units": ["comp"], "control_units": ["comp"]},  # overlap
        {"ci_alpha": 1.5},                                # out of (0,1)
        {"niter": 3},                                     # below minimum
    ])
    def test_invalid_config_raises(self, over):
        with pytest.raises(MlsynthConfigError):
            CMBSTS(base_cfg(make_panel(), **over))


# --------------------------------------------------------------------------- #
# Layer 3 -- estimator orchestration
# --------------------------------------------------------------------------- #
class TestEstimator:
    def test_smoke_returns_results(self):
        res = CMBSTS(base_cfg(make_panel())).fit()
        assert isinstance(res, CMBSTSResults)
        assert np.isfinite(res.att)

    def test_recovers_known_effect(self):
        res = CMBSTS(base_cfg(make_panel(effect_store=8.0, effect_comp=3.0),
                              niter=800, burn=200)).fit()
        lo, hi = res.att_ci
        assert lo <= 8.0 <= hi                            # truth in the credible band
        det = res.inference_detail
        assert det.att_mean.shape == (2,)
        assert det.att_mean[0] > det.att_mean[1] > 0      # store effect > comp > 0

    def test_univariate_runs(self):
        res = CMBSTS(base_cfg(make_panel(), group_units=None,
                              control_units=None, covariates=None)).fit()
        assert res.inference_detail.att_mean.shape == (1,)
        assert res.posterior.inclusion_probs is None

    def test_covariates_spike_slab_inclusion(self):
        res = CMBSTS(base_cfg(make_panel(), covariates=["sat"])).fit()
        incl = res.posterior.inclusion_probs
        assert set(incl) == {"sat", "wineA", "wineB", "wineC"}
        assert all(0.0 <= p <= 1.0 for p in incl.values())

    def test_seasonal_component_runs(self):
        res = CMBSTS(base_cfg(make_panel(), components=["trend", "seasonal"],
                              seas_period=7, niter=300, burn=60)).fit()
        assert np.isfinite(res.att)
        assert res.posterior.components == ["trend", "seasonal"]

    def test_horizon_config_runs(self):
        res = CMBSTS(base_cfg(make_panel(), horizon=10, niter=300, burn=60)).fit()
        assert np.isfinite(res.att)
        lo, hi = res.att_ci
        assert lo <= hi

    def test_reproducible_with_seed(self):
        a = CMBSTS(base_cfg(make_panel(), seed=7)).fit()
        b = CMBSTS(base_cfg(make_panel(), seed=7)).fit()
        assert a.att == pytest.approx(b.att)

    def test_dict_and_config_object_agree(self):
        from mlsynth.config_models import CMBSTSConfig
        cfg = base_cfg(make_panel(), seed=3)
        r1 = CMBSTS(cfg).fit()
        r2 = CMBSTS(CMBSTSConfig(**cfg)).fit()
        assert r1.att == pytest.approx(r2.att)

    def test_no_treated_unit_raises(self):
        # T0 == T means the treatment condition never fires -> no treated unit.
        df = make_panel(T=40, T0=40)
        with pytest.raises((MlsynthDataError, MlsynthEstimationError, MlsynthConfigError)):
            CMBSTS(base_cfg(df)).fit()


# --------------------------------------------------------------------------- #
# Layer 4 -- public API / result-contract conformance
# --------------------------------------------------------------------------- #
class TestResultContract:
    def test_two_family_result_contract(self):
        """CMBSTS conforms to the observational (EffectResult) contract.

        CMBSTS needs group/control units beyond the single-df harness inputs, so
        the contract is pinned here with a fast config.
        """
        res = CMBSTS(base_cfg(make_panel(), niter=300, burn=60)).fit()
        assert isinstance(res, MlsynthResult)
        assert isinstance(res, (EffectResult, BaseEstimatorResults))
        assert res.effects is not None and res.effects.att is not None
        assert res.time_series is not None
        assert res.time_series.counterfactual_outcome is not None
        assert res.weights is not None                    # empty but present (§4.4)
        assert res.method_details is not None and res.method_details.method_name == "CMBSTS"
        assert isinstance(res.att, float)
        assert res.att == pytest.approx(res.effects.att)
        cf = np.asarray(res.counterfactual)
        gap = np.asarray(res.gap)
        assert cf.ndim == 1 and cf.shape == gap.shape
        ci = res.att_ci
        assert ci is None or (len(ci) == 2 and ci[0] <= ci[1])
        assert res.inference is not None and res.inference.method == "bayesian_posterior"

    def test_frozen_result_immutable(self):
        import pydantic
        res = CMBSTS(base_cfg(make_panel(), niter=200, burn=40)).fit()
        with pytest.raises(pydantic.ValidationError):
            res.inputs = None                             # frozen field assignment


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
class TestCoverageBranches:
    """Exercise the remaining reachable validation / option branches."""

    def test_dtw_selection_with_stub(self, monkeypatch):
        import sys
        import types
        from mlsynth.config_models import CMBSTSConfig
        fake = types.ModuleType("fastdtw")
        fake.fastdtw = lambda a, b: (
            float(np.sum(np.abs(np.asarray(a) - np.asarray(b)[: len(a)]))), None)
        monkeypatch.setitem(sys.modules, "fastdtw", fake)
        inp = prepare_cmbsts_inputs(CMBSTSConfig(**base_cfg(
            make_panel(), control_units=None, control_selection="dtw", n_controls=2)))
        assert len(inp.control_names) == 2
        assert all(c in ("wineA", "wineB", "wineC") for c in inp.control_names)

    def test_dtw_pool_too_small_raises(self):
        from mlsynth.config_models import CMBSTSConfig
        with pytest.raises(MlsynthDataError):
            prepare_cmbsts_inputs(CMBSTSConfig(**base_cfg(
                make_panel(), control_units=None, control_selection="dtw", n_controls=99)))

    def test_nan_covariate_raises(self):
        from mlsynth.config_models import CMBSTSConfig
        df = make_panel()
        df.loc[(df["item"] == "store") & (df["week"] == 3), "sat"] = np.nan
        with pytest.raises(MlsynthDataError):
            prepare_cmbsts_inputs(CMBSTSConfig(**base_cfg(df, covariates=["sat"])))

    def test_multiple_treated_units_raises(self):
        from mlsynth.config_models import CMBSTSConfig
        df = make_panel()
        df.loc[(df["item"] == "comp") & (df["week"] >= 60), "treated"] = 1  # 2 treated
        with pytest.raises(MlsynthDataError):
            prepare_cmbsts_inputs(CMBSTSConfig(**base_cfg(df, group_units=None)))

    def test_treated_in_group_raises(self):
        from mlsynth.config_models import CMBSTSConfig
        with pytest.raises(MlsynthDataError):
            prepare_cmbsts_inputs(CMBSTSConfig(**base_cfg(make_panel(), group_units=["store"])))

    def test_control_not_found_raises(self):
        from mlsynth.config_models import CMBSTSConfig
        with pytest.raises(MlsynthDataError):
            prepare_cmbsts_inputs(CMBSTSConfig(**base_cfg(make_panel(), control_units=["ghost"])))

    def test_excl_dates_missing_column_raises(self):
        from mlsynth.config_models import CMBSTSConfig
        with pytest.raises(MlsynthDataError):
            prepare_cmbsts_inputs(CMBSTSConfig(**base_cfg(make_panel(), excl_dates="absent")))

    def test_inputs_period_properties(self):
        from mlsynth.config_models import CMBSTSConfig
        inp = prepare_cmbsts_inputs(CMBSTSConfig(**base_cfg(make_panel())))
        assert inp.pre_periods == 60
        assert inp.post_periods == 20

    @pytest.mark.parametrize("over", [
        {"components": ["slope", "seasonal"], "seas_period": 7},  # slope without trend
        {"control_selection": "bogus", "n_controls": 2},          # bad selection mode
    ])
    def test_more_invalid_configs(self, over):
        with pytest.raises(MlsynthConfigError):
            CMBSTS(base_cfg(make_panel(), **over))

    def test_estimation_error_is_translated(self, monkeypatch):
        import mlsynth.estimators.cmbsts as mod

        def boom(_config):
            raise ValueError("synthetic engine failure")

        monkeypatch.setattr(mod, "run_cmbsts", boom)
        with pytest.raises(MlsynthEstimationError):
            CMBSTS(base_cfg(make_panel())).fit()

    def test_display_graphs_runs(self):
        CMBSTS(base_cfg(make_panel(), niter=150, burn=30, display_graphs=True)).fit()
        plt.close("all")


class TestPlotting:
    def test_plot_cmbsts_returns_figure(self, tmp_path):
        res = CMBSTS(base_cfg(make_panel(), niter=200, burn=40)).fit()
        fig = plot_cmbsts(res, save=str(tmp_path / "cmbsts.png"))
        assert fig is not None
        assert (tmp_path / "cmbsts.png").exists()
        plt.close(fig)

    def test_result_plot_smoke(self):
        res = CMBSTS(base_cfg(make_panel(), niter=200, burn=40)).fit()
        res.plot()                                        # base single-series plot
        plt.close("all")
