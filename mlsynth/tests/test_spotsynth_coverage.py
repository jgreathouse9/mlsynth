"""Line/branch coverage tests for the SPOTSYNTH subsystem.

These exercise the remaining branches in the helpers that the behavioural
suite in ``test_spotsynth.py`` does not reach: the plotter render path, the
replication/driver harness, and the small fallback branches in
``screen`` / ``sc`` / ``debias`` / ``pipeline``. Heavy MCMC-bound paths reuse
tiny draw counts.
"""

from __future__ import annotations

import sys
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from mlsynth.exceptions import (  # noqa: E402
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.spotsynth_helpers import (  # noqa: E402
    prepare_spotsynth_inputs,
    proximal_debias,
    run_spotsynth,
    simplex_weights,
    simulate_spillover_panel,
    spillover_screen,
)
from mlsynth.utils.spotsynth_helpers.plotter import plot_spotsynth  # noqa: E402
from mlsynth.utils.spotsynth_helpers.screen import _bucketize  # noqa: E402


def _small_inputs(seed=0, n_donors=8, T0=20, n_post=6):
    df, _ = simulate_spillover_panel(
        n_donors=n_donors, T0=T0, n_post=n_post, sigma_x=0.3, seed=seed)
    return prepare_spotsynth_inputs(df, "Y", "treated", "unit", "time")


# ----------------------------------------------------------------------
# plotter.py
# ----------------------------------------------------------------------

class TestPlotter:
    def teardown_method(self):
        plt.close("all")

    def test_plot_frequentist_no_band(self):
        inp = _small_inputs()
        res = run_spotsynth(inp, selection="S1", forecast="loo",
                            inference="frequentist", n_donors=4)
        # No uncertainty band -> the plain plot path (extra={}).
        plot_spotsynth(res, time_axis_label="time", unit_label="unit",
                       treatment_label="treated", outcome_label="Y")
        plt.close("all")

    def test_plot_with_uncertainty_band(self):
        inp = _small_inputs()
        res = run_spotsynth(inp, selection="S1", forecast="loo",
                            inference="frequentist", n_donors=4)
        # Inject a fake credible band so the uncertainty branch is taken.
        cf = res.counterfactual
        object.__setattr__(res, "counterfactual_lower", cf - 1.0)
        object.__setattr__(res, "counterfactual_upper", cf + 1.0)
        plot_spotsynth(res)
        plt.close("all")

    def test_plot_typeerror_fallback(self):
        inp = _small_inputs()
        res = run_spotsynth(inp, selection="S1", forecast="loo",
                            inference="frequentist", n_donors=4)
        cf = res.counterfactual
        object.__setattr__(res, "counterfactual_lower", cf - 1.0)
        object.__setattr__(res, "counterfactual_upper", cf + 1.0)

        calls = {"n": 0}
        real = plt.figure  # keep matplotlib happy in the fallback

        def fake_plot(**kwargs):
            calls["n"] += 1
            if "uncertainty_intervals_array" in kwargs:
                raise TypeError("no uncertainty support")
            # fallback call -> succeed quietly
            real()
            plt.close("all")

        with mock.patch(
            "mlsynth.utils.spotsynth_helpers.plotter.plot_estimates",
            side_effect=fake_plot,
        ):
            plot_spotsynth(res)
        assert calls["n"] == 2  # first raises TypeError, fallback called
        plt.close("all")


# ----------------------------------------------------------------------
# estimator: display_graphs + error-wrapping branches
# ----------------------------------------------------------------------

class TestEstimator:
    def teardown_method(self):
        plt.close("all")

    def test_display_graphs_true(self):
        from mlsynth import SPOTSYNTH
        df, _ = simulate_spillover_panel(n_donors=12, T0=24, n_post=6,
                                         sigma_x=0.3, seed=0)
        res = SPOTSYNTH({"df": df, "outcome": "Y", "treat": "treated",
                         "unitid": "unit", "time": "time", "selection": "S1",
                         "n_donors": 4, "inference": "frequentist",
                         "display_graphs": True}).fit()
        assert res.att is not None
        plt.close("all")

    def test_accepts_config_object(self):
        # Passing a SPOTSYNTHConfig instance (not a dict) skips the dict branch.
        from mlsynth import SPOTSYNTH
        from mlsynth.config_models import SPOTSYNTHConfig
        df, _ = simulate_spillover_panel(n_donors=8, T0=20, n_post=6, seed=0)
        cfg = SPOTSYNTHConfig(df=df, outcome="Y", treat="treated", unitid="unit",
                              time="time", selection="S1", n_donors=4,
                              inference="frequentist", display_graphs=False)
        res = SPOTSYNTH(cfg).fit()
        assert res.metadata["estimator"] == "SPOTSYNTH"

    def test_data_error_passthrough(self):
        # A MlsynthDataError from setup propagates unchanged (not re-wrapped).
        from mlsynth import SPOTSYNTH
        df, _ = simulate_spillover_panel(n_donors=6, T0=18, n_post=5, seed=0)
        df["treated"] = 0  # no treated unit
        with pytest.raises(MlsynthDataError):
            SPOTSYNTH({"df": df, "outcome": "Y", "treat": "treated",
                       "unitid": "unit", "time": "time",
                       "inference": "frequentist",
                       "display_graphs": False}).fit()

    def test_generic_exception_wrapped(self):
        # A non-mlsynth exception inside fit() is wrapped as
        # MlsynthEstimationError.
        from mlsynth import SPOTSYNTH
        df, _ = simulate_spillover_panel(n_donors=8, T0=20, n_post=6, seed=0)
        est = SPOTSYNTH({"df": df, "outcome": "Y", "treat": "treated",
                         "unitid": "unit", "time": "time",
                         "inference": "frequentist", "display_graphs": False})
        with mock.patch(
            "mlsynth.estimators.spotsynth.run_spotsynth",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(MlsynthEstimationError, match="SPOTSYNTH estimation failed"):
                est.fit()


# ----------------------------------------------------------------------
# screen.py fallback branches
# ----------------------------------------------------------------------

class TestScreenBranches:
    def test_bucketize_no_full_groups(self):
        # idx shorter than k -> no full groups -> fallback to [idx].
        Z = np.arange(12, dtype=float).reshape(6, 2)
        out = _bucketize(Z, np.array([0, 1]), k=5)
        assert out.shape[0] == 1

    def test_unknown_forecast_raises(self):
        inp = _small_inputs()
        with pytest.raises(ValueError, match="Unknown forecast"):
            spillover_screen(inp.D, inp.T0, inp.donor_names, forecast="bogus")

    def test_lag_too_few_buckets_raises(self):
        inp = _small_inputs(T0=20, n_post=6)
        # bucket so large that fewer than 3 pre-buckets remain.
        with pytest.raises(ValueError, match="Too few"):
            spillover_screen(inp.D, inp.T0, inp.donor_names,
                             forecast="lag", time_average=10)

    def test_s2_fallback_to_s1_warns(self):
        # Tight PPI -> S2 keeps < 2 donors -> warns and falls back.
        inp = _small_inputs(seed=4)
        with pytest.warns(RuntimeWarning, match="fewer than 2 donors"):
            sc = spillover_screen(inp.D, inp.T0, inp.donor_names,
                                  selection="S2", forecast="lag", ppi=0.01)
        assert sc.selected_idx.size >= 2

    def test_default_keep_with_explicit_n(self):
        inp = _small_inputs()
        sc = spillover_screen(inp.D, inp.T0, inp.donor_names,
                              selection="S1", n_donors=3)
        assert sc.selected_idx.size == 3
        # n_donors above pool size clamps to the pool.
        sc2 = spillover_screen(inp.D, inp.T0, inp.donor_names,
                               selection="S1", n_donors=999)
        assert sc2.selected_idx.size == inp.n_donors


# ----------------------------------------------------------------------
# sc.py edge branches
# ----------------------------------------------------------------------

class TestSimplexWeights:
    def test_empty_donor_matrix_raises(self):
        y = np.arange(10, dtype=float)
        D = np.empty((10, 0))
        with pytest.raises(MlsynthEstimationError, match="No donors"):
            simplex_weights(y, D, 6)

    def test_solver_returns_none(self):
        # A solve that leaves w.value None -> the "no solution" guard fires.
        y = np.arange(10, dtype=float)
        D = np.random.default_rng(0).normal(size=(10, 3))
        with mock.patch(
            "mlsynth.utils.spotsynth_helpers.sc.cp.Problem.solve",
            lambda self, *a, **k: None,
        ):
            with pytest.raises(MlsynthEstimationError, match="no solution"):
                simplex_weights(y, D, 6)

    def test_all_zero_weights_branch(self):
        # weights sum to 0 after clipping -> the s>0 normalisation is skipped.
        y = np.arange(10, dtype=float)
        D = np.random.default_rng(1).normal(size=(10, 3))

        def fake_solve(self, *a, **k):
            self.variables()[0].value = np.zeros(3)

        with mock.patch(
            "mlsynth.utils.spotsynth_helpers.sc.cp.Problem.solve",
            fake_solve,
        ):
            w, cf = simplex_weights(y, D, 6)
        assert w.sum() == 0.0

    def test_clarabel_failure_falls_back_to_scs(self):
        y = np.arange(12, dtype=float)
        D = np.random.default_rng(2).normal(size=(12, 4))
        real_solve = simplex_weights.__globals__["cp"].Problem.solve
        calls = {"n": 0}

        def flaky_solve(self, solver=None, **k):
            calls["n"] += 1
            if calls["n"] == 1:  # first (CLARABEL) attempt fails
                raise RuntimeError("clarabel boom")
            return real_solve(self, **k)

        with mock.patch(
            "mlsynth.utils.spotsynth_helpers.sc.cp.Problem.solve",
            flaky_solve,
        ):
            w, cf = simplex_weights(y, D, 8)
        assert calls["n"] >= 2
        assert abs(w.sum() - 1.0) < 1e-6


# ----------------------------------------------------------------------
# debias.py fallback (fewer instruments than kept donors)
# ----------------------------------------------------------------------

class TestDebiasFallback:
    def test_too_few_instruments_falls_back_to_ols(self):
        rng = np.random.default_rng(0)
        T, T0, k = 30, 22, 3
        X = rng.normal(size=(T, k))
        y = X @ np.array([0.5, 0.3, 0.2]) + rng.normal(0, 0.1, T)
        # only 1 instrument < k=3 -> OLS-on-X fallback (lines 89-93).
        Z = rng.normal(size=(T, 1))
        fit = proximal_debias(y, X, Z, T0)
        assert fit.n_instruments == 1
        assert fit.weights.shape == (k,)

    def test_zero_instruments_falls_back(self):
        rng = np.random.default_rng(1)
        T, T0, k = 30, 22, 2
        X = rng.normal(size=(T, k))
        y = X @ np.array([0.6, 0.4]) + rng.normal(0, 0.1, T)
        Z = np.empty((T, 0))
        fit = proximal_debias(y, X, Z, T0)
        assert fit.n_instruments == 0


# ----------------------------------------------------------------------
# pipeline.py debias branch
# ----------------------------------------------------------------------

class TestPipelineDebias:
    def test_debias_path(self):
        inp = _small_inputs(seed=2, n_donors=10, T0=24, n_post=6)
        res = run_spotsynth(inp, selection="S1", forecast="loo", n_donors=4,
                            inference="frequentist", debias=True)
        assert res.att_debiased is not None

    def test_debias_skipped_when_nothing_excluded(self):
        inp = _small_inputs(seed=3, n_donors=6, T0=20, n_post=5)
        res = run_spotsynth(inp, selection="all", inference="frequentist",
                            debias=True)
        # 'all' excludes nobody -> debias short-circuits.
        assert res.att_debiased is None


# ----------------------------------------------------------------------
# replication.py: simulation, power analysis, semi-synthetic demos
# ----------------------------------------------------------------------

class TestReplicationLight:
    def test_run_spotsynth_simulation_verbose(self, capsys):
        from mlsynth.utils.spotsynth_helpers import (
            SpotSimConfig, run_spotsynth_simulation)
        cfg = SpotSimConfig(n_donors=12, T0=18, n_post=5, n_keep=4, n_reps=2,
                            n_factors=3, noise_levels=(0.3,))
        out = run_spotsynth_simulation(cfg, seed=0, verbose=True)
        assert set(out[0.3]) == {"All", "Valid", "S1", "S2"}
        printed = capsys.readouterr().out
        assert "noise" in printed

    def test_run_spotsynth_simulation_quiet(self):
        from mlsynth.utils.spotsynth_helpers import (
            SpotSimConfig, run_spotsynth_simulation)
        cfg = SpotSimConfig(n_donors=10, T0=16, n_post=4, n_keep=4, n_reps=1,
                            n_factors=3, noise_levels=(0.5,))
        out = run_spotsynth_simulation(cfg, seed=1, verbose=False)
        assert 0.5 in out

    def test_power_analysis_verbose(self, capsys):
        from mlsynth.utils.spotsynth_helpers import run_forecast_power_analysis
        out = run_forecast_power_analysis(
            n_donors=14, T0=18, n_post=8, invalid_fracs=(0.3,),
            ramps=(1, 24), n_factors=4, n_reps=2, verbose=True)
        assert (0.3, 1) in out and (0.3, 24) in out
        printed = capsys.readouterr().out
        assert "lag" in printed and "loo" in printed

    def test_auc_degenerate_returns_nan(self):
        from mlsynth.utils.spotsynth_helpers.replication import _auc
        score = np.array([1.0, 2.0, 3.0])
        # all valid -> no invalid -> nan
        assert np.isnan(_auc(score, np.array([False, False, False])))
        # all invalid -> no valid -> nan
        assert np.isnan(_auc(score, np.array([True, True, True])))


class TestBayesWarnings:
    """Cover the NUTS divergence / low-acceptance RuntimeWarnings (bayes.py)."""

    def _data(self):
        rng = np.random.default_rng(0)
        T, T0, n = 24, 18, 3
        D = rng.normal(size=(T, n))
        y = D @ np.array([0.5, 0.3, 0.2]) + rng.normal(0, 0.2, T)
        return y, D, T0

    def test_divergence_warning(self):
        pytest.importorskip("numpyro")
        from mlsynth.utils.spotsynth_helpers import bayes as bmod

        y, D, T0 = self._data()
        real_get = bmod.__dict__  # placeholder

        class _FakeExtra(dict):
            pass

        orig_run = None

        import numpyro.infer as ni

        orig_extra = ni.MCMC.get_extra_fields

        def fake_extra(self, *a, **k):
            out = dict(orig_extra(self, *a, **k))
            out["diverging"] = np.ones_like(np.asarray(out["accept_prob"]),
                                            dtype=bool)
            return out

        with mock.patch.object(ni.MCMC, "get_extra_fields", fake_extra):
            with pytest.warns(RuntimeWarning, match="divergent transition"):
                bmod.bayesian_simplex_sc(y, D, T0, n_samples=300, n_warmup=200,
                                         n_chains=1, seed=0)
        del real_get, orig_run

    def test_low_acceptance_warning(self):
        pytest.importorskip("numpyro")
        from mlsynth.utils.spotsynth_helpers import bayes as bmod

        y, D, T0 = self._data()
        import numpyro.infer as ni

        orig_extra = ni.MCMC.get_extra_fields

        def fake_extra(self, *a, **k):
            out = dict(orig_extra(self, *a, **k))
            ap = np.asarray(out["accept_prob"])
            out["accept_prob"] = np.zeros_like(ap)   # force low acceptance
            out["diverging"] = np.zeros_like(ap, dtype=bool)
            return out

        with mock.patch.object(ni.MCMC, "get_extra_fields", fake_extra):
            with pytest.warns(RuntimeWarning, match="acceptance probability is low"):
                bmod.bayesian_simplex_sc(y, D, T0, n_samples=300, n_warmup=200,
                                         n_chains=1, seed=0)


class TestSemiSyntheticDemos:
    """Drive _semi_synthetic_demo branches with a small in-memory panel."""

    def _panel(self, n=8, T=24, t0=18, seed=0):
        rng = np.random.default_rng(seed)
        f = np.cumsum(rng.normal(0, 1, T))
        rows = []
        for t in range(T):
            rows.append({"state": "California", "year": 1970 + t,
                         "cigsale": float(f[t] + (-20.0 if t >= t0 - 1970 else 0)
                                          if False else f[t])})
            for j in range(n):
                rows.append({"state": f"s{j}", "year": 1970 + t,
                             "cigsale": float(f[t] + rng.normal(0, 0.4))})
        return pd.DataFrame(rows)

    def test_prop99_with_dataframe_frequentist_path(self):
        # Patch the Bayesian fit to frequentist so the demo runs fast and the
        # verbose printing branch (with att_ci=None) is exercised.
        import mlsynth.utils.spotsynth_helpers.replication as rep

        df = self._panel(seed=1)
        captured = {}

        from mlsynth.estimators.spotsynth import SPOTSYNTH

        orig_init = SPOTSYNTH.__init__

        def patched_init(self, config):
            if isinstance(config, dict):
                config = dict(config)
                config["inference"] = "frequentist"
            orig_init(self, config)

        with mock.patch.object(SPOTSYNTH, "__init__", patched_init):
            r = rep.replicate_prop99_spillover(
                data=df.rename(columns={}), n_keep=4, sigma=0.3, seed=0,
                verbose=True)
        for key in ("oracle_att", "contaminated_att", "screened_att",
                    "synthetic_donor_excluded", "results"):
            assert key in r
        captured  # silence linter

    def test_germany_and_basque_use_correct_columns(self):
        # Just confirm the column-subset wiring runs by feeding renamed frames.
        import mlsynth.utils.spotsynth_helpers.replication as rep

        from mlsynth.estimators.spotsynth import SPOTSYNTH

        orig_init = SPOTSYNTH.__init__

        def patched_init(self, config):
            if isinstance(config, dict):
                config = dict(config)
                config["inference"] = "frequentist"
            orig_init(self, config)

        rng = np.random.default_rng(0)
        T, t0, n = 22, 1985, 8
        f = np.cumsum(rng.normal(0, 1, T))
        grows, brows = [], []
        for t in range(T):
            grows.append({"country": "West Germany", "year": 1974 + t,
                          "gdp": float(f[t] * 100)})
            brows.append({"regionname": "Basque Country (Pais Vasco)",
                          "year": 1960 + t, "gdpcap": float(f[t])})
            for j in range(n):
                grows.append({"country": f"c{j}", "year": 1974 + t,
                              "gdp": float(f[t] * 100 + rng.normal(0, 5))})
                brows.append({"regionname": f"r{j}", "year": 1960 + t,
                              "gdpcap": float(f[t] + rng.normal(0, 0.3))})
        gdf = pd.DataFrame(grows)
        bdf = pd.DataFrame(brows)

        with mock.patch.object(SPOTSYNTH, "__init__", patched_init):
            rg = rep.replicate_germany_spillover(data=gdf, n_keep=4, sigma=5.0,
                                                 seed=0, verbose=False)
            rb = rep.replicate_basque_spillover(data=bdf, n_keep=4, sigma=0.3,
                                                seed=0, verbose=False)
        assert "screened_att" in rg and "screened_att" in rb

    def test_url_default_branches(self):
        # data=None -> the module reads its built-in URL via pd.read_csv. Mock
        # read_csv so no network is needed; this covers the `data = URL` lines.
        import mlsynth.utils.spotsynth_helpers.replication as rep

        seen = {}

        cols = {"state": [], "year": [], "cigsale": [], "country": [],
                "gdp": [], "regionname": [], "gdpcap": []}

        def fake_read_csv(url):
            seen["url"] = url
            # Return a frame carrying every column any demo selects; the demo
            # itself is stubbed out below so its content is irrelevant.
            return pd.DataFrame(cols)

        def fake_demo(*a, **k):
            return {"label": k.get("label", "x")}

        for fn, url_attr in (
            (rep.replicate_prop99_spillover, "PROP99_URL"),
            (rep.replicate_germany_spillover, "GERMANY_URL"),
            (rep.replicate_basque_spillover, "BASQUE_URL"),
        ):
            with mock.patch.object(rep.pd, "read_csv", fake_read_csv), \
                 mock.patch.object(rep, "_semi_synthetic_demo", fake_demo):
                fn(data=None, verbose=False)
            assert seen["url"] == getattr(rep, url_attr)

    def test_replicate_all_spillover_and_main(self, capsys):
        # Cover replicate_all_spillover (incl. the verbose summary) by stubbing
        # each per-panel replicate to return a lightweight result dict.
        import mlsynth.utils.spotsynth_helpers.replication as rep

        stub = {
            "label": "Stub", "oracle_att": 1.0, "contaminated_att": 0.5,
            "screened_att": 0.9, "pre_rmse": 0.1,
            "synthetic_donor_excluded": True,
        }
        with mock.patch.object(rep, "replicate_prop99_spillover",
                               return_value=stub), \
             mock.patch.object(rep, "replicate_germany_spillover",
                               return_value=stub), \
             mock.patch.object(rep, "replicate_basque_spillover",
                               return_value=stub):
            out = rep.replicate_all_spillover(verbose=True)
        assert set(out) == {"prop99", "germany", "basque"}
        assert "Summary" in capsys.readouterr().out
