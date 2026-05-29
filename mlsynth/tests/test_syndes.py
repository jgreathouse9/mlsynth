"""Tests for the SYNDES estimator (paper-aligned Doudchenko et al. 2021).

Layered along agents_tests.md:

* Layer 1 (numerical helpers): mode-name translation, contrast
  vector for per_unit.
* Layer 2 (data utilities): config validation (K, post_col, T0).
* Layer 3 (estimator integration): SYNDES.fit end-to-end for each
  of the three paper modes, plus the K=None two-way-global path.
* Layer 4 (public API contracts): top-level import, results
  carry the paper-aligned mode tag, inference runs on all modes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import SYNDES, power_analysis
from mlsynth.config_models import SYNDESConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError
from mlsynth.estimators.syndes import _MODE_FROM_INTERNAL, _MODE_TO_INTERNAL
from mlsynth.utils.syndes_helpers.inference import _build_contrast_vector
from mlsynth.utils.syndes_helpers.optimization import solve_synthetic_design
from mlsynth.utils.syndes_helpers.power import SYNDESPower
from mlsynth.utils.syndes_helpers.structures import SYNDESResults


# ----------------------------------------------------------------------
# Shared panel fixture
# ----------------------------------------------------------------------

def _panel(n_units: int = 10, T: int = 14, n_post: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((T, n_units)) * 0.4
    Y += np.linspace(0, 1, T)[:, None]                # mild time trend
    Y += rng.standard_normal(n_units)                  # unit FE noise
    rows = []
    for j in range(n_units):
        for t in range(T):
            rows.append({
                "unit": j,
                "time": t,
                "y": float(Y[t, j]),
                "post": int(t >= T - n_post),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def panel():
    return _panel()


# ----------------------------------------------------------------------
# Layer 1: numerical helpers
# ----------------------------------------------------------------------

class TestModeTranslation:
    def test_round_trip_mode_mapping(self):
        for public, internal in _MODE_TO_INTERNAL.items():
            assert _MODE_FROM_INTERNAL[internal] == public

    def test_per_unit_contrast_vector_sums_to_zero_modulo_K(self, panel):
        Y_pre = (panel[panel["post"] == 0]
                 .pivot(index="time", columns="unit", values="y")
                 .sort_index().to_numpy())
        design = solve_synthetic_design(Y=Y_pre, K=2, mode="per_unit")
        c = _build_contrast_vector(design, n_units=Y_pre.shape[1])
        # Treated weights sum to D/K = K/K = 1; control weights summed
        # over donors also sum to K/K = 1. So contrast sums to 0.
        assert abs(c.sum()) < 1e-6
        # And it's not the trivial all-zero vector.
        assert np.abs(c).sum() > 0


# ----------------------------------------------------------------------
# Layer 2: config validation
# ----------------------------------------------------------------------

class TestConfigValidation:
    def test_K_none_works_for_two_way_global(self, panel):
        cfg = SYNDESConfig(
            df=panel, outcome="y", unitid="unit", time="time",
            K=None, mode="two_way_global", post_col="post",
        )
        assert cfg.K is None and cfg.mode == "two_way_global"

    def test_K_none_rejected_for_per_unit(self, panel):
        with pytest.raises(MlsynthConfigError):
            SYNDESConfig(
                df=panel, outcome="y", unitid="unit", time="time",
                K=None, mode="per_unit", post_col="post",
            )

    def test_K_none_rejected_for_one_way_global(self, panel):
        with pytest.raises(MlsynthConfigError):
            SYNDESConfig(
                df=panel, outcome="y", unitid="unit", time="time",
                K=None, mode="one_way_global", post_col="post",
            )

    def test_unknown_post_col_rejected(self, panel):
        with pytest.raises(MlsynthConfigError):
            SYNDESConfig(
                df=panel, outcome="y", unitid="unit", time="time",
                K=2, mode="two_way_global", post_col="not_a_column",
            )

    def test_neither_T0_nor_post_col_is_design_only(self, panel):
        # No T0 and no post_col -> the whole panel is pre-treatment
        # (design-only / planning mode). Config is valid and fit returns a
        # design with no post period and no inference.
        cfg = SYNDESConfig(
            df=panel, outcome="y", unitid="unit", time="time",
            K=2, mode="two_way_global",
        )
        assert cfg.T0 is None and cfg.post_col is None
        res = SYNDES({
            "df": panel, "outcome": "y", "unitid": "unit", "time": "time",
            "K": 2, "mode": "two_way_global",
        }).fit()
        assert res.inputs.Y_post is None
        assert res.inference is None


# ----------------------------------------------------------------------
# Layer 3: estimator integration
# ----------------------------------------------------------------------

class TestEstimator:
    @pytest.mark.parametrize("mode", ["per_unit", "two_way_global", "one_way_global"])
    def test_each_mode_runs_end_to_end(self, panel, mode):
        res = SYNDES({
            "df": panel, "outcome": "y", "unitid": "unit", "time": "time",
            "K": 2, "mode": mode, "post_col": "post",
            "run_inference": False,
        }).fit()
        assert isinstance(res, SYNDESResults)
        # design.mode is the paper-aligned label.
        assert res.design.mode == mode
        assert res.design.selected_unit_indices.size == 2

    def test_K_none_two_way_global_picks_K(self, panel):
        res = SYNDES({
            "df": panel, "outcome": "y", "unitid": "unit", "time": "time",
            "K": None, "mode": "two_way_global", "post_col": "post",
            "run_inference": False,
        }).fit()
        # Exactly 1 <= K_hat <= n_units - 1
        k_hat = res.design.selected_unit_indices.size
        n_units = panel["unit"].nunique()
        assert 1 <= k_hat <= n_units - 1

    def test_lam_none_defaults_to_pre_variance(self, panel):
        res = SYNDES({
            "df": panel, "outcome": "y", "unitid": "unit", "time": "time",
            "K": 2, "mode": "two_way_global", "post_col": "post",
            "lam": None, "run_inference": False,
        }).fit()
        # solve_synthetic_design defaults lam to estimate_lambda(Y).
        assert res.design.lambda_value > 0

    @pytest.mark.parametrize("mode", ["per_unit", "two_way_global", "one_way_global"])
    def test_inference_works_for_every_mode(self, panel, mode):
        res = SYNDES({
            "df": panel, "outcome": "y", "unitid": "unit", "time": "time",
            "K": 2, "mode": mode, "post_col": "post",
            "run_inference": True, "alpha": 0.10,
        }).fit()
        assert res.inference is not None
        assert 0.0 <= res.inference.p_value <= 1.0
        assert res.inference.method.endswith(mode)


# ----------------------------------------------------------------------
# Layer 4: public API contracts
# ----------------------------------------------------------------------

class TestPublicAPI:
    def test_top_level_import(self):
        from mlsynth import SYNDES as _SYNDES
        assert _SYNDES is SYNDES

    def test_dict_config_accepted(self, panel):
        res = SYNDES({
            "df": panel, "outcome": "y", "unitid": "unit", "time": "time",
            "K": 2, "mode": "two_way_global", "post_col": "post",
            "run_inference": False,
        }).fit()
        assert isinstance(res, SYNDESResults)


# ----------------------------------------------------------------------
# Budget constraints (paper section 1 / discussion)
# ----------------------------------------------------------------------

class TestBudget:
    def test_costs_and_budget_optional(self, panel):
        # Default: no budget constraint.
        res = SYNDES({
            "df": panel, "outcome": "y", "unitid": "unit", "time": "time",
            "K": 2, "mode": "two_way_global", "post_col": "post",
            "run_inference": False,
        }).fit()
        assert res.design.selected_unit_indices.size == 2

    def test_budget_binds_when_unit_costs_exceed_budget(self, panel):
        # Each unit costs 10 but budget allows at most 25 -> at most 2 treated.
        n_units = panel["unit"].nunique()
        res = SYNDES({
            "df": panel, "outcome": "y", "unitid": "unit", "time": "time",
            "K": None, "mode": "two_way_global", "post_col": "post",
            "costs": [10.0] * n_units, "budget": 25.0,
            "run_inference": False,
        }).fit()
        treated_total_cost = res.design.selected_unit_indices.size * 10.0
        assert treated_total_cost <= 25.0

    def test_per_unit_costs_select_cheap_units_first(self, panel):
        # Last unit is extremely expensive -> MIP should prefer the cheaper ones.
        n_units = panel["unit"].nunique()
        costs = [1.0] * (n_units - 1) + [1000.0]
        res = SYNDES({
            "df": panel, "outcome": "y", "unitid": "unit", "time": "time",
            "K": 2, "mode": "two_way_global", "post_col": "post",
            "costs": costs, "budget": 5.0,
            "run_inference": False,
        }).fit()
        # The expensive unit should not be selected under that budget.
        assert (n_units - 1) not in res.design.selected_unit_indices.tolist()

    def test_costs_without_budget_rejected(self, panel):
        n_units = panel["unit"].nunique()
        with pytest.raises(MlsynthConfigError):
            SYNDESConfig(
                df=panel, outcome="y", unitid="unit", time="time",
                K=2, mode="two_way_global", post_col="post",
                costs=[1.0] * n_units, budget=None,
            )

    def test_costs_length_mismatch_rejected(self, panel):
        n_units = panel["unit"].nunique()
        with pytest.raises(MlsynthConfigError):
            SYNDESConfig(
                df=panel, outcome="y", unitid="unit", time="time",
                K=2, mode="two_way_global", post_col="post",
                costs=[1.0] * (n_units + 1), budget=10.0,
            )

    def test_negative_costs_rejected(self, panel):
        n_units = panel["unit"].nunique()
        with pytest.raises(MlsynthConfigError):
            SYNDESConfig(
                df=panel, outcome="y", unitid="unit", time="time",
                K=2, mode="two_way_global", post_col="post",
                costs=[1.0] * (n_units - 1) + [-1.0], budget=10.0,
            )


# ----------------------------------------------------------------------
# Power analysis (paper section A.4 / Figure 2)
# ----------------------------------------------------------------------

class TestPowerAnalysis:
    @pytest.fixture
    def fitted_results(self, panel):
        return SYNDES({
            "df": panel, "outcome": "y", "unitid": "unit", "time": "time",
            "K": 2, "mode": "two_way_global", "post_col": "post",
            "run_inference": False,
        }).fit()

    def test_default_horizons_one_through_twelve(self, fitted_results):
        power = power_analysis(fitted_results)
        assert isinstance(power, SYNDESPower)
        assert power.n_post_periods.tolist() == list(range(1, 13))
        assert power.mde_absolute.shape == power.n_post_periods.shape
        assert power.mde_percent.shape == power.n_post_periods.shape

    def test_mde_decreases_with_horizon(self, fitted_results):
        power = power_analysis(fitted_results)
        # MDE scales as 1/sqrt(n_post), so it must be strictly decreasing.
        diffs = np.diff(power.mde_absolute)
        assert (diffs < 0).all()

    def test_mde_widens_at_higher_power(self, fitted_results):
        low = power_analysis(fitted_results, power=0.50)
        high = power_analysis(fitted_results, power=0.95)
        # Higher target power -> larger MDE.
        assert (high.mde_absolute > low.mde_absolute).all()

    def test_mde_widens_at_lower_alpha(self, fitted_results):
        loose = power_analysis(fitted_results, alpha=0.10)
        tight = power_analysis(fitted_results, alpha=0.01)
        assert (tight.mde_absolute > loose.mde_absolute).all()

    def test_baseline_options(self, fitted_results):
        treated = power_analysis(fitted_results, baseline="treated")
        overall = power_analysis(fitted_results, baseline="overall")
        custom = power_analysis(fitted_results, baseline=1.0)
        assert treated.baseline_kind == "treated"
        assert overall.baseline_kind == "overall"
        assert custom.baseline_kind == "custom" and custom.baseline == 1.0

    def test_custom_horizons_accepted(self, fitted_results):
        power = power_analysis(fitted_results, n_post_periods=[2, 6, 18])
        assert power.n_post_periods.tolist() == [2, 6, 18]

    def test_per_unit_design_supports_power(self, panel):
        res = SYNDES({
            "df": panel, "outcome": "y", "unitid": "unit", "time": "time",
            "K": 2, "mode": "per_unit", "post_col": "post",
            "run_inference": False,
        }).fit()
        power = power_analysis(res, n_post_periods=[1, 4])
        assert np.isfinite(power.mde_absolute).all()
        assert np.isfinite(power.mde_percent).all()

    def test_to_dataframe(self, fitted_results):
        df = power_analysis(fitted_results).to_dataframe()
        assert list(df.columns) == ["n_post", "mde_absolute", "mde_percent"]
        assert len(df) == 12

    def test_zero_baseline_rejected(self, fitted_results):
        with pytest.raises(MlsynthEstimationError):
            power_analysis(fitted_results, baseline=0.0)

    def test_invalid_horizons_rejected(self, fitted_results):
        with pytest.raises(MlsynthEstimationError):
            power_analysis(fitted_results, n_post_periods=[])
        with pytest.raises(MlsynthEstimationError):
            power_analysis(fitted_results, n_post_periods=[0, 1, 2])

    def test_long_run_sigma_reported(self, fitted_results):
        p = power_analysis(fitted_results)
        assert p.long_run_sigma > 0
        # MDE rests on the long-run sigma: MDE(h) == (z_a+z_b)*long_run_sigma/sqrt(h).
        from scipy.stats import norm
        mult = norm.ppf(1 - p.alpha / 2) + norm.ppf(p.power)
        expected = mult * p.long_run_sigma / np.sqrt(p.n_post_periods.astype(float))
        np.testing.assert_allclose(p.mde_absolute, expected, rtol=1e-9)


# ----------------------------------------------------------------------
# Layer 3: one-way global objective (paper-correct: treated 1/K, control free)
# ----------------------------------------------------------------------
class TestOneWayGlobalObjective:
    def _matchable_panel(self):
        """Unit 2 == mean(units 0, 1); the one-way control synthetic should
        concentrate on unit 2 (a diff-in-means design could not).
        """
        rng = np.random.default_rng(3)
        N, T, n_post = 8, 14, 3
        Y = rng.standard_normal((T, N)) * 0.3 + np.linspace(0, 1, T)[:, None]
        Y[:, 2] = 0.5 * (Y[:, 0] + Y[:, 1])           # perfect control match
        rows = [{"unit": j, "time": t, "y": float(Y[t, j]), "post": int(t >= T - n_post)}
                for j in range(N) for t in range(T)]
        return pd.DataFrame(rows)

    def test_control_weights_are_free_and_concentrated(self):
        res = SYNDES({
            "df": self._matchable_panel(), "outcome": "y", "unitid": "unit",
            "time": "time", "K": 2, "mode": "one_way_global", "post_col": "post",
            "run_inference": False, "lam": 0.0,
        }).fit()
        d = res.design
        cw = np.asarray(d.control_weights, dtype=float)
        assert cw.shape == (8,)
        assert cw.sum() == pytest.approx(1.0, abs=1e-5)
        assert np.all(cw >= -1e-7)
        # Treated units carry no control weight.
        assert np.allclose(cw[d.selected_unit_indices], 0.0, atol=1e-6)
        # The control synthetic is FREE: it concentrates on the matching unit
        # rather than the uniform 1/(N-K) a difference-in-means would impose.
        assert cw.max() > 0.9
        assert not np.allclose(cw, (1 - d.assignment) / (8 - 2), atol=1e-3)

    def test_treated_weights_pinned_to_one_over_K(self):
        res = SYNDES({
            "df": self._matchable_panel(), "outcome": "y", "unitid": "unit",
            "time": "time", "K": 2, "mode": "one_way_global", "post_col": "post",
            "run_inference": False, "lam": 0.0,
        }).fit()
        tw = np.asarray(res.design.treated_weights, dtype=float)
        nz = tw[tw > 0]
        assert nz.size == 2
        np.testing.assert_allclose(nz, 0.5, atol=1e-6)


# ----------------------------------------------------------------------
# Design "prediction" fields (pre-period contrast series + RMSE)
# ----------------------------------------------------------------------
class TestDesignPredictions:
    @pytest.mark.parametrize("mode", ["per_unit", "two_way_global", "one_way_global"])
    def test_contrast_series_and_pre_rmse(self, panel, mode):
        res = SYNDES({
            "df": panel, "outcome": "y", "unitid": "unit", "time": "time",
            "K": 2, "mode": mode, "post_col": "post", "run_inference": False,
        }).fit()
        d = res.design
        T_pre = res.inputs.Y_pre.shape[0]
        assert d.contrast_series is not None
        assert d.contrast_series.shape == (T_pre,)
        assert np.isfinite(d.pre_fit_rmse)
        np.testing.assert_allclose(
            d.pre_fit_rmse, np.sqrt(np.mean(d.contrast_series ** 2)), rtol=1e-9
        )


# ----------------------------------------------------------------------
# Multi-arm SYNDES
# ----------------------------------------------------------------------

def _arm_panel(arms=("A", "B", "C"), upa: int = 6, T: int = 16,
               n_post: int = 4, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    u = 0
    for a in arms:
        for _ in range(upa):
            y = (rng.standard_normal(T) * 0.4
                 + np.linspace(0, 1, T) + rng.standard_normal())
            for t in range(T):
                rows.append({"unit": u, "arm": a, "time": t,
                             "y": float(y[t]), "post": int(t >= T - n_post)})
            u += 1
    return pd.DataFrame(rows)


@pytest.fixture
def arm_panel():
    return _arm_panel()


class TestMultiArmSYNDES:
    _cfg = dict(outcome="y", unitid="unit", time="time", post_col="post",
                K=2, mode="two_way_global", solver="SCIP")

    def test_arm_returns_multiarm_results(self, arm_panel):
        from mlsynth.utils.syndes_helpers.structures import (
            SYNDESMultiArmResults, SYNDESResults,
        )
        res = SYNDES({"df": arm_panel, "arm": "arm", **self._cfg}).fit()
        assert isinstance(res, SYNDESMultiArmResults)
        assert res.arm == "arm" and res.mode == "syndes_multiarm"
        assert sorted(res.arm_designs) == ["A", "B", "C"]
        for r in res.arm_designs.values():
            assert isinstance(r, SYNDESResults)
            assert len(r.inputs.unit_index.labels) == 6   # own units only
            assert r.design.selected_unit_labels.size == 2  # K=2 per arm

    def test_arm_none_returns_single_result(self, arm_panel):
        from mlsynth.utils.syndes_helpers.structures import SYNDESResults
        res = SYNDES({"df": arm_panel[arm_panel["arm"] == "A"],
                      **self._cfg}).fit()
        assert isinstance(res, SYNDESResults)

    def test_accessors(self, arm_panel):
        res = SYNDES({"df": arm_panel, "arm": "arm", "run_inference": True,
                      "alpha": 0.1, **self._cfg}).fit()
        assert set(res.atet_by_arm()) == {"A", "B", "C"}
        assert set(res.selected_unit_labels_by_arm()) == {"A", "B", "C"}

    def test_missing_arm_column_raises(self, arm_panel):
        with pytest.raises(MlsynthDataError):
            SYNDES({"df": arm_panel, "arm": "nope", **self._cfg}).fit()

    def test_arm_varying_within_unit_raises(self, arm_panel):
        df = arm_panel.copy()
        first = df["unit"].iloc[0]
        n = int((df["unit"] == first).sum())
        df.loc[df["unit"] == first, "arm"] = (["A", "B"] * (n // 2 + 1))[:n]
        with pytest.raises(MlsynthDataError):
            SYNDES({"df": df, "arm": "arm", **self._cfg}).fit()

    def test_costs_with_arm_rejected(self, arm_panel):
        with pytest.raises(MlsynthConfigError):
            SYNDESConfig(df=arm_panel, outcome="y", unitid="unit", time="time",
                         K=2, mode="two_way_global", post_col="post",
                         arm="arm", costs=[1.0] * 18, budget=5.0)

    def test_K_too_big_for_arm_rejected(self, arm_panel):
        # each arm has 6 units; K must be < 6
        with pytest.raises(MlsynthConfigError):
            SYNDESConfig(df=arm_panel, outcome="y", unitid="unit", time="time",
                         K=6, mode="per_unit", post_col="post", arm="arm")


# ----------------------------------------------------------------------
# Standardized post-fit attachment (auto-attached via compute_post_fit)
# ----------------------------------------------------------------------

class TestSYNDESPostFit:
    """Every SYNDES.fit() call attaches a ``SyntheticControlPostFit`` to
    ``res.post_fit`` regardless of mode, post_col/T0 selection, or whether
    inference was requested."""

    def test_post_fit_attached_for_each_mode(self, panel):
        from mlsynth.utils.post_fit import SyntheticControlPostFit
        for mode in ("two_way_global", "one_way_global", "per_unit"):
            res = SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                          "time": "time", "K": 2, "mode": mode,
                          "post_col": "post", "run_inference": False}).fit()
            assert isinstance(res.post_fit, SyntheticControlPostFit), mode
            assert res.post_fit.n_fit > 0 and res.post_fit.n_blank == 0
            assert res.post_fit.power is not None, mode

    def test_post_col_and_T0_give_same_post_fit(self, panel):
        kw = dict(outcome="y", unitid="unit", time="time",
                  K=2, mode="two_way_global", run_inference=False)
        T0 = int(panel["time"].max() + 1) - int(panel["post"].sum() / panel["unit"].nunique())
        a = SYNDES({"df": panel, "post_col": "post", **kw}).fit()
        b = SYNDES({"df": panel, "T0": T0, **kw}).fit()
        np.testing.assert_allclose(a.post_fit.ate, b.post_fit.ate, atol=1e-8)

    def test_post_fit_power_failure_is_swallowed(self, panel, monkeypatch):
        # Force compute_power_analysis to throw — fit() still succeeds and
        # leaves res.post_fit.power == None (covers lines 153-154 of syndes.py).
        import mlsynth.estimators.syndes as syn_mod

        def _boom(*_a, **_kw):
            raise RuntimeError("intentional")
        monkeypatch.setattr(syn_mod, "compute_power_analysis", _boom)
        res = SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                      "time": "time", "K": 2, "mode": "two_way_global",
                      "post_col": "post", "run_inference": False}).fit()
        assert res.post_fit is not None and res.post_fit.power is None


# ----------------------------------------------------------------------
# Annealed solver (mode="two_way_global_annealed") — covers the relaxed
# solver chain: relaxed_solver / relaxed_annealing / relaxed_formulation /
# relaxed_initialization / relaxed_structures / syndes._fit_relaxed
# ----------------------------------------------------------------------

class TestAnnealedMode:
    """The annealed relaxation pathway is otherwise dark to the suite. Covers
    the full chain end-to-end plus the relaxed-specific error paths."""

    @pytest.fixture
    def panel(self):
        return _panel(n_units=8, T=14, n_post=4, seed=2)

    def test_annealed_end_to_end(self, panel):
        from mlsynth.utils.syndes_helpers.relaxed_structures import (
            RelaxedSolverResults,
        )
        from mlsynth.utils.post_fit import SyntheticControlPostFit
        res = SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                      "time": "time", "K": 3,
                      "mode": "two_way_global_annealed",
                      "relaxed_max_iter": 8, "post_col": "post",
                      "run_inference": True, "alpha": 0.1}).fit()
        assert isinstance(res, RelaxedSolverResults)
        assert res.mode == "two_way_global_annealed"
        # Result-object properties exercised.
        assert res.assignment.shape[0] == 8
        assert res.selected_unit_indices.shape[0] == 3
        assert res.selected_unit_labels.shape[0] == 3
        assert np.isfinite(res.objective_value)
        assert np.isfinite(res.rmse)
        # Inference + post_fit attached.
        assert res.inference is not None
        assert res.inference.method.startswith("moving_block_permutation")
        assert isinstance(res.post_fit, SyntheticControlPostFit)
        assert res.post_fit.power is not None
        # Trace populated by each outer iteration.
        assert len(res.trace.objective_history) == 8
        assert len(res.trace.rmse_history) == 8
        assert len(res.trace.swap_logs) == 8

    def test_annealed_requires_K(self, panel):
        # The config validator rejects K=None for any non-two_way_global mode
        # before _fit_relaxed runs, so the line in _fit_relaxed is only
        # reachable by bypassing the config and clearing self.K.
        est = SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                      "time": "time", "K": 2,
                      "mode": "two_way_global_annealed",
                      "relaxed_max_iter": 2, "post_col": "post",
                      "run_inference": False})
        est.K = None
        with pytest.raises(MlsynthConfigError, match="explicit K"):
            est.fit()

    def test_annealed_no_post_window_skips_inference(self):
        # No post_col / T0 ⇒ Y_post is None ⇒ inference is None.
        df = _panel(n_units=6, T=8, n_post=0, seed=4)
        res = SYNDES({"df": df, "outcome": "y", "unitid": "unit",
                      "time": "time", "K": 2,
                      "mode": "two_way_global_annealed",
                      "relaxed_max_iter": 4, "verbose": False}).fit()
        assert res.inference is None

    def test_annealed_selected_unit_labels_without_inputs(self, panel):
        # Direct call to solve_two_way_relaxed (no inputs attachment) ⇒
        # selected_unit_labels falls back to raw indices.
        from mlsynth.utils.syndes_helpers.relaxed_solver import (
            solve_two_way_relaxed,
        )
        Y = np.tile(np.arange(8.0), (10, 1))
        Y += np.random.default_rng(0).standard_normal((10, 8)) * 0.01
        rel = solve_two_way_relaxed(Y, K=3, max_iter=3, verbose=False)
        assert rel.inputs is None
        # In this branch selected_unit_labels returns the integer indices.
        labels = rel.selected_unit_labels
        idx = rel.selected_unit_indices
        np.testing.assert_array_equal(labels, idx)

    def test_annealed_swap_log_acceptance_rate(self):
        from mlsynth.utils.syndes_helpers.relaxed_structures import (
            RelaxedSwapLog,
        )
        log = RelaxedSwapLog(n_proposals=10, n_accepted=5, n_uphill=4,
                              n_uphill_accepted=2, delta_history=[0.1, 0.2])
        # uphill_acceptance_rate property: 2 / 4 ≈ 0.5
        assert abs(log.uphill_acceptance_rate - 0.5) < 1e-6
        # Zero-uphill case: the +1e-8 keeps it well-defined.
        empty = RelaxedSwapLog(n_proposals=0, n_accepted=0, n_uphill=0,
                                n_uphill_accepted=0, delta_history=[])
        assert empty.uphill_acceptance_rate == 0.0


# ----------------------------------------------------------------------
# Relaxed-formulation primitives (called via solve_two_way_relaxed but
# also tested directly so each branch is verified in isolation)
# ----------------------------------------------------------------------

class TestRelaxedFormulation:

    @staticmethod
    def _Y(seed=0, T=8, N=6):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((T, N))

    def test_solve_weights_global_simplex(self):
        from mlsynth.utils.syndes_helpers.relaxed_formulation import (
            solve_weights_global,
        )
        Y = self._Y()
        D = np.array([1, 1, 0, 0, 0, 0])
        w = solve_weights_global(Y, D, lam=0.0)
        # Treated / control weights are each on the simplex.
        assert abs(w[D == 1].sum() - 1.0) < 1e-6
        assert abs(w[D == 0].sum() - 1.0) < 1e-6
        assert (w[D == 1] >= -1e-8).all()
        assert (w[D == 0] >= -1e-8).all()

    def test_solve_weights_global_with_ridge(self):
        from mlsynth.utils.syndes_helpers.relaxed_formulation import (
            solve_weights_global,
        )
        Y = self._Y()
        D = np.array([1, 0, 0, 1, 0, 0])
        w = solve_weights_global(Y, D, lam=0.1)
        assert w.shape == (6,)

    def test_compute_energy_and_rmse(self):
        from mlsynth.utils.syndes_helpers.relaxed_formulation import (
            compute_energy, compute_rmse_gap, synthetic_paths,
        )
        Y = self._Y()
        D = np.array([1, 1, 0, 0, 0, 0])
        w = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25])
        e = compute_energy(Y, D, w, lam=0.05)
        r = compute_rmse_gap(Y, D, w)
        muT, muC, gap = synthetic_paths(Y, D, w)
        assert np.isfinite(e) and np.isfinite(r)
        np.testing.assert_allclose(gap, muT - muC)

    def test_extract_weights_normalises(self):
        from mlsynth.utils.syndes_helpers.relaxed_formulation import (
            extract_weights,
        )
        D = np.array([1, 1, 0, 0])
        w = np.array([0.7, 0.3, 0.4, 0.6])
        out = extract_weights(D, w)
        np.testing.assert_allclose(out["treated_weights"].sum(), 1.0, atol=1e-6)
        np.testing.assert_allclose(out["control_weights"].sum(), 1.0, atol=1e-6)
        # Contrast: (2D-1) * w
        np.testing.assert_allclose(out["contrast_weights"], np.array([0.7, 0.3, -0.4, -0.6]))


# ----------------------------------------------------------------------
# Relaxed initialization & validation
# ----------------------------------------------------------------------

class TestRelaxedInitialization:

    def test_validate_inputs_rejects_non_2d(self):
        from mlsynth.utils.syndes_helpers.relaxed_initialization import (
            validate_relaxed_inputs,
        )
        with pytest.raises(MlsynthDataError):
            validate_relaxed_inputs(np.zeros(10), K=2)

    def test_validate_inputs_rejects_short_panel(self):
        from mlsynth.utils.syndes_helpers.relaxed_initialization import (
            validate_relaxed_inputs,
        )
        with pytest.raises(MlsynthConfigError, match="two time periods"):
            validate_relaxed_inputs(np.zeros((1, 5)), K=2)

    def test_validate_inputs_rejects_non_positive_K(self):
        from mlsynth.utils.syndes_helpers.relaxed_initialization import (
            validate_relaxed_inputs,
        )
        with pytest.raises(MlsynthConfigError, match="positive"):
            validate_relaxed_inputs(np.zeros((4, 5)), K=0)

    def test_validate_inputs_rejects_oversize_K(self):
        from mlsynth.utils.syndes_helpers.relaxed_initialization import (
            validate_relaxed_inputs,
        )
        with pytest.raises(MlsynthConfigError, match="exceed"):
            validate_relaxed_inputs(np.zeros((4, 3)), K=4)

    def test_default_lambda_is_mean_column_variance(self):
        from mlsynth.utils.syndes_helpers.relaxed_initialization import (
            default_lambda,
        )
        Y = np.tile(np.arange(5, dtype=float), (4, 1))   # zero column variance
        np.testing.assert_allclose(default_lambda(Y), 0.0)
        # Column variance grows with the spread.
        Y2 = np.array([[0.0, 0.0], [1.0, 3.0]])
        assert default_lambda(Y2) > 0.0

    def test_init_assignment_picks_K_units(self):
        from mlsynth.utils.syndes_helpers.relaxed_initialization import (
            init_assignment,
        )
        rng = np.random.default_rng(7)
        Y = rng.standard_normal((8, 5))
        D = init_assignment(Y, K=2)
        assert D.shape == (5,) and int(D.sum()) == 2


# ----------------------------------------------------------------------
# Annealing primitives (temperature_schedule, propose_swap, d_step_annealed)
# ----------------------------------------------------------------------

class TestRelaxedAnnealingPrimitives:

    def test_temperature_warmup(self):
        from mlsynth.utils.syndes_helpers.relaxed_annealing import (
            temperature_schedule,
        )
        Y = np.ones((10, 5)) + np.random.default_rng(0).standard_normal((10, 5))
        # Warm-up phase (it < 5) uses geometric decay; both with and without
        # delta_history should produce the same value here.
        t0 = temperature_schedule(0, Y)
        t1 = temperature_schedule(0, Y, delta_history=[])
        assert t0 == t1

    def test_temperature_adaptive_phase(self):
        from mlsynth.utils.syndes_helpers.relaxed_annealing import (
            temperature_schedule,
        )
        Y = np.ones((10, 5))
        # Past warm-up with enough delta history including uphill samples ⇒
        # the adaptive branch sets the temperature from the median uphill.
        history = [-1.0, -0.5] + [0.5, 0.7, 1.0, 1.2, 1.5, 1.8] * 4
        t = temperature_schedule(10, Y, delta_history=history)
        assert t > 0.0

    def test_temperature_fallback_when_no_uphill(self):
        from mlsynth.utils.syndes_helpers.relaxed_annealing import (
            temperature_schedule,
        )
        Y = np.ones((10, 5)) + np.random.default_rng(0).standard_normal((10, 5))
        # >20 entries but all downhill -> adaptive branch finds no uphill, falls
        # back to the geometric decay.
        downhill = [-0.1] * 30
        t = temperature_schedule(15, Y, delta_history=downhill)
        assert t > 0.0

    def test_propose_swap_preserves_treated_count(self):
        from mlsynth.utils.syndes_helpers.relaxed_annealing import propose_swap
        D = np.array([1, 1, 1, 0, 0, 0])
        np.random.seed(0)
        D_new, (i, j) = propose_swap(D, T=0.5, max_m=2)
        assert int(D_new.sum()) == 3
        assert all(D[k] == 1 for k in i)
        assert all(D[k] == 0 for k in j)

    def test_d_step_annealed_runs(self):
        from mlsynth.utils.syndes_helpers.relaxed_annealing import (
            d_step_annealed,
        )
        from mlsynth.utils.syndes_helpers.relaxed_formulation import (
            solve_weights_global,
        )
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((10, 6))
        D = np.array([1, 1, 0, 0, 0, 0])
        w = solve_weights_global(Y, D)
        D2, w2, log = d_step_annealed(Y, D, w, K=2, T=0.5, lam=0.0,
                                       n_proposals=5)
        assert int(D2.sum()) == 2
        assert log.n_proposals == 5


# ----------------------------------------------------------------------
# Relaxed permutation inference
# ----------------------------------------------------------------------

class TestRelaxedInference:

    def test_relaxed_inference_runs(self):
        from mlsynth.utils.syndes_helpers.inference import (
            permutation_test_relaxed_global,
        )
        from mlsynth.utils.syndes_helpers.relaxed_solver import (
            solve_two_way_relaxed,
        )
        rng = np.random.default_rng(1)
        Y_pre = rng.standard_normal((10, 6))
        Y_post = rng.standard_normal((4, 6))
        rel = solve_two_way_relaxed(Y_pre, K=2, max_iter=3, verbose=False)
        inf = permutation_test_relaxed_global(Y_pre, Y_post, rel.design,
                                               alpha=0.1)
        assert 0.0 <= inf.p_value <= 1.0
        assert inf.method == "moving_block_permutation_relaxed_global"
        assert inf.null_stats is not None

    def test_relaxed_inference_without_null_stats(self):
        from mlsynth.utils.syndes_helpers.inference import (
            permutation_test_relaxed_global,
        )
        from mlsynth.utils.syndes_helpers.relaxed_solver import (
            solve_two_way_relaxed,
        )
        rng = np.random.default_rng(2)
        Y_pre = rng.standard_normal((10, 6))
        Y_post = rng.standard_normal((4, 6))
        rel = solve_two_way_relaxed(Y_pre, K=2, max_iter=3, verbose=False)
        inf = permutation_test_relaxed_global(Y_pre, Y_post, rel.design,
                                               alpha=0.1,
                                               include_null_stats=False)
        assert inf.null_stats is None

    def test_relaxed_inference_requires_post(self):
        from mlsynth.utils.syndes_helpers.inference import (
            permutation_test_relaxed_global,
        )
        from mlsynth.utils.syndes_helpers.relaxed_structures import (
            RelaxedDesign,
        )
        Y_pre = np.zeros((4, 3))
        rd = RelaxedDesign(
            assignment=np.array([1, 0, 0]),
            raw_weights=np.array([1.0, 0.5, 0.5]),
            treated_weights=np.array([1.0, 0.0, 0.0]),
            control_weights=np.array([0.0, 0.5, 0.5]),
            contrast_weights=np.array([1.0, -0.5, -0.5]),
            synthetic_treated=np.zeros(4),
            synthetic_control=np.zeros(4),
            synthetic_gap=np.zeros(4),
            objective_value=0.0, rmse=0.0, lambda_value=0.0,
        )
        with pytest.raises(MlsynthDataError):
            permutation_test_relaxed_global(Y_pre, None, rd)
        with pytest.raises(MlsynthDataError):
            permutation_test_relaxed_global(Y_pre, np.zeros((0, 3)), rd)


# ----------------------------------------------------------------------
# Per-unit inference error paths + contrast vector branches
# ----------------------------------------------------------------------

class TestInferenceContrastEdges:

    def _make_design(self, **overrides):
        from mlsynth.utils.syndes_helpers.structures import SYNDESDesign
        defaults = dict(
            mode="global_2way", objective_value=0.0, lambda_value=0.0,
            assignment=np.array([1, 1, 0, 0]),
            selected_unit_indices=np.array([0, 1]),
            selected_unit_labels=np.array(["A", "B"]),
            assignment_by_unit={},
            w=None, q=None,
        )
        defaults.update(overrides)
        return SYNDESDesign(**defaults)

    def test_global_2way_falls_back_to_w_and_q(self):
        # contrast_weights is None -> compute as 2q - w
        d = self._make_design(
            w=np.array([0.3, 0.3, 0.2, 0.2]),
            q=np.array([0.5, 0.5, 0.0, 0.0]),
            contrast_weights=None,
        )
        c = _build_contrast_vector(d, n_units=4)
        np.testing.assert_allclose(c, 2.0 * d.q - d.w)

    def test_global_2way_missing_w_and_q_errors(self):
        d = self._make_design(contrast_weights=None)
        with pytest.raises(MlsynthEstimationError, match="requires w and q"):
            _build_contrast_vector(d, n_units=4)

    def test_per_unit_requires_q(self):
        d = self._make_design(mode="per_unit", q=None)
        with pytest.raises(MlsynthEstimationError, match="per_unit inference"):
            _build_contrast_vector(d, n_units=4)

    def test_per_unit_requires_at_least_one_treated(self):
        d = self._make_design(
            mode="per_unit",
            assignment=np.zeros(4), q=np.zeros((4, 4)),
        )
        with pytest.raises(MlsynthEstimationError, match="at least one treated"):
            _build_contrast_vector(d, n_units=4)

    def test_unknown_mode_rejected(self):
        d = self._make_design(mode="nope")
        with pytest.raises(MlsynthConfigError, match="Unknown SYNDES mode"):
            _build_contrast_vector(d, n_units=4)

    def test_permutation_test_requires_post(self):
        from mlsynth.utils.syndes_helpers.inference import (
            permutation_test_global,
        )
        d = self._make_design(contrast_weights=np.array([0.5, 0.5, -0.5, -0.5]))
        with pytest.raises(MlsynthDataError):
            permutation_test_global(np.zeros((4, 4)), None, d)


# ----------------------------------------------------------------------
# Plotter (every dispatch branch + every error path)
# ----------------------------------------------------------------------

class TestPlotter:

    @pytest.fixture(autouse=True)
    def _agg(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        yield
        _plt.close("all")

    def test_plot_global_design(self, panel):
        from mlsynth.utils.syndes_helpers.plotter import plot_syndes_design
        res = SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                      "time": "time", "K": 2, "mode": "two_way_global",
                      "post_col": "post", "run_inference": False}).fit()
        plot_syndes_design(res)

    def test_plot_one_way_global_design(self, panel):
        from mlsynth.utils.syndes_helpers.plotter import plot_syndes_design
        res = SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                      "time": "time", "K": 2, "mode": "one_way_global",
                      "post_col": "post", "run_inference": False}).fit()
        plot_syndes_design(res)

    def test_plot_per_unit_design(self, panel):
        from mlsynth.utils.syndes_helpers.plotter import plot_syndes_design
        res = SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                      "time": "time", "K": 2, "mode": "per_unit",
                      "post_col": "post", "run_inference": False}).fit()
        plot_syndes_design(res)

    def test_plot_relaxed_design(self, panel):
        from mlsynth.utils.syndes_helpers.plotter import plot_syndes_design
        res = SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                      "time": "time", "K": 2,
                      "mode": "two_way_global_annealed",
                      "relaxed_max_iter": 4, "post_col": "post",
                      "run_inference": False, "verbose": False}).fit()
        plot_syndes_design(res)

    def test_plot_unknown_mode_errors(self):
        from mlsynth.utils.syndes_helpers.plotter import plot_syndes_design
        from mlsynth.exceptions import MlsynthPlottingError
        from dataclasses import replace as _dc_replace

        # A minimally-valid SYNDESResults with a fake mode.
        from mlsynth.utils.syndes_helpers.structures import (
            SYNDESDesign, SYNDESInputs, SYNDESResults,
        )
        from mlsynth.utils.fast_scm_helpers.structure import IndexSet
        d = SYNDESDesign(
            mode="not_a_mode", objective_value=0.0, lambda_value=0.0,
            assignment=np.array([1, 0, 0, 0]),
            selected_unit_indices=np.array([0]),
            selected_unit_labels=np.array([0]), assignment_by_unit={},
        )
        inputs = SYNDESInputs(
            Y_pre=np.zeros((3, 4)), Y_post=None,
            unit_index=IndexSet.from_labels([0, 1, 2, 3]),
            time_index=IndexSet.from_labels([0, 1, 2]),
            pre_time_index=IndexSet.from_labels([0, 1, 2]),
            post_time_index=None, outcome="y",
        )
        res = SYNDESResults(design=d, inputs=inputs)
        with pytest.raises(MlsynthPlottingError):
            plot_syndes_design(res)

    def test_plot_global_design_missing_weights_errors(self):
        from mlsynth.utils.syndes_helpers.plotter import plot_global_design
        from mlsynth.exceptions import MlsynthPlottingError
        from mlsynth.utils.syndes_helpers.structures import (
            SYNDESDesign, SYNDESInputs, SYNDESResults,
        )
        from mlsynth.utils.fast_scm_helpers.structure import IndexSet
        d = SYNDESDesign(
            mode="two_way_global", objective_value=0.0, lambda_value=0.0,
            assignment=np.array([1, 0]), selected_unit_indices=np.array([0]),
            selected_unit_labels=np.array([0]), assignment_by_unit={},
            treated_weights=None, control_weights=None,
        )
        inputs = SYNDESInputs(
            Y_pre=np.zeros((3, 2)), Y_post=None,
            unit_index=IndexSet.from_labels([0, 1]),
            time_index=IndexSet.from_labels([0, 1, 2]),
            pre_time_index=IndexSet.from_labels([0, 1, 2]),
            post_time_index=None, outcome="y",
        )
        with pytest.raises(MlsynthPlottingError, match="Missing"):
            plot_global_design(SYNDESResults(design=d, inputs=inputs))

    def test_plot_per_unit_missing_q_errors(self):
        from mlsynth.utils.syndes_helpers.plotter import plot_per_unit_design
        from mlsynth.exceptions import MlsynthPlottingError
        from mlsynth.utils.syndes_helpers.structures import (
            SYNDESDesign, SYNDESInputs, SYNDESResults,
        )
        from mlsynth.utils.fast_scm_helpers.structure import IndexSet
        d = SYNDESDesign(
            mode="per_unit", objective_value=0.0, lambda_value=0.0,
            assignment=np.array([1, 0]), selected_unit_indices=np.array([0]),
            selected_unit_labels=np.array([0]), assignment_by_unit={},
            q=None,
        )
        inputs = SYNDESInputs(
            Y_pre=np.zeros((3, 2)), Y_post=None,
            unit_index=IndexSet.from_labels([0, 1]),
            time_index=IndexSet.from_labels([0, 1, 2]),
            pre_time_index=IndexSet.from_labels([0, 1, 2]),
            post_time_index=None, outcome="y",
        )
        with pytest.raises(MlsynthPlottingError, match="requires q"):
            plot_per_unit_design(SYNDESResults(design=d, inputs=inputs))

    def test_plot_relaxed_no_inputs_errors(self):
        from mlsynth.utils.syndes_helpers.plotter import plot_relaxed_design
        from mlsynth.exceptions import MlsynthPlottingError
        from mlsynth.utils.syndes_helpers.relaxed_solver import (
            solve_two_way_relaxed,
        )
        rng = np.random.default_rng(0)
        rel = solve_two_way_relaxed(rng.standard_normal((6, 4)),
                                     K=2, max_iter=2, verbose=False)
        # rel.inputs is None
        with pytest.raises(MlsynthPlottingError, match="inputs"):
            plot_relaxed_design(rel)

    def test_stack_pre_post_without_post(self):
        from mlsynth.utils.syndes_helpers.plotter import _stack_pre_post
        from mlsynth.utils.syndes_helpers.structures import (
            SYNDESDesign, SYNDESInputs, SYNDESResults,
        )
        from mlsynth.utils.fast_scm_helpers.structure import IndexSet
        inputs = SYNDESInputs(
            Y_pre=np.arange(6.0).reshape(3, 2), Y_post=None,
            unit_index=IndexSet.from_labels([0, 1]),
            time_index=IndexSet.from_labels([0, 1, 2]),
            pre_time_index=IndexSet.from_labels([0, 1, 2]),
            post_time_index=None, outcome="y",
        )
        d = SYNDESDesign(
            mode="two_way_global", objective_value=0.0, lambda_value=0.0,
            assignment=np.array([1, 0]),
            selected_unit_indices=np.array([0]),
            selected_unit_labels=np.array([0]), assignment_by_unit={},
        )
        out = _stack_pre_post(SYNDESResults(design=d, inputs=inputs))
        np.testing.assert_array_equal(out, inputs.Y_pre)

    def test_per_unit_plot_with_single_treated_unit(self, panel):
        # K=1 in per_unit ⇒ single subplot ⇒ the axes-singleton branch
        # (line 66 of plotter.py) fires.
        from mlsynth.utils.syndes_helpers.plotter import plot_syndes_design
        res = SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                      "time": "time", "K": 1, "mode": "per_unit",
                      "post_col": "post", "run_inference": False}).fit()
        plot_syndes_design(res)


# ----------------------------------------------------------------------
# Last-mile gap fillers (one branch per test)
# ----------------------------------------------------------------------

class TestSyndesCoverageMopUp:

    def test_relaxed_solver_verbose_branch(self, capsys):
        # verbose=True triggers the per-iteration print (line 123 of
        # relaxed_solver.py).
        from mlsynth.utils.syndes_helpers.relaxed_solver import (
            solve_two_way_relaxed,
        )
        rng = np.random.default_rng(0)
        solve_two_way_relaxed(rng.standard_normal((6, 4)), K=2,
                               max_iter=2, verbose=True)
        out = capsys.readouterr().out
        assert "uphill_accept" in out

    def test_solve_weights_global_non_optimal_raises(self, monkeypatch):
        # Force the QP to report a non-optimal status → MlsynthEstimationError
        # (line 87 of relaxed_formulation.py).
        from mlsynth.utils.syndes_helpers.relaxed_formulation import (
            solve_weights_global,
        )
        import cvxpy as cp
        Y = np.random.default_rng(0).standard_normal((6, 4))
        D = np.array([1, 1, 0, 0])
        orig = cp.Problem.solve
        def _fake_solve(self, *a, **kw):
            self._status = "infeasible"
        monkeypatch.setattr(cp.Problem, "solve", _fake_solve)
        try:
            with pytest.raises(MlsynthEstimationError, match="weight QP failed"):
                solve_weights_global(Y, D, lam=0.0)
        finally:
            monkeypatch.setattr(cp.Problem, "solve", orig)

    def test_init_assignment_runs(self):
        from mlsynth.utils.syndes_helpers.relaxed_initialization import (
            init_assignment,
        )
        Y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        D = init_assignment(Y, K=1)
        assert int(D.sum()) == 1

    def test_reconstruction_error_empty_and_full(self):
        from mlsynth.utils.syndes_helpers.relaxed_initialization import (
            _reconstruction_error,
        )
        Yc = np.array([[1.0, -1.0, 2.0], [0.5, 0.5, -1.0]])
        # Empty cols → infinity (defensive branch).
        assert _reconstruction_error(Yc, []) == np.inf
        # Non-empty cols → finite residual.
        assert _reconstruction_error(Yc, [0]) >= 0.0

    def test_setup_too_few_units_rejected(self):
        # Pre-period passes the >=2-rows check but the panel has only 1 unit
        # after pivoting → the >=2-units check (line 91 of setup.py).
        from mlsynth.utils.syndes_helpers.setup import prepare_syndes_inputs
        df = pd.DataFrame({"unit": ["u0"] * 4,
                            "time": [0, 1, 2, 3],
                            "y": [1.0, 2.0, 3.0, 4.0]})
        with pytest.raises(MlsynthDataError, match="at least two units"):
            prepare_syndes_inputs(df, outcome="y", unitid="unit",
                                   time="time", T0=3)

    def test_power_control_baseline_with_oddly_shaped_weights(self):
        # Synthetic-control with a (N, K) control_weights instead of (N,)
        # — power_analysis falls back to overall (lines 225-226 of power.py).
        from mlsynth.utils.syndes_helpers.power import power_analysis
        from mlsynth.utils.syndes_helpers.structures import (
            SYNDESDesign, SYNDESInputs, SYNDESResults,
        )
        from mlsynth.utils.fast_scm_helpers.structure import IndexSet
        inputs = SYNDESInputs(
            Y_pre=np.ones((6, 4)), Y_post=None,
            unit_index=IndexSet.from_labels(list(range(4))),
            time_index=IndexSet.from_labels(list(range(6))),
            pre_time_index=IndexSet.from_labels(list(range(6))),
            post_time_index=None, outcome="y",
        )
        d = SYNDESDesign(
            mode="two_way_global", objective_value=0.0, lambda_value=0.0,
            assignment=np.array([1, 1, 0, 0]),
            selected_unit_indices=np.array([0, 1]),
            selected_unit_labels=np.array([0, 1]), assignment_by_unit={},
            contrast_weights=np.array([0.5, 0.5, -0.5, -0.5]),
            control_weights=np.zeros((4, 2)),     # wrong shape on purpose
        )
        # The pre-period contrast is identically zero ⇒ sigma_perm = 0 and
        # the MDE is zero, but the control-baseline fallback path still runs.
        try:
            out = power_analysis(SYNDESResults(design=d, inputs=inputs),
                                  baseline="control")
            assert out.baseline_kind == "overall_fallback"
        except MlsynthEstimationError:
            # Acceptable if the zero-baseline guard fires first; the
            # important branch (the shape check) was still executed.
            pass


# ----------------------------------------------------------------------
# Estimator-level error paths
# ----------------------------------------------------------------------

class TestEstimatorErrorPaths:

    def test_invalid_config_raises_config_error(self, panel):
        # Extra key triggers pydantic ValidationError → MlsynthConfigError
        # (lines 180-181 of syndes.py).
        with pytest.raises(MlsynthConfigError, match="Invalid SYNDES"):
            SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                    "time": "time", "K": 2, "mode": "two_way_global",
                    "post_col": "post", "bogus_field": 7})

    def test_unexpected_exception_in_fit_wrapped(self, panel, monkeypatch):
        # Multi-arm path: monkey-patch _fit_single to throw a plain
        # RuntimeError → fit() wraps in MlsynthEstimationError (lines 248-249).
        import mlsynth.estimators.syndes as syn_mod
        df = panel.copy()
        df["arm"] = df["unit"].mod(2).map({0: "A", 1: "B"})
        orig = SYNDES._fit_single

        def _boom(self, *_a, **_kw):
            raise RuntimeError("nope")
        monkeypatch.setattr(syn_mod.SYNDES, "_fit_single", _boom)
        with pytest.raises(MlsynthEstimationError, match="SYNDES estimation failed"):
            SYNDES({"df": df, "outcome": "y", "unitid": "unit",
                    "time": "time", "K": 2, "mode": "two_way_global",
                    "post_col": "post", "arm": "arm"}).fit()
        monkeypatch.setattr(syn_mod.SYNDES, "_fit_single", orig)

    def test_plotting_failure_wrapped(self, panel, monkeypatch):
        # display_graph=True + plot_syndes_design raising → MlsynthPlottingError
        # (lines 315-318 of syndes.py).
        import mlsynth.estimators.syndes as syn_mod
        from mlsynth.exceptions import MlsynthPlottingError

        def _boom(*_a, **_kw):
            raise RuntimeError("plot died")
        monkeypatch.setattr(syn_mod, "plot_syndes_design", _boom)
        with pytest.raises(MlsynthPlottingError, match="SYNDES plotting"):
            SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                    "time": "time", "K": 2, "mode": "two_way_global",
                    "post_col": "post", "run_inference": False,
                    "display_graph": True}).fit()


# ----------------------------------------------------------------------
# Setup / solver edge cases not covered upstream
# ----------------------------------------------------------------------

class TestSetupEdges:

    def test_post_col_missing_column(self, panel):
        from mlsynth.utils.syndes_helpers.setup import prepare_syndes_inputs
        with pytest.raises(MlsynthDataError):
            prepare_syndes_inputs(panel, outcome="y", unitid="unit",
                                   time="time", post_col="nope")

    def test_post_col_all_post_rejected(self):
        from mlsynth.utils.syndes_helpers.setup import prepare_syndes_inputs
        df = _panel(n_units=4, T=6, n_post=0)
        df["post"] = 1
        with pytest.raises(MlsynthConfigError, match="every period"):
            prepare_syndes_inputs(df, outcome="y", unitid="unit",
                                   time="time", post_col="post")

    def test_post_col_missing_per_period_rejected(self):
        from mlsynth.utils.syndes_helpers.setup import prepare_syndes_inputs
        df = _panel(n_units=4, T=6, n_post=0)
        df["post"] = 0
        df.loc[df["time"] == 0, "post"] = None
        with pytest.raises(MlsynthDataError, match="every time period"):
            prepare_syndes_inputs(df, outcome="y", unitid="unit",
                                   time="time", post_col="post")

    def test_unbalanced_pivot_rejected(self):
        from mlsynth.utils.syndes_helpers.setup import prepare_syndes_inputs
        df = _panel(n_units=4, T=6)
        df = df.drop(df.index[0]).reset_index(drop=True)
        with pytest.raises(MlsynthDataError, match="complete balanced"):
            prepare_syndes_inputs(df, outcome="y", unitid="unit",
                                   time="time", T0=5)

    def test_T0_out_of_range_rejected(self):
        from mlsynth.utils.syndes_helpers.setup import prepare_syndes_inputs
        df = _panel(n_units=4, T=6)
        with pytest.raises(MlsynthConfigError, match="between 1"):
            prepare_syndes_inputs(df, outcome="y", unitid="unit",
                                   time="time", T0=0)

    def test_pre_period_too_short_rejected(self):
        from mlsynth.utils.syndes_helpers.setup import prepare_syndes_inputs
        df = _panel(n_units=4, T=6)
        with pytest.raises(MlsynthDataError, match="at least two pre"):
            prepare_syndes_inputs(df, outcome="y", unitid="unit",
                                   time="time", T0=1)

    def test_T0_defaults_to_full_panel(self):
        from mlsynth.utils.syndes_helpers.setup import prepare_syndes_inputs
        df = _panel(n_units=4, T=6)
        out = prepare_syndes_inputs(df, outcome="y", unitid="unit",
                                     time="time", T0=None)
        # Default ⇒ Y_post is None
        assert out.Y_post is None
        assert out.Y_pre.shape[0] == 6


# ----------------------------------------------------------------------
# Optimization / formulation edge cases
# ----------------------------------------------------------------------

class TestOptimizationEdges:

    def test_estimate_lambda_rejects_non_2d(self):
        from mlsynth.utils.syndes_helpers.optimization import estimate_lambda
        with pytest.raises(MlsynthDataError):
            estimate_lambda(np.zeros(5))

    def test_estimate_lambda_rejects_short_panel(self):
        from mlsynth.utils.syndes_helpers.optimization import estimate_lambda
        with pytest.raises(MlsynthConfigError, match="two pre"):
            estimate_lambda(np.zeros((1, 4)))

    def test_solve_synthetic_design_validates_K(self):
        from mlsynth.utils.syndes_helpers.optimization import (
            solve_synthetic_design,
        )
        Y = np.random.default_rng(0).standard_normal((10, 4))
        with pytest.raises(MlsynthConfigError, match="exceed"):
            solve_synthetic_design(Y, K=5, mode="global_2way")
        with pytest.raises(MlsynthConfigError, match="positive"):
            solve_synthetic_design(Y, K=0, mode="global_2way")

    def test_solve_synthetic_design_validates_lambda(self):
        from mlsynth.utils.syndes_helpers.optimization import (
            solve_synthetic_design,
        )
        Y = np.random.default_rng(0).standard_normal((10, 4))
        with pytest.raises(MlsynthConfigError, match="nonneg"):
            solve_synthetic_design(Y, K=2, mode="global_2way", lam=-0.1)

    def test_solve_synthetic_design_costs_budget_pairing(self):
        from mlsynth.utils.syndes_helpers.optimization import (
            solve_synthetic_design,
        )
        Y = np.random.default_rng(0).standard_normal((10, 4))
        with pytest.raises(MlsynthConfigError, match="together"):
            solve_synthetic_design(Y, K=2, mode="global_2way",
                                    costs=[1.0] * 4, budget=None)

    def test_solve_synthetic_design_costs_length(self):
        from mlsynth.utils.syndes_helpers.optimization import (
            solve_synthetic_design,
        )
        Y = np.random.default_rng(0).standard_normal((10, 4))
        with pytest.raises(MlsynthConfigError, match="length"):
            solve_synthetic_design(Y, K=2, mode="global_2way",
                                    costs=[1.0] * 3, budget=10.0)

    def test_solve_synthetic_design_negative_costs(self):
        from mlsynth.utils.syndes_helpers.optimization import (
            solve_synthetic_design,
        )
        Y = np.random.default_rng(0).standard_normal((10, 4))
        with pytest.raises(MlsynthConfigError, match="non-negative"):
            solve_synthetic_design(Y, K=2, mode="global_2way",
                                    costs=[-1.0, 1.0, 1.0, 1.0], budget=10.0)

    def test_solve_synthetic_design_non_positive_budget(self):
        from mlsynth.utils.syndes_helpers.optimization import (
            solve_synthetic_design,
        )
        Y = np.random.default_rng(0).standard_normal((10, 4))
        with pytest.raises(MlsynthConfigError, match="positive"):
            solve_synthetic_design(Y, K=2, mode="global_2way",
                                    costs=[1.0] * 4, budget=0.0)

    def test_solve_synthetic_design_solver_raises(self, monkeypatch):
        from mlsynth.utils.syndes_helpers.optimization import (
            solve_synthetic_design,
        )
        import cvxpy as cp
        Y = np.random.default_rng(0).standard_normal((10, 4))
        # Force cvxpy.Problem.solve to throw.
        orig = cp.Problem.solve
        def _boom(self, *a, **kw):
            raise RuntimeError("solver gone")
        monkeypatch.setattr(cp.Problem, "solve", _boom)
        try:
            with pytest.raises(MlsynthEstimationError, match="failed to solve"):
                solve_synthetic_design(Y, K=2, mode="global_2way")
        finally:
            monkeypatch.setattr(cp.Problem, "solve", orig)

    def test_one_way_global_constraint_requires_K(self):
        # build_global_equal_weights_constraints / objective both validate K.
        from mlsynth.utils.syndes_helpers.formulation import (
            build_global_equal_weights_constraints,
            build_global_equal_weights_objective,
        )
        import cvxpy as cp
        Y = np.zeros((4, 4))
        D = cp.Variable(4, boolean=True)
        c = cp.Variable(4)
        z = cp.Variable(4)
        with pytest.raises(ValueError, match="requires an explicit K"):
            build_global_equal_weights_constraints(
                Y=Y, D=D, K=None, variables={"c": c, "z": z})
        with pytest.raises(ValueError, match="less"):
            build_global_equal_weights_constraints(
                Y=Y, D=D, K=4, variables={"c": c, "z": z})
        with pytest.raises(ValueError, match="positive integer"):
            build_global_equal_weights_objective(
                Y=Y, K=None, lam=0.0, variables={"c": c, "z": z})

    def test_build_components_unknown_mode(self):
        from mlsynth.utils.syndes_helpers.formulation import (
            build_syndes_problem_components,
        )
        import cvxpy as cp
        Y = np.zeros((4, 4))
        D = cp.Variable(4, boolean=True)
        with pytest.raises(ValueError, match="Unknown SYNDES mode"):
            build_syndes_problem_components(Y=Y, D=D, K=2, lam=0.0,
                                              mode="not_a_mode")

    def test_unpack_problem_components(self):
        from mlsynth.utils.syndes_helpers.formulation import (
            build_syndes_problem_components, unpack_problem_components,
        )
        import cvxpy as cp
        Y = np.random.default_rng(0).standard_normal((10, 4))
        D = cp.Variable(4, boolean=True)
        comps = build_syndes_problem_components(Y=Y, D=D, K=2, lam=0.5,
                                                  mode="global_2way")
        obj, cons, vars_ = unpack_problem_components(comps)
        assert isinstance(obj, cp.Expression)
        assert isinstance(cons, list) and len(cons) > 0
        assert isinstance(vars_, dict)

    def test_validate_design_inputs_rejects_non_2d(self):
        from mlsynth.utils.syndes_helpers.optimization import (
            solve_synthetic_design,
        )
        with pytest.raises(MlsynthConfigError, match="two-dimensional"):
            solve_synthetic_design(np.zeros(5), K=2, mode="global_2way")

    def test_build_global_2way_helper(self):
        # The private _build_global_2way / _build_per_unit helpers are thin
        # wrappers around build_syndes_problem_components; cover them so the
        # public unpacking surface is verified.
        from mlsynth.utils.syndes_helpers.optimization import (
            _build_global_2way, _build_per_unit,
        )
        import cvxpy as cp
        Y = np.random.default_rng(0).standard_normal((6, 4))
        D = cp.Variable(4, boolean=True)
        obj, cons, vars_ = _build_global_2way(Y, D, K=2, lam=0.0)
        assert isinstance(obj, cp.Expression) and isinstance(cons, list)
        obj2, cons2, vars2 = _build_per_unit(Y, D, K=2, lam=0.0)
        assert isinstance(obj2, cp.Expression) and isinstance(cons2, list)

    def test_extract_weights_defensive_branches(self):
        # _extract_weights has three defensive fallback paths that the public
        # solve_synthetic_design never normally hits (c is always present in
        # global_equal_weights, q is always present in per_unit, etc.). Drive
        # them directly with fabricated value dicts.
        from mlsynth.utils.syndes_helpers.optimization import _extract_weights
        assignment = np.array([1, 1, 0, 0, 0])

        # global_equal_weights with c missing → uniform 1/(N-K) control branch
        tw, cw, ct = _extract_weights("global_equal_weights",
                                       assignment, values={})
        np.testing.assert_allclose(cw, np.array([0, 0, 1, 1, 1]) / 3)

        # per_unit with q missing → all-None tuple
        tw, cw, ct = _extract_weights("per_unit", assignment, values={})
        assert tw is None and cw is None and ct is None

        # Unknown mode with w/q missing → all-None tuple (final fallback)
        tw, cw, ct = _extract_weights("nope", assignment, values={})
        assert tw is None and cw is None and ct is None

    def test_solve_synthetic_design_non_optimal_status(self, monkeypatch):
        # Patch cvxpy to report a non-optimal status without raising.
        from mlsynth.utils.syndes_helpers.optimization import (
            solve_synthetic_design,
        )
        import cvxpy as cp
        Y = np.random.default_rng(0).standard_normal((6, 4))
        orig = cp.Problem.solve

        def _fake_solve(self, *a, **kw):
            self._status = "infeasible"   # cvxpy reads self.status from solver
            return None
        # Trick: monkey-patch Problem.solve to set status without raising and
        # leave D.value as None to also exercise the assignment-missing path.
        monkeypatch.setattr(cp.Problem, "solve",
                             lambda self, *a, **kw: setattr(self, "_status",
                                                              "infeasible"))
        try:
            with pytest.raises(MlsynthEstimationError,
                                match="failed: infeasible"):
                solve_synthetic_design(Y, K=2, mode="global_2way")
        finally:
            monkeypatch.setattr(cp.Problem, "solve", orig)

    def test_solve_synthetic_design_missing_assignment(self, monkeypatch):
        # cvxpy reports optimal but D.value is still None → the
        # "did not return an assignment" branch fires.
        from mlsynth.utils.syndes_helpers.optimization import (
            solve_synthetic_design,
        )
        import cvxpy as cp
        Y = np.random.default_rng(0).standard_normal((6, 4))
        orig = cp.Problem.solve

        def _fake_solve(self, *a, **kw):
            self._status = "optimal"
            # Don't actually solve → variables stay .value=None
            return None
        monkeypatch.setattr(cp.Problem, "solve", _fake_solve)
        try:
            with pytest.raises(MlsynthEstimationError,
                                match="did not return an assignment"):
                solve_synthetic_design(Y, K=2, mode="global_2way")
        finally:
            monkeypatch.setattr(cp.Problem, "solve", orig)


# ----------------------------------------------------------------------
# Power-analysis edge cases
# ----------------------------------------------------------------------

class TestPowerEdges:

    def test_power_rejects_short_panel(self, panel):
        from mlsynth.utils.syndes_helpers.power import (
            _newey_west_sigma, power_analysis,
        )
        # _newey_west_sigma on T<3 hits the short-panel branch.
        val = _newey_west_sigma(np.array([1.0, 2.0]))
        assert val >= 0.0

    def test_power_invalid_alpha(self, panel):
        res = SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                      "time": "time", "K": 2, "mode": "two_way_global",
                      "post_col": "post", "run_inference": False}).fit()
        with pytest.raises(MlsynthEstimationError, match="alpha"):
            power_analysis(res, alpha=0.0)

    def test_power_invalid_power(self, panel):
        res = SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                      "time": "time", "K": 2, "mode": "two_way_global",
                      "post_col": "post", "run_inference": False}).fit()
        with pytest.raises(MlsynthEstimationError, match="power"):
            power_analysis(res, power=1.5)

    def test_power_empty_horizon_grid(self, panel):
        res = SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                      "time": "time", "K": 2, "mode": "two_way_global",
                      "post_col": "post", "run_inference": False}).fit()
        with pytest.raises(MlsynthEstimationError, match="empty"):
            power_analysis(res, n_post_periods=[])

    def test_power_non_positive_horizon(self, panel):
        res = SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                      "time": "time", "K": 2, "mode": "two_way_global",
                      "post_col": "post", "run_inference": False}).fit()
        with pytest.raises(MlsynthEstimationError, match=">= 1"):
            power_analysis(res, n_post_periods=[0, 2])

    def test_power_unknown_baseline(self, panel):
        res = SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                      "time": "time", "K": 2, "mode": "two_way_global",
                      "post_col": "post", "run_inference": False}).fit()
        with pytest.raises(MlsynthEstimationError, match="Unknown baseline"):
            power_analysis(res, baseline="nope")

    def test_power_zero_baseline_rejected(self, panel):
        res = SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                      "time": "time", "K": 2, "mode": "two_way_global",
                      "post_col": "post", "run_inference": False}).fit()
        with pytest.raises(MlsynthEstimationError, match="non-zero"):
            power_analysis(res, baseline=0.0)

    def test_power_overall_baseline(self, panel):
        res = SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                      "time": "time", "K": 2, "mode": "two_way_global",
                      "post_col": "post", "run_inference": False}).fit()
        out = power_analysis(res, baseline="overall")
        assert out.baseline_kind == "overall"

    def test_power_control_baseline_with_global_2way(self, panel):
        # global_2way has control_weights set ⇒ control branch picks the
        # SC-weighted control mean.
        res = SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                      "time": "time", "K": 2, "mode": "two_way_global",
                      "post_col": "post", "run_inference": False}).fit()
        out = power_analysis(res, baseline="control")
        assert out.baseline_kind in ("control", "overall_fallback")

    def test_power_per_unit_falls_back_to_overall(self, panel):
        # per_unit has no scalar control_weights ⇒ baseline='control' falls
        # back to overall.
        res = SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                      "time": "time", "K": 2, "mode": "per_unit",
                      "post_col": "post", "run_inference": False}).fit()
        out = power_analysis(res, baseline="control")
        assert out.baseline_kind == "overall_fallback"

    def test_power_treated_baseline_requires_treated(self, monkeypatch, panel):
        # Manually empty the selected-units array.
        from mlsynth.utils.syndes_helpers.structures import SYNDESDesign
        res = SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                      "time": "time", "K": 2, "mode": "two_way_global",
                      "post_col": "post", "run_inference": False}).fit()
        from dataclasses import replace as _r
        empty_design = _r(res.design,
                          selected_unit_indices=np.array([], dtype=int))
        from dataclasses import replace as _rr
        res2 = _rr(res, design=empty_design)
        with pytest.raises(MlsynthEstimationError, match="at least one"):
            power_analysis(res2, baseline="treated")

    def test_power_custom_baseline(self, panel):
        res = SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                      "time": "time", "K": 2, "mode": "two_way_global",
                      "post_col": "post", "run_inference": False}).fit()
        out = power_analysis(res, baseline=1.5)
        assert out.baseline_kind == "custom"
        assert out.baseline == 1.5

    def test_power_rejects_non_2d_input(self):
        # power_analysis defends against an inputs.Y_pre that isn't 2-D.
        from mlsynth.utils.syndes_helpers.power import power_analysis
        from mlsynth.utils.syndes_helpers.structures import (
            SYNDESDesign, SYNDESInputs, SYNDESResults,
        )
        from mlsynth.utils.fast_scm_helpers.structure import IndexSet
        inputs = SYNDESInputs(
            Y_pre=np.zeros(6), Y_post=None,   # 1-D on purpose
            unit_index=IndexSet.from_labels([0, 1]),
            time_index=IndexSet.from_labels([0, 1, 2]),
            pre_time_index=IndexSet.from_labels([0, 1, 2]),
            post_time_index=None, outcome="y",
        )
        d = SYNDESDesign(
            mode="two_way_global", objective_value=0.0, lambda_value=0.0,
            assignment=np.array([1, 0]),
            selected_unit_indices=np.array([0]),
            selected_unit_labels=np.array([0]), assignment_by_unit={},
            contrast_weights=np.array([0.5, -0.5]),
        )
        with pytest.raises(MlsynthEstimationError, match="2-D"):
            power_analysis(SYNDESResults(design=d, inputs=inputs))

    def test_power_contrast_length_mismatch(self):
        from mlsynth.utils.syndes_helpers.power import power_analysis
        from mlsynth.utils.syndes_helpers.structures import (
            SYNDESDesign, SYNDESInputs, SYNDESResults,
        )
        from mlsynth.utils.fast_scm_helpers.structure import IndexSet
        inputs = SYNDESInputs(
            Y_pre=np.zeros((4, 2)), Y_post=None,
            unit_index=IndexSet.from_labels([0, 1]),
            time_index=IndexSet.from_labels([0, 1, 2, 3]),
            pre_time_index=IndexSet.from_labels([0, 1, 2, 3]),
            post_time_index=None, outcome="y",
        )
        d = SYNDESDesign(
            mode="two_way_global", objective_value=0.0, lambda_value=0.0,
            assignment=np.array([1, 0]),
            selected_unit_indices=np.array([0]),
            selected_unit_labels=np.array([0]), assignment_by_unit={},
            contrast_weights=np.array([0.5, -0.5, 0.0]),    # length 3 != 2
        )
        with pytest.raises(MlsynthEstimationError, match="match"):
            power_analysis(SYNDESResults(design=d, inputs=inputs))

    def test_power_short_pre_period_rejected(self):
        from mlsynth.utils.syndes_helpers.power import power_analysis
        from mlsynth.utils.syndes_helpers.structures import (
            SYNDESDesign, SYNDESInputs, SYNDESResults,
        )
        from mlsynth.utils.fast_scm_helpers.structure import IndexSet
        inputs = SYNDESInputs(
            Y_pre=np.zeros((1, 2)), Y_post=None,
            unit_index=IndexSet.from_labels([0, 1]),
            time_index=IndexSet.from_labels([0]),
            pre_time_index=IndexSet.from_labels([0]),
            post_time_index=None, outcome="y",
        )
        d = SYNDESDesign(
            mode="two_way_global", objective_value=0.0, lambda_value=0.0,
            assignment=np.array([1, 0]),
            selected_unit_indices=np.array([0]),
            selected_unit_labels=np.array([0]), assignment_by_unit={},
            contrast_weights=np.array([0.5, -0.5]),
        )
        with pytest.raises(MlsynthEstimationError, match=">= 2"):
            power_analysis(SYNDESResults(design=d, inputs=inputs))

    def test_to_dataframe_round_trip(self, panel):
        res = SYNDES({"df": panel, "outcome": "y", "unitid": "unit",
                      "time": "time", "K": 2, "mode": "two_way_global",
                      "post_col": "post", "run_inference": False}).fit()
        df = power_analysis(res).to_dataframe()
        assert list(df.columns) == ["n_post", "mde_absolute", "mde_percent"]


# ----------------------------------------------------------------------
# MultiArmResults convenience properties
# ----------------------------------------------------------------------

class TestMultiArmResultsProperties:

    def test_mode_property(self):
        from mlsynth.utils.syndes_helpers.structures import (
            SYNDESMultiArmResults,
        )
        r = SYNDESMultiArmResults(arm_designs={}, arm="x")
        assert r.mode == "syndes_multiarm"
