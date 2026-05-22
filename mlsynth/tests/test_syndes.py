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

from mlsynth import SYNDES, SCDI, power_analysis
from mlsynth.config_models import SYNDESConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError
from mlsynth.estimators.syndes import _MODE_FROM_INTERNAL, _MODE_TO_INTERNAL
from mlsynth.utils.scdi_helpers.inference import _build_contrast_vector
from mlsynth.utils.scdi_helpers.optimization import solve_synthetic_design
from mlsynth.utils.scdi_helpers.power import SYNDESPower
from mlsynth.utils.scdi_helpers.structures import SCDIResults


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

    def test_T0_or_post_col_required(self, panel):
        with pytest.raises(MlsynthConfigError):
            SYNDESConfig(
                df=panel, outcome="y", unitid="unit", time="time",
                K=2, mode="two_way_global",
            )


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
        assert isinstance(res, SCDIResults)
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

    def test_scdi_alias_still_works(self, panel):
        # Legacy SCDI must keep working with its old mode names.
        res = SCDI({
            "df": panel, "outcome": "y", "unitid": "unit", "time": "time",
            "K": 2, "mode": "global_2way", "post_col": "post",
            "run_inference": False,
        }).fit()
        assert res.design.mode == "global_2way"

    def test_dict_config_accepted(self, panel):
        res = SYNDES({
            "df": panel, "outcome": "y", "unitid": "unit", "time": "time",
            "K": 2, "mode": "two_way_global", "post_col": "post",
            "run_inference": False,
        }).fit()
        assert isinstance(res, SCDIResults)


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
