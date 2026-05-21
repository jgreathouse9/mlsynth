"""Tests for the SparseSC estimator.

Python port of Vives-i-Bastida's MATLAB ``sparse_synth.m``: an
L1-penalized predictor-weighting variant of canonical SCM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import SparseSC
from mlsynth.config_models import SparseSCConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.sparse_sc_helpers.inner import solve_w
from mlsynth.utils.sparse_sc_helpers.objective import training_loss, validation_mse
from mlsynth.utils.sparse_sc_helpers.optimization import (
    default_lambda_grid,
    default_v20,
    recover_w,
    sweep_lambda,
)
from mlsynth.utils.sparse_sc_helpers.setup import prepare_sparse_sc_inputs
from mlsynth.utils.sparse_sc_helpers.structures import (
    SparseSCDesign,
    SparseSCInference,
    SparseSCInputs,
    SparseSCResults,
)


def _factor_panel(
    *, seed: int = 0, N_donors: int = 6, T: int = 20, T0: int = 14,
    P: int = 4, true_effect: float = -2.5, noise: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Linear-factor panel + matched predictor table."""
    rng = np.random.default_rng(seed)
    n_units = N_donors + 1
    F = rng.standard_normal((T, P))
    Lambda = rng.standard_normal((n_units, P))
    Y = F @ Lambda.T + noise * rng.standard_normal((T, n_units))
    # Treated unit is unit 0; effect from T0 onward.
    Y[T0:, 0] += true_effect

    records = []
    for u in range(n_units):
        name = f"unit_{u}"
        for t in range(T):
            records.append({
                "unit": name, "year": 2000 + t,
                "y": float(Y[t, u]),
                "tr": int(u == 0 and t >= T0),
            })
    df = pd.DataFrame(records)

    # Predictor table: pre-treatment factor loadings as observable proxies.
    pred = pd.DataFrame({
        "unit": [f"unit_{u}" for u in range(n_units)],
        **{f"p{p}": Lambda[:, p] for p in range(P)},
    })
    return df, pred


@pytest.fixture(scope="module")
def small_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _factor_panel()


# ---------------------------------------------------------------------------
# Layer 1: inner QP + objectives
# ---------------------------------------------------------------------------

class TestInnerQP:
    def test_solve_w_on_simplex(self):
        rng = np.random.default_rng(0)
        X0 = rng.standard_normal((4, 5))
        X1 = X0[:, 0] + 0.01 * rng.standard_normal(4)
        v = np.ones(4)
        w = solve_w(v, X1, X0)
        assert w.shape == (5,)
        assert w.sum() == pytest.approx(1.0, abs=1e-5)
        assert (w >= -1e-8).all()

    def test_solve_w_recovers_replicated_donor(self):
        # If X1 exactly equals a donor column, w should put most mass on it.
        rng = np.random.default_rng(1)
        X0 = rng.standard_normal((3, 4))
        X1 = X0[:, 2].copy()
        w = solve_w(np.ones(3), X1, X0)
        assert w[2] > 0.9

    def test_training_loss_includes_penalty(self):
        rng = np.random.default_rng(2)
        X0 = rng.standard_normal((3, 4))
        X1 = rng.standard_normal(3)
        Z0 = rng.standard_normal((5, 4))
        Z1 = rng.standard_normal(5)
        v2 = np.array([1.0, 1.0])
        base = training_loss(v2, X1, X0, Z1, Z0, lam=0.0)
        pen = training_loss(v2, X1, X0, Z1, Z0, lam=0.5)
        # L1 penalty on v = [1, 1, 1] adds 0.5 * 3 = 1.5
        assert pen == pytest.approx(base + 1.5, abs=1e-6)

    def test_validation_mse_nonnegative(self):
        rng = np.random.default_rng(3)
        X0 = rng.standard_normal((3, 4))
        X1 = rng.standard_normal(3)
        Z0 = rng.standard_normal((5, 4))
        Z1 = rng.standard_normal(5)
        assert validation_mse(np.ones(2), X1, X0, Z1, Z0) >= 0


# ---------------------------------------------------------------------------
# Layer 2: setup
# ---------------------------------------------------------------------------

class TestSetup:
    def test_basic_shapes(self, small_panel):
        df, pred = small_panel
        inputs = prepare_sparse_sc_inputs(
            df=df, outcome="y", treat="tr",
            unitid="unit", time="year",
            predictors_df=pred, predictors_unitid="unit",
        )
        assert inputs.Y0.shape == (20, 6)
        assert inputs.Y1.shape == (20,)
        assert inputs.X0.shape[1] == 6  # N donors
        assert inputs.X1.shape[0] == inputs.X0.shape[0]
        assert inputs.T0_total == 14
        assert 1 < inputs.T0_train < inputs.T0_total

    def test_explicit_predictor_subset(self, small_panel):
        df, pred = small_panel
        inputs = prepare_sparse_sc_inputs(
            df=df, outcome="y", treat="tr",
            unitid="unit", time="year",
            predictors_df=pred, predictors_unitid="unit",
            predictor_cols=["p0", "p1"],
        )
        assert inputs.P == 2
        assert list(inputs.predictor_names) == ["p0", "p1"]

    def test_missing_predictor_unit_rejected(self, small_panel):
        df, pred = small_panel
        truncated = pred[pred["unit"] != "unit_2"]
        with pytest.raises(MlsynthDataError):
            prepare_sparse_sc_inputs(
                df=df, outcome="y", treat="tr",
                unitid="unit", time="year",
                predictors_df=truncated, predictors_unitid="unit",
            )

    def test_missing_predictor_column_rejected(self, small_panel):
        df, pred = small_panel
        with pytest.raises(MlsynthDataError):
            prepare_sparse_sc_inputs(
                df=df, outcome="y", treat="tr",
                unitid="unit", time="year",
                predictors_df=pred, predictors_unitid="unit",
                predictor_cols=["does_not_exist"],
            )

    def test_T0_train_validated(self, small_panel):
        df, pred = small_panel
        with pytest.raises(MlsynthDataError):
            prepare_sparse_sc_inputs(
                df=df, outcome="y", treat="tr",
                unitid="unit", time="year",
                predictors_df=pred, predictors_unitid="unit",
                T0_train=0,
            )

    def test_short_pre_period_rejected(self, small_panel):
        df, _ = small_panel
        # Move treatment up so pre-period < 4.
        df2 = df.copy()
        df2["tr"] = ((df2["unit"] == "unit_0") & (df2["year"] >= 2003)).astype(int)
        with pytest.raises(MlsynthDataError):
            prepare_sparse_sc_inputs(
                df=df2, outcome="y", treat="tr",
                unitid="unit", time="year",
                predictors_df=_factor_panel()[1],
                predictors_unitid="unit",
            )


# ---------------------------------------------------------------------------
# Layer 3: optimization + integration
# ---------------------------------------------------------------------------

class TestOptimization:
    def test_default_lambda_grid_starts_at_zero(self):
        grid = default_lambda_grid(size=11)
        assert grid.size == 11
        assert grid[0] == 0.0
        assert grid[-1] == pytest.approx(1.0, abs=1e-9)
        assert (np.diff(grid[1:]) > 0).all()  # log-spaced strictly increasing

    def test_default_v20_anchors_to_first_predictor(self):
        rng = np.random.default_rng(0)
        X0 = rng.standard_normal((4, 6))
        v20 = default_v20(X0)
        assert v20.shape == (3,)
        assert (v20 > 0).all()

    def test_sweep_lambda_returns_consistent_shapes(self, small_panel):
        df, pred = small_panel
        inputs = prepare_sparse_sc_inputs(
            df=df, outcome="y", treat="tr",
            unitid="unit", time="year",
            predictors_df=pred, predictors_unitid="unit",
        )
        grid = np.array([0.0, 0.01, 0.1])
        optv, opt_lam, grid_used, train, val, v_path = sweep_lambda(
            X1=inputs.X1, X0=inputs.X0,
            Y1=inputs.Y1, Y0=inputs.Y0,
            T0_total=inputs.T0_total, T0_train=inputs.T0_train,
            lambda_grid=grid,
        )
        assert optv.shape == (inputs.P,)
        assert optv[0] == 1.0
        assert opt_lam in grid_used
        assert train.shape == val.shape == grid.shape
        assert v_path.shape == (3, inputs.P)
        assert np.all(v_path[:, 0] == 1.0)


class TestSyntheticRecovery:
    def test_recovers_true_effect(self, small_panel):
        df, pred = small_panel
        res = SparseSC({
            "df": df, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "predictors_df": pred, "predictors_unitid": "unit",
            "lambda_grid": [0.0, 0.001, 0.01, 0.1],
            "run_inference": False, "display_graphs": False,
        }).fit()
        assert res.att == pytest.approx(-2.5, abs=1.0)

    def test_w_sums_to_one(self, small_panel):
        df, pred = small_panel
        res = SparseSC({
            "df": df, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "predictors_df": pred, "predictors_unitid": "unit",
            "lambda_grid": [0.0, 0.01], "run_inference": False,
            "display_graphs": False,
        }).fit()
        assert res.design.w.sum() == pytest.approx(1.0, abs=1e-5)
        assert (res.design.w >= -1e-8).all()

    def test_v_path_first_column_pinned(self, small_panel):
        df, pred = small_panel
        res = SparseSC({
            "df": df, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "predictors_df": pred, "predictors_unitid": "unit",
            "lambda_grid": [0.0, 0.01, 0.1],
            "run_inference": False, "display_graphs": False,
        }).fit()
        assert np.all(res.design.v_path[:, 0] == 1.0)

    def test_large_lambda_zeroes_v_weights(self, small_panel):
        df, pred = small_panel
        res = SparseSC({
            "df": df, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "predictors_df": pred, "predictors_unitid": "unit",
            "lambda_grid": [1e3],  # huge penalty
            "run_inference": False, "display_graphs": False,
        }).fit()
        # All non-anchored V-weights driven to 0 at very large lambda.
        assert np.allclose(res.design.v[1:], 0.0, atol=1e-3)


class TestPlacebo:
    def test_placebo_disabled_by_default_in_test(self, small_panel):
        df, pred = small_panel
        res = SparseSC({
            "df": df, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "predictors_df": pred, "predictors_unitid": "unit",
            "lambda_grid": [0.01], "run_inference": False,
            "display_graphs": False,
        }).fit()
        assert res.inference.method == "none"
        assert res.inference.n_placebo == 0
        assert np.isnan(res.inference.p_value)

    def test_placebo_yields_valid_p_value(self, small_panel):
        df, pred = small_panel
        res = SparseSC({
            "df": df, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "predictors_df": pred, "predictors_unitid": "unit",
            "lambda_grid": [0.01], "run_inference": True,
            "n_placebo": 4, "display_graphs": False, "seed": 7,
        }).fit()
        assert res.inference.method == "abadie_placebo_permutation"
        assert 0.0 < res.inference.p_value <= 1.0
        assert res.inference.n_placebo > 0
        assert res.inference.placebo_atts.size == res.inference.n_placebo


# ---------------------------------------------------------------------------
# Layer 4: public API
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_import(self):
        from mlsynth import SparseSC as Imported  # noqa: F401
        assert Imported is SparseSC

    def test_results_object_types(self, small_panel):
        df, pred = small_panel
        res = SparseSC({
            "df": df, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "predictors_df": pred, "predictors_unitid": "unit",
            "lambda_grid": [0.01], "run_inference": False,
            "display_graphs": False,
        }).fit()
        assert isinstance(res, SparseSCResults)
        assert isinstance(res.inputs, SparseSCInputs)
        assert isinstance(res.design, SparseSCDesign)
        assert isinstance(res.inference, SparseSCInference)

    def test_dict_vs_config_object(self, small_panel):
        df, pred = small_panel
        cfg_dict = {
            "df": df, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "predictors_df": pred, "predictors_unitid": "unit",
            "lambda_grid": [0.01], "run_inference": False,
            "display_graphs": False,
        }
        cfg_obj = SparseSCConfig(**cfg_dict)
        r1 = SparseSC(cfg_dict).fit()
        r2 = SparseSC(cfg_obj).fit()
        assert r1.att == pytest.approx(r2.att)

    def test_invalid_config_raises(self):
        with pytest.raises(MlsynthConfigError):
            SparseSC({"df": "not a dataframe"})

    def test_donor_and_predictor_weights_aligned(self, small_panel):
        df, pred = small_panel
        res = SparseSC({
            "df": df, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "predictors_df": pred, "predictors_unitid": "unit",
            "lambda_grid": [0.01], "run_inference": False,
            "display_graphs": False,
        }).fit()
        assert set(res.donor_weights.keys()) == set(
            str(n) for n in res.inputs.donor_names
        )
        assert set(res.predictor_weights.keys()) == set(
            res.inputs.predictor_names
        )
        assert res.predictor_weights[res.inputs.predictor_names[0]] == pytest.approx(1.0)
