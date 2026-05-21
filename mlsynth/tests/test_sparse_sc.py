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
) -> pd.DataFrame:
    """Linear-factor panel with covariate columns embedded in long form."""
    rng = np.random.default_rng(seed)
    n_units = N_donors + 1
    F = rng.standard_normal((T, P))
    Lambda = rng.standard_normal((n_units, P))
    Y = F @ Lambda.T + noise * rng.standard_normal((T, n_units))
    Y[T0:, 0] += true_effect

    records = []
    for u in range(n_units):
        name = f"unit_{u}"
        # Unit-level covariates p0..p{P-1} are constant across time.
        covs = {f"p{p}": float(Lambda[u, p]) for p in range(P)}
        for t in range(T):
            row = {
                "unit": name, "year": 2000 + t,
                "y": float(Y[t, u]),
                "tr": int(u == 0 and t >= T0),
            }
            row.update(covs)
            records.append(row)
    return pd.DataFrame(records)


@pytest.fixture(scope="module")
def small_panel() -> pd.DataFrame:
    return _factor_panel()


COVS = ["p0", "p1", "p2", "p3"]


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
# Layer 2: setup (long-form covariate plumbing)
# ---------------------------------------------------------------------------

class TestSetup:
    def test_covariates_collapsed_to_unit_means(self, small_panel):
        inputs = prepare_sparse_sc_inputs(
            df=small_panel, outcome="y", treat="tr",
            unitid="unit", time="year",
            covariates=COVS,
        )
        assert inputs.Y0.shape == (20, 6)
        assert inputs.Y1.shape == (20,)
        assert inputs.X0.shape == (4, 6)
        assert inputs.X1.shape == (4,)
        assert inputs.T0_total == 14
        assert 1 < inputs.T0_train < inputs.T0_total
        assert list(inputs.predictor_names) == COVS

    def test_outcome_lag_periods_add_predictor_rows(self, small_panel):
        inputs = prepare_sparse_sc_inputs(
            df=small_panel, outcome="y", treat="tr",
            unitid="unit", time="year",
            covariates=COVS,
            outcome_lag_periods=[2003, 2007],
        )
        assert inputs.X0.shape[0] == 6  # 4 covs + 2 lagged outcomes
        assert inputs.predictor_names[-2:] == ["y@2003", "y@2007"]

    def test_requires_at_least_one_predictor(self, small_panel):
        with pytest.raises(MlsynthDataError):
            prepare_sparse_sc_inputs(
                df=small_panel, outcome="y", treat="tr",
                unitid="unit", time="year",
            )

    def test_unknown_covariate_rejected(self, small_panel):
        with pytest.raises(MlsynthDataError):
            prepare_sparse_sc_inputs(
                df=small_panel, outcome="y", treat="tr",
                unitid="unit", time="year",
                covariates=["does_not_exist"],
            )

    def test_post_period_lag_rejected(self, small_panel):
        # 2017 is post-treatment (T0=14 -> last pre-year is 2013).
        with pytest.raises(MlsynthDataError):
            prepare_sparse_sc_inputs(
                df=small_panel, outcome="y", treat="tr",
                unitid="unit", time="year",
                covariates=COVS, outcome_lag_periods=[2017],
            )

    def test_T0_train_validated(self, small_panel):
        with pytest.raises(MlsynthDataError):
            prepare_sparse_sc_inputs(
                df=small_panel, outcome="y", treat="tr",
                unitid="unit", time="year",
                covariates=COVS, T0_train=0,
            )

    def test_short_pre_period_rejected(self, small_panel):
        df2 = small_panel.copy()
        df2["tr"] = ((df2["unit"] == "unit_0") & (df2["year"] >= 2003)).astype(int)
        with pytest.raises(MlsynthDataError):
            prepare_sparse_sc_inputs(
                df=df2, outcome="y", treat="tr",
                unitid="unit", time="year",
                covariates=COVS,
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
        assert (np.diff(grid[1:]) > 0).all()

    def test_default_v20_anchors_to_first_predictor(self):
        rng = np.random.default_rng(0)
        X0 = rng.standard_normal((4, 6))
        v20 = default_v20(X0)
        assert v20.shape == (3,)
        assert (v20 > 0).all()

    def test_sweep_lambda_returns_consistent_shapes(self, small_panel):
        inputs = prepare_sparse_sc_inputs(
            df=small_panel, outcome="y", treat="tr",
            unitid="unit", time="year",
            covariates=COVS,
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
        res = SparseSC({
            "df": small_panel, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "covariates": COVS,
            "lambda_grid": [0.0, 0.001, 0.01, 0.1],
            "run_inference": False, "display_graphs": False,
        }).fit()
        assert res.att == pytest.approx(-2.5, abs=1.0)

    def test_w_sums_to_one(self, small_panel):
        res = SparseSC({
            "df": small_panel, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "covariates": COVS,
            "lambda_grid": [0.0, 0.01], "run_inference": False,
            "display_graphs": False,
        }).fit()
        assert res.design.w.sum() == pytest.approx(1.0, abs=1e-5)
        assert (res.design.w >= -1e-8).all()

    def test_v_path_first_column_pinned(self, small_panel):
        res = SparseSC({
            "df": small_panel, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "covariates": COVS,
            "lambda_grid": [0.0, 0.01, 0.1],
            "run_inference": False, "display_graphs": False,
        }).fit()
        assert np.all(res.design.v_path[:, 0] == 1.0)

    def test_large_lambda_zeroes_v_weights(self, small_panel):
        res = SparseSC({
            "df": small_panel, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "covariates": COVS,
            "lambda_grid": [1e3],
            "run_inference": False, "display_graphs": False,
        }).fit()
        assert np.allclose(res.design.v[1:], 0.0, atol=1e-3)

    def test_outcome_lag_predictors_work_end_to_end(self, small_panel):
        # 2008, 2010, 2012 are pre-treatment.
        res = SparseSC({
            "df": small_panel, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "covariates": COVS,
            "outcome_lag_periods": [2008, 2010, 2012],
            "lambda_grid": [0.0, 0.01], "run_inference": False,
            "display_graphs": False,
        }).fit()
        assert res.design.v.shape == (7,)
        assert res.design.v[0] == 1.0


class TestPlacebo:
    def test_placebo_disabled(self, small_panel):
        res = SparseSC({
            "df": small_panel, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "covariates": COVS,
            "lambda_grid": [0.01], "run_inference": False,
            "display_graphs": False,
        }).fit()
        assert res.inference.method == "none"
        assert res.inference.n_placebo == 0
        assert np.isnan(res.inference.p_value)

    def test_placebo_yields_valid_p_value(self, small_panel):
        res = SparseSC({
            "df": small_panel, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "covariates": COVS,
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
        res = SparseSC({
            "df": small_panel, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "covariates": COVS,
            "lambda_grid": [0.01], "run_inference": False,
            "display_graphs": False,
        }).fit()
        assert isinstance(res, SparseSCResults)
        assert isinstance(res.inputs, SparseSCInputs)
        assert isinstance(res.design, SparseSCDesign)
        assert isinstance(res.inference, SparseSCInference)

    def test_dict_vs_config_object(self, small_panel):
        cfg_dict = {
            "df": small_panel, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "covariates": COVS,
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
        res = SparseSC({
            "df": small_panel, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "covariates": COVS,
            "lambda_grid": [0.01], "run_inference": False,
            "display_graphs": False,
        }).fit()
        assert set(res.donor_weights.keys()) == set(
            str(n) for n in res.inputs.donor_names
        )
        assert set(res.predictor_weights.keys()) == set(
            res.inputs.predictor_names
        )
        # First predictor (the anchor) gets v = 1.
        assert res.predictor_weights[res.inputs.predictor_names[0]] == pytest.approx(1.0)
