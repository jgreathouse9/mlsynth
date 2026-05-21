"""Comprehensive tests for the SCDI estimator and its helper subpackage.

Covers:
    * SCDIConfig validation paths.
    * prepare_scdi_inputs (data prep / panel pivoting).
    * The MIP solver paths via ``solve_synthetic_design`` (all three modes).
    * The relaxed simulated-annealing pipeline:
        - relaxed_initialization (validate, default lambda, init_assignment)
        - relaxed_formulation (energy, RMSE, paths, weight extraction)
        - relaxed_annealing (temperature schedule, propose_swap, d-step)
        - relaxed_solver (solve_two_way_relaxed)
    * Permutation inference for both global and relaxed designs.
    * The SCDI estimator class as a whole.
    * Plotting helpers (smoke).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from mlsynth import SCDI
from mlsynth.config_models import SCDIConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.scdi_helpers.structures import (
    SCDIDesign,
    SCDIInference,
    SCDIInputs,
    SCDIResults,
)
from mlsynth.utils.scdi_helpers.relaxed_structures import (
    RelaxedAnnealingTrace,
    RelaxedDesign,
    RelaxedInference,
    RelaxedSolverResults,
    RelaxedSwapLog,
)
from mlsynth.utils.scdi_helpers.relaxed_formulation import (
    compute_energy,
    compute_rmse_gap,
    extract_weights,
    solve_weights_global,
    synthetic_paths,
)
from mlsynth.utils.scdi_helpers.relaxed_initialization import (
    default_lambda,
    init_assignment,
    validate_relaxed_inputs,
)
from mlsynth.utils.scdi_helpers.relaxed_annealing import (
    d_step_annealed,
    propose_swap,
    temperature_schedule,
)
from mlsynth.utils.scdi_helpers.relaxed_solver import solve_two_way_relaxed
from mlsynth.utils.scdi_helpers.inference import (
    permutation_test_global,
    permutation_test_relaxed_global,
)
from mlsynth.utils.scdi_helpers.optimization import (
    estimate_lambda,
    solve_synthetic_design,
)
from mlsynth.utils.scdi_helpers.plotter import plot_scdi_design
from mlsynth.utils.scdi_helpers.setup import prepare_scdi_inputs


# =========================================================================
# FIXTURES
# =========================================================================

def _make_panel(n_units=8, T=20, T_post=5, L=2, sigma=0.3, seed=0):
    rng = np.random.default_rng(seed)
    gamma = rng.standard_normal((n_units, L))
    nu = rng.standard_normal((T, L))
    Y = nu @ gamma.T + sigma * rng.standard_normal((T, n_units))
    rows = []
    for i in range(n_units):
        for t in range(T):
            rows.append({
                "unitid": f"u{i:02d}",
                "time": t,
                "y": Y[t, i],
                "post": int(t >= T - T_post),
            })
    return pd.DataFrame(rows), Y


@pytest.fixture
def panel_df():
    df, _ = _make_panel()
    return df


@pytest.fixture
def panel_no_post():
    df, _ = _make_panel(T=15, T_post=0)
    return df.drop(columns=["post"])


@pytest.fixture
def Y_small():
    rng = np.random.default_rng(0)
    return rng.normal(size=(20, 6))


@pytest.fixture
def inputs(panel_df):
    return prepare_scdi_inputs(panel_df, "y", "unitid", "time",
                                post_col="post")


# =========================================================================
# CONFIG VALIDATION
# =========================================================================

class TestSCDIConfig:

    def test_valid_dict_accepted(self, panel_df):
        cfg = SCDIConfig(df=panel_df, outcome="y", unitid="unitid",
                          time="time", K=3, post_col="post")
        assert cfg.K == 3
        assert cfg.mode == "global_2way"
        assert cfg.alpha == 0.10
        assert cfg.run_inference is True

    def test_K_must_be_positive(self, panel_df):
        with pytest.raises(Exception):
            SCDIConfig(df=panel_df, outcome="y", unitid="unitid",
                        time="time", K=0)

    def test_K_must_be_less_than_n_units(self, panel_df):
        with pytest.raises(MlsynthConfigError):
            SCDIConfig(df=panel_df, outcome="y", unitid="unitid",
                        time="time", K=8)   # exactly the unit count

    def test_K_too_large_rejected(self, panel_df):
        with pytest.raises(MlsynthConfigError):
            SCDIConfig(df=panel_df, outcome="y", unitid="unitid",
                        time="time", K=100)

    def test_post_col_must_exist(self, panel_df):
        with pytest.raises(MlsynthConfigError):
            SCDIConfig(df=panel_df, outcome="y", unitid="unitid",
                        time="time", K=3, post_col="not_a_column")

    def test_T0_too_large_rejected(self, panel_df):
        with pytest.raises(MlsynthConfigError):
            SCDIConfig(df=panel_df, outcome="y", unitid="unitid",
                        time="time", K=3, T0=10_000)

    def test_invalid_mode_rejected(self, panel_df):
        with pytest.raises(Exception):
            SCDIConfig(df=panel_df, outcome="y", unitid="unitid",
                        time="time", K=3, mode="nonexistent_mode")

    def test_dict_to_constructor_invalid_wrapped(self, panel_df):
        with pytest.raises(MlsynthConfigError, match="Invalid SCDI configuration"):
            SCDI({"df": panel_df, "outcome": "y", "unitid": "unitid",
                   "time": "time", "K": 0})


# =========================================================================
# DATA PREP
# =========================================================================

class TestPrepareScdiInputs:

    def test_basic_post_col(self, panel_df):
        inp = prepare_scdi_inputs(panel_df, "y", "unitid", "time",
                                    post_col="post")
        assert inp.Y_pre.shape == (15, 8)
        assert inp.Y_post.shape == (5, 8)
        assert inp.outcome == "y"

    def test_T0_split(self, panel_df):
        inp = prepare_scdi_inputs(panel_df, "y", "unitid", "time", T0=10)
        assert inp.Y_pre.shape == (10, 8)
        assert inp.Y_post.shape == (10, 8)

    def test_no_post_returns_none(self, panel_no_post):
        inp = prepare_scdi_inputs(panel_no_post, "y", "unitid", "time")
        assert inp.Y_post is None

    def test_unbalanced_panel_rejected(self, panel_df):
        partial = panel_df.iloc[:-1].copy()   # drop one observation
        with pytest.raises(MlsynthDataError):
            prepare_scdi_inputs(partial, "y", "unitid", "time",
                                 post_col="post")

    def test_invalid_post_col_rejected(self, panel_df):
        with pytest.raises(MlsynthDataError):
            prepare_scdi_inputs(panel_df, "y", "unitid", "time",
                                 post_col="nope")

    def test_T0_zero_rejected(self, panel_df):
        with pytest.raises(MlsynthConfigError):
            prepare_scdi_inputs(panel_df, "y", "unitid", "time", T0=0)

    def test_all_post_rejected(self, panel_df):
        df = panel_df.copy()
        df["post"] = 1
        with pytest.raises(MlsynthConfigError):
            prepare_scdi_inputs(df, "y", "unitid", "time", post_col="post")

    def test_T0_at_max_means_no_post(self, panel_df):
        n_periods = panel_df["time"].nunique()
        inp = prepare_scdi_inputs(panel_df, "y", "unitid", "time", T0=n_periods)
        assert inp.Y_post is None


# =========================================================================
# OPTIMIZATION: estimate_lambda + _validate_design_inputs
# =========================================================================

class TestEstimateLambda:

    def test_returns_positive_on_random_data(self, Y_small):
        lam = estimate_lambda(Y_small)
        assert lam > 0

    def test_zero_variance_zero_lambda(self):
        Y = np.ones((10, 5))
        assert estimate_lambda(Y) == 0.0

    def test_rejects_1d(self):
        with pytest.raises(MlsynthDataError):
            estimate_lambda(np.zeros(5))

    def test_rejects_one_period(self):
        with pytest.raises(MlsynthConfigError):
            estimate_lambda(np.zeros((1, 5)))


# =========================================================================
# OPTIMIZATION: solve_synthetic_design (MIP modes)
# =========================================================================

class TestSolveSyntheticDesign:

    def test_global_2way_smoke(self, inputs):
        d = solve_synthetic_design(
            Y=inputs.Y_pre, K=2, mode="global_2way",
            unit_index=inputs.unit_index,
        )
        assert isinstance(d, SCDIDesign)
        assert d.mode == "global_2way"
        assert d.n_treated == 2 if hasattr(d, "n_treated") else d.assignment.sum() == 2

    def test_global_equal_weights_smoke(self, inputs):
        d = solve_synthetic_design(
            Y=inputs.Y_pre, K=3, mode="global_equal_weights",
            unit_index=inputs.unit_index,
        )
        assert d.mode == "global_equal_weights"
        # By construction: K treated each at 1/K
        treated_idx = d.selected_unit_indices
        assert np.allclose(d.treated_weights[treated_idx], 1.0 / 3)
        # Non-treated entries are 0 on the treated side
        assert np.all(d.treated_weights.sum() == pytest.approx(1.0))

    def test_per_unit_smoke(self, inputs):
        d = solve_synthetic_design(
            Y=inputs.Y_pre, K=2, mode="per_unit",
            unit_index=inputs.unit_index,
        )
        assert d.mode == "per_unit"
        assert d.q is not None

    def test_K_too_large(self, Y_small):
        with pytest.raises(MlsynthConfigError):
            solve_synthetic_design(Y_small, K=100, mode="global_2way")

    def test_negative_lambda_rejected(self, Y_small):
        with pytest.raises(MlsynthConfigError):
            solve_synthetic_design(Y_small, K=2, lam=-0.5, mode="global_2way")

    def test_user_lambda_respected(self, inputs):
        d = solve_synthetic_design(
            Y=inputs.Y_pre, K=2, mode="global_2way", lam=0.5,
            unit_index=inputs.unit_index,
        )
        assert d.lambda_value == 0.5

    def test_selected_unit_labels_are_inputs_labels(self, inputs):
        d = solve_synthetic_design(
            Y=inputs.Y_pre, K=2, mode="global_2way",
            unit_index=inputs.unit_index,
        )
        for lbl in d.selected_unit_labels:
            assert lbl in inputs.unit_index.labels


# =========================================================================
# RELAXED INITIALIZATION
# =========================================================================

class TestRelaxedInitialization:

    def test_default_lambda_positive(self, Y_small):
        assert default_lambda(Y_small) > 0

    def test_default_lambda_zero_for_constant(self):
        Y = np.ones((10, 5))
        assert default_lambda(Y) == 0.0

    def test_init_assignment_picks_K(self, Y_small):
        D = init_assignment(Y_small, K=2)
        assert D.shape == (6,)
        assert D.sum() == 2
        assert set(np.unique(D).tolist()) <= {0, 1}

    def test_init_assignment_deterministic(self, Y_small):
        D1 = init_assignment(Y_small, K=2)
        D2 = init_assignment(Y_small, K=2)
        np.testing.assert_array_equal(D1, D2)

    def test_validate_relaxed_rejects_1d(self):
        with pytest.raises(MlsynthDataError):
            validate_relaxed_inputs(np.zeros(5), K=1)

    def test_validate_relaxed_rejects_one_period(self):
        with pytest.raises(MlsynthConfigError):
            validate_relaxed_inputs(np.zeros((1, 5)), K=1)

    def test_validate_relaxed_rejects_nonpositive_K(self, Y_small):
        with pytest.raises(MlsynthConfigError):
            validate_relaxed_inputs(Y_small, K=0)

    def test_validate_relaxed_rejects_K_too_large(self, Y_small):
        with pytest.raises(MlsynthConfigError):
            validate_relaxed_inputs(Y_small, K=10)

    def test_validate_relaxed_accepts_valid(self, Y_small):
        validate_relaxed_inputs(Y_small, K=3)   # should not raise


# =========================================================================
# RELAXED FORMULATION
# =========================================================================

class TestRelaxedFormulation:

    def test_solve_weights_global_basic(self, Y_small):
        D = np.array([1, 1, 0, 0, 0, 0])
        w = solve_weights_global(Y_small, D, lam=0.0)
        treated_idx = np.where(D == 1)[0]
        control_idx = np.where(D == 0)[0]
        assert np.isclose(w[treated_idx].sum(), 1.0, atol=1e-4)
        assert np.isclose(w[control_idx].sum(), 1.0, atol=1e-4)
        assert np.all(w >= -1e-6)

    def test_compute_energy_nonnegative(self, Y_small):
        D = np.array([1, 1, 0, 0, 0, 0])
        w = solve_weights_global(Y_small, D)
        e = compute_energy(Y_small, D, w, lam=0.1)
        assert e >= 0

    def test_compute_rmse_gap_nonnegative(self, Y_small):
        D = np.array([1, 1, 0, 0, 0, 0])
        w = solve_weights_global(Y_small, D)
        rmse = compute_rmse_gap(Y_small, D, w)
        assert rmse >= 0

    def test_synthetic_paths_shapes(self, Y_small):
        D = np.array([1, 1, 0, 0, 0, 0])
        w = solve_weights_global(Y_small, D)
        mu_T, mu_C, gap = synthetic_paths(Y_small, D, w)
        T = Y_small.shape[0]
        assert mu_T.shape == (T,)
        assert mu_C.shape == (T,)
        np.testing.assert_allclose(gap, mu_T - mu_C)

    def test_extract_weights_normalization(self):
        D = np.array([1, 1, 0, 0])
        w = np.array([0.3, 0.7, 0.4, 0.6])
        out = extract_weights(D, w)
        assert np.isclose(out["treated_weights"].sum(), 1.0)
        assert np.isclose(out["control_weights"].sum(), 1.0)
        np.testing.assert_allclose(
            out["contrast_weights"], (2 * D - 1) * w
        )


# =========================================================================
# RELAXED ANNEALING
# =========================================================================

class TestRelaxedAnnealing:

    def test_temperature_warmup(self, Y_small):
        T_init = temperature_schedule(0, Y_small)
        T_later = temperature_schedule(4, Y_small)
        # Strict geometric decay during warm-up.
        assert T_later < T_init

    def test_temperature_floor_nonzero(self, Y_small):
        # Lots of iterations with no history should still return >0
        T = temperature_schedule(100, Y_small, delta_history=[])
        assert T > 0

    def test_temperature_adaptive_after_warmup(self, Y_small):
        history = [0.05] * 30
        T = temperature_schedule(50, Y_small, delta_history=history)
        assert T > 0

    def test_propose_swap_preserves_K(self, Y_small):
        np.random.seed(0)
        D = np.array([1, 1, 0, 0, 0, 0])
        D_new, (i_idx, j_idx) = propose_swap(D, T=0.1)
        assert D_new.sum() == D.sum()
        # Treated->control and vice versa
        assert all(D[i] == 1 for i in i_idx)
        assert all(D[j] == 0 for j in j_idx)

    def test_propose_swap_high_temp_more_swaps(self, Y_small):
        np.random.seed(0)
        D = np.array([1, 1, 1, 0, 0, 0])
        # very high temperature => m ≈ max_m
        _, (i_low, _) = propose_swap(D.copy(), T=0.001, max_m=3)
        np.random.seed(0)
        _, (i_high, _) = propose_swap(D.copy(), T=100.0, max_m=3)
        assert len(i_high) >= len(i_low)

    def test_d_step_returns_swap_log(self, Y_small):
        np.random.seed(0)
        D = np.array([1, 1, 0, 0, 0, 0])
        w = solve_weights_global(Y_small, D)
        best_D, best_w, log = d_step_annealed(
            Y_small, D, w, K=2, T=0.1, lam=0.0, n_proposals=5,
        )
        assert isinstance(log, RelaxedSwapLog)
        assert log.n_proposals == 5
        assert best_D.sum() == 2


class TestRelaxedSwapLog:

    def test_uphill_acceptance_rate(self):
        log = RelaxedSwapLog(
            n_proposals=10, n_accepted=4, n_uphill=4,
            n_uphill_accepted=1, delta_history=[],
        )
        assert log.uphill_acceptance_rate == pytest.approx(1 / 4, abs=1e-6)

    def test_uphill_acceptance_zero_safe(self):
        log = RelaxedSwapLog(
            n_proposals=0, n_accepted=0, n_uphill=0,
            n_uphill_accepted=0, delta_history=[],
        )
        # Division-by-eps guard returns ~0
        assert log.uphill_acceptance_rate == pytest.approx(0.0, abs=1e-6)


# =========================================================================
# RELAXED SOLVER (end-to-end)
# =========================================================================

class TestRelaxedSolver:

    def test_solve_two_way_relaxed_smoke(self, Y_small):
        np.random.seed(0)
        result = solve_two_way_relaxed(Y_small, K=2, max_iter=5, verbose=False)
        assert isinstance(result, RelaxedSolverResults)
        assert result.mode == "global_2way_relaxed"
        assert result.design.assignment.sum() == 2
        assert np.isclose(result.design.treated_weights.sum(), 1.0, atol=1e-4)
        assert np.isclose(result.design.control_weights.sum(), 1.0, atol=1e-4)
        assert len(result.trace.objective_history) == 5

    def test_returns_synthetic_paths(self, Y_small):
        np.random.seed(0)
        result = solve_two_way_relaxed(Y_small, K=2, max_iter=3, verbose=False)
        assert result.design.synthetic_treated.shape == (Y_small.shape[0],)
        np.testing.assert_allclose(
            result.design.synthetic_gap,
            result.design.synthetic_treated - result.design.synthetic_control,
        )

    def test_user_lambda_propagates(self, Y_small):
        np.random.seed(0)
        result = solve_two_way_relaxed(Y_small, K=2, lam=0.42,
                                       max_iter=2, verbose=False)
        assert result.design.lambda_value == 0.42

    def test_invalid_inputs_raise(self):
        with pytest.raises(MlsynthDataError):
            solve_two_way_relaxed(np.zeros(5), K=1, verbose=False)

    def test_results_accessor_aliases(self, Y_small):
        np.random.seed(0)
        result = solve_two_way_relaxed(Y_small, K=2, max_iter=2, verbose=False)
        np.testing.assert_array_equal(result.assignment, result.design.assignment)
        assert result.objective_value == result.design.objective_value
        assert result.rmse == result.design.rmse


# =========================================================================
# INFERENCE
# =========================================================================

class TestPermutationInference:

    def test_global_inference(self, inputs):
        d = solve_synthetic_design(
            Y=inputs.Y_pre, K=2, mode="global_2way",
            unit_index=inputs.unit_index,
        )
        inf = permutation_test_global(
            Y_pre=inputs.Y_pre, Y_post=inputs.Y_post,
            design=d, alpha=0.10,
        )
        assert isinstance(inf, SCDIInference)
        assert 0.0 <= inf.p_value <= 1.0
        assert inf.alpha == 0.10
        assert isinstance(inf.reject, (bool, np.bool_))
        assert inf.null_stats is not None
        assert inf.null_stats.shape[0] == inputs.Y_pre.shape[0] + inputs.Y_post.shape[0]

    def test_global_inference_wrong_mode_rejected(self, inputs):
        d = solve_synthetic_design(
            Y=inputs.Y_pre, K=2, mode="global_equal_weights",
            unit_index=inputs.unit_index,
        )
        with pytest.raises(MlsynthConfigError):
            permutation_test_global(
                Y_pre=inputs.Y_pre, Y_post=inputs.Y_post,
                design=d,
            )

    def test_global_inference_no_post_rejected(self, inputs):
        d = solve_synthetic_design(
            Y=inputs.Y_pre, K=2, mode="global_2way",
            unit_index=inputs.unit_index,
        )
        with pytest.raises(MlsynthDataError):
            permutation_test_global(
                Y_pre=inputs.Y_pre, Y_post=np.array([]).reshape(0, inputs.Y_pre.shape[1]),
                design=d,
            )

    def test_relaxed_inference(self, Y_small, inputs):
        np.random.seed(0)
        result = solve_two_way_relaxed(inputs.Y_pre, K=2,
                                        max_iter=3, verbose=False)
        inf = permutation_test_relaxed_global(
            Y_pre=inputs.Y_pre, Y_post=inputs.Y_post,
            design=result.design, alpha=0.10,
        )
        assert isinstance(inf, RelaxedInference)
        assert 0.0 <= inf.p_value <= 1.0
        assert inf.null_stats is not None

    def test_relaxed_inference_no_post_rejected(self, Y_small, inputs):
        np.random.seed(0)
        result = solve_two_way_relaxed(inputs.Y_pre, K=2,
                                        max_iter=3, verbose=False)
        with pytest.raises(MlsynthDataError):
            permutation_test_relaxed_global(
                Y_pre=inputs.Y_pre,
                Y_post=np.array([]).reshape(0, inputs.Y_pre.shape[1]),
                design=result.design,
            )


# =========================================================================
# SCDI ESTIMATOR (public API)
# =========================================================================

class TestSCDIEstimator:

    def test_fit_global_2way(self, panel_df):
        est = SCDI({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "K": 2, "post_col": "post",
            "run_inference": False,
        })
        res = est.fit()
        assert isinstance(res, SCDIResults)
        assert res.mode == "global_2way"

    def test_fit_global_equal_weights(self, panel_df):
        res = SCDI({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "K": 2, "post_col": "post",
            "mode": "global_equal_weights", "run_inference": False,
        }).fit()
        assert res.mode == "global_equal_weights"

    def test_fit_per_unit(self, panel_df):
        res = SCDI({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "K": 2, "post_col": "post",
            "mode": "per_unit", "run_inference": False,
        }).fit()
        assert res.mode == "per_unit"

    def test_fit_relaxed(self, panel_df):
        np.random.seed(0)
        res = SCDI({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "K": 2, "post_col": "post",
            "mode": "global_2way_relaxed", "run_inference": False,
            "relaxed_max_iter": 3, "verbose": False,
        }).fit()
        assert res.mode == "global_2way_relaxed"
        assert isinstance(res, RelaxedSolverResults)

    def test_fit_with_inference(self, panel_df):
        res = SCDI({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "K": 2, "post_col": "post",
            "run_inference": True,
        }).fit()
        assert res.inference is not None
        assert isinstance(res.inference, SCDIInference)

    def test_fit_relaxed_with_inference(self, panel_df):
        np.random.seed(0)
        res = SCDI({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "K": 2, "post_col": "post",
            "mode": "global_2way_relaxed", "run_inference": True,
            "relaxed_max_iter": 3, "verbose": False,
        }).fit()
        assert res.inference is not None
        assert isinstance(res.inference, RelaxedInference)

    def test_fit_inference_non_global_rejected(self, panel_df):
        # MIP-mode inference is only implemented for global_2way.
        with pytest.raises(MlsynthConfigError):
            SCDI({
                "df": panel_df, "outcome": "y", "unitid": "unitid",
                "time": "time", "K": 2, "post_col": "post",
                "mode": "global_equal_weights", "run_inference": True,
            }).fit()

    def test_fit_unexpected_error_wrapped(self, monkeypatch, panel_df):
        def boom(*args, **kwargs):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(
            "mlsynth.estimators.scdi.solve_synthetic_design", boom
        )
        with pytest.raises(MlsynthEstimationError):
            SCDI({"df": panel_df, "outcome": "y", "unitid": "unitid",
                   "time": "time", "K": 2, "post_col": "post",
                   "run_inference": False}).fit()


# =========================================================================
# PLOTTING (smoke)
# =========================================================================

class TestPlotting:

    @pytest.fixture(autouse=True)
    def _matplotlib_agg(self, monkeypatch):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda: None)

    def test_plot_global_2way(self, panel_df):
        res = SCDI({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "K": 2, "post_col": "post",
            "run_inference": False,
        }).fit()
        plot_scdi_design(res)

    def test_plot_per_unit(self, panel_df):
        res = SCDI({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "K": 2, "post_col": "post",
            "mode": "per_unit", "run_inference": False,
        }).fit()
        plot_scdi_design(res)

    def test_plot_relaxed(self, panel_df):
        np.random.seed(0)
        res = SCDI({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "K": 2, "post_col": "post",
            "mode": "global_2way_relaxed", "run_inference": False,
            "relaxed_max_iter": 2, "verbose": False,
        }).fit()
        plot_scdi_design(res)

    def test_plot_unknown_mode_raises(self, panel_df):
        from mlsynth.exceptions import MlsynthPlottingError
        # Build a fake result with an unrecognized mode.
        class FakeResults:
            mode = "made_up_mode"
        with pytest.raises(MlsynthPlottingError):
            plot_scdi_design(FakeResults())


# =========================================================================
# IMMUTABILITY
# =========================================================================

class TestImmutability:

    def test_design_is_frozen(self, inputs):
        d = solve_synthetic_design(
            Y=inputs.Y_pre, K=2, mode="global_2way",
            unit_index=inputs.unit_index,
        )
        with pytest.raises(FrozenInstanceError):
            d.mode = "per_unit"   # type: ignore[misc]

    def test_relaxed_design_is_frozen(self, Y_small):
        np.random.seed(0)
        result = solve_two_way_relaxed(Y_small, K=2, max_iter=2,
                                        verbose=False)
        with pytest.raises(FrozenInstanceError):
            result.design.assignment = np.zeros(6)   # type: ignore[misc]
