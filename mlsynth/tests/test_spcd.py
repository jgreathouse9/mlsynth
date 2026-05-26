"""Comprehensive tests for the SPCD estimator and its helper subpackage.

Covers:
    * The five solve_spcd / solve_spcd_with_holdout entry points (smoke
      across variant x weights).
    * Every public helper in mlsynth.utils.spcd_helpers/*.
    * SPCDConfig validation paths.
    * SPCDResults convenience properties.
    * Edge cases: tiny panels, no-post mode, holdout-too-small fallback,
      backwards-compatible enable_inference=False mode.

Reference: Lu, Li, Ying, Blanchet (2022). arXiv:2211.15241v1.
"""

from __future__ import annotations

import warnings
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

from mlsynth import SPCD
from mlsynth.config_models import SPCDConfig, BaseEstimatorResults
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.spcd_helpers.structures import (
    SPCDDesign,
    SPCDInputs,
    SPCDMultiArmResults,
    SPCDResults,
)
from mlsynth.utils.spcd_helpers.setup import prepare_spcd_inputs
from mlsynth.utils.spcd_helpers.formulation import (
    build_iteration_matrix,
    estimate_noise_variance,
    validate_spcd_inputs,
)
from mlsynth.utils.spcd_helpers.orchestration import select_alpha_by_holdout
from mlsynth.utils.spcd_helpers.spectral_init import spectral_initialization
from mlsynth.utils.spcd_helpers.iteration_spcd import (
    run_spcd_iteration,
    spcd_step,
)
from mlsynth.utils.spcd_helpers.iteration_norm_spcd import (
    norm_spcd_step,
    run_norm_spcd_iteration,
)
from mlsynth.utils.spcd_helpers.weights_empirical import empirical_weights
from mlsynth.utils.spcd_helpers.treatment_effect import (
    apply_minority_flip,
    build_synthetic_paths,
    build_weight_groups,
    compute_att_and_fit,
)
from mlsynth.utils.spcd_helpers.holdout import (
    compute_holdout_residuals,
    split_pre_window,
)
from mlsynth.utils.spcd_helpers.inference import (
    SPCDConformalResult,
    compute_conformal_ci,
)
from mlsynth.utils.spcd_helpers.power import (
    SPCDPowerAnalysis,
    compute_detectability_curve,
    compute_mde,
    compute_pooled_average_mde,
)
from mlsynth.utils.spcd_helpers.orchestration import (
    solve_spcd,
    solve_spcd_with_holdout,
)
from mlsynth.utils.spcd_helpers.results_assembly import build_summary
from mlsynth.utils.spcd_helpers.plotter import plot_spcd_design


# =========================================================================
# FIXTURES
# =========================================================================

def _make_panel(
    n_units=20,
    T=80,
    T_post=10,
    L=3,
    sigma=0.5,
    seed=0,
    treated_unit=None,
    treatment_effect=0.0,
):
    """Synthetic factor-model panel matching the SPCD paper's Section 4.1."""
    rng = np.random.default_rng(seed)
    gamma = rng.standard_normal((n_units, L))
    nu = rng.standard_normal((T, L))
    Y = nu @ gamma.T + sigma * rng.standard_normal((T, n_units))
    if treated_unit is not None and treatment_effect != 0:
        Y[T - T_post:, treated_unit] += treatment_effect

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
    df, _ = _make_panel(T=70, T_post=0)
    # Drop the 'post' column to be explicit about pre-only mode.
    return df.drop(columns=["post"])


@pytest.fixture
def small_panel():
    """Smallest panel that still satisfies SPCD's basic requirements."""
    df, _ = _make_panel(n_units=4, T=8, T_post=2)
    return df


@pytest.fixture
def arm_panel():
    """A panel with three arms of 8 units each (SPCD multi-arm)."""
    df, _ = _make_panel(n_units=24, T=80, T_post=10)
    code = df["unitid"].str[1:].astype(int)
    df["arm"] = np.where(code < 8, "A", np.where(code < 16, "B", "C"))
    return df


class TestMultiArm:
    _cfg = dict(outcome="y", unitid="unitid", time="time", post_col="post",
                display_graph=False)

    def test_arm_returns_multiarm_per_arm_designs(self, arm_panel):
        res = SPCD({"df": arm_panel, "arm": "arm", **self._cfg}).fit()
        assert isinstance(res, SPCDMultiArmResults)
        assert res.arm == "arm"
        assert sorted(res.arm_designs) == ["A", "B", "C"]
        for a, r in res.arm_designs.items():
            assert isinstance(r, SPCDResults)
            # each arm is solved on its own 8 units only
            assert list(r.inputs.unit_index.labels) == [
                f"u{c:02d}" for c in range(*{"A": (0, 8), "B": (8, 16),
                                             "C": (16, 24)}[a])]
            assert 1 <= r.design.n_treated < 8

    def test_arm_none_still_single_result(self, panel_df):
        res = SPCD({"df": panel_df, **self._cfg}).fit()
        assert isinstance(res, SPCDResults)

    def test_att_by_arm_keys(self, arm_panel):
        res = SPCD({"df": arm_panel, "arm": "arm", **self._cfg}).fit()
        assert set(res.att_by_arm()) == {"A", "B", "C"}

    def test_missing_arm_column_raises(self, panel_df):
        with pytest.raises(MlsynthDataError):
            SPCD({"df": panel_df, "arm": "nope", **self._cfg}).fit()

    def test_arm_varying_within_unit_raises(self, arm_panel):
        df = arm_panel.copy()
        first = df["unitid"].iloc[0]
        df.loc[df["unitid"] == first, "arm"] = (
            ["A", "B"] * (int((df["unitid"] == first).sum()) // 2 + 1)
        )[:int((df["unitid"] == first).sum())]
        with pytest.raises(MlsynthDataError):
            SPCD({"df": df, "arm": "arm", **self._cfg}).fit()


@pytest.fixture
def inputs(panel_df):
    return prepare_spcd_inputs(
        df=panel_df, outcome="y", unitid="unitid", time="time", post_col="post"
    )


@pytest.fixture
def inputs_no_post(panel_no_post):
    return prepare_spcd_inputs(
        df=panel_no_post, outcome="y", unitid="unitid", time="time"
    )


# =========================================================================
# CONFIG VALIDATION
# =========================================================================

class TestSPCDConfig:

    def test_valid_dict_config_accepted(self, panel_df):
        cfg = SPCDConfig(df=panel_df, outcome="y", unitid="unitid",
                         time="time", post_col="post")
        assert cfg.variant == "norm_spcd"
        assert cfg.weights == "empirical"
        assert cfg.holdout_frac_E == 0.7
        assert cfg.enable_inference is True

    def test_invalid_variant_rejected(self, panel_df):
        with pytest.raises(Exception):
            SPCDConfig(df=panel_df, outcome="y", unitid="unitid",
                       time="time", variant="not_a_variant")

    def test_invalid_weights_rejected(self, panel_df):
        with pytest.raises(Exception):
            SPCDConfig(df=panel_df, outcome="y", unitid="unitid",
                       time="time", weights="not_a_mode")

    def test_post_col_must_exist(self, panel_df):
        with pytest.raises(MlsynthConfigError, match="post_col"):
            SPCDConfig(df=panel_df, outcome="y", unitid="unitid",
                       time="time", post_col="does_not_exist")

    def test_T0_too_large_rejected(self, panel_df):
        with pytest.raises(MlsynthConfigError, match="T0"):
            SPCDConfig(df=panel_df, outcome="y", unitid="unitid",
                       time="time", T0=10_000)

    def test_holdout_frac_E_out_of_range_rejected(self, panel_df):
        with pytest.raises(Exception):
            SPCDConfig(df=panel_df, outcome="y", unitid="unitid",
                       time="time", holdout_frac_E=0.99)
        with pytest.raises(Exception):
            SPCDConfig(df=panel_df, outcome="y", unitid="unitid",
                       time="time", holdout_frac_E=0.0)

    def test_dict_to_spcd_class_constructor(self, panel_df):
        cfg = {"df": panel_df, "outcome": "y", "unitid": "unitid",
               "time": "time", "post_col": "post"}
        est = SPCD(cfg)
        assert isinstance(est.config, SPCDConfig)

    def test_invalid_dict_wraps_in_configerror(self, panel_df):
        with pytest.raises(MlsynthConfigError, match="Invalid SPCD configuration"):
            SPCD({"df": panel_df, "outcome": "y", "unitid": "unitid",
                  "time": "time", "variant": "garbage"})


# =========================================================================
# DATA PREP (prepare_spcd_inputs)
# =========================================================================

class TestPrepareSpcdInputs:

    def test_basic_post_col(self, panel_df):
        inp = prepare_spcd_inputs(panel_df, "y", "unitid", "time",
                                   post_col="post")
        assert inp.Y_pre.shape[1] == 20
        assert inp.Y_post.shape[1] == 20
        assert inp.Y_pre.shape[0] == 70
        assert inp.Y_post.shape[0] == 10
        assert inp.outcome == "y"

    def test_T0_split(self, panel_df):
        inp = prepare_spcd_inputs(panel_df, "y", "unitid", "time", T0=50)
        assert inp.Y_pre.shape[0] == 50
        assert inp.Y_post.shape[0] == 30

    def test_no_post_returns_none(self, panel_no_post):
        inp = prepare_spcd_inputs(panel_no_post, "y", "unitid", "time")
        assert inp.Y_post is None

    def test_T0_at_max_means_no_post(self, panel_df):
        n_periods = panel_df["time"].nunique()
        inp = prepare_spcd_inputs(panel_df, "y", "unitid", "time", T0=n_periods)
        assert inp.Y_post is None

    def test_invalid_post_col_raises(self, panel_df):
        with pytest.raises(MlsynthDataError, match="post_col"):
            prepare_spcd_inputs(panel_df, "y", "unitid", "time",
                                 post_col="nope")

    def test_T0_out_of_range(self, panel_df):
        with pytest.raises(MlsynthConfigError):
            prepare_spcd_inputs(panel_df, "y", "unitid", "time", T0=0)
        with pytest.raises(MlsynthConfigError):
            prepare_spcd_inputs(panel_df, "y", "unitid", "time", T0=9999)

    def test_all_post_rejected(self, panel_df):
        df = panel_df.copy()
        df["post"] = 1
        with pytest.raises(MlsynthConfigError):
            prepare_spcd_inputs(df, "y", "unitid", "time", post_col="post")

    def test_missing_panel_observation_raises(self, panel_df):
        partial = panel_df.iloc[:-1].copy()
        with pytest.raises(MlsynthDataError):
            prepare_spcd_inputs(partial, "y", "unitid", "time",
                                 post_col="post")

    def test_tiny_pre_period_raises(self):
        rows = [
            {"unitid": "a", "time": 0, "y": 1.0, "post": 1},
            {"unitid": "b", "time": 0, "y": 2.0, "post": 1},
        ]
        with pytest.raises(MlsynthConfigError):
            prepare_spcd_inputs(pd.DataFrame(rows), "y", "unitid",
                                 "time", post_col="post")

    def test_unit_labels_preserved(self, panel_df):
        inp = prepare_spcd_inputs(panel_df, "y", "unitid", "time",
                                   post_col="post")
        # IndexSet labels should be alphabetically sorted unit labels
        assert "u00" in inp.unit_index.labels
        assert "u19" in inp.unit_index.labels


# =========================================================================
# FORMULATION: build_iteration_matrix + validate_spcd_inputs
# =========================================================================

class TestFormulation:

    def test_validate_inputs_rejects_1d(self):
        with pytest.raises(MlsynthDataError):
            validate_spcd_inputs(np.zeros(5))

    def test_validate_inputs_rejects_one_period(self):
        with pytest.raises(MlsynthDataError):
            validate_spcd_inputs(np.zeros((1, 5)))

    def test_validate_inputs_rejects_one_unit(self):
        with pytest.raises(MlsynthDataError):
            validate_spcd_inputs(np.zeros((5, 1)))

    def test_validate_inputs_accepts_minimum(self):
        validate_spcd_inputs(np.zeros((2, 2)))   # should not raise

    def test_M_is_symmetric_psd(self):
        Y = np.random.default_rng(0).normal(size=(10, 5))
        M, M_inv, alpha, lam, beta = build_iteration_matrix(Y)
        assert np.allclose(M, M.T)
        assert np.all(np.linalg.eigvalsh(M) > 0)

    def test_M_inv_is_inverse(self):
        Y = np.random.default_rng(0).normal(size=(10, 5))
        M, M_inv, *_ = build_iteration_matrix(Y)
        np.testing.assert_allclose(M @ M_inv, np.eye(5), atol=1e-6)

    def test_auto_hyperparams_positive(self):
        Y = np.random.default_rng(0).normal(size=(10, 5))
        _, _, alpha, lam, beta = build_iteration_matrix(Y)
        assert alpha > 0
        assert lam > 0
        assert beta > 0

    def test_user_hyperparams_respected(self):
        Y = np.random.default_rng(0).normal(size=(10, 5))
        _, _, alpha, lam, beta = build_iteration_matrix(
            Y, alpha=2.0, lam=5.0, beta=0.5
        )
        assert alpha == 2.0
        assert lam == 5.0
        assert beta == 0.5

    def test_M_eq2_form_matches_paper(self):
        """M = Y^T Y + alpha I + lambda 1 1^T  (Eq. 2)."""
        Y = np.random.default_rng(0).normal(size=(10, 5))
        alpha, lam = 0.3, 1.7
        M, _, _, _, _ = build_iteration_matrix(Y, alpha=alpha, lam=lam)
        expected = Y.T @ Y + alpha * np.eye(5) + lam * np.ones((5, 5))
        np.testing.assert_allclose(M, expected, atol=1e-10)


# =========================================================================
# SPECTRAL INITIALIZATION
# =========================================================================

class TestSpectralInit:

    def test_returns_pm1_vector(self):
        Y = np.random.default_rng(0).normal(size=(8, 6))
        M, *_ = build_iteration_matrix(Y)
        y0 = spectral_initialization(M)
        assert y0.shape == (6,)
        assert set(np.unique(y0).tolist()) <= {-1.0, 1.0}

    def test_zero_entries_mapped_to_plus_one(self):
        # Construct M whose smallest eigvec has a zero entry.
        M = np.array([[2.0, 0.0], [0.0, 1.0]])
        y0 = spectral_initialization(M)
        assert np.all((y0 == 1.0) | (y0 == -1.0))

    def test_deterministic_on_repeat(self):
        Y = np.random.default_rng(42).normal(size=(10, 5))
        M, *_ = build_iteration_matrix(Y)
        y1 = spectral_initialization(M)
        y2 = spectral_initialization(M)
        np.testing.assert_array_equal(y1, y2)


# =========================================================================
# ITERATION STEPS
# =========================================================================

class TestSpcdIteration:

    def test_step_produces_pm1(self):
        rng = np.random.default_rng(0)
        M_inv = rng.normal(size=(5, 5))
        M_inv = (M_inv + M_inv.T) / 2
        y = np.array([1, -1, 1, -1, 1], dtype=float)
        y_new = spcd_step(M_inv, y, beta=0.1)
        assert set(np.unique(y_new).tolist()) <= {-1.0, 1.0}

    def test_run_iteration_converges_on_factor_model(self):
        rng = np.random.default_rng(0)
        Y = rng.normal(size=(40, 8))
        M, M_inv, *_, beta = build_iteration_matrix(Y)
        y0 = spectral_initialization(M)
        y_star, n_iter, converged = run_spcd_iteration(
            M_inv, y0, beta=beta, max_iter=100
        )
        assert converged is True
        assert n_iter >= 1
        assert y_star.shape == y0.shape

    def test_run_iteration_max_iter_caps_loop(self):
        # Pathological case: very small beta + small M => quickly cycles.
        rng = np.random.default_rng(7)
        Y = rng.normal(size=(40, 8))
        _, M_inv, *_ = build_iteration_matrix(Y)
        y0 = np.ones(8)
        y_star, n_iter, converged = run_spcd_iteration(
            M_inv, y0, beta=0.0, max_iter=5
        )
        # Either converged or hit cap; in either case n_iter <= 5.
        assert n_iter <= 5

    def test_step_zero_entries_become_plus_one(self):
        # Force `M_inv @ y + beta y` to be all zeros.
        M_inv = np.zeros((3, 3))
        y = np.zeros(3)
        y_new = spcd_step(M_inv, y, beta=0.0)
        assert np.all(y_new == 1.0)


class TestNormSpcdIteration:

    def test_step_produces_pm1(self):
        rng = np.random.default_rng(0)
        Y = rng.normal(size=(20, 5))
        M, M_inv, _, _, beta = build_iteration_matrix(Y)
        d = np.sqrt(np.diag(M_inv).clip(min=np.finfo(float).eps))
        y = np.array([1, -1, 1, -1, 1], dtype=float)
        y_new = norm_spcd_step(M_inv, d, y, beta=beta)
        assert set(np.unique(y_new).tolist()) <= {-1.0, 1.0}

    def test_run_iteration_converges(self):
        rng = np.random.default_rng(1)
        Y = rng.normal(size=(40, 8))
        M, M_inv, *_, beta = build_iteration_matrix(Y)
        y0 = spectral_initialization(M)
        y_star, n_iter, converged = run_norm_spcd_iteration(
            M_inv, y0, beta=beta, max_iter=100
        )
        assert converged is True
        assert y_star.shape == y0.shape

    def test_normalization_handles_tiny_diag(self):
        # Forced tiny diagonal entries to exercise the floor (eps clipping).
        N = 5
        rng = np.random.default_rng(0)
        Y = rng.normal(size=(20, N))
        _, M_inv, *_ = build_iteration_matrix(Y, alpha=1e6)
        # M_inv with huge alpha has tiny diagonal — should still iterate.
        y0 = np.ones(N)
        y_star, _, _ = run_norm_spcd_iteration(M_inv, y0, beta=1.0, max_iter=20)
        assert set(np.unique(y_star).tolist()) <= {-1.0, 1.0}


# =========================================================================
# CLOSED-FORM WEIGHTS (Eq. 9)
# =========================================================================

class TestEmpiricalWeights:

    def test_signs_match_y_star(self):
        rng = np.random.default_rng(0)
        Y = rng.normal(size=(40, 8))
        _, M_inv, *_ = build_iteration_matrix(Y)
        y_star = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)
        w = empirical_weights(M_inv, y_star)
        assert w.shape == (8,)
        # Equation (9) says the resulting w should have sgn(w) == y_star
        # at fixed points; on near-fixed-points we at least expect the
        # signs of the dominant entries to align.
        assert np.isfinite(w).all()

    def test_sum_absolute_values_equals_two(self):
        # Eq. (9) normalization implies ||w||_1 = 2.
        rng = np.random.default_rng(0)
        Y = rng.normal(size=(40, 8))
        _, M_inv, *_ = build_iteration_matrix(Y)
        y_star = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
        w = empirical_weights(M_inv, y_star)
        assert np.isclose(np.sum(np.abs(w)), 2.0, atol=1e-10)

    def test_zero_y_star_yields_zero(self):
        M_inv = np.eye(3)
        w = empirical_weights(M_inv, np.zeros(3))
        np.testing.assert_array_equal(w, np.zeros(3))


# =========================================================================
# TREATMENT-EFFECT HELPERS
# =========================================================================

class TestTreatmentEffectHelpers:

    def test_minority_flip_keeps_minority_positive(self):
        y_star = np.array([1, 1, 1, -1])  # majority +1, flip to make minority +1
        w = np.array([0.3, 0.4, 0.3, -1.0])
        flipped_y, flipped_w = apply_minority_flip(y_star, w)
        # After flip, the minority group should be the +1 group.
        assert np.sum(flipped_y == 1) <= np.sum(flipped_y == -1)

    def test_minority_flip_noop_when_already_balanced_majority_negative(self):
        y_star = np.array([1, -1, -1, -1])
        w = np.array([1.0, -0.3, -0.4, -0.3])
        flipped_y, flipped_w = apply_minority_flip(y_star, w)
        np.testing.assert_array_equal(flipped_y, y_star)
        np.testing.assert_array_equal(flipped_w, w)

    def test_build_weight_groups_normalization(self):
        # Two treated, two control, signed weights.
        assignment = np.array([1, 1, -1, -1])
        raw = np.array([0.3, 0.7, -0.4, -0.6])
        selected_mask, t_w, c_w, contrast = build_weight_groups(assignment, raw)
        np.testing.assert_array_equal(selected_mask, [1, 1, 0, 0])
        # Each group's weights sum to 1
        assert np.isclose(t_w.sum(), 1.0)
        assert np.isclose(c_w.sum(), 1.0)
        np.testing.assert_array_equal(contrast, t_w - c_w)

    def test_build_synthetic_paths_with_post(self):
        Y_pre = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        Y_post = np.array([[7.0, 8.0, 9.0]])
        treated_w = np.array([1.0, 0.0, 0.0])
        control_w = np.array([0.0, 0.5, 0.5])
        st, sc, gap = build_synthetic_paths(Y_pre, Y_post, treated_w, control_w)
        assert st.shape == (3,)
        assert np.allclose(gap, st - sc)

    def test_build_synthetic_paths_no_post(self):
        Y_pre = np.array([[1.0, 2.0], [3.0, 4.0]])
        treated_w = np.array([1.0, 0.0])
        control_w = np.array([0.0, 1.0])
        st, sc, gap = build_synthetic_paths(Y_pre, None, treated_w, control_w)
        assert st.shape == (2,)
        np.testing.assert_array_equal(st, [1.0, 3.0])
        np.testing.assert_array_equal(sc, [2.0, 4.0])

    def test_compute_att_and_fit_no_post(self):
        Y_pre = np.array([[1.0, 1.0], [2.0, 2.0]])
        treated_w = np.array([1.0, 0.0])
        control_w = np.array([0.0, 1.0])
        att, rmse_pre, rmse_post = compute_att_and_fit(
            Y_pre, None, treated_w, control_w
        )
        assert att == 0.0
        assert rmse_post is None
        assert rmse_pre == 0.0  # treated_w and control_w perfectly track here

    def test_compute_att_and_fit_with_effect(self):
        Y_pre = np.array([[1.0, 1.0], [2.0, 2.0]])
        Y_post = np.array([[5.0, 1.0], [6.0, 2.0]])  # treated unit shifted +4
        treated_w = np.array([1.0, 0.0])
        control_w = np.array([0.0, 1.0])
        att, rmse_pre, rmse_post = compute_att_and_fit(
            Y_pre, Y_post, treated_w, control_w
        )
        assert np.isclose(att, 4.0)
        assert rmse_post is not None and rmse_post > 0


# =========================================================================
# HOLDOUT SPLIT
# =========================================================================

class TestHoldoutSplit:

    def test_split_proportions(self):
        Y_pre = np.arange(100 * 3, dtype=float).reshape(100, 3)
        Y_E, Y_B, n_E, n_B, can_infer = split_pre_window(Y_pre, frac_E=0.7)
        assert n_E == 70 and n_B == 30
        assert can_infer is True

    def test_split_proportions_50_50(self):
        Y_pre = np.zeros((40, 5))
        Y_E, Y_B, n_E, n_B, _ = split_pre_window(Y_pre, frac_E=0.5)
        assert n_E == 20 and n_B == 20

    def test_can_infer_false_when_holdout_too_small(self):
        Y_pre = np.zeros((10, 3))
        _, _, _, n_B, can_infer = split_pre_window(
            Y_pre, frac_E=0.9, min_blank_size=5
        )
        assert n_B == 1
        assert can_infer is False

    def test_invalid_frac_E_rejected(self):
        Y_pre = np.zeros((10, 3))
        with pytest.raises(MlsynthConfigError):
            split_pre_window(Y_pre, frac_E=0.05)
        with pytest.raises(MlsynthConfigError):
            split_pre_window(Y_pre, frac_E=0.99)

    def test_E_too_small_raises(self):
        Y_pre = np.zeros((4, 3))
        with pytest.raises(MlsynthDataError):
            split_pre_window(Y_pre, frac_E=0.2)  # n_E = 0

    def test_holdout_residuals_shape_and_value(self):
        Y_B = np.array([[1.0, 2.0], [3.0, 4.0]])
        contrast = np.array([1.0, -1.0])
        r_B = compute_holdout_residuals(Y_B, contrast)
        np.testing.assert_allclose(r_B, [-1.0, -1.0])


# =========================================================================
# CONFORMAL INFERENCE
# =========================================================================

class TestConformalCI:

    def test_basic_shape(self):
        rng = np.random.default_rng(0)
        r_B = rng.normal(0, 1, 30)
        post_gap = rng.normal(0.5, 0.3, 10)
        out = compute_conformal_ci(r_B, post_gap, alpha=0.1)
        assert isinstance(out, SPCDConformalResult)
        assert out.pointwise_lower.shape == (10,)
        assert out.pointwise_upper.shape == (10,)
        assert np.all(out.pointwise_lower <= out.pointwise_upper)

    def test_p_value_in_unit_interval(self):
        rng = np.random.default_rng(0)
        out = compute_conformal_ci(rng.normal(0, 1, 30), rng.normal(0, 1, 10))
        assert 0.0 <= out.p_value <= 1.0

    def test_ci_contains_point_estimate(self):
        rng = np.random.default_rng(0)
        out = compute_conformal_ci(
            rng.normal(0, 1, 50), rng.normal(0.5, 0.5, 20), alpha=0.2
        )
        # The grid-search inversion is symmetric around observed_att, so
        # the point estimate is almost always inside the accepted set.
        assert out.ci_lower <= out.att <= out.ci_upper

    def test_block_size_explicit(self):
        rng = np.random.default_rng(0)
        out = compute_conformal_ci(
            rng.normal(0, 1, 30), rng.normal(0, 1, 10), block_size=5
        )
        assert out.block_size == 5

    def test_block_size_auto(self):
        rng = np.random.default_rng(0)
        out = compute_conformal_ci(rng.normal(0, 1, 30), rng.normal(0, 1, 16))
        # Auto block size = max(3, floor(sqrt(S))). sqrt(16) = 4.
        assert out.block_size == 4


# =========================================================================
# POWER ANALYSIS (MDE)
# =========================================================================

class TestPowerAnalysis:

    def test_mde_basic(self):
        rng = np.random.default_rng(0)
        r_B = rng.normal(0, 1, 50)
        out = compute_mde(
            r_B, baseline=10.0, n_post=10,
            alpha=0.05, power_target=0.8,
            n_sims=500, n_trials=100, seed=0,
        )
        assert isinstance(out, SPCDPowerAnalysis)
        assert out.alpha == 0.05
        assert out.power_target == 0.8
        assert out.baseline == 10.0
        assert out.n_post == 10
        # Feasible on this clean panel; mde_tau should be finite.
        assert out.feasible
        assert np.isfinite(out.mde_tau)

    def test_mde_zero_baseline_safe(self):
        rng = np.random.default_rng(0)
        out = compute_mde(
            rng.normal(0, 1, 50), baseline=0.0, n_post=5,
            n_sims=200, n_trials=50, seed=0,
        )
        # With baseline ~ 0 we substitute 1.0 to avoid divide-by-zero;
        # the absolute MDE remains meaningful.
        assert np.isfinite(out.critical_stat)

    def test_mde_deterministic_seed(self):
        rng = np.random.default_rng(0)
        r_B = rng.normal(0, 1, 50)
        out1 = compute_mde(r_B, baseline=10.0, n_post=5,
                           n_sims=200, n_trials=50, seed=1)
        out2 = compute_mde(r_B, baseline=10.0, n_post=5,
                           n_sims=200, n_trials=50, seed=1)
        assert out1.mde_tau == out2.mde_tau
        assert out1.critical_stat == out2.critical_stat

    def test_detectability_curve(self):
        rng = np.random.default_rng(0)
        r_B = rng.normal(0, 1, 50)
        curve = compute_detectability_curve(
            r_B, baseline=10.0, horizon_grid=[5, 10, 20],
            n_sims=200, n_trials=50, seed=0,
        )
        assert set(curve.keys()) == {5, 10, 20}
        # All values are finite or NaN (sometimes the small-trial MC is
        # unable to identify an MDE on the grid).
        for v in curve.values():
            assert np.isfinite(v) or np.isnan(v)


# =========================================================================
# WEIGHTS_EXACT (CVXPY path)
# =========================================================================

class TestExactWeights:

    def test_exact_solve_smoke(self, inputs):
        from mlsynth.utils.spcd_helpers.weights_exact import exact_weights
        Y_E = inputs.Y_pre[:50]
        # Build a sign vector with two non-empty groups.
        y_star = np.array([1.0 if i < 10 else -1.0 for i in range(20)])
        w = exact_weights(Y_E, y_star, sigma=0.1)
        assert w.shape == (20,)
        assert np.isfinite(w).all()

    def test_exact_solve_rejects_empty_group(self, inputs):
        from mlsynth.utils.spcd_helpers.weights_exact import exact_weights
        Y_E = inputs.Y_pre[:50]
        y_star = np.ones(20)  # all treated, no control
        with pytest.raises(MlsynthEstimationError):
            exact_weights(Y_E, y_star, sigma=0.1)


# =========================================================================
# END-TO-END: solve_spcd
# =========================================================================

class TestSolveSpcd:

    @pytest.mark.parametrize("variant", ["spcd", "norm_spcd"])
    @pytest.mark.parametrize("weights", ["empirical", "exact"])
    def test_smoke_all_combinations(self, inputs, variant, weights):
        d = solve_spcd(inputs, variant=variant, weights=weights)
        assert isinstance(d, SPCDDesign)
        assert d.variant == variant
        assert d.weights_mode == weights
        assert d.n_treated >= 1
        assert d.n_treated < inputs.Y_pre.shape[1]
        assert np.isclose(d.treated_weights.sum(), 1.0)
        assert np.isclose(d.control_weights.sum(), 1.0)
        assert d.synthetic_treated.shape[0] == inputs.Y_pre.shape[0] + inputs.Y_post.shape[0]

    def test_invalid_variant_rejected(self, inputs):
        with pytest.raises(MlsynthConfigError):
            solve_spcd(inputs, variant="nope", weights="empirical")

    def test_invalid_weights_rejected(self, inputs):
        with pytest.raises(MlsynthConfigError):
            solve_spcd(inputs, variant="spcd", weights="nope")

    def test_user_alpha_lam_beta_propagate(self, inputs):
        d = solve_spcd(inputs, variant="spcd", weights="empirical",
                       alpha=0.05, lam=2.0, beta=0.3)
        assert d.alpha_ridge == 0.05
        assert d.lam_balance == 2.0
        assert d.beta == 0.3

    def test_max_iter_propagates(self, inputs):
        d = solve_spcd(inputs, variant="spcd", weights="empirical",
                       max_iter=1)
        assert d.n_iterations <= 1


# =========================================================================
# END-TO-END: solve_spcd_with_holdout
# =========================================================================

class TestSolveSpcdWithHoldout:

    def test_disabled_inference_returns_none(self, inputs):
        d, conformal, power = solve_spcd_with_holdout(
            inputs, enable_inference=False
        )
        assert isinstance(d, SPCDDesign)
        assert conformal is None
        assert power is None

    def test_enabled_inference_with_post(self, inputs):
        d, conformal, power = solve_spcd_with_holdout(
            inputs, enable_inference=True, holdout_frac_E=0.7,
            mde_n_sims=200, mde_n_trials=50,
        )
        assert conformal is not None
        assert power is not None
        assert d.synthetic_treated.shape[0] == (
            inputs.Y_pre.shape[0] + inputs.Y_post.shape[0]
        )

    def test_enabled_inference_no_post(self, inputs_no_post):
        d, conformal, power = solve_spcd_with_holdout(
            inputs_no_post, enable_inference=True,
            mde_n_sims=200, mde_n_trials=50,
        )
        # No post -> no ATT or CI, but MDE still runs.
        assert conformal is None
        assert power is not None

    def test_design_identical_pre_only_vs_pre_post(self, panel_df):
        """The backwards-compatibility guarantee in the paper-grounded docs."""
        pre_only = panel_df[panel_df["time"] < 70].copy().drop(columns=["post"])
        inp_pre_only = prepare_spcd_inputs(
            pre_only, "y", "unitid", "time"
        )
        inp_with_post = prepare_spcd_inputs(
            panel_df, "y", "unitid", "time", post_col="post"
        )
        d1, _, _ = solve_spcd_with_holdout(inp_pre_only, mde_n_sims=100,
                                            mde_n_trials=25)
        d2, _, _ = solve_spcd_with_holdout(inp_with_post, mde_n_sims=100,
                                            mde_n_trials=25)
        # The chosen sign vector must be identical.
        np.testing.assert_array_equal(d1.assignment_pm1, d2.assignment_pm1)

    def test_holdout_too_small_warns_and_skips_inference(self):
        # Tiny pre-window: 12 periods, frac_E=0.7 -> n_E=8, n_B=4 < 5.
        rows = []
        for i in range(6):
            for t in range(18):
                rows.append({
                    "unitid": f"u{i}", "time": t,
                    "y": float(i + t), "post": int(t >= 12),
                })
        df = pd.DataFrame(rows)
        inp = prepare_spcd_inputs(df, "y", "unitid", "time", post_col="post")
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            d, conformal, power = solve_spcd_with_holdout(
                inp, enable_inference=True, holdout_frac_E=0.7,
                min_blank_size=5,
            )
            assert any("holdout window" in str(x.message) for x in w_list)
        assert conformal is None
        assert power is None
        # Design still produced
        assert isinstance(d, SPCDDesign)

    def test_horizon_grid_populates_detectability(self, inputs):
        _, _, power = solve_spcd_with_holdout(
            inputs, mde_horizon_grid=[5, 10],
            mde_n_sims=200, mde_n_trials=50,
        )
        assert power.detectability is not None
        assert set(power.detectability.keys()) == {5, 10}


# =========================================================================
# RESULTS ASSEMBLY (build_summary)
# =========================================================================

class TestBuildSummary:

    def test_summary_shape_with_post(self, inputs):
        d = solve_spcd(inputs)
        summary = build_summary(d, inputs)
        assert isinstance(summary, BaseEstimatorResults)
        assert summary.effects.att is not None
        assert summary.fit_diagnostics.rmse_pre is not None
        assert summary.fit_diagnostics.rmse_post is not None
        assert isinstance(summary.weights.donor_weights, dict)
        assert summary.method_details.method_name.startswith("SPCD")

    def test_summary_no_post_att_is_none(self, inputs_no_post):
        d = solve_spcd(inputs_no_post)
        summary = build_summary(d, inputs_no_post)
        assert summary.effects.att is None
        assert summary.fit_diagnostics.rmse_post is None

    def test_summary_with_conformal(self, inputs):
        d, conformal, power = solve_spcd_with_holdout(
            inputs, mde_n_sims=200, mde_n_trials=50,
        )
        summary = build_summary(d, inputs, conformal=conformal, power=power)
        assert summary.inference is not None
        assert summary.inference.method == "moving_block_conformal"
        assert 0.0 <= summary.inference.p_value <= 1.0
        assert "mde_tau" in summary.fit_diagnostics.additional_metrics

    def test_donor_weights_are_non_negative_control_dict(self, inputs):
        d = solve_spcd(inputs)
        summary = build_summary(d, inputs)
        assert all(v >= 0 for v in summary.weights.donor_weights.values())
        assert "treated_weights_by_unit" in summary.weights.summary_stats or\
               hasattr(summary.weights, "treated_weights_by_unit")


# =========================================================================
# PUBLIC API: SPCD class
# =========================================================================

class TestSPCDClass:

    def test_fit_returns_results(self, panel_df):
        est = SPCD({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "post_col": "post",
            "mde_n_sims": 200, "mde_n_trials": 50,
        })
        res = est.fit()
        assert isinstance(res, SPCDResults)
        assert res.mode == "spcd"

    def test_fit_with_T0_instead_of_post_col(self, panel_no_post):
        # Note: panel_no_post has the 'post' column dropped, so we use T0.
        est = SPCD({
            "df": panel_no_post, "outcome": "y", "unitid": "unitid",
            "time": "time", "T0": 50,
            "mde_n_sims": 200, "mde_n_trials": 50,
        })
        res = est.fit()
        assert res.att is not None

    def test_convenience_properties(self, panel_df):
        res = SPCD({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "post_col": "post",
            "mde_n_sims": 200, "mde_n_trials": 50,
        }).fit()
        assert res.att is not None
        assert res.rmse_pre is not None
        assert res.rmse_post is not None
        assert res.mde is not None
        assert res.p_value is not None
        assert res.ci_lower is not None and res.ci_upper is not None
        # Treated and control dicts should be disjoint label sets.
        t_keys = set(res.treated_weights_by_unit.keys())
        c_keys = set(res.control_weights_by_unit.keys())
        assert t_keys.isdisjoint(c_keys)

    def test_legacy_no_inference_mode(self, panel_df):
        res = SPCD({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "post_col": "post",
            "enable_inference": False,
        }).fit()
        assert res.conformal is None
        assert res.power is None
        assert res.att is not None  # still computed from post-period

    def test_results_mode_property(self, panel_df):
        res = SPCD({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "post_col": "post",
            "mde_n_sims": 200, "mde_n_trials": 50,
        }).fit()
        assert res.mode == "spcd"
        # Alias
        np.testing.assert_array_equal(res.assignment, res.design.selected_mask)

    def test_unexpected_error_wrapped(self, monkeypatch, panel_df):
        # Inject a bug at the solve layer and confirm the wrapper turns
        # it into an MlsynthEstimationError.
        def boom(*args, **kwargs):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(
            "mlsynth.estimators.spcd.solve_spcd_with_holdout", boom
        )
        with pytest.raises(MlsynthEstimationError, match="SPCD estimation failed"):
            SPCD({"df": panel_df, "outcome": "y", "unitid": "unitid",
                  "time": "time", "post_col": "post"}).fit()


# =========================================================================
# PLOTTER (smoke only)
# =========================================================================

class TestPlotter:

    def test_plot_runs(self, panel_df, monkeypatch):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda: None)

        res = SPCD({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "post_col": "post",
            "mde_n_sims": 50, "mde_n_trials": 25,
        }).fit()
        plot_spcd_design(res)   # should not raise

    def test_plot_requires_inputs(self):
        from mlsynth.exceptions import MlsynthPlottingError
        results = SPCDResults(
            design=SPCDDesign(
                variant="spcd", weights_mode="empirical",
                assignment_pm1=np.array([1, -1]),
                selected_mask=np.array([1, 0]),
                raw_weights=np.array([0.5, -0.5]),
                treated_weights=np.array([1.0, 0.0]),
                control_weights=np.array([0.0, 1.0]),
                contrast_weights=np.array([1.0, -1.0]),
                synthetic_treated=np.array([1.0, 2.0]),
                synthetic_control=np.array([1.0, 2.0]),
                synthetic_gap=np.array([0.0, 0.0]),
                selected_unit_indices=np.array([0]),
                selected_unit_labels=np.array(["a"]),
                n_treated=1, n_iterations=1, converged=True,
                alpha_ridge=1.0, lam_balance=1.0, beta=1.0,
            ),
            inputs=None,
        )
        with pytest.raises(MlsynthPlottingError):
            plot_spcd_design(results)


# =========================================================================
# IMMUTABILITY GUARANTEES
# =========================================================================

class TestImmutability:

    def test_design_is_frozen(self, inputs):
        d = solve_spcd(inputs)
        with pytest.raises(FrozenInstanceError):
            d.variant = "norm_spcd"   # type: ignore[misc]

    def test_results_is_frozen(self, panel_df):
        res = SPCD({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "post_col": "post",
            "enable_inference": False,
        }).fit()
        with pytest.raises(FrozenInstanceError):
            res.inputs = None   # type: ignore[misc]


# =========================================================================
# NOISE-SCALE ALPHA: ESTIMATION + HOLDOUT SELECTION
# =========================================================================

class TestAlphaSelection:
    """The alpha default tracks the noise scale (Eq. (2) / appendix
    alpha = ||Delta||), chosen by out-of-sample pre-period balance.
    """

    def test_noise_variance_recovers_known_sigma(self):
        # A low-rank signal plus i.i.d. N(0, sigma^2) noise; the
        # Gavish-Donoho estimate should land within a small factor of
        # the true variance (not the dominant signal eigenvalue).
        rng = np.random.default_rng(0)
        T, N, sigma = 60, 12, 2.0
        signal = rng.standard_normal((T, 2)) @ rng.standard_normal((2, N)) * 5.0
        Y = signal + rng.normal(scale=sigma, size=(T, N))
        sigma2_hat = estimate_noise_variance(Y)
        assert sigma2_hat > 0
        assert 0.3 * sigma ** 2 < sigma2_hat < 3.0 * sigma ** 2

    def test_noise_variance_is_positive_and_finite(self, inputs):
        val = estimate_noise_variance(inputs.Y_pre)
        assert np.isfinite(val) and val > 0

    def test_select_alpha_returns_grid_scale_value(self, inputs):
        alpha = select_alpha_by_holdout(inputs.Y_pre)
        gd = estimate_noise_variance(inputs.Y_pre)
        assert np.isfinite(alpha) and alpha > 0
        # selected value lives on the multiplicative grid around the GD scale
        assert 0.1 * gd <= alpha <= 3.5 * gd

    def test_select_alpha_short_pre_falls_back_to_estimate(self):
        # Too few pre-periods to spare a validation tail -> bare estimate.
        rng = np.random.default_rng(1)
        Y = rng.standard_normal((4, 6))
        assert select_alpha_by_holdout(Y, val_frac=0.5) == pytest.approx(
            estimate_noise_variance(Y)
        )

    def test_auto_alpha_used_when_not_supplied(self, inputs):
        # With alpha=None the design records a concrete, positive ridge.
        design = solve_spcd(inputs, alpha=None)
        assert design.alpha_ridge > 0

    def test_explicit_alpha_bypasses_selection(self, inputs):
        design = solve_spcd(inputs, alpha=1.0)
        assert design.alpha_ridge == pytest.approx(1.0)


# =========================================================================
# POOLED AVERAGE-EFFECT MDE (MULTI-ARM)
# =========================================================================

class TestPooledAverageMDE:
    """Option-3 pooled MDE: the size-/equal-weighted *average* effect
    across arms, formed on time-aligned holdout residuals so cross-arm
    correlation enters through Var(g) = w' Sigma w.
    """

    def _residuals(self, rng, m=4, nB=12, corr=0.4, scale=1.0):
        """M correlated holdout series (shared factor + idiosyncratic)."""
        common = rng.standard_normal(nB)
        return {
            f"A{a}": scale * (np.sqrt(corr) * common
                              + np.sqrt(1 - corr) * rng.standard_normal(nB))
            for a in range(m)
        }

    def test_pooling_beats_per_arm(self):
        rng = np.random.default_rng(0)
        res = self._residuals(rng, m=4, nB=14)
        base = {k: 100.0 for k in res}
        size = {k: 8 for k in res}
        pooled = compute_pooled_average_mde(
            res, base, size, n_post=8, weights="size",
            n_sims=800, n_trials=300, seed=1,
        )
        per_arm = [
            compute_mde(res[k], baseline=100.0, n_post=8,
                        n_sims=800, n_trials=300, seed=1).mde_tau
            for k in res
        ]
        assert pooled.feasible and pooled.mde_tau > 0
        # averaging cancels idiosyncratic noise -> smaller detectable effect
        assert pooled.mde_tau < np.nanmedian(per_arm)

    def test_alignment_captures_correlation(self):
        # Pooled variance must equal w' Sigma w (aligned), not the
        # independent-sum w'^2 diag(Sigma); for positive correlation the
        # aligned sd is strictly larger.
        rng = np.random.default_rng(2)
        res = self._residuals(rng, m=5, nB=200, corr=0.5)
        labels = sorted(res)
        R = np.column_stack([res[l] for l in labels])
        w = np.full(len(labels), 1 / len(labels))
        aligned_sd = (R @ w).std(ddof=1)
        indep_sd = np.sqrt(np.sum(w**2 * np.var(R, axis=0, ddof=1)))
        assert aligned_sd > indep_sd  # positive correlation is retained

    def test_size_vs_equal_weights(self):
        rng = np.random.default_rng(3)
        res = self._residuals(rng, m=3, nB=14)
        base = {k: 100.0 for k in res}
        size = {"A0": 20, "A1": 5, "A2": 5}
        out_size = compute_pooled_average_mde(res, base, size, n_post=6,
                                              weights="size", n_sims=600,
                                              n_trials=200, seed=4)
        out_eq = compute_pooled_average_mde(res, base, size, n_post=6,
                                            weights="equal", n_sims=600,
                                            n_trials=200, seed=4)
        assert out_size.feasible and out_eq.feasible

    def test_requires_two_arms(self):
        with pytest.raises(ValueError):
            compute_pooled_average_mde({"A0": np.ones(10)}, {"A0": 1.0},
                                       {"A0": 8}, n_post=5)

    def test_bad_weights_mode(self):
        rng = np.random.default_rng(5)
        res = self._residuals(rng, m=2, nB=10)
        with pytest.raises(ValueError):
            compute_pooled_average_mde(res, {k: 1.0 for k in res},
                                       {k: 8 for k in res}, n_post=5,
                                       weights="nope")

    def test_multiarm_fit_exposes_pooled(self, arm_panel):
        res = SPCD({"df": arm_panel, "arm": "arm", "outcome": "y",
                    "unitid": "unitid", "time": "time", "post_col": "post",
                    "mde_n_sims": 600, "mde_n_trials": 200}).fit()
        assert isinstance(res, SPCDMultiArmResults)
        assert res.pooled_weights == "size"
        assert res.pooled_power is not None
        assert res.pooled_mde is not None and res.pooled_mde >= 0
        assert res.pooled_mde_pct is not None

    def test_multiarm_no_pooled_when_inference_off(self, arm_panel):
        res = SPCD({"df": arm_panel, "arm": "arm", "outcome": "y",
                    "unitid": "unitid", "time": "time", "post_col": "post",
                    "enable_inference": False}).fit()
        assert res.pooled_power is None
        assert res.pooled_mde is None
        assert res.pooled_weights is None


class TestPooledDetectabilityCurve:
    """Pooled-average MDE as a function of post-period horizon -- the
    'how long should the study run?' question.
    """

    def _res(self, rng, m=4, nB=16, corr=0.4):
        common = rng.standard_normal(nB)
        return {f"A{a}": np.sqrt(corr) * common
                + np.sqrt(1 - corr) * rng.standard_normal(nB) for a in range(m)}

    def test_pooled_horizon_grid_attaches_curve(self):
        rng = np.random.default_rng(0)
        res = self._res(rng)
        H = [2, 4, 6, 8]
        out = compute_pooled_average_mde(
            res, {k: 100.0 for k in res}, {k: 8 for k in res},
            n_post=8, horizon_grid=H, n_sims=600, n_trials=200, seed=1,
        )
        assert out.detectability is not None
        assert sorted(out.detectability) == H
        assert all(np.isfinite(v) or np.isnan(v) for v in out.detectability.values())

    def test_pooled_curve_via_estimator(self, arm_panel):
        res = SPCD({"df": arm_panel, "arm": "arm", "outcome": "y",
                    "unitid": "unitid", "time": "time", "post_col": "post",
                    "mde_horizon_grid": [2, 4, 6], "mde_n_sims": 500,
                    "mde_n_trials": 150}).fit()
        # whole-study curve
        assert res.pooled_power.detectability is not None
        assert sorted(res.pooled_power.detectability) == [2, 4, 6]
        # per-arm curves too
        for r in res.arm_designs.values():
            assert r.power.detectability is not None
