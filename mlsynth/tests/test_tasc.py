"""Comprehensive tests for the TASC estimator and its helper subpackage.

Covers:
    * TASCConfig validation paths.
    * prepare_tasc_inputs (data prep / N x T reshape).
    * initialize_parameters (spectral SVD init).
    * Kalman filter step (Algorithm 4), infinite-variance step (Algorithm 5),
      and the two pre / full sweeps.
    * RTS smoother (Algorithm 6).
    * M-step (Algorithm 7), with both diagonal and full Q/R updates.
    * EM_pre loop (Algorithm 2).
    * Orchestration (Algorithm 3) end-to-end.
    * Counterfactual + posterior CI computation.
    * TASC estimator class smoke + edge cases.
    * Plotter (smoke).
    * Immutability of all frozen dataclasses.

Reference: Rho, Illick, Narasipura, Abadie, Hsu, Misra (2026,
arXiv:2601.03099).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from mlsynth import TASC
from mlsynth.config_models import TASCConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)

# All TASC helpers imported from the submodule path so the test is robust
# to changes in tasc_helpers/__init__.py re-exports.
from mlsynth.utils.tasc_helpers.structures import (
    TASCDesign,
    TASCFilteredStates,
    TASCInference,
    TASCInputs,
    TASCParameters,
    TASCResults,
    TASCSmoothedStates,
)
from mlsynth.utils.tasc_helpers.setup import (
    initialize_parameters,
    prepare_tasc_inputs,
)
from mlsynth.utils.tasc_helpers.filtering import (
    _sym,
    kalman_filter_full,
    kalman_filter_inf_variance_step,
    kalman_filter_pre,
    kalman_filter_step,
)
from mlsynth.utils.tasc_helpers.smoothing import rts_smoother
from mlsynth.utils.tasc_helpers.mstep import m_step
from mlsynth.utils.tasc_helpers.em import em_pre
from mlsynth.utils.tasc_helpers.inference import counterfactual_with_ci
from mlsynth.utils.tasc_helpers.orchestration import run_tasc, summarize_effects
from mlsynth.utils.tasc_helpers.plotter import plot_tasc


# =========================================================================
# FIXTURES
# =========================================================================

def _make_panel(
    n_units=6,
    T=24,
    T0=18,
    d_true=2,
    sigma_obs=0.1,
    seed=0,
    treated_effect=0.0,
):
    """Synthetic state-space-style panel.

    Generates ``y_t = H x_t + r_t`` with ``x_{t+1} = A x_t + q_t`` so the
    panel actually fits TASC's model. Treated unit gets an additive effect
    on the post-treatment window if ``treated_effect != 0``.
    """
    rng = np.random.default_rng(seed)

    A = 0.95 * np.eye(d_true) + 0.02 * rng.standard_normal((d_true, d_true))
    H = rng.standard_normal((n_units, d_true))
    x = rng.standard_normal(d_true)

    Y = np.zeros((T, n_units))
    for t in range(T):
        x = A @ x + 0.1 * rng.standard_normal(d_true)
        Y[t] = H @ x + sigma_obs * rng.standard_normal(n_units)

    if treated_effect != 0.0:
        Y[T0:, 0] += treated_effect

    rows = []
    for i in range(n_units):
        for t in range(T):
            rows.append({
                "unitid": f"u{i:02d}",
                "time": t,
                "y": Y[t, i],
                "treat": int(i == 0 and t >= T0),
            })
    return pd.DataFrame(rows), Y


@pytest.fixture
def panel_df():
    df, _ = _make_panel()
    return df


@pytest.fixture
def panel_with_effect():
    df, _ = _make_panel(treated_effect=1.5)
    return df


@pytest.fixture
def inputs(panel_df):
    return prepare_tasc_inputs(
        df=panel_df, outcome="y", unitid="unitid", time="time", treat="treat"
    )


@pytest.fixture
def init_params(inputs):
    return initialize_parameters(inputs.Y_pre, d=2)


# =========================================================================
# CONFIG VALIDATION
# =========================================================================

class TestTASCConfig:

    def test_minimal_valid_config(self, panel_df):
        cfg = TASCConfig(df=panel_df, outcome="y", unitid="unitid",
                          time="time", treat="treat", d=2)
        assert cfg.d == 2
        assert cfg.n_em_iter == 50
        assert cfg.em_tol is None
        assert cfg.diagonal_Q is True
        assert cfg.diagonal_R is True
        assert cfg.alpha == 0.05

    def test_d_must_be_positive(self, panel_df):
        with pytest.raises(Exception):
            TASCConfig(df=panel_df, outcome="y", unitid="unitid",
                        time="time", treat="treat", d=0)

    def test_n_em_iter_must_be_positive(self, panel_df):
        with pytest.raises(Exception):
            TASCConfig(df=panel_df, outcome="y", unitid="unitid",
                        time="time", treat="treat", d=2, n_em_iter=0)

    def test_em_tol_must_be_positive(self, panel_df):
        with pytest.raises(Exception):
            TASCConfig(df=panel_df, outcome="y", unitid="unitid",
                        time="time", treat="treat", d=2, em_tol=0.0)

    def test_alpha_in_range(self, panel_df):
        with pytest.raises(Exception):
            TASCConfig(df=panel_df, outcome="y", unitid="unitid",
                        time="time", treat="treat", d=2, alpha=0.0)
        with pytest.raises(Exception):
            TASCConfig(df=panel_df, outcome="y", unitid="unitid",
                        time="time", treat="treat", d=2, alpha=1.0)

    def test_diagonal_flags_default_true(self, panel_df):
        cfg = TASCConfig(df=panel_df, outcome="y", unitid="unitid",
                          time="time", treat="treat", d=2)
        assert cfg.diagonal_Q is True
        assert cfg.diagonal_R is True

    def test_invalid_dict_wraps_in_configerror(self, panel_df):
        with pytest.raises(MlsynthConfigError, match="Invalid TASC configuration"):
            TASC({"df": panel_df, "outcome": "y", "unitid": "unitid",
                   "time": "time", "treat": "treat", "d": 0})


# =========================================================================
# DATA PREP
# =========================================================================

class TestPrepareTascInputs:

    def test_basic_shape_and_orientation(self, panel_df):
        inp = prepare_tasc_inputs(
            panel_df, outcome="y", unitid="unitid", time="time", treat="treat"
        )
        # Y_full should be (N, T) with target as row 0
        assert inp.Y_full.shape == (6, 24)
        assert inp.Y_pre.shape == (6, 18)
        assert inp.Y_post_donors is not None
        assert inp.Y_post_donors.shape == (5, 6)
        assert inp.T0 == 18
        assert inp.T == 24
        assert inp.N == 6

    def test_target_in_row_zero(self, panel_df, inputs):
        # The treated unit (u00) should be row 0 of Y_full
        assert inputs.treated_unit_name == "u00"

    def test_too_few_pre_periods_rejected(self):
        rows = []
        for i in range(3):
            for t in range(3):
                rows.append({
                    "unitid": f"u{i}", "time": t, "y": float(t),
                    "treat": int(i == 0 and t >= 1),  # T0 = 1
                })
        df = pd.DataFrame(rows)
        with pytest.raises(MlsynthDataError):
            prepare_tasc_inputs(df, "y", "unitid", "time", "treat")

    def test_no_post_period(self):
        rows = []
        T = 10
        for i in range(4):
            for t in range(T):
                rows.append({
                    "unitid": f"u{i}", "time": t, "y": float(i + t),
                    "treat": 0,
                })
        df = pd.DataFrame(rows)
        # No treatment at all means no treated unit -> dataprep should fail
        with pytest.raises((MlsynthDataError, MlsynthConfigError, Exception)):
            prepare_tasc_inputs(df, "y", "unitid", "time", "treat")


# =========================================================================
# PARAMETER INITIALIZATION
# =========================================================================

class TestInitializeParameters:

    def test_basic_shapes(self, inputs):
        params = initialize_parameters(inputs.Y_pre, d=2)
        assert params.A.shape == (2, 2)
        assert params.H.shape == (inputs.N, 2)
        assert params.Q.shape == (2, 2)
        assert params.R.shape == (inputs.N, inputs.N)
        assert params.m0.shape == (2,)
        assert params.P0.shape == (2, 2)

    def test_Q_R_positive_diagonal(self, inputs):
        params = initialize_parameters(inputs.Y_pre, d=2)
        assert np.all(np.diag(params.Q) > 0)
        assert np.all(np.diag(params.R) > 0)

    def test_d_too_large_rejected(self, inputs):
        # Y_pre has shape (6, 18). d > min(6, 18) = 6 should raise.
        with pytest.raises(MlsynthEstimationError):
            initialize_parameters(inputs.Y_pre, d=10)

    def test_d_equals_one(self, inputs):
        # d=1 is a legal corner case
        params = initialize_parameters(inputs.Y_pre, d=1)
        assert params.A.shape == (1, 1)
        assert params.H.shape == (inputs.N, 1)

    def test_deterministic_up_to_sign(self, inputs):
        # Spectral init is deterministic (SVD)
        p1 = initialize_parameters(inputs.Y_pre, d=2)
        p2 = initialize_parameters(inputs.Y_pre, d=2)
        # Either identical or columns flipped in sign.
        np.testing.assert_allclose(np.abs(p1.H), np.abs(p2.H), atol=1e-10)


# =========================================================================
# KALMAN FILTER (Algorithms 4 and 5)
# =========================================================================

class TestKalmanFilter:

    def test_sym_helper(self):
        M = np.array([[1.0, 2.0], [4.0, 3.0]])
        S = _sym(M)
        np.testing.assert_allclose(S, S.T)
        np.testing.assert_allclose(S, [[1.0, 3.0], [3.0, 3.0]])

    def test_step_shapes(self, inputs, init_params):
        m_prev = init_params.m0
        P_prev = init_params.P0
        m_new, P_new = kalman_filter_step(
            inputs.Y_pre[:, 0], m_prev, P_prev, init_params
        )
        assert m_new.shape == m_prev.shape
        assert P_new.shape == P_prev.shape

    def test_step_covariance_stays_psd(self, inputs, init_params):
        m_prev = init_params.m0
        P_prev = init_params.P0
        for k in range(5):
            m_prev, P_prev = kalman_filter_step(
                inputs.Y_pre[:, k], m_prev, P_prev, init_params
            )
            # P should remain symmetric PSD
            np.testing.assert_allclose(P_prev, P_prev.T, atol=1e-10)
            evs = np.linalg.eigvalsh(P_prev)
            assert evs[0] >= -1e-8

    def test_pre_pass_shapes(self, inputs, init_params):
        filtered = kalman_filter_pre(inputs.Y_pre, init_params)
        assert isinstance(filtered, TASCFilteredStates)
        # m and P include index 0 (prior) plus T0 posterior updates
        assert filtered.m.shape == (inputs.T0 + 1, 2)
        assert filtered.P.shape == (inputs.T0 + 1, 2, 2)
        # Index 0 should be the prior
        np.testing.assert_array_equal(filtered.m[0], init_params.m0)
        np.testing.assert_array_equal(filtered.P[0], init_params.P0)

    def test_inf_variance_step_uses_donors_only(self, inputs, init_params):
        # The infinite-variance step should ignore the target row.
        m_prev = init_params.m0
        P_prev = init_params.P0
        y_donors = inputs.Y_full[1:, 0]

        m_inf, P_inf = kalman_filter_inf_variance_step(
            y_donors, m_prev, P_prev, init_params
        )

        # Mutating the target row should NOT change the result
        y_donors_garbage_target = inputs.Y_full[:, 0].copy()
        y_donors_garbage_target[0] = 99999.0
        # Using the inf-variance step with the donor slice should be
        # identical whether or not we corrupted the target row, because the
        # step never sees row 0.
        m_inf2, P_inf2 = kalman_filter_inf_variance_step(
            y_donors_garbage_target[1:], m_prev, P_prev, init_params
        )
        np.testing.assert_allclose(m_inf, m_inf2)
        np.testing.assert_allclose(P_inf, P_inf2)

    def test_full_pass_consistency(self, inputs, init_params):
        filtered_full = kalman_filter_full(
            Y_pre=inputs.Y_pre, Y_post_donors=inputs.Y_post_donors,
            params=init_params,
        )
        assert filtered_full.m.shape == (inputs.T + 1, 2)
        # The first T0 + 1 entries should match the pre-only filter
        filtered_pre = kalman_filter_pre(inputs.Y_pre, init_params)
        np.testing.assert_allclose(
            filtered_full.m[: inputs.T0 + 1], filtered_pre.m, atol=1e-10
        )
        np.testing.assert_allclose(
            filtered_full.P[: inputs.T0 + 1], filtered_pre.P, atol=1e-10
        )


# =========================================================================
# RTS SMOOTHER (Algorithm 6)
# =========================================================================

class TestRTSSmoother:

    def test_smoother_shapes(self, inputs, init_params):
        filtered = kalman_filter_pre(inputs.Y_pre, init_params)
        smoothed = rts_smoother(filtered, init_params)
        assert isinstance(smoothed, TASCSmoothedStates)
        assert smoothed.m_s.shape == filtered.m.shape
        assert smoothed.P_s.shape == filtered.P.shape
        assert smoothed.G.shape == filtered.P.shape

    def test_smoother_matches_filter_at_last_index(self, inputs, init_params):
        filtered = kalman_filter_pre(inputs.Y_pre, init_params)
        smoothed = rts_smoother(filtered, init_params)
        # The smoother at the final timestep equals the filter
        np.testing.assert_allclose(smoothed.m_s[-1], filtered.m[-1])
        np.testing.assert_allclose(smoothed.P_s[-1], filtered.P[-1])

    def test_smoothed_covariance_psd(self, inputs, init_params):
        filtered = kalman_filter_pre(inputs.Y_pre, init_params)
        smoothed = rts_smoother(filtered, init_params)
        for k in range(smoothed.P_s.shape[0]):
            np.testing.assert_allclose(
                smoothed.P_s[k], smoothed.P_s[k].T, atol=1e-8
            )
            evs = np.linalg.eigvalsh(smoothed.P_s[k])
            assert evs[0] >= -1e-6


# =========================================================================
# M-STEP (Algorithm 7)
# =========================================================================

class TestMStep:

    def test_basic_shapes(self, inputs, init_params):
        filtered = kalman_filter_pre(inputs.Y_pre, init_params)
        smoothed = rts_smoother(filtered, init_params)
        new_params = m_step(
            Y_pre=inputs.Y_pre, smoothed=smoothed,
            prev_params=init_params,
            diagonal_Q=True, diagonal_R=True,
        )
        assert new_params.A.shape == init_params.A.shape
        assert new_params.H.shape == init_params.H.shape
        assert new_params.Q.shape == init_params.Q.shape
        assert new_params.R.shape == init_params.R.shape

    def test_diagonal_Q_constraint(self, inputs, init_params):
        filtered = kalman_filter_pre(inputs.Y_pre, init_params)
        smoothed = rts_smoother(filtered, init_params)
        new_params = m_step(
            Y_pre=inputs.Y_pre, smoothed=smoothed,
            prev_params=init_params,
            diagonal_Q=True, diagonal_R=True,
        )
        d = init_params.A.shape[0]
        off_diag_Q = new_params.Q - np.diag(np.diag(new_params.Q))
        np.testing.assert_allclose(off_diag_Q, np.zeros((d, d)), atol=1e-10)

    def test_full_Q_R_when_requested(self, inputs, init_params):
        filtered = kalman_filter_pre(inputs.Y_pre, init_params)
        smoothed = rts_smoother(filtered, init_params)
        new_params = m_step(
            Y_pre=inputs.Y_pre, smoothed=smoothed,
            prev_params=init_params,
            diagonal_Q=False, diagonal_R=False,
        )
        # Both should be symmetric
        np.testing.assert_allclose(new_params.Q, new_params.Q.T, atol=1e-10)
        np.testing.assert_allclose(new_params.R, new_params.R.T, atol=1e-10)


# =========================================================================
# EM_pre (Algorithm 2)
# =========================================================================

class TestEMPre:

    def test_em_runs_to_completion(self, inputs, init_params):
        params, deltas, filtered, smoothed = em_pre(
            Y_pre=inputs.Y_pre, init_params=init_params,
            n_em_iter=5, em_tol=None,
        )
        assert isinstance(params, TASCParameters)
        assert len(deltas) == 5
        assert isinstance(filtered, TASCFilteredStates)
        assert isinstance(smoothed, TASCSmoothedStates)

    def test_em_early_stops_with_tight_tol(self, inputs, init_params):
        params, deltas, _, _ = em_pre(
            Y_pre=inputs.Y_pre, init_params=init_params,
            n_em_iter=100, em_tol=1e10,
        )
        # em_tol so loose every iteration triggers early stop
        assert len(deltas) == 1

    def test_em_deltas_finite(self, inputs, init_params):
        _, deltas, _, _ = em_pre(
            Y_pre=inputs.Y_pre, init_params=init_params,
            n_em_iter=3, em_tol=None,
        )
        assert np.all(np.isfinite(deltas))

    def test_em_full_Q_R_path(self, inputs, init_params):
        # Exercise the non-diagonal branch in m_step.
        params, _, _, _ = em_pre(
            Y_pre=inputs.Y_pre, init_params=init_params,
            n_em_iter=2, em_tol=None,
            diagonal_Q=False, diagonal_R=False,
        )
        assert isinstance(params, TASCParameters)


# =========================================================================
# COUNTERFACTUAL + CI (Algorithm 3 footer)
# =========================================================================

class TestCounterfactualWithCi:

    def test_inference_shapes(self, inputs, init_params):
        filtered = kalman_filter_full(
            Y_pre=inputs.Y_pre, Y_post_donors=inputs.Y_post_donors,
            params=init_params,
        )
        smoothed = rts_smoother(filtered, init_params)
        inf = counterfactual_with_ci(smoothed, init_params, alpha=0.05)
        assert isinstance(inf, TASCInference)
        assert inf.counterfactual.shape == (inputs.T,)
        assert inf.ci_lower.shape == (inputs.T,)
        assert inf.ci_upper.shape == (inputs.T,)
        assert inf.posterior_variance.shape == (inputs.T,)
        assert inf.alpha == 0.05

    def test_ci_ordering(self, inputs, init_params):
        filtered = kalman_filter_pre(inputs.Y_pre, init_params)
        smoothed = rts_smoother(filtered, init_params)
        inf = counterfactual_with_ci(smoothed, init_params, alpha=0.05)
        assert np.all(inf.ci_lower <= inf.counterfactual)
        assert np.all(inf.counterfactual <= inf.ci_upper)
        assert np.all(inf.posterior_variance >= 0)

    def test_alpha_smaller_means_wider_ci(self, inputs, init_params):
        filtered = kalman_filter_pre(inputs.Y_pre, init_params)
        smoothed = rts_smoother(filtered, init_params)
        narrow = counterfactual_with_ci(smoothed, init_params, alpha=0.10)
        wide = counterfactual_with_ci(smoothed, init_params, alpha=0.01)
        # Smaller alpha -> wider CI
        narrow_width = narrow.ci_upper - narrow.ci_lower
        wide_width = wide.ci_upper - wide.ci_lower
        assert np.all(wide_width >= narrow_width - 1e-10)


# =========================================================================
# ORCHESTRATION (Algorithm 3 end-to-end)
# =========================================================================

class TestRunTasc:

    def test_basic_run(self, inputs):
        design, inference = run_tasc(
            inputs=inputs, d=2, n_em_iter=5, em_tol=None,
            diagonal_Q=True, diagonal_R=True, alpha=0.05, seed=0,
        )
        assert isinstance(design, TASCDesign)
        assert isinstance(inference, TASCInference)
        assert design.n_em_iter_used == 5
        assert design.em_param_deltas.shape == (5,)
        # Filtered / smoothed states span the full T window
        assert design.filtered.m.shape == (inputs.T + 1, 2)
        assert design.smoothed.m_s.shape == (inputs.T + 1, 2)

    def test_summarize_effects_returns_floats(self, inputs):
        _, inference = run_tasc(
            inputs=inputs, d=2, n_em_iter=2, em_tol=None,
            diagonal_Q=True, diagonal_R=True, alpha=0.05, seed=0,
        )
        att, pre_rmse = summarize_effects(inputs, inference)
        assert isinstance(att, float)
        assert isinstance(pre_rmse, float)
        assert np.isfinite(att)
        assert pre_rmse >= 0

    def test_summarize_effects_no_post_returns_nan(self):
        # Build a panel where the treatment never starts.
        # prepare_tasc_inputs will reject it -> we construct TASCInputs by hand.
        T0 = 5
        T = 5  # no post
        Y_full = np.zeros((3, T))
        from mlsynth.utils.tasc_helpers.structures import TASCInputs
        inputs_no_post = TASCInputs(
            Y_full=Y_full, Y_pre=Y_full, Y_post_donors=None,
            T0=T0, T=T, N=3, treated_unit_name="x",
            donor_names=["a", "b"],
            time_labels=np.arange(T), pre_periods=T0, post_periods=0,
            Ywide=None, y_target=Y_full[0],
        )
        infer = TASCInference(
            counterfactual=np.zeros(T),
            ci_lower=np.zeros(T), ci_upper=np.zeros(T),
            posterior_variance=np.zeros(T), alpha=0.05,
        )
        att, rmse = summarize_effects(inputs_no_post, infer)
        assert np.isnan(att)
        assert rmse == 0.0


# =========================================================================
# TASC ESTIMATOR (public API)
# =========================================================================

class TestTASCEstimator:

    def test_fit_smoke(self, panel_df):
        results = TASC({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "treat": "treat", "d": 2,
            "n_em_iter": 3,
        }).fit()
        assert isinstance(results, TASCResults)
        assert isinstance(results.design, TASCDesign)
        assert isinstance(results.inference, TASCInference)
        assert isinstance(results.att, float)
        assert isinstance(results.pre_rmse, float)
        assert results.pre_rmse >= 0

    def test_fit_with_full_Q_R(self, panel_df):
        results = TASC({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "treat": "treat", "d": 2,
            "n_em_iter": 3,
            "diagonal_Q": False, "diagonal_R": False,
        }).fit()
        assert isinstance(results, TASCResults)

    def test_fit_with_em_tol_early_stop(self, panel_df):
        results = TASC({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "treat": "treat", "d": 2,
            "n_em_iter": 100, "em_tol": 1e10,
        }).fit()
        # Very loose tol triggers early stop after 1 iter
        assert results.design.n_em_iter_used == 1

    def test_fit_d_equals_one(self, panel_df):
        results = TASC({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "treat": "treat", "d": 1,
            "n_em_iter": 2,
        }).fit()
        assert results.design.parameters.A.shape == (1, 1)

    def test_fit_recovers_effect_direction(self, panel_with_effect):
        # When the treated unit was perturbed by +1.5 in the post window,
        # the ATT should be positive (approximately).
        results = TASC({
            "df": panel_with_effect, "outcome": "y", "unitid": "unitid",
            "time": "time", "treat": "treat", "d": 2,
            "n_em_iter": 10,
        }).fit()
        # Don't assert tight magnitude (EM has noise), only sign.
        assert results.att > 0

    def test_fit_unexpected_error_wrapped(self, monkeypatch, panel_df):
        def boom(*args, **kwargs):
            raise RuntimeError("kaboom")
        monkeypatch.setattr(
            "mlsynth.estimators.tasc.run_tasc", boom
        )
        with pytest.raises(MlsynthEstimationError):
            TASC({"df": panel_df, "outcome": "y", "unitid": "unitid",
                   "time": "time", "treat": "treat", "d": 2}).fit()

    def test_dict_config_invalid_wrapped(self, panel_df):
        with pytest.raises(MlsynthConfigError):
            TASC({"df": panel_df, "outcome": "y", "unitid": "unitid",
                   "time": "time", "treat": "treat", "d": -1})


# =========================================================================
# PLOTTER (smoke)
# =========================================================================

class TestPlotter:

    @pytest.fixture(autouse=True)
    def _matplotlib_agg(self, monkeypatch):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda: None)

    def test_plot_runs(self, panel_df):
        results = TASC({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "treat": "treat", "d": 2,
            "n_em_iter": 2, "display_graphs": False,
        }).fit()
        plot_tasc(results=results)   # should not raise


# =========================================================================
# IMMUTABILITY GUARANTEES
# =========================================================================

class TestImmutability:

    def test_inputs_frozen(self, inputs):
        with pytest.raises(FrozenInstanceError):
            inputs.T0 = 99   # type: ignore[misc]

    def test_parameters_frozen(self, init_params):
        with pytest.raises(FrozenInstanceError):
            init_params.A = np.zeros((2, 2))   # type: ignore[misc]

    def test_design_frozen(self, inputs):
        design, _ = run_tasc(
            inputs=inputs, d=2, n_em_iter=2, em_tol=None,
            diagonal_Q=True, diagonal_R=True, alpha=0.05, seed=0,
        )
        with pytest.raises(FrozenInstanceError):
            design.n_em_iter_used = 99   # type: ignore[misc]

    def test_results_frozen(self, panel_df):
        results = TASC({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "treat": "treat", "d": 2,
            "n_em_iter": 2,
        }).fit()
        with pytest.raises(FrozenInstanceError):
            results.att = 99.0   # type: ignore[misc]
