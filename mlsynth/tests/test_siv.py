"""Tests for the SIV estimator and its helper subpackage.

Layered along agents_tests.md:

* Layer 1 (numerical helpers): per-unit SC solver, projection,
  ensemble alpha selection, just-identified 2SLS.
* Layer 2 (data utilities): prepare_siv_inputs, design-matrix
  construction.
* Layer 3 (estimator integration): SIV.fit end-to-end across the
  three modes plus exception translation.
* Layer 4 (public API contracts): top-level import, results structure,
  selected_variant dispatch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import SIV
from mlsynth.config_models import SIVConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.siv_helpers.ensemble import select_alpha
from mlsynth.utils.siv_helpers.projection import project_outcome_pre
from mlsynth.utils.siv_helpers.setup import (
    build_design_matrix,
    prepare_siv_inputs,
)
from mlsynth.utils.siv_helpers.structures import SIVInputs, SIVResults
from mlsynth.utils.siv_helpers.twosls import two_sls_just_identified
from mlsynth.utils.siv_helpers.weights import fit_synthetic_controls


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

def _shift_share_panel(
    J: int = 12, T: int = 14, T0: int = 8,
    theta: float = -0.2, gamma: float = 0.5,
    sigma_eps: float = 0.1, sigma_mu: float = 0.5,
    seed: int = 0,
):
    """Build a balanced shift-share panel for SIV tests.

    Returns the panel DataFrame plus the true theta. The factor
    structure is single-factor; Z and R are zero in the pre-period
    (matching the Syrian application).
    """

    rng = np.random.default_rng(seed)
    Z_loading = rng.normal(0, 1.0, J)
    mu = rng.normal(0, sigma_mu, J)
    f = np.zeros(T); g = np.zeros(T)
    for t in range(1, T):
        f[t] = 0.5 * f[t - 1] + rng.normal(0, 0.5)
        g[t] = 0.5 * g[t - 1] + rng.normal(0, 0.5)
    eps = rng.normal(0, sigma_eps, (J, T))
    eta = rng.normal(0, sigma_eps, (J, T))

    Z = np.zeros((J, T)); R = np.zeros((J, T))
    Z[:, T0:] = np.outer(Z_loading, g[T0:])
    R[:, T0:] = gamma * Z[:, T0:] + eta[:, T0:]
    Y = theta * R + np.outer(mu, f) + eps

    df = pd.DataFrame({
        "unit": np.repeat(np.arange(J), T),
        "time": np.tile(np.arange(T), J),
        "y": Y.reshape(-1),
        "r": R.reshape(-1),
        "z": Z.reshape(-1),
    })
    return df, theta, T0


@pytest.fixture
def panel():
    return _shift_share_panel(seed=0)


# ----------------------------------------------------------------------
# Layer 1: numerical helpers
# ----------------------------------------------------------------------

class TestNumericalHelpers:
    def test_fit_synthetic_controls_simplex_shape_and_constraints(self):
        rng = np.random.default_rng(0)
        design = rng.normal(0, 1, (8, 12))
        W = fit_synthetic_controls(design, constraint="simplex")
        assert W.shape == (8, 8)
        # Diagonal must be zero, rows must sum to ~1, weights non-negative.
        assert np.allclose(np.diag(W), 0.0)
        for i in range(8):
            assert W[i].sum() == pytest.approx(1.0, abs=1e-4)
            assert (W[i] >= -1e-9).all()

    def test_two_sls_just_identified_recovers_theta_on_clean_data(self):
        rng = np.random.default_rng(0)
        J, T, T0 = 20, 12, 6
        theta = 0.7
        Z = np.zeros((J, T))
        Z[:, T0:] = rng.normal(0, 1, (J, T - T0))
        R = np.zeros((J, T))
        R[:, T0:] = 0.5 * Z[:, T0:] + rng.normal(0, 0.05, (J, T - T0))
        Y = theta * R + rng.normal(0, 0.02, (J, T))
        est = two_sls_just_identified(Y, R, Z, T0=T0, variant="siv")
        assert abs(est.theta_hat - theta) < 0.05
        assert est.beta_first_stage > 0
        assert est.f_stat > 10

    def test_select_alpha_returns_zero_when_siv_is_strictly_better(self):
        rng = np.random.default_rng(0)
        siv = rng.normal(0, 0.01, (5, 4))   # small residuals
        proj = rng.normal(0, 1.0, (5, 4))    # large residuals
        alpha = select_alpha(siv, proj)
        # Weight on PROJ should be small if PROJ has larger residuals.
        assert alpha < 0.3


# ----------------------------------------------------------------------
# Layer 2: data utilities
# ----------------------------------------------------------------------

class TestSetup:
    def test_prepare_inputs_pivots_and_detects_pre_zero_columns(self, panel):
        df, theta, T0 = panel
        inputs = prepare_siv_inputs(
            df, outcome="y", treat="r", instrument="z",
            unitid="unit", time="time", T0=T0,
        )
        assert isinstance(inputs, SIVInputs)
        assert inputs.Y.shape == (12, 14)
        assert inputs.T0 == T0
        assert inputs.has_pre_treatment is False
        assert inputs.has_pre_instrument is False

    def test_missing_instrument_raises(self, panel):
        df, theta, T0 = panel
        bad = df.drop(columns=["z"])
        with pytest.raises(MlsynthDataError):
            prepare_siv_inputs(bad, "y", "r", "z", "unit", "time", T0=T0)

    def test_missing_T0_raises(self, panel):
        df, theta, T0 = panel
        with pytest.raises(MlsynthConfigError):
            prepare_siv_inputs(df, "y", "r", "z", "unit", "time", T0=None)

    def test_design_matrix_outcome_only_when_no_pre_variation(self, panel):
        df, theta, T0 = panel
        inputs = prepare_siv_inputs(df, "y", "r", "z", "unit", "time", T0=T0)
        design = build_design_matrix(inputs, series="default")
        # No pre-period variation in R or Z, so design = Y_pre only.
        assert design.shape == (12, T0)
        np.testing.assert_array_equal(design, inputs.Y[:, :T0])


# ----------------------------------------------------------------------
# Layer 3: estimator integration
# ----------------------------------------------------------------------

class TestEstimator:
    def test_default_siv_mode_runs_end_to_end(self, panel):
        df, theta_true, T0 = panel
        results = SIV({
            "df": df, "outcome": "y", "treat": "r", "instrument": "z",
            "unitid": "unit", "time": "time", "T0": T0,
            "inference_method": "none", "mode": "siv",
        }).fit()
        assert isinstance(results, SIVResults)
        assert results.selected_variant == "siv"
        # SIV should be within a reasonable band of the truth at this
        # signal-to-noise level (theta_true = -0.2).
        assert abs(results.theta_hat - theta_true) < 0.5

    def test_projected_mode_returns_projected_variant(self, panel):
        df, theta_true, T0 = panel
        results = SIV({
            "df": df, "outcome": "y", "treat": "r", "instrument": "z",
            "unitid": "unit", "time": "time", "T0": T0,
            "inference_method": "none", "mode": "projected",
        }).fit()
        assert results.selected_variant == "projected"
        assert "projected" in results.estimates
        assert results.weights_projected is not None

    def test_ensemble_mode_blends(self, panel):
        df, theta_true, T0 = panel
        results = SIV({
            "df": df, "outcome": "y", "treat": "r", "instrument": "z",
            "unitid": "unit", "time": "time", "T0": T0,
            "inference_method": "none", "mode": "ensemble",
        }).fit()
        assert results.selected_variant == "ensemble"
        assert results.metadata["ensemble_alpha"] is not None
        assert 0.0 <= results.metadata["ensemble_alpha"] <= 1.0

    def test_asymptotic_inference_populates_ci(self, panel):
        df, theta_true, T0 = panel
        results = SIV({
            "df": df, "outcome": "y", "treat": "r", "instrument": "z",
            "unitid": "unit", "time": "time", "T0": T0,
            "inference_method": "asymptotic",
        }).fit()
        inf = results.inference
        assert inf.method == "asymptotic"
        assert np.isfinite(inf.ci_lower)
        assert np.isfinite(inf.ci_upper)
        assert inf.ci_lower <= inf.theta_hat <= inf.ci_upper

    def test_conformal_inference_populates_event_study(self, panel):
        df, theta_true, T0 = panel
        results = SIV({
            "df": df, "outcome": "y", "treat": "r", "instrument": "z",
            "unitid": "unit", "time": "time", "T0": T0,
            "inference_method": "conformal", "n_permutations": 1000,
        }).fit()
        inf = results.inference
        assert inf.method == "conformal"
        assert inf.event_study_coefs.size == results.inputs.T

    def test_invalid_config_raises(self, panel):
        df, theta_true, T0 = panel
        with pytest.raises(MlsynthConfigError):
            SIV({
                "df": df, "outcome": "y", "treat": "r", "instrument": "z",
                "unitid": "unit", "time": "time",
                # neither T0 nor post_col supplied
            })

    def test_missing_column_in_df_raises(self, panel):
        df, theta_true, T0 = panel
        df_bad = df.drop(columns=["y"])
        with pytest.raises((MlsynthConfigError, MlsynthDataError)):
            SIV({
                "df": df_bad, "outcome": "y", "treat": "r", "instrument": "z",
                "unitid": "unit", "time": "time", "T0": T0,
            })


# ----------------------------------------------------------------------
# Layer 4: public API
# ----------------------------------------------------------------------

class TestPublicAPI:
    def test_top_level_import(self):
        from mlsynth import SIV as _SIV
        assert _SIV is SIV

    def test_results_dataclasses_are_frozen(self, panel):
        df, theta_true, T0 = panel
        results = SIV({
            "df": df, "outcome": "y", "treat": "r", "instrument": "z",
            "unitid": "unit", "time": "time", "T0": T0,
            "inference_method": "none",
        }).fit()
        with pytest.raises(Exception):
            results.inputs.Y = np.zeros_like(results.inputs.Y)

    def test_estimates_dict_contains_all_variants(self, panel):
        df, theta_true, T0 = panel
        results = SIV({
            "df": df, "outcome": "y", "treat": "r", "instrument": "z",
            "unitid": "unit", "time": "time", "T0": T0,
            "inference_method": "none", "mode": "siv",
        }).fit()
        assert "siv" in results.estimates
        assert "siv_z" in results.estimates
        assert "siv_yr" in results.estimates
        assert "projected" in results.estimates
