import pytest
import numpy as np
import pandas as pd
from collections import defaultdict
from copy import deepcopy
from scipy import stats
from typing import Tuple, List, Optional, Dict, Any, DefaultDict
from unittest.mock import patch
import cvxpy as cp # For mocking SolverError

from mlsynth.utils.sdidutils import (
    estimate_cohort_sdid_effects,
    estimate_event_study_sdid,
    estimate_placebo_variance,
    fit_time_weights,
    compute_regularization,
    unit_weights,
)
from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError, MlsynthEstimationError

# Fixtures
@pytest.fixture
def sample_Y0_pre_donors() -> np.ndarray:
    """Sample donor outcomes in pre-treatment period with varied differences."""
    return np.array([[1, 2, 3], [4, 5, 7], [7, 9, 10], [10, 12, 14]], dtype=float)


@pytest.fixture
def sample_y_pre_mean_treated() -> np.ndarray:
    """Sample mean outcome of treated units in pre-treatment period."""
    return np.array([2, 5, 8, 11], dtype=float)


@pytest.fixture
def sample_Y0_post_donors_mean_across_units(sample_Y0_pre_donors: np.ndarray) -> np.ndarray:
    """Sample mean outcome of each donor unit in post-treatment period."""
    # For simplicity, let's make it similar to the mean of pre-treatment donors
    return sample_Y0_pre_donors.mean(axis=0) + np.array([0.5, -0.5, 0.2])


# Tests for helper functions
def test_compute_regularization_smoke(sample_Y0_pre_donors: np.ndarray):
    """Smoke test for compute_regularization."""
    T_post_treatment_periods = 5
    zeta = compute_regularization(sample_Y0_pre_donors, T_post_treatment_periods)
    assert isinstance(zeta, float)
    assert zeta > 0 # With varied data, zeta should be > 0

def test_compute_regularization_zero_if_std_is_zero():
    """Test compute_regularization returns 0 if std_dev_diffs is 0."""
    Y0_pre_donors_const_diff = np.array([[1,1],[2,2],[3,3]], dtype=float) # diffs are [1,1], std of diffs is 0
    T_post_treatment_periods = 5
    zeta = compute_regularization(Y0_pre_donors_const_diff, T_post_treatment_periods)
    assert isinstance(zeta, float)
    assert np.isclose(zeta, 0.0)

def test_compute_regularization_no_donors(sample_Y0_pre_donors: np.ndarray):
    """Test compute_regularization with no donors."""
    Y0_pre_donors_no_donors = np.empty((sample_Y0_pre_donors.shape[0], 0))
    T_post_treatment_periods = 5
    zeta = compute_regularization(Y0_pre_donors_no_donors, T_post_treatment_periods)
    assert isinstance(zeta, float)
    assert zeta == (T_post_treatment_periods**0.25) * 1.0 # Fallback

def test_compute_regularization_all_nans_diff(sample_Y0_pre_donors: np.ndarray):
    """Test compute_regularization when diffs result in all NaNs."""
    Y0_pre_donors_nans = np.full_like(sample_Y0_pre_donors, np.nan)
    T_post_treatment_periods = 5
    zeta = compute_regularization(Y0_pre_donors_nans, T_post_treatment_periods)
    assert isinstance(zeta, float)
    assert zeta == (T_post_treatment_periods**0.25) * 1.0 # Fallback due to NaN std

def test_compute_regularization_invalid_inputs(sample_Y0_pre_donors: np.ndarray):
    """Test compute_regularization with invalid inputs."""
    with pytest.raises(MlsynthDataError, match="donor_outcomes_pre_treatment must be a NumPy array"):
        compute_regularization([[1,2],[3,4]], 5) # type: ignore
    with pytest.raises(MlsynthDataError, match="donor_outcomes_pre_treatment must be a 2D array"):
        compute_regularization(np.array([1,2,3]), 5)
    with pytest.raises(MlsynthConfigError, match="num_post_treatment_periods must be a non-negative integer"):
        compute_regularization(sample_Y0_pre_donors, -1)
    with pytest.raises(MlsynthConfigError, match="num_post_treatment_periods must be a non-negative integer"):
        compute_regularization(sample_Y0_pre_donors, 5.5) # type: ignore

def test_compute_regularization_no_diff(sample_Y0_pre_donors: np.ndarray):
    """Test compute_regularization with insufficient data for diff."""
    Y0_pre_donors_short = sample_Y0_pre_donors[:1, :]
    T_post_treatment_periods = 5
    zeta = compute_regularization(Y0_pre_donors_short, T_post_treatment_periods)
    assert isinstance(zeta, float)
    # Based on fallback std_dev_diffs = 1.0
    assert zeta == (T_post_treatment_periods**0.25) * 1.0


def test_unit_weights_smoke(sample_Y0_pre_donors: np.ndarray, sample_y_pre_mean_treated: np.ndarray):
    """Smoke test for unit_weights."""
    zeta_reg = 0.1
    intercept, weights = unit_weights(sample_Y0_pre_donors, sample_y_pre_mean_treated, zeta_reg)
    assert intercept is not None
    assert weights is not None
    intercept_val = intercept.item() if isinstance(intercept, np.ndarray) else intercept
    assert isinstance(intercept_val, float)
    assert isinstance(weights, np.ndarray)
    assert weights.shape == (sample_Y0_pre_donors.shape[1],)
    assert np.all(weights >= -1e-6) # Allow for small numerical errors around 0
    assert np.isclose(np.sum(weights), 1.0)

def test_unit_weights_invalid_inputs(sample_Y0_pre_donors: np.ndarray, sample_y_pre_mean_treated: np.ndarray):
    """Test unit_weights with various invalid inputs."""
    zeta_reg = 0.1
    # Invalid types
    with pytest.raises(MlsynthDataError, match="donor_outcomes_pre_treatment must be a NumPy array"):
        unit_weights("not_an_array", sample_y_pre_mean_treated, zeta_reg) # type: ignore
    with pytest.raises(MlsynthDataError, match="mean_treated_outcome_pre_treatment must be a NumPy array"):
        unit_weights(sample_Y0_pre_donors, "not_an_array", zeta_reg) # type: ignore
    with pytest.raises(MlsynthConfigError, match="regularization_parameter_zeta must be a non-negative float or int"):
        unit_weights(sample_Y0_pre_donors, sample_y_pre_mean_treated, "not_a_float") # type: ignore
    with pytest.raises(MlsynthConfigError, match="regularization_parameter_zeta must be a non-negative float or int"):
        unit_weights(sample_Y0_pre_donors, sample_y_pre_mean_treated, -0.1)

    # Invalid ndims
    with pytest.raises(MlsynthDataError, match="donor_outcomes_pre_treatment must be a 2D array"):
        unit_weights(sample_Y0_pre_donors.flatten(), sample_y_pre_mean_treated, zeta_reg)
    with pytest.raises(MlsynthDataError, match="mean_treated_outcome_pre_treatment must be a 1D array"):
        unit_weights(sample_Y0_pre_donors, sample_y_pre_mean_treated.reshape(-1,1), zeta_reg)

    # Zero pre-periods or donors
    with pytest.raises(MlsynthDataError, match="cannot have zero pre-treatment periods"):
        unit_weights(np.empty((0,3)), sample_y_pre_mean_treated[:0], zeta_reg)
    with pytest.raises(MlsynthDataError, match="cannot have zero donors"):
        unit_weights(np.empty((4,0)), sample_y_pre_mean_treated, zeta_reg)

    # Shape mismatch
    with pytest.raises(MlsynthDataError, match="Shape mismatch: donor_outcomes_pre_treatment has"):
        unit_weights(sample_Y0_pre_donors, sample_y_pre_mean_treated[:-1], zeta_reg)

@patch('cvxpy.Problem.solve')
def test_unit_weights_solver_error(mock_solve, sample_Y0_pre_donors: np.ndarray, sample_y_pre_mean_treated: np.ndarray):
    """Test unit_weights raises MlsynthEstimationError on cp.SolverError."""
    mock_solve.side_effect = cp.error.SolverError("CVXPY Test Solver Error")
    with pytest.raises(MlsynthEstimationError, match="CVXPY solver failed in unit_weights"):
        unit_weights(sample_Y0_pre_donors, sample_y_pre_mean_treated, 0.1)

def test_unit_weights_optimization_failure_status(sample_Y0_pre_donors, sample_y_pre_mean_treated):
    """Test unit_weights returns None, None if solver status is not optimal."""
    # This test relies on CVXPY returning a non-optimal status for some input.
    # A more robust way is to mock problem.status after solve.
    # For now, using potentially problematic data (all zeros for donors)
    Y0_pre_donors_zeros = np.zeros_like(sample_Y0_pre_donors)
    zeta_reg = 0.1
    
    # Mock problem.status to be 'infeasible'
    with patch('cvxpy.Problem.solve') as mock_solve_method:
        # Create a dummy problem object to attach the status to
        class MockProblem:
            def __init__(self):
                self.status = 'infeasible_inaccurate' # Or any non-optimal status
            def solve(self, *args, **kwargs):
                # This method will be called by the unit_weights function
                pass # Do nothing, status is already set
        
        # When cp.Problem is instantiated, return our mock problem
        with patch('cvxpy.Problem', return_value=MockProblem()) as mock_problem_init:
            intercept, weights = unit_weights(Y0_pre_donors_zeros, sample_y_pre_mean_treated, zeta_reg)
            assert intercept is None
            assert weights is None


def test_fit_time_weights_smoke(sample_Y0_pre_donors: np.ndarray, sample_Y0_post_donors_mean_across_units: np.ndarray):
    """Smoke test for fit_time_weights."""
    intercept, weights = fit_time_weights(sample_Y0_pre_donors, sample_Y0_post_donors_mean_across_units)
    assert intercept is not None
    assert weights is not None
    intercept_val = intercept.item() if isinstance(intercept, np.ndarray) else intercept
    assert isinstance(intercept_val, float)
    assert isinstance(weights, np.ndarray)
    assert weights.shape == (sample_Y0_pre_donors.shape[0],)
    assert np.all(weights >= -1e-6) # Allow for small numerical errors
    assert np.isclose(np.sum(weights), 1.0)

def test_fit_time_weights_invalid_inputs(sample_Y0_pre_donors: np.ndarray, sample_Y0_post_donors_mean_across_units: np.ndarray):
    """Test fit_time_weights with various invalid inputs."""
    # Invalid types
    with pytest.raises(MlsynthDataError, match="donor_outcomes_pre_treatment must be a NumPy array"):
        fit_time_weights("not_an_array", sample_Y0_post_donors_mean_across_units) # type: ignore
    with pytest.raises(MlsynthDataError, match="mean_donor_outcomes_post_treatment must be a NumPy array"):
        fit_time_weights(sample_Y0_pre_donors, "not_an_array") # type: ignore

    # Invalid ndims
    with pytest.raises(MlsynthDataError, match="donor_outcomes_pre_treatment must be a 2D array"):
        fit_time_weights(sample_Y0_pre_donors.flatten(), sample_Y0_post_donors_mean_across_units)
    with pytest.raises(MlsynthDataError, match="mean_donor_outcomes_post_treatment must be a 1D array"):
        fit_time_weights(sample_Y0_pre_donors, sample_Y0_post_donors_mean_across_units.reshape(-1,1))

    # Zero pre-periods or donors
    with pytest.raises(MlsynthDataError, match="cannot have zero pre-treatment periods"):
        fit_time_weights(np.empty((0,3)), sample_Y0_post_donors_mean_across_units)
    with pytest.raises(MlsynthDataError, match="cannot have zero donors if mean_donor_outcomes_post_treatment has donors"):
        fit_time_weights(np.empty((4,0)), sample_Y0_post_donors_mean_across_units)
    
    # Shape mismatch
    with pytest.raises(MlsynthDataError, match="Shape mismatch: donor_outcomes_pre_treatment has"):
        fit_time_weights(sample_Y0_pre_donors, sample_Y0_post_donors_mean_across_units[:-1])

@patch('cvxpy.Problem.solve')
def test_fit_time_weights_solver_error(mock_solve, sample_Y0_pre_donors: np.ndarray, sample_Y0_post_donors_mean_across_units: np.ndarray):
    """Test fit_time_weights raises MlsynthEstimationError on cp.SolverError."""
    mock_solve.side_effect = cp.error.SolverError("CVXPY Test Solver Error")
    with pytest.raises(MlsynthEstimationError, match="CVXPY solver failed in fit_time_weights"):
        fit_time_weights(sample_Y0_pre_donors, sample_Y0_post_donors_mean_across_units)

# More tests for estimate_cohort_sdid_effects
def test_estimate_cohort_sdid_effects_no_pre_periods(sample_cohort_data: Dict[str, Any]):
    """Test estimate_cohort_sdid_effects with no pre-treatment periods."""
    data = deepcopy(sample_cohort_data)
    data["pre_periods"] = 0
    # Y0_pre_donors will be empty.
    # unit_weights and fit_time_weights will be called with empty arrays if not caught earlier.
    # The function should handle this by returning NaNs.
    # compute_regularization will use fallback.
    # unit_weights will be called with donor_outcomes_pre_treatment_cohort of shape (0, N_donors)
    # This should lead to NaNs in weights and subsequently in results.
    
    a = 1 # Adoption at the first period
    pooled_tau_ell: DefaultDict[float, List[Tuple[int, float]]] = defaultdict(list)
    
    results = estimate_cohort_sdid_effects(a, data, pooled_tau_ell)
    assert np.isnan(results["att"])
    assert np.all(np.isnan(results["treatment_effects_series"]))
    assert np.all(np.isnan(results["counterfactual"]))
    assert np.all(np.isnan(results["fitted_counterfactual"]))


def test_estimate_cohort_sdid_effects_no_post_periods(sample_cohort_data: Dict[str, Any]):
    """Test estimate_cohort_sdid_effects with no post-treatment periods."""
    data = deepcopy(sample_cohort_data)
    data["post_periods"] = 0
    data["total_periods"] = data["pre_periods"]
    data["y"] = data["y"][:data["pre_periods"], :]
    data["donor_matrix"] = data["donor_matrix"][:data["pre_periods"], :]
    
    a = data["pre_periods"] + 1 # Adoption period (conceptually, though no post data)
    pooled_tau_ell: DefaultDict[float, List[Tuple[int, float]]] = defaultdict(list)
    
    results = estimate_cohort_sdid_effects(a, data, pooled_tau_ell)
    assert np.isnan(results["att"]) # No post periods to average
    assert results["post_effects"].size == 0 # No post_effects
    assert len(results["treatment_effects_series"]) == data["total_periods"]
    # Effects in pre-period might still be calculated based on weights from pre-period data
    # but overall ATT should be NaN.
    # fit_time_weights will not be called with post_periods = 0, so optimal_time_weights_vector remains None.
    # This leads to bias_correction being NaN.

def test_estimate_cohort_sdid_effects_invalid_inputs(sample_cohort_data: Dict[str, Any]):
    """Test estimate_cohort_sdid_effects with various invalid inputs."""
    a = sample_cohort_data["pre_periods"] + 1
    pooled_tau_ell: DefaultDict[float, List[Tuple[int, float]]] = defaultdict(list)
    valid_data = deepcopy(sample_cohort_data)

    with pytest.raises(MlsynthConfigError, match="cohort_adoption_period must be an integer"):
        estimate_cohort_sdid_effects("not_int", valid_data, pooled_tau_ell) # type: ignore
    with pytest.raises(MlsynthDataError, match="cohort_data_dict must be a dictionary"):
        estimate_cohort_sdid_effects(a, "not_dict", pooled_tau_ell) # type: ignore
    
    bad_data_missing_key = deepcopy(valid_data)
    del bad_data_missing_key["y"]
    with pytest.raises(MlsynthDataError, match="Missing required key 'y'"):
        estimate_cohort_sdid_effects(a, bad_data_missing_key, pooled_tau_ell)

    bad_data_wrong_type = deepcopy(valid_data)
    bad_data_wrong_type["total_periods"] = "not_int" # type: ignore
    with pytest.raises(MlsynthDataError, match="'total_periods' must be a positive integer"):
        estimate_cohort_sdid_effects(a, bad_data_wrong_type, pooled_tau_ell)

    bad_data_shape_mismatch = deepcopy(valid_data)
    bad_data_shape_mismatch["y"] = bad_data_shape_mismatch["y"][:-1,:] # Mismatch with total_periods
    with pytest.raises(MlsynthDataError, match="Shape mismatch: 'y' has"):
        estimate_cohort_sdid_effects(a, bad_data_shape_mismatch, pooled_tau_ell)
    
    bad_data_treated_indices = deepcopy(valid_data)
    bad_data_treated_indices["treated_indices"] = [0] # Mismatch with y.shape[1]
    with pytest.raises(MlsynthDataError, match="Shape mismatch: 'y' has 2 treated units"):
        estimate_cohort_sdid_effects(a, bad_data_treated_indices, pooled_tau_ell)

    bad_data_negative_periods = deepcopy(valid_data)
    bad_data_negative_periods["pre_periods"] = -1
    with pytest.raises(MlsynthDataError, match="'pre_periods' must be a non-negative integer"):
        estimate_cohort_sdid_effects(a, bad_data_negative_periods, pooled_tau_ell)

    # Test warning for pre_periods + post_periods > total_periods
    warn_data = deepcopy(valid_data)
    warn_data["post_periods"] = warn_data["total_periods"] # Makes sum > total_periods
    with pytest.warns(UserWarning, match="Sum of pre_periods"):
        estimate_cohort_sdid_effects(a, warn_data, pooled_tau_ell)


@pytest.fixture
def sample_cohort_data(sample_Y0_pre_donors: np.ndarray, sample_y_pre_mean_treated: np.ndarray) -> Dict[str, Any]:
    """Sample data for a single cohort."""
    T_total = 8
    T0_pre = sample_Y0_pre_donors.shape[0] # Should be 4 from fixture
    T_post = T_total - T0_pre # Should be 4

    # Create dummy post-treatment data
    y_post_mean_treated_flat = np.array([13, 15, 14, 16], dtype=float)
    
    # Reshape for broadcasting: (T0_pre, 1) or (T_post, 1)
    sample_y_pre_mean_treated_col = sample_y_pre_mean_treated.reshape(-1, 1)
    y_post_mean_treated_col = y_post_mean_treated_flat.reshape(-1, 1)

    # Create y_mat_treated for pre-period (T0_pre, N_treated_in_cohort=2)
    y_mat_treated_pre = sample_y_pre_mean_treated_col + np.random.normal(0, 0.1, (T0_pre, 2))
    
    # Create y_mat_treated for post-period (T_post, N_treated_in_cohort=2)
    y_mat_treated_post = y_post_mean_treated_col + np.random.normal(0, 0.1, (T_post, 2))

    y_mat_treated_full = np.vstack((y_mat_treated_pre, y_mat_treated_post))


    Y0_post_donors = np.array([[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]], dtype=float)
    full_donor_matrix = np.vstack((sample_Y0_pre_donors, Y0_post_donors))

    return {
        "y": y_mat_treated_full, # (T_total, N_treated_in_cohort)
        "donor_matrix": full_donor_matrix, # (T_total, N_donors)
        "total_periods": T_total,
        "pre_periods": T0_pre,
        "post_periods": T_post,
        "treated_indices": [0, 1], # Original indices of treated units
    }


def test_estimate_cohort_sdid_effects_smoke(sample_cohort_data: Dict[str, Any]):
    """Smoke test for estimate_cohort_sdid_effects."""
    a = sample_cohort_data["pre_periods"] + 1 # Adoption period
    pooled_tau_ell: DefaultDict[float, List[Tuple[int, float]]] = defaultdict(list)

    results = estimate_cohort_sdid_effects(a, sample_cohort_data, pooled_tau_ell)

    assert isinstance(results, dict)
    expected_keys = [
        "effects", "pre_effects", "post_effects", "actual", "counterfactual",
        "fitted_counterfactual", "att", "treatment_effects_series", "ell"
    ]
    for key in expected_keys:
        assert key in results

    T_total = sample_cohort_data["total_periods"]
    assert results["effects"].shape == (T_total, 2)
    assert results["actual"].shape == (T_total,)
    assert results["counterfactual"].shape == (T_total,)
    assert results["fitted_counterfactual"].shape == (T_total,)
    assert results["treatment_effects_series"].shape == (T_total,)
    assert results["ell"].shape == (T_total,)
    assert isinstance(results["att"], float)

    # Check pooled_tau_ell was updated
    assert len(pooled_tau_ell) > 0
    # Check that effects are added to pooled_tau_ell only if not NaN
    has_valid_effect = False
    for _, effect_val_list in pooled_tau_ell.items():
        for _, effect_val_single in effect_val_list:
            if not np.isnan(effect_val_single):
                has_valid_effect = True
                break
        if has_valid_effect:
            break
    
    # If all effects were NaN (e.g. due to no pre_periods), pooled_tau_ell might be empty
    # or contain only NaNs which are filtered out.
    # This part of the test needs to be robust to that.
    # If unit_weights or time_weights failed and returned None, then effects would be NaN.
    # The current smoke test data should produce valid weights.
    assert has_valid_effect, "No valid (non-NaN) effects were added to pooled_tau_ell"

    for ell_val, effect_list in pooled_tau_ell.items():
        assert isinstance(ell_val, float) # np.float64 is fine
        assert isinstance(effect_list, list)
        for item in effect_list:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], int) # num_treated_in_cohort
            assert isinstance(item[1], float) # effect_value

    # Check pre_effects and post_effects structure
    if results["pre_effects"].size > 0:
        assert results["pre_effects"].ndim == 2
        assert results["pre_effects"].shape[1] == 2
    if results["post_effects"].size > 0:
        assert results["post_effects"].ndim == 2
        assert results["post_effects"].shape[1] == 2


@pytest.fixture
def sample_prepped_data(sample_cohort_data: Dict[str, Any]) -> Dict[str, Any]:
    """Sample prepped data with multiple cohorts for event study functions."""
    # Create a second cohort by slightly modifying the first one
    cohort_data_2 = deepcopy(sample_cohort_data)
    cohort_data_2["y"] = sample_cohort_data["y"] * 1.1 # Slightly different outcomes
    cohort_data_2["treated_indices"] = [2, 3] # Different treated units

    # Assume cohort 1 adopts at period 5 (index 4), cohort 2 at period 6 (index 5)
    # pre_periods is 4 for sample_cohort_data
    adoption_period_cohort1 = sample_cohort_data["pre_periods"] + 1
    adoption_period_cohort2 = sample_cohort_data["pre_periods"] + 2

    return {
        "cohorts": {
            adoption_period_cohort1: sample_cohort_data,
            adoption_period_cohort2: cohort_data_2,
        },
        # Add other keys if dataprep_event_study_sdid typically adds them,
        # for now, focusing on what's directly used by the tested functions.
    }


def test_estimate_placebo_variance_smoke(sample_prepped_data: Dict[str, Any]):
    """Smoke test for estimate_placebo_variance."""
    B_iterations = 2 # Small number for a smoke test
    seed = 123
    results: Optional[Dict[str, Any]] = None # Initialize results
    
    # The current sample_prepped_data fixture (3 unique donors, 2+2 treated units)
    # is expected to trigger the warning.
    with pytest.warns(UserWarning, match="Placebo inference might be unreliable"):
        results = estimate_placebo_variance(sample_prepped_data, B_iterations, seed)
    
    assert results is not None, "Results dictionary should not be None"
    assert isinstance(results, dict)
    expected_keys = [
        "att_variance", "cohort_variances", "event_variances", "pooled_event_variances"
    ]
    for key in expected_keys:
        assert key in results

    assert isinstance(results["att_variance"], float) # Can be nan if B_iterations is too small
    assert isinstance(results["cohort_variances"], dict)
    assert isinstance(results["event_variances"], dict)
    assert isinstance(results["pooled_event_variances"], dict)

    # Check structure of nested dicts if they are not empty
    if results["cohort_variances"]:
        first_cohort_key = list(results["cohort_variances"].keys())[0]
        assert isinstance(results["cohort_variances"][first_cohort_key], float)

    if results["event_variances"]:
        first_cohort_key = list(results["event_variances"].keys())[0]
        assert isinstance(results["event_variances"][first_cohort_key], dict)
        if results["event_variances"][first_cohort_key]:
            first_event_key = list(results["event_variances"][first_cohort_key].keys())[0]
            assert isinstance(results["event_variances"][first_cohort_key][first_event_key], float)
    
    if results["pooled_event_variances"]:
        first_event_key = list(results["pooled_event_variances"].keys())[0]
        assert isinstance(results["pooled_event_variances"][first_event_key], float)


def test_estimate_event_study_sdid_smoke(sample_prepped_data: Dict[str, Any]):
    """Smoke test for estimate_event_study_sdid."""
    placebo_iterations = 2 # Small number for smoke test
    seed = 456
    results: Optional[Dict[str, Any]] = None # Initialize results

    # The current sample_prepped_data fixture (3 unique donors, 2+2 treated units)
    # is expected to trigger the warning.
    with pytest.warns(UserWarning, match="Placebo inference might be unreliable"):
        results = estimate_event_study_sdid(sample_prepped_data, placebo_iterations, seed)

    assert results is not None, "Results dictionary should not be None"
    assert isinstance(results, dict)
    expected_keys = [
        "tau_a_ell", "tau_ell", "att", "att_se", "att_ci",
        "cohort_estimates", "pooled_estimates"
    ]
    for key in expected_keys:
        assert key in results

    assert isinstance(results["tau_a_ell"], dict)
    assert isinstance(results["tau_ell"], dict)
    assert isinstance(results["att"], float) # Can be NaN
    assert isinstance(results["att_se"], float) # can be nan
    assert isinstance(results["att_ci"], list)
    assert len(results["att_ci"]) == 2 # Elements can be NaN
    assert isinstance(results["cohort_estimates"], dict)
    assert isinstance(results["pooled_estimates"], dict)
    assert "placebo_att_values" in results # Check new key
    assert isinstance(results["placebo_att_values"], list) or results["placebo_att_values"] is None


    # Check structure of cohort_estimates
    if results["cohort_estimates"]:
        first_cohort_key = list(results["cohort_estimates"].keys())[0]
        cohort_est = results["cohort_estimates"][first_cohort_key]
        assert "att" in cohort_est
        assert "att_se" in cohort_est
        assert "att_ci" in cohort_est
        assert "event_estimates" in cohort_est
        assert isinstance(cohort_est["event_estimates"], dict)
        if cohort_est["event_estimates"]:
            first_event_key = list(cohort_est["event_estimates"].keys())[0]
            event_detail = cohort_est["event_estimates"][first_event_key]
            assert "tau" in event_detail
            assert "se" in event_detail
            assert "ci" in event_detail

    # Check structure of pooled_estimates
    if results["pooled_estimates"]:
        first_event_key = list(results["pooled_estimates"].keys())[0]
        pooled_est_detail = results["pooled_estimates"][first_event_key]
        assert "tau" in pooled_est_detail
        assert "se" in pooled_est_detail
        assert "ci" in pooled_est_detail

def test_estimate_event_study_sdid_invalid_inputs(sample_prepped_data: Dict[str, Any]):
    """Test estimate_event_study_sdid with invalid inputs."""
    valid_data = deepcopy(sample_prepped_data)
    with pytest.raises(MlsynthDataError, match="prepped_event_study_data must be a dict with a 'cohorts' key"):
        estimate_event_study_sdid("not_dict", 2, 123) # type: ignore
    
    bad_data_no_cohorts = deepcopy(valid_data)
    del bad_data_no_cohorts["cohorts"]
    with pytest.raises(MlsynthDataError, match="prepped_event_study_data must be a dict with a 'cohorts' key"):
        estimate_event_study_sdid(bad_data_no_cohorts, 2, 123)

    bad_data_cohorts_not_dict = deepcopy(valid_data)
    bad_data_cohorts_not_dict["cohorts"] = "not_a_dict" # type: ignore
    with pytest.raises(MlsynthDataError, match="'cohorts' in prepped_event_study_data must be a dictionary"):
        estimate_event_study_sdid(bad_data_cohorts_not_dict, 2, 123)

    with pytest.raises(MlsynthConfigError, match="placebo_iterations must be a non-negative integer"):
        estimate_event_study_sdid(valid_data, -1, 123)
    with pytest.raises(MlsynthConfigError, match="seed must be an integer"):
        estimate_event_study_sdid(valid_data, 2, "not_int") # type: ignore

    bad_data_cohort_key_not_int = deepcopy(valid_data)
    # Make a cohort key a string
    first_key = list(bad_data_cohort_key_not_int["cohorts"].keys())[0]
    bad_data_cohort_key_not_int["cohorts"][str(first_key)] = bad_data_cohort_key_not_int["cohorts"].pop(first_key)
    with pytest.raises(MlsynthDataError, match="Cohort key .* must be an integer"):
        estimate_event_study_sdid(bad_data_cohort_key_not_int, 2, 123)


def test_estimate_placebo_variance_invalid_inputs(sample_prepped_data: Dict[str, Any]):
    """Test estimate_placebo_variance with invalid inputs."""
    valid_data = deepcopy(sample_prepped_data)
    with pytest.raises(MlsynthDataError, match="prepped_event_study_data must be a dict with a 'cohorts' key"):
        estimate_placebo_variance("not_dict", 2, 123) # type: ignore
    # Other data validation is similar to estimate_event_study_sdid and covered there or in estimate_cohort_sdid_effects

    with pytest.raises(MlsynthConfigError, match="num_placebo_iterations must be a non-negative integer"):
        estimate_placebo_variance(valid_data, -1, 123)
    with pytest.raises(MlsynthConfigError, match="seed must be an integer"):
        estimate_placebo_variance(valid_data, 2, "not_int") # type: ignore
