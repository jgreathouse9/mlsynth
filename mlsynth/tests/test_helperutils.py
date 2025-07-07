import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from mlsynth.utils.helperutils import (
    prenorm,
    ssdid_w,
    ssdid_lambda,
    ssdid_est,
    sc_diagplot,
)
from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError, MlsynthEstimationError # Import custom exceptions
import cvxpy # For cvxpy.error.SolverError
from matplotlib import pyplot as plt # For mocking

# --- Fixtures ---

@pytest.fixture
def ssdid_sample_data() -> dict:
    """Provides sample data for SSDID related function tests."""
    T = 10  # Total time periods
    J = 3   # Number of donor units
    a = 5   # Number of pre-treatment periods for matching/lambda
    k_horizon = 2 # Horizon for optimization or estimation
    eta = 0.1 # Regularization parameter

    # Ensure reproducibility and simple values
    np.random.seed(0)
    treated_y = np.arange(T, dtype=float) * 2.0 # Simple linear trend
    # donor_matrix with some variation
    donor_matrix = np.array([np.arange(T) + i*0.5 for i in range(J)], dtype=float).T + np.random.rand(T, J) * 0.1

    return {
        "treated_y": treated_y,
        "donor_matrix": donor_matrix,
        "a": a,
        "k_horizon": k_horizon,
        "eta": eta,
        "J": J,
    }

# --- Tests for prenorm (existing) ---

def test_prenorm_vector(): # Existing test
    x = np.array([10, 20, 40])
    result = prenorm(x)
    expected = x / 40 * 100
    np.testing.assert_allclose(result, expected)

def test_prenorm_matrix(): # Existing test
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 10]])
    result = prenorm(X)
    expected = X / np.array([5, 10]) * 100
    np.testing.assert_allclose(result, expected)

def test_prenorm_custom_target(): # Existing test
    x = np.array([5, 10])
    result = prenorm(x, target=200)
    expected = x / 10 * 200
    np.testing.assert_allclose(result, expected)

def test_prenorm_list_input(): # Existing test
    x = [2, 4, 8]
    result = prenorm(x)
    expected = np.array(x) / 8 * 100
    np.testing.assert_allclose(result, expected)

def test_prenorm_division_by_zero(): # Existing test
    x = np.array([1, 0])
    with pytest.raises(MlsynthDataError, match="Division by zero: Denominator for normalization is zero."):
        prenorm(x)

# --- Tests for ssdid_w ---

def test_ssdid_w_smoke(ssdid_sample_data):
    """Smoke test for ssdid_w function."""
    data = ssdid_sample_data
    omega, omega_0 = ssdid_w(
        treated_unit_outcomes_all_periods=data["treated_y"],
        donor_units_outcomes_all_periods=data["donor_matrix"],
        num_matching_pre_periods=data["a"],
        matching_horizon_offset=data["k_horizon"],
        l2_penalty_regularization_strength=data["eta"],
    )
    assert isinstance(omega, np.ndarray), "omega should be a numpy array"
    assert omega.shape == (data["J"],), f"omega shape mismatch, expected ({data['J']},)"
    assert isinstance(omega_0, (float, np.floating)), "omega_0 should be a float"
    assert np.isclose(np.sum(omega), 1.0), "omega weights should sum to 1"


def test_ssdid_w_invalid_input_types(ssdid_sample_data):
    data = ssdid_sample_data
    with pytest.raises(MlsynthDataError, match="`treated_unit_outcomes_all_periods` must be a 1D NumPy array."):
        ssdid_w([1,2,3], data["donor_matrix"], data["a"], data["k_horizon"], data["eta"])
    with pytest.raises(MlsynthDataError, match="`donor_units_outcomes_all_periods` must be a 2D NumPy array."):
        ssdid_w(data["treated_y"], [1,2,3], data["a"], data["k_horizon"], data["eta"])
    with pytest.raises(MlsynthDataError, match="`donor_prior_weights_for_penalty` must be a 1D NumPy array if provided"):
        ssdid_w(data["treated_y"], data["donor_matrix"], data["a"], data["k_horizon"], data["eta"], donor_prior_weights_for_penalty=[1,2])

def test_ssdid_w_period_mismatch(ssdid_sample_data):
    data = ssdid_sample_data
    with pytest.raises(MlsynthDataError, match="Mismatch in total periods"):
        ssdid_w(data["treated_y"][:-1], data["donor_matrix"], data["a"], data["k_horizon"], data["eta"])

def test_ssdid_w_invalid_config_params(ssdid_sample_data):
    data = ssdid_sample_data
    with pytest.raises(MlsynthConfigError, match="`l2_penalty_regularization_strength` cannot be negative."):
        ssdid_w(data["treated_y"], data["donor_matrix"], data["a"], data["k_horizon"], -0.1)
    with pytest.raises(MlsynthConfigError, match="`num_matching_pre_periods` cannot be negative."):
        ssdid_w(data["treated_y"], data["donor_matrix"], -1, data["k_horizon"], data["eta"])
    with pytest.raises(MlsynthConfigError, match="`matching_horizon_offset` cannot be negative."):
        ssdid_w(data["treated_y"], data["donor_matrix"], data["a"], -1, data["eta"])
    with pytest.raises(MlsynthConfigError, match="`matching_period_end_index` .* is out of bounds"): # Too large
        ssdid_w(data["treated_y"], data["donor_matrix"], len(data["treated_y"]), data["k_horizon"] + 1, data["eta"])

def test_ssdid_w_invalid_pi_shape(ssdid_sample_data):
    data = ssdid_sample_data
    pi_wrong_shape = np.random.rand(data["J"] + 1)
    with pytest.raises(MlsynthDataError, match="Length of `donor_prior_weights_for_penalty` .* does not match number of donor units"):
        ssdid_w(data["treated_y"], data["donor_matrix"], data["a"], data["k_horizon"], data["eta"], donor_prior_weights_for_penalty=pi_wrong_shape)

@patch("cvxpy.Problem.solve")
def test_ssdid_w_solver_error(mock_solve, ssdid_sample_data):
    data = ssdid_sample_data
    mock_solve.side_effect = cvxpy.error.SolverError("Solver failed")
    with pytest.raises(MlsynthEstimationError, match="CVXPY solver failed in ssdid_w: Solver failed"):
        ssdid_w(data["treated_y"], data["donor_matrix"], data["a"], data["k_horizon"], data["eta"])

@patch.object(cvxpy.Problem, "solve", autospec=True)
def test_ssdid_w_solver_no_solution(mock_problem_solve_method, ssdid_sample_data):
    data = ssdid_sample_data

    def solve_side_effect(problem_instance, *args, **kwargs):
        # Access variables from the problem instance and set their .value to None
        for var in problem_instance.variables():
            var.value = None
        return None # Simulate solve completing but finding no optimal value

    mock_problem_solve_method.side_effect = solve_side_effect

    with pytest.raises(MlsynthEstimationError, match="CVXPY optimization failed to find a solution for donor weights in ssdid_w."):
        ssdid_w(data["treated_y"], data["donor_matrix"], data["a"], data["k_horizon"], data["eta"])


def test_ssdid_w_with_pi(ssdid_sample_data):
    """Test ssdid_w function with custom pi."""
    data = ssdid_sample_data
    pi_custom = np.random.rand(data["J"])
    pi_custom /= np.sum(pi_custom)  # Normalize pi

    omega, omega_0 = ssdid_w(
        treated_unit_outcomes_all_periods=data["treated_y"],
        donor_units_outcomes_all_periods=data["donor_matrix"],
        num_matching_pre_periods=data["a"],
        matching_horizon_offset=data["k_horizon"],
        l2_penalty_regularization_strength=data["eta"],
        donor_prior_weights_for_penalty=pi_custom,
    )
    assert isinstance(omega, np.ndarray), "omega should be a numpy array"
    assert omega.shape == (data["J"],), f"omega shape mismatch, expected ({data['J']},)"
    assert isinstance(omega_0, (float, np.floating)), "omega_0 should be a float"
    assert np.isclose(np.sum(omega), 1.0), "omega weights should sum to 1"

# --- Tests for ssdid_lambda ---

def test_ssdid_lambda_smoke(ssdid_sample_data):
    """Smoke test for ssdid_lambda function."""
    data = ssdid_sample_data
    lambda_val, lambda_0_val = ssdid_lambda(
        treated_unit_outcomes_all_periods=data["treated_y"],
        donor_units_outcomes_all_periods=data["donor_matrix"],
        num_pre_treatment_periods_for_lambda=data["a"],
        post_treatment_horizon_offset=data["k_horizon"],
        l2_penalty_regularization_strength=data["eta"],
    )
    assert isinstance(lambda_val, np.ndarray), "lambda_val should be a numpy array"
    assert lambda_val.shape == (data["a"],), f"lambda_val shape mismatch, expected ({data['a']},)"
    assert isinstance(lambda_0_val, (float, np.floating)), "lambda_0_val should be a float"
    assert np.isclose(np.sum(lambda_val), 1.0), "lambda weights should sum to 1"


def test_ssdid_lambda_invalid_input_types(ssdid_sample_data):
    data = ssdid_sample_data
    with pytest.raises(MlsynthDataError, match="`treated_unit_outcomes_all_periods` must be a 1D NumPy array."):
        ssdid_lambda([1,2,3], data["donor_matrix"], data["a"], data["k_horizon"], data["eta"])
    with pytest.raises(MlsynthDataError, match="`donor_units_outcomes_all_periods` must be a 2D NumPy array."):
        ssdid_lambda(data["treated_y"], [1,2,3], data["a"], data["k_horizon"], data["eta"])

def test_ssdid_lambda_period_mismatch(ssdid_sample_data):
    data = ssdid_sample_data
    with pytest.raises(MlsynthDataError, match="Mismatch in total periods"):
        ssdid_lambda(data["treated_y"][:-1], data["donor_matrix"], data["a"], data["k_horizon"], data["eta"])

def test_ssdid_lambda_invalid_config_params(ssdid_sample_data):
    data = ssdid_sample_data
    with pytest.raises(MlsynthConfigError, match="`l2_penalty_regularization_strength` cannot be negative."):
        ssdid_lambda(data["treated_y"], data["donor_matrix"], data["a"], data["k_horizon"], -0.1)
    with pytest.raises(MlsynthConfigError, match="`num_pre_treatment_periods_for_lambda` cannot be negative."):
        ssdid_lambda(data["treated_y"], data["donor_matrix"], -1, data["k_horizon"], data["eta"])
    with pytest.raises(MlsynthConfigError, match="`post_treatment_horizon_offset` cannot be negative."):
        ssdid_lambda(data["treated_y"], data["donor_matrix"], data["a"], -1, data["eta"])
    with pytest.raises(MlsynthConfigError, match="`target_donor_period_index` .* is out of bounds"): # Too large
        ssdid_lambda(data["treated_y"], data["donor_matrix"], data["a"], len(data["treated_y"]), data["eta"])
    with pytest.raises(MlsynthConfigError, match="`num_pre_treatment_periods_for_lambda` .* is out of bounds"): # Too large
        ssdid_lambda(data["treated_y"], data["donor_matrix"], len(data["treated_y"]) +1 , 0, data["eta"])


@patch("cvxpy.Problem.solve")
def test_ssdid_lambda_solver_error(mock_solve, ssdid_sample_data):
    data = ssdid_sample_data
    mock_solve.side_effect = cvxpy.error.SolverError("Solver failed")
    with pytest.raises(MlsynthEstimationError, match="CVXPY solver failed in ssdid_lambda: Solver failed"):
        ssdid_lambda(data["treated_y"], data["donor_matrix"], data["a"], data["k_horizon"], data["eta"])

@patch.object(cvxpy.Problem, "solve", autospec=True)
def test_ssdid_lambda_solver_no_solution(mock_problem_solve_method, ssdid_sample_data):
    data = ssdid_sample_data

    def solve_side_effect(problem_instance, *args, **kwargs):
        for var in problem_instance.variables():
            var.value = None
        return None

    mock_problem_solve_method.side_effect = solve_side_effect
    
    with pytest.raises(MlsynthEstimationError, match="CVXPY optimization failed to find a solution for time weights in ssdid_lambda."):
        ssdid_lambda(data["treated_y"], data["donor_matrix"], data["a"], data["k_horizon"], data["eta"])


# --- Tests for ssdid_est ---

def test_ssdid_est_smoke(ssdid_sample_data):
    """Smoke test for ssdid_est function."""
    data = ssdid_sample_data
    # Get some valid omega and lambda_vec first
    omega, _ = ssdid_w(
        treated_unit_outcomes_all_periods=data["treated_y"],
        donor_units_outcomes_all_periods=data["donor_matrix"],
        num_matching_pre_periods=data["a"],
        matching_horizon_offset=data["k_horizon"],
        l2_penalty_regularization_strength=data["eta"],
    )
    lambda_vec, _ = ssdid_lambda(
        treated_unit_outcomes_all_periods=data["treated_y"],
        donor_units_outcomes_all_periods=data["donor_matrix"],
        num_pre_treatment_periods_for_lambda=data["a"],
        post_treatment_horizon_offset=data["k_horizon"],
        l2_penalty_regularization_strength=data["eta"],
    )

    estimate = ssdid_est(
        treated_unit_outcomes_all_periods=data["treated_y"],
        donor_units_outcomes_all_periods=data["donor_matrix"],
        donor_weights=omega,
        time_weights_vector=lambda_vec,
        num_pre_treatment_periods=data["a"],
        post_treatment_horizon_offset=data["k_horizon"],
    )
    assert isinstance(estimate, (float, np.floating)), "Estimate should be a float"


# --- Tests for ssdid_est input validation ---

def test_ssdid_est_invalid_treated_y_type(ssdid_sample_data):
    data = ssdid_sample_data
    with pytest.raises(MlsynthDataError, match="`treated_unit_outcomes_all_periods` must be a 1D NumPy array."):
        ssdid_est(
            treated_unit_outcomes_all_periods=[1, 2, 3], # Not a numpy array
            donor_units_outcomes_all_periods=data["donor_matrix"],
            donor_weights=np.random.rand(data["J"]),
            time_weights_vector=np.random.rand(data["a"]),
            num_pre_treatment_periods=data["a"],
            post_treatment_horizon_offset=data["k_horizon"],
        )

def test_ssdid_est_invalid_treated_y_ndim(ssdid_sample_data):
    data = ssdid_sample_data
    with pytest.raises(MlsynthDataError, match="`treated_unit_outcomes_all_periods` must be a 1D NumPy array."):
        ssdid_est(
            treated_unit_outcomes_all_periods=np.array([[1,2],[3,4]]), # 2D array
            donor_units_outcomes_all_periods=data["donor_matrix"],
            donor_weights=np.random.rand(data["J"]),
            time_weights_vector=np.random.rand(data["a"]),
            num_pre_treatment_periods=data["a"],
            post_treatment_horizon_offset=data["k_horizon"],
        )

def test_ssdid_est_invalid_donor_matrix_type(ssdid_sample_data):
    data = ssdid_sample_data
    with pytest.raises(MlsynthDataError, match="`donor_units_outcomes_all_periods` must be a 2D NumPy array."):
        ssdid_est(
            treated_unit_outcomes_all_periods=data["treated_y"],
            donor_units_outcomes_all_periods="not_an_array",
            donor_weights=np.random.rand(data["J"]),
            time_weights_vector=np.random.rand(data["a"]),
            num_pre_treatment_periods=data["a"],
            post_treatment_horizon_offset=data["k_horizon"],
        )

def test_ssdid_est_invalid_donor_matrix_ndim(ssdid_sample_data):
    data = ssdid_sample_data
    with pytest.raises(MlsynthDataError, match="`donor_units_outcomes_all_periods` must be a 2D NumPy array."):
        ssdid_est(
            treated_unit_outcomes_all_periods=data["treated_y"],
            donor_units_outcomes_all_periods=np.array([1,2,3]), # 1D array
            donor_weights=np.random.rand(data["J"]),
            time_weights_vector=np.random.rand(data["a"]),
            num_pre_treatment_periods=data["a"],
            post_treatment_horizon_offset=data["k_horizon"],
        )

def test_ssdid_est_invalid_donor_weights_type(ssdid_sample_data):
    data = ssdid_sample_data
    with pytest.raises(MlsynthDataError, match="`donor_weights` must be a 1D NumPy array."):
        ssdid_est(
            treated_unit_outcomes_all_periods=data["treated_y"],
            donor_units_outcomes_all_periods=data["donor_matrix"],
            donor_weights=[0.1, 0.9], # List, not array
            time_weights_vector=np.random.rand(data["a"]),
            num_pre_treatment_periods=data["a"],
            post_treatment_horizon_offset=data["k_horizon"],
        )

def test_ssdid_est_invalid_time_weights_type(ssdid_sample_data):
    data = ssdid_sample_data
    with pytest.raises(MlsynthDataError, match="`time_weights_vector` must be a 1D NumPy array."):
        ssdid_est(
            treated_unit_outcomes_all_periods=data["treated_y"],
            donor_units_outcomes_all_periods=data["donor_matrix"],
            donor_weights=np.random.rand(data["J"]),
            time_weights_vector=(0.5, 0.5), # Tuple, not array
            num_pre_treatment_periods=data["a"],
            post_treatment_horizon_offset=data["k_horizon"],
        )

def test_ssdid_est_negative_num_pre_treatment_periods(ssdid_sample_data):
    data = ssdid_sample_data
    with pytest.raises(MlsynthDataError, match="`num_pre_treatment_periods` cannot be negative."):
        ssdid_est(
            treated_unit_outcomes_all_periods=data["treated_y"],
            donor_units_outcomes_all_periods=data["donor_matrix"],
            donor_weights=np.random.rand(data["J"]),
            time_weights_vector=np.random.rand(data["a"]),
            num_pre_treatment_periods=-1,
            post_treatment_horizon_offset=data["k_horizon"],
        )

def test_ssdid_est_negative_post_treatment_horizon_offset(ssdid_sample_data):
    data = ssdid_sample_data
    with pytest.raises(MlsynthDataError, match="`post_treatment_horizon_offset` cannot be negative."):
        ssdid_est(
            treated_unit_outcomes_all_periods=data["treated_y"],
            donor_units_outcomes_all_periods=data["donor_matrix"],
            donor_weights=np.random.rand(data["J"]),
            time_weights_vector=np.random.rand(data["a"]),
            num_pre_treatment_periods=data["a"],
            post_treatment_horizon_offset=-1,
        )

def test_ssdid_est_period_mismatch_treated_donors(ssdid_sample_data):
    data = ssdid_sample_data
    short_donor_matrix = data["donor_matrix"][:-1, :] # One period shorter
    with pytest.raises(MlsynthDataError, match="Mismatch in total periods"):
        ssdid_est(
            treated_unit_outcomes_all_periods=data["treated_y"],
            donor_units_outcomes_all_periods=short_donor_matrix,
            donor_weights=np.random.rand(data["J"]),
            time_weights_vector=np.random.rand(data["a"]),
            num_pre_treatment_periods=data["a"],
            post_treatment_horizon_offset=data["k_horizon"],
        )

def test_ssdid_est_post_treatment_index_out_of_bounds(ssdid_sample_data):
    data = ssdid_sample_data
    with pytest.raises(MlsynthDataError, match="Calculated `post_treatment_period_index` .* is out of bounds"):
        ssdid_est(
            treated_unit_outcomes_all_periods=data["treated_y"],
            donor_units_outcomes_all_periods=data["donor_matrix"],
            donor_weights=np.random.rand(data["J"]),
            time_weights_vector=np.random.rand(data["a"]),
            num_pre_treatment_periods=data["a"],
            post_treatment_horizon_offset=len(data["treated_y"]), # This will make index too large
        )

def test_ssdid_est_num_pre_periods_out_of_bounds(ssdid_sample_data):
    data = ssdid_sample_data
    # This test sets num_pre_treatment_periods to be too large.
    # This also causes post_treatment_period_index to be too large,
    # and that check comes first in the ssdid_est function.
    with pytest.raises(MlsynthDataError, match="Calculated `post_treatment_period_index` .* is out of bounds"):
        ssdid_est(
            treated_unit_outcomes_all_periods=data["treated_y"],
            donor_units_outcomes_all_periods=data["donor_matrix"],
            donor_weights=np.random.rand(data["J"]),
            time_weights_vector=np.random.rand(data["a"]),
            num_pre_treatment_periods=len(data["treated_y"]) + 1, # Too large
            post_treatment_horizon_offset=data["k_horizon"],
        )

def test_ssdid_est_time_weights_length_mismatch(ssdid_sample_data):
    data = ssdid_sample_data
    wrong_length_time_weights = np.random.rand(data["a"] + 1)
    with pytest.raises(MlsynthDataError, match="Length of `time_weights_vector` .* does not match `num_pre_treatment_periods`"):
        ssdid_est(
            treated_unit_outcomes_all_periods=data["treated_y"],
            donor_units_outcomes_all_periods=data["donor_matrix"],
            donor_weights=np.random.rand(data["J"]),
            time_weights_vector=wrong_length_time_weights,
            num_pre_treatment_periods=data["a"],
            post_treatment_horizon_offset=data["k_horizon"],
        )

def test_ssdid_est_donor_weights_length_mismatch(ssdid_sample_data):
    data = ssdid_sample_data
    wrong_length_donor_weights = np.random.rand(data["J"] + 1)
    with pytest.raises(MlsynthDataError, match="Length of `donor_weights` .* does not match number of donors"):
        ssdid_est(
            treated_unit_outcomes_all_periods=data["treated_y"],
            donor_units_outcomes_all_periods=data["donor_matrix"],
            donor_weights=wrong_length_donor_weights,
            time_weights_vector=np.random.rand(data["a"]),
            num_pre_treatment_periods=data["a"],
            post_treatment_horizon_offset=data["k_horizon"],
        )

# --- Fixtures for sc_diagplot ---

@pytest.fixture
def sc_diagplot_sample_config() -> dict:
    """Provides a sample configuration for sc_diagplot tests."""
    df = pd.DataFrame({
        "unit": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "time": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "outcome_var": np.random.rand(9),
        "treated_unit_flag": [1, 1, 1, 0, 0, 0, 0, 0, 0]
    })
    return {
        "df": df,
        "unitid": "unit",
        "time": "time",
        "outcome": "outcome_var",
        "treat": "treated_unit_flag",
    }

# --- Tests for sc_diagplot ---


@patch("mlsynth.utils.helperutils.plt.show")
@patch("mlsynth.utils.datautils.dataprep")  # Corrected patch target
def test_sc_diagplot_smoke(mock_dataprep, mock_plt_show, sc_diagplot_sample_config):
    """Smoke test for sc_diagplot function."""
    # Mock dataprep to return a structure for a single treated unit
    mock_dataprep.return_value = {
        "y": np.random.rand(10, 1),  # Treated unit outcome
        "donor_matrix": np.random.rand(10, 5),  # Donor outcomes
        "pre_periods": 5,
        "treated_unit_name": "Unit1",
        "Ywide": pd.DataFrame(index=np.arange(10))  # For time axis
    }

    config_list = [sc_diagplot_sample_config]

    sc_diagplot(config_list)

    # ✅ Updated to match actual function call signature
    mock_dataprep.assert_called_once_with(
        df=sc_diagplot_sample_config["df"],
        unit_id_column_name=sc_diagplot_sample_config["unitid"],
        time_period_column_name=sc_diagplot_sample_config["time"],
        outcome_column_name=sc_diagplot_sample_config["outcome"],
        treatment_indicator_column_name=sc_diagplot_sample_config["treat"],
    )

    mock_plt_show.assert_called_once()




def test_sc_diagplot_invalid_config_list_type():
    """Test sc_diagplot with non-list config_list."""
    with pytest.raises(MlsynthConfigError, match="Input 'config_list' must be a list of configuration dictionaries."):
        sc_diagplot("not_a_list")

@patch("mlsynth.utils.helperutils.plt.show")
@patch("mlsynth.utils.datautils.dataprep") # Corrected patch target
def test_sc_diagplot_multiple_cohorts_no_cohort_key(mock_dataprep, mock_plt_show, sc_diagplot_sample_config):
    """Test sc_diagplot with multiple cohorts but no cohort key specified."""
    mock_dataprep.return_value = {
        "cohorts": {
            "cohort1": {"y": np.random.rand(10,1), "donor_matrix": np.random.rand(10, 2), "pre_periods": 5, "treated_units": ["T1"]},
            "cohort2": {"y": np.random.rand(10,1), "donor_matrix": np.random.rand(10, 2), "pre_periods": 5, "treated_units": ["T2"]},
        },
        "Ywide": pd.DataFrame(index=np.arange(10))
    }
    config_list = [sc_diagplot_sample_config] # Config itself doesn't have 'cohort'
    
    with pytest.raises(MlsynthConfigError, match="Multiple cohorts found in data. Please specify a 'cohort' in the plot configuration to resolve ambiguity."):
        sc_diagplot(config_list)
    mock_plt_show.assert_not_called() # Should fail before plotting

@patch("mlsynth.utils.helperutils.plt.show")
@patch("mlsynth.utils.datautils.dataprep") # Corrected patch target
def test_sc_diagplot_multiple_cohorts_with_cohort_key(mock_dataprep, mock_plt_show, sc_diagplot_sample_config):
    """Test sc_diagplot with multiple cohorts and a cohort key specified."""
    mock_dataprep.return_value = {
        "cohorts": {
            "C1": {"y": np.random.rand(10,1), "donor_matrix": np.random.rand(10, 2), "pre_periods": 5, "treated_units": ["T1"]},
            "C2": {"y": np.random.rand(10,1), "donor_matrix": np.random.rand(10, 2), "pre_periods": 5, "treated_units": ["T2"]},
        },
        "Ywide": pd.DataFrame(index=np.arange(10))
    }
    config_with_cohort = {**sc_diagplot_sample_config, "cohort": "C1"}
    config_list = [config_with_cohort]
    
    sc_diagplot(config_list)
    mock_dataprep.assert_called_once()
    mock_plt_show.assert_called_once()

@patch("mlsynth.utils.helperutils.plt.show")
@patch("mlsynth.utils.datautils.dataprep") # Corrected patch target
def test_sc_diagplot_single_cohort_in_cohorts_structure(mock_dataprep, mock_plt_show, sc_diagplot_sample_config):
    """Test sc_diagplot when dataprep returns a 'cohorts' dict with only one cohort, and no cohort key is in config."""
    mock_dataprep.return_value = {
        "cohorts": {
            "TheOnlyCohort": {"y": np.random.rand(10,1), "donor_matrix": np.random.rand(10, 2), "pre_periods": 5, "treated_units": ["T1"]},
        },
        "Ywide": pd.DataFrame(index=np.arange(10))
    }
    config_list = [sc_diagplot_sample_config] # No 'cohort' key in config
    
    sc_diagplot(config_list)
    mock_dataprep.assert_called_once()
    mock_plt_show.assert_called_once()

@patch("mlsynth.utils.helperutils.plt.show")
@patch("mlsynth.utils.datautils.dataprep") # Corrected patch target
def test_sc_diagplot_multiple_plots(mock_dataprep, mock_plt_show, sc_diagplot_sample_config):
    """Test sc_diagplot with multiple configurations to generate multiple subplots."""
    mock_dataprep.side_effect = [
        { # First call
            "y": np.random.rand(10, 1), "donor_matrix": np.random.rand(10, 3),
            "pre_periods": 5, "treated_unit_name": "UnitA", "Ywide": pd.DataFrame(index=np.arange(10))
        },
        { # Second call
            "cohorts": {"C1": {"y": np.random.rand(10,1), "donor_matrix": np.random.rand(10, 2), "pre_periods": 5, "treated_units": ["T1_C1"]}},
            "Ywide": pd.DataFrame(index=np.arange(10))
        }
    ]
    
    config1 = {**sc_diagplot_sample_config, "outcome": "outcome1"}
    config2 = {**sc_diagplot_sample_config, "outcome": "outcome2", "cohort": "C1"}
    config_list = [config1, config2]
    
    # Mock plt.subplots to check number of subplots
    mock_fig = MagicMock()
    mock_axes = [MagicMock(), MagicMock()] # Two axes for two plots
    with patch("mlsynth.utils.helperutils.plt.subplots", return_value=(mock_fig, mock_axes)) as mock_subplots:
        sc_diagplot(config_list)
    
    mock_subplots.assert_called_once_with(1, 2, figsize=(12, 5), sharey=True) # 2 plots
    assert mock_dataprep.call_count == 2
    mock_plt_show.assert_called_once()
    # Check titles were set on mocked axes
    mock_axes[0].set_title.assert_called_with("Unit: UnitA", loc="left")
    mock_axes[1].set_title.assert_called_with("Unit(s): T1_C1", loc="left")
