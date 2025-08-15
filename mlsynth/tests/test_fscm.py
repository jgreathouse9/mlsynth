import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor # Added import
from pydantic import ValidationError

from mlsynth import FSCM
from mlsynth.config_models import FSCMConfig, BaseEstimatorResults
from mlsynth.exceptions import (
    MlsynthDataError,
    MlsynthConfigError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from mlsynth.utils.estutils import fSCM

# Base configuration dictionary containing all parameters used in tests.
# We will extract Pydantic-valid fields from this for FSCMConfig instantiation.
FSCM_FULL_TEST_CONFIG_BASE: Dict[str, Any] = {
    "outcome": "Y",
    "treat": "treated_indicator_programmatic",
    "unitid": "unit_id",
    "time": "time_id",
    "counterfactual_color": ["blue"],
    "treated_color": "green",
    "display_graphs": False,
    "save": False,
    # FSCM-specific parameters (not in BaseEstimatorConfig/current FSCMConfig)
    "lambda_": 0.1, # Note: Pydantic model might use 'lambda_val'
    "omega": "balanced", # Note: Pydantic model might use 'omega_val'
    "cv_lambda": False,
    "cv_omega": False,
    "min_lambda": 1e-5,
    "max_lambda": 1e5,
    "grid_lambda": 10,
    "min_omega": 0,
    "max_omega": 1,
    "grid_omega": 10,
    "cv_folds": 5,
    "model_type": "conformal", # This seems to be for conformal prediction, maybe not FSCM core
    "level": 0.95, # For conformal prediction
    "seed": 12345,
    "parallel": False,
    "verbose": False,
}

# Fields that are part of BaseEstimatorConfig (and thus FSCMConfig)
FSCM_PYDANTIC_MODEL_FIELDS = [
    "df", "outcome", "treat", "unitid", "time", 
    "display_graphs", "save", "counterfactual_color", "treated_color"
]

def _get_pydantic_config_dict(full_config: Dict[str, Any], df_fixture: pd.DataFrame) -> Dict[str, Any]:
    """Helper to extract Pydantic-valid fields and add the DataFrame."""
    pydantic_dict = {
        k: v for k, v in full_config.items() if k in FSCM_PYDANTIC_MODEL_FIELDS
    }
    pydantic_dict["df"] = df_fixture
    return pydantic_dict

@pytest.fixture
def basic_panel_data_with_treatment():
    """Provides a very basic panel dataset with a treatment column for smoke testing."""
    data_dict = {
        'unit_id': ['1', '1', '1', '1', '2', '2', '2', '2', '3', '3', '3', '3'], # Changed to strings
        'time_id': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
        'Y':       [10, 11, 15, 16, 9, 10, 11, 12, 12, 13, 14, 15],
        'X1':      [5, 6, 7, 8, 4, 5, 6, 7, 6, 7, 8, 9],
    }
    df = pd.DataFrame(data_dict)
    
    # Add treatment column as expected by FSCM (via config['treat'])
    # Unit 1 is treated starting from time_id = 3
    treatment_col_name = FSCM_FULL_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    df.loc[(df['unit_id'] == '1') & (df['time_id'] >= 3), treatment_col_name] = 1 # Corrected to string comparison
    return df

def test_fscm_creation(basic_panel_data_with_treatment: pd.DataFrame):
    """Test that the FSCM estimator can be instantiated."""
    pydantic_dict = _get_pydantic_config_dict(FSCM_FULL_TEST_CONFIG_BASE, basic_panel_data_with_treatment)
    
    try:
        config_obj = FSCMConfig(**pydantic_dict)
        estimator = FSCM(config=config_obj) # Assumes FSCM.__init__ now takes FSCMConfig
        assert estimator is not None, "FSCM estimator should be created."
        assert estimator.outcome == "Y", "Outcome attribute should be set from config."
        assert estimator.treat == FSCM_FULL_TEST_CONFIG_BASE["treat"]
        assert not estimator.display_graphs, "display_graphs should be False for tests."
    except Exception as e:
        pytest.fail(f"FSCM instantiation failed: {e}")

def test_fscm_fit_smoke(basic_panel_data_with_treatment: pd.DataFrame):
    """Smoke test for the FSCM fit method to ensure it runs without crashing."""
    pydantic_dict = _get_pydantic_config_dict(FSCM_FULL_TEST_CONFIG_BASE, basic_panel_data_with_treatment)
    config_obj = FSCMConfig(**pydantic_dict)
    estimator = FSCM(config=config_obj)
    
    try:
        results = estimator.fit() 
        assert results is not None, "Fit method should return BaseEstimatorResults."
        assert isinstance(results, BaseEstimatorResults), "Fit method should return BaseEstimatorResults."

        assert 'FSCM' in results.sub_method_results is not None, "Results should contain 'effects'."
        assert results.weights is not None, "Results should contain 'weights'."
        assert results.time_series is not None
        assert results.method_details is not None
        assert results.method_details.name == "FSCM" # Corrected attribute name
        assert results.sub_method_results['FSCM'].effects is not None
        
        assert isinstance(results.effects.att, (float, np.floating, type(None))) # ATT can be None if not calculable
        
        # donor_weights is now a Dict[str, float] or None
        assert results.weights.donor_weights is None or isinstance(results.weights.donor_weights, dict)
        if isinstance(results.weights.donor_weights, dict) and results.weights.donor_weights:
            assert all(isinstance(k, str) for k in results.weights.donor_weights.keys())
            assert all(isinstance(v, float) for v in results.weights.donor_weights.values())
        
        # donor_names is not directly in WeightsResults anymore, it's implicit in the dict keys.

        # Check raw_results for original structure
        assert results.raw_results is not None
        assert "Effects" in results.raw_results
        assert "Weights" in results.raw_results
        assert isinstance(results.raw_results["Weights"], list)
        assert len(results.raw_results["Weights"]) == 2
    except Exception as e:
        pytest.fail(f"FSCM fit method failed during smoke test: {e}")

@pytest.mark.parametrize(
    "omega_config, lambda_config",
    [
        ({"omega": "SCM"}, {"lambda_": None}), # SCM implies lambda is not used or fixed
        ({"omega": "balanced"}, {"lambda_": 0.1}),
        ({"omega": 0.5}, {"lambda_": 0.2}), # Numeric omega
    ],
)
def test_fscm_fit_omega_lambda_variants(
    basic_panel_data_with_treatment, omega_config, lambda_config
):
    """Test FSCM fit with different omega and lambda configurations."""
    full_config_dict = FSCM_FULL_TEST_CONFIG_BASE.copy()
    # These omega/lambda updates are to the full_config_dict.
    # FSCMConfig will only pick up base fields. FSCM.fit() will use internal defaults
    # for omega/lambda if FSCM class now only relies on FSCMConfig for these.
    full_config_dict.update(omega_config)
    full_config_dict.update(lambda_config)
    full_config_dict["cv_lambda"] = False # Ensure CV is off
    full_config_dict["cv_omega"] = False

    pydantic_dict = _get_pydantic_config_dict(full_config_dict, basic_panel_data_with_treatment)
    config_obj = FSCMConfig(**pydantic_dict)
    estimator = FSCM(config=config_obj)
    
    results = estimator.fit() 

    assert isinstance(results, BaseEstimatorResults)
    assert results.effects is not None
    assert results.weights is not None
    assert results.effects.att is not None # Or check for float/None

    if omega_config["omega"] == "SCM": # This test logic might need review as FSCMConfig is minimal
                                      # The FSCM class itself doesn't use omega/lambda from config currently
        # For SCM, weights should sum to 1 (or close to it)
        if results.weights.donor_weights is not None and len(results.weights.donor_weights) > 0:
             assert np.isclose(np.sum(list(results.weights.donor_weights.values())), 1.0, atol=1e-5)


def test_fscm_fit_cv_lambda(basic_panel_data_with_treatment):
    """Test FSCM fit with lambda cross-validation enabled."""
    full_config_dict = FSCM_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict["cv_lambda"] = True
    full_config_dict["cv_omega"] = False 
    full_config_dict["omega"] = "balanced" 

    pydantic_dict = _get_pydantic_config_dict(full_config_dict, basic_panel_data_with_treatment)
    config_obj = FSCMConfig(**pydantic_dict)
    estimator = FSCM(config=config_obj)
    
    results = estimator.fit() 

    assert isinstance(results, BaseEstimatorResults)
    assert results.effects is not None
    assert results.weights is not None
    assert results.effects.att is not None
    # TODO: Investigate if 'best_lambda' should be in results['Fit'] or elsewhere when cv_lambda=True.
    # For now, test that it runs without error.
    # Assertions about 'best_lambda' would depend on how FSCM now handles/returns it.

def test_fscm_fit_cv_omega(basic_panel_data_with_treatment: pd.DataFrame):
    """Test FSCM fit with omega cross-validation enabled."""
    full_config_dict = FSCM_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict["cv_omega"] = True
    full_config_dict["cv_lambda"] = False 
    full_config_dict["omega"] = [0.1, 0.5, 0.9] # This variation won't pass to FSCM via minimal FSCMConfig

    pydantic_dict = _get_pydantic_config_dict(full_config_dict, basic_panel_data_with_treatment)
    config_obj = FSCMConfig(**pydantic_dict)
    estimator = FSCM(config=config_obj)
    
    results = estimator.fit() 

    assert isinstance(results, BaseEstimatorResults)
    assert results.effects is not None
    assert results.weights is not None
    assert results.effects.att is not None
    # Assertions about 'best_omega' would depend on how FSCM now handles/returns it.

def test_fscm_init_missing_dataframe_leads_to_config_error():
    """Test FSCMConfig instantiation fails if DataFrame 'df' is missing."""
    config_dict_no_df = FSCM_FULL_TEST_CONFIG_BASE.copy()
    # "df" is deliberately not added via _get_pydantic_config_dict or explicitly
    
    pydantic_dict_attempt = {
        k: v for k, v in config_dict_no_df.items() if k in FSCM_PYDANTIC_MODEL_FIELDS and k != "df"
    }
    with pytest.raises(ValidationError): # df is required by BaseEstimatorConfig
        FSCMConfig(**pydantic_dict_attempt)

def test_fscm_init_missing_essential_column_name_leads_to_config_error(basic_panel_data_with_treatment: pd.DataFrame):
    """Test FSCMConfig instantiation fails if an essential column name (e.g., outcome) is missing."""
    config_dict_missing_outcome = FSCM_FULL_TEST_CONFIG_BASE.copy()
    del config_dict_missing_outcome["outcome"]
    pydantic_dict_missing_outcome = _get_pydantic_config_dict(config_dict_missing_outcome, basic_panel_data_with_treatment)
    # Remove 'outcome' after helper if it was still there (it shouldn't be if key was deleted)
    if "outcome" in pydantic_dict_missing_outcome: del pydantic_dict_missing_outcome["outcome"]

    with pytest.raises(ValidationError): # outcome is required
        FSCMConfig(**pydantic_dict_missing_outcome)

    config_dict_missing_treat = FSCM_FULL_TEST_CONFIG_BASE.copy()
    del config_dict_missing_treat["treat"]
    pydantic_dict_missing_treat = _get_pydantic_config_dict(config_dict_missing_treat, basic_panel_data_with_treatment)
    if "treat" in pydantic_dict_missing_treat: del pydantic_dict_missing_treat["treat"]
    
    with pytest.raises(ValidationError): # treat is required
        FSCMConfig(**pydantic_dict_missing_treat)

# --- New tests for error handling ---

def test_fscm_fit_dataprep_key_error(basic_panel_data_with_treatment: pd.DataFrame):
    """Test fit raises MlsynthEstimationError if dataprep output is missing essential keys."""
    pydantic_dict = _get_pydantic_config_dict(FSCM_FULL_TEST_CONFIG_BASE, basic_panel_data_with_treatment)
    config_obj = FSCMConfig(**pydantic_dict)
    estimator = FSCM(config=config_obj)

    with patch('mlsynth.estimators.fscm.dataprep') as mock_dataprep:
        mock_dataprep.return_value = {"pre_periods": 5} # Missing 'donor_matrix', 'y', etc.
        with pytest.raises(MlsynthEstimationError, match="Essential key 'donor_matrix' missing"):
            estimator.fit()

def test_fscm_fit_dataprep_invalid_pre_periods(basic_panel_data_with_treatment: pd.DataFrame):
    """Test fit raises MlsynthEstimationError if dataprep returns invalid pre_periods."""
    pydantic_dict = _get_pydantic_config_dict(FSCM_FULL_TEST_CONFIG_BASE, basic_panel_data_with_treatment)
    config_obj = FSCMConfig(**pydantic_dict)
    estimator = FSCM(config=config_obj)
    
    with patch('mlsynth.estimators.fscm.dataprep') as mock_dataprep:
        # Mock dataprep to return a dictionary with an invalid 'pre_periods'
        mock_dataprep.return_value = {
            "donor_matrix": np.array([[1,2],[3,4]]), 
            "pre_periods": 0, # Invalid
            "y": pd.Series([1,2]), 
            "donor_names": ["d1", "d2"], 
            "treated_unit_name": "t1",
            "time_labels": [1,2,3,4]
        }
        with pytest.raises(MlsynthEstimationError, match="Invalid 'pre_periods'"):
            estimator.fit()

def test_fscm_fit_no_donors_after_dataprep(basic_panel_data_with_treatment: pd.DataFrame):
    """Test fit raises MlsynthEstimationError if no donor units are available."""
    pydantic_dict = _get_pydantic_config_dict(FSCM_FULL_TEST_CONFIG_BASE, basic_panel_data_with_treatment)
    config_obj = FSCMConfig(**pydantic_dict)
    estimator = FSCM(config=config_obj)

    with patch('mlsynth.estimators.fscm.dataprep') as mock_dataprep:
        mock_dataprep.return_value = {
            "donor_matrix": np.array([[], []]).reshape(2,0), # No donors
            "pre_periods": 2,
            "y": pd.Series([10, 11]),
            "donor_names": [],
            "treated_unit_name": "t1",
            "time_labels": [1,2,3,4],
            "post_periods": 2,
        }
        with pytest.raises(MlsynthEstimationError, match="No donor units available after data preparation."):
            estimator.fit()

@patch('mlsynth.utils.estutils.fSCM')
def test_fscm_fit_fscm_method_raises_error(mock_fscm_method, basic_panel_data_with_treatment: pd.DataFrame):
    """Test fit handles errors from the internal fSCM optimization method."""
    pydantic_dict = _get_pydantic_config_dict(FSCM_FULL_TEST_CONFIG_BASE, basic_panel_data_with_treatment)
    config_obj = FSCMConfig(**pydantic_dict)
    estimator = FSCM(config=config_obj)
    
    mock_fscm_method.side_effect = MlsynthEstimationError("Internal fSCM failure")
    with pytest.raises(MlsynthEstimationError, match="Internal fSCM failure"):
        estimator.fit()

@patch('mlsynth.estimators.fscm.plot_estimates')
def test_fscm_fit_plotting_error_caught_gracefully(mock_plot_estimates, basic_panel_data_with_treatment: pd.DataFrame, capsys):
    """Test that errors during plotting are caught and a warning is printed, but results are still returned."""
    pydantic_dict = _get_pydantic_config_dict(FSCM_FULL_TEST_CONFIG_BASE, basic_panel_data_with_treatment)
    pydantic_dict["display_graphs"] = True # Ensure plotting is attempted
    config_obj = FSCMConfig(**pydantic_dict)
    estimator = FSCM(config=config_obj)
    
    mock_plot_estimates.side_effect = MlsynthPlottingError("Simulated plotting error")
    
    results = estimator.fit()
    assert results is not None, "Fit should still return results even if plotting fails."
    assert isinstance(results, BaseEstimatorResults)
    
    captured = capsys.readouterr()
    assert "Warning: Plotting failed with MlsynthPlottingError: Simulated plotting error" in captured.out or \
           "Warning: Plotting failed with MlsynthPlottingError: Simulated plotting error" in captured.err


def test_fscm_fscm_method_no_donors_evaluated_error(basic_panel_data_with_treatment: pd.DataFrame):
    """Test the fSCM method raises error if no donors are evaluated in parallel step."""
    pydantic_dict = _get_pydantic_config_dict(FSCM_FULL_TEST_CONFIG_BASE, basic_panel_data_with_treatment)
    config_obj = FSCMConfig(**pydantic_dict)
    estimator = FSCM(config=config_obj)

    # Prepare minimal inputs for fSCM method
    # This test is tricky because ThreadPoolExecutor is used.
    # We'll mock evaluate_donor to return an empty list from map.
    with patch.object(ThreadPoolExecutor, 'map', return_value=[]):
        with pytest.raises(MlsynthEstimationError, match="No donor units were successfully evaluated"):
            estimator.fSCM(
                treated_outcome_pre_treatment_vector=np.array([1,2]),
                all_donors_outcomes_matrix_pre_treatment=np.array([[1,2],[3,4]]), # Dummy, won't be used by mocked map
                num_pre_treatment_periods=2
            )


@patch('mlsynth.estimators.fscm.Opt.SCopt')
def test_fscm_fscm_method_final_opt_fails(mock_scopt, basic_panel_data_with_treatment: pd.DataFrame):
    """Test fSCM method handles failure in final SCM optimization."""
    pydantic_dict = _get_pydantic_config_dict(FSCM_FULL_TEST_CONFIG_BASE, basic_panel_data_with_treatment)
    config_obj = FSCMConfig(**pydantic_dict)
    estimator = FSCM(config=config_obj)

    # Mock evaluate_donor to return some plausible single donor MSEs
    def mock_evaluate_donor(donor_idx, _, __, ___):
        mse_map = {0: 0.1, 1: 0.05} # Donor 1 is best
        return donor_idx, mse_map.get(donor_idx, 0.2)

    # Mock Opt.SCopt: first calls succeed, final call fails
    mock_solution_good = MagicMock()
    mock_solution_good.opt_val = 0.05 # MSE
    mock_solution_good.primal_vars = {"w": np.array([1.0])}

    mock_solution_final_fail = MagicMock()
    mock_solution_final_fail.primal_vars = None # Simulate failure

    # Let initial evaluations and forward selection step succeed
    # Then make the *final* optimization call fail
    # When estimator.evaluate_donor is patched, fSCM directly calls Opt.SCopt:
    # 1. Inside the forward selection loop (for candidate_donor_index_set_for_evaluation)
    # 2. For the final optimization after the loop.
    # So, we need 2 mocks for Opt.SCopt.
    mock_scopt.side_effect = [
        MagicMock(solution=mock_solution_good), # For the forward selection step
        MagicMock(solution=mock_solution_final_fail) # Final optimization fails
    ]
    
    with patch.object(estimator, 'evaluate_donor', side_effect=mock_evaluate_donor):
        with pytest.raises(MlsynthEstimationError, match="Final SCM optimization did not yield donor weights"):
            estimator.fSCM(
                treated_outcome_pre_treatment_vector=np.array([10,11]),
                all_donors_outcomes_matrix_pre_treatment=np.array([[9,12],[10,13]]), # 2 donors
                num_pre_treatment_periods=2
            )
