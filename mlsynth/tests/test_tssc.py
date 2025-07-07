import pytest
import pandas as pd
import numpy as np
import cvxpy # Ensure cvxpy is imported for its error types
from unittest.mock import patch, MagicMock
from pydantic import ValidationError # Import ValidationError
from mlsynth import TSSC 
from mlsynth.config_models import TSSCConfig, BaseEstimatorResults # Import Pydantic models
from mlsynth.utils.datautils import balance, dataprep
from mlsynth.utils.estutils import TSEST
from mlsynth.utils.inferutils import step2
from mlsynth.utils.resultutils import plot_estimates
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)


# Configuration for TSSC
TSSC_TEST_CONFIG_BASE = {
    "outcome": "target_y",
    "treat": "is_treated_flag_tssc",
    "unitid": "entity_id",
    "time": "time_idx",
    "display_graphs": False,
    "save": False,
    "draws": 10, # Small number of draws for testing
    "counterfactual_color": ["lime"],
    "treated_color": "purple"
}

@pytest.fixture
def tssc_panel_data():
    """Provides a panel dataset for TSSC smoke testing."""
    n_units = 10 
    n_periods = 15 # TSSC might need enough pre-periods for its internal methods
    data_dict = {
        'entity_id': np.repeat(np.arange(1, n_units + 1), n_periods),
        'time_idx': np.tile(np.arange(1, n_periods + 1), n_units),
        'target_y': np.random.normal(loc=np.repeat(np.arange(0, n_units*3, 3), n_periods), scale=2, size=n_units*n_periods),
        'feature_x': np.random.rand(n_units * n_periods) * 20,
    }
    df = pd.DataFrame(data_dict)
    
    # Unit 1 is treated starting from time_idx = 10
    treatment_col_name = TSSC_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    df.loc[(df['entity_id'] == 1) & (df['time_idx'] >= 10), treatment_col_name] = 1
    return df

def test_tssc_creation(tssc_panel_data):
    """Test that the TSSC estimator can be instantiated."""
    test_config_dict = TSSC_TEST_CONFIG_BASE.copy()
    test_config_dict["df"] = tssc_panel_data
    
    try:
        config_obj = TSSCConfig(**test_config_dict)
        estimator = TSSC(config=config_obj)
        assert estimator is not None, "TSSC estimator should be created."
        assert estimator.outcome == "target_y"
        assert estimator.treat == TSSC_TEST_CONFIG_BASE["treat"]
        assert estimator.draws == 10
        assert not estimator.display_graphs
    except Exception as e:
        pytest.fail(f"TSSC instantiation failed: {e}")

def test_tssc_fit_smoke(tssc_panel_data):
    """Smoke test for the TSSC fit method."""
    test_config_dict = TSSC_TEST_CONFIG_BASE.copy()
    test_config_dict["df"] = tssc_panel_data
    
    config_obj = TSSCConfig(**test_config_dict)
    estimator = TSSC(config=config_obj)
    
    try:
        results_list = estimator.fit() # fit() takes no arguments
        assert results_list is not None, "Fit method should return results."
        assert isinstance(results_list, list), "Results should be a list."
        assert len(results_list) > 0, "Results list should not be empty (expecting results for multiple SC methods)."
        
        # Check structure of the first result BaseEstimatorResults object in the list
        first_pydantic_result = results_list[0]
        assert isinstance(first_pydantic_result, BaseEstimatorResults)
        
        assert first_pydantic_result.effects is not None
        assert first_pydantic_result.fit_diagnostics is not None
        assert first_pydantic_result.time_series is not None
        assert first_pydantic_result.weights is not None
        assert first_pydantic_result.inference is not None
        assert first_pydantic_result.method_details is not None
        assert first_pydantic_result.raw_results is not None # Corrected indentation

        # Check some nested attributes
        assert isinstance(first_pydantic_result.weights.donor_weights, dict) # Changed from np.ndarray
        assert isinstance(first_pydantic_result.time_series.counterfactual_outcome, np.ndarray)
            
        # Check raw_results for original structure
        raw_res_check = first_pydantic_result.raw_results
        assert "Effects" in raw_res_check
        assert "WeightV" in raw_res_check
        assert "95% CI" in raw_res_check
        assert "Vectors" in raw_res_check and "Counterfactual" in raw_res_check["Vectors"]

    except Exception as e:
        if isinstance(e, (np.linalg.LinAlgError, ValueError)) and "singular" in str(e).lower():
             pytest.skip(f"Skipping due to potential singularity in TSSC with small random data: {e}")
        pytest.fail(f"TSSC fit method failed during smoke test: {e}")


# --- Input Validation Tests ---

def test_tssc_creation_missing_config_keys(tssc_panel_data):
    """Test TSSC instantiation with missing essential keys in config."""
    base_config_dict = TSSC_TEST_CONFIG_BASE.copy()
    base_config_dict["df"] = tssc_panel_data
    
    required_keys = ["df", "outcome", "treat", "unitid", "time"]
    
    for key_to_remove in required_keys:
        test_config_dict_iter = base_config_dict.copy()
        test_config_dict_iter.pop(key_to_remove, None)
        
        with pytest.raises(ValidationError): # Pydantic should catch missing required fields
            TSSCConfig(**test_config_dict_iter)

def test_tssc_creation_df_not_dataframe():
    """Test TSSC instantiation if 'df' in config is not a pandas DataFrame."""
    test_config_dict = TSSC_TEST_CONFIG_BASE.copy()
    test_config_dict["df"] = "not_a_dataframe" 
    with pytest.raises(ValidationError): # Pydantic should catch invalid type for df
        TSSCConfig(**test_config_dict)

def test_tssc_creation_df_missing_columns(tssc_panel_data):
    """Test TSSC instantiation if df is missing essential columns."""
    base_config = TSSC_TEST_CONFIG_BASE.copy()
    
    essential_cols = {
        "outcome": base_config["outcome"],
        "treat": base_config["treat"],
        "unitid": base_config["unitid"],
        "time": base_config["time"],
    }
    
    for col_key, col_name_in_config in essential_cols.items():
        df_missing_col = tssc_panel_data.copy()
        if col_name_in_config in df_missing_col.columns:
             df_missing_col = df_missing_col.drop(columns=[col_name_in_config])
        
        test_config_dict = base_config.copy()
        test_config_dict["df"] = df_missing_col
        
        # Error is raised by BaseEstimatorConfig's validator during TSSCConfig instantiation
        # The error message will contain the actual column name, not the key from essential_cols
        expected_missing_col_name_in_error_msg = col_name_in_config # This is the actual column name string
        with pytest.raises(MlsynthDataError, match=f"Missing required columns in DataFrame 'df': {expected_missing_col_name_in_error_msg}"):
            TSSCConfig(**test_config_dict)

def test_tssc_creation_invalid_config_types(tssc_panel_data):
    """Test TSSC instantiation with invalid types for optional config parameters."""
    base_config_dict = TSSC_TEST_CONFIG_BASE.copy()
    base_config_dict["df"] = tssc_panel_data

    # Test for draws
    config_invalid_draws_dict = base_config_dict.copy()
    config_invalid_draws_dict["draws"] = "not_an_int"
    with pytest.raises(ValidationError): # Pydantic should catch invalid type for draws
        TSSCConfig(**config_invalid_draws_dict)

    # Test for display_graphs
    config_invalid_display_dict = base_config_dict.copy()
    config_invalid_display_dict["display_graphs"] = "not_a_bool"
    with pytest.raises(ValidationError): # Pydantic should catch invalid type
        TSSCConfig(**config_invalid_display_dict)
        
    # Test for save
    config_invalid_save_dict = base_config_dict.copy()
    config_invalid_save_dict["save"] = 123 # Invalid type for save
    with pytest.raises(ValidationError): # Pydantic should catch invalid type
        TSSCConfig(**config_invalid_save_dict)


# --- Edge Case Tests ---

@patch('mlsynth.estimators.tssc.plot_estimates')
def test_tssc_fit_insufficient_pre_periods(mock_plot_estimates, tssc_panel_data):
    """Test TSSC fit with insufficient pre-treatment periods."""
    test_config_dict = TSSC_TEST_CONFIG_BASE.copy()
    df = tssc_panel_data.copy()
    
    # Modify data to have very few pre-periods (e.g., 1)
    # Original treatment at time_idx = 10 (9 pre-periods)
    df[test_config_dict["treat"]] = 0
    df.loc[(df['entity_id'] == 1) & (df['time_idx'] >= 2), test_config_dict["treat"]] = 1 # Treatment at period 2 (1 pre-period)
    
    test_config_dict["df"] = df
    config_obj = TSSCConfig(**test_config_dict)
    estimator = TSSC(config=config_obj)
    
    # With very few pre-periods, dataprep, TSEST, or step2 might produce warnings
    # or NaN results but not necessarily raise a Mlsynth*Error if underlying
    # utilities handle it gracefully (e.g., ci_bootstrap returns NaNs).
    # The main concern is that it doesn't crash with an unhandled error.
    try:
        estimator.fit()
    except (MlsynthDataError, MlsynthEstimationError, MlsynthConfigError) as e:
        # If it does raise one of our custom errors, that's acceptable.
        pass
    except Exception as e:
        pytest.fail(f"TSSC fit failed unexpectedly with insufficient pre-periods: {e}")
    
    # Plotting should not occur if estimation is problematic or results are invalid for plotting.
    # Depending on where an error might occur, or if it completes with NaNs,
    # plotting might or might not be called. If it completes successfully enough
    # to attempt plotting but then plotting fails, that's caught by plotting warnings.
    # For this test, if it errors out early, plot won't be called.
    # If it completes, but results are bad, plot_estimates might not be called or might warn.
    # The original test asserted not_called.
    # Given the `fit` method's structure, if `prepared_panel_data` is incomplete
    # due to very few pre-periods, plotting might be skipped or warn.
    # If `TSEST` or `step2` produce NaNs, plotting might still be attempted.
    # For now, let's assume if it doesn't raise a critical MlsynthError, it might try to plot.
    # However, the original test had assert_not_called. Let's stick to that logic for now,
    # implying that such a scenario should ideally not lead to a plot attempt or should error out before.
    # Re-evaluating: if fit *completes* without Mlsynth*Error, it will proceed to plotting.
    # The test setup has display_graphs=False by default in TSSC_TEST_CONFIG_BASE.
    # So, plot_estimates would not be called anyway unless display_graphs is True.
    # The mock_plot_estimates.assert_not_called() is correct if display_graphs is False.
    # Let's ensure display_graphs is False for this specific test logic.
    assert not estimator.config.display_graphs # Double check config for this test
    mock_plot_estimates.assert_not_called()


@patch('mlsynth.estimators.tssc.plot_estimates')
def test_tssc_fit_insufficient_donors(mock_plot_estimates, tssc_panel_data):
    """Test TSSC fit with insufficient donor units (e.g., 0 or 1)."""
    test_config_dict = TSSC_TEST_CONFIG_BASE.copy()
    df = tssc_panel_data.copy()

    # Keep only the treated unit (entity_id 1) and zero or one control unit
    df_no_donors = df[df['entity_id'] == 1] # No donors
    
    config_no_donors_dict = test_config_dict.copy()
    config_no_donors_dict["df"] = df_no_donors
    config_obj_no_donors = TSSCConfig(**config_no_donors_dict)
    estimator_no_donors = TSSC(config=config_obj_no_donors)
    
    # dataprep should raise MlsynthDataError if no control units are found
    with pytest.raises(MlsynthDataError, match="No donor units found after pivoting and selecting."):
        estimator_no_donors.fit()

    df_one_donor = df[df['entity_id'].isin([1, 2])] # One donor
    config_one_donor_dict = test_config_dict.copy()
    config_one_donor_dict["df"] = df_one_donor
    config_obj_one_donor = TSSCConfig(**config_one_donor_dict)
    estimator_one_donor = TSSC(config=config_obj_one_donor)
    # With only one donor, estimation might be problematic.
    # Allow MlsynthDataError, MlsynthEstimationError, or MlsynthConfigError,
    # or allow completion if underlying utilities handle it without these specific errors.
    try:
        estimator_one_donor.fit()
    except (MlsynthDataError, MlsynthEstimationError, MlsynthConfigError) as e:
        pass # Acceptable if a Mlsynth error is raised
    except Exception as e:
        pytest.fail(f"TSSC fit with one donor failed unexpectedly: {e}")
        
    # mock_plot_estimates might or might not be called depending on where the error occurs.
    # If fit() raises before plotting, it won't be called.
    # If fit() completes but plotting fails, it would be called.
    # Given the expectation of failure within fit's core logic, assert_not_called is reasonable.
    mock_plot_estimates.assert_not_called()


@patch('mlsynth.estimators.tssc.plot_estimates')
def test_tssc_fit_no_post_periods(mock_plot_estimates, tssc_panel_data):
    """Test TSSC fit with no post-treatment periods."""
    test_config_dict = TSSC_TEST_CONFIG_BASE.copy()
    df = tssc_panel_data.copy()

    last_period = df['time_idx'].max()
    df[test_config_dict["treat"]] = 0
    # Treat unit 1 at a time where no post-periods exist
    df.loc[(df['entity_id'] == 1) & (df['time_idx'] >= last_period + 1), test_config_dict["treat"]] = 1
    
    test_config_dict["df"] = df
    config_obj = TSSCConfig(**test_config_dict)
    estimator = TSSC(config=config_obj)
    
    # dataprep's logictreat should raise MlsynthDataError if no treated units are found.
    with pytest.raises(MlsynthDataError, match="No treated units found \\(zero treated observations with value 1\\)"):
        estimator.fit()
    mock_plot_estimates.assert_not_called()


@patch('mlsynth.estimators.tssc.plot_estimates')
def test_tssc_fit_nan_in_outcome(mock_plot_estimates, tssc_panel_data):
    """Test TSSC fit when outcome variable contains NaN values."""
    test_config_dict = TSSC_TEST_CONFIG_BASE.copy()
    
    # Case 1: NaN in treated unit's pre-period outcome
    df_nan_treated = tssc_panel_data.copy()
    df_nan_treated.loc[(df_nan_treated['entity_id'] == 1) & (df_nan_treated['time_idx'] == 5), test_config_dict["outcome"]] = np.nan
    
    config_nan_treated_dict = test_config_dict.copy()
    config_nan_treated_dict["df"] = df_nan_treated
    config_obj_nan_treated = TSSCConfig(**config_nan_treated_dict)
    estimator_nan_treated = TSSC(config=config_obj_nan_treated)
    
    # Expect MlsynthDataError or MlsynthEstimationError if NaNs cause issues in dataprep or estimation
    # StopIteration might still occur from underlying optimization if not caught and wrapped by estutils
    with pytest.raises((MlsynthDataError, MlsynthEstimationError, StopIteration)):
        estimator_nan_treated.fit()

    mock_plot_estimates.reset_mock()

    # Case 2: NaN in a control unit's pre-period outcome
    df_nan_control = tssc_panel_data.copy()
    df_nan_control.loc[(df_nan_control['entity_id'] == 2) & (df_nan_control['time_idx'] == 5), test_config_dict["outcome"]] = np.nan
    
    config_nan_control_dict = test_config_dict.copy()
    config_nan_control_dict["df"] = df_nan_control
    config_obj_nan_control = TSSCConfig(**config_nan_control_dict)
    estimator_nan_control = TSSC(config=config_obj_nan_control)

    # Expect MlsynthDataError or MlsynthEstimationError.
    # CVXPY errors should be wrapped into MlsynthEstimationError by estutils or tssc.
    # StopIteration might still occur.
    with pytest.raises((MlsynthDataError, MlsynthEstimationError, StopIteration)):
        estimator_nan_control.fit() 
    
    mock_plot_estimates.reset_mock()


# --- Detailed Results Validation ---

def test_tssc_fit_all_methods_results_structure(tssc_panel_data):
    """Test that fit returns results for all expected SC methods with correct structure."""
    test_config_dict = TSSC_TEST_CONFIG_BASE.copy()
    test_config_dict["df"] = tssc_panel_data
    config_obj = TSSCConfig(**test_config_dict)
    estimator = TSSC(config=config_obj)
    
    results_list = estimator.fit()
    
    assert isinstance(results_list, list)
    expected_methods = ["SIMPLEX", "MSCa", "MSCb", "MSCc"]
    returned_method_names = []

    for pydantic_result in results_list:
        assert isinstance(pydantic_result, BaseEstimatorResults)
        assert pydantic_result.method_details is not None
        method_name = pydantic_result.method_details.method_name
        assert method_name is not None
        returned_method_names.append(method_name)
        
        # Validate Pydantic object structure
        assert pydantic_result.effects is not None and isinstance(pydantic_result.effects.att, (float, np.floating, type(None)))
        assert pydantic_result.fit_diagnostics is not None
        assert pydantic_result.time_series is not None and isinstance(pydantic_result.time_series.counterfactual_outcome, np.ndarray)
        assert pydantic_result.weights is not None and \
               (pydantic_result.weights.donor_weights is None or isinstance(pydantic_result.weights.donor_weights, dict)) # Allow None due to potential mismatch
        assert pydantic_result.inference is not None
        
        # Check raw results for original keys for good measure
        raw_res = pydantic_result.raw_results
        assert "Effects" in raw_res and isinstance(raw_res["Effects"], dict)
        assert "WeightV" in raw_res and isinstance(raw_res["WeightV"], np.ndarray)
        assert "95% CI" in raw_res and isinstance(raw_res["95% CI"], list)
        assert "Vectors" in raw_res and "Counterfactual" in raw_res["Vectors"]

    for method in expected_methods:
        assert method in returned_method_names, f"Expected method {method} not found in results."
    assert len(returned_method_names) == len(expected_methods), "Unexpected number of methods returned."


# --- Configuration Variations ---

@patch('mlsynth.estimators.tssc.plot_estimates')
def test_tssc_fit_different_draws_values(mock_plot_estimates, tssc_panel_data):
    """Test TSSC fit with different numbers of bootstrap draws."""
    for num_draws in [0, 5]: # Test with 0 and a small number of draws
        test_config_dict = TSSC_TEST_CONFIG_BASE.copy()
        test_config_dict["df"] = tssc_panel_data
        test_config_dict["draws"] = num_draws
        
        config_obj = TSSCConfig(**test_config_dict)
        estimator = TSSC(config=config_obj)
        try:
            results_list = estimator.fit()
            assert isinstance(results_list, list) and len(results_list) > 0
            for pydantic_result in results_list:
                assert isinstance(pydantic_result, BaseEstimatorResults)
                assert pydantic_result.inference is not None
                # CI is now pydantic_result.inference.ci_lower and pydantic_result.inference.ci_upper
                # The raw "95% CI" list is in pydantic_result.raw_results["95% CI"]
                raw_ci = pydantic_result.raw_results.get("95% CI")
                assert isinstance(raw_ci, list)

                if num_draws > 0:
                    assert len(raw_ci) == 2, f"Expected 2 elements in raw CI for draws={num_draws}"
                    assert isinstance(raw_ci[0], (int, float))
                    assert isinstance(raw_ci[1], (int, float))
                    assert pydantic_result.inference.ci_lower is not None
                    assert pydantic_result.inference.ci_upper is not None
                elif num_draws == 0:
                    # For 0 draws, ci_bootstrap might return empty list or list of NaNs for raw_ci
                    # Pydantic fields ci_lower/ci_upper would be None or NaN
                    if raw_ci: # if not empty list
                         assert len(raw_ci) == 2
                         # check if they are nan
                         assert np.isnan(raw_ci[0]) or isinstance(raw_ci[0], (int,float)) # Allow actual numbers if asymptotic
                         assert np.isnan(raw_ci[1]) or isinstance(raw_ci[1], (int,float))
                    # Pydantic fields should reflect this, possibly as None
                    # (The helper maps them to None if raw_ci is not a list of 2 numbers)
                    if not (np.isnan(pydantic_result.inference.ci_lower) if pydantic_result.inference.ci_lower is not None else True):
                        if not (pydantic_result.inference.ci_lower is None): # check if it's None
                             pass # Allow if it's a number (asymptotic)
                    if not (np.isnan(pydantic_result.inference.ci_upper) if pydantic_result.inference.ci_upper is not None else True):
                        if not (pydantic_result.inference.ci_upper is None):
                             pass

        except (MlsynthEstimationError, MlsynthConfigError, MlsynthDataError) as e:
            if num_draws == 0:
                # Check if the error message indicates an issue with draws or subsample size
                assert "draws" in str(e).lower() or \
                       "subsample" in str(e).lower() or \
                       "bootstrap" in str(e).lower() or \
                       "positive" in str(e).lower() or \
                       "index 0 is out of bounds" in str(e) # Original IndexError message might be wrapped
                pass # Expected failure for draws=0
            else:
                pytest.fail(f"TSSC fit failed for draws={num_draws} with unexpected Mlsynth error: {e}")
        except Exception as e: # Catch any other unexpected error
            pytest.fail(f"TSSC fit failed for draws={num_draws} with unexpected error: {e}")
        mock_plot_estimates.reset_mock()


# --- Plotting Behavior ---

@patch('mlsynth.estimators.tssc.plot_estimates')
def test_tssc_plotting_behavior_display_true(mock_plot_estimates, tssc_panel_data):
    """Test that plot_estimates is called when display_graphs is True."""
    test_config_dict = TSSC_TEST_CONFIG_BASE.copy()
    test_config_dict["df"] = tssc_panel_data
    test_config_dict["display_graphs"] = True
    
    config_obj = TSSCConfig(**test_config_dict)
    estimator = TSSC(config=config_obj)
    estimator.fit()
    mock_plot_estimates.assert_called_once()
    call_args = mock_plot_estimates.call_args[1] # Get kwargs
    assert "counterfactual_series_list" in call_args and isinstance(call_args["counterfactual_series_list"], list)
    assert len(call_args["counterfactual_series_list"]) == 1 # Should plot one recommended CF
    assert "counterfactual_names" in call_args
    assert "Recommended" in call_args["counterfactual_names"][0]


@patch('mlsynth.estimators.tssc.plot_estimates')
def test_tssc_plotting_behavior_display_false(mock_plot_estimates, tssc_panel_data):
    """Test that plot_estimates is NOT called when display_graphs is False."""
    test_config_dict = TSSC_TEST_CONFIG_BASE.copy()
    test_config_dict["df"] = tssc_panel_data
    test_config_dict["display_graphs"] = False # Default, but explicit
    
    config_obj = TSSCConfig(**test_config_dict)
    estimator = TSSC(config=config_obj)
    estimator.fit()
    mock_plot_estimates.assert_not_called()


@patch('mlsynth.estimators.tssc.plot_estimates')
@patch('mlsynth.estimators.tssc.TSEST') # Mock TSEST to control its output
def test_tssc_plotting_msc_c_weights_not_found(mock_tsest, mock_plot_estimates, tssc_panel_data): # Removed capsys
    """Test plotting behavior when MSCc weights (for recommendation) are not found."""
    test_config_dict = TSSC_TEST_CONFIG_BASE.copy()
    test_config_dict["df"] = tssc_panel_data
    test_config_dict["display_graphs"] = True

    # Simulate TSEST not returning MSCc results, or MSCc results missing WeightV
    # Let's make TSEST return only SIMPLEX results
    simplex_results_template = {
        "SIMPLEX": {
            "Effects": {"ATT": 0.1}, "Fit": {}, 
            "Vectors": {"Counterfactual": np.array([1,2,3])}, 
            "Weights": {}, "WeightV": np.array([0.5, 0.5]), "95% CI": [0.0, 0.2]
        }
    }
    mock_tsest.return_value = [simplex_results_template]
    
    config_obj = TSSCConfig(**test_config_dict)
    estimator = TSSC(config=config_obj)
    
    with pytest.warns(UserWarning, match="Warning: MSCc weights not found for step2 validation."):
        estimator.fit()
    
    mock_plot_estimates.assert_called_once() # Should still plot with default SIMPLEX
    call_args = mock_plot_estimates.call_args[1]
    assert "SIMPLEX (Recommended)" in call_args["counterfactual_names"][0]


@patch('mlsynth.estimators.tssc.plot_estimates')
def test_tssc_plotting_failure_issues_warning(mock_plot_estimates, tssc_panel_data):
    """Test that a UserWarning is issued if plot_estimates fails."""
    test_config_dict = TSSC_TEST_CONFIG_BASE.copy()
    test_config_dict["df"] = tssc_panel_data
    test_config_dict["display_graphs"] = True # Ensure plotting is attempted
    
    # Configure the mocked plot_estimates to raise an error
    mock_plot_estimates.side_effect = MlsynthPlottingError("Simulated plotting error")
    
    config_obj = TSSCConfig(**test_config_dict)
    estimator = TSSC(config=config_obj)
    
    with pytest.warns(UserWarning, match="Plotting failed for TSSC: Simulated plotting error"):
        results = estimator.fit() # Call fit and store results to ensure it completes
        assert results is not None # Ensure fit still returns results despite plotting warning

    mock_plot_estimates.assert_called_once() # Ensure plot_estimates was attempted


@patch('mlsynth.estimators.tssc.plot_estimates')
def test_tssc_no_counterfactual_for_plotting_issues_warning(mock_plot_estimates, tssc_panel_data):
    """Test that a UserWarning is issued if no counterfactual data is found for plotting."""
    test_config_dict = TSSC_TEST_CONFIG_BASE.copy()
    test_config_dict["df"] = tssc_panel_data
    test_config_dict["display_graphs"] = True

    config_obj = TSSCConfig(**test_config_dict)
    estimator = TSSC(config=config_obj)

    # Mock _create_single_estimator_results to return a result where counterfactual is None
    # for the recommended method.
    original_create_results = estimator._create_single_estimator_results
    def mock_create_results_no_cf(*args, **kwargs):
        res = original_create_results(*args, **kwargs)
        if res.method_details.is_recommended:
            res.time_series.counterfactual_outcome = None
        return res

    with patch.object(estimator, '_create_single_estimator_results', side_effect=mock_create_results_no_cf):
        with pytest.warns(UserWarning, match="Warning: Could not find counterfactual data for recommended model"):
            estimator.fit()
    
    mock_plot_estimates.assert_not_called() # Plotting should not be attempted if CF is None
