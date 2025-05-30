import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional 
from pydantic import ValidationError

from mlsynth import SCMO
from mlsynth.exceptions import (
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthConfigError, # Though not directly tested for raising here, good to have if needed
)
from mlsynth.config_models import (
    SCMOConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
    InferenceResults,
    MethodDetailsResults,
)

# Full configuration dictionary used in tests.
SCMO_FULL_TEST_CONFIG_BASE: Dict[str, Any] = {
    "outcome": "Y_main",
    "treat": "treated_indicator_programmatic_scmo",
    "unitid": "unit_id",
    "time": "time_id",
    "addout": "Y_aux1", 
    "method": "TLP", 
    "counterfactual_color": "cyan",
    "treated_color": "magenta",
    "display_graphs": False,
    "save": False,
    "seed": 13579, # Not part of SCMOConfig, but used in original tests
    "verbose": False, # Not part of SCMOConfig
}

# Fields that are part of SCMOConfig (BaseEstimatorConfig + SCMO specific)
SCMO_PYDANTIC_MODEL_FIELDS = [
    "df", "outcome", "treat", "unitid", "time", 
    "display_graphs", "save", "counterfactual_color", "treated_color",
    "addout", "method" 
]

def _get_pydantic_config_dict_scmo(full_config: Dict[str, Any], df_fixture: pd.DataFrame) -> Dict[str, Any]:
    """Helper to extract Pydantic-valid fields for SCMOConfig and add the DataFrame."""
    pydantic_dict = {
        k: v for k, v in full_config.items() if k in SCMO_PYDANTIC_MODEL_FIELDS
    }
    pydantic_dict["df"] = df_fixture
    return pydantic_dict

@pytest.fixture
def basic_panel_data_with_treatment_scmo():
    """Provides a panel dataset with main and auxiliary outcomes for SCMO smoke testing."""
    data_dict = {
        'unit_id': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4], # 4 units, 5 periods
        'time_id': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'Y_main':  [10,11,15,16,17, 9, 10,11,12,13, 12,13,14,15,16, 11,12,13,14,15], # Main outcome
        'Y_aux1':  [20,22,28,30,32, 18,20,22,24,26, 22,24,26,28,30, 21,23,25,27,29], # Auxiliary outcome
        'X1':      [5, 6, 7, 8, 9,  4, 5, 6, 7, 8,  6, 7, 8, 9,10,  5, 6, 7, 8, 9], # A predictor
    }
    df = pd.DataFrame(data_dict)
    
    treatment_col_name = SCMO_FULL_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    # Unit 1 is treated starting from time_id = 4
    df.loc[(df['unit_id'] == 1) & (df['time_id'] >= 4), treatment_col_name] = 1
    return df

def test_scmo_creation(basic_panel_data_with_treatment_scmo: pd.DataFrame):
    """Test that the SCMO estimator can be instantiated."""
    pydantic_dict = _get_pydantic_config_dict_scmo(SCMO_FULL_TEST_CONFIG_BASE, basic_panel_data_with_treatment_scmo)
    
    try:
        config_obj = SCMOConfig(**pydantic_dict)
        estimator = SCMO(config=config_obj)
        assert estimator is not None, "SCMO estimator should be created."
        assert estimator.outcome == "Y_main"
        assert estimator.treat == SCMO_FULL_TEST_CONFIG_BASE["treat"]
        assert estimator.method == "TLP"
        assert estimator.addout == "Y_aux1" # SCMOConfig has addout
        assert not estimator.display_graphs
    except Exception as e:
        pytest.fail(f"SCMO instantiation failed: {e}")

def test_scmo_fit_smoke_tlp(basic_panel_data_with_treatment_scmo: pd.DataFrame):
    """Smoke test for the SCMO fit method (TLP) to ensure it runs without crashing."""
    full_config_dict = SCMO_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict["method"] = "TLP"
    pydantic_dict = _get_pydantic_config_dict_scmo(full_config_dict, basic_panel_data_with_treatment_scmo)
    config_obj = SCMOConfig(**pydantic_dict)
    estimator = SCMO(config=config_obj)
    
    try:
        results = estimator.fit()
        assert isinstance(results, BaseEstimatorResults), "Fit method should return BaseEstimatorResults."
        assert results.weights is not None
        assert results.effects is not None
        assert results.fit_diagnostics is not None
        assert results.time_series is not None
        assert results.inference is not None
        assert results.method_details is not None

        assert isinstance(results.weights.donor_weights, dict)
        assert isinstance(results.effects.att, (float, np.floating)) # Or check for specific effect
        assert isinstance(results.fit_diagnostics.pre_treatment_rmse, (float, np.floating))
        assert isinstance(results.time_series.counterfactual_outcome, np.ndarray)
        assert isinstance(results.inference.details, np.ndarray) # Conformal Prediction details

    except Exception as e:
        pytest.fail(f"SCMO fit method (TLP) failed during smoke test: {e}")

def test_scmo_fit_smoke_sbmf(basic_panel_data_with_treatment_scmo):
    """Smoke test for the SCMO fit method (SBMF) to ensure it runs without crashing."""
    full_config_dict = SCMO_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict["method"] = "SBMF"
    pydantic_dict = _get_pydantic_config_dict_scmo(full_config_dict, basic_panel_data_with_treatment_scmo)
    config_obj = SCMOConfig(**pydantic_dict)
    estimator = SCMO(config=config_obj)
    
    try:
        results = estimator.fit()
        assert isinstance(results, BaseEstimatorResults), "Fit method (SBMF) should return BaseEstimatorResults."
        assert results.weights is not None
        assert results.effects is not None
        assert results.fit_diagnostics is not None
        assert results.time_series is not None
        assert results.inference is not None
        assert results.method_details is not None

        assert isinstance(results.weights.donor_weights, dict)
        assert isinstance(results.effects.att, (float, np.floating))
        assert isinstance(results.fit_diagnostics.pre_treatment_rmse, (float, np.floating))
        assert isinstance(results.time_series.counterfactual_outcome, np.ndarray)
        assert isinstance(results.inference.details, np.ndarray)

    except Exception as e:
        pytest.fail(f"SCMO fit method (SBMF) failed during smoke test: {e}")

def test_scmo_fit_smoke_both(basic_panel_data_with_treatment_scmo):
    """Smoke test for the SCMO fit method (BOTH) to ensure it runs without crashing."""
    full_config_dict = SCMO_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict["method"] = "BOTH"
    pydantic_dict = _get_pydantic_config_dict_scmo(full_config_dict, basic_panel_data_with_treatment_scmo)
    config_obj = SCMOConfig(**pydantic_dict)
    estimator = SCMO(config=config_obj)
    
    try:
        results = estimator.fit()
        assert isinstance(results, BaseEstimatorResults), "Fit method (BOTH) should return BaseEstimatorResults."
        assert results.effects is not None
        assert results.fit_diagnostics is not None
        assert results.time_series is not None
        assert results.inference is not None
        assert results.weights is not None
        assert results.method_details is not None
        assert results.method_details.additional_details is not None
        assert "lambdas" in results.method_details.additional_details

        assert isinstance(results.effects.att, (float, np.floating))
        assert isinstance(results.fit_diagnostics.pre_treatment_rmse, (float, np.floating))
        assert isinstance(results.time_series.counterfactual_outcome, np.ndarray)
        assert isinstance(results.inference.details, np.ndarray) # Conformal Prediction
        
        assert isinstance(results.weights.donor_weights, dict)
        assert results.weights.summary_stats is not None
        assert "positive_weights" in results.weights.summary_stats
        assert isinstance(results.weights.summary_stats["positive_weights"], dict)
        
        lambdas = results.method_details.additional_details["lambdas"]
        assert isinstance(lambdas, dict)
        assert "TLP" in lambdas
        assert "SBMF" in lambdas


    except Exception as e:
        pytest.fail(f"SCMO fit method (BOTH) failed during smoke test: {e}")

# --- Input Validation Tests ---

def test_scmo_invalid_method(basic_panel_data_with_treatment_scmo: pd.DataFrame):
    """Test SCMOConfig instantiation fails with an invalid method."""
    full_config_dict = SCMO_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict["method"] = "INVALID_METHOD"
    pydantic_dict = _get_pydantic_config_dict_scmo(full_config_dict, basic_panel_data_with_treatment_scmo)
    
    with pytest.raises(ValidationError): # SCMOConfig has pattern validation for method
        SCMOConfig(**pydantic_dict)

# Only include fields that are truly required (no defaults in SCMOConfig or BaseEstimatorConfig)
@pytest.mark.parametrize("missing_key", ["df", "outcome", "treat", "unitid", "time"])
def test_scmo_missing_essential_config(basic_panel_data_with_treatment_scmo: pd.DataFrame, missing_key: str):
    """Test SCMOConfig instantiation fails if essential config keys are missing."""
    full_config_dict = SCMO_FULL_TEST_CONFIG_BASE.copy()
    original_value = full_config_dict.pop(missing_key, None) 
    
    pydantic_dict = _get_pydantic_config_dict_scmo(full_config_dict, basic_panel_data_with_treatment_scmo)
    # Ensure the key is truly removed from the dict passed to Pydantic
    if missing_key in pydantic_dict:
        del pydantic_dict[missing_key]

    with pytest.raises(ValidationError):
        SCMOConfig(**pydantic_dict)
    
    if original_value is not None: # Restore for other tests if fixture is session-scoped (not an issue here)
        full_config_dict[missing_key] = original_value


@pytest.mark.parametrize("col_key, wrong_col_name", [
    ("outcome", "NonExistent_Y_main"),
    ("treat", "NonExistent_Treat"),
    ("unitid", "NonExistent_Unit"),
    ("time", "NonExistent_Time"),
    ("addout", "NonExistent_Aux") 
])
def test_scmo_column_not_in_df(basic_panel_data_with_treatment_scmo: pd.DataFrame, col_key: str, wrong_col_name: str):
    """Test SCMO fit fails if specified columns are not in df.
    Pydantic config creation should pass, error expected during fit.
    """
    full_config_dict = SCMO_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict[col_key] = wrong_col_name
    
    pydantic_dict = _get_pydantic_config_dict_scmo(full_config_dict, basic_panel_data_with_treatment_scmo)

    if col_key == "addout": # addout is not checked by BaseEstimatorConfig validator
        config_obj = SCMOConfig(**pydantic_dict)
        estimator = SCMO(config=config_obj)
        # Error from dataprep, wrapped in fit()
        with pytest.raises(MlsynthDataError, match=f"Missing expected key during SCMO data processing: '{wrong_col_name}'"):
            estimator.fit()
    else: # outcome, treat, unitid, time are checked by BaseEstimatorConfig validator
        with pytest.raises(MlsynthDataError, match=f"Missing required columns in DataFrame 'df': {wrong_col_name}"):
            SCMOConfig(**pydantic_dict)


def test_scmo_addout_empty_list(basic_panel_data_with_treatment_scmo: pd.DataFrame):
    """Test SCMO runs with addout as an empty list (uses only main outcome)."""
    full_config_dict = SCMO_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict["addout"] = [] 
    
    pydantic_dict = _get_pydantic_config_dict_scmo(full_config_dict, basic_panel_data_with_treatment_scmo)
    config_obj = SCMOConfig(**pydantic_dict)
    estimator = SCMO(config=config_obj)
    try:
        results = estimator.fit()
        assert results is not None
    except Exception as e:
        pytest.fail(f"SCMO fit failed with empty list addout: {e}")

def test_scmo_addout_empty_string(basic_panel_data_with_treatment_scmo: pd.DataFrame):
    """Test SCMO fit fails when addout is an empty string (treated as missing column)."""
    full_config_dict = SCMO_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict["addout"] = "" 
                               
    pydantic_dict = _get_pydantic_config_dict_scmo(full_config_dict, basic_panel_data_with_treatment_scmo)
    config_obj = SCMOConfig(**pydantic_dict) # SCMOConfig allows str for addout
    estimator = SCMO(config=config_obj)
    # dataprep will try to use "" as a column name, which becomes a KeyError, wrapped by fit()
    with pytest.raises(MlsynthDataError, match="Missing expected key during SCMO data processing: ''"):
        estimator.fit()


def test_scmo_addout_list_with_nonexistent_col(basic_panel_data_with_treatment_scmo: pd.DataFrame):
    """Test SCMO fit fails if addout list contains a non-existent column."""
    full_config_dict = SCMO_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict["addout"] = ["Y_aux1", "NonExistent_Aux2"]
    
    pydantic_dict = _get_pydantic_config_dict_scmo(full_config_dict, basic_panel_data_with_treatment_scmo)
    config_obj = SCMOConfig(**pydantic_dict) # SCMOConfig allows list of str
    estimator = SCMO(config=config_obj)
    # dataprep for "NonExistent_Aux2" should fail with KeyError, wrapped by fit()
    with pytest.raises(MlsynthDataError, match="Missing expected key during SCMO data processing: 'NonExistent_Aux2'"):
        estimator.fit()

@pytest.fixture
def data_with_nan_in_outcome_scmo(basic_panel_data_with_treatment_scmo: pd.DataFrame):
    """Modifies basic data to include NaN in one of the outcome variables."""
    df_nan = basic_panel_data_with_treatment_scmo.copy()
    df_nan.loc[(df_nan['unit_id'] == 2) & (df_nan['time_id'] == 3), 'Y_main'] = np.nan
    return df_nan

def test_scmo_nan_in_stacked_data(data_with_nan_in_outcome_scmo: pd.DataFrame):
    """Test SCMO fit fails if stacked data contains NaN."""
    full_config_dict = SCMO_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict["method"] = "TLP" 
    
    pydantic_dict = _get_pydantic_config_dict_scmo(full_config_dict, data_with_nan_in_outcome_scmo)
    config_obj = SCMOConfig(**pydantic_dict)
    estimator = SCMO(config=config_obj)
    with pytest.raises(MlsynthDataError, match="Stacked data contains non-numeric values"):
        estimator.fit()

# --- Tests for edge cases in data structure ---

@pytest.fixture
def single_control_unit_data_scmo():
    """Panel data with only one control unit for SCMO."""
    data_dict = {
        'unit_id': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2], # Unit 1 treated, Unit 2 control
        'time_id': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'Y_main':  [10,11,15,16,17, 9, 10,11,12,13],
        'Y_aux1':  [20,22,28,30,32, 18,20,22,24,26],
    }
    df = pd.DataFrame(data_dict)
    treatment_col_name = SCMO_FULL_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    df.loc[(df['unit_id'] == 1) & (df['time_id'] >= 4), treatment_col_name] = 1
    return df

def test_scmo_single_control_unit(single_control_unit_data_scmo: pd.DataFrame):
    """Test SCMO with only one control unit."""
    full_config_dict = SCMO_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict["method"] = "TLP"
    pydantic_dict = _get_pydantic_config_dict_scmo(full_config_dict, single_control_unit_data_scmo)
    config_obj = SCMOConfig(**pydantic_dict)
    estimator = SCMO(config=config_obj)
    try:
        results = estimator.fit()
        assert isinstance(results, BaseEstimatorResults)
        assert results.weights is not None
        assert results.weights.donor_weights is not None
        assert len(results.weights.donor_weights) == 1
        assert list(results.weights.donor_weights.keys())[0] == str(single_control_unit_data_scmo['unit_id'].unique()[1])
    except Exception as e:
        pytest.fail(f"SCMO fit failed with single control unit: {e}")

@pytest.fixture
def no_control_units_data_scmo():
    """Panel data with no control units for SCMO."""
    data_dict = {
        'unit_id': [1, 1, 1, 1, 1], 
        'time_id': [1, 2, 3, 4, 5],
        'Y_main':  [10,11,15,16,17],
        'Y_aux1':  [20,22,28,30,32],
    }
    df = pd.DataFrame(data_dict)
    treatment_col_name = SCMO_FULL_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    df.loc[(df['unit_id'] == 1) & (df['time_id'] >= 4), treatment_col_name] = 1
    return df

def test_scmo_no_control_units(no_control_units_data_scmo: pd.DataFrame):
    """Test SCMO fails gracefully when there are no control units."""
    pydantic_dict = _get_pydantic_config_dict_scmo(SCMO_FULL_TEST_CONFIG_BASE, no_control_units_data_scmo)
    config_obj = SCMOConfig(**pydantic_dict)
    estimator = SCMO(config=config_obj)
    with pytest.raises(MlsynthDataError, match="No donor units found"):
        estimator.fit()

@pytest.fixture
def few_pre_periods_data_scmo():
    """Panel data with very few pre-treatment periods for SCMO."""
    data_dict = { 
        'unit_id': [1, 1, 1, 2, 2, 2, 3, 3, 3], 
        'time_id': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'Y_main':  [10,16,17, 9, 10,11, 12,13,14],
        'Y_aux1':  [20,30,32, 18,20,22, 22,24,26],
    }
    df = pd.DataFrame(data_dict)
    treatment_col_name = SCMO_FULL_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    df.loc[(df['unit_id'] == 1) & (df['time_id'] >= 2), treatment_col_name] = 1 # Treat at T=2 (1 pre period)
    return df

def test_scmo_few_pre_periods(few_pre_periods_data_scmo: pd.DataFrame):
    """Test SCMO with few pre-treatment periods."""
    full_config_dict = SCMO_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict["method"] = "TLP"
    pydantic_dict = _get_pydantic_config_dict_scmo(full_config_dict, few_pre_periods_data_scmo)
    config_obj = SCMOConfig(**pydantic_dict)
    estimator = SCMO(config=config_obj)
    try:
        results = estimator.fit()
        assert results is not None
    except Exception as e:
        pytest.fail(f"SCMO fit failed with few pre-treatment periods: {e}")


@pytest.fixture
def few_post_periods_data_scmo():
    """Panel data with very few post-treatment periods for SCMO."""
    data_dict = { 
        'unit_id': [1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3], 
        'time_id': [1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5],
        'Y_main':  [10,11,12,13,17, 9,10,11,12,13, 12,13,14,15,16],
        'Y_aux1':  [20,22,24,26,32, 18,20,22,24,26, 22,24,26,28,30],
    }
    df = pd.DataFrame(data_dict)
    treatment_col_name = SCMO_FULL_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    df.loc[(df['unit_id'] == 1) & (df['time_id'] >= 5), treatment_col_name] = 1 # Treat at T=5 (1 post period)
    return df

def test_scmo_few_post_periods(few_post_periods_data_scmo: pd.DataFrame):
    """Test SCMO with few post-treatment periods."""
    full_config_dict = SCMO_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict["method"] = "TLP"
    pydantic_dict = _get_pydantic_config_dict_scmo(full_config_dict, few_post_periods_data_scmo)
    config_obj = SCMOConfig(**pydantic_dict)
    estimator = SCMO(config=config_obj)
    try:
        results = estimator.fit()
        assert isinstance(results, BaseEstimatorResults)
        assert results.effects is not None
        assert results.effects.additional_effects is not None
        att_time = results.effects.additional_effects.get("ATT_Time")
        assert att_time is not None
        assert len(att_time) == 1
    except Exception as e:
        pytest.fail(f"SCMO fit failed with few post-treatment periods: {e}")

@pytest.fixture
def no_pre_periods_data_scmo():
    """Panel data where treatment starts at the first period for SCMO."""
    data_dict = { 
        'unit_id': [1, 1, 1, 2, 2, 2, 3, 3, 3], 
        'time_id': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'Y_main':  [16,17,18, 9, 10,11, 12,13,14],
        'Y_aux1':  [30,32,34, 18,20,22, 22,24,26],
    }
    df = pd.DataFrame(data_dict)
    treatment_col_name = SCMO_FULL_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    df.loc[(df['unit_id'] == 1) & (df['time_id'] >= 1), treatment_col_name] = 1 # Treat at T=1 (0 pre periods)
    return df

def test_scmo_no_pre_periods(no_pre_periods_data_scmo: pd.DataFrame):
    """Test SCMO fails or handles no pre-treatment periods."""
    pydantic_dict = _get_pydantic_config_dict_scmo(SCMO_FULL_TEST_CONFIG_BASE, no_pre_periods_data_scmo)
    config_obj = SCMOConfig(**pydantic_dict)
    estimator = SCMO(config=config_obj)
    with pytest.raises(MlsynthDataError, match="Not enough pre-treatment periods"): # dataprep raises this
        estimator.fit()

@pytest.fixture
def no_post_periods_data_scmo():
    """Panel data where treatment starts after the last observed period for SCMO."""
    data_dict = {
        'unit_id': [1, 1, 1, 2, 2, 2, 3, 3, 3], 
        'time_id': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'Y_main':  [10,11,12, 9, 10,11, 12,13,14],
        'Y_aux1':  [20,22,24, 18,20,22, 22,24,26],
    }
    df = pd.DataFrame(data_dict)
    treatment_col_name = SCMO_FULL_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    df.loc[(df['unit_id'] == 1) & (df['time_id'] >= 4), treatment_col_name] = 1 # Treat at T=4, max T=3
    return df

def test_scmo_no_post_periods(no_post_periods_data_scmo: pd.DataFrame):
    """Test SCMO fails or handles no post-treatment periods."""
    pydantic_dict = _get_pydantic_config_dict_scmo(SCMO_FULL_TEST_CONFIG_BASE, no_post_periods_data_scmo)
    config_obj = SCMOConfig(**pydantic_dict)
    estimator = SCMO(config=config_obj)
    # logictreat (called by dataprep) raises MlsynthDataError if no treated units are found
    with pytest.raises(MlsynthDataError, match="No treated units found"): 
        estimator.fit()

# --- Tests for detailed results validation ---

def check_single_method_results_structure(results: BaseEstimatorResults, method_name: str, panel_data: pd.DataFrame, config_dict: dict):
    """Helper function to check results structure for single SCMO methods (TLP, SBMF)."""
    assert isinstance(results, BaseEstimatorResults)

    # Method Details
    assert results.method_details is not None
    assert results.method_details.name == method_name
    assert results.method_details.parameters_used is not None
    assert results.method_details.parameters_used["method"] == method_name

    # Weights
    assert results.weights is not None
    assert isinstance(results.weights.donor_weights, dict)
    assert all(isinstance(k, str) for k in results.weights.donor_weights.keys())
    assert all(isinstance(v, (float, np.floating)) for v in results.weights.donor_weights.values())

    # Effects
    assert results.effects is not None
    assert results.effects.att is not None
    assert results.effects.att_percent is not None
    assert results.effects.additional_effects is not None
    expected_effects_keys = ["SATT", "TTE", "ATT_Time", "PercentATT_Time", "SATT_Time"]
    for key in expected_effects_keys:
        assert key in results.effects.additional_effects, f"Expected key '{key}' not in {method_name} Effects additional_effects"
    
    # Fit Diagnostics
    assert results.fit_diagnostics is not None
    assert results.fit_diagnostics.pre_treatment_rmse is not None
    assert results.fit_diagnostics.pre_treatment_r_squared is not None
    assert results.fit_diagnostics.additional_metrics is not None
    expected_fit_keys = ["T1 RMSE", "Pre-Periods", "Post-Periods"] # "T0 RMSE", "R-Squared" are main fields
    for key in expected_fit_keys:
        assert key in results.fit_diagnostics.additional_metrics, f"Expected key '{key}' not in {method_name} Fit additional_metrics"

    # Time Series
    assert results.time_series is not None
    expected_vector_keys = ["observed_outcome", "counterfactual_outcome", "estimated_gap", "time_periods"]
    for key in expected_vector_keys:
        assert getattr(results.time_series, key) is not None, f"Expected attribute '{key}' not in {method_name} TimeSeriesResults"
        assert isinstance(getattr(results.time_series, key), np.ndarray)
    
    # Inference
    assert results.inference is not None
    assert results.inference.method == "conformal"
    assert results.inference.confidence_level is not None
    assert isinstance(results.inference.details, np.ndarray) # Conformal Prediction intervals
    assert results.inference.details.shape[1] == 2 # Lower and Upper bounds

    # Check vector lengths
    num_time_periods_treated = panel_data[panel_data[config_dict["unitid"]] == 1][config_dict["time"]].nunique()
    num_post_periods_treated = panel_data[
        (panel_data[config_dict["unitid"]] == 1) & (panel_data[config_dict["treat"]] == 1)
    ][config_dict["time"]].nunique()

    assert len(results.time_series.observed_outcome) == num_time_periods_treated
    assert len(results.time_series.counterfactual_outcome) == num_time_periods_treated
    assert len(results.time_series.estimated_gap) == num_time_periods_treated
    assert len(results.time_series.time_periods) == num_time_periods_treated
    
    att_time = results.effects.additional_effects.get("ATT_Time")
    assert att_time is not None and len(att_time) == num_post_periods_treated
    assert results.inference.details.shape[0] == num_post_periods_treated


def test_scmo_fit_results_tlp(basic_panel_data_with_treatment_scmo: pd.DataFrame):
    """Detailed test for SCMO fit results (TLP method)."""
    full_config_dict = SCMO_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict["method"] = "TLP"
    pydantic_dict = _get_pydantic_config_dict_scmo(full_config_dict, basic_panel_data_with_treatment_scmo)
    config_obj = SCMOConfig(**pydantic_dict)
    estimator = SCMO(config=config_obj)
    results = estimator.fit()
    check_single_method_results_structure(results, "TLP", basic_panel_data_with_treatment_scmo, pydantic_dict)

def test_scmo_fit_results_sbmf(basic_panel_data_with_treatment_scmo: pd.DataFrame):
    """Detailed test for SCMO fit results (SBMF method)."""
    full_config_dict = SCMO_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict["method"] = "SBMF"
    pydantic_dict = _get_pydantic_config_dict_scmo(full_config_dict, basic_panel_data_with_treatment_scmo)
    config_obj = SCMOConfig(**pydantic_dict)
    estimator = SCMO(config=config_obj)
    results = estimator.fit()
    check_single_method_results_structure(results, "SBMF", basic_panel_data_with_treatment_scmo, pydantic_dict)

def test_scmo_fit_results_both(basic_panel_data_with_treatment_scmo: pd.DataFrame):
    """Detailed test for SCMO fit results (BOTH method - model averaging)."""
    full_config_dict = SCMO_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict["method"] = "BOTH"
    pydantic_dict = _get_pydantic_config_dict_scmo(full_config_dict, basic_panel_data_with_treatment_scmo)
    config_obj = SCMOConfig(**pydantic_dict)
    estimator = SCMO(config=config_obj)
    results = estimator.fit()

    assert isinstance(results, BaseEstimatorResults)

    # Method Details
    assert results.method_details is not None
    assert results.method_details.name == "SCMO_MA" # Specific name for BOTH
    assert results.method_details.parameters_used is not None
    assert results.method_details.parameters_used["method"] == "BOTH"
    assert results.method_details.additional_details is not None
    lambdas = results.method_details.additional_details.get("lambdas")
    assert isinstance(lambdas, dict)
    assert "TLP" in lambdas and "SBMF" in lambdas
    assert isinstance(lambdas["TLP"], (float, np.floating))
    assert isinstance(lambdas["SBMF"], (float, np.floating))
    assert np.isclose(lambdas["TLP"] + lambdas["SBMF"], 1.0)

    # Weights
    assert results.weights is not None
    assert isinstance(results.weights.donor_weights, dict) # Averaged weights
    assert all(isinstance(k, str) for k in results.weights.donor_weights.keys())
    assert results.weights.summary_stats is not None
    assert "positive_weights" in results.weights.summary_stats
    assert isinstance(results.weights.summary_stats["positive_weights"], dict) # Positive averaged weights
    assert all(isinstance(k, str) for k in results.weights.summary_stats["positive_weights"].keys())

    # Effects, Fit, TimeSeries, Inference (similar structure to single method)
    assert results.effects is not None
    assert results.effects.att is not None
    assert results.effects.additional_effects is not None
    expected_effects_keys = ["SATT", "TTE", "ATT_Time", "PercentATT_Time", "SATT_Time"] # ATT, Percent ATT are main
    for key in expected_effects_keys:
        assert key in results.effects.additional_effects, f"Expected key '{key}' not in BOTH Effects additional_effects"
    
    assert results.fit_diagnostics is not None
    assert results.fit_diagnostics.pre_treatment_rmse is not None
    assert results.fit_diagnostics.additional_metrics is not None
    expected_fit_keys = ["T1 RMSE", "Pre-Periods", "Post-Periods"] # T0 RMSE, R-Squared are main
    for key in expected_fit_keys:
        assert key in results.fit_diagnostics.additional_metrics, f"Expected key '{key}' not in BOTH Fit additional_metrics"

    assert results.time_series is not None
    expected_vector_keys = ["observed_outcome", "counterfactual_outcome", "estimated_gap", "time_periods"]
    for key in expected_vector_keys:
        assert getattr(results.time_series, key) is not None, f"Expected attribute '{key}' not in BOTH TimeSeriesResults"
        assert isinstance(getattr(results.time_series, key), np.ndarray)
    
    assert results.inference is not None
    assert results.inference.method == "conformal"
    assert results.inference.confidence_level is not None
    assert isinstance(results.inference.details, np.ndarray) # Conformal Prediction intervals
    assert results.inference.details.shape[1] == 2

    # Check vector lengths
    num_time_periods_treated = basic_panel_data_with_treatment_scmo[
        basic_panel_data_with_treatment_scmo[pydantic_dict["unitid"]] == 1
    ][pydantic_dict["time"]].nunique()
    num_post_periods_treated = basic_panel_data_with_treatment_scmo[
        (basic_panel_data_with_treatment_scmo[pydantic_dict["unitid"]] == 1) & 
        (basic_panel_data_with_treatment_scmo[pydantic_dict["treat"]] == 1)
    ][pydantic_dict["time"]].nunique()

    assert len(results.time_series.observed_outcome) == num_time_periods_treated
    assert len(results.time_series.counterfactual_outcome) == num_time_periods_treated
    assert len(results.time_series.estimated_gap) == num_time_periods_treated
    assert len(results.time_series.time_periods) == num_time_periods_treated
    
    att_time = results.effects.additional_effects.get("ATT_Time")
    assert att_time is not None and len(att_time) == num_post_periods_treated
    assert results.inference.details.shape[0] == num_post_periods_treated

# TODO: Add tests for specific weight values or effect sizes if known/stable.
