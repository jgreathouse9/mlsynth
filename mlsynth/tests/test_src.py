import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any
from pydantic import ValidationError

from mlsynth import SRC
from mlsynth.config_models import SRCConfig, BaseEstimatorResults
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError

# Full configuration dictionary used in tests.
# Pydantic-valid fields will be extracted for SRCConfig instantiation.
SRC_FULL_TEST_CONFIG_BASE: Dict[str, Any] = {
    "outcome": "Y",
    "treat": "treated_indicator_programmatic_src",
    "unitid": "unit_id",
    "time": "time_id",
    "counterfactual_color": "purple",
    "treated_color": "orange",
    "display_graphs": False,
    "save": False,
    "seed": 54321, 
    "verbose": False,
}

# Fields that are part of BaseEstimatorConfig (and thus SRCConfig)
SRC_PYDANTIC_MODEL_FIELDS = [
    "df", "outcome", "treat", "unitid", "time", 
    "display_graphs", "save", "counterfactual_color", "treated_color"
]

def _get_pydantic_config_dict_src(full_config: Dict[str, Any], df_fixture: pd.DataFrame) -> Dict[str, Any]:
    """Helper to extract Pydantic-valid fields for SRCConfig and add the DataFrame."""
    pydantic_dict = {
        k: v for k, v in full_config.items() if k in SRC_PYDANTIC_MODEL_FIELDS
    }
    pydantic_dict["df"] = df_fixture
    return pydantic_dict


@pytest.fixture
def basic_panel_data_with_treatment_src():
    """Provides a very basic panel dataset with a treatment column for SRC smoke testing."""
    data_dict = {
        'unit_id': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4], # 4 units, 5 periods
        'time_id': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'Y':       [10,11,15,16,17, 9, 10,11,12,13, 12,13,14,15,16, 11,12,13,14,15],
        'X1':      [5, 6, 7, 8, 9,  4, 5, 6, 7, 8,  6, 7, 8, 9,10,  5, 6, 7, 8, 9],
    }
    df = pd.DataFrame(data_dict)
    
    treatment_col_name = SRC_FULL_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    # Unit 1 is treated starting from time_id = 4
    df.loc[(df['unit_id'] == 1) & (df['time_id'] >= 4), treatment_col_name] = 1
    return df

def test_src_creation(basic_panel_data_with_treatment_src: pd.DataFrame):
    """Test that the SRC estimator can be instantiated."""
    pydantic_dict = _get_pydantic_config_dict_src(SRC_FULL_TEST_CONFIG_BASE, basic_panel_data_with_treatment_src)
    
    try:
        config_obj = SRCConfig(**pydantic_dict)
        estimator = SRC(config=config_obj)
        assert estimator is not None, "SRC estimator should be created."
        assert estimator.outcome == "Y", "Outcome attribute should be set from config."
        assert estimator.treat == SRC_FULL_TEST_CONFIG_BASE["treat"]
        assert not estimator.display_graphs, "display_graphs should be False for tests."
    except Exception as e:
        pytest.fail(f"SRC instantiation failed: {e}")

def test_src_fit_smoke(basic_panel_data_with_treatment_src: pd.DataFrame):
    """Smoke test for the SRC fit method to ensure it runs without crashing."""
    pydantic_dict = _get_pydantic_config_dict_src(SRC_FULL_TEST_CONFIG_BASE, basic_panel_data_with_treatment_src)
    config_obj = SRCConfig(**pydantic_dict)
    estimator = SRC(config=config_obj)
    
    try:
        results = estimator.fit() 
        assert results is not None, "Fit method should return BaseEstimatorResults."
        assert isinstance(results, BaseEstimatorResults), "Fit method should return BaseEstimatorResults."

        assert results.effects is not None, "Results should contain 'effects'."
        assert results.weights is not None, "Results should contain 'weights'."
        assert results.time_series is not None
        assert results.time_series.counterfactual_outcome is not None
        assert results.method_details is not None
        assert results.method_details.method_name == "SRC"
        
        assert isinstance(results.time_series.counterfactual_outcome, np.ndarray)
        assert isinstance(results.weights.donor_weights, dict) # Changed from np.ndarray
        assert isinstance(results.weights.donor_names, list)

        # Check raw_results for original structure
        assert results.raw_results is not None
        assert "Counterfactual" in results.raw_results
        assert "Weights" in results.raw_results
        assert "ATT" in results.raw_results
        assert "Fit" in results.raw_results
        assert "Vectors" in results.raw_results

    except Exception as e:
        pytest.fail(f"SRC fit method failed during smoke test: {e}")

# --- Tests for invalid configurations and data ---

def test_src_missing_df_in_config():
    """Test SRCConfig instantiation fails if 'df' is missing in config."""
    config_dict_no_df = SRC_FULL_TEST_CONFIG_BASE.copy()
    # "df" is deliberately not added
    
    pydantic_dict_attempt = {
        k: v for k, v in config_dict_no_df.items() if k in SRC_PYDANTIC_MODEL_FIELDS and k != "df"
    }
    with pytest.raises(ValidationError): # df is required by BaseEstimatorConfig
        SRCConfig(**pydantic_dict_attempt)

def test_src_missing_outcome_in_config(basic_panel_data_with_treatment_src: pd.DataFrame):
    """Test SRCConfig instantiation fails if 'outcome' is missing in config."""
    config_dict_missing_outcome = SRC_FULL_TEST_CONFIG_BASE.copy()
    del config_dict_missing_outcome["outcome"] 
    pydantic_dict = _get_pydantic_config_dict_src(config_dict_missing_outcome, basic_panel_data_with_treatment_src)
    if "outcome" in pydantic_dict: del pydantic_dict["outcome"] # Ensure it's removed for test

    with pytest.raises(ValidationError): # outcome is required
        SRCConfig(**pydantic_dict)

def test_src_outcome_not_in_df(basic_panel_data_with_treatment_src: pd.DataFrame):
    """Test SRC fit fails if 'outcome' column is not in df.
    Pydantic config creation should pass, error expected during fit.
    """
    full_config_dict = SRC_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict["outcome"] = "NonExistentOutcomeColumn"
    
    pydantic_dict = _get_pydantic_config_dict_src(full_config_dict, basic_panel_data_with_treatment_src)
    # BaseEstimatorConfig validates column existence
    with pytest.raises(MlsynthDataError, match=f"Missing required columns in DataFrame 'df': {full_config_dict['outcome']}"):
        SRCConfig(**pydantic_dict)

def test_src_missing_treat_in_config(basic_panel_data_with_treatment_src: pd.DataFrame):
    """Test SRCConfig instantiation fails if 'treat' is missing in config."""
    config_dict_missing_treat = SRC_FULL_TEST_CONFIG_BASE.copy()
    del config_dict_missing_treat["treat"]
    pydantic_dict = _get_pydantic_config_dict_src(config_dict_missing_treat, basic_panel_data_with_treatment_src)
    if "treat" in pydantic_dict: del pydantic_dict["treat"]

    with pytest.raises(ValidationError): # treat is required
        SRCConfig(**pydantic_dict)

def test_src_treat_not_in_df(basic_panel_data_with_treatment_src: pd.DataFrame):
    """Test SRC fit fails if 'treat' column is not in df.
    Pydantic config creation should pass, error expected during fit.
    """
    full_config_dict = SRC_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict["treat"] = "NonExistentTreatColumn"
    
    pydantic_dict = _get_pydantic_config_dict_src(full_config_dict, basic_panel_data_with_treatment_src)
    # BaseEstimatorConfig validates column existence
    with pytest.raises(MlsynthDataError, match=f"Missing required columns in DataFrame 'df': {full_config_dict['treat']}"):
        SRCConfig(**pydantic_dict)

# --- Tests for edge cases in data structure ---

@pytest.fixture
def single_control_unit_data_src():
    """Panel data with only one control unit."""
    data_dict = {
        'unit_id': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2], # Unit 1 treated, Unit 2 control
        'time_id': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'Y':       [10,11,15,16,17, 9, 10,11,12,13],
        'X1':      [5, 6, 7, 8, 9,  4, 5, 6, 7, 8],
    }
    df = pd.DataFrame(data_dict)
    treatment_col_name = SRC_FULL_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    df.loc[(df['unit_id'] == 1) & (df['time_id'] >= 4), treatment_col_name] = 1
    return df

def test_src_single_control_unit(single_control_unit_data_src: pd.DataFrame):
    """Test SRC with only one control unit."""
    pydantic_dict = _get_pydantic_config_dict_src(SRC_FULL_TEST_CONFIG_BASE, single_control_unit_data_src)
    config_obj = SRCConfig(**pydantic_dict)
    estimator = SRC(config=config_obj)
    try:
        results = estimator.fit()
        assert isinstance(results, BaseEstimatorResults)
        assert results.weights is not None
        assert results.weights.donor_weights is not None
        assert isinstance(results.weights.donor_weights, dict)
        assert len(results.weights.donor_weights) == 1 # Should have weight for the single donor
        # Get the single donor's name (key) and check its weight
        single_donor_name = list(results.weights.donor_weights.keys())[0]
        assert results.weights.donor_weights[single_donor_name] == pytest.approx(1.0, abs=1e-2) # Single donor gets all weight
    except Exception as e:
        pytest.fail(f"SRC fit failed with single control unit: {e}")

@pytest.fixture
def no_control_units_data_src():
    """Panel data with no control units (only a treated unit)."""
    data_dict = {
        'unit_id': [1, 1, 1, 1, 1], # Only Unit 1 (treated)
        'time_id': [1, 2, 3, 4, 5],
        'Y':       [10,11,15,16,17],
        'X1':      [5, 6, 7, 8, 9],
    }
    df = pd.DataFrame(data_dict)
    treatment_col_name = SRC_FULL_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    df.loc[(df['unit_id'] == 1) & (df['time_id'] >= 4), treatment_col_name] = 1
    return df

def test_src_no_control_units(no_control_units_data_src: pd.DataFrame):
    """Test SRC fails gracefully when there are no control units."""
    pydantic_dict = _get_pydantic_config_dict_src(SRC_FULL_TEST_CONFIG_BASE, no_control_units_data_src)
    config_obj = SRCConfig(**pydantic_dict)
    estimator = SRC(config=config_obj)
    # dataprep (in datautils.py) now raises MlsynthDataError
    with pytest.raises(MlsynthDataError, match="No donor units found"): 
        estimator.fit()

@pytest.fixture
def few_pre_periods_data_src():
    """Panel data with very few pre-treatment periods (e.g., 1)."""
    data_dict = { # Unit 1 treated at time_id = 2
        'unit_id': [1, 1, 1, 2, 2, 2, 3, 3, 3], 
        'time_id': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'Y':       [10,16,17, 9, 10,11, 12,13,14],
        'X1':      [5, 8, 9,  4, 5, 6,  6, 7, 8],
    }
    df = pd.DataFrame(data_dict)
    treatment_col_name = SRC_FULL_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    df.loc[(df['unit_id'] == 1) & (df['time_id'] >= 2), treatment_col_name] = 1
    return df

def test_src_few_pre_periods(few_pre_periods_data_src: pd.DataFrame):
    """Test SRC with few pre-treatment periods."""
    pydantic_dict = _get_pydantic_config_dict_src(SRC_FULL_TEST_CONFIG_BASE, few_pre_periods_data_src)
    config_obj = SRCConfig(**pydantic_dict)
    estimator = SRC(config=config_obj)
    # With 1 pre-period, SRCest might lead to LinAlgError if matrices are singular,
    # or effects.calculate might have issues.
    # The refactored src.py should catch these and raise MlsynthEstimationError.
    # If SRCest itself has a check for min pre-periods (e.g. >1), it would be MlsynthDataError.
    # From estutils.SRCest, it doesn't seem to have a direct check for num_pre_periods > 1.
    # It's more likely that numerical issues or issues in effects.calculate arise.
    # For now, let's assume it might complete or raise a numerical error wrapped as MlsynthEstimationError.
    # If it completes, results should be checked. If it errors, the specific error should be caught.
    # Given the complexity, we'll assert it runs and produces results.
    # If specific numerical issues arise with 1 pre-period, they'd be caught by MlsynthEstimationError.
    results = estimator.fit() # Expect MlsynthEstimationError if it fails numerically
    assert isinstance(results, BaseEstimatorResults)
    # If it ran, check that results are populated (even if potentially NaN or unusual due to few pre-periods)
    assert results.effects is not None
    assert results.time_series is not None
    assert results.weights is not None


@pytest.fixture
def few_post_periods_data_src():
    """Panel data with very few post-treatment periods (e.g., 1)."""
    data_dict = { # Unit 1 treated at time_id = 5
        'unit_id': [1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3], 
        'time_id': [1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5],
        'Y':       [10,11,12,13,17, 9,10,11,12,13, 12,13,14,15,16],
        'X1':      [5,6,7,8,9,   4,5,6,7,8,   6,7,8,9,10],
    }
    df = pd.DataFrame(data_dict)
    treatment_col_name = SRC_FULL_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    df.loc[(df['unit_id'] == 1) & (df['time_id'] >= 5), treatment_col_name] = 1
    return df

def test_src_few_post_periods(few_post_periods_data_src: pd.DataFrame):
    """Test SRC with few post-treatment periods."""
    pydantic_dict = _get_pydantic_config_dict_src(SRC_FULL_TEST_CONFIG_BASE, few_post_periods_data_src)
    config_obj = SRCConfig(**pydantic_dict)
    estimator = SRC(config=config_obj)
    try:
        results = estimator.fit()
        assert isinstance(results, BaseEstimatorResults)
        assert results.effects is not None
        raw_att_time = results.raw_results.get("ATT", {}).get("ATT_Time") # Check raw results for this
        assert raw_att_time is not None and len(raw_att_time) == 1 # Only one post-treatment ATT
    except Exception as e:
        pytest.fail(f"SRC fit failed with few post-treatment periods: {e}")

@pytest.fixture
def no_pre_periods_data_src():
    """Panel data where treatment starts at the first period."""
    data_dict = { # Unit 1 treated at time_id = 1
        'unit_id': [1, 1, 1, 2, 2, 2, 3, 3, 3], 
        'time_id': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'Y':       [16,17,18, 9, 10,11, 12,13,14],
        'X1':      [8, 9,10,  4, 5, 6,  6, 7, 8],
    }
    df = pd.DataFrame(data_dict)
    treatment_col_name = SRC_FULL_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    df.loc[(df['unit_id'] == 1) & (df['time_id'] >= 1), treatment_col_name] = 1
    return df

def test_src_no_pre_periods(no_pre_periods_data_src: pd.DataFrame):
    """Test SRC fails or handles no pre-treatment periods."""
    pydantic_dict = _get_pydantic_config_dict_src(SRC_FULL_TEST_CONFIG_BASE, no_pre_periods_data_src)
    config_obj = SRCConfig(**pydantic_dict)
    estimator = SRC(config=config_obj)
    # dataprep raises MlsynthDataError for zero pre-periods.
    with pytest.raises(MlsynthDataError, match="Not enough pre-treatment periods \\(0 pre-periods found\\)."):
        estimator.fit()

@pytest.fixture
def no_post_periods_data_src():
    """Panel data where treatment starts after the last observed period (effectively no post-period)."""
    data_dict = { # Unit 1 treated at time_id = 4, but data only up to 3
        'unit_id': [1, 1, 1, 2, 2, 2, 3, 3, 3], 
        'time_id': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'Y':       [10,11,12, 9, 10,11, 12,13,14],
        'X1':      [5, 6, 7,  4, 5, 6,  6, 7, 8],
    }
    df = pd.DataFrame(data_dict)
    treatment_col_name = SRC_FULL_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    # Treatment starts at time 4, but max time is 3
    df.loc[(df['unit_id'] == 1) & (df['time_id'] >= 4), treatment_col_name] = 1 
    return df

def test_src_no_post_periods(no_post_periods_data_src: pd.DataFrame):
    """Test SRC fails or handles no post-treatment periods."""
    pydantic_dict = _get_pydantic_config_dict_src(SRC_FULL_TEST_CONFIG_BASE, no_post_periods_data_src)
    config_obj = SRCConfig(**pydantic_dict)
    estimator = SRC(config=config_obj)
    # dataprep -> logictreat now raises MlsynthDataError
    with pytest.raises(MlsynthDataError, match="No treated units found"):
        estimator.fit()

# --- Tests for detailed results validation ---

def test_src_fit_results_structure_and_types(basic_panel_data_with_treatment_src: pd.DataFrame):
    """Test the structure and types of the results dictionary from fit()."""
    pydantic_dict = _get_pydantic_config_dict_src(SRC_FULL_TEST_CONFIG_BASE, basic_panel_data_with_treatment_src)
    config_obj = SRCConfig(**pydantic_dict)
    estimator = SRC(config=config_obj)
    results = estimator.fit()
    assert isinstance(results, BaseEstimatorResults)

    # Time Series - Counterfactual
    assert results.time_series is not None
    assert isinstance(results.time_series.counterfactual_outcome, np.ndarray)
    
    # Weights
    assert results.weights is not None
    assert results.weights.donor_names is not None
    assert results.weights.donor_weights is not None
    assert all(isinstance(k, str) for k in results.weights.donor_names)
    assert isinstance(results.weights.donor_weights, dict) # Changed from np.ndarray
    assert len(results.weights.donor_names) == len(results.weights.donor_weights.keys()) # Compare length of names list to keys in weights dict
    # For SRC, weights don't necessarily sum to 1 or are non-negative.

    # Effects
    effects_res = results.effects
    assert effects_res is not None
    raw_att_dict = results.raw_results.get("ATT", {}) # Check raw for SATT, TTE, etc.
    expected_att_keys = ["ATT", "Percent ATT", "SATT", "TTE", "ATT_Time", "PercentATT_Time", "SATT_Time"]
    for key in expected_att_keys:
        assert key in raw_att_dict, f"Expected key '{key}' not found in ATT results (raw)."
    assert isinstance(effects_res.att, (float, np.floating, type(None)))
    assert isinstance(effects_res.att_percent, (float, np.floating, type(None)))
    assert isinstance(raw_att_dict["ATT_Time"], np.ndarray)


    # Fit Diagnostics
    fit_res = results.fit_diagnostics
    assert fit_res is not None
    expected_fit_keys = ["T0 RMSE", "T1 RMSE", "R-Squared"] 
    raw_fit_dict = results.raw_results.get("Fit", {})
    for key in expected_fit_keys:
        assert key in raw_fit_dict, f"Expected key '{key}' not found in Fit results (raw)."
    assert isinstance(fit_res.rmse_pre, (float, np.floating, type(None)))
    assert isinstance(fit_res.rmse_post, (float, np.floating, type(None)))
    assert isinstance(fit_res.r_squared_pre, (float, np.floating, type(None)))

    # Time Series - Vectors
    ts_res = results.time_series
    assert isinstance(ts_res.observed_outcome, np.ndarray)
    assert isinstance(ts_res.counterfactual_outcome, np.ndarray) # Already checked
    assert isinstance(ts_res.gap, np.ndarray)
    
    # Check consistency of vector lengths
    num_time_periods = basic_panel_data_with_treatment_src[
        basic_panel_data_with_treatment_src['unit_id'] == 1 
    ]['time_id'].nunique()
    
    assert len(ts_res.counterfactual_outcome) == num_time_periods
    assert len(ts_res.observed_outcome) == num_time_periods
    assert len(ts_res.gap) == num_time_periods
    
    num_post_periods = basic_panel_data_with_treatment_src[
        (basic_panel_data_with_treatment_src['unit_id'] == 1) & 
        (basic_panel_data_with_treatment_src[SRC_FULL_TEST_CONFIG_BASE["treat"]] == 1)
    ]['time_id'].nunique()
    assert len(raw_att_dict["ATT_Time"]) == num_post_periods


def test_src_weights_correspond_to_donors(basic_panel_data_with_treatment_src: pd.DataFrame):
    """Test that donor weights correspond to actual donor unit IDs."""
    pydantic_dict = _get_pydantic_config_dict_src(SRC_FULL_TEST_CONFIG_BASE, basic_panel_data_with_treatment_src)
    config_obj = SRCConfig(**pydantic_dict)
    estimator = SRC(config=config_obj)
    results = estimator.fit()

    # Identify treated unit name from the fixture's setup
    # Unit 1 is treated in basic_panel_data_with_treatment_src
    treated_unit_id_value = 1 
    
    all_unit_ids_in_data = set(
        basic_panel_data_with_treatment_src['unit_id'].unique()
    )
    
    # Donor IDs are all unit IDs except the treated one. Convert to string for comparison.
    donor_ids_from_data_str = {str(uid) for uid in all_unit_ids_in_data if uid != treated_unit_id_value}
    
    assert results.weights is not None
    assert results.weights.donor_names is not None
    weight_names_set = set(results.weights.donor_names)

    assert weight_names_set == donor_ids_from_data_str, \
        f"Weight names {weight_names_set} do not match expected donor IDs {donor_ids_from_data_str}"


def test_src_missing_unitid_in_config(basic_panel_data_with_treatment_src: pd.DataFrame):
    """Test SRCConfig instantiation fails if 'unitid' is missing in config."""
    config_dict_missing_unitid = SRC_FULL_TEST_CONFIG_BASE.copy()
    del config_dict_missing_unitid["unitid"]
    pydantic_dict = _get_pydantic_config_dict_src(config_dict_missing_unitid, basic_panel_data_with_treatment_src)
    if "unitid" in pydantic_dict: del pydantic_dict["unitid"]
    
    with pytest.raises(ValidationError): # unitid is required
        SRCConfig(**pydantic_dict)

def test_src_unitid_not_in_df(basic_panel_data_with_treatment_src: pd.DataFrame):
    """Test SRC fit fails if 'unitid' column is not in df.
    Pydantic config creation should pass, error expected during fit.
    """
    full_config_dict = SRC_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict["unitid"] = "NonExistentUnitIdColumn"
    
    pydantic_dict = _get_pydantic_config_dict_src(full_config_dict, basic_panel_data_with_treatment_src)
    # BaseEstimatorConfig validates column existence
    with pytest.raises(MlsynthDataError, match=f"Missing required columns in DataFrame 'df': {full_config_dict['unitid']}"):
        SRCConfig(**pydantic_dict)

def test_src_missing_time_in_config(basic_panel_data_with_treatment_src: pd.DataFrame):
    """Test SRCConfig instantiation fails if 'time' is missing in config."""
    config_dict_missing_time = SRC_FULL_TEST_CONFIG_BASE.copy()
    del config_dict_missing_time["time"]
    pydantic_dict = _get_pydantic_config_dict_src(config_dict_missing_time, basic_panel_data_with_treatment_src)
    if "time" in pydantic_dict: del pydantic_dict["time"]
    
    with pytest.raises(ValidationError): # time is required
        SRCConfig(**pydantic_dict)

def test_src_time_not_in_df(basic_panel_data_with_treatment_src: pd.DataFrame):
    """Test SRC fit fails if 'time' column is not in df.
    Pydantic config creation should pass, error expected during fit.
    """
    full_config_dict = SRC_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict["time"] = "NonExistentTimeColumn"
    
    pydantic_dict = _get_pydantic_config_dict_src(full_config_dict, basic_panel_data_with_treatment_src)
    # BaseEstimatorConfig validates column existence
    with pytest.raises(MlsynthDataError, match=f"Missing required columns in DataFrame 'df': {full_config_dict['time']}"):
        SRCConfig(**pydantic_dict)
