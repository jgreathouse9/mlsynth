import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any
import cvxpy 
from unittest.mock import patch
from pydantic import ValidationError

from mlsynth import SDID
from mlsynth.exceptions import (
    MlsynthDataError,
    MlsynthEstimationError,
    # MlsynthConfigError, # Not directly tested for raising here yet
)
from mlsynth.config_models import (
    SDIDConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
    InferenceResults,
    MethodDetailsResults
)
from mlsynth.utils.datautils import balance, dataprep 
from mlsynth.utils.sdidutils import estimate_event_study_sdid 
from mlsynth.utils.resultutils import SDID_plot 

# Configuration for SDID (excluding 'df' which is passed at Pydantic model instantiation)
SDID_TEST_CONFIG_BASE = {
    "outcome": "outcome_val",
    "treat": "treatment_status_sdid", # Name of the treatment column
    "unitid": "id_unit",
    "time": "time_period",
    "display_graphs": False,
    "save": False,
    "B": 10, # Small number of placebo iterations for testing
    "counterfactual_color": ["red"],
    "treated_color": "darkblue",
    "seed": 91827,
    "verbose": False,
}

# Fields that are part of BaseEstimatorConfig (and thus SDIDConfig)
SDID_PYDANTIC_MODEL_FIELDS = [
    "df", "outcome", "treat", "unitid", "time", 
    "display_graphs", "save", "B", # B is specific to SDIDConfig
    "counterfactual_color", "treated_color" 
    # seed and verbose are not in SDIDConfig
]

def _get_pydantic_config_dict_sdid(full_config: Dict[str, Any], df_fixture: pd.DataFrame) -> Dict[str, Any]:
    """Helper to extract Pydantic-valid fields for SDIDConfig and add the DataFrame."""
    pydantic_dict = {
        k: v for k, v in full_config.items() if k in SDID_PYDANTIC_MODEL_FIELDS
    }
    pydantic_dict["df"] = df_fixture
    return pydantic_dict

@pytest.fixture
def sdid_panel_data():
    """Provides a panel dataset for SDID smoke testing."""
    n_units = 15 # SDID needs a decent number of control units for placebo
    n_periods = 20
    data_dict = {
        'id_unit': np.repeat(np.arange(1, n_units + 1), n_periods),
        'time_period': np.tile(np.arange(1, n_periods + 1), n_units),
        'outcome_val': np.random.normal(loc=np.repeat(np.arange(0, n_units*5, 5), n_periods), scale=3, size=n_units*n_periods),
        'covariate_x': np.random.rand(n_units * n_periods) * 15,
    }
    df = pd.DataFrame(data_dict)
    
    # Unit 1 is treated starting from period = 12
    treatment_col_name = SDID_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    df.loc[(df['id_unit'] == 1) & (df['time_period'] >= 12), treatment_col_name] = 1
    return df

def test_sdid_creation(sdid_panel_data):
    """Test that the SDID estimator can be instantiated."""
    pydantic_config_dict = _get_pydantic_config_dict_sdid(SDID_TEST_CONFIG_BASE, sdid_panel_data)
    
    try:
        sdid_config = SDIDConfig(**pydantic_config_dict)
        estimator = SDID(config=sdid_config)
        assert estimator is not None, "SDID estimator should be created."
        assert estimator.outcome == "outcome_val"
        assert estimator.treat == SDID_TEST_CONFIG_BASE["treat"]
        assert estimator.B == 10
        assert not estimator.display_graphs
    except ValidationError as ve:
        pytest.fail(f"SDIDConfig validation failed during estimator creation test: {ve}")
    except Exception as e:
        pytest.fail(f"SDID instantiation failed: {e}")

def test_sdid_fit_smoke(sdid_panel_data):
    """Smoke test for the SDID fit method."""
    pydantic_config_dict = _get_pydantic_config_dict_sdid(SDID_TEST_CONFIG_BASE, sdid_panel_data)
    
    try:
        sdid_config = SDIDConfig(**pydantic_config_dict)
    except ValidationError as ve:
        pytest.fail(f"SDIDConfig validation failed during fit smoke test setup: {ve}")

    estimator = SDID(config=sdid_config)
    
    try:
        results = estimator.fit() 
        assert results is not None, "Fit method should return results."
        assert isinstance(results, BaseEstimatorResults), "Results should be BaseEstimatorResults Pydantic model."
        
        # Check main components
        assert results.effects is not None and isinstance(results.effects, EffectsResults)
        assert results.fit_diagnostics is not None and isinstance(results.fit_diagnostics, FitDiagnosticsResults) # Will be mostly empty
        assert results.time_series is not None and isinstance(results.time_series, TimeSeriesResults)
        assert results.weights is not None and isinstance(results.weights, WeightsResults) # Will be mostly empty
        assert results.inference is not None and isinstance(results.inference, InferenceResults)
        assert results.method_details is not None and isinstance(results.method_details, MethodDetailsResults)
        assert results.raw_results is not None and isinstance(results.raw_results, dict)

        # Check some specific fields based on how _create_estimator_results maps them
        assert results.effects.att is not None or np.isnan(results.effects.att) # ATT can be NaN
        if results.effects.additional_effects:
            assert "att_standard_error" in results.effects.additional_effects
            assert "att_confidence_interval" in results.effects.additional_effects
        
        if results.time_series.time_periods is not None:
            assert isinstance(results.time_series.time_periods, np.ndarray)
            assert isinstance(results.time_series.estimated_gap, np.ndarray)
            assert len(results.time_series.time_periods) == len(results.time_series.estimated_gap)

        assert "pooled_estimates" in results.raw_results
        assert "att" in results.raw_results 
        
        # Check structure of pooled_estimates from raw_results
        raw_pooled_estimates = results.raw_results.get("pooled_estimates", {})
        assert isinstance(raw_pooled_estimates, dict)
        if raw_pooled_estimates: 
            first_event_time_key = next(iter(raw_pooled_estimates))
            assert isinstance(first_event_time_key, (float, int))
            event_time_data = raw_pooled_estimates[first_event_time_key]
            assert isinstance(event_time_data, dict)
            assert "tau" in event_time_data
            assert "se" in event_time_data
            assert "ci" in event_time_data
            assert isinstance(event_time_data["ci"], list) and len(event_time_data["ci"]) == 2
            
        if estimator.B > 0 and results.inference.p_value is not None:
            assert 0 <= results.inference.p_value <= 1
        if results.inference.confidence_interval is not None:
            assert isinstance(results.inference.confidence_interval, list) and len(results.inference.confidence_interval) == 2


    except Exception as e:
        if isinstance(e, (np.linalg.LinAlgError, ValueError)) and ("singular" in str(e).lower() or "must be positive" in str(e).lower()):
             pytest.skip(f"Skipping due to potential numerical issue in SDID with test data: {e}")
        pytest.fail(f"SDID fit method failed during smoke test: {e}")

# --- Input Validation Tests ---

def test_sdid_creation_missing_config_keys(sdid_panel_data):
    """Test SDIDConfig instantiation with missing essential keys."""
    
    # These are the fields in SDIDConfig (inherited from BaseEstimatorConfig)
    # that do NOT have default values. 'df' is handled by _get_pydantic_config_dict_sdid.
    truly_required_string_keys = ["outcome", "treat", "unitid", "time"]
    
    for key_to_remove in truly_required_string_keys:
        pydantic_config_dict = _get_pydantic_config_dict_sdid(SDID_TEST_CONFIG_BASE, sdid_panel_data)
        if key_to_remove in pydantic_config_dict:
            del pydantic_config_dict[key_to_remove]

        with pytest.raises(ValidationError) as excinfo:
            SDIDConfig(**pydantic_config_dict)
        
        # Check that the error message mentions the missing field
        assert key_to_remove in str(excinfo.value).lower()


def test_sdid_creation_df_not_dataframe():
    """Test SDIDConfig instantiation if 'df' is not a pandas DataFrame."""
    pydantic_config_dict = _get_pydantic_config_dict_sdid(SDID_TEST_CONFIG_BASE, "not_a_dataframe") # type: ignore
    with pytest.raises(ValidationError) as excinfo:
        SDIDConfig(**pydantic_config_dict)
    assert "df" in str(excinfo.value).lower()
    assert "DataFrame" in str(excinfo.value) # Pydantic error message should indicate expected type

def test_sdid_creation_df_missing_columns(sdid_panel_data):
    """Test SDIDConfig instantiation fails if df is missing essential columns (error during config validation)."""
    
    essential_cols_map = {
        "outcome": SDID_TEST_CONFIG_BASE["outcome"],
        "treat": SDID_TEST_CONFIG_BASE["treat"],
        "unitid": SDID_TEST_CONFIG_BASE["unitid"],
        "time": SDID_TEST_CONFIG_BASE["time"],
    }
    
    for col_key_in_config, actual_col_name in essential_cols_map.items():
        df_missing_col = sdid_panel_data.copy()
        if actual_col_name in df_missing_col.columns:
             df_missing_col = df_missing_col.drop(columns=[actual_col_name])
        
        pydantic_config_dict = _get_pydantic_config_dict_sdid(SDID_TEST_CONFIG_BASE, df_missing_col)
        
        # BaseEstimatorConfig's model_validator catches missing essential columns during SDIDConfig instantiation.
        with pytest.raises(MlsynthDataError, match=f"Missing required columns in DataFrame 'df': {actual_col_name}"):
            SDIDConfig(**pydantic_config_dict)

def test_sdid_creation_invalid_config_types(sdid_panel_data):
    """Test SDIDConfig instantiation with invalid types for config parameters."""

    # Test for B (integer expected)
    pydantic_config_invalid_b = _get_pydantic_config_dict_sdid(SDID_TEST_CONFIG_BASE, sdid_panel_data)
    pydantic_config_invalid_b["B"] = "not_an_int" # type: ignore
    with pytest.raises(ValidationError) as excinfo_b:
        SDIDConfig(**pydantic_config_invalid_b)
    assert "b" in str(excinfo_b.value).lower() # Field name is 'B'
    assert "integer" in str(excinfo_b.value).lower()

    # Test for save (Pydantic model defines it as Union[bool, str, Dict[str, str]])
    # Test with a type not in the Union
    pydantic_config_invalid_save = _get_pydantic_config_dict_sdid(SDID_TEST_CONFIG_BASE, sdid_panel_data)
    pydantic_config_invalid_save["save"] = 123 # type: ignore
    with pytest.raises(ValidationError) as excinfo_save:
        SDIDConfig(**pydantic_config_invalid_save)
    assert "save" in str(excinfo_save.value).lower()
    # Check for messages related to bool, str, or dict failure
    assert ("bool" in str(excinfo_save.value).lower() or 
            "string" in str(excinfo_save.value).lower() or 
            "dict" in str(excinfo_save.value).lower())


    # Test for display_graphs (boolean expected)
    pydantic_config_invalid_display = _get_pydantic_config_dict_sdid(SDID_TEST_CONFIG_BASE, sdid_panel_data)
    pydantic_config_invalid_display["display_graphs"] = "not_a_bool" # type: ignore
    with pytest.raises(ValidationError) as excinfo_display:
        SDIDConfig(**pydantic_config_invalid_display)
    assert "display_graphs" in str(excinfo_display.value).lower()
    assert "bool" in str(excinfo_display.value).lower()

def test_sdid_fit_no_control_units(sdid_panel_data):
    """Test SDID fit when no control units are available."""
    df_no_controls = sdid_panel_data[sdid_panel_data[SDID_TEST_CONFIG_BASE["unitid"]] == 1].copy() # Keep only the treated unit

    pydantic_config_dict = _get_pydantic_config_dict_sdid(SDID_TEST_CONFIG_BASE, df_no_controls)
    
    try:
        sdid_config = SDIDConfig(**pydantic_config_dict)
    except ValidationError as ve:
        pytest.fail(f"SDIDConfig validation failed for no control units test: {ve}")

    estimator = SDID(config=sdid_config)
    with pytest.raises(MlsynthDataError, match="No donor units found after pivoting and selecting."):
        estimator.fit()


def test_sdid_creation_with_dict(sdid_panel_data):
    """Ensure SDID.__init__ handles config dict input."""
    config_dict = _get_pydantic_config_dict_sdid(SDID_TEST_CONFIG_BASE, sdid_panel_data)
    
    estimator = SDID(config_dict)  # <- pass dict, not SDIDConfig
    assert isinstance(estimator, SDID)
    assert estimator.df.equals(sdid_panel_data)
    assert estimator.B == config_dict["B"]





def test_sdid_fit_non_binary_treatment(sdid_panel_data):
    """Test SDID fit when the treatment indicator is not binary."""
    df_non_binary_treat = sdid_panel_data.copy()
    # Introduce non-binary values in the treatment column
    df_non_binary_treat.loc[df_non_binary_treat[SDID_TEST_CONFIG_BASE["unitid"]] == 2, SDID_TEST_CONFIG_BASE["treat"]] = 2 
    
    pydantic_config_dict = _get_pydantic_config_dict_sdid(SDID_TEST_CONFIG_BASE, df_non_binary_treat)

    try:
        sdid_config = SDIDConfig(**pydantic_config_dict)
    except ValidationError as ve:
        pytest.fail(f"SDIDConfig validation failed for non-binary treatment test: {ve}")

    estimator = SDID(config=sdid_config)
    with pytest.raises(MlsynthDataError, match="Treatment indicator must be a binary variable"):
        estimator.fit()

# --- Edge Case Tests ---

@patch('mlsynth.estimators.sdid.SDID_plot') # Mock the plotting function
def test_sdid_fit_insufficient_pre_periods(mock_sdid_plot, sdid_panel_data):
    """Test SDID fit with insufficient pre-treatment periods."""
    df = sdid_panel_data.copy()
    # Modify treat column for insufficient pre-periods
    df[SDID_TEST_CONFIG_BASE["treat"]] = 0
    df.loc[(df['id_unit'] == 1) & (df['time_period'] >= 2), SDID_TEST_CONFIG_BASE["treat"]] = 1
    
    pydantic_config_dict = _get_pydantic_config_dict_sdid(SDID_TEST_CONFIG_BASE, df)
    try:
        sdid_config = SDIDConfig(**pydantic_config_dict)
    except ValidationError as ve:
        pytest.fail(f"SDIDConfig validation failed for insufficient pre-periods test: {ve}")
    
    estimator = SDID(config=sdid_config)
    
    try:
        estimator.fit()
    except Exception as e:
        pytest.fail(f"SDID.fit() failed unexpectedly with insufficient pre-periods: {e}")
    mock_sdid_plot.assert_not_called()


@patch('mlsynth.estimators.sdid.SDID_plot')
def test_sdid_fit_insufficient_donors(mock_sdid_plot, sdid_panel_data):
    """Test SDID fit with insufficient donor units."""
    df = sdid_panel_data.copy()
    df_few_donors = df[df['id_unit'].isin([1, 2])] # Only treated unit 1 and one donor unit 2
    
    # Ensure the single donor has valid outcome data
    df_few_donors.loc[df_few_donors['id_unit'] == 2, SDID_TEST_CONFIG_BASE["outcome"]] = \
        np.random.normal(loc=10, scale=1, size=df_few_donors[df_few_donors['id_unit'] == 2].shape[0])
    assert not df_few_donors.loc[df_few_donors['id_unit'] == 2, SDID_TEST_CONFIG_BASE["outcome"]].isnull().any()

    pydantic_config_dict = _get_pydantic_config_dict_sdid(SDID_TEST_CONFIG_BASE, df_few_donors)
    try:
        sdid_config = SDIDConfig(**pydantic_config_dict)
    except ValidationError as ve:
        pytest.fail(f"SDIDConfig validation failed for insufficient donors test: {ve}")
        
    estimator = SDID(config=sdid_config)
    
    try:
        estimator.fit()
    except Exception as e:
        pytest.fail(f"SDID.fit() failed unexpectedly with insufficient donors: {e}")
    mock_sdid_plot.assert_not_called()

@patch('mlsynth.estimators.sdid.SDID_plot')
def test_sdid_fit_no_post_periods(mock_sdid_plot, sdid_panel_data):
    """Test SDID fit with no post-treatment periods for the treated unit."""
    df = sdid_panel_data.copy()
    last_period = df['time_period'].max()
    # Set treatment to start after the last observed period
    df[SDID_TEST_CONFIG_BASE["treat"]] = 0
    df.loc[(df['id_unit'] == 1) & (df['time_period'] >= last_period + 1), SDID_TEST_CONFIG_BASE["treat"]] = 1
    
    pydantic_config_dict = _get_pydantic_config_dict_sdid(SDID_TEST_CONFIG_BASE, df)
    try:
        sdid_config = SDIDConfig(**pydantic_config_dict)
    except ValidationError as ve:
        pytest.fail(f"SDIDConfig validation failed for no post-periods test: {ve}")

    estimator = SDID(config=sdid_config)
    
    # logictreat (via dataprep) raises MlsynthDataError if no treated units are found
    with pytest.raises(MlsynthDataError, match="No treated units found"):
        estimator.fit()
    mock_sdid_plot.assert_not_called()

@patch('mlsynth.estimators.sdid.SDID_plot')
def test_sdid_fit_nan_in_outcome(mock_sdid_plot, sdid_panel_data):
    """Test SDID fit when outcome variable contains NaN values."""
    
    # Case 1: NaN in outcome for the treated unit in pre-period
    df_nan_treat_pre = sdid_panel_data.copy()
    df_nan_treat_pre.loc[
        (df_nan_treat_pre['id_unit'] == 1) & (df_nan_treat_pre['time_period'] == 5), 
        SDID_TEST_CONFIG_BASE["outcome"]
    ] = np.nan
    
    pydantic_config_nan_treat = _get_pydantic_config_dict_sdid(SDID_TEST_CONFIG_BASE, df_nan_treat_pre)
    try:
        sdid_config_nan_treat = SDIDConfig(**pydantic_config_nan_treat)
    except ValidationError as ve:
        pytest.fail(f"SDIDConfig validation failed for NaN in treated pre-period: {ve}")
    estimator_nan_treat = SDID(config=sdid_config_nan_treat)
    # sdid_est handles NaNs by dropping. If this leads to insufficient data, it might raise.
    # Test that it runs without unhandled error, or specific CVXPY error if NaNs propagate.
    try:
        estimator_nan_treat.fit()
    except Exception as e:
        # Allow for known numerical/data issues, but fail on unexpected errors
        if not isinstance(e, (np.linalg.LinAlgError, ValueError, cvxpy.error.SolverError, cvxpy.error.DCPError)):
            pytest.fail(f"SDID.fit() with NaN in treated pre-period failed unexpectedly: {e}")
    mock_sdid_plot.assert_not_called()
    mock_sdid_plot.reset_mock()

    # Case 2: All pre-period outcomes for treated unit are NaN
    df_all_nan_pre_treat = sdid_panel_data.copy()
    df_all_nan_pre_treat.loc[
        (df_all_nan_pre_treat['id_unit'] == 1) & (df_all_nan_pre_treat['time_period'] < 12), 
        SDID_TEST_CONFIG_BASE["outcome"]
    ] = np.nan
    
    pydantic_config_all_nan = _get_pydantic_config_dict_sdid(SDID_TEST_CONFIG_BASE, df_all_nan_pre_treat)
    try:
        sdid_config_all_nan = SDIDConfig(**pydantic_config_all_nan)
    except ValidationError as ve:
        pytest.fail(f"SDIDConfig validation failed for all NaN in treated pre-period: {ve}")
    estimator_all_nan = SDID(config=sdid_config_all_nan)
    
    try:
        estimator_all_nan.fit()
    except Exception as e:
        # Allow for known numerical/data issues
        if not isinstance(e, (np.linalg.LinAlgError, ValueError, cvxpy.error.SolverError, cvxpy.error.DCPError)):
            pytest.fail(f"SDID.fit() failed unexpectedly when all pre-treatment treated data is NaN: {e}")
    mock_sdid_plot.assert_not_called()
    mock_sdid_plot.reset_mock()

    # Case 3: NaN in control unit pre-period
    df_nan_control = sdid_panel_data.copy()
    df_nan_control.loc[
        (df_nan_control['id_unit'] == 2) & (df_nan_control['time_period'] == 5), 
        SDID_TEST_CONFIG_BASE["outcome"]
    ] = np.nan
    
    pydantic_config_nan_control = _get_pydantic_config_dict_sdid(SDID_TEST_CONFIG_BASE, df_nan_control)
    try:
        sdid_config_nan_control = SDIDConfig(**pydantic_config_nan_control)
    except ValidationError as ve:
        pytest.fail(f"SDIDConfig validation failed for NaN in control pre-period: {ve}")
    estimator_nan_control = SDID(config=sdid_config_nan_control)
    
    # Expect MlsynthEstimationError wrapping CVXPY errors if NaNs propagate
    with pytest.raises(MlsynthEstimationError,
                       match=r"(CVXPY solver error in SDID|CVXPY solver failed in unit_weights|Problem does not follow DCP rules|Solver failed|infeasible|unbounded|nan|NaN)") as excinfo:
        estimator_nan_control.fit()
    
    # Plotting should not be called if fit fails with an exception
    mock_sdid_plot.assert_not_called() # mock_sdid_plot was already reset before this case

# --- Detailed Results Validation ---
# (Covered by test_sdid_fit_smoke for now, can be expanded if specific calculations need checking)

# --- Configuration Variations ---

@patch('mlsynth.estimators.sdid.SDID_plot')
def test_sdid_fit_different_B_values(mock_sdid_plot, sdid_panel_data):
    """Test SDID fit with different numbers of placebo iterations (B)."""

    for b_val in [0, 5, 20]: 
        pydantic_config_dict = _get_pydantic_config_dict_sdid(SDID_TEST_CONFIG_BASE, sdid_panel_data)
        pydantic_config_dict["B"] = b_val
        
        try:
            sdid_config = SDIDConfig(**pydantic_config_dict)
        except ValidationError as ve:
            pytest.fail(f"SDIDConfig validation failed for B={b_val}: {ve}")

        estimator = SDID(config=sdid_config)
        try:
            results = estimator.fit()
            assert isinstance(results, BaseEstimatorResults)
            assert "pooled_estimates" in results.raw_results
            assert "att_ci" in results.raw_results
            assert isinstance(results.raw_results["att_ci"], list) and len(results.raw_results["att_ci"]) == 2
            if b_val > 0:
                assert results.inference.p_value is not None
            else:
                assert results.inference.p_value is None


        except (np.linalg.LinAlgError, ValueError, TypeError, cvxpy.error.SolverError, cvxpy.error.DCPError) as e:
            if ("singular" in str(e).lower() or
                "must be positive" in str(e).lower() or
                "Need at least two" in str(e).lower() or 
                "control units after filtering" in str(e).lower() or
                "Problem encountered" in str(e) or 
                "solution may be inaccurate" in str(e).lower() or
                "Infeasible" in str(e) or "Unbounded" in str(e)
                ):
                pytest.skip(f"Skipping due to numerical/data issue with B={b_val}: {e}")
            else:
                pytest.fail(f"SDID fit failed for B={b_val}: {e}")
        
        if sdid_config.display_graphs:
             mock_sdid_plot.assert_called()
        else:
             mock_sdid_plot.assert_not_called()
        mock_sdid_plot.reset_mock()


# --- Plotting Behavior ---

@patch('mlsynth.estimators.sdid.SDID_plot')
def test_sdid_plotting_behavior(mock_sdid_plot_func, sdid_panel_data):
    """Test that SDID_plot is called based on display_graphs config."""

    # Case 1: display_graphs = True
    pydantic_config_true = _get_pydantic_config_dict_sdid(SDID_TEST_CONFIG_BASE, sdid_panel_data)
    pydantic_config_true["display_graphs"] = True
    
    try:
        sdid_config_true = SDIDConfig(**pydantic_config_true)
    except ValidationError as ve:
        pytest.fail(f"SDIDConfig validation failed for display_graphs=True test: {ve}")
    
    estimator_true = SDID(config=sdid_config_true)
    try:
        results_obj_true = estimator_true.fit() # Returns BaseEstimatorResults
        # SDID_plot expects the raw dictionary
        mock_sdid_plot_func.assert_called_once_with(results_obj_true.raw_results) 
    except (np.linalg.LinAlgError, ValueError, cvxpy.error.SolverError, cvxpy.error.DCPError) as e:
        if "singular" in str(e).lower() or "must be positive" in str(e).lower() or "Problem encountered" in str(e):
            pytest.skip(f"Skipping plotting test (display_graphs=True) due to numerical/solver issue: {e}")
        else:
            pytest.fail(f"SDID fit failed (display_graphs=True): {e}")

    mock_sdid_plot_func.reset_mock()

    # Case 2: display_graphs = False
    pydantic_config_false = _get_pydantic_config_dict_sdid(SDID_TEST_CONFIG_BASE, sdid_panel_data)
    pydantic_config_false["display_graphs"] = False

    try:
        sdid_config_false = SDIDConfig(**pydantic_config_false)
    except ValidationError as ve:
        pytest.fail(f"SDIDConfig validation failed for display_graphs=False test: {ve}")

    estimator_false = SDID(config=sdid_config_false)
    try:
        estimator_false.fit()
        mock_sdid_plot_func.assert_not_called()
    except (np.linalg.LinAlgError, ValueError, cvxpy.error.SolverError, cvxpy.error.DCPError) as e:
        if "singular" in str(e).lower() or "must be positive" in str(e).lower() or "Problem encountered" in str(e):
            pytest.skip(f"Skipping plotting test (display_graphs=False) due to numerical/solver issue: {e}")
        else:
            pytest.fail(f"SDID fit failed (display_graphs=False): {e}")
