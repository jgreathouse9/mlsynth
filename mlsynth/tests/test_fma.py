import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any
from unittest.mock import patch
from pydantic import ValidationError

from mlsynth.estimators.fma import FMA
from mlsynth.config_models import (
    FMAConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    InferenceResults
)
from mlsynth.exceptions import MlsynthDataError, MlsynthEstimationError, MlsynthConfigError
from mlsynth.utils.datautils import balance, dataprep # For potential error sources
from mlsynth.utils.resultutils import plot_estimates # For mocking

@pytest.fixture
def sample_fma_data() -> pd.DataFrame:
    """Creates a sample DataFrame for FMA tests with more time periods."""
    n_units = 16
    n_periods = 40
    treatment_start_period = 20

    units = np.repeat(np.arange(1, n_units + 1), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_units)
    
    # Create some varied outcome data
    np.random.seed(42) # for reproducibility
    outcomes = []
    for i in range(n_units):
        base_trend = np.linspace(start=10 + i*2, stop=25 + i*2, num=n_periods)
        noise = np.random.normal(0, 1, n_periods)
        outcomes.extend(base_trend + noise)

    data = {
        "Unit": units,
        "Time": times,
        "Outcome": outcomes,
        "Treated": np.zeros(n_units * n_periods, dtype=int),
    }
    df = pd.DataFrame(data)

    # Unit 1 treated from treatment_start_period
    # Pre-periods: 1 to treatment_start_period-1. Post-periods: treatment_start_period to n_periods.
    # If treatment_start_period = 13, pre_periods = 12.
    df.loc[(df['Unit'] == 1) & (df['Time'] >= treatment_start_period), 'Treated'] = 1
    
    return df

def test_fma_creation(sample_fma_data: pd.DataFrame) -> None:
    """Test FMA estimator creation."""
    config_dict: Dict[str, Any] = {
        "df": sample_fma_data,
        "treat": "Treated",
        "time": "Time",
        "outcome": "Outcome",
        "unitid": "Unit",
        "display_graphs": False,
    }
    try:
        config_obj = FMAConfig(**config_dict)
        estimator = FMA(config_obj)
        assert estimator is not None
        assert estimator.df.equals(sample_fma_data)
        assert estimator.treat == "Treated"
        assert estimator.time == "Time"
        assert estimator.outcome == "Outcome"
        assert estimator.unitid == "Unit"
        assert not estimator.display_graphs
    except Exception as e:
        pytest.fail(f"FMA creation failed: {e}")

def test_fma_fit_smoke(sample_fma_data: pd.DataFrame) -> None:
    """Smoke test for FMA fit method."""
    config_dict: Dict[str, Any] = {
        "df": sample_fma_data,
        "treat": "Treated",
        "time": "Time",
        "outcome": "Outcome",
        "unitid": "Unit",
        "display_graphs": False,
        "criti": 11, # Default, nonstationary
        "DEMEAN": 1, # Default, demean
    }
    config_obj = FMAConfig(**config_dict)
    estimator = FMA(config_obj)
    try:
        results = estimator.fit()
        assert isinstance(results, BaseEstimatorResults)
        assert results.effects is not None
        assert results.fit_diagnostics is not None
        assert results.time_series is not None
        assert results.inference is not None

        # Check sub-model types
        assert isinstance(results.effects, EffectsResults)
        assert isinstance(results.fit_diagnostics, FitDiagnosticsResults)
        assert isinstance(results.time_series, TimeSeriesResults)
        assert isinstance(results.inference, InferenceResults)

    except Exception as e:
        pytest.fail(f"FMA fit failed: {e}")

# --- Input Validation Tests ---

def test_fma_creation_missing_config_keys(sample_fma_data: pd.DataFrame):
    """Test FMA instantiation with missing essential keys in config."""
    base_config_dict: Dict[str, Any] = {
        "df": sample_fma_data,
        "treat": "Treated",
        "time": "Time",
        "outcome": "Outcome",
        "unitid": "Unit",
    }
    # Pydantic models define required fields in their schema.
    # We test by removing keys and expecting ValidationError.
    required_keys = ["df", "outcome", "treat", "unitid", "time"] 
    
    for key_to_remove in required_keys:
        test_config_dict = base_config_dict.copy()
        test_config_dict.pop(key_to_remove, None)
        
        with pytest.raises(ValidationError):
            FMAConfig(**test_config_dict)

def test_fma_creation_df_not_dataframe():
    """Test FMA instantiation if 'df' in config is not a pandas DataFrame."""
    test_config_dict: Dict[str, Any] = {
        "df": "not_a_dataframe", # Invalid type
        "treat": "Treated", "time": "Time", "outcome": "Outcome", "unitid": "Unit",
    }
    with pytest.raises(ValidationError):
        FMAConfig(**test_config_dict)

def test_fma_creation_df_missing_columns(sample_fma_data: pd.DataFrame):
    """Test FMA instantiation if df is missing essential columns.
    This validation typically occurs during fit(), not Pydantic model creation,
    Validation now occurs at config instantiation due to BaseEstimatorConfig.
    """
    base_config_dict: Dict[str, Any] = {
        # "df" will be set in the loop
        "treat": "Treated", "time": "Time", "outcome": "Outcome", "unitid": "Unit",
    }
    essential_cols_map = { # map key to actual column name in config
        "outcome": base_config_dict["outcome"],
        "treat": base_config_dict["treat"],
        "unitid": base_config_dict["unitid"],
        "time": base_config_dict["time"],
    }
    
    for col_key_in_map, actual_col_name_to_drop in essential_cols_map.items():
        df_missing_col = sample_fma_data.copy()
        if actual_col_name_to_drop in df_missing_col.columns:
             df_missing_col = df_missing_col.drop(columns=[actual_col_name_to_drop])
        else:
            # This should not happen if base_config_dict keys match sample_fma_data columns
            pytest.skip(f"Column '{actual_col_name_to_drop}' not found in fixture to drop.")
            continue
        
        current_config_dict = base_config_dict.copy()
        current_config_dict["df"] = df_missing_col
        
        expected_message = f"Missing required columns in DataFrame 'df': {actual_col_name_to_drop}"
        with pytest.raises(MlsynthDataError, match=expected_message):
            FMAConfig(**current_config_dict)

def test_fma_creation_invalid_config_values(sample_fma_data: pd.DataFrame):
    """Test FMA instantiation with invalid types or values for 'criti' and 'DEMEAN'."""
    base_config_dict: Dict[str, Any] = {
        "df": sample_fma_data,
        "treat": "Treated", "time": "Time", "outcome": "Outcome", "unitid": "Unit",
    }

    # Invalid types for Pydantic validation
    config_invalid_type_criti_dict = base_config_dict.copy()
    config_invalid_type_criti_dict["criti"] = "not_an_int"
    with pytest.raises(ValidationError):
        FMAConfig(**config_invalid_type_criti_dict)
    
    config_invalid_type_demean_dict = base_config_dict.copy()
    config_invalid_type_demean_dict["DEMEAN"] = "not_an_int"
    with pytest.raises(ValidationError):
        FMAConfig(**config_invalid_type_demean_dict)

    # Semantically invalid values (type is correct, but value might be out of expected range)
    # FMAConfig might not validate these ranges unless explicitly defined (e.g. Literal[1,2] for DEMEAN)
    # The original test checked fit behavior. We'll keep that part.

    # Invalid criti (e.g., 99, if FMAConfig allows any int but FMA handles specific values)
    # FMA defaults to nfactor_xu_val if criti is not 10 or 11.
    config_sem_invalid_criti_dict = base_config_dict.copy()
    config_sem_invalid_criti_dict["criti"] = 99
    # FMAConfig should raise ValidationError for criti=99 due to ge=10, le=11 constraint
    with pytest.raises(ValidationError, match="Input should be less than or equal to 11"):
        FMAConfig(**config_sem_invalid_criti_dict)
    # The following lines testing fit() with criti=99 are unreachable if FMAConfig validation is active.
    # If FMAConfig were to allow criti=99, then the fit() behavior would be tested.
    # try:
    #     # This part assumes FMAConfig allows criti=99, which it doesn't by default.
    #     # To test fit's internal handling of criti not in [10, 11], one might need to
    #     # bypass or temporarily alter FMAConfig validation for the test.
    #     # config_obj_criti = FMAConfig(**config_sem_invalid_criti_dict) 
    #     # estimator_criti = FMA(config=config_obj_criti)
    #     # estimator_criti.fit() # Should run with default factor selection
    #     pass # Test for fit() with invalid criti (if config allowed it) would go here
    # except Exception as e:
    #     pytest.fail(f"FMA fit failed unexpectedly with criti=99: {e}")


    # Invalid DEMEAN (e.g., 3, if FMAConfig allows any int but FMA/denoiseutils handles 1 or 2)
    config_sem_invalid_demean_dict = base_config_dict.copy()
    config_sem_invalid_demean_dict["DEMEAN"] = 3
    # FMAConfig should raise ValidationError for DEMEAN=3 due to ge=1, le=2 constraint
    with pytest.raises(ValidationError, match="Input should be less than or equal to 2"):
        FMAConfig(**config_sem_invalid_demean_dict)
    # The following lines testing fit() with DEMEAN=3 are unreachable if FMAConfig validation is active.
    # try:
    #     # config_obj_demean = FMAConfig(**config_sem_invalid_demean_dict) # This would fail
    #     # estimator_demean = FMA(config=config_obj_demean)
    #     # estimator_demean.fit()
    #     pass # Test for fit() with invalid DEMEAN (if config allowed it) would go here
    # except Exception as e:
    #     pytest.fail(f"FMA fit failed unexpectedly with DEMEAN=3: {e}")


# --- Edge Case Tests ---

@patch('mlsynth.estimators.fma.plot_estimates')
def test_fma_fit_insufficient_pre_periods(mock_plot_estimates, sample_fma_data: pd.DataFrame):
    """Test FMA fit with insufficient pre-treatment periods."""
    config_dict: Dict[str, Any] = {
        "df": sample_fma_data.copy(),
        "treat": "Treated", "time": "Time", "outcome": "Outcome", "unitid": "Unit",
        "display_graphs": False, "criti": 11, "DEMEAN": 1,
    }
    # Modify data to have very few pre-periods. Original has 12 pre-periods.
    # FMA's LOO CV needs t1_pre_periods > 0. Factor estimation needs enough rows.
    # Let's set treatment start to period 2, so 1 pre-period.
    df_mod = config_dict["df"] # type: ignore
    treatment_start_period = 2 
    df_mod.loc[(df_mod['Unit'] == 1) & (df_mod['Time'] >= treatment_start_period), 'Treated'] = 1
    df_mod.loc[(df_mod['Unit'] == 1) & (df_mod['Time'] < treatment_start_period), 'Treated'] = 0
    # Ensure other units are not treated for this test
    df_mod.loc[df_mod['Unit'] != 1, 'Treated'] = 0

    config_obj = FMAConfig(**config_dict)
    estimator = FMA(config=config_obj)
    # Expecting failure during LOO CV (e.g., np.delete on array of size 1) or matrix inversion
    # or in nbpiid if t1_pre_periods is too small for its internal logic.
    
    with pytest.raises((ValueError, IndexError, np.linalg.LinAlgError, MlsynthEstimationError)):
        estimator.fit()

    mock_plot_estimates.assert_not_called()


@patch('mlsynth.estimators.fma.plot_estimates')
def test_fma_fit_insufficient_donors(mock_plot_estimates, sample_fma_data):
    """Test FMA fit behavior when there are insufficient donor units."""
    base_config_dict: Dict[str, Any] = {
        "treat": "Treated",
        "time": "Time",
        "outcome": "Outcome",
        "unitid": "Unit",
        "display_graphs": False,
        "criti": 11,
        "DEMEAN": 1,
    }

    # --- Case 1: No donor units ---
    df_no_donors = sample_fma_data[sample_fma_data["Unit"] == 1].copy()
    config_no_donors_dict = {**base_config_dict, "df": df_no_donors}
    estimator_no_donors = FMA(FMAConfig(**config_no_donors_dict))

    # Expect a specific failure due to no donor units
    with pytest.raises(MlsynthDataError, match="No donor units found"):
        estimator_no_donors.fit()

    # --- Case 2: Only one donor unit ---
    df_one_donor = sample_fma_data[sample_fma_data["Unit"].isin([1, 2])].copy()
    config_one_donor_dict = {**base_config_dict, "df": df_one_donor}
    estimator_one_donor = FMA(FMAConfig(**config_one_donor_dict))

    try:
        results = estimator_one_donor.fit()
        assert isinstance(results, BaseEstimatorResults), "Fit should return a BaseEstimatorResults object."
    except (ValueError, IndexError, np.linalg.LinAlgError, MlsynthDataError, MlsynthConfigError) as e:
        # Expected for degenerate cases like singular matrix or low donor count
        print(f"Caught expected error with one donor: {e}")
        pass
    except Exception as e:
        pytest.fail(f"FMA fit with one donor failed unexpectedly: {e}")

    # Ensure plotting is not triggered during smoke tests
    mock_plot_estimates.assert_not_called()


@patch('mlsynth.estimators.fma.plot_estimates')
def test_fma_fit_no_post_periods(mock_plot_estimates, sample_fma_data: pd.DataFrame):
    """Test FMA fit with no post-treatment periods."""
    config_dict: Dict[str, Any] = {
        "df": sample_fma_data.copy(),
        "treat": "Treated", "time": "Time", "outcome": "Outcome", "unitid": "Unit",
        "display_graphs": False, "criti": 11, "DEMEAN": 1,
    }
    df_mod = config_dict["df"] # type: ignore
    last_period = df_mod['Time'].max()
    # Treat unit 1 at a time where no post-periods exist
    df_mod.loc[(df_mod['Unit'] == 1) & (df_mod['Time'] >= last_period + 1), 'Treated'] = 1
    df_mod.loc[(df_mod['Unit'] == 1) & (df_mod['Time'] < last_period + 1), 'Treated'] = 0
    df_mod.loc[df_mod['Unit'] != 1, 'Treated'] = 0

    config_obj = FMAConfig(**config_dict)
    estimator = FMA(config=config_obj)
    # dataprep's logictreat should raise an AssertionError
    with pytest.raises(AssertionError, match="No treated units found"):
        estimator.fit()
    mock_plot_estimates.assert_not_called()


@patch('mlsynth.estimators.fma.plot_estimates')
def test_fma_fit_nan_in_outcome(mock_plot_estimates, sample_fma_data: pd.DataFrame):
    """Test FMA fit when outcome variable contains NaN values."""
    base_config_dict: Dict[str, Any] = {
        "treat": "Treated", "time": "Time", "outcome": "Outcome", "unitid": "Unit",
        "display_graphs": False, "criti": 11, "DEMEAN": 1,
    }
    
    # Case 1: NaN in treated unit's pre-period outcome
    df_mod_treated_nan = sample_fma_data.copy()
    df_mod_treated_nan.loc[(df_mod_treated_nan['Unit'] == 1) & (df_mod_treated_nan['Time'] == 5), 'Outcome'] = np.nan
    config_treated_nan_dict = {**base_config_dict, "df": df_mod_treated_nan}
    config_obj_treated_nan = FMAConfig(**config_treated_nan_dict)
    estimator_nan_treated = FMA(config=config_obj_treated_nan)
    try:
        results = estimator_nan_treated.fit()
        # If fit completes, ATT or SE might be NaN
        assert (results.effects is None or np.isnan(results.effects.att)) or \
               (results.inference is None or np.isnan(results.inference.standard_error)), \
               "ATT or SE should be NaN if fit completes with NaNs in treated outcome"
    except (ValueError, np.linalg.LinAlgError, TypeError) as e:
        # LinAlgError from inv() if matrices become singular due to NaNs
        # ValueError from np.mean of all-NaN slice, etc.
        print(f"Caught expected error for NaN in treated unit: {e}")
        pass 
    except Exception as e:
        pytest.fail(f"FMA fit with NaN in treated unit failed unexpectedly: {e}")
    
    mock_plot_estimates.reset_mock()

    # Case 2: NaN in a control unit's pre-period outcome
    # This will affect donor_matrix, then X_processed, then factor estimation.
    df_mod_control_nan = sample_fma_data.copy() # Reset df
    df_mod_control_nan.loc[(df_mod_control_nan['Unit'] == 2) & (df_mod_control_nan['Time'] == 5), 'Outcome'] = np.nan
    config_control_nan_dict = {**base_config_dict, "df": df_mod_control_nan}
    config_obj_control_nan = FMAConfig(**config_control_nan_dict)
    estimator_nan_control = FMA(config=config_obj_control_nan)
    try:
        results = estimator_nan_control.fit()
        # Factor model might still compute, but results could be affected
        assert (results.effects is None or np.isnan(results.effects.att)) or \
               (results.inference is None or np.isnan(results.inference.standard_error)) or \
               (results.time_series is None or results.time_series.counterfactual_outcome is None or np.all(np.isnan(results.time_series.counterfactual_outcome))), \
               "ATT, SE, or Counterfactual should be NaN if fit completes with NaNs in control outcome"
    except (ValueError, np.linalg.LinAlgError, TypeError) as e:
        print(f"Caught expected error for NaN in control unit: {e}")
        pass
    except Exception as e:
        pytest.fail(f"FMA fit with NaN in control unit failed unexpectedly: {e}")

    mock_plot_estimates.reset_mock()

# --- Detailed Results Validation ---

def test_fma_fit_results_structure_detailed(sample_fma_data: pd.DataFrame):
    """Test the detailed structure of the results dictionary from FMA fit."""
    config_dict: Dict[str, Any] = {
        "df": sample_fma_data,
        "treat": "Treated", "time": "Time", "outcome": "Outcome", "unitid": "Unit",
        "display_graphs": False, "criti": 11, "DEMEAN": 1,
    }
    config_obj = FMAConfig(**config_dict)
    estimator = FMA(config=config_obj)
    results: BaseEstimatorResults = estimator.fit()

    assert results.effects is not None
    effects_res = results.effects
    assert effects_res.att is not None and isinstance(effects_res.att, (float, np.floating))
    # ATT_Time was in additional_effects in the mapping
    assert effects_res.additional_effects is not None
    assert "ATT_Time" in effects_res.additional_effects and isinstance(effects_res.additional_effects["ATT_Time"], np.ndarray)
    
    assert results.fit_diagnostics is not None
    fit_diag_res = results.fit_diagnostics
    assert fit_diag_res.pre_treatment_rmse is not None and isinstance(fit_diag_res.pre_treatment_rmse, (float, np.floating))
    assert fit_diag_res.pre_treatment_r_squared is not None and isinstance(fit_diag_res.pre_treatment_r_squared, (float, np.floating))

    assert results.time_series is not None
    ts_res = results.time_series
    assert ts_res.observed_outcome is not None and isinstance(ts_res.observed_outcome, np.ndarray)
    assert ts_res.counterfactual_outcome is not None and isinstance(ts_res.counterfactual_outcome, np.ndarray)
    assert ts_res.estimated_gap is not None and isinstance(ts_res.estimated_gap, np.ndarray)
    assert len(ts_res.observed_outcome) == len(ts_res.counterfactual_outcome) == len(ts_res.estimated_gap)
    
    unique_time_periods = config_dict["df"]["Time"].nunique() # type: ignore
    assert len(ts_res.observed_outcome) == unique_time_periods
    assert ts_res.time_periods is not None and len(ts_res.time_periods) == unique_time_periods


    assert results.inference is not None
    inf_res = results.inference
    assert inf_res.standard_error is not None and isinstance(inf_res.standard_error, (float, np.floating))
    assert inf_res.details is not None and "t_statistic" in inf_res.details and isinstance(inf_res.details["t_statistic"], (float, np.floating))
    assert inf_res.ci_lower_bound is not None and inf_res.ci_upper_bound is not None
    assert isinstance(inf_res.ci_lower_bound, (float, np.floating))
    assert isinstance(inf_res.ci_upper_bound, (float, np.floating))
    assert inf_res.p_value is not None and isinstance(inf_res.p_value, (float, np.floating))


# --- Configuration Variations ---

@patch('mlsynth.estimators.fma.plot_estimates')
def test_fma_fit_config_variations(mock_plot_estimates, sample_fma_data: pd.DataFrame):
    """Test FMA fit with different 'criti' and 'DEMEAN' configurations."""
    base_config_dict: Dict[str, Any] = {
        "df": sample_fma_data,
        "treat": "Treated", "time": "Time", "outcome": "Outcome", "unitid": "Unit",
        "display_graphs": False,
    }

    configs_to_test_params = [
        {"criti": 10, "DEMEAN": 1}, # Stationary, demean
        {"criti": 11, "DEMEAN": 2}, # Nonstationary, standardize
        {"criti": 10, "DEMEAN": 2}, # Stationary, standardize
    ]

    for var_params in configs_to_test_params:
        current_config_dict = {**base_config_dict, **var_params}
        config_obj = FMAConfig(**current_config_dict)
        estimator = FMA(config=config_obj)
        try:
            results = estimator.fit()
            assert isinstance(results, BaseEstimatorResults)
            assert results.effects is not None and results.effects.att is not None
        except Exception as e:
            pytest.fail(f"FMA fit failed for config {var_params}: {e}") # Corrected var_config to var_params
    mock_plot_estimates.assert_not_called() # display_graphs is False


# --- Plotting Behavior ---

@patch('mlsynth.estimators.fma.plot_estimates')
def test_fma_plotting_behavior_display_true(mock_plot_func, sample_fma_data: pd.DataFrame):
    """Test that plot_estimates is called when display_graphs is True."""
    config_dict: Dict[str, Any] = {
        "df": sample_fma_data,
        "treat": "Treated", "time": "Time", "outcome": "Outcome", "unitid": "Unit",
        "display_graphs": True, "criti": 11, "DEMEAN": 1,
    }
    config_obj = FMAConfig(**config_dict)
    estimator = FMA(config=config_obj)
    estimator.fit()
    mock_plot_func.assert_called_once()
    # Check some key args passed to plot_estimates
    args, kwargs = mock_plot_func.call_args
    assert "df" in kwargs # prepped data
    assert "y" in kwargs
    assert "cf_list" in kwargs and isinstance(kwargs["cf_list"], list) and len(kwargs["cf_list"]) == 1
    assert "method" in kwargs and kwargs["method"] == "FMA"


@patch('mlsynth.estimators.fma.plot_estimates')
def test_fma_plotting_behavior_display_false(mock_plot_func, sample_fma_data: pd.DataFrame):
    """Test that plot_estimates is NOT called when display_graphs is False."""
    config_dict: Dict[str, Any] = {
        "df": sample_fma_data,
        "treat": "Treated", "time": "Time", "outcome": "Outcome", "unitid": "Unit",
        "display_graphs": False, "criti": 11, "DEMEAN": 1,
    }
    config_obj = FMAConfig(**config_dict)
    estimator = FMA(config=config_obj)
    estimator.fit()
    mock_plot_func.assert_not_called()
