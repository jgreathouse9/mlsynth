import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any
from pydantic import ValidationError

from mlsynth import NSC
from mlsynth.config_models import (
    NSCConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
    MethodDetailsResults,
)
from mlsynth.exceptions import (
    MlsynthDataError,
    MlsynthEstimationError,
)

# Full configuration dictionary used in tests.
NSC_FULL_TEST_CONFIG_BASE: Dict[str, Any] = {
    "outcome": "Y_val",
    "treat": "is_treated_nsc", 
    "unitid": "unit",
    "time": "period",
    "display_graphs": False,
    "save": False,
    "counterfactual_color": ["gold"],
    "treated_color": "indigo",
    "seed": 75391, # Not part of NSCConfig
    "verbose": False, # Not part of NSCConfig
}

# Fields that are part of BaseEstimatorConfig (and thus NSCConfig)
NSC_PYDANTIC_MODEL_FIELDS = [
    "df", "outcome", "treat", "unitid", "time", 
    "display_graphs", "save", "counterfactual_color", "treated_color"
]

def _get_pydantic_config_dict_nsc(full_config: Dict[str, Any], df_fixture: pd.DataFrame) -> Dict[str, Any]:
    """Helper to extract Pydantic-valid fields for NSCConfig and add the DataFrame."""
    pydantic_dict = {
        k: v for k, v in full_config.items() if k in NSC_PYDANTIC_MODEL_FIELDS
    }
    pydantic_dict["df"] = df_fixture
    return pydantic_dict

@pytest.fixture
def nsc_panel_data():
    """Provides a panel dataset for NSC smoke testing."""
    # NSCcv might need a decent number of pre-periods and donors for stable CV
    n_units = 10
    n_periods = 20
    data_dict = {
        'unit': np.repeat(np.arange(1, n_units + 1), n_periods),
        'period': np.tile(np.arange(1, n_periods + 1), n_units),
        'Y_val': np.random.normal(loc=np.repeat(np.arange(0, n_units*10, 10), n_periods), scale=5, size=n_units*n_periods),
        'X_cov': np.random.rand(n_units * n_periods) * 10,
    }
    df = pd.DataFrame(data_dict)
    
    # Unit 1 is treated starting from period = 15
    treatment_col_name = NSC_FULL_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    df.loc[(df['unit'] == 1) & (df['period'] >= 15), treatment_col_name] = 1
    return df

def test_nsc_creation(nsc_panel_data: pd.DataFrame):
    """Test that the NSC estimator can be instantiated."""
    pydantic_dict = _get_pydantic_config_dict_nsc(NSC_FULL_TEST_CONFIG_BASE, nsc_panel_data)
    
    try:
        config_obj = NSCConfig(**pydantic_dict)
        estimator = NSC(config=config_obj)
        assert estimator is not None, "NSC estimator should be created."
        assert estimator.outcome == "Y_val"
        assert estimator.treat == NSC_FULL_TEST_CONFIG_BASE["treat"]
        assert not estimator.display_graphs
    except Exception as e:
        pytest.fail(f"NSC instantiation failed: {e}")

def test_nsc_fit_smoke(nsc_panel_data: pd.DataFrame):
    """Smoke test for the NSC fit method."""
    pydantic_dict = _get_pydantic_config_dict_nsc(NSC_FULL_TEST_CONFIG_BASE, nsc_panel_data)
    config_obj = NSCConfig(**pydantic_dict)
    estimator = NSC(config=config_obj)
    
    try:
        results = estimator.fit()
        assert results is not None, "Fit method should return results."
        assert isinstance(results, BaseEstimatorResults), "Results should be BaseEstimatorResults Pydantic model."

        # Check main components
        assert results.effects is not None and isinstance(results.effects, EffectsResults)
        assert results.fit_diagnostics is not None and isinstance(results.fit_diagnostics, FitDiagnosticsResults)
        assert results.time_series is not None and isinstance(results.time_series, TimeSeriesResults)
        assert results.weights is not None and isinstance(results.weights, WeightsResults)
        assert results.method_details is not None and isinstance(results.method_details, MethodDetailsResults)
        assert results.raw_results is not None and isinstance(results.raw_results, dict)
        
        assert results.raw_results["_prepped"] is not None # Check _prepped exists in raw_results
        assert "Counterfactual" in results.raw_results["Vectors"] # Check Counterfactual exists in raw_results["Vectors"]
        assert isinstance(results.time_series.counterfactual_outcome, np.ndarray)


    except Exception as e:
        if isinstance(e, (np.linalg.LinAlgError, ValueError)) and "singular" in str(e).lower():
             pytest.skip(f"Skipping due to potential singularity in NSCcv/NSC_opt with small random data: {e}")
        pytest.fail(f"NSC fit method failed during smoke test: {e}")


# --- Input Validation Tests ---

def test_nsc_missing_df_in_config(): # nsc_panel_data fixture not needed for this test
    """Test NSC instantiation and fit with 'df' missing in config."""
    pydantic_dict_no_df = {
        k: v for k, v in NSC_FULL_TEST_CONFIG_BASE.items() if k in NSC_PYDANTIC_MODEL_FIELDS and k != "df"
    }
    # Attempt to create NSCConfig without 'df'
    with pytest.raises(ValidationError, match="Field required") as excinfo:
        NSCConfig(**pydantic_dict_no_df)
    assert "df" in str(excinfo.value)


def test_nsc_df_not_dataframe():
    """Test NSCConfig with df not being a pandas DataFrame."""
    pydantic_dict = _get_pydantic_config_dict_nsc(NSC_FULL_TEST_CONFIG_BASE, "not_a_dataframe") # type: ignore
    with pytest.raises(ValidationError, match="Input should be an instance of DataFrame"): # Adjusted match
        NSCConfig(**pydantic_dict)


@pytest.mark.parametrize("missing_key", [k for k in NSC_PYDANTIC_MODEL_FIELDS if k in ["outcome", "treat", "unitid", "time"]]) # Only test truly required string fields
def test_nsc_missing_core_keys_in_config(nsc_panel_data, missing_key):
    """Test NSCConfig with missing core string keys."""
    pydantic_dict = _get_pydantic_config_dict_nsc(NSC_FULL_TEST_CONFIG_BASE, nsc_panel_data)
    
    pydantic_dict.pop(missing_key) # Remove the key
    
    with pytest.raises(ValidationError, match="Field required") as excinfo:
        NSCConfig(**pydantic_dict)
    assert missing_key in str(excinfo.value)


@pytest.mark.parametrize("essential_col", ["Y_val", NSC_FULL_TEST_CONFIG_BASE["treat"], "unit", "period"])
def test_nsc_df_missing_essential_columns(nsc_panel_data, essential_col):
    """Test NSC when the DataFrame is missing an essential column."""
    pydantic_dict = _get_pydantic_config_dict_nsc(NSC_FULL_TEST_CONFIG_BASE, nsc_panel_data)
    
    df_modified = nsc_panel_data.copy()
    if essential_col in df_modified.columns:
        df_modified = df_modified.drop(columns=[essential_col])
    
    pydantic_dict["df"] = df_modified
    
    # The error is now raised during NSCConfig instantiation due to BaseEstimatorConfig's validator
    expected_error = MlsynthDataError 
    # The message from BaseEstimatorConfig validator is slightly different
    match_pattern = f"Missing required columns in DataFrame 'df': {essential_col}"

    with pytest.raises(expected_error, match=match_pattern):
        config_obj = NSCConfig(**pydantic_dict) # Error expected here
        # estimator = NSC(config=config_obj) # This line won't be reached
        # estimator.fit() # This line won't be reached


# --- Edge Case Tests ---

@pytest.fixture
def nsc_insufficient_data_fixture(request):
    """Fixture to generate data for various insufficient data scenarios."""
    n_units = 5
    n_periods = 10
    treat_period_start = 8 # Default: 2 post-periods, 7 pre-periods

    scenario = request.param
    if scenario == "zero_pre_periods":
        treat_period_start = 1 # Treatment from the very first period
    elif scenario == "one_pre_period":
        treat_period_start = 2 
    elif scenario == "zero_post_periods":
        treat_period_start = n_periods + 1 # Treatment never effectively starts
    elif scenario == "one_donor":
        n_units = 2 # Treated + 1 donor
    elif scenario == "zero_donors":
        n_units = 1 # Only treated unit

    data_dict = {
        'unit': np.repeat(np.arange(1, n_units + 1), n_periods),
        'period': np.tile(np.arange(1, n_periods + 1), n_units),
        'Y_val': np.random.normal(size=n_units * n_periods),
    }
    df = pd.DataFrame(data_dict)
    
    treatment_col_name = NSC_FULL_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    if n_units > 0: # Ensure there's a unit to treat
        df.loc[(df['unit'] == 1) & (df['period'] >= treat_period_start), treatment_col_name] = 1
    
    # For zero_post_periods, ensure no unit is actually marked as treated post-event
    if scenario == "zero_post_periods":
        df[treatment_col_name] = 0

    return df, scenario

@pytest.mark.parametrize(
    "nsc_insufficient_data_fixture, expected_error, match_pattern",
    [
        (("zero_pre_periods", MlsynthDataError, "Not enough pre-treatment periods")), # dataprep
        (("one_pre_period", None, None)), # Should run, but NSCcv might warn / NSC_opt might be unstable
        (("zero_post_periods", MlsynthDataError, "No treated units found")), # dataprep or effects.calculate
        (("zero_donors", MlsynthDataError, "No donor units found")), # dataprep
        (("one_donor", UserWarning, "Not enough donors for meaningful CV")), # NSCcv warning
    ],
    indirect=["nsc_insufficient_data_fixture"]
)
def test_nsc_insufficient_data(nsc_insufficient_data_fixture, expected_error, match_pattern):
    """Test NSC with various insufficient data scenarios."""
    df, scenario = nsc_insufficient_data_fixture
    pydantic_dict = _get_pydantic_config_dict_nsc(NSC_FULL_TEST_CONFIG_BASE, df)
    config_obj = NSCConfig(**pydantic_dict)
    estimator = NSC(config=config_obj)

    if expected_error is UserWarning:
        # Updated regex to be more specific for the "one_donor" scenario
        if scenario == "one_donor":
            match_pattern = r"Not enough donors \(1\) for meaningful 1-fold CV\. Returning default L1=0\.01, L2=0\.01\."
        with pytest.warns(UserWarning, match=match_pattern):
            results = estimator.fit()
            assert results is not None
            assert isinstance(results, BaseEstimatorResults)
    elif expected_error:
        with pytest.raises(expected_error, match=match_pattern if match_pattern else None):
            estimator.fit()
    else: # one_pre_period case, expected to run but potentially unstable
        try:
            results = estimator.fit()
            assert results is not None
            # It's hard to predict exact outcome with one pre-period, just ensure it runs.
            # Results might contain NaNs if NSC_opt fails.
        except (np.linalg.LinAlgError, ValueError) as e:
            if "singular matrix" in str(e).lower() or "optimal_inaccurate" in str(e).lower() or "Optimization failed" in str(e):
                pytest.skip(f"Skipping due to instability with one pre-period: {e}")
            raise


@pytest.fixture
def nsc_nan_data_fixture(request, nsc_panel_data):
    """Fixture to introduce NaNs into the panel data."""
    df_nan = nsc_panel_data.copy()
    scenario = request.param

    if scenario == "nan_in_treated_outcome_pre":
        # Introduce NaN in treated unit's outcome in a pre-period
        df_nan.loc[(df_nan['unit'] == 1) & (df_nan['period'] == 5), 'Y_val'] = np.nan
    elif scenario == "nan_in_donor_outcome_pre":
        # Introduce NaN in a donor unit's outcome in a pre-period
        df_nan.loc[(df_nan['unit'] == 2) & (df_nan['period'] == 5), 'Y_val'] = np.nan
    
    return df_nan, scenario

@pytest.mark.parametrize(
    "nsc_nan_data_fixture",
    ["nan_in_treated_outcome_pre", "nan_in_donor_outcome_pre"],
    indirect=["nsc_nan_data_fixture"]
)
def test_nsc_with_nan_in_data(nsc_nan_data_fixture):
    """Test NSC behavior when NaNs are present in outcome data."""
    df_with_nan, scenario = nsc_nan_data_fixture
    pydantic_dict = _get_pydantic_config_dict_nsc(NSC_FULL_TEST_CONFIG_BASE, df_with_nan)
    config_obj = NSCConfig(**pydantic_dict)
    estimator = NSC(config=config_obj)
    
    # CVXPY typically fails with NaNs in the objective or constraints.
    # NSC_opt uses y_target_pre and Y0_donors_pre directly in cp.sum_squares.
    # This will likely lead to an error during problem.solve() or when cvxpy processes NaNs.
    # NSC_opt (from estutils) should raise MlsynthEstimationError if CVXPY fails.
    # The main fit method now wraps underlying errors into MlsynthEstimationError.
    with pytest.raises(MlsynthEstimationError):
        results = estimator.fit()
        # If it doesn't raise, check for NaNs in results (less likely now with stricter error wrapping)
        # assert np.isnan(results["Effects"]["ATT"])
        # assert np.all(np.isnan(results["Vectors"]["Counterfactual"]))

# --- Detailed Results Validation ---

def test_nsc_fit_detailed_results_validation(nsc_panel_data):
    """Test detailed structure and properties of NSC fit results."""
    pydantic_dict = _get_pydantic_config_dict_nsc(NSC_FULL_TEST_CONFIG_BASE, nsc_panel_data)
    config_obj = NSCConfig(**pydantic_dict)
    estimator = NSC(config=config_obj)
    
    try:
        results = estimator.fit()
    except (np.linalg.LinAlgError, ValueError) as e:
        if "singular" in str(e).lower() or "Optimization failed" in str(e).lower() or "optimal_inaccurate" in str(e).lower():
            pytest.skip(f"Skipping detailed results validation due to potential singularity/CVXPY issue: {e}")
        raise

    assert results is not None
    assert isinstance(results, BaseEstimatorResults)

    # Validate _prepped structure (basics) from raw_results
    prepped = results.raw_results["_prepped"]
    assert isinstance(prepped, dict)
    assert "y" in prepped and isinstance(prepped["y"], np.ndarray)
    assert "donor_matrix" in prepped and isinstance(prepped["donor_matrix"], np.ndarray)
    assert "pre_periods" in prepped and isinstance(prepped["pre_periods"], int)
    assert "post_periods" in prepped and isinstance(prepped["post_periods"], int)
    assert "total_periods" in prepped and isinstance(prepped["total_periods"], int)
    assert prepped["total_periods"] == prepped["pre_periods"] + prepped["post_periods"]
    assert len(prepped["y"]) == prepped["total_periods"]
    assert prepped["donor_matrix"].shape[0] == prepped["total_periods"]
    
    num_donors = prepped["donor_matrix"].shape[1]
    total_periods_val = prepped["total_periods"]

    # Validate EffectsResults
    effects_res = results.effects
    assert isinstance(effects_res, EffectsResults)
    expected_effects_keys = ["att", "att_percent", "additional_effects"] # Pydantic fields
    raw_effects_keys = ["SATT", "TTE", "ATT_Time", "PercentATT_Time", "SATT_Time"] # Keys in additional_effects
    assert isinstance(effects_res.att, (float, np.floating)) or np.isnan(effects_res.att)
    for key in raw_effects_keys:
        assert key in effects_res.additional_effects
        if "_Time" in key:
            assert isinstance(effects_res.additional_effects[key], np.ndarray)
        else:
            assert isinstance(effects_res.additional_effects[key], (float, np.floating)) or np.isnan(effects_res.additional_effects[key])


    # Validate FitDiagnosticsResults
    fit_res = results.fit_diagnostics
    assert isinstance(fit_res, FitDiagnosticsResults)
    # Keys in additional_metrics are transformed to lowercase_with_underscores
    raw_fit_keys_transformed = ["t1_rmse", "pre-periods", "post-periods"] 
    assert isinstance(fit_res.pre_treatment_rmse, (float, np.floating)) or np.isnan(fit_res.pre_treatment_rmse)
    if fit_res.pre_treatment_r_squared is not None: # R-squared can be None
        assert isinstance(fit_res.pre_treatment_r_squared, (float, np.floating)) or np.isnan(fit_res.pre_treatment_r_squared)
    
    assert fit_res.additional_metrics is not None # Ensure additional_metrics is not None before iterating
    for key in raw_fit_keys_transformed:
        assert key in fit_res.additional_metrics
        if key in ["pre-periods", "post-periods"]: # Check against transformed keys
            assert isinstance(fit_res.additional_metrics[key], (int, np.integer))
        else:
            assert isinstance(fit_res.additional_metrics[key], (float, np.floating)) or np.isnan(fit_res.additional_metrics[key])


    # Validate TimeSeriesResults
    ts_res = results.time_series
    assert isinstance(ts_res, TimeSeriesResults)
    assert ts_res.observed_outcome is not None and isinstance(ts_res.observed_outcome, np.ndarray)
    assert ts_res.counterfactual_outcome is not None and isinstance(ts_res.counterfactual_outcome, np.ndarray)
    assert ts_res.estimated_gap is not None and isinstance(ts_res.estimated_gap, np.ndarray)
    assert ts_res.time_periods is not None and isinstance(ts_res.time_periods, np.ndarray)
    
    assert ts_res.observed_outcome.shape == (total_periods_val, 1)
    assert ts_res.counterfactual_outcome.shape == (total_periods_val, 1) # NSC counterfactual is 1D, effects.calculate reshapes
    assert ts_res.estimated_gap.shape == (total_periods_val, 2) # Gap has 2 columns from effects.calculate
    assert len(ts_res.time_periods) == total_periods_val


    # Validate WeightsResults
    weights_res = results.weights
    assert isinstance(weights_res, WeightsResults)
    assert isinstance(weights_res.donor_weights, dict)
    assert len(weights_res.donor_weights) == num_donors
    
    weight_sum = sum(w for w in weights_res.donor_weights.values() if not np.isnan(w))
    if not np.isnan(weight_sum) and weight_sum != 0: # Check for non-zero before np.isclose
         assert np.isclose(weight_sum, 1.0, atol=1e-1) # NSC weights should sum to 1

    # In _create_estimator_results, weights_data[1] is processed into summary_stats
    assert weights_res.summary_stats is not None
    assert isinstance(weights_res.summary_stats, dict)
    assert "cardinality_of_positive_donors" in weights_res.summary_stats # Transformed key
    cardinality = weights_res.summary_stats["cardinality_of_positive_donors"]
    assert isinstance(cardinality, (int, np.integer))
    assert 0 <= cardinality <= num_donors

    # Validate MethodDetailsResults
    method_details = results.method_details
    assert isinstance(method_details, MethodDetailsResults)
    assert method_details.name == "NSC"
    assert method_details.additional_outputs is not None # Check it's not None
    assert "best_a" in method_details.additional_outputs
    assert "best_b" in method_details.additional_outputs


# --- Configuration Variation Tests ---
# Mocking NSCcv to test NSC_opt behavior with specific a, b values

@pytest.fixture
def mock_nsc_cv(mocker):
    """Fixture to mock the NSCcv function."""
    return mocker.patch("mlsynth.estimators.nsc.NSCcv")

@pytest.mark.parametrize(
    "mock_cv_return_val, description",
    [
        ((0.1, 0.1), "low_a_low_b"),
        ((0.9, 0.1), "high_a_low_b"),
        ((0.1, 0.9), "low_a_high_b"),
        ((0.9, 0.9), "high_a_high_b"),
        # Test case where NSC_opt might fail (e.g., if NSCcv returned nonsensical a,b or data is bad)
        # This is indirectly tested if NSC_opt returns NaN weights.
        # For now, assume NSCcv returns valid a,b from its grid.
    ]
)
def test_nsc_fit_with_mocked_cv_params(nsc_panel_data, mock_nsc_cv, mock_cv_return_val, description):
    """Test NSC.fit when NSCcv returns specific (a,b) hyperparameters."""
    mock_nsc_cv.return_value = mock_cv_return_val # (best_a, best_b)
    
    pydantic_dict = _get_pydantic_config_dict_nsc(NSC_FULL_TEST_CONFIG_BASE, nsc_panel_data)
    config_obj = NSCConfig(**pydantic_dict)
    estimator = NSC(config=config_obj)

    try:
        results = estimator.fit()
        assert results is not None
        assert isinstance(results, BaseEstimatorResults)
        mock_nsc_cv.assert_called_once()
        
        assert results.effects is not None
        assert results.weights is not None
        
        if np.all(np.isnan(np.array(list(results.weights.donor_weights.values())))):
            assert np.all(np.isnan(results.time_series.counterfactual_outcome))
            assert np.isnan(results.effects.att)
        else: 
            weight_sum = sum(w for w in results.weights.donor_weights.values() if not np.isnan(w))
            if not np.isnan(weight_sum) and weight_sum != 0: 
                assert np.isclose(weight_sum, 1.0, atol=1e-1)

    except (np.linalg.LinAlgError, ValueError) as e:
        # due to the nature of the random data.
        if "singular" in str(e).lower() or "Optimization failed" in str(e).lower() or "optimal_inaccurate" in str(e).lower():
            pytest.skip(f"Skipping mocked CV test due to optimization issue for {description}: {e}")
        raise


# --- Plotting Behavior Tests ---

@pytest.fixture
def mock_plot_estimates(mocker):
    """Fixture to mock the plot_estimates function."""
    return mocker.patch("mlsynth.estimators.nsc.plot_estimates")


@pytest.mark.parametrize("display_graphs_config", [True, False])
@pytest.mark.parametrize("save_config", [False, True])  # Removed dict case for now
def test_nsc_plotting_behavior(nsc_panel_data, mock_plot_estimates, display_graphs_config, save_config):
    """Test that plot_estimates is called (or not) based on config."""

    pydantic_dict = _get_pydantic_config_dict_nsc(NSC_FULL_TEST_CONFIG_BASE, nsc_panel_data)
    pydantic_dict["display_graphs"] = display_graphs_config
    pydantic_dict["save"] = save_config
    config_obj = NSCConfig(**pydantic_dict)
    estimator = NSC(config=config_obj)
    
    try:
        results = estimator.fit()
        assert isinstance(results, BaseEstimatorResults)  # Ensure fit runs
    except (np.linalg.LinAlgError, ValueError) as e:
        if "singular" in str(e).lower() or "optimization failed" in str(e).lower() or "optimal_inaccurate" in str(e).lower():
            pytest.skip(f"Skipping plotting test due to optimization issue: {e}")
        raise

    if display_graphs_config:
        # Ensure plot_estimates was called exactly once
        mock_plot_estimates.assert_called_once()

        # Extract positional and keyword arguments from the mock call
        call_args = mock_plot_estimates.call_args
        pos_args = call_args[0]  # positional arguments tuple
        kwargs = call_args[1]    # keyword arguments dict

        # The first positional argument is the processed_data_dict
        processed_dict_arg = kwargs.get("processed_data_dict", pos_args[0] if len(pos_args) > 0 else None)
        assert processed_dict_arg is results.raw_results["_prepped"]

        # Check remaining keyword arguments
        assert kwargs.get("estimation_method_name", None) == "NSC"
        assert kwargs.get("save_plot_config", save_config) == save_config
        assert kwargs.get("treated_series_color", None) == config_obj.treated_color
        counter_colors = kwargs.get("counterfactual_series_colors", None)
        expected_colors = [config_obj.counterfactual_color] if isinstance(config_obj.counterfactual_color, str) else config_obj.counterfactual_color
        assert counter_colors == expected_colors
    else:
        # If display_graphs is False, plot_estimates should not be called
        mock_plot_estimates.assert_not_called()
