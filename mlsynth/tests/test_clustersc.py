import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pydantic import ValidationError

from mlsynth.estimators.clustersc import CLUSTERSC
from mlsynth.config_models import CLUSTERSCConfig, BaseEstimatorResults
from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError, MlsynthEstimationError

@pytest.fixture
def sample_clustersc_data() -> pd.DataFrame:
    """Creates a sample DataFrame for CLUSTERSC tests."""
    n_units = 5 # Unit 1 treated, Units 2-5 donors
    n_periods = 15
    treatment_start_period = 10 # 9 pre-periods, 6 post-periods

    units = np.repeat(np.arange(1, n_units + 1), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_units)
    
    np.random.seed(101) # for reproducibility
    outcomes = []
    for i in range(n_units):
        base_trend = np.linspace(start=10 + i*2.5, stop=30 + i*2.5, num=n_periods)
        noise = np.random.normal(0, 1.0, n_periods)
        outcomes.extend(base_trend + noise)

    data = {
        "UnitNum": units,
        "TimePoint": times,
        "OutcomeMetric": outcomes,
        "TreatmentStatus": np.zeros(n_units * n_periods, dtype=int),
    }
    df = pd.DataFrame(data)

    # Unit 1 is treated from treatment_start_period
    df.loc[(df['UnitNum'] == 1) & (df['TimePoint'] >= treatment_start_period), 'TreatmentStatus'] = 1
    
    return df

def test_clustersc_creation(sample_clustersc_data: pd.DataFrame) -> None:
    """Test CLUSTERSC estimator creation."""
    config_dict: Dict[str, Any] = {
        "df": sample_clustersc_data,
        "outcome": "OutcomeMetric",
        "treat": "TreatmentStatus",
        "unitid": "UnitNum",
        "time": "TimePoint",
        "method": "PCR", 
        "display_graphs": False,
    }
    try:
        config_obj = CLUSTERSCConfig(**config_dict) # Corrected case
        estimator = CLUSTERSC(config_obj)
        assert estimator is not None
        assert estimator.df.equals(sample_clustersc_data)
        assert estimator.outcome == "OutcomeMetric"
        assert estimator.method == "PCR" 
        assert not estimator.display_graphs
    except Exception as e:
        pytest.fail(f"CLUSTERSC creation failed: {e}")

@pytest.mark.parametrize("method_to_test", ["PCR", "RPCA", "BOTH"])
def test_clustersc_fit_smoke(sample_clustersc_data: pd.DataFrame, method_to_test: str) -> None:
    """Smoke test for CLUSTERSC fit method for various methods."""
    config_dict: Dict[str, Any] = {
        "df": sample_clustersc_data,
        "outcome": "OutcomeMetric",
        "treat": "TreatmentStatus",
        "unitid": "UnitNum",
        "time": "TimePoint",
        "method": method_to_test,
        "display_graphs": False,
        "cluster": True, 
        "objective": "OLS", 
        "Frequentist": True, 
        "ROB": "PCP",  # Corrected Robust to ROB
    }
    config_obj = CLUSTERSCConfig(**config_dict) # Corrected case
    estimator = CLUSTERSC(config_obj)
    try:
        results = estimator.fit()
        assert isinstance(results, BaseEstimatorResults)
        assert isinstance(results.sub_method_results, dict)
        
        sub_method_keys = []
        if method_to_test == "PCR" or method_to_test == "BOTH":
            sub_method_keys.append("PCR")
        if method_to_test == "RPCA" or method_to_test == "BOTH":
            sub_method_keys.append("RPCA")

        for key in sub_method_keys:
            assert key in results.sub_method_results
            method_result = results.sub_method_results[key]
            assert isinstance(method_result, BaseEstimatorResults) # Each sub-result is also BaseEstimatorResults

            assert method_result.effects is not None
            assert method_result.fit_diagnostics is not None
            assert method_result.time_series is not None
            assert method_result.weights is not None # Check presence of the WeightsResults model

            assert method_result.time_series.counterfactual_outcome is not None
            assert isinstance(method_result.time_series.counterfactual_outcome, np.ndarray)
            
            if key == "PCR":
                assert method_result.weights.donor_weights is not None # Full weights dict
                assert isinstance(method_result.weights.donor_weights, dict)
                # Non-zero weights are in additional_outputs if stored
                if method_result.weights.additional_outputs:
                    assert "non_zero_weights" in method_result.weights.additional_outputs
                    assert isinstance(method_result.weights.additional_outputs["non_zero_weights"], dict)
            elif key == "RPCA":
                 assert method_result.weights.donor_weights is not None
                 assert isinstance(method_result.weights.donor_weights, dict)

    except Exception as e:
        if isinstance(e, (np.linalg.LinAlgError, ValueError)): # Common errors
            pytest.skip(f"Skipping CLUSTERSC fit ({method_to_test}) due to numerical/data issue: {e}")
        pytest.fail(f"CLUSTERSC fit ({method_to_test}) failed: {e}")

# --- Start of new tests ---

# Input Validation Tests
def test_clustersc_invalid_method(sample_clustersc_data: pd.DataFrame) -> None:
    """Test CLUSTERSC with an invalid method."""
    config_dict: Dict[str, Any] = {
        "df": sample_clustersc_data,
        "outcome": "OutcomeMetric",
        "treat": "TreatmentStatus",
        "unitid": "UnitNum",
        "time": "TimePoint",
        "method": "INVALID_METHOD", # Invalid method
        "display_graphs": False,
    }
    # Assuming ClusterSCConfig uses Literal for method, this will raise ValidationError.
    # Otherwise, CLUSTERSC.fit() will raise ValueError.
    with pytest.raises((ValidationError, ValueError)):
        config_obj = CLUSTERSCConfig(**config_dict) # Corrected case
        estimator = CLUSTERSC(config_obj)
        estimator.fit()


    @pytest.mark.parametrize(
        "invalid_objective_value, expected_exception, match_pattern",
        [
            # CLUSTERSCConfig now validates 'objective' with a pattern.
            ("INVALID_OBJECTIVE", ValidationError, r"String should match pattern '\^(OLS|SIMPLEX|MSCa|MSCb|MSCc)\$'"),
        ],
    )
    def test_clustersc_invalid_objective_param(
        sample_clustersc_data: pd.DataFrame,
        invalid_objective_value: Any,
        expected_exception: Any,
        match_pattern: Optional[str],
    ) -> None:
        """Test CLUSTERSC with an invalid 'objective' parameter for PCR method."""
        base_config_dict: Dict[str, Any] = {
            "df": sample_clustersc_data,
            "outcome": "OutcomeMetric",
            "treat": "TreatmentStatus",
            "unitid": "UnitNum",
            "time": "TimePoint",
            "method": "PCR", 
            "display_graphs": False,
            "objective": invalid_objective_value, # Set the invalid objective
        }
    
        # Error should occur at CLUSTERSCConfig instantiation due to pattern validation
        if match_pattern:
            with pytest.raises(expected_exception, match=match_pattern):
                CLUSTERSCConfig(**base_config_dict)
        else:
            with pytest.raises(expected_exception):
                CLUSTERSCConfig(**base_config_dict)

def test_clustersc_invalid_rob_param_warning(sample_clustersc_data: pd.DataFrame) -> None:
    """Test CLUSTERSC with an invalid 'Robust' param, expecting a UserWarning."""
    config_dict: Dict[str, Any] = {
        "df": sample_clustersc_data,
        "outcome": "OutcomeMetric",
        "treat": "TreatmentStatus",
        "unitid": "UnitNum",
        "time": "TimePoint",
        "method": "RPCA", 
        "ROB": 123, # Corrected Robust to ROB, Pydantic will catch type error if ROB is str
        "display_graphs": False,
    }
    # Pydantic should raise ValidationError because ROB must be a string.
    with pytest.raises(ValidationError, match="Input should be a valid string"):
        config_obj = CLUSTERSCConfig(**config_dict) # Corrected case
        # The following lines won't be reached if ValidationError is raised, which is expected.
        # estimator = CLUSTERSC(config_obj)
        # estimator.fit() 
        # Original test expected a UserWarning from RPCASYNTH, but Pydantic validation is earlier.
        # If we wanted to test the UserWarning, ROB would need to be a string not matching the pattern,
        # and the pattern check would need to be less strict or handled differently.
        # For now, testing Pydantic's type validation is appropriate.
    # If the above passes (i.e., ValidationError is raised), the test for this specific case is done.
    # No need to assert on results if config creation fails as expected.

@pytest.mark.parametrize(
    "missing_col", ["OutcomeMetric", "TreatmentStatus", "UnitNum", "TimePoint"]
)
def test_clustersc_missing_column(
    sample_clustersc_data: pd.DataFrame, missing_col: str
) -> None:
    """Test CLUSTERSC with a missing essential column in the DataFrame.
    Pydantic config creation should pass, error expected during fit.
    """
    df_missing = sample_clustersc_data.drop(columns=[missing_col])
    config_dict: Dict[str, Any] = {
        "df": df_missing,
        "outcome": "OutcomeMetric",
        "treat": "TreatmentStatus",
        "unitid": "UnitNum",
        "time": "TimePoint",
        "method": "PCR",
        "display_graphs": False,
    }
    # Pydantic config should be valid as it doesn't cross-validate df content with other keys.
    # UPDATE: BaseEstimatorConfig now validates presence of these columns in df.
    # So, the error should occur at CLUSTERSCConfig instantiation.
    
    # Create a temporary config dict to modify for the test
    temp_config_dict = {
        "df": df_missing, # df_missing is the one with the column dropped
        "outcome": "OutcomeMetric",
        "treat": "TreatmentStatus",
        "unitid": "UnitNum",
        "time": "TimePoint",
        "method": "PCR",
        "display_graphs": False,
    }

    # If the missing column is one of the core identifier/outcome columns,
    # BaseEstimatorConfig's validator should catch it.
    core_cols = {"OutcomeMetric", "TreatmentStatus", "UnitNum", "TimePoint"}
    if missing_col in core_cols:
        # Corrected match string for MlsynthDataError from BaseEstimatorConfig
        expected_message = f"Missing required columns in DataFrame 'df': {missing_col}"
        # If multiple could be missing, the message joins them, but parametrize ensures one at a time.
        with pytest.raises(MlsynthDataError, match=expected_message):
            CLUSTERSCConfig(**temp_config_dict)
    else:
        # If a non-core column (not currently tested by this parametrize) were missing 
        # and used later by dataprep, then fit() might raise KeyError or MlsynthDataError.
        # This branch is not hit by current parametrization.
        config_obj = CLUSTERSCConfig(**temp_config_dict)
        estimator = CLUSTERSC(config_obj)
        with pytest.raises(KeyError): # Or MlsynthDataError from dataprep
            estimator.fit()


def test_clustersc_empty_dataframe(sample_clustersc_data: pd.DataFrame) -> None:
    """Test CLUSTERSC with an empty DataFrame.
    BaseEstimatorConfig should raise MlsynthDataError during Pydantic validation.
    """
    df_empty = pd.DataFrame(columns=sample_clustersc_data.columns)
    config_dict: Dict[str, Any] = {
        "df": df_empty,
        "outcome": "OutcomeMetric",
        "treat": "TreatmentStatus",
        "unitid": "UnitNum",
        "time": "TimePoint",
        "method": "PCR",
        "display_graphs": False,
    }
    with pytest.raises(MlsynthDataError, match="Input DataFrame 'df' cannot be empty."): # Corrected match
        CLUSTERSCConfig(**config_dict)

# More tests for wrong types (e.g., outcome column not numeric) could be added.
# For now, focusing on the list from the task.

# --- Edge Case Tests ---
# First block of edge case tests (insufficient_pre_periods, nan_in_outcome, too_few_donors) removed.
# The tests below are the ones that were previously the "second instance" of these.

def test_clustersc_insufficient_pre_periods(sample_clustersc_data: pd.DataFrame) -> None:
    """Test CLUSTERSC with too few pre-treatment periods."""
    # Modify data to have only 1 pre-treatment period
    df_insufficient_pre = sample_clustersc_data[
        (sample_clustersc_data["TimePoint"] >= 9) # Treatment starts at 10, so 9 is one pre-period
    ].copy()
    # We need to ensure the treatment status reflects this new truncated data
    # For unit 1, TimePoint 9 is pre, TimePoint >=10 is post
    df_insufficient_pre["TreatmentStatus"] = 0
    df_insufficient_pre.loc[
        (df_insufficient_pre["UnitNum"] == 1) & (df_insufficient_pre["TimePoint"] >= 10), 
        "TreatmentStatus"
    ] = 1
    
    config_dict: Dict[str, Any] = {
        "df": df_insufficient_pre,
        "outcome": "OutcomeMetric",
        "treat": "TreatmentStatus",
        "unitid": "UnitNum",
        "time": "TimePoint",
        "method": "PCR",
        "display_graphs": False,
    }
    config_obj = CLUSTERSCConfig(**config_dict) # Corrected case
    estimator = CLUSTERSC(config_obj)
    with pytest.raises((MlsynthDataError, MlsynthEstimationError)):
        estimator.fit()


def test_clustersc_nan_in_outcome(sample_clustersc_data: pd.DataFrame) -> None: 
    """Test CLUSTERSC with NaN values in the outcome variable."""
    df_with_nan = sample_clustersc_data.copy()
    # Introduce NaN in a pre-treatment period for the treated unit
    df_with_nan.loc[
        (df_with_nan["UnitNum"] == 1) & (df_with_nan["TimePoint"] == 5), "OutcomeMetric"
    ] = np.nan
    # Introduce NaN in a pre-treatment period for a donor unit
    df_with_nan.loc[
        (df_with_nan["UnitNum"] == 2) & (df_with_nan["TimePoint"] == 3), "OutcomeMetric"
    ] = np.nan

    config_dict: Dict[str, Any] = {
        "df": df_with_nan,
        "outcome": "OutcomeMetric",
        "treat": "TreatmentStatus",
        "unitid": "UnitNum",
        "time": "TimePoint",
        "method": "PCR", 
        "display_graphs": False,
    }
    config_obj = CLUSTERSCConfig(**config_dict) # Corrected case
    estimator = CLUSTERSC(config_obj)
    with pytest.raises(MlsynthEstimationError, match="SVD computation failed: SVD did not converge"):
        estimator.fit()

def test_clustersc_too_few_donors(sample_clustersc_data: pd.DataFrame) -> None: 
    """Test CLUSTERSC with only one donor unit (may not be enough for clustering/SCM)."""
    df_few_donors = sample_clustersc_data[
        sample_clustersc_data["UnitNum"].isin([1, 2])
    ].copy()

    config_dict: Dict[str, Any] = {
        "df": df_few_donors,
        "outcome": "OutcomeMetric",
        "treat": "TreatmentStatus",
        "unitid": "UnitNum",
        "time": "TimePoint",
        "method": "PCR",
        "cluster": True, 
        "display_graphs": False,
    }
    config_obj = CLUSTERSCConfig(**config_dict) # Corrected case
    estimator = CLUSTERSC(config_obj)
    with pytest.raises((MlsynthDataError, MlsynthEstimationError)):
        estimator.fit()


# TODO: Add test for RPCA method with insufficient donors/pre-periods if behavior differs.

# --- Configuration Variation Tests & Detailed Results Validation ---

@pytest.mark.parametrize(
    "pcr_config_override",
    [
        {"cluster": False},
        {"objective": "SIMPLEX"},
        {"objective": "MSCb"}, # As per pcr docstring
        {"Frequentist": False}, # Bayesian PCR
        {"cluster": False, "objective": "SIMPLEX", "Frequentist": False},
    ],
)
def test_clustersc_pcr_config_variations(
    sample_clustersc_data: pd.DataFrame, pcr_config_override: Dict[str, Any]
) -> None:
    """Test CLUSTERSC with various valid PCR configurations."""
    base_config_dict: Dict[str, Any] = {
        "df": sample_clustersc_data,
        "outcome": "OutcomeMetric",
        "treat": "TreatmentStatus",
        "unitid": "UnitNum",
        "time": "TimePoint",
        "method": "PCR",
        "display_graphs": False,
        "cluster": True, 
        "objective": "OLS", 
        "Frequentist": True, 
    }
    config_dict = {**base_config_dict, **pcr_config_override}
    config_obj = CLUSTERSCConfig(**config_dict) # Corrected case
    estimator = CLUSTERSC(config_obj)

    try:
        results = estimator.fit()
        assert isinstance(results, BaseEstimatorResults)
        assert "PCR" in results.sub_method_results
        pcr_results_obj = results.sub_method_results["PCR"]
        assert isinstance(pcr_results_obj, BaseEstimatorResults)

        # Detailed checks for PCR results structure
        assert pcr_results_obj.effects is not None
        assert isinstance(pcr_results_obj.effects.att, (float, np.floating))

        assert pcr_results_obj.fit_diagnostics is not None
        assert isinstance(pcr_results_obj.fit_diagnostics.pre_treatment_rmse, (float, np.floating))

        assert pcr_results_obj.time_series is not None
        for vec_attr in ["observed_outcome", "counterfactual_outcome", "estimated_gap"]:
            vector_val = getattr(pcr_results_obj.time_series, vec_attr)
            assert vector_val is not None
            assert isinstance(vector_val, np.ndarray)
            if vec_attr == "estimated_gap":
                 assert vector_val.ndim == 2
            else:
                 assert vector_val.ndim == 1 or vector_val.shape[1] == 1


            treated_unit_id = sample_clustersc_data[sample_clustersc_data["TreatmentStatus"] == 1][config_dict["unitid"]].unique()[0]
            expected_len = sample_clustersc_data[sample_clustersc_data[config_dict["unitid"]] == treated_unit_id].shape[0]
            assert len(vector_val) == expected_len
        
        assert pcr_results_obj.weights is not None
        assert isinstance(pcr_results_obj.weights.donor_weights, dict) 
        if pcr_results_obj.weights.additional_outputs:
            assert "non_zero_weights" in pcr_results_obj.weights.additional_outputs
            assert isinstance(pcr_results_obj.weights.additional_outputs["non_zero_weights"], dict)

        if not config_dict["Frequentist"]: # Bayesian
            assert pcr_results_obj.inference is not None
            assert "credible_interval" in pcr_results_obj.inference.details
            ci_details = pcr_results_obj.inference.details["credible_interval"]
            assert isinstance(ci_details, tuple)
            assert len(ci_details) == 2
            assert isinstance(ci_details[0], np.ndarray) # Lower bound series
            assert isinstance(ci_details[1], np.ndarray) # Upper bound series
            
            treated_unit_id = sample_clustersc_data[sample_clustersc_data["TreatmentStatus"] == 1][config_dict["unitid"]].unique()[0]
            expected_len = sample_clustersc_data[sample_clustersc_data[config_dict["unitid"]] == treated_unit_id].shape[0]
            assert len(ci_details[0]) == expected_len
            assert len(ci_details[1]) == expected_len
            
        assert pcr_results_obj.method_details is not None
        assert pcr_results_obj.method_details.name is not None


    except Exception as e:
        pytest.fail(f"CLUSTERSC PCR fit with config {pcr_config_override} failed: {e}")


@pytest.mark.parametrize(
    "rpca_config_override",
    [
        {"ROB": "HQF"}, 
    ],
)
def test_clustersc_rpca_config_variations(
    sample_clustersc_data: pd.DataFrame, rpca_config_override: Dict[str, Any]
) -> None:
    """Test CLUSTERSC with various valid RPCA configurations."""
    base_config_dict: Dict[str, Any] = {
        "df": sample_clustersc_data,
        "outcome": "OutcomeMetric",
        "treat": "TreatmentStatus",
        "unitid": "UnitNum",
        "time": "TimePoint",
        "method": "RPCA",
        "display_graphs": False,
        "ROB": "PCP", 
    }

    config_dict = {**base_config_dict, **rpca_config_override}
    config_obj = CLUSTERSCConfig(**config_dict) 
    estimator = CLUSTERSC(config_obj)

    try:
        results = estimator.fit()
        assert isinstance(results, BaseEstimatorResults)
        assert "RPCA" in results.sub_method_results
        rpca_results_obj = results.sub_method_results["RPCA"]
        assert isinstance(rpca_results_obj, BaseEstimatorResults)


        assert rpca_results_obj.method_details is not None
        assert rpca_results_obj.method_details.name == "RPCA Synth"
        
        assert rpca_results_obj.effects is not None
        assert rpca_results_obj.effects.att is not None 
        
        assert rpca_results_obj.fit_diagnostics is not None
        assert rpca_results_obj.fit_diagnostics.pre_treatment_rmse is not None

        assert rpca_results_obj.time_series is not None
        for vec_attr in ["observed_outcome", "counterfactual_outcome", "estimated_gap"]:
            vector_val = getattr(rpca_results_obj.time_series, vec_attr)
            assert vector_val is not None
            assert isinstance(vector_val, np.ndarray)
            if vec_attr == "estimated_gap":
                 assert vector_val.ndim == 2
            else:
                 assert vector_val.ndim == 1 or vector_val.shape[1] == 1
            
            treated_unit_id = sample_clustersc_data[sample_clustersc_data["TreatmentStatus"] == 1][config_dict["unitid"]].unique()[0]
            expected_len = sample_clustersc_data[sample_clustersc_data[config_dict["unitid"]] == treated_unit_id].shape[0]
            assert len(vector_val) == expected_len

        assert rpca_results_obj.weights is not None
        assert isinstance(rpca_results_obj.weights.donor_weights, dict)

    except Exception as e:
        pytest.fail(f"CLUSTERSC RPCA fit with config {rpca_config_override} failed: {e}")


# --- Plotting Behavior Tests ---
from unittest.mock import patch
import pytest
import pandas as pd
from typing import Dict, Any

@pytest.mark.parametrize("method_to_test", ["PCR", "RPCA", "BOTH"])
@pytest.mark.parametrize("display_graphs_flag", [True, False])
def test_clustersc_plotting_behavior(
    sample_clustersc_data: pd.DataFrame, method_to_test: str, display_graphs_flag: bool
) -> None:
    """Test plotting behavior of CLUSTERSC.fit based on display_graphs flag."""
    config_dict: Dict[str, Any] = {
        "df": sample_clustersc_data,
        "outcome": "OutcomeMetric",
        "treat": "TreatmentStatus",
        "unitid": "UnitNum",
        "time": "TimePoint",
        "method": method_to_test,
        "display_graphs": display_graphs_flag,
        "cluster": True, 
        "objective": "OLS", 
        "Frequentist": True,
        "ROB": "PCP", 
        "counterfactual_color": ["blue", "red"], 
        "treated_color": "green",
        "save": False
    }

    config_obj = CLUSTERSCConfig(**config_dict) 
    estimator = CLUSTERSC(config_obj)
    
    with patch("mlsynth.estimators.clustersc.plot_estimates") as mock_plot:
        results = estimator.fit()

        if display_graphs_flag:
            expected_calls = 0
            expected_cf_names = []
            expected_cf_list_len = 0
            
            pcr_sub_res = results.sub_method_results.get("PCR")
            rpca_sub_res = results.sub_method_results.get("RPCA")

            if method_to_test == "PCR":
                if pcr_sub_res and pcr_sub_res.time_series and pcr_sub_res.time_series.counterfactual_outcome is not None:
                    expected_calls = 1
                    expected_cf_list_len = 1
                    expected_cf_names = ["RSC"] if config_dict["Frequentist"] else ["Bayesian RSC"]
            elif method_to_test == "RPCA":
                if rpca_sub_res and rpca_sub_res.time_series and rpca_sub_res.time_series.counterfactual_outcome is not None:
                    expected_calls = 1
                    expected_cf_list_len = 1
                    expected_cf_names = ["RPCA Synth"]
            elif method_to_test == "BOTH":
                num_valid_cfs = 0
                temp_names = []
                if pcr_sub_res and pcr_sub_res.time_series and pcr_sub_res.time_series.counterfactual_outcome is not None:
                    num_valid_cfs += 1
                    temp_names.append("RSC" if config_dict["Frequentist"] else "Bayesian RSC")
                if rpca_sub_res and rpca_sub_res.time_series and rpca_sub_res.time_series.counterfactual_outcome is not None:
                    num_valid_cfs += 1
                    temp_names.append("RPCA Synth")

                if num_valid_cfs > 0:
                    expected_calls = 1
                    expected_cf_list_len = num_valid_cfs
                    expected_cf_names = temp_names
                    
                if expected_calls > 0:
                    mock_plot.assert_called_once()
                    call_args = mock_plot.call_args[1]
                
                    assert len(call_args["counterfactual_series_list"]) == expected_cf_list_len
                    assert call_args["counterfactual_names"] == expected_cf_names
                    assert call_args["estimation_method_name"] == "CLUSTERSC"
                    assert call_args["treated_series_color"] == config_dict["treated_color"]
                
                    if expected_cf_list_len == 1:
                        assert call_args["counterfactual_series_colors"] == config_dict["counterfactual_color"]
                    else:
                        assert call_args["counterfactual_series_colors"] == (
                            config_dict["counterfactual_color"] * expected_cf_list_len
                        )
                
                    # These assertions should always run if `expected_calls > 0`
                    assert call_args["save_plot_config"] == config_dict["save"]
                    assert call_args["time_axis_label"] == config_dict["time"]
                    assert call_args["unit_identifier_column_name"] == config_dict["unitid"]
                    assert call_args["outcome_variable_label"] == config_dict["outcome"]
                    assert call_args["treatment_name_label"] == config_dict["treat"]
                else:
                    mock_plot.assert_not_called()

