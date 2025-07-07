import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pydantic import ValidationError

from mlsynth import SI
from mlsynth.config_models import (
    SIConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
    InferenceResults, # Though SI might not populate this much
    MethodDetailsResults,
)
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError

# Full configuration dictionary used in tests.
SI_FULL_TEST_CONFIG_BASE: Dict[str, Any] = {
    "outcome": "Y_main",
    "treat": "focal_treatment",
    "unitid": "unit_id",
    "time": "time_id",
    "inters": ["inter_A", "inter_B"],
    "display_graphs": False,
    "save": False,
    "counterfactual_color": ["teal"],
    "treated_color": "maroon",
    "seed": 98765, # Not part of SIConfig
    "verbose": False, # Not part of SIConfig
}

# Fields that are part of SIConfig (BaseEstimatorConfig + SI specific)
SI_PYDANTIC_MODEL_FIELDS = [
    "df", "outcome", "treat", "unitid", "time",
    "display_graphs", "save", "counterfactual_color", "treated_color",
    "inters"
]

def _get_pydantic_config_dict_si(full_config: Dict[str, Any], df_fixture: pd.DataFrame) -> Dict[str, Any]:
    """Helper to extract Pydantic-valid fields for SIConfig and add the DataFrame."""
    pydantic_dict = {
        k: v for k, v in full_config.items() if k in SI_PYDANTIC_MODEL_FIELDS
    }
    pydantic_dict["df"] = df_fixture
    return pydantic_dict

@pytest.fixture
def si_panel_data():
    """Provides a panel dataset for SI smoke testing."""
    data_dict = {
        'unit_id': [1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4, 5,5,5,5,5], # 5 units, 5 periods
        'time_id': [1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5],
        'Y_main':  [10,11,15,16,17, 9,10,11,12,13, 12,13,14,15,16, 11,12,13,14,15, 13,14,15,16,17],
        'X1':      [5,6,7,8,9,   4,5,6,7,8,   6,7,8,9,10,  5,6,7,8,9,   7,8,9,10,11],
    }
    df = pd.DataFrame(data_dict)

    # Focal treatment for unit 1, starting at time_id = 4
    df[SI_FULL_TEST_CONFIG_BASE["treat"]] = 0
    df.loc[(df['unit_id'] == 1) & (df['time_id'] >= 4), SI_FULL_TEST_CONFIG_BASE["treat"]] = 1

    # Alternative interventions
    # inter_A: units 2, 3 treated at time_id = 3
    df["inter_A"] = 0
    df.loc[(df['unit_id'].isin([2,3])) & (df['time_id'] >= 3), "inter_A"] = 1

    # inter_B: unit 4 treated at time_id = 2
    df["inter_B"] = 0
    df.loc[(df['unit_id'] == 4) & (df['time_id'] >= 2), "inter_B"] = 1

    return df

def test_si_creation(si_panel_data: pd.DataFrame):
    """Test that the SI estimator can be instantiated."""
    pydantic_dict = _get_pydantic_config_dict_si(SI_FULL_TEST_CONFIG_BASE, si_panel_data)

    try:
        config_obj = SIConfig(**pydantic_dict)
        estimator = SI(config=config_obj)
        assert estimator is not None, "SI estimator should be created."
        assert estimator.outcome == "Y_main"
        assert estimator.treat == SI_FULL_TEST_CONFIG_BASE["treat"]
        assert estimator.inters == ["inter_A", "inter_B"]
        assert not estimator.display_graphs
    except Exception as e:
        pytest.fail(f"SI instantiation failed: {e}")

def test_si_fit_smoke(si_panel_data: pd.DataFrame):
    """Smoke test for the SI fit method."""
    pydantic_dict = _get_pydantic_config_dict_si(SI_FULL_TEST_CONFIG_BASE, si_panel_data)
    config_obj = SIConfig(**pydantic_dict)
    estimator = SI(config=config_obj)

    try:
        results_dict = estimator.fit()
        assert results_dict is not None, "Fit method should return results."
        assert isinstance(results_dict, dict), "Results should be a dictionary."

        for inter_name in SI_FULL_TEST_CONFIG_BASE["inters"]:
            assert inter_name in results_dict, f"Results for intervention '{inter_name}' should be present."
            inter_results: BaseEstimatorResults = results_dict[inter_name]
            assert isinstance(inter_results, BaseEstimatorResults)

            if inter_results.execution_summary and "error" in inter_results.execution_summary:
                print(f"Warning for {inter_name}: {inter_results.execution_summary['error']}")
                continue

            assert inter_results.effects is not None
            assert inter_results.fit_diagnostics is not None
            assert inter_results.time_series is not None
            assert inter_results.weights is not None
            assert inter_results.method_details is not None

            assert isinstance(inter_results.effects.att, (float, np.floating, type(None)))
            assert isinstance(inter_results.fit_diagnostics.pre_treatment_rmse, (float, np.floating, type(None)))
            assert isinstance(inter_results.time_series.counterfactual_outcome, (np.ndarray, type(None)))
            assert isinstance(inter_results.weights.donor_weights, (dict, type(None)))

    except Exception as e:
        pytest.fail(f"SI fit method failed during smoke test: {e}")

# --- Input Validation Tests ---

@pytest.mark.parametrize("invalid_inters", [
    None, # Missing
    "not_a_list",
    [],
])
def test_si_invalid_inters_config(si_panel_data: pd.DataFrame, invalid_inters: Any):
    """Test SIConfig instantiation fails if 'inters' is invalid or missing."""
    full_config_dict = SI_FULL_TEST_CONFIG_BASE.copy()
    if invalid_inters is None: # Simulate missing 'inters' key
        if "inters" in full_config_dict:
            del full_config_dict["inters"]
    else: # Set 'inters' to the invalid value
        full_config_dict["inters"] = invalid_inters

    pydantic_dict = _get_pydantic_config_dict_si(full_config_dict, si_panel_data)
    # Adjust pydantic_dict based on how _get_pydantic_config_dict_si handles missing keys
    if invalid_inters is None and "inters" in pydantic_dict:
         del pydantic_dict["inters"] # Ensure 'inters' is truly missing for pydantic validation
    elif invalid_inters is not None:
         pydantic_dict["inters"] = invalid_inters


    # SIConfig has min_length=1 for 'inters'.
    # So, if 'inters' is missing, not a list, or an empty list, SIConfig should raise ValidationError.
    with pytest.raises(ValidationError):
        SIConfig(**pydantic_dict)


def test_si_inters_column_not_in_df(si_panel_data: pd.DataFrame):
    """Test SI instantiation fails if an intervention column in 'inters' is not in df."""
    full_config_dict = SI_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict["inters"] = ["inter_A", "NonExistentInter"]
    pydantic_dict = _get_pydantic_config_dict_si(full_config_dict, si_panel_data)

    config_obj = SIConfig(**pydantic_dict)
    # Changed from ValueError to MlsynthConfigError
    with pytest.raises(MlsynthConfigError, match="Intervention column 'NonExistentInter' specified in 'inters' not found in dataframe."):
        SI(config=config_obj)

@pytest.mark.parametrize("missing_key", ["df", "outcome", "treat", "unitid", "time", "inters"])
def test_si_missing_essential_config(si_panel_data: pd.DataFrame, missing_key: str):
    """Test SIConfig instantiation fails if essential config keys are missing."""
    full_config_dict = SI_FULL_TEST_CONFIG_BASE.copy()
    original_value = full_config_dict.pop(missing_key, None)

    pydantic_dict = _get_pydantic_config_dict_si(full_config_dict, si_panel_data)
    if missing_key in pydantic_dict:
        del pydantic_dict[missing_key]

    with pytest.raises(ValidationError):
        SIConfig(**pydantic_dict)

    if original_value is not None:
        full_config_dict[missing_key] = original_value


@pytest.mark.parametrize("col_key, wrong_col_name", [
    ("outcome", "NonExistent_Y_main"),
    ("treat", "NonExistent_Focal_Treat"),
    ("unitid", "NonExistent_Unit"),
    ("time", "NonExistent_Time"),
])
def test_si_core_column_not_in_df(si_panel_data: pd.DataFrame, col_key: str, wrong_col_name: str):
    """Test SI fit fails if core specified columns are not in df."""
    full_config_dict = SI_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict[col_key] = wrong_col_name

    pydantic_dict = _get_pydantic_config_dict_si(full_config_dict, si_panel_data)
    
    # BaseEstimatorConfig (parent of SIConfig) validates these columns upon instantiation
    # and directly raises MlsynthDataError.
    with pytest.raises(MlsynthDataError, match=f"Missing required columns in DataFrame 'df': {wrong_col_name}"):
        SIConfig(**pydantic_dict)

# --- Edge Case Tests ---

def test_si_no_common_donors_for_an_intervention(si_panel_data: pd.DataFrame):
    """Test SI handles an intervention with no common donors."""
    full_config_dict = SI_FULL_TEST_CONFIG_BASE.copy()
    df_mod = si_panel_data.copy()

    df_mod["inter_C_no_donors"] = 0
    full_config_dict["inters"] = ["inter_A", "inter_C_no_donors"]

    pydantic_dict = _get_pydantic_config_dict_si(full_config_dict, df_mod)
    config_obj = SIConfig(**pydantic_dict)
    estimator = SI(config=config_obj)
    results_dict = estimator.fit()

    assert "inter_C_no_donors" in results_dict
    inter_c_results: BaseEstimatorResults = results_dict["inter_C_no_donors"]
    assert isinstance(inter_c_results, BaseEstimatorResults)
    assert inter_c_results.execution_summary is not None
    assert inter_c_results.execution_summary.get("error") == "No common donors found"

    assert "inter_A" in results_dict
    inter_a_results: BaseEstimatorResults = results_dict["inter_A"]
    assert isinstance(inter_a_results, BaseEstimatorResults)
    assert inter_a_results.execution_summary is None or "error" not in inter_a_results.execution_summary


@pytest.fixture
def si_single_common_donor_data(si_panel_data):
    df = si_panel_data.copy()
    df["inter_single_donor"] = 0
    df.loc[(df['unit_id'] == 2) & (df['time_id'] >= 1), "inter_single_donor"] = 1
    return df

def test_si_single_common_donor(si_single_common_donor_data: pd.DataFrame):
    full_config_dict = SI_FULL_TEST_CONFIG_BASE.copy()
    full_config_dict["inters"] = ["inter_single_donor"]

    pydantic_dict = _get_pydantic_config_dict_si(full_config_dict, si_single_common_donor_data)
    config_obj = SIConfig(**pydantic_dict)
    estimator = SI(config=config_obj)
    results_dict = estimator.fit()

    assert "inter_single_donor" in results_dict
    inter_res: BaseEstimatorResults = results_dict["inter_single_donor"]
    assert isinstance(inter_res, BaseEstimatorResults)
    assert inter_res.execution_summary is None or "error" not in inter_res.execution_summary, \
        f"Fit failed for single common donor: {inter_res.execution_summary.get('error') if inter_res.execution_summary else 'No error info'}"
    assert inter_res.weights is not None
    assert inter_res.weights.donor_weights is not None
    assert len(inter_res.weights.donor_weights) == 1
    assert str(2) in inter_res.weights.donor_weights

@pytest.fixture
def si_few_pre_periods_data():
    data_dict = {
        'unit_id': [1,1,1, 2,2,2, 3,3,3, 4,4,4],
        'time_id': [1,2,3, 1,2,3, 1,2,3, 1,2,3],
        'Y_main':  [10,16,17, 9,10,11, 12,13,14, 11,12,13],
        'X1':      [5,8,9,   4,5,6,   6,7,8,   5,6,7],
    }
    df = pd.DataFrame(data_dict)
    df[SI_FULL_TEST_CONFIG_BASE["treat"]] = 0
    df.loc[(df['unit_id'] == 1) & (df['time_id'] >= 2), SI_FULL_TEST_CONFIG_BASE["treat"]] = 1
    df["inter_A"] = 0
    df.loc[(df['unit_id'].isin([2,3])) & (df['time_id'] >= 2), "inter_A"] = 1
    df["inter_B"] = 0
    df.loc[(df['unit_id'] == 4) & (df['time_id'] >= 1), "inter_B"] = 1
    return df

def test_si_few_pre_periods(si_few_pre_periods_data: pd.DataFrame):
    pydantic_dict = _get_pydantic_config_dict_si(SI_FULL_TEST_CONFIG_BASE, si_few_pre_periods_data)
    config_obj = SIConfig(**pydantic_dict)
    estimator = SI(config=config_obj)
    results_dict = estimator.fit()
    assert results_dict is not None
    assert "inter_A" in results_dict
    inter_a_res: BaseEstimatorResults = results_dict["inter_A"]
    assert isinstance(inter_a_res, BaseEstimatorResults)
    assert inter_a_res.execution_summary is not None
    assert "error" in inter_a_res.execution_summary
    # Check for the specific error related to insufficient pre-periods for PCR
    # This error originates from estutils.Opt.SCopt, wrapped by SI.fit
    assert "PCR requires at least 2 pre-treatment periods" in inter_a_res.execution_summary["error"] or \
           "Donor matrix Y0 has no rows" in inter_results.execution_summary["error"] # Alternative from SCopt

    # Check inter_B as well, which might have enough pre-periods depending on its own treatment time
    assert "inter_B" in results_dict
    inter_b_res: BaseEstimatorResults = results_dict["inter_B"]
    assert isinstance(inter_b_res, BaseEstimatorResults)
    # For inter_B, unit 4 is treated at time_id=1. Focal unit 1 is treated at time_id=2.
    # So, for inter_B, focal unit 1 has 1 pre-period (time_id=1). This should also error.
    assert inter_b_res.execution_summary is not None
    assert "error" in inter_b_res.execution_summary
    assert "PCR requires at least 2 pre-treatment periods" in inter_b_res.execution_summary["error"] or \
           "Donor matrix Y0 has no rows" in inter_results.execution_summary["error"]


@pytest.fixture
def si_no_pre_periods_data(si_few_pre_periods_data: pd.DataFrame):
    df = si_few_pre_periods_data.copy()
    df[SI_FULL_TEST_CONFIG_BASE["treat"]] = 0
    df.loc[(df['unit_id'] == 1) & (df['time_id'] >= 1), SI_FULL_TEST_CONFIG_BASE["treat"]] = 1
    return df

def test_si_no_pre_periods(si_no_pre_periods_data: pd.DataFrame):
    pydantic_dict = _get_pydantic_config_dict_si(SI_FULL_TEST_CONFIG_BASE, si_no_pre_periods_data)
    config_obj = SIConfig(**pydantic_dict)
    estimator = SI(config=config_obj)
    # This scenario (0 pre-periods) should be caught by dataprep or pcr (via Opt.SCopt)
    # Opt.SCopt raises MlsynthDataError: "Donor matrix Y0 has no rows (no pre-treatment periods)."
    # The fit method in SI.py now catches MlsynthDataError during setup and populates error results.
    # So, fit() itself should not raise, but return results with error messages.
    
    results_dict = estimator.fit()
    assert results_dict is not None
    for inter_name in SI_FULL_TEST_CONFIG_BASE["inters"]:
        assert inter_name in results_dict
        inter_results: BaseEstimatorResults = results_dict[inter_name]
        assert inter_results.execution_summary is not None
        assert "error" in inter_results.execution_summary
        # The exact error message might come from dataprep (if it fails to define periods)
        # or from pcr/SCopt if dataprep produces 0 pre-periods.
        # Let's check for a part of the expected MlsynthDataError message.
        assert "SI fit failed during initial data/config setup" in inter_results.execution_summary["error"] or \
               "no pre-treatment periods" in inter_results.execution_summary["error"].lower() or \
               "key 'pre_periods' is missing" in inter_results.execution_summary["error"].lower()

# --- Detailed Results Validation ---

def check_si_intervention_results_structure(
    inter_results: BaseEstimatorResults,
    inter_name: str,
    panel_data: pd.DataFrame,
    config_dict: dict
):
    """Helper to check structure of results for a single intervention in SI."""
    assert isinstance(inter_results, BaseEstimatorResults)

    if inter_results.execution_summary and "error" in inter_results.execution_summary:
        return # Skip detailed checks if there was an error during fit for this intervention

    # Method Details
    assert inter_results.method_details is not None
    assert inter_results.method_details.name == f"SI_for_{inter_name}"
    assert inter_results.method_details.parameters_used is not None

    # Effects
    assert inter_results.effects is not None
    assert inter_results.effects.att is not None
    assert inter_results.effects.att_percent is not None
    assert inter_results.effects.additional_effects is not None
    expected_effects_keys = ["SATT", "TTE", "ATT_Time", "PercentATT_Time", "SATT_Time"]
    for key in expected_effects_keys:
        assert key in inter_results.effects.additional_effects

    # Fit Diagnostics
    assert inter_results.fit_diagnostics is not None
    assert inter_results.fit_diagnostics.pre_treatment_rmse is not None
    assert inter_results.fit_diagnostics.pre_treatment_r_squared is not None
    assert inter_results.fit_diagnostics.additional_metrics is not None
    expected_fit_keys = ["T1 RMSE", "Pre-Periods", "Post-Periods"]
    for key in expected_fit_keys:
        assert key in inter_results.fit_diagnostics.additional_metrics

    # Time Series
    assert inter_results.time_series is not None
    ts_attrs = ["observed_outcome", "counterfactual_outcome", "estimated_gap", "time_periods"]
    for attr in ts_attrs:
        assert getattr(inter_results.time_series, attr) is not None
        assert isinstance(getattr(inter_results.time_series, attr), np.ndarray)

    # Weights
    assert inter_results.weights is not None
    assert isinstance(inter_results.weights.donor_weights, dict)
    assert all(isinstance(k, str) for k in inter_results.weights.donor_weights.keys())
    assert all(isinstance(v, (float, np.floating)) for v in inter_results.weights.donor_weights.values())

    # Inference (SI via PCR doesn't add much here by default)
    assert inter_results.inference is not None
    assert inter_results.inference.method == "PCR-based point estimate"


    # Check vector lengths (focal unit perspective)
    focal_unit_id_val = None
    # Find the focal unit ID from the 'treat' column in the original df from config
    # This is a bit indirect; ideally dataprep would return focal_unit_id more directly
    # or it would be part of self.config if it's a single unit.
    # For now, assuming self.treat identifies a single unit over time.
    focal_unit_ids = panel_data[panel_data[config_dict["treat"]] == 1][config_dict["unitid"]].unique()
    if len(focal_unit_ids) > 0:
        focal_unit_id_val = focal_unit_ids[0]

    if focal_unit_id_val is not None:
        num_time_periods_focal = panel_data[panel_data[config_dict["unitid"]] == focal_unit_id_val][config_dict["time"]].nunique()
        num_post_periods_focal = panel_data[
            (panel_data[config_dict["unitid"]] == focal_unit_id_val) & (panel_data[config_dict["treat"]] == 1)
        ][config_dict["time"]].nunique()

        assert len(inter_results.time_series.observed_outcome) == num_time_periods_focal
        assert len(inter_results.time_series.counterfactual_outcome) == num_time_periods_focal
        assert len(inter_results.time_series.estimated_gap) == num_time_periods_focal
        assert len(inter_results.time_series.time_periods) == num_time_periods_focal
        
        att_time = inter_results.effects.additional_effects.get("ATT_Time")
        assert att_time is not None
        assert len(att_time) == num_post_periods_focal


def test_si_fit_results_structure(si_panel_data: pd.DataFrame):
    """Test the overall structure and types of the results dictionary from SI.fit()."""
    pydantic_dict = _get_pydantic_config_dict_si(SI_FULL_TEST_CONFIG_BASE, si_panel_data)
    config_obj = SIConfig(**pydantic_dict)
    estimator = SI(config=config_obj)
    results_dict = estimator.fit()

    assert isinstance(results_dict, dict)
    # Ensure all requested interventions are keys in the results
    # Some might have errors, but the key should be there.
    assert set(results_dict.keys()) == set(pydantic_dict["inters"]), \
        f"Result keys {set(results_dict.keys())} do not match intervention names {set(pydantic_dict['inters'])}"

    for inter_name, inter_results_obj in results_dict.items():
        assert isinstance(inter_results_obj, BaseEstimatorResults)
        check_si_intervention_results_structure(inter_results_obj, inter_name, si_panel_data, pydantic_dict)

# TODO: Add tests for specific weight values or effect sizes if known/stable for PCR.
