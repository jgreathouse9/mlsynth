import pytest
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional
from unittest.mock import MagicMock, patch

# Import SCM classes and utility functions from mlsynth
from mlsynth import CLUSTERSC
from mlsynth.config_models import CLUSTERSCConfig # Import Pydantic configs
from mlsynth.utils import spillover # Import the module for patch.object
from mlsynth.utils.spillover import _get_data, _estimate_counterfactual, iterative_scm
from mlsynth.utils.datautils import dataprep # For creating test data structures
from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError, MlsynthEstimationError


@pytest.fixture
def sample_spillover_df() -> pd.DataFrame:
    """Creates a sample DataFrame for spillover tests."""
    data = {
        "unit": ["T1", "C1", "C2", "S1", "S2"] * 10,
        "time": np.repeat(np.arange(1, 11), 5),
        "outcome": np.random.rand(50) * 10,
        "treated": [1 if unit == "T1" and time >= 6 else 0 for unit, time in zip(["T1", "C1", "C2", "S1", "S2"] * 10, np.repeat(np.arange(1, 11), 5))]
    }
    df = pd.DataFrame(data)
    # Ensure 'treated' column is integer type as expected by some SCMs or dataprep
    df['treated'] = df['treated'].astype(int)
    return df

@pytest.fixture
def base_scm_config(sample_spillover_df: pd.DataFrame) -> Dict[str, Any]:
    """Base configuration for SCM models."""
    return {
        "df": sample_spillover_df,
        "unitid": "unit",
        "time": "time",
        "outcome": "outcome",
        "treat": "treated",
        "display_graphs": False, # Keep tests quiet
        "save": False,
    }

@pytest.fixture
def sample_clustersc_scm(base_scm_config: Dict[str, Any]) -> CLUSTERSC:
    """Fixture for an initialized CLUSTERSC instance."""
    config_dict = {**base_scm_config, "method": "PCR", "Frequentist": True}
    pydantic_config = CLUSTERSCConfig(**config_dict)
    return CLUSTERSC(config=pydantic_config)

# Tests for _get_data
@pytest.mark.parametrize(
    "scm_fixture_name",
    ["sample_clustersc_scm"]
)
def test_get_data_smoke(scm_fixture_name: str, request: pytest.FixtureRequest):
    """Smoke test for _get_data with different SCM types."""
    scm_instance = request.getfixturevalue(scm_fixture_name)
    data_dict = _get_data(scm_instance)

    expected_keys = ["donor_matrix", "y", "pre_periods", "post_periods", "donor_names", "T0"]
    for key in expected_keys:
        assert key in data_dict

    assert isinstance(data_dict["donor_matrix"], np.ndarray)
    assert isinstance(data_dict["y"], np.ndarray)
    assert isinstance(data_dict["pre_periods"], int)
    assert isinstance(data_dict["post_periods"], int)
    assert isinstance(data_dict["donor_names"], list)
    assert isinstance(data_dict["T0"], int)

    assert data_dict["donor_matrix"].ndim == 2
    assert data_dict["y"].ndim == 1
    assert len(data_dict["donor_names"]) == data_dict["donor_matrix"].shape[1] # 4 control units: C1, C2, S1, S2
    assert data_dict["pre_periods"] == 5 # Treated from period 6, so 5 pre-periods (1,2,3,4,5)
    assert data_dict["post_periods"] == 5 # Periods 6,7,8,9,10
    assert data_dict["T0"] == data_dict["pre_periods"]


def test_get_data_raises_error_for_cohorts(sample_clustersc_scm: CLUSTERSC):
    """Test _get_data raises MlsynthDataError if dataprep returns cohort data."""
    with patch('mlsynth.utils.spillover.dataprep') as mock_dataprep:
        # Simulate dataprep output for a cohort scenario
        mock_dataprep.return_value = {"cohorts": {1: "data"}, "some_other_key": "value"}
        with pytest.raises(MlsynthDataError, match="iterative_scm supports only single treated unit case, but cohort data was detected."):
            _get_data(sample_clustersc_scm)


def test_get_data_raises_error_on_dataprep_failure(sample_clustersc_scm: CLUSTERSC):
    """Test _get_data raises MlsynthDataError if dataprep itself fails."""
    with patch('mlsynth.utils.spillover.dataprep') as mock_dataprep:
        mock_dataprep.side_effect = Exception("Dataprep internal error")
        with pytest.raises(MlsynthDataError, match="Failed to extract data from SCM via dataprep: Dataprep internal error"):
            _get_data(sample_clustersc_scm)

def test_get_data_missing_scm_attrs(sample_clustersc_scm: CLUSTERSC):
    """Test _get_data raises MlsynthConfigError for missing SCM attributes."""
    scm_bad_attr = MagicMock(spec=CLUSTERSC)
    del scm_bad_attr.df # Remove a required attribute
    with pytest.raises(MlsynthConfigError, match="SCM instance is missing required attribute 'df'"):
        _get_data(scm_bad_attr)

    scm_df_not_dataframe = MagicMock(spec=CLUSTERSC)
    scm_df_not_dataframe.df = "not a dataframe" # df is not a DataFrame
    # Add other required attributes to avoid failing earlier
    scm_df_not_dataframe.unitid = "unit"
    scm_df_not_dataframe.time = "time"
    scm_df_not_dataframe.outcome = "outcome"
    scm_df_not_dataframe.treat = "treated"
    scm_df_not_dataframe.config = {} # Dummy config
    with pytest.raises(MlsynthDataError, match="SCM attribute 'df' must be a pandas DataFrame."):
        _get_data(scm_df_not_dataframe)


@pytest.fixture
def sample_counterfactual_inputs(sample_clustersc_scm: CLUSTERSC) -> Dict[str, Any]:
    """Provides sample inputs for _estimate_counterfactual tests."""
    # Use _get_data to get a realistic data structure
    data_dict = _get_data(sample_clustersc_scm)
    T = data_dict["y"].shape[0] # Total time periods
    
    # Assume 'S1' (index depends on donor_names ordering from dataprep) is the target spillover unit
    # and C1, C2 are clean donors.
    # donor_names from sample_df via dataprep: ['C1', 'C2', 'S1', 'S2'] (order might vary)
    # Let's assume S1 is at index 2, C1 at 0, C2 at 1 for this example.
    # This part is a bit fragile if donor_names order changes.
    # A more robust fixture would explicitly set up donor_names and indices.
    # For now, let's assume a fixed order for simplicity of the fixture.
    # Actual donor_names: data_dict["donor_names"] should now be a list ['C1', 'C2', 'S1', 'S2']
    # due to the fix in _get_data
    
    donor_names_list = data_dict["donor_names"] # This is now a list
    target_spillover_idx_in_donors = donor_names_list.index("S1")
    
    clean_donor_indices = [
        i for i, name in enumerate(donor_names_list) 
        if name in ["C1", "C2"]
    ]

    return {
        "X_donors": data_dict["donor_matrix"][:, clean_donor_indices], # Clean donors' outcomes
        "Y_target": data_dict["donor_matrix"][:, target_spillover_idx_in_donors], # Spillover unit's outcome
        "donor_names_subset": [donor_names_list[i] for i in clean_donor_indices], # Names of clean donors
        "pre_periods": data_dict["pre_periods"],
        "idx": target_spillover_idx_in_donors, # Original index of the spillover unit being processed
        "spillover_indices": [donor_names_list.index("S1"), donor_names_list.index("S2")] # All spillover units
    }

# Tests for _estimate_counterfactual
@patch('mlsynth.utils.spillover.pcr')
def test_estimate_counterfactual_clustersc_pcr(mock_pcr: MagicMock, sample_clustersc_scm: CLUSTERSC, sample_counterfactual_inputs: Dict[str, Any]):
    """Test _estimate_counterfactual with CLUSTERSC (PCR method)."""
    mock_pcr.return_value = {"cf_mean": np.random.rand(sample_counterfactual_inputs["Y_target"].shape[0])}
    sample_clustersc_scm.method = "PCR" # Ensure method is PCR

    cf = _estimate_counterfactual(
        scm=sample_clustersc_scm,
        donor_outcomes_for_cf_estimation=sample_counterfactual_inputs["X_donors"],
        target_spillover_donor_outcome=sample_counterfactual_inputs["Y_target"],
        subset_donor_identifiers=sample_counterfactual_inputs["donor_names_subset"],
        num_pre_treatment_periods=sample_counterfactual_inputs["pre_periods"],
        spillover_donor_original_index=sample_counterfactual_inputs["idx"],
        all_spillover_donor_original_indices=sample_counterfactual_inputs["spillover_indices"],
        method="PCR"
    )
    mock_pcr.assert_called_once()
    assert isinstance(cf, np.ndarray)
    assert cf.shape == sample_counterfactual_inputs["Y_target"].shape



@patch('mlsynth.utils.spillover.RPCASYNTH')
@patch('mlsynth.utils.spillover.dataprep')  # RPCASYNTH path calls dataprep internally
def test_estimate_counterfactual_clustersc_rpca(
    mock_dataprep: MagicMock,
    mock_rpca: MagicMock,
    sample_clustersc_scm,
    sample_counterfactual_inputs: Dict[str, Any]
):
    """Test _estimate_counterfactual with CLUSTERSC (RPCA method)."""

    # Setup
    T_total_mock = sample_counterfactual_inputs["Y_target"].shape[0]
    pre_periods_mock = sample_counterfactual_inputs["pre_periods"]
    donor_names_mock = sample_counterfactual_inputs["donor_names_subset"]

    # Mocked dataprep outputs for both unmodified and modified dataframes
    mock_prepped_data = {
        "donor_matrix": np.random.rand(T_total_mock, len(donor_names_mock)),
        "y": np.random.rand(T_total_mock),
        "treated_unit_name": donor_names_mock[0],
        "pre_periods": pre_periods_mock,
        "post_periods": T_total_mock - pre_periods_mock,
        "total_periods": T_total_mock,
        "donor_names": pd.Index(donor_names_mock)
    }

    mock_dataprep.side_effect = [mock_prepped_data, mock_prepped_data]

    # Mock RPCASYNTH output
    mock_rpca.return_value = {
        "Vectors": {
            "Counterfactual": np.random.rand(T_total_mock)
        }
    }

    sample_clustersc_scm.method = "RPCA"

    # Run the function under test
    cf = _estimate_counterfactual(
        scm=sample_clustersc_scm,
        donor_outcomes_for_cf_estimation=sample_counterfactual_inputs["X_donors"],
        target_spillover_donor_outcome=sample_counterfactual_inputs["Y_target"],
        subset_donor_identifiers=sample_counterfactual_inputs["donor_names_subset"],
        num_pre_treatment_periods=sample_counterfactual_inputs["pre_periods"],
        spillover_donor_original_index=sample_counterfactual_inputs["idx"],
        all_spillover_donor_original_indices=sample_counterfactual_inputs["spillover_indices"],
        method="RPCA"
    )

    # Assertions
    mock_rpca.assert_called_once()
    assert isinstance(cf, np.ndarray)
    assert cf.shape == sample_counterfactual_inputs["Y_target"].shape




def test_estimate_counterfactual_invalid_method(sample_clustersc_scm: CLUSTERSC, sample_counterfactual_inputs: Dict[str, Any]):
    """Test _estimate_counterfactual raises MlsynthConfigError for invalid method."""
    with pytest.raises(MlsynthConfigError, match="method must be 'PCR', 'RPCA', or 'BOTH' for CLUSTERSC"):
        _estimate_counterfactual(
            scm=sample_clustersc_scm, 
            donor_outcomes_for_cf_estimation=sample_counterfactual_inputs["X_donors"],
            target_spillover_donor_outcome=sample_counterfactual_inputs["Y_target"],
            subset_donor_identifiers=sample_counterfactual_inputs["donor_names_subset"],
            num_pre_treatment_periods=sample_counterfactual_inputs["pre_periods"],
            spillover_donor_original_index=sample_counterfactual_inputs["idx"],
            all_spillover_donor_original_indices=sample_counterfactual_inputs["spillover_indices"],
            method="INVALID_METHOD"
        )

def test_estimate_counterfactual_invalid_input_types(sample_clustersc_scm: CLUSTERSC, sample_counterfactual_inputs: Dict[str, Any]):
    """Test _estimate_counterfactual with invalid input data types and shapes."""
    valid_args = {
        "scm": sample_clustersc_scm,
        "donor_outcomes_for_cf_estimation": sample_counterfactual_inputs["X_donors"],
        "target_spillover_donor_outcome": sample_counterfactual_inputs["Y_target"],
        "subset_donor_identifiers": sample_counterfactual_inputs["donor_names_subset"],
        "num_pre_treatment_periods": sample_counterfactual_inputs["pre_periods"],
        "spillover_donor_original_index": sample_counterfactual_inputs["idx"],
        "all_spillover_donor_original_indices": sample_counterfactual_inputs["spillover_indices"],
        "method": "PCR"
    }

    # Invalid donor_outcomes_for_cf_estimation (not ndarray)
    with pytest.raises(MlsynthDataError, match="donor_outcomes_for_cf_estimation must be a 2D NumPy array"):
        _estimate_counterfactual(**{**valid_args, "donor_outcomes_for_cf_estimation": [[1,2],[3,4]]}) # type: ignore
    # Invalid donor_outcomes_for_cf_estimation (not 2D)
    with pytest.raises(MlsynthDataError, match="donor_outcomes_for_cf_estimation must be a 2D NumPy array"):
        _estimate_counterfactual(**{**valid_args, "donor_outcomes_for_cf_estimation": np.array([1,2,3])})
    # Invalid target_spillover_donor_outcome (not 1D)
    with pytest.raises(MlsynthDataError, match="target_spillover_donor_outcome must be a 1D NumPy array"):
        _estimate_counterfactual(**{**valid_args, "target_spillover_donor_outcome": np.array([[1],[2]])})
    # Invalid subset_donor_identifiers (not list of str)
    with pytest.raises(MlsynthDataError, match="subset_donor_identifiers must be a list of strings"):
        _estimate_counterfactual(**{**valid_args, "subset_donor_identifiers": [1,2,3]}) # type: ignore
    # Invalid num_pre_treatment_periods (negative)
    with pytest.raises(MlsynthDataError, match="num_pre_treatment_periods must be a non-negative integer"):
        _estimate_counterfactual(**{**valid_args, "num_pre_treatment_periods": -1})
    # Shape mismatch: time dimension
    with pytest.raises(MlsynthDataError, match="Time dimension mismatch"):
        _estimate_counterfactual(**{**valid_args, "target_spillover_donor_outcome": sample_counterfactual_inputs["Y_target"][:-1]})
    # Shape mismatch: donor dimension
    with pytest.raises(MlsynthDataError, match="Number of donors in donor_outcomes_for_cf_estimation does not match"):
        _estimate_counterfactual(**{**valid_args, "subset_donor_identifiers": sample_counterfactual_inputs["donor_names_subset"][:-1]})


def test_estimate_counterfactual_unsupported_scm(sample_counterfactual_inputs: Dict[str, Any]):
    """Test _estimate_counterfactual raises NotImplementedError for unsupported SCM."""
    class UnsupportedSCM: # Dummy class
        pass
    
    with pytest.raises(NotImplementedError, match="Iterative SCM not implemented for UnsupportedSCM"):
        _estimate_counterfactual(
            scm=UnsupportedSCM(),
            donor_outcomes_for_cf_estimation=sample_counterfactual_inputs["X_donors"],
            target_spillover_donor_outcome=sample_counterfactual_inputs["Y_target"],
            subset_donor_identifiers=sample_counterfactual_inputs["donor_names_subset"],
            num_pre_treatment_periods=sample_counterfactual_inputs["pre_periods"],
            spillover_donor_original_index=sample_counterfactual_inputs["idx"],
            all_spillover_donor_original_indices=sample_counterfactual_inputs["spillover_indices"]
        )

# Tests for iterative_scm

# Parametrize over SCM types for iterative_scm tests
SCM_FIXTURE_NAMES = ["sample_clustersc_scm"]

@pytest.mark.parametrize("scm_fixture_name", SCM_FIXTURE_NAMES)
@patch('mlsynth.utils.spillover._estimate_counterfactual')
@patch('mlsynth.utils.spillover._get_data')
def test_iterative_scm_smoke(
    mock_get_data: MagicMock,
    mock_estimate_cf: MagicMock,
    scm_fixture_name: str,
    request: pytest.FixtureRequest,
    sample_spillover_df: pd.DataFrame # Used to construct mock SCM for fit call
):
    """Smoke test for iterative_scm with different SCM types."""
    scm_instance = request.getfixturevalue(scm_fixture_name)
    
    # Mock _get_data return value
    T = 10 # Total periods from sample_spillover_df
    mock_donor_names = ["C1", "C2", "S1", "S2"]
    mock_get_data.return_value = {
        "donor_matrix": np.random.rand(T, len(mock_donor_names)),
        "y": np.random.rand(T),
        "pre_periods": 5,
        "post_periods": 5,
        "donor_names": mock_donor_names,
        "T0": 5,
    }

    # Mock _estimate_counterfactual return value
    mock_estimate_cf.return_value = np.random.rand(T)

    # Mock the 'fit' method of the SCM class that will be instantiated inside iterative_scm
    # This requires knowing the type of scm_instance to patch its 'fit' method correctly.
    # We can create a mock SCM instance that iterative_scm will create and use.
    
    mock_returned_scm_instance = MagicMock(spec=type(scm_instance))
    mock_returned_scm_instance.fit.return_value = {"Effects": "mock_effects_data"}
    
    # Patch the SCM class constructor as it's looked up in the spillover module.
    # mocked_scm_class will replace the SCM class (e.g., CLUSTERSC) in spillover.py's namespace.
    # We then configure it so that when it's instantiated, it returns our mock_returned_scm_instance.
    patch_target_str = f"mlsynth.utils.spillover.{scm_instance.__class__.__name__}"
    with patch(patch_target_str) as mocked_scm_class:
        mocked_scm_class.return_value = mock_returned_scm_instance
        
        spillover_ids = ["S1", "S2"]
        results = iterative_scm(scm_instance, spillover_ids, method=getattr(scm_instance, 'method', None))

        mock_get_data.assert_called_once_with(scm_instance)
        # _estimate_counterfactual should be called for each spillover_id
        assert mock_estimate_cf.call_count == len(spillover_ids)
        
        # Check that the SCM class (now mocked_scm_class) was instantiated once.
        # The arguments to the constructor can be complex due to **scm.__dict__, so we might
        # not check them specifically unless needed, or use mock.ANY for the config dict.
        mocked_scm_class.assert_called_once() 
        
        # Check that the fit method of the instance returned by the SCM constructor was called
        mock_returned_scm_instance.fit.assert_called_once()
        
        assert results == {"Effects": "mock_effects_data"}


def test_iterative_scm_empty_spillover_ids(sample_clustersc_scm: CLUSTERSC):
    """Test iterative_scm raises ValueError for empty spillover_unit_ids."""
    with pytest.raises(MlsynthConfigError, match="spillover_unit_identifiers must be a non-empty list."):
        iterative_scm(sample_clustersc_scm, [])

def test_iterative_scm_invalid_spillover_ids_type(sample_clustersc_scm: CLUSTERSC):
    """Test iterative_scm with invalid type for spillover_unit_identifiers."""
    with pytest.raises(MlsynthConfigError, match="spillover_unit_identifiers must be a non-empty list"):
        iterative_scm(sample_clustersc_scm, "not_a_list") # type: ignore
    with pytest.raises(MlsynthConfigError, match="All elements in spillover_unit_identifiers must be strings"):
        iterative_scm(sample_clustersc_scm, ["S1", 123]) # type: ignore

@patch('mlsynth.utils.spillover._get_data')
def test_iterative_scm_invalid_spillover_id_not_found(mock_get_data: MagicMock, sample_clustersc_scm: CLUSTERSC):
    """Test iterative_scm raises MlsynthConfigError for spillover_unit_id not in donor_names."""
    mock_get_data.return_value = {
        "donor_matrix": np.random.rand(10, 3), "y": np.random.rand(10),
        "pre_periods": 5, "post_periods": 5, "donor_names": ["C1", "C2", "C3"], "T0": 5,
    }
    with pytest.raises(MlsynthConfigError, match="Spillover unit ID 'InvalidID' not found in donor names"):
        iterative_scm(sample_clustersc_scm, ["InvalidID"])

@patch('mlsynth.utils.spillover._get_data')
def test_iterative_scm_non_unique_spillover_ids(mock_get_data: MagicMock, sample_clustersc_scm: CLUSTERSC):
    """Test iterative_scm raises MlsynthConfigError for non-unique spillover_unit_ids."""
    mock_get_data.return_value = {
        "donor_matrix": np.random.rand(10, 3), "y": np.random.rand(10),
        "pre_periods": 5, "post_periods": 5, "donor_names": ["C1", "S1", "S2"], "T0": 5,
    }
    with pytest.raises(MlsynthConfigError, match="Spillover unit IDs must be unique"):
        iterative_scm(sample_clustersc_scm, ["S1", "S1"])


@patch('mlsynth.utils.spillover._get_data')
def test_iterative_scm_insufficient_clean_donors(mock_get_data: MagicMock, sample_clustersc_scm: CLUSTERSC):
    """Test iterative_scm raises MlsynthConfigError if less than 2 clean donors remain."""
    mock_get_data.return_value = {
        "donor_matrix": np.random.rand(10, 3), "y": np.random.rand(10),
        "pre_periods": 5, "post_periods": 5, "donor_names": ["S1", "S2", "C1"], "T0": 5,
    } # Only C1 would be clean if S1, S2 are spillover
    with pytest.raises(MlsynthConfigError, match="At least 2 initial clean donors are required"):
        iterative_scm(sample_clustersc_scm, ["S1", "S2"])

@patch('mlsynth.utils.spillover._estimate_counterfactual')
@patch('mlsynth.utils.spillover._get_data')
def test_iterative_scm_counterfactual_estimation_fails(
    mock_get_data: MagicMock, mock_estimate_cf: MagicMock, sample_clustersc_scm: CLUSTERSC
):
    """Test iterative_scm raises MlsynthEstimationError if _estimate_counterfactual fails."""
    mock_get_data.return_value = {
        "donor_matrix": np.random.rand(10, 3), "y": np.random.rand(10),
        "pre_periods": 5, "post_periods": 5, "donor_names": ["C1", "S1", "S2"], "T0": 5,
    }
    mock_estimate_cf.side_effect = Exception("CF estimation error") # Generic exception to be wrapped
    with pytest.raises(MlsynthEstimationError, match="Counterfactual estimation failed for spillover donor 'S1': CF estimation error"):
        iterative_scm(sample_clustersc_scm, ["S1"])


@patch('mlsynth.utils.spillover._estimate_counterfactual') 
@patch('mlsynth.utils.spillover._get_data') 
def test_iterative_scm_final_fit_fails(
    mock_get_data_actual: MagicMock, 
    mock_estimate_cf_actual: MagicMock, 
    sample_clustersc_scm: CLUSTERSC,
    sample_spillover_df: pd.DataFrame 
):
    """Test iterative_scm raises MlsynthEstimationError if the final SCM fit fails."""
    mock_get_data_actual.return_value = { 
        "donor_matrix": np.random.rand(10, 3), "y": np.random.rand(10),
        "pre_periods": 5, "post_periods": 5, "donor_names": ["C1", "S1", "S2"], "T0": 5,
    }
    mock_estimate_cf_actual.return_value = np.random.rand(10) 
    
    mock_returned_scm_instance = MagicMock(spec=type(sample_clustersc_scm))
    mock_returned_scm_instance.fit.side_effect = Exception("Final SCM fit error") # Generic to be wrapped

    patch_target_str = f"mlsynth.utils.spillover.{sample_clustersc_scm.__class__.__name__}"
    with patch(patch_target_str) as mocked_scm_class:
        mocked_scm_class.return_value = mock_returned_scm_instance
        
        with pytest.raises(MlsynthEstimationError, match="Final SCM fitting failed after spillover cleaning: Final SCM fit error"):
            iterative_scm(sample_clustersc_scm, ["S1"])
        
        mocked_scm_class.assert_called_once()
        mock_returned_scm_instance.fit.assert_called_once()

def test_iterative_scm_invalid_method_type(sample_clustersc_scm: CLUSTERSC):
    """Test iterative_scm with invalid type for method parameter."""
    with pytest.raises(MlsynthConfigError, match="method, if provided, must be a string"):
        iterative_scm(sample_clustersc_scm, ["S1"], method=123) # type: ignore
