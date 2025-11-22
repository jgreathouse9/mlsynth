import pytest
import pandas as pd
from pydantic import ValidationError
from typing import List, Dict, Any

# Assuming config_models.py is in mlsynth directory, adjust import path as necessary
from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError
from mlsynth.config_models import (
    BaseEstimatorConfig,
    TSSCConfig,
    FMAConfig,
    PDAConfig,
    FDIDConfig,
    GSCConfig,
    CLUSTERSCConfig,
    PROXIMALConfig,
    FSCMConfig,
    SRCConfig,
    SCMOConfig,
    SIConfig,
    StableSCConfig,
    NSCConfig,
    SDIDConfig,
    MAREXConfig
)

# Sample valid DataFrame for testing
@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        'unit': [1, 1, 2, 2, 3, 3],
        'time': [1, 2, 1, 2, 1, 2],
        'outcome_var': [10, 12, 15, 16, 11, 13],
        'treat_var': [0, 0, 1, 1, 0, 0],
        'aux_outcome': [1, 2, 3, 4, 5, 6],
        'inter_1': [0,0,1,1,0,0],
        'inter_2': [0,0,0,0,1,1],
        'donor_proxy_1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'surrogate_var_1': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    })

@pytest.fixture
def base_config_data(sample_df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "df": sample_df,
        "outcome": "outcome_var",
        "treat": "treat_var",
        "unitid": "unit",
        "time": "time"
    }

def test_base_estimator_config_valid(base_config_data: Dict[str, Any]):
    """Test valid instantiation of BaseEstimatorConfig."""
    config = BaseEstimatorConfig(**base_config_data)
    assert config.df.equals(base_config_data["df"])
    assert config.outcome == "outcome_var"
    assert config.display_graphs is True # Check default
    assert config.counterfactual_color == ["red"]

def test_base_estimator_config_missing_required(base_config_data: Dict[str, Any]):
    """Test ValidationError for missing required fields in BaseEstimatorConfig."""
    incomplete_data = base_config_data.copy()
    del incomplete_data["df"]
    with pytest.raises(ValidationError):
        BaseEstimatorConfig(**incomplete_data)

def test_base_estimator_config_invalid_type(base_config_data: Dict[str, Any]):
    """Test ValidationError for invalid type in BaseEstimatorConfig."""
    invalid_data = base_config_data.copy()
    invalid_data["display_graphs"] = "not_a_bool"
    with pytest.raises(ValidationError):
        BaseEstimatorConfig(**invalid_data)

# --- Tests for BaseEstimatorConfig model_validator ---

def test_base_config_df_empty(base_config_data: Dict[str, Any]):
    """Test MlsynthDataError if df is empty."""
    config_data = base_config_data.copy()
    config_data["df"] = pd.DataFrame() # Empty DataFrame
    with pytest.raises(MlsynthDataError, match="Input DataFrame 'df' cannot be empty."):
        BaseEstimatorConfig(**config_data)

def test_base_config_missing_outcome_column(base_config_data: Dict[str, Any], sample_df: pd.DataFrame):
    """Test MlsynthDataError if outcome column is missing in df."""
    config_data = base_config_data.copy()
    df_missing_col = sample_df.drop(columns=["outcome_var"])
    config_data["df"] = df_missing_col
    with pytest.raises(MlsynthDataError, match="Missing required columns in DataFrame 'df': outcome_var"):
        BaseEstimatorConfig(**config_data)

def test_base_config_missing_treat_column(base_config_data: Dict[str, Any], sample_df: pd.DataFrame):
    """Test MlsynthDataError if treat column is missing in df."""
    config_data = base_config_data.copy()
    df_missing_col = sample_df.drop(columns=["treat_var"])
    config_data["df"] = df_missing_col
    with pytest.raises(MlsynthDataError, match="Missing required columns in DataFrame 'df': treat_var"):
        BaseEstimatorConfig(**config_data)

def test_base_config_missing_unitid_column(base_config_data: Dict[str, Any], sample_df: pd.DataFrame):
    """Test MlsynthDataError if unitid column is missing in df."""
    config_data = base_config_data.copy()
    df_missing_col = sample_df.drop(columns=["unit"])
    config_data["df"] = df_missing_col
    with pytest.raises(MlsynthDataError, match="Missing required columns in DataFrame 'df': unit"):
        BaseEstimatorConfig(**config_data)

def test_base_config_missing_time_column(base_config_data: Dict[str, Any], sample_df: pd.DataFrame):
    """Test MlsynthDataError if time column is missing in df."""
    config_data = base_config_data.copy()
    df_missing_col = sample_df.drop(columns=["time"])
    config_data["df"] = df_missing_col
    with pytest.raises(MlsynthDataError, match="Missing required columns in DataFrame 'df': time"):
        BaseEstimatorConfig(**config_data)

def test_base_config_missing_multiple_columns(base_config_data: Dict[str, Any], sample_df: pd.DataFrame):
    """Test MlsynthDataError if multiple required columns are missing in df."""
    config_data = base_config_data.copy()
    df_missing_cols = sample_df.drop(columns=["outcome_var", "time"])
    config_data["df"] = df_missing_cols
    with pytest.raises(MlsynthDataError, match="Missing required columns in DataFrame 'df': outcome_var, time"):
        BaseEstimatorConfig(**config_data)

# --- End of tests for BaseEstimatorConfig model_validator ---

def test_tssc_config_valid(base_config_data: Dict[str, Any]):
    """Test valid instantiation of TSSCConfig with defaults and specific args."""
    config = TSSCConfig(**base_config_data)
    assert config.draws == 500 # Default
    assert config.ci == 0.95 # Default

    config_custom = TSSCConfig(**base_config_data, draws=100, ci=0.9)
    assert config_custom.draws == 100
    assert config_custom.ci == 0.9

def test_fma_config_valid(base_config_data: Dict[str, Any]):
    """Test valid instantiation of FMAConfig."""
    config = FMAConfig(**base_config_data)
    assert config.criti == 11
    assert config.DEMEAN == 1

def test_pda_config_valid(base_config_data: Dict[str, Any]):
    """Test valid instantiation of PDAConfig and method validation."""
    config = PDAConfig(**base_config_data, method="LASSO")
    assert config.method == "LASSO"
    with pytest.raises(ValidationError): # Invalid method
        PDAConfig(**base_config_data, method="INVALID_METHOD")

def test_fdid_config_valid(base_config_data: Dict[str, Any]):
    """Test valid instantiation of FDIDConfig."""
    FDIDConfig(**base_config_data) # Should not raise

def test_gsc_config_valid(base_config_data: Dict[str, Any]):
    """Test valid instantiation of GSCConfig."""
    GSCConfig(**base_config_data) # Should not raise

def test_clustersc_config_valid(base_config_data: Dict[str, Any]):
    """Test valid instantiation of CLUSTERSCConfig."""
    config = CLUSTERSCConfig(**base_config_data, method="RPCA", ROB="HQF")
    assert config.method == "RPCA"
    assert config.ROB == "HQF"
    with pytest.raises(ValidationError):
        CLUSTERSCConfig(**base_config_data, objective="INVALID_OBJECTIVE")

def test_proximal_config_valid(base_config_data: Dict[str, Any], sample_df: pd.DataFrame):
    """Test valid instantiation of PROXIMALConfig."""
    # 'donors' and 'vars' with 'donorproxies' are required
    prox_data = {
        **base_config_data,
        "donors": [1, 2],
        "vars": {"donorproxies": ["donor_proxy_1"]}
    }
    config = PROXIMALConfig(**prox_data)
    assert config.donors == [1, 2]
    assert config.surrogates == []  # Default
    assert config.vars == {"donorproxies": ["donor_proxy_1"]}
    assert config.counterfactual_color == ["grey", "red", "blue"]

    # Test case for invalid 'vars' (missing 'donorproxies')
    invalid_vars_data = {**base_config_data, "donors": [1, 2], "vars": {}}
    with pytest.raises(MlsynthConfigError, match="Config 'vars' must contain a non-empty list for 'donorproxies'."):
        PROXIMALConfig(**invalid_vars_data)
    
    # Test case for invalid 'vars' (missing 'surrogatevars' when surrogates are present)
    invalid_surrogate_vars_data = {
        **base_config_data,
        "donors": [1, 2],
        "surrogates": [3], # Surrogates are present
        "vars": {"donorproxies": ["donor_proxy_1"]} # Missing surrogatevars
    }
    with pytest.raises(MlsynthConfigError, match="Config 'vars' must contain a non-empty list for 'surrogatevars' when surrogates are provided."):
        PROXIMALConfig(**invalid_surrogate_vars_data)

    with pytest.raises(ValidationError):  # Missing 'donors' (original test for ValidationError)
        PROXIMALConfig(**base_config_data)

def test_fscm_config_valid(base_config_data: Dict[str, Any]):
    """Test valid instantiation of FSCMConfig."""
    FSCMConfig(**base_config_data) # Should not raise

def test_src_config_valid(base_config_data: Dict[str, Any]):
    """Test valid instantiation of SRCConfig."""
    SRCConfig(**base_config_data) # Should not raise

def test_scmo_config_valid(base_config_data: Dict[str, Any]):
    """Test valid instantiation of SCMOConfig."""
    config = SCMOConfig(**base_config_data, addout="aux_outcome", method="SBMF")
    assert config.addout == "aux_outcome"
    assert config.method == "SBMF"
    with pytest.raises(ValidationError):
        SCMOConfig(**base_config_data, method="INVALID")

def test_si_config_valid(base_config_data: Dict[str, Any]):
    """Test valid instantiation of SIConfig."""
    # 'inters' is required
    si_data = {**base_config_data, "inters": ["inter_1", "inter_2"]}
    config = SIConfig(**si_data)
    assert config.inters == ["inter_1", "inter_2"]
    with pytest.raises(ValidationError): # Missing 'inters'
        SIConfig(**base_config_data)

def test_stablesc_config_valid(base_config_data: Dict[str, Any]):
    """Test valid instantiation of StableSCConfig."""
    StableSCConfig(**base_config_data) # Should not raise

def test_nsc_config_valid(base_config_data: Dict[str, Any]):
    """Test valid instantiation of NSCConfig."""
    NSCConfig(**base_config_data) # Should not raise

    # Test cases for NSCConfig validator
    invalid_a_space_type = {**base_config_data, "a_search_space": ["not_a_float", 0.5]}
    # Pydantic's own type validation for List[float] will catch this first.
    with pytest.raises(ValidationError): # Expect Pydantic's ValidationError for type mismatches
        NSCConfig(**invalid_a_space_type)

    invalid_b_space_type = {**base_config_data, "b_search_space": [0.1, "not_a_float"]}
    # Pydantic's own type validation for List[float] will catch this first.
    with pytest.raises(ValidationError): # Expect Pydantic's ValidationError for type mismatches
        NSCConfig(**invalid_b_space_type)

    # These custom MlsynthConfigError checks are for valid types but invalid content (e.g. empty list)
    empty_a_space = {**base_config_data, "a_search_space": []}
    with pytest.raises(MlsynthConfigError, match="a_search_space cannot be an empty list if provided."):
        NSCConfig(**empty_a_space)

    empty_b_space = {**base_config_data, "b_search_space": []}
    with pytest.raises(MlsynthConfigError, match="b_search_space cannot be an empty list if provided."):
        NSCConfig(**empty_b_space)

def test_sdid_config_valid(base_config_data: Dict[str, Any]):
    """Test valid instantiation of SDIDConfig."""
    config = SDIDConfig(**base_config_data, B=100)
    assert config.B == 100
    with pytest.raises(ValidationError): # B must be >= 0
        SDIDConfig(**base_config_data, B=-1)


# ---------------------------
# Numeric time column
# ---------------------------
def test_numeric_time_consecutive():
    df = pd.DataFrame({
        "unit": [1, 1, 2, 2],
        "time": [1, 2, 1, 2],
        "y": [0, 1, 2, 3]
    })
    # Should not raise
    config = MAREXConfig(df=df, outcome="y", unitid="unit", time="time")

def test_numeric_time_non_consecutive():
    df = pd.DataFrame({
        "unit": [1, 1, 2, 2],
        "time": [1, 3, 1, 3],  # non-consecutive
        "y": [0, 1, 2, 3]
    })
    with pytest.raises(MlsynthDataError, match="Time periods in 'time' are not consecutive"):
        MAREXConfig(df=df, outcome="y", unitid="unit", time="time")

# ---------------------------
# Datetime time column
# ---------------------------
def test_datetime_time_consecutive():
    df = pd.DataFrame({
        "unit": [1, 1, 2, 2],
        "time": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"]),
        "y": [0, 1, 2, 3]
    })
    # Should not raise
    config = MAREXConfig(df=df, outcome="y", unitid="unit", time="time")

def test_datetime_time_non_consecutive():
    # Non-consecutive datetimes for time column
    df = pd.DataFrame({
        "unit": [1, 1, 2, 2],
        "time": pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-01", "2025-01-03"]),
        "y": [0, 1, 2, 3]
    })

    with pytest.raises(MlsynthDataError, match="Datetime time periods in 'time' are not consecutive"):
        MAREXConfig(df=df, outcome="y", unitid="unit", time="time")

# ---------------------------
# Unsupported dtype
# ---------------------------
def test_time_unsupported_dtype():
    df = pd.DataFrame({
        "unit": [1, 1, 2, 2],
        "time": ["a", "b", "a", "b"],  # string dtype
        "y": [0, 1, 2, 3]
    })
    with pytest.raises(MlsynthDataError, match="Unsupported dtype for time column"):
        MAREXConfig(df=df, outcome="y", unitid="unit", time="time")
