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
    """Test valid instantiation of the modernized FMAConfig (Li & Sonnier 2023)."""
    config = FMAConfig(**base_config_data)
    # Paper-aligned defaults
    assert config.stationarity == "nonstationary"
    assert config.preprocessing == "demean"
    assert config.n_factors is None
    assert config.max_factors == 10
    assert config.alpha == 0.05
    assert config.inference_methods == ["asymptotic"]
    assert config.n_bootstrap == 1000

    # Explicit overrides
    cfg2 = FMAConfig(
        **base_config_data,
        stationarity="stationary", preprocessing="standardize",
        n_factors=3, inference_methods=["asymptotic", "bootstrap"],
    )
    assert cfg2.stationarity == "stationary"
    assert cfg2.preprocessing == "standardize"
    assert cfg2.n_factors == 3
    assert "bootstrap" in cfg2.inference_methods

    # Invalid inference method rejected
    with pytest.raises(MlsynthConfigError):
        FMAConfig(**base_config_data, inference_methods=["nope"])

    # Invalid stationarity rejected
    with pytest.raises(ValidationError):
        FMAConfig(**base_config_data, stationarity="bogus")

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
    """Test valid instantiation of CLUSTERSCConfig.

    Covers (a) modern field names, (b) legacy-field accommodation
    via the `_normalise_legacy_fields` validator (objective, cluster,
    Frequentist, ROB, uppercase method tags), and (c) Literal rejection.
    """
    # Modern field names
    config = CLUSTERSCConfig(
        **base_config_data,
        method="rpca",
        rpca_method="HQF",
        pcr_objective="OLS",
        estimator="frequentist",
    )
    assert config.method == "rpca"
    assert config.rpca_method == "HQF"
    assert config.pcr_objective == "OLS"
    assert config.estimator == "frequentist"

    # Legacy field names should be accepted and normalised
    legacy = CLUSTERSCConfig(
        **base_config_data,
        method="BOTH",
        objective="SIMPLEX",
        cluster=False,
        Frequentist=False,
        ROB="PCP",
    )
    assert legacy.method == "both"
    assert legacy.pcr_objective == "SIMPLEX"
    assert legacy.clustering is False
    assert legacy.estimator == "bayesian"
    assert legacy.rpca_method == "PCP"

    # Invalid method rejected
    with pytest.raises(ValidationError):
        CLUSTERSCConfig(**base_config_data, method="bogus")

    # Invalid pcr_objective rejected
    with pytest.raises(ValidationError):
        CLUSTERSCConfig(**base_config_data, pcr_objective="INVALID")

    # Invalid rpca_method rejected
    with pytest.raises(ValidationError):
        CLUSTERSCConfig(**base_config_data, rpca_method="INVALID")

def test_proximal_config_valid(base_config_data: Dict[str, Any], sample_df: pd.DataFrame):
    """Test valid instantiation of PROXIMALConfig with the methods API."""
    # 'methods', 'donors', and (for PI) 'vars' with 'donorproxies' are required.
    prox_data = {
        **base_config_data,
        "methods": ["PI"],
        "donors": [1, 2],
        "vars": {"donorproxies": ["donor_proxy_1"]}
    }
    config = PROXIMALConfig(**prox_data)
    assert config.methods == ["PI"]
    assert config.donors == [1, 2]
    assert config.surrogates == []  # Default
    assert config.vars == {"donorproxies": ["donor_proxy_1"]}
    assert config.counterfactual_color == ["grey", "red", "blue"]

    # SPSC needs no proxies at all.
    spsc_only = PROXIMALConfig(**{**base_config_data, "methods": ["SPSC"], "donors": [1, 2]})
    assert spsc_only.methods == ["SPSC"]

    # PI without 'donorproxies' is rejected.
    invalid_vars_data = {**base_config_data, "methods": ["PI"], "donors": [1, 2], "vars": {}}
    with pytest.raises(MlsynthConfigError, match="donorproxies"):
        PROXIMALConfig(**invalid_vars_data)

    # PIS without 'surrogatevars' is rejected.
    invalid_surrogate_vars_data = {
        **base_config_data, "methods": ["PI", "PIS"], "donors": [1, 2],
        "surrogates": [3], "vars": {"donorproxies": ["donor_proxy_1"]},
    }
    with pytest.raises(MlsynthConfigError, match="surrogatevars"):
        PROXIMALConfig(**invalid_surrogate_vars_data)

    # Unknown method is rejected.
    with pytest.raises(MlsynthConfigError, match="Unknown PROXIMAL method"):
        PROXIMALConfig(**{**base_config_data, "methods": ["WRONG"], "donors": [1, 2]})

    with pytest.raises(ValidationError):  # Missing 'methods' and 'donors'
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
    """Test valid instantiation of the modernized NSCConfig (Tian 2023)."""
    # Defaults: a / b = None triggers CV; cv_target = "controls"; etc.
    cfg = NSCConfig(**base_config_data)
    assert cfg.a is None and cfg.b is None
    assert cfg.cv_target == "controls"
    assert cfg.cv_grid_size == 0.1
    assert cfg.alpha == 0.05
    assert cfg.run_inference is True

    # Explicit (a, b) inside the dimensionless [0, 1] range.
    cfg2 = NSCConfig(**base_config_data, a=0.3, b=0.2)
    assert cfg2.a == 0.3 and cfg2.b == 0.2

    # a outside [0, 1] is rejected by Pydantic.
    with pytest.raises(ValidationError):
        NSCConfig(**base_config_data, a=1.5)

    # Unknown CV target is rejected.
    with pytest.raises(ValidationError):
        NSCConfig(**base_config_data, cv_target="bogus")

    # cv_grid_size outside (0, 0.5] is rejected.
    with pytest.raises(ValidationError):
        NSCConfig(**base_config_data, cv_grid_size=0.0)
    with pytest.raises(ValidationError):
        NSCConfig(**base_config_data, cv_grid_size=0.6)

    # Unknown covariate column raises MlsynthConfigError via validator.
    with pytest.raises(MlsynthConfigError):
        NSCConfig(**base_config_data, covariates=["not_a_column"])

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
    import pandas as pd
    import pytest
    from datetime import datetime, timedelta
    from mlsynth.config_models import MAREXConfig
    from mlsynth.exceptions import MlsynthDataError

    df = pd.DataFrame({
        "unit": [1, 1, 1],
        "time": [datetime(2020, 1, 1),
                 datetime(2020, 1, 3),  # Gap here
                 datetime(2020, 1, 4)],
        "y": [1, 2, 3]
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
