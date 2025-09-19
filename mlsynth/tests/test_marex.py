# tests/test_marex.py
import pytest
import pandas as pd
import numpy as np
from mlsynth import MAREX
from mlsynth.config_models import MAREXConfig
from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError
from pydantic import ValidationError

# ----------------------------------------------------
# Initialization Tests
# ----------------------------------------------------

def test_initialization_valid_config(curacao_sim_data):
    config_data = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "T0": 104,
        "cluster": "Region",
        "design": "eq11",
        "m_eq": 1,
        "lambda1": 0.2,
        "lambda2": 0.2
    }
    try:
        marex = MAREX(config=MAREXConfig(**config_data))
        assert marex.df is not None
        assert marex.outcome == "Y_obs"
        assert marex.T0 == 104
        assert marex.cluster == "Region"
    except ValidationError as e:
        assert False, f"Initialization failed with ValidationError: {e}"

def test_initialization_invalid_config(curacao_sim_data):
    config_data = {
        "df": curacao_sim_data["df"],
        "unitid": "town",  # Missing outcome
        "time": "time",
        "T0": 104,
        "cluster": "Region",
        "design": "eq11",
        "m_eq": 1,
        "lambda1": 0.2,
        "lambda2": 0.2
    }
    with pytest.raises(ValidationError):
        MAREXConfig(**config_data)

def test_initialization_invalid_cluster_column(curacao_sim_data):
    config_data = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "T0": 104,
        "cluster": "InvalidColumn",  # Non-existent column
        "design": "eq11",
        "m_eq": 1,
        "lambda1": 0.2,
        "lambda2": 0.2
    }
    with pytest.raises(MlsynthDataError):
        MAREX(config=MAREXConfig(**config_data))

def test_initialization_invalid_data_type(curacao_sim_data):
    config_data = {
        "df": "not_a_dataframe",
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "T0": 104,
        "cluster": "Region",
        "design": "eq11",
        "m_eq": 1,
        "lambda1": 0.2,
        "lambda2": 0.2
    }
    with pytest.raises(ValidationError):
        MAREXConfig(**config_data)

def test_initialization_missing_column(curacao_sim_data):
    config_data = {
        "df": curacao_sim_data["df"].drop(columns=["Y_obs"]),
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "T0": 104,
        "cluster": "Region",
        "design": "eq11",
        "m_eq": 1,
        "lambda1": 0.2,
        "lambda2": 0.2
    }
    with pytest.raises(MlsynthDataError):
        MAREX(config=MAREXConfig(**config_data))

def test_init_empty_dataframe():
    df_empty = pd.DataFrame(columns=["town", "time", "Y_obs"])
    config = {
        "df": df_empty,
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time"
    }
    with pytest.raises(MlsynthDataError):
        MAREXConfig(**config)

def test_init_duplicate_rows(curacao_sim_data):
    df_dup = pd.concat([curacao_sim_data["df"], curacao_sim_data["df"].iloc[:1]])
    config = {
        "df": df_dup,
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time"
    }
    with pytest.raises(MlsynthDataError):
        MAREXConfig(**config)

def test_init_unsorted_dataframe_warning(curacao_sim_data):
    df_shuffled = curacao_sim_data["df"].sample(frac=1).reset_index(drop=True)
    config = {
        "df": df_shuffled,
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time"
    }
    with pytest.warns(UserWarning):
        cfg = MAREXConfig(**config)
        assert cfg.df.equals(cfg.df.sort_values(["town", "time"]).reset_index(drop=True))

def test_init_invalid_design(curacao_sim_data):
    config = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "design": "invalid_design"
    }
    with pytest.raises(MlsynthDataError):
        MAREXConfig(**config)

def test_init_invalid_T0(curacao_sim_data):
    max_periods = curacao_sim_data["df"]["time"].nunique()
    config = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "T0": max_periods + 1
    }
    with pytest.raises(MlsynthDataError):
        MAREXConfig(**config)

def test_clusters_column_as_string(curacao_sim_data):
    # Make a copy and force Region to string
    df = curacao_sim_data["df"].copy()
    df["Region"] = df["Region"].astype(str)

    config_data = {
        "df": df,
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "cluster": "Region",  # this column has strings
        "m_eq": 1
    }

    # Check that a UserWarning is raised for automatic integer conversion
    with pytest.warns(UserWarning, match="Cluster column 'Region' contains strings or categories"):
        marex_config = MAREXConfig(**config_data)

    # Check that the df column is now integer-coded
    assert pd.api.types.is_integer_dtype(marex_config.df["Region"])




def test_m_eq_greater_than_cluster(curacao_sim_data):
    # Find the max cluster size
    cluster_sizes = curacao_sim_data["df"].groupby("Region").size()
    too_large = cluster_sizes.max() + 1

    config = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "cluster": "Region",
        "m_eq": too_large,
    }
    with pytest.raises(MlsynthDataError, match="cannot be greater than"):
        MAREXConfig(**config)


def test_m_min_less_than_one(curacao_sim_data):
    config = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "cluster": "Region",
        "m_min": 0,  # invalid
    }
    with pytest.raises(MlsynthDataError, match="m_min must be >= 1"):
        MAREXConfig(**config)


def test_m_max_greater_than_smallest_cluster(curacao_sim_data):
    # Find the smallest cluster size
    min_cluster_size = curacao_sim_data["df"].groupby("Region").size().min()
    too_large = min_cluster_size + 1

    config = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "cluster": "Region",
        "m_max": too_large,
    }
    with pytest.raises(MlsynthDataError, match="cannot be greater than the smallest cluster size"):
        MAREXConfig(**config)


def test_m_min_greater_than_m_max(curacao_sim_data):
    config = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "cluster": "Region",
        "m_min": 5,
        "m_max": 3,
    }
    with pytest.raises(MlsynthDataError, match="m_min .* cannot be greater than m_max"):
        MAREXConfig(**config)


def test_valid_m_min_and_m_max(curacao_sim_data):
    # Choose values within bounds
    cluster_sizes = curacao_sim_data["df"].groupby("Region").size()
    valid_min = 1
    valid_max = cluster_sizes.min()

    config = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "cluster": "Region",
        "m_min": valid_min,
        "m_max": valid_max,
    }

    # Should not raise
    cfg = MAREXConfig(**config)
    assert cfg.m_min == valid_min
    assert cfg.m_max == valid_max



def test_fit_minimal_config(curacao_sim_data):
    config = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "T0": 50,  # within range
        "m_eq": 1
    }
    marex = MAREX(config=MAREXConfig(**config))
    results = marex.fit()
    assert results is not None
    assert hasattr(results, "study")



def test_default_lambdas(curacao_sim_data):
    config = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "m_eq": 1
    }
    cfg = MAREXConfig(**config)
    assert cfg.lambda1 == 0.0
    assert cfg.lambda2 == 0.0


def test_unit_invariant_cluster(curacao_sim_data):
    """
    Ensure that if a cluster column is specified, each unit is assigned to
    exactly one cluster across all time periods. If a unit appears in multiple
    clusters, the validator should raise a MlsynthDataError.
    """
    df = curacao_sim_data["df"].copy()

    # Pick a unit and break invariance: assign different clusters at different times
    example_unit = df["town"].iloc[0]
    unit_mask = df["town"] == example_unit
    df.loc[unit_mask, "Region"] = [0, 1] + [0] * (unit_mask.sum() - 2)  # first two rows conflicting

    config_data = {
        "df": df,
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "cluster": "Region",
        "T0": 50,
        "m_eq": 1
    }

    # Should raise error due to multiple clusters per unit
    with pytest.raises(MlsynthDataError, match="multiple cluster assignments"):
        MAREXConfig(**config_data)

    # Now test a valid case: all units invariant
    df_valid = curacao_sim_data["df"].copy()
    config_data_valid = {
        "df": df_valid,
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "cluster": "Region",
        "T0": 50,
        "m_eq": 1
    }

    cfg = MAREXConfig(**config_data_valid)
    assert cfg.df.equals(df_valid)



# ----------------------------------------------------
# fit() Method Tests
# ----------------------------------------------------

def test_fit_valid_config(curacao_sim_data):
    config_data = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "T0": 104,
        "cluster": "Region",
        "design": "eq11",
        "m_eq": 1,
        "lambda1": 0.2,
        "lambda2": 0.2
    }
    marex = MAREX(config=MAREXConfig(**config_data))
    results = marex.fit()
    assert results is not None
    assert hasattr(results, "clusters")
    assert hasattr(results, "study")
    assert hasattr(results, "globres")

def test_fit_no_cluster(curacao_sim_data):
    config_data = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "T0": 104,
        "cluster": None,
        "design": "eq11",
        "m_eq": 1,
        "lambda1": 0.2,
        "lambda2": 0.2
    }
    marex = MAREX(config=MAREXConfig(**config_data))
    results = marex.fit()
    assert results is not None
    assert hasattr(results, "clusters")

def test_fit_conflicting_m_eq_m_min_max(curacao_sim_data):
    config = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "m_eq": 1,
        "m_min": 1,
        "m_max": 2
    }
    marex = MAREX(config=MAREXConfig(**config))
    with pytest.raises(MlsynthConfigError):
        marex.fit()

def test_fit_missing_m_eq_or_range(curacao_sim_data):
    config = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time"
    }
    marex = MAREX(config=MAREXConfig(**config))
    with pytest.raises(MlsynthConfigError):
        marex.fit()

def test_fit_extreme_values(curacao_sim_data):
    config_data = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "T0": 127,
        "cluster": "Region",
        "design": "eq11",
        "m_eq": 3,
        "lambda1": 1000.0,
        "lambda2": 1000.0
    }
    marex = MAREX(config=MAREXConfig(**config_data))
    results = marex.fit()
    assert results is not None
    assert hasattr(results, "clusters")

def test_fit_cluster_none(curacao_sim_data):
    config = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "cluster": None,
        "m_eq": 1
    }
    marex = MAREX(config=MAREXConfig(**config))
    results = marex.fit()
    assert results is not None
    assert all(isinstance(c, type(next(iter(results.clusters.values())))) for c in results.clusters.values())

# ----------------------------------------------------
# Results Structure Tests
# ----------------------------------------------------

def test_results_structure(curacao_sim_data):
    config = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "cluster": "Region",
        "m_eq": 1
    }
    marex = MAREX(config=MAREXConfig(**config))
    results = marex.fit()

    # Global results
    glob = results.globres
    assert hasattr(glob, "Y_fit")
    assert hasattr(glob, "treated_weights_agg")
    assert hasattr(glob, "control_weights_agg")

    # Cluster results
    for cluster_id, cluster_res in results.clusters.items():
        assert isinstance(cluster_res.members, list)
        assert cluster_res.cluster_cardinality == len(cluster_res.members)
        assert cluster_res.synthetic_treated.shape[0] <= results.study.T0
        assert cluster_res.synthetic_control.shape[0] <= results.study.T0
        assert cluster_res.treated_weights.shape[0] == len(cluster_res.members)
        assert cluster_res.control_weights.shape[0] == len(cluster_res.members)
        assert cluster_res.selection_indicators.shape[0] == len(cluster_res.members)
        assert isinstance(cluster_res.unit_weight_map, dict)

# ----------------------------------------------------
# Extreme / Stress Tests
# ----------------------------------------------------

def test_fit_extreme_penalty_values(curacao_sim_data):
    config = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "cluster": "Region",
        "lambda1": 1e6,
        "lambda2": 1e6,
        "design": "eq11",
        "m_eq": 1
    }
    marex = MAREX(config=MAREXConfig(**config))
    results = marex.fit()
    assert results is not None
    assert len(results.clusters) > 0

def test_all_units_one_cluster(curacao_sim_data):
    df = curacao_sim_data["df"].copy()
    df["Region"] = 0  # all units in a single cluster
    config = {
        "df": df,
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "cluster": "Region",
        "m_eq": 2,
    }
    marex = MAREX(config=MAREXConfig(**config))
    results = marex.fit()
    assert results is not None
    assert len(results.clusters) == 1

def test_T0_edge_cases(curacao_sim_data):
    max_periods = curacao_sim_data["df"]["time"].nunique()
    
    # T0 = 1 (minimal pre-treatment)
    config1 = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "T0": 1,
        "m_eq": 1
    }
    marex1 = MAREX(config=MAREXConfig(**config1))
    results1 = marex1.fit()
    assert results1.study.T0 == 1

    # T0 = max_periods - 1
    config2 = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "T0": max_periods - 1,
        "m_eq": 1
    }
    marex2 = MAREX(config=MAREXConfig(**config2))
    results2 = marex2.fit()
    assert results2.study.T0 == max_periods - 1

def test_non_consecutive_time_indices(curacao_sim_data):
    df = curacao_sim_data["df"].copy()
    # Drop some time points
    df = df[df["time"] % 5 != 0]
    config = {
        "df": df,
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "m_eq": 1
    }
    marex = MAREX(config=MAREXConfig(**config))
    results = marex.fit()
    assert results is not None
    assert results.globres.Y_fit.shape[1] == df["time"].nunique()

def test_explicit_blank_periods(curacao_sim_data):
    df = curacao_sim_data["df"].copy()
    max_periods = df["time"].nunique()
    T0 = max_periods - 2
    explicit_blanks = 2  # should match max_periods - T0
    config = {
        "df": df,
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "T0": T0,
        "blank_periods": explicit_blanks,
        "m_eq": 1
    }
    marex = MAREX(config=MAREXConfig(**config))
    results = marex.fit()
    assert results.study.blank_periods == explicit_blanks
