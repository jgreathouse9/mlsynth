# tests/test_marex.py
import pytest
import pandas as pd
import numpy as np
from mlsynth import MAREX
from mlsynth.config_models import MAREXConfig
from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError
from pydantic import ValidationError
import warnings


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

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = MAREXConfig(**config)

        # Expected sorted df: alphabetical town, then increasing time
        df_expected = cfg.df.sort_values(["town", "time"], kind="mergesort").reset_index(drop=True)
        pd.testing.assert_frame_equal(cfg.df, df_expected)




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
    df = curacao_sim_data["df"].copy()
    df["Region"] = df["Region"].astype(str)  # force Region to string

    config_data = {
        "df": df,  # use the modified df here
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "cluster": "Region",  # this column has strings
        "m_eq": 1
    }

    # Expect a UserWarning about automatic integer conversion
    with pytest.warns(UserWarning, match="Cluster column 'Region' contains strings or categories"):
        MAREXConfig(**config_data)



def test_cluster_missing_values(curacao_sim_data):
    df_base = curacao_sim_data["df"].copy()

    # --------------------------
    # Case 1: Entire cluster column missing → should raise error
    # --------------------------
    df_all_missing = df_base.copy()
    df_all_missing["Region"] = pd.NA  # all missing

    config_all_missing = {
        "df": df_all_missing,
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "cluster": "Region",
        "m_eq": 1
    }

    with pytest.raises(MlsynthDataError, match="Cluster column 'Region' contains only missing values"):
        MAREXConfig(**config_all_missing)

    # --------------------------
    # Case 2: Some missing values → should warn
    # --------------------------
    df_some_missing = df_base.copy()
    df_some_missing.loc[df_some_missing.index[:3], "Region"] = pd.NA  # first 3 missing

    config_some_missing = {
        "df": df_some_missing,
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "cluster": "Region",
        "m_eq": 1
    }

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = MAREXConfig(**config_some_missing)
        # Expect a warning about missing cluster values
        assert any("contains 3 missing values" in str(warn.message) for warn in w)


def test_m_eq_greater_than_cluster(curacao_sim_data):
    config = MAREXConfig(df=curacao_sim_data["df"], outcome="Y_obs", unitid="town", time="time",
                         cluster="Region", m_eq=1000)
    marex = MAREX(config)
    with pytest.raises(MlsynthConfigError):
        marex.fit()











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
