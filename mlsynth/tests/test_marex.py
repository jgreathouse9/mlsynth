# tests/test_marex.py
import pytest
import pandas as pd
import numpy as np
import cvxpy as cp
from mlsynth import MAREX
from mlsynth.config_models import MAREXConfig
from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError
from pydantic import ValidationError
from mlsynth.utils.exputils import (
    SCMEXP,
    _get_per_cluster_param,
    _prepare_clusters,
    _validate_scm_inputs,
    _validate_costs_budget,
    _prepare_fit_slices,
    _build_membership_mask,
    _compute_cluster_means_members,
    _precompute_distances,
    _init_cvxpy_variables,
    _build_constraints,
    _build_objective, inference_scm_vectorized, 
_compute_placebo_ci_vectorized)

from mlsynth.utils.exprelutils import _post_hoc_discretize, SCMEXP_REL



def test_miqp_solver_available():
    miqp_solvers = ["ECOS_BB", "SCIP", "GUROBI", "MOSEK"]
    installed = cp.installed_solvers()
    # Print installed solvers for visibility
    print("Installed solvers:", installed)
    # Assert that at least one MIQP solver is available
    assert any(solver in installed for solver in miqp_solvers), \
        f"No MIQP solver installed. Installed solvers: {installed}"



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
    # force Region to string
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



def test_unit_invariant_cluster_valid(curacao_sim_data):
    # All units are invariant in cluster by default
    config_data = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "cluster": "Region",
    }

    # Should not raise
    cfg = MAREXConfig(**config_data)

    # Validate that each unit belongs to a single cluster
    unit_groups = cfg.df.groupby(config_data["unitid"])[config_data["cluster"]].nunique()
    assert (unit_groups == 1).all(), "All units should be invariant within their cluster"


def test_unit_invariant_cluster_invalid(curacao_sim_data):
    df_invalid = curacao_sim_data["df"].copy()
    # Introduce a unit with multiple cluster assignments
    df_invalid.loc[df_invalid.index[:2], "Region"] = [0, 1]  # same unit, different clusters

    config_data = {
        "df": df_invalid,
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "cluster": "Region",
    }

    with pytest.raises(MlsynthDataError, match=r"The following units have multiple cluster assignments"):
        MAREXConfig(**config_data)


def make_dummy_df(n_units=4, n_periods=3):
    """Helper to make a minimal valid df for config tests."""
    df = pd.DataFrame({
        "unit": np.repeat(np.arange(n_units), n_periods),
        "time": list(range(n_periods)) * n_units,
        "y": np.random.randn(n_units * n_periods),
        "cluster": np.repeat([0, 1], n_units * n_periods // 2),
    })
    return df


def test_negative_cost_raises():
    df = make_dummy_df()
    with pytest.raises(MlsynthDataError, match="strictly positive"):
        MAREXConfig(
            df=df,
            unitid="unit",
            time="time",
            outcome="y",
            cluster="cluster",
            costs=[1, -5, 3, 4],
            budget=10
        )


def test_zero_cost_raises():
    df = make_dummy_df()
    with pytest.raises(MlsynthDataError, match="strictly positive"):
        MAREXConfig(
            df=df,
            unitid="unit",
            time="time",
            outcome="y",
            cluster="cluster",
            costs=[0, 2, 3, 4],
            budget=10
        )


def test_zero_budget_scalar_raises():
    df = make_dummy_df()
    with pytest.raises(MlsynthDataError, match="strictly positive"):
        MAREXConfig(
            df=df,
            unitid="unit",
            time="time",
            outcome="y",
            cluster="cluster",
            costs=[1, 2, 3, 4],
            budget=0
        )


def test_negative_budget_scalar_raises():
    df = make_dummy_df()
    with pytest.raises(MlsynthDataError, match="strictly positive"):
        MAREXConfig(
            df=df,
            unitid="unit",
            time="time",
            outcome="y",
            cluster="cluster",
            costs=[1, 2, 3, 4],
            budget=-10
        )


def test_missing_cluster_in_budget_dict_raises():
    df = make_dummy_df()
    with pytest.raises(MlsynthDataError, match="missing entry for cluster"):
        MAREXConfig(
            df=df,
            unitid="unit",
            time="time",
            outcome="y",
            cluster="cluster",
            costs=[1, 2, 3, 4],
            budget={0: 5}  # missing cluster 1
        )


def test_nonpositive_budget_dict_raises():
    df = make_dummy_df()
    with pytest.raises(MlsynthDataError, match="strictly positive"):
        MAREXConfig(
            df=df,
            unitid="unit",
            time="time",
            outcome="y",
            cluster="cluster",
            costs=[1, 2, 3, 4],
            budget={0: 5, 1: 0}
        )



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
    # Remove some time points to create gaps
    df = df[~df["time"].isin([5, 10, 15, 20])]
    config = {
        "df": df,
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time"
    }

    # Should raise error due to non-consecutive times
    with pytest.raises(MlsynthDataError, match=r"Time periods in column 'time' are not consecutive"):
        MAREXConfig(**config)


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


# -------------------------
# Fixtures
# -------------------------


@pytest.fixture
def simple_data():
    """
    Generates a small synthetic dataset for SCMEXP testing.
    Returns:
        Y (np.ndarray): outcome matrix, shape (N_units, T_total)
        clusters (np.ndarray): cluster labels, shape (N_units,)
    """
    N_units = 6
    T_total = 5
    # simple synthetic data
    Y = np.array([
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8],
        [5, 6, 7, 8, 9],
        [6, 7, 8, 9, 10]
    ])
    clusters = np.array([0, 0, 1, 1, 2, 2])  # 3 clusters
    return Y, clusters


@pytest.mark.parametrize("design", ["base", "weak", "eq11", "unit"])
def test_scmexp_basic(simple_data, design):
    """Tests SCMEXP for all design modes."""
    Y, clusters = simple_data
    # Convert Y to DataFrame to satisfy SCMEXP
    Y_df = pd.DataFrame(Y, columns=[f"t{i+1}" for i in range(Y.shape[1])])

    res = SCMEXP(Y_full=Y_df, T0=3, clusters=clusters, design=design)

    N = Y.shape[0]
    cluster_labels = np.unique(clusters)
    K = len(cluster_labels)

    # --- Shape checks ---
    assert res["w_opt"].shape == (N, K)
    assert res["v_opt"].shape == (N, K)
    assert res["z_opt"].shape == (N, K)
    assert len(res["y_syn_treated_clusters"]) == K
    assert len(res["y_syn_control_clusters"]) == K
    assert len(res["Xbar_clusters"]) == K

    # --- Weight sum sanity check within clusters ---
    for k_idx, lab in enumerate(cluster_labels):
        members = np.where(clusters == lab)[0]
        np.testing.assert_allclose(res["w_opt"][members, k_idx].sum(), 1, rtol=1e-6)
        np.testing.assert_allclose(res["v_opt"][members, k_idx].sum(), 1, rtol=1e-6)


def test_scmexp_with_cost_budget(simple_data):
    """Tests SCMEXP with costs and budget constraints."""
    Y, clusters = simple_data
    Y_df = pd.DataFrame(Y, columns=[f"t{i+1}" for i in range(Y.shape[1])])

    # Simple cost vector
    costs = np.array([1, 2, 1, 2, 1, 2])
    budget = {0: 1.5, 1: 2.5, 2: 3}  # budget per cluster

    res = SCMEXP(Y_full=Y_df, T0=3, clusters=clusters, design="base",
                 costs=costs, budget=budget)

    # --- Check costs applied ---
    for k_idx, lab in enumerate(np.unique(clusters)):
        members = np.where(clusters == lab)[0]
        total_cost = np.sum(costs[members] * res["w_opt"][members, k_idx])
        assert total_cost <= budget[lab] + 1e-6



# ---------------------------
# Blank periods
# ---------------------------
def test_scmexp_blank_periods(simple_data):
    Y, clusters = simple_data
    Y_df = pd.DataFrame(Y, columns=[f"t{i+1}" for i in range(Y.shape[1])])
    res = SCMEXP(Y_full=Y_df, T0=3, clusters=clusters, blank_periods=1)
    assert res["T_fit"] == 2
    if res["Y_blank"] is not None:
        assert res["Y_blank"].shape[1] == 1

# ---------------------------
# RMSE sanity
# ---------------------------
def test_rmse_positive(simple_data):
    Y, clusters = simple_data
    Y_df = pd.DataFrame(Y, columns=[f"t{i+1}" for i in range(Y.shape[1])])
    res = SCMEXP(Y_full=Y_df, T0=3, clusters=clusters)
    for rmse in res["rmse_cluster"]:
        assert rmse >= 0 and np.isfinite(rmse)

# ---------------------------
# DataFrame input
# ---------------------------
def test_scmexp_dataframe_input(simple_data):
    Y, clusters = simple_data
    Y_df = pd.DataFrame(Y, columns=[f"t{i+1}" for i in range(Y.shape[1])])
    res = SCMEXP(Y_full=Y_df, T0=3, clusters=clusters)
    assert isinstance(res["df"], pd.DataFrame)



# -------------------------
# _get_per_cluster_param
# -------------------------
def test_get_per_cluster_param_scalar():
    assert _get_per_cluster_param(5, "A") == 5
    assert _get_per_cluster_param(None, "A", default=10) == 10
    assert _get_per_cluster_param({"A": 7}, "A", default=0) == 7
    assert _get_per_cluster_param({"B": 7}, "A", default=0) == 0

# -------------------------
# _prepare_clusters
# -------------------------
def test_prepare_clusters_array_and_dataframe():
    Y = np.array([[1,2],[3,4]])
    clusters = ["X", "Y"]
    Y_np, clust_np, N, labels, K, label_to_k = _prepare_clusters(Y, clusters)
    assert isinstance(Y_np, np.ndarray)
    assert N == 2
    assert K == 2
    assert labels.tolist() == ["X","Y"]
    assert label_to_k == {"X":0, "Y":1}

    df = pd.DataFrame(Y)
    Y_np2, clust_np2, N2, labels2, K2, _ = _prepare_clusters(df, clusters)
    assert np.all(Y_np2 == Y_np)

def test_prepare_clusters_mismatched_length():
    Y = np.array([[1,2],[3,4]])
    clusters = ["X"]
    with pytest.raises(ValueError):
        _prepare_clusters(Y, clusters)

# -------------------------
# _validate_scm_inputs
# -------------------------
def test_validate_scm_inputs_basic_and_design_checks():
    Y = np.zeros((3,5))
    # T0 invalid
    with pytest.raises(ValueError):
        _validate_scm_inputs(Y, 0, 0, "base")
    with pytest.raises(ValueError):
        _validate_scm_inputs(Y, 5, 0, "base")
    # blank_periods invalid
    with pytest.raises(ValueError):
        _validate_scm_inputs(Y, 3, 3, "base")
    # design incompatible parameters
    with pytest.raises(ValueError):
        _validate_scm_inputs(Y, 3, 1, "base", beta=0.1)
    with pytest.raises(ValueError):
        _validate_scm_inputs(Y, 3, 1, "weak", lambda1=0.1)
    with pytest.raises(ValueError):
        _validate_scm_inputs(Y, 3, 1, "base", xi=0.1)

# -------------------------
# _validate_costs_budget
# -------------------------
def test_validate_costs_budget_scalar_and_dict():
    N = 3
    clusters = ["A", "B"]
    costs = [1,2,3]
    budget = 10
    costs_np, budget_dict = _validate_costs_budget(costs, budget, N, clusters, 2)
    assert np.all(costs_np == np.array(costs))
    assert budget_dict == {"A":5, "B":5}

    budget_dict_input = {"A": 6, "B": 4}
    _, bd = _validate_costs_budget(costs, budget_dict_input, N, clusters, 2)
    assert bd == budget_dict_input

def test_validate_costs_budget_errors():
    N = 3
    clusters = ["A", "B"]
    costs = [1,2]
    with pytest.raises(ValueError):
        _validate_costs_budget(costs, 10, N, clusters, 2)
    with pytest.raises(ValueError):
        _validate_costs_budget([1,2,3], None, N, clusters, 2)
    with pytest.raises(TypeError):
        _validate_costs_budget([1,2,3], ["not a dict"], N, clusters, 2)

# -------------------------
# _prepare_fit_slices
# -------------------------
def test_prepare_fit_slices_shapes():
    Y = np.zeros((5,10))
    Y_fit, Y_blank, T_fit = _prepare_fit_slices(Y, 8, 3)
    assert Y_fit.shape[1] == 5
    assert Y_blank.shape[1] == 3
    assert T_fit == 5

# -------------------------
# _build_membership_mask
# -------------------------
def test_build_membership_mask_basic():
    clusters = ["A","B","A"]
    label_to_k = {"A":0,"B":1}
    M = _build_membership_mask(clusters, label_to_k, 3, 2)
    assert M.shape == (3,2)
    assert np.all(M[0] == [True, False])
    assert np.all(M[1] == [False, True])

# -------------------------
# _compute_cluster_means_members
# -------------------------
def test_compute_cluster_means_members_basic():
    Y_fit = np.array([[1,2],[3,4],[5,6]])
    M = np.array([[1,0],[0,1],[1,0]], dtype=bool)
    labels = ["A","B"]
    Xbar, members = _compute_cluster_means_members(Y_fit, M, labels)
    assert len(Xbar) == 2
    assert np.allclose(Xbar[0], np.mean([[1,2],[5,6]], axis=0))
    assert np.allclose(Xbar[1], [3,4])
    assert len(members[0]) == 2
    assert len(members[1]) == 1

# -------------------------
# _precompute_distances
# -------------------------
def test_precompute_distances_shapes():
    Y_fit = np.random.rand(3,4)
    Xbar_clusters = [np.mean(Y_fit[[0,2]], axis=0), np.mean(Y_fit[[1]], axis=0)]
    cluster_members = [np.array([0,2]), np.array([1])]
    D1, D2_list = _precompute_distances(Y_fit, Xbar_clusters, cluster_members)
    assert D1.shape == (3,2)
    assert len(D2_list) == 2
    assert D2_list[0].shape == (2,2)
    assert D2_list[1].shape == (1,1)

# -------------------------
# _init_cvxpy_variables
# -------------------------
def test_init_cvxpy_variables_shapes():
    w, v, z = _init_cvxpy_variables(3,2)
    assert w.shape == (3,2)
    assert v.shape == (3,2)
    assert z.shape == (3,2)

# -------------------------
# _build_constraints
# -------------------------
def test_build_constraints_shapes_and_basic():
    w, v, z = _init_cvxpy_variables(3,2)
    M = np.array([[1,0],[0,1],[1,0]], dtype=bool)
    cluster_members = [np.array([0,2]), np.array([1])]
    labels = ["A","B"]
    constraints = _build_constraints(w,v,z,M,cluster_members,labels,
                                     m_eq=None,m_min=None,m_max=None,
                                     costs=None,budget_dict=None,exclusive=True)
    assert isinstance(constraints, list)
    assert len(constraints) > 0

# -------------------------
# _build_objective
# -------------------------
def test_build_objective_creates_cvxpy_expression():
    Y_fit = np.random.rand(3,4)
    Xbar_clusters = [np.mean(Y_fit[[0,2]], axis=0), np.mean(Y_fit[[1]], axis=0)]
    cluster_members = [np.array([0,2]), np.array([1])]
    w, v, z = _init_cvxpy_variables(3,2)
    obj = _build_objective(Y_fit, Xbar_clusters, cluster_members, w, v, z, design="base")
    assert isinstance(obj, cp.Minimize)


# ---------------------------
# SINGLE-CLUSTER TESTS
# ---------------------------

def test_single_cluster_basic_selection():
    w_opt = np.array([[0.6], [0.3], [0.1]])
    v_opt = np.array([[0.1], [0.3], [0.6]])
    cluster_members = [np.array([0, 1, 2])]
    cluster_labels = ["A"]

    w_d, v_d, sel_treat, sel_ctrl, rmse = _post_hoc_discretize(
        w_opt, v_opt, cluster_members, cluster_labels,
        m_min={"A": 1}, m_max={"A": 2}
    )

    # Treated/control disjoint
    assert set(sel_treat[0]).isdisjoint(sel_ctrl[0])
    # Number of treated within bounds
    assert 1 <= len(sel_treat[0]) <= 2
    # Weights sum to 1 for selected treated
    if len(sel_treat[0]) > 0:
        assert np.isclose(w_d[sel_treat[0], 0].sum(), 1.0)
    # Control weights sum to 1 for selected
    if len(sel_ctrl[0]) > 0:
        assert np.isclose(v_d[sel_ctrl[0], 0].sum(), 1.0)

def test_single_cluster_trim_small_weights():
    w_opt = np.array([[0.01], [0.98], [0.01]])
    v_opt = np.array([[0.5], [0.25], [0.25]])
    cluster_members = [np.array([0, 1, 2])]
    cluster_labels = ["A"]

    w_d, v_d, sel_treat, sel_ctrl, _ = _post_hoc_discretize(
        w_opt, v_opt, cluster_members, cluster_labels,
        m_min={"A": 1}, m_max={"A": 2},
        trim_threshold=0.05
    )

    # Tiny weight (0.01) should be ignored
    assert 0 not in sel_treat[0]  # Index 0 trimmed
    assert 1 in sel_treat[0]

def test_single_cluster_exact_m_eq():
    w_opt = np.array([[0.4], [0.3], [0.3]])
    v_opt = np.array([[0.3], [0.4], [0.3]])
    cluster_members = [np.array([0, 1, 2])]
    cluster_labels = ["A"]

    w_d, v_d, sel_treat, sel_ctrl, _ = _post_hoc_discretize(
        w_opt, v_opt, cluster_members, cluster_labels,
        m_eq={"A": 2}
    )

    assert len(sel_treat[0]) == 2
    # Treated/control disjoint
    assert set(sel_treat[0]).isdisjoint(sel_ctrl[0])

# ---------------------------
# MULTI-CLUSTER TESTS
# ---------------------------

def test_multi_cluster_basic_selection():
    w_opt = np.array([
        [0.6, 0.1],
        [0.4, 0.2],
        [0.0, 0.7],
        [0.0, 0.0]
    ])
    v_opt = np.ones_like(w_opt) * 0.25
    cluster_members = [np.array([0, 1]), np.array([2, 3])]
    cluster_labels = ["A", "B"]

    w_d, v_d, sel_treat, sel_ctrl, _ = _post_hoc_discretize(
        w_opt, v_opt, cluster_members, cluster_labels,
        m_min={"A": 1, "B": 1}, 
        m_max={"A": 2, "B": 2}
    )

    for k_idx in range(2):
        # Number of treated within bounds
        assert 1 <= len(sel_treat[k_idx]) <= 2
        # Treated/control disjoint
        assert set(sel_treat[k_idx]).isdisjoint(sel_ctrl[k_idx])
        # Weights sum to 1 for selected treated
        if len(sel_treat[k_idx]) > 0:
            assert np.isclose(w_d[sel_treat[k_idx], k_idx].sum(), 1.0)
        # Weights sum to 1 for control
        if len(sel_ctrl[k_idx]) > 0:
            assert np.isclose(v_d[sel_ctrl[k_idx], k_idx].sum(), 1.0)

def test_multi_cluster_respects_m_eq():
    w_opt = np.array([
        [0.6, 0.1],
        [0.4, 0.2],
        [0.0, 0.7],
        [0.0, 0.0]
    ])
    v_opt = np.ones_like(w_opt) * 0.25
    cluster_members = [np.array([0, 1]), np.array([2, 3])]
    cluster_labels = ["A", "B"]

    w_d, v_d, sel_treat, sel_ctrl, _ = _post_hoc_discretize(
        w_opt, v_opt, cluster_members, cluster_labels,
        m_eq={"A": 2, "B": 1}
    )

    assert len(sel_treat[0]) == 2
    assert len(sel_treat[1]) == 1
    # Treated/control disjoint
    for k_idx in range(2):
        assert set(sel_treat[k_idx]).isdisjoint(sel_ctrl[k_idx])

# ---------------------------
# EDGE CASES
# ---------------------------

def test_empty_w_nonzero_min():
    w_opt = np.zeros((3, 1))
    v_opt = np.ones((3, 1)) / 3
    cluster_members = [np.array([0, 1, 2])]
    cluster_labels = ["A"]

    w_d, v_d, sel_treat, sel_ctrl, _ = _post_hoc_discretize(
        w_opt, v_opt, cluster_members, cluster_labels,
        m_min={"A": 2}, m_max={"A": 2}
    )

    assert len(sel_treat[0]) == 2  # fallback uses m_min
    # Control weights sum to 1
    if len(sel_ctrl[0]) > 0:
        assert np.isclose(v_d[sel_ctrl[0], 0].sum(), 1.0)

# -------------------------------
# Fixtures for multi-cluster testing
# -------------------------------
@pytest.fixture
def simple_clusters():
    # 6 units in 2 clusters
    cluster_members = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    cluster_labels = ["A", "B"]
    return cluster_members, cluster_labels

@pytest.fixture
def simple_weights(simple_clusters):
    cluster_members, cluster_labels = simple_clusters
    N = 6
    K = len(cluster_labels)
    # Create relaxed weights
    w_opt = np.zeros((N, K))
    v_opt = np.zeros((N, K))
    w_opt[0,0] = 0.6
    w_opt[1,0] = 0.3
    w_opt[2,0] = 0.05
    w_opt[3,1] = 0.7
    w_opt[4,1] = 0.2
    w_opt[5,1] = 0.05

    v_opt[0,0] = 0.2
    v_opt[1,0] = 0.5
    v_opt[2,0] = 0.3
    v_opt[3,1] = 0.3
    v_opt[4,1] = 0.4
    v_opt[5,1] = 0.3

    return w_opt, v_opt, cluster_members, cluster_labels

@pytest.fixture
def blank_periods_matrix():
    # Simple blank periods for RMSE
    Y_blank = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [1.5, 2.5],
        [2.0, 1.0],
        [1.0, 0.5],
        [0.5, 1.0]
    ])
    return Y_blank

# -------------------------------
# Tests
# -------------------------------

def test_basic_posthoc_selection(simple_weights):
    w_opt, v_opt, cluster_members, cluster_labels = simple_weights
    w_discrete, v_discrete, sel_treat, sel_control, rmse_blank = _post_hoc_discretize(
        w_opt, v_opt, cluster_members, cluster_labels, trim_threshold=0.05
    )
    # Weights sum to 1 per cluster for treated
    for k_idx, members in enumerate(cluster_members):
        assert np.isclose(w_discrete[members, k_idx].sum(), 1.0)
        # Control weights sum to 1 if any controls
        if len(sel_control[k_idx]) > 0:
            assert np.isclose(v_discrete[members, k_idx].sum(), 1.0)

def test_trim_threshold_behavior(simple_weights):
    w_opt, v_opt, cluster_members, cluster_labels = simple_weights
    # Use high trim threshold to zero out some weights
    w_discrete, _, sel_treat, _, _ = _post_hoc_discretize(
        w_opt, v_opt, cluster_members, cluster_labels, trim_threshold=0.5
    )
    # Only the largest weight should be selected per cluster
    for k_idx in range(len(cluster_labels)):
        assert len(sel_treat[k_idx]) == 1

def test_zero_weights_fallback(simple_clusters):
    cluster_members, cluster_labels = simple_clusters
    N = 6; K = len(cluster_labels)
    # All zeros
    w_opt = np.zeros((N, K))
    v_opt = np.zeros((N, K))
    w_discrete, v_discrete, sel_treat, sel_control, _ = _post_hoc_discretize(
        w_opt, v_opt, cluster_members, cluster_labels
    )
    for k_idx, members in enumerate(cluster_members):
        # Should fallback to uniform selection of first member
        assert len(sel_treat[k_idx]) == 1
        assert np.isclose(w_discrete[members, k_idx].sum(), 1.0)

def test_m_eq_m_min_m_max_bounds(simple_weights):
    w_opt, v_opt, cluster_members, cluster_labels = simple_weights
    m_eq = {"A": 2, "B": 1}
    m_min = {"A": 1, "B": 1}
    m_max = {"A": 3, "B": 2}
    w_discrete, _, sel_treat, _, _ = _post_hoc_discretize(
        w_opt, v_opt, cluster_members, cluster_labels,
        m_eq=m_eq, m_min=m_min, m_max=m_max
    )
    assert len(sel_treat[0]) == 2  # A
    assert len(sel_treat[1]) == 1  # B

def test_multi_cluster_selection(simple_weights):
    w_opt, v_opt, cluster_members, cluster_labels = simple_weights
    w_discrete, v_discrete, sel_treat, sel_control, _ = _post_hoc_discretize(
        w_opt, v_opt, cluster_members, cluster_labels
    )
    # Each selected unit should belong to its cluster
    for k_idx, members in enumerate(cluster_members):
        for idx in sel_treat[k_idx]:
            assert idx in members
        for idx in sel_control[k_idx]:
            assert idx in members

def test_rmse_blank_computation(simple_weights, blank_periods_matrix):
    w_opt, v_opt, cluster_members, cluster_labels = simple_weights
    Y_blank = blank_periods_matrix
    _, _, sel_treat, sel_control, rmse_blank = _post_hoc_discretize(
        w_opt, v_opt, cluster_members, cluster_labels, Y_blank=Y_blank
    )
    # RMSE should be numeric for clusters with treated and control
    for k_idx in range(len(cluster_labels)):
        if len(sel_treat[k_idx]) > 0 and len(sel_control[k_idx]) > 0:
            assert rmse_blank[k_idx] is not None
            assert rmse_blank[k_idx] >= 0.0
        else:
            assert rmse_blank[k_idx] is None

def test_single_member_cluster():
    w_opt = np.array([[1.0], [0.0]])
    v_opt = np.array([[0.0], [1.0]])
    cluster_members = [np.array([0, 1])]
    cluster_labels = ["X"]
    w_discrete, v_discrete, sel_treat, sel_control, rmse_blank = _post_hoc_discretize(
        w_opt, v_opt, cluster_members, cluster_labels
    )
    # Only one treated selected
    assert len(sel_treat[0]) == 1
    # Weights sum to 1
    assert np.isclose(w_discrete.sum(), 1.0)

def test_all_zero_weights_multi_cluster():
    w_opt = np.zeros((4, 2))
    v_opt = np.zeros((4, 2))
    cluster_members = [np.array([0,1]), np.array([2,3])]
    cluster_labels = ["A", "B"]
    w_discrete, v_discrete, sel_treat, sel_control, _ = _post_hoc_discretize(
        w_opt, v_opt, cluster_members, cluster_labels
    )
    for k_idx in range(2):
        assert len(sel_treat[k_idx]) == 1
        assert np.isclose(w_discrete[cluster_members[k_idx], k_idx].sum(), 1.0)


def test_scmexp_rel_basic_int_clusters():
    # --- Tiny synthetic dataset ---
    # 4 units, 3 time periods
    Y = pd.DataFrame({
        'unit1': [1, 2, 3],
        'unit2': [2, 1, 0],
        'unit3': [0, 1, 2],
        'unit4': [1, 0, 1]
    }).T  # units x time

    clusters = np.array([0, 0, 1, 1])  # integer cluster labels

    # --- Run relaxed SCM ---
    res = SCMEXP_REL(
        Y_full=Y,
        T0=2,  # first 2 periods pre-treatment
        clusters=clusters,
        blank_periods=1,
        m_min=1,
        m_max=2,
        design='eq11',
        zeta=0.0
    )

    # --- Assertions ---
    # Check dimensions
    assert res['w_opt'].shape == (Y.shape[0], len(np.unique(clusters))), "w_opt shape mismatch"
    assert res['v_opt'].shape == (Y.shape[0], len(np.unique(clusters))), "v_opt shape mismatch"
    assert len(res['selected_treated']) == 2, "Number of clusters mismatch"

    # Check that at least one treated unit per cluster is selected
    for sel in res['selected_treated']:
        assert len(sel) >= 1, "No treated units selected in cluster"

    print("Minimal relaxed SCM test passed.")

# Run the test
test_scmexp_rel_basic_int_clusters()



def test_scmexp_rel_basic_shapes():
    """Check that SCMEXP_REL returns expected keys and shapes for small input."""
    N, T, K = 6, 10, 2
    Y_full = np.random.randn(N, T)
    clusters = np.array([0, 0, 0, 1, 1, 1])
    
    result = SCMEXP_REL(Y_full, T0=5, clusters=clusters, blank_periods=2, verbose=False)
    
    # Expected keys
    keys = [
        "w_opt_rel", "v_opt_rel", "z_opt_rel", "w_opt", "v_opt",
        "selected_treated", "selected_control", "y_syn_treated_clusters",
        "y_syn_control_clusters", "Xbar_clusters", "cluster_labels",
        "cluster_members", "w_agg", "v_agg", "cluster_sizes",
        "T0", "blank_periods", "T_fit", "Y_fit", "Y_blank",
        "rmse_cluster", "rmse_blank", "design", "beta", "lambda1",
        "lambda2", "xi", "zeta", "original_cluster_vector"
    ]
    for k in keys:
        assert k in result

    # Shapes sanity checks
    assert result["w_opt_rel"].shape == (N, K)
    assert result["v_opt_rel"].shape == (N, K)
    assert result["w_opt"].shape == (N, K)
    assert result["v_opt"].shape == (N, K)
    assert len(result["cluster_members"]) == K
    assert result["w_agg"].shape[0] == N
    assert result["v_agg"].shape[0] == N

def test_scmexp_rel_zero_budget():
    """Check that the function handles zero budget."""
    N, T, K = 4, 8, 2
    Y_full = np.random.randn(N, T)
    clusters = np.array([0, 0, 1, 1])
    
    result = SCMEXP_REL(Y_full, T0=4, clusters=clusters, costs=np.ones(N), budget=0)
    
    # Should select no units
    for selected in result["selected_treated"]:
        assert len(selected) == 0

def test_scmexp_rel_empty_clusters():
    """Check that the function raises if a cluster has no members."""
    N, T = 5, 10
    Y_full = np.random.randn(N, T)
    clusters = np.array([0, 0, 1, 2, 3])  # K=4, cluster 3 has 1 member, but let's remove it
    clusters[4] = -1  # Invalid cluster
    
    import pytest
    with pytest.raises(ValueError):
        SCMEXP_REL(Y_full, T0=5, clusters=clusters)

def test_scmexp_rel_posthoc_nonnegative_weights():
    """Check that post-hoc discretization produces non-negative weights."""
    N, T, K = 6, 12, 2
    Y_full = np.random.randn(N, T)
    clusters = np.array([0, 0, 0, 1, 1, 1])
    
    result = SCMEXP_REL(Y_full, T0=6, clusters=clusters)
    
    assert np.all(result["w_opt"] >= 0)
    assert np.all(result["v_opt"] >= 0)

def test_scmexp_rel_relaxed_vs_posthoc_consistency():
    """Check that relaxed weights are reasonably related to post-hoc weights."""
    N, T, K = 8, 10, 2
    Y_full = np.random.randn(N, T)
    clusters = np.array([0]*4 + [1]*4)
    
    result = SCMEXP_REL(Y_full, T0=5, clusters=clusters)
    
    w_rel = result["w_opt_rel"]
    w_post = result["w_opt"]
    
    # relaxed weights should be > 0 somewhere in each cluster
    for k in range(K):
        assert np.any(w_rel[:, k] > 0)
        # Post-hoc should be <= relaxed weights (since trimmed)
        assert np.all(w_post[:, k] <= w_rel[:, k] + 1e-12)




# Inference 

def test_global_inference_shapes():
    np.random.seed(123)
    N_units = 10
    T_total = 20
    T_post = 5
    Y_full = np.random.randn(N_units, T_total)
    Y_fit = Y_full[:, :15]
    Y_blank = np.random.randn(N_units, 3)
    w_agg = np.random.rand(N_units)
    v_agg = np.random.rand(N_units)

    result = {
        "T0": 15,
        "w_agg": w_agg,
        "v_agg": v_agg,
        "w_opt": w_agg[:, None],
        "v_opt": v_agg[:, None],
        "Y_fit": Y_fit,
        "Y_blank": Y_blank,
        "rmse_cluster": np.array([0.5])
    }

    out = inference_scm_vectorized(result, Y_full, T_post)
    assert out["tau_hat"].shape == (T_post,)
    assert out["tau_hat_cluster"].shape == (1, T_post)
    assert out["ci_lower"].shape == (T_post,)
    assert out["ci_lower_cluster"].shape == (1, T_post)
    assert out["p_values"].shape == (T_post,)
    assert out["p_values_cluster"].shape == (1, T_post)

def test_compute_placebo_ci_vectorized_global():
    np.random.seed(42)
    tau_hat = np.array([1.0, 2.0])
    Y_blank_T = np.random.randn(5, 3)
    w = np.array([0.2, 0.5, 0.3])
    v = np.array([0.1, 0.6, 0.3])
    rmspe_pre = 0.5

    ci_lower, ci_upper, p_values = _compute_placebo_ci_vectorized(tau_hat, Y_blank_T, w, v, rmspe_pre)
    assert ci_lower.shape == tau_hat.shape
    assert ci_upper.shape == tau_hat.shape
    assert p_values.shape == tau_hat.shape

def test_compute_placebo_ci_vectorized_cluster():
    np.random.seed(42)
    tau_hat = np.array([[1.0, 2.0], [0.5, 1.5]])
    Y_blank_T = np.random.randn(5, 4)
    w = np.random.rand(4, 2)
    v = np.random.rand(4, 2)
    rmspe_pre = 1.0
    rmse_cluster = np.array([0.5, 0.8])

    ci_lower, ci_upper, p_values = _compute_placebo_ci_vectorized(tau_hat, Y_blank_T, w, v, rmspe_pre, rmse_cluster)
    assert ci_lower.shape == tau_hat.shape
    assert ci_upper.shape == tau_hat.shape
    assert p_values.shape == tau_hat.shape


def make_dummy_data(N=5, T0=10, T_post=3, K=2, blank_periods=4, seed=123):
    np.random.seed(seed)
    Y_full = np.random.rand(N, T0 + T_post)
    Y_fit = np.random.rand(N, T0)
    Y_blank = np.random.rand(N, blank_periods)
    w_agg = np.random.rand(N)
    w_agg /= w_agg.sum()
    v_agg = np.random.rand(N)
    v_agg /= v_agg.sum()
    w_opt = np.random.rand(N, K)
    w_opt /= w_opt.sum(axis=0)
    v_opt = np.random.rand(N, K)
    v_opt /= v_opt.sum(axis=0)
    rmse_cluster = np.random.rand(K) * 0.1 + 0.05
    return Y_full, Y_fit, Y_blank, w_agg, v_agg, w_opt, v_opt, rmse_cluster

def test_global_output_shapes():
    Y_full, Y_fit, Y_blank, w_agg, v_agg, w_opt, v_opt, rmse_cluster = make_dummy_data()
    result = {
        "T0": 10,
        "w_agg": w_agg,
        "v_agg": v_agg,
        "w_opt": w_opt,
        "v_opt": v_opt,
        "Y_fit": Y_fit,
        "Y_blank": Y_blank,
        "rmse_cluster": rmse_cluster
    }
    T_post = 3
    output = inference_scm_vectorized(result, Y_full, T_post)
    assert output["tau_hat"].shape == (T_post,)
    assert output["tau_hat_cluster"].shape == (w_opt.shape[1], T_post)
    assert output["ci_lower"].shape == (T_post,)
    assert output["ci_lower_cluster"].shape == (w_opt.shape[1], T_post)
    assert output["p_values"].shape == (T_post,)
    assert output["p_values_cluster"].shape == (w_opt.shape[1], T_post)

def test_ci_bounds_order():
    Y_full, Y_fit, Y_blank, w_agg, v_agg, w_opt, v_opt, rmse_cluster = make_dummy_data()
    result = {
        "T0": 10,
        "w_agg": w_agg,
        "v_agg": v_agg,
        "w_opt": w_opt,
        "v_opt": v_opt,
        "Y_fit": Y_fit,
        "Y_blank": Y_blank,
        "rmse_cluster": rmse_cluster
    }
    output = inference_scm_vectorized(result, Y_full, T_post=3)
    # CI lower <= tau <= CI upper
    assert np.all(output["ci_lower"] <= output["tau_hat"])
    assert np.all(output["ci_upper"] >= output["tau_hat"])
    assert np.all(output["ci_lower_cluster"] <= output["tau_hat_cluster"])
    assert np.all(output["ci_upper_cluster"] >= output["tau_hat_cluster"])

def test_p_values_between_0_1():
    Y_full, Y_fit, Y_blank, w_agg, v_agg, w_opt, v_opt, rmse_cluster = make_dummy_data()
    result = {
        "T0": 10,
        "w_agg": w_agg,
        "v_agg": v_agg,
        "w_opt": w_opt,
        "v_opt": v_opt,
        "Y_fit": Y_fit,
        "Y_blank": Y_blank,
        "rmse_cluster": rmse_cluster
    }
    output = inference_scm_vectorized(result, Y_full, T_post=3)
    assert np.all((output["p_values"] >= 0) & (output["p_values"] <= 1))
    assert np.all((output["p_values_cluster"] >= 0) & (output["p_values_cluster"] <= 1))

def test_edge_case_zero_rmspe():
    # Pre-RMSPE = 0
    Y_full, Y_fit, Y_blank, w_agg, v_agg, w_opt, v_opt, rmse_cluster = make_dummy_data()
    result = {
        "T0": 10,
        "w_agg": w_agg,
        "v_agg": w_agg.copy(),  # identical weights -> zero RMSPE
        "w_opt": w_opt,
        "v_opt": w_opt.copy(),
        "Y_fit": Y_fit,
        "Y_blank": Y_blank,
        "rmse_cluster": rmse_cluster
    }
    output = inference_scm_vectorized(result, Y_full, T_post=3)
    # CI should still be finite
    assert np.all(np.isfinite(output["ci_lower"]))
    assert np.all(np.isfinite(output["ci_upper"]))

def test_invalid_method():
    Y_full, Y_fit, Y_blank, w_agg, v_agg, w_opt, v_opt, rmse_cluster = make_dummy_data()
    result = {
        "T0": 10,
        "w_agg": w_agg,
        "v_agg": v_agg,
        "w_opt": w_opt,
        "v_opt": v_opt,
        "Y_fit": Y_fit,
        "Y_blank": Y_blank,
        "rmse_cluster": rmse_cluster
    }
    with pytest.raises(ValueError):
        inference_scm_vectorized(result, Y_full, T_post=3, method="invalid")


def test_ci_widens_with_blank_variance():
    Y_full, Y_fit, Y_blank, w_agg, v_agg, w_opt, v_opt, rmse_cluster = make_dummy_data()
    result = {
        "T0": 10,
        "w_agg": w_agg,
        "v_agg": v_agg,
        "w_opt": w_opt,
        "v_opt": v_opt,
        "Y_fit": Y_fit,
        "Y_blank": Y_blank,
        "rmse_cluster": rmse_cluster
    }
    output1 = inference_scm_vectorized(result, Y_full, T_post=3)
    
    # Inflate Y_blank variance
    Y_blank_high_var = Y_blank * 10
    result["Y_blank"] = Y_blank_high_var
    output2 = inference_scm_vectorized(result, Y_full, T_post=3)
    
    assert np.all(output2["ci_upper"] - output2["ci_lower"] > output1["ci_upper"] - output1["ci_lower"])
def test_zero_rmse_cluster_handling():
    Y_full, Y_fit, Y_blank, w_agg, v_agg, w_opt, v_opt, rmse_cluster = make_dummy_data(K=3)
    rmse_cluster[1] = 0.0
    result = {
        "T0": 10,
        "w_agg": w_agg,
        "v_agg": v_agg,
        "w_opt": w_opt,
        "v_opt": v_opt,
        "Y_fit": Y_fit,
        "Y_blank": Y_blank,
        "rmse_cluster": rmse_cluster
    }
    output = inference_scm_vectorized(result, Y_full, T_post=3)
    # CI should remain finite
    assert np.all(np.isfinite(output["ci_lower_cluster"]))
    assert np.all(np.isfinite(output["ci_upper_cluster"]))
def test_p_value_monotonicity():
    # Simple 1-cluster setup
    Y_full = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    Y_fit = Y_full[:, :2]
    Y_blank = np.ones((2, 2))
    w_agg = np.array([0.5, 0.5])
    v_agg = np.array([0.5, 0.0])
    w_opt = np.array([[0.5], [0.5]])
    v_opt = np.array([[0.5], [0.0]])
    rmse_cluster = np.array([1.0])
    result = {
        "T0": 2,
        "w_agg": w_agg,
        "v_agg": v_agg,
        "w_opt": w_opt,
        "v_opt": v_opt,
        "Y_fit": Y_fit,
        "Y_blank": Y_blank,
        "rmse_cluster": rmse_cluster
    }
    output = inference_scm_vectorized(result, Y_full, T_post=2)
    assert np.all(output["p_values"] <= 1.0)
    assert np.all(output["p_values"] >= 0.0)
    assert np.all(output["p_values_cluster"] <= 1.0)
    assert np.all(output["p_values_cluster"] >= 0.0)
def test_all_zero_effects():
    Y_full, Y_fit, Y_blank, w_agg, v_agg, w_opt, v_opt, rmse_cluster = make_dummy_data()
    v_agg = w_agg.copy()
    v_opt = w_opt.copy()
    result = {
        "T0": 10,
        "w_agg": w_agg,
        "v_agg": v_agg,
        "w_opt": w_opt,
        "v_opt": v_opt,
        "Y_fit": Y_fit,
        "Y_blank": Y_blank,
        "rmse_cluster": rmse_cluster
    }
    output = inference_scm_vectorized(result, Y_full, T_post=3)
    np.testing.assert_allclose(output["tau_hat"], 0, atol=1e-10)
    np.testing.assert_allclose(output["tau_hat_cluster"], 0, atol=1e-10)
    assert np.all(output["p_values"] >= 0.5)


@pytest.mark.parametrize("T_post", [0, 1, 5])
def test_post_period_edge_cases(T_post):
    """Test very small T_post values (0 or 1)"""
    N = 4
    K = 2
    T0 = 10
    Y_full = np.random.randn(N, T0 + max(1, T_post))
    w_opt = np.random.rand(N, K)
    v_opt = np.random.rand(N, K)
    w_agg = np.random.rand(N)
    v_agg = np.random.rand(N)
    Y_fit = np.random.randn(N, T0)
    Y_blank = np.random.randn(N, T0)
    rmse_cluster = np.ones(K)

    result = {
        "w_opt": w_opt,
        "v_opt": v_opt,
        "w_agg": w_agg,
        "v_agg": v_agg,
        "Y_fit": Y_fit,
        "Y_blank": Y_blank,
        "rmse_cluster": rmse_cluster,
        "T0": T0
    }

    output = inference_scm_vectorized(result, Y_full, T_post)
    # Should return shapes correctly even for T_post=0 or 1
    assert output["tau_hat"].shape[0] == T_post
    assert output["tau_hat_cluster"].shape[1] == T_post

def test_single_unit_cluster():
    """Test cluster inference when a cluster has only one unit"""
    N = 3
    K = 3
    T0 = 5
    T_post = 2
    Y_full = np.random.randn(N, T0 + T_post)
    w_opt = np.eye(N)  # each unit its own cluster
    v_opt = np.zeros((N, N))
    w_agg = np.ones(N)/N
    v_agg = np.zeros(N)
    Y_fit = np.random.randn(N, T0)
    Y_blank = np.random.randn(N, T0)
    rmse_cluster = np.ones(N)

    result = {
        "w_opt": w_opt,
        "v_opt": v_opt,
        "w_agg": w_agg,
        "v_agg": v_agg,
        "Y_fit": Y_fit,
        "Y_blank": Y_blank,
        "rmse_cluster": rmse_cluster,
        "T0": T0
    }

    output = inference_scm_vectorized(result, Y_full, T_post)
    assert np.all(np.isfinite(output["ci_lower_cluster"]))
    assert np.all(np.isfinite(output["ci_upper_cluster"]))

def test_nan_in_blank_periods():
    """Ensure inference handles NaNs in Y_blank gracefully"""
    N = 4
    K = 2
    T0 = 10
    T_post = 3
    Y_full = np.random.randn(N, T0 + T_post)
    w_opt = np.random.rand(N, K)
    v_opt = np.random.rand(N, K)
    w_agg = np.random.rand(N)
    v_agg = np.random.rand(N)
    Y_fit = np.random.randn(N, T0)
    Y_blank = np.random.randn(N, T0)
    Y_blank[0, 1] = np.nan
    rmse_cluster = np.ones(K)

    result = {
        "w_opt": w_opt,
        "v_opt": v_opt,
        "w_agg": w_agg,
        "v_agg": v_agg,
        "Y_fit": Y_fit,
        "Y_blank": Y_blank,
        "rmse_cluster": rmse_cluster,
        "T0": T0
    }

    with pytest.raises(ValueError):
        # Could also modify the function to handle NaNs, but currently should error
        inference_scm_vectorized(result, Y_full, T_post)

def test_extreme_values_scaling():
    """Check that extremely large values do not break CI computation"""
    N = 5
    K = 2
    T0 = 10
    T_post = 2
    Y_full = np.random.randn(N, T0 + T_post) * 1e6
    w_opt = np.random.rand(N, K)
    v_opt = np.random.rand(N, K)
    w_agg = np.random.rand(N)
    v_agg = np.random.rand(N)
    Y_fit = np.random.randn(N, T0) * 1e6
    Y_blank = np.random.randn(N, T0) * 1e6
    rmse_cluster = np.ones(K)

    result = {
        "w_opt": w_opt,
        "v_opt": v_opt,
        "w_agg": w_agg,
        "v_agg": v_agg,
        "Y_fit": Y_fit,
        "Y_blank": Y_blank,
        "rmse_cluster": rmse_cluster,
        "T0": T0
    }

    output = inference_scm_vectorized(result, Y_full, T_post)
    assert np.all(np.isfinite(output["ci_lower"]))
    assert np.all(np.isfinite(output["ci_upper"]))
    assert np.all(np.isfinite(output["ci_lower_cluster"]))
    assert np.all(np.isfinite(output["ci_upper_cluster"]))
