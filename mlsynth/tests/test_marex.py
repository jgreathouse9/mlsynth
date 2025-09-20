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
    _build_objective
)
from mlsynth.utils.exprelutils import _post_hoc_discretize



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


def test_basic_discretization_with_eq():
    # 1 cluster, 4 members
    w_opt = np.array([[0.6], [0.3], [0.05], [0.05]])
    v_opt = np.ones_like(w_opt) * 0.25
    cluster_members = [np.array([0, 1, 2, 3])]
    cluster_labels = ["A"]

    # Force exactly 2 treated units
    w_d, v_d, sel_treat, sel_ctrl, rmse = _post_hoc_discretize(
        w_opt, v_opt, cluster_members, cluster_labels, m_eq=2
    )

    # Treated: top 2 weights (indices 0,1)
    assert set(sel_treat[0]) == {0, 1}
    assert np.isclose(w_d[0,0] + w_d[1,0], 1.0)
    # Controls: remaining
    assert set(sel_ctrl[0]) == {2, 3}
    assert np.isclose(v_d[2,0] + v_d[3,0], 1.0)
    # RMSE not computed
    assert rmse[0] is None

def test_min_max_bounds_enforced():
    w_opt = np.array([[0.9], [0.05], [0.05]])
    v_opt = np.ones_like(w_opt) / 3
    cluster_members = [np.array([0, 1, 2])]
    cluster_labels = ["B"]

    # Require at least 2 treated units
    w_d, v_d, sel_treat, sel_ctrl, _ = _post_hoc_discretize(
        w_opt, v_opt, cluster_members, cluster_labels, m_min=2
    )

    assert len(sel_treat[0]) == 2
    assert np.isclose(w_d[:,0].sum(), 1.0)

    # Now require at most 1 treated unit
    w_d, v_d, sel_treat, sel_ctrl, _ = _post_hoc_discretize(
        w_opt, v_opt, cluster_members, cluster_labels, m_max=1
    )
    assert len(sel_treat[0]) == 1
    assert np.isclose(w_d[:,0].sum(), 1.0)

def test_trim_and_fallback_uniform():
    w_opt = np.array([[1e-6], [1e-6], [1e-6]])
    v_opt = np.ones_like(w_opt) / 3
    cluster_members = [np.array([0, 1, 2])]
    cluster_labels = ["C"]

    w_d, v_d, sel_treat, sel_ctrl, _ = _post_hoc_discretize(
        w_opt, v_opt, cluster_members, cluster_labels, m_eq=1, trim_threshold=1e-2
    )

    # With trimming, all original weights vanish, fallback uniform applies
    assert len(sel_treat[0]) == 1
    assert np.isclose(w_d[:,0].sum(), 1.0)

def test_blank_rmse_computed():
    w_opt = np.array([[0.7], [0.3]])
    v_opt = np.ones_like(w_opt) * 0.5
    cluster_members = [np.array([0, 1])]
    cluster_labels = ["D"]

    Y_blank = np.array([[1, 2, 3], [2, 3, 4]])  # treated=unit0, control=unit1

    w_d, v_d, sel_treat, sel_ctrl, rmse = _post_hoc_discretize(
        w_opt, v_opt, cluster_members, cluster_labels, m_eq=1, Y_blank=Y_blank
    )

    assert rmse[0] is not None
    assert rmse[0] >= 0

def test_multi_cluster_case():
    # 2 clusters, different labels
    w_opt = np.array([
        [0.8, 0.0],  # cluster X, member 0
        [0.2, 0.0],  # cluster X, member 1
        [0.0, 0.9],  # cluster Y, member 2
        [0.0, 0.1],  # cluster Y, member 3
    ])
    v_opt = np.ones_like(w_opt) * 0.25
    cluster_members = [np.array([0, 1]), np.array([2, 3])]
    cluster_labels = ["X", "Y"]

    w_d, v_d, sel_treat, sel_ctrl, rmse = _post_hoc_discretize(
        w_opt, v_opt, cluster_members, cluster_labels, m_eq={"X": 1, "Y": 2}
    )

    # Cluster X: should pick unit 0
    assert sel_treat[0] == [0]
    assert set(sel_ctrl[0]) == {1}
    assert np.isclose(w_d[:,0].sum(), 1.0)

    # Cluster Y: both units must be treated (m_eq=2)
    assert set(sel_treat[1]) == {2, 3}
    assert sel_ctrl[1] == []
    assert np.isclose(w_d[:,1].sum(), 1.0)

    # Controls in cluster Y vanish since no members left
    assert np.isclose(v_d[:,1].sum(), 0.0)
    assert rmse == [None, None]

