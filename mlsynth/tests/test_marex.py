# tests/test_marex.py
import pytest
import pandas as pd
import numpy as np
import cvxpy as cp
from mlsynth import MAREX
from mlsynth.config_models import MAREXConfig
from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError
from pydantic import ValidationError
from mlsynth.utils.exputils import _get_per_cluster_param, SCMEXP, _validate_scm_inputs, _prepare_clusters, _validate_costs_budget

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



# ---------------------------
# Validation Tests
# ---------------------------


def test_validate_scm_inputs():
    Y = np.random.rand(5, 10)  # 5 units, 10 time periods

    # 1) Valid input should not raise
    _validate_scm_inputs(Y, T0=5, blank_periods=1, design="base")
    _validate_scm_inputs(Y, T0=5, blank_periods=0, design="weak", beta=0.1)
    _validate_scm_inputs(Y, T0=5, blank_periods=2, design="eq11", lambda1=0.1, lambda2=0.2)
    _validate_scm_inputs(Y, T0=5, blank_periods=0, design="unit", xi=0.1, lambda1_unit=0.2, lambda2_unit=0.3)

    # 2) Invalid T0
    with pytest.raises(ValueError, match="T0 must be 1 <= T0"):
        _validate_scm_inputs(Y, T0=0, blank_periods=0, design="base")
    with pytest.raises(ValueError, match="T0 must be 1 <= T0"):
        _validate_scm_inputs(Y, T0=10, blank_periods=0, design="base")  # equal to n_periods

    # 3) Invalid blank_periods
    with pytest.raises(ValueError, match="blank_periods must be 0 <= blank_periods"):
        _validate_scm_inputs(Y, T0=5, blank_periods=-1, design="base")
    with pytest.raises(ValueError, match="blank_periods must be 0 <= blank_periods"):
        _validate_scm_inputs(Y, T0=5, blank_periods=5, design="base")  # equal to T0

    # 4) Invalid parameter combos
    with pytest.raises(ValueError, match="beta is only valid"):
        _validate_scm_inputs(Y, T0=5, blank_periods=0, design="base", beta=0.1)
    with pytest.raises(ValueError, match="lambda1/lambda2 are only valid"):
        _validate_scm_inputs(Y, T0=5, blank_periods=0, design="base", lambda1=0.1)
    with pytest.raises(ValueError, match="xi/lambda1_unit/lambda2_unit are only valid"):
        _validate_scm_inputs(Y, T0=5, blank_periods=0, design="base", xi=0.1)

def test_prepare_clusters():
    # --- normal case with DataFrame ---
    Y_df = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
    clusters = ["A", "B", "A"]
    Y_np, clusters_np, N, cluster_labels, K, label_to_k = _prepare_clusters(Y_df, clusters)

    # Checks
    assert isinstance(Y_np, np.ndarray)
    assert Y_np.shape == (3, 2)
    assert isinstance(clusters_np, np.ndarray)
    assert clusters_np.tolist() == clusters
    assert N == 3
    assert set(cluster_labels) == {"A", "B"}
    assert K == 2
    assert label_to_k == {"A": 0, "B": 1} or label_to_k == {"B": 0, "A": 1}

    # --- normal case with ndarray ---
    Y_arr = np.array([[10, 20], [30, 40]])
    clusters_arr = np.array([1, 2])
    Y_np, clusters_np, N, cluster_labels, K, label_to_k = _prepare_clusters(Y_arr, clusters_arr)
    assert isinstance(Y_np, np.ndarray)
    assert N == 2
    assert set(cluster_labels) == {1, 2}

    # --- failure case: cluster length mismatch ---
    clusters_wrong = ["X", "Y", "Z"]
    with pytest.raises(ValueError, match="clusters must have length N"):
        _prepare_clusters(Y_arr, clusters_wrong)

def test_validate_costs_budget():
    N = 4
    cluster_labels = ["A", "B"]
    K = 2

    # --- normal case: scalar budget ---
    costs = [10, 20, 30, 40]
    budget = 100
    costs_np, budget_dict = _validate_costs_budget(costs, budget, N, cluster_labels, K)
    assert isinstance(costs_np, np.ndarray)
    assert costs_np.tolist() == costs
    assert budget_dict == {"A": 50.0, "B": 50.0}

    # --- normal case: dict budget ---
    budget_d = {"A": 60, "B": 40}
    costs_np, budget_dict = _validate_costs_budget(costs, budget_d, N, cluster_labels, K)
    assert budget_dict == budget_d

    # --- error: costs length mismatch ---
    costs_wrong = [1, 2]
    with pytest.raises(ValueError, match="costs must have length N"):
        _validate_costs_budget(costs_wrong, budget, N, cluster_labels, K)

    # --- error: budget missing ---
    with pytest.raises(ValueError, match="budget must be provided"):
        _validate_costs_budget(costs, None, N, cluster_labels, K)

    # --- error: missing cluster in dict budget ---
    budget_bad = {"A": 50}
    with pytest.raises(ValueError, match="budget missing entry for cluster 'B'"):
        _validate_costs_budget(costs, budget_bad, N, cluster_labels, K)


