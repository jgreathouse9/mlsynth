import numpy as np
import pytest

from mlsynth.utils.fast_scm_helpers.fast_scm_bb import branch_and_bound_topK
from mlsynth.utils.fast_scm_helpers.fast_scm_setup import IndexSet


# =========================================================
# FIXTURE
# =========================================================

def make_G():
    """Small symmetric PSD-like matrix for deterministic tests."""
    return np.array([
        [1.0, 0.2, 0.1],
        [0.2, 0.9, 0.3],
        [0.1, 0.3, 0.8],
    ])


def make_candidates():
    return np.array([0, 1, 2])


def make_index():
    return IndexSet.from_labels(["A", "B", "C"])


# =========================================================
# SMOKE TEST
# =========================================================

def test_branch_and_bound_runs_smoke():
    """Main goal: ensure algorithm executes without crashing."""
    G = make_G()
    candidate_idx = make_candidates()

    result = branch_and_bound_topK(
        G=G,
        candidate_idx=candidate_idx,
        m=2,
        top_K=3,
    )

    assert "top_tuples" in result
    assert "stats" in result

    assert len(result["top_tuples"]) > 0
    assert isinstance(result["stats"], dict)


# =========================================================
# OUTPUT STRUCTURE TEST
# =========================================================

def test_topk_output_structure():
    G = make_G()

    result = branch_and_bound_topK(
        G=G,
        candidate_idx=make_candidates(),
        m=2,
        top_K=2,
        unit_index=make_index(),
    )

    top = result["top_tuples"][0]

    # Core structure
    assert hasattr(top, "loss")
    assert hasattr(top, "indices")
    assert hasattr(top, "weights")

    # Labels should be attached when unit_index provided
    assert hasattr(top, "labels")
    assert isinstance(top.labels, list)


# =========================================================
# DETERMINISM TEST
# =========================================================

def test_topk_deterministic():
    G = make_G()
    idx = make_candidates()

    r1 = branch_and_bound_topK(G, idx, m=2, top_K=2)
    r2 = branch_and_bound_topK(G, idx, m=2, top_K=2)

    # Loss ordering should be stable
    losses_1 = [t.loss for t in r1["top_tuples"]]
    losses_2 = [t.loss for t in r2["top_tuples"]]

    np.testing.assert_allclose(losses_1, losses_2)


# =========================================================
# PRESOLVE + BUDGET EDGE CASES
# =========================================================

def test_budget_and_lam_execution():
    G = make_G()

    result = branch_and_bound_topK(
        G=G,
        candidate_idx=make_candidates(),
        m=2,
        lam=0.1,
        budget=1.0,
        top_K=2,
    )

    stats = result["stats"]

    assert "performance" in stats
    assert "pruning" in stats
    assert stats["search_space"]["total_subsets"] >= 0


# =========================================================
# UNIT INDEX LABELING
# =========================================================

def test_unit_index_label_mapping():
    G = make_G()

    unit_index = IndexSet.from_labels(["A", "B", "C"])

    result = branch_and_bound_topK(
        G=G,
        candidate_idx=make_candidates(),
        m=2,
        top_K=2,
        unit_index=unit_index,
    )

    top = result["top_tuples"][0]

    # Ensure label mapping exists
    assert isinstance(top.labels, list)
    assert isinstance(top.weight_dict, dict)

    # Check consistency
    for k in top.weight_dict:
        assert k in unit_index.labels


# =========================================================
# EDGE: SMALL SEARCH SPACE
# =========================================================

def test_small_problem_edge_case():
    G = np.array([[1.0]])

    result = branch_and_bound_topK(
        G=G,
        candidate_idx=np.array([0]),
        m=1,
        top_K=1,
    )

    assert len(result["top_tuples"]) == 1
    assert np.isfinite(result["top_tuples"][0].loss)


# =========================================================
# METRICS SANITY CHECK
# =========================================================

def test_stats_structure():
    result = branch_and_bound_topK(
        G=make_G(),
        candidate_idx=make_candidates(),
        m=2,
        top_K=2,
    )

    stats = result["stats"]

    assert "search_space" in stats
    assert "exploration" in stats
    assert "pruning" in stats
    assert "performance" in stats
    assert "optimality" in stats

    assert "total_subsets" in stats["search_space"]



import numpy as np
import pytest

from mlsynth.utils.fast_scm_helpers.fast_scm_bb_helpers import (
    presolve_candidates,
    compute_search_space_size,
    greedy_initial_solution,
    expand_weights_to_full,
    simplex_lower_bound,
    lookahead_lower_bound,
    solve_qp_simplex_value,
    expand_tuple,
    get_qp_call_count,
    reset_qp_call_count,
    clear_solver_cache,
    Solution,
)

# =========================================================
# PRESOLVE
# =========================================================

def test_presolve_budget_filter():
    G = np.eye(4)
    candidate_idx = np.array([0, 1, 2, 3])
    costs = np.array([1, 5, 2, 10])

    out = presolve_candidates(G, candidate_idx, budget=3, unit_costs=costs)

    assert set(out.tolist()) == {0, 2}


def test_presolve_variance_filter():
    G = np.diag([1, 1, 1, 1000])  # huge outlier
    idx = np.array([0, 1, 2, 3])

    out = presolve_candidates(G, idx)

    assert 3 not in out


def test_presolve_collinearity_filter():
    G = np.array([
        [1.0, 0.9999],
        [0.9999, 1.0],
    ])
    idx = np.array([0, 1])
    costs = np.array([1.0, 2.0])  # second is more expensive

    out = presolve_candidates(G, idx, unit_costs=costs)

    assert len(out) == 1
    assert out[0] == 0  # cheaper survives


# =========================================================
# SEARCH SPACE
# =========================================================

def test_search_space_size():
    subsets, nodes = compute_search_space_size(5, 2)

    assert subsets == 10  # C(5,2)
    assert nodes == 5 + 10  # C(5,1) + C(5,2)


# =========================================================
# QP SOLVER
# =========================================================

def test_qp_solver_basic():
    Q = np.eye(2)

    loss, w = solve_qp_simplex_value(Q)

    assert np.isclose(loss, 0.5)
    np.testing.assert_allclose(w.sum(), 1.0)


def test_qp_solver_fallback(monkeypatch):
    def fake_solve(*args, **kwargs):
        class Dummy:
            status = "failed"
            value = None
        return Dummy()

    import cvxpy as cp
    monkeypatch.setattr(cp.Problem, "solve", fake_solve)

    Q = np.eye(2)
    loss, w = solve_qp_simplex_value(Q)

    assert np.isfinite(loss)
    assert np.isclose(w.sum(), 1.0)


# =========================================================
# LOWER BOUNDS
# =========================================================

def test_simplex_lower_bound_basic():
    Q = np.eye(3)
    lb = simplex_lower_bound(Q)

    assert lb >= 0
    assert lb <= 1


def test_lookahead_lower_bound_monotonic():
    G = np.eye(5)

    lb = lookahead_lower_bound(
        G,
        current_lb=0.5,
        remaining_candidates=np.array([2, 3, 4]),
        slots_left=2,
        m=3,
    )

    assert lb >= 0


# =========================================================
# GREEDY INIT
# =========================================================

def test_greedy_initial_solution():
    G = np.eye(4)
    idx = np.array([0, 1, 2, 3])

    loss, sel, w = greedy_initial_solution(G, idx, m=2)

    assert len(sel) == 2
    assert np.isclose(w.sum(), 1.0)


# =========================================================
# WEIGHT EXPANSION
# =========================================================

def test_expand_weights_to_full():
    indices = [1, 3]
    weights = np.array([0.3, 0.7])

    full = expand_weights_to_full(indices, weights, total_units=5)

    assert full.shape == (5,)
    assert np.isclose(full.sum(), 1.0)
    assert full[1] == 0.3
    assert full[3] == 0.7


# =========================================================
# GLOBAL STATE
# =========================================================

def test_qp_call_counter():
    reset_qp_call_count()

    Q = np.eye(2)
    solve_qp_simplex_value(Q)
    solve_qp_simplex_value(Q)

    assert get_qp_call_count() == 2


def test_clear_solver_cache():
    clear_solver_cache()
    assert get_qp_call_count() == 0


# =========================================================
# EXPAND TUPLE (CORE RECURSION)
# =========================================================

def test_expand_tuple_smoke():
    G = np.eye(4)
    candidate_idx = np.array([0, 1, 2, 3])

    stats = {
        "nodes_visited": 0,
        "subsets_evaluated": 0,
        "branches_pruned": 0,
        "branches_considered": 0,
    }

    top = [Solution(loss=10.0, indices=[0], weights=np.array([1.0]))]

    expand_tuple(
        G=G,
        candidate_idx=candidate_idx,
        m=2,
        top_K=3,
        top_tuples=top,
        indices=[0],
        stats=stats,
        start_pos=1,
        Q_partial=np.array([[1.0]]),
    )

    assert stats["nodes_visited"] > 0
    assert len(top) >= 1


def test_expand_tuple_budget_pruning():
    G = np.eye(3)
    idx = np.array([0, 1, 2])
    costs = np.array([1.0, 10.0, 1.0])

    stats = {
        "nodes_visited": 0,
        "subsets_evaluated": 0,
        "branches_pruned": 0,
        "branches_considered": 0,
    }

    top = [Solution(loss=10.0, indices=[0], weights=np.array([1.0]))]

    expand_tuple(
        G=G,
        candidate_idx=idx,
        m=2,
        top_K=2,
        top_tuples=top,
        indices=[0],
        stats=stats,
        start_pos=1,
        Q_partial=np.array([[1.0]]),
        unit_costs=costs,
        budget=1.5,
        current_cost=1.0,
    )

    assert stats["branches_pruned"] > 0


# =========================================================
# NUMERICAL STABILITY
# =========================================================

def test_near_singular_Q():
    Q = np.array([
        [1.0, 0.999999],
        [0.999999, 1.0],
    ])

    loss, w = solve_qp_simplex_value(Q)

    assert np.isfinite(loss)
    assert np.isclose(w.sum(), 1.0)


# =========================================================
# RANDOMIZED STRESS
# =========================================================

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_psd_inputs(seed):
    rng = np.random.default_rng(seed)

    X = rng.normal(size=(5, 5))
    G = X.T @ X  # PSD

    idx = np.arange(5)

    loss, sel, w = greedy_initial_solution(G, idx, m=2)

    assert np.isfinite(loss)
    assert np.isclose(w.sum(), 1.0)
