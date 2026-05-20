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

def test_budget_execution():
    G = make_G()

    result = branch_and_bound_topK(
        G=G,
        candidate_idx=make_candidates(),
        m=2,
        budget=1.0,
        unit_costs=np.array([0.4, 0.5, 0.6]),
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


# =========================================================
# LOW-LEVEL HELPER TESTS
# =========================================================

from mlsynth.utils.fast_scm_helpers.fast_scm_bb_helpers import (
    Precomputed,
    Solution,
    clear_cache,
    compute_search_space_size,
    diagonal_bound_Q,
    expand_weights_to_full,
    fw_completion_bound,
    get_qp_call_count,
    greedy_init,
    inverse_rank_bound,
    presolve,
    reset_qp_call_count,
    solve_qp,
    spectral_lower_bound,
)


# =========================================================
# PRESOLVE
# =========================================================

def test_presolve_budget_filter():
    """Budget filter drops units whose cost plus the cheapest partners
    exceeds the budget."""
    G = np.eye(4)
    candidate_idx = np.array([0, 1, 2, 3])
    costs = np.array([1.0, 5.0, 2.0, 10.0])

    pre = Precomputed(G)
    out = presolve(pre, candidate_idx, m=2, budget=3.0, unit_costs=costs)

    # m=2 needs one partner; cheapest partner cost = min(costs[other]) = 1.
    # Affordable picks satisfy unit_cost + 1 <= 3 -> {0, 2}.
    assert set(out.tolist()) == {0, 2}


def test_presolve_variance_filter():
    """Units with diagonal far above the median are dropped."""
    G = np.diag([1.0, 1.0, 1.0, 1000.0])  # huge outlier
    idx = np.array([0, 1, 2, 3])

    pre = Precomputed(G)
    out = presolve(pre, idx, m=2)

    assert 3 not in out


def test_presolve_collinearity_filter():
    """Near-collinear units are deduplicated."""
    G = np.array([
        [1.0, 0.9999],
        [0.9999, 1.0],
    ])
    idx = np.array([0, 1])

    pre = Precomputed(G)
    out = presolve(pre, idx, m=1)

    # Only one of the two near-collinear units should survive.
    assert len(out) == 1


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

    loss, w = solve_qp(Q)

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
    # The solver cache may already hold a result; clear it so the fallback
    # path is the one we hit.
    clear_cache()

    Q = np.eye(2)
    loss, w = solve_qp(Q)

    assert np.isfinite(loss)
    assert np.isclose(w.sum(), 1.0)


# =========================================================
# LOWER BOUNDS (the renamed / refactored set)
# =========================================================

def test_diagonal_bound_basic():
    Q = np.diag([0.5, 1.0, 2.0])
    lb = diagonal_bound_Q(Q)

    assert lb == 0.5


def test_spectral_lower_bound_nonnegative_psd():
    Q = np.eye(3)
    lb = spectral_lower_bound(Q)

    assert lb >= 0


def test_inverse_rank_bound_single_unit():
    Q = np.array([[2.5]])
    lb = inverse_rank_bound(Q)
    assert lb == 2.5


def test_fw_completion_bound_runs():
    G = np.eye(5)
    pre = Precomputed(G)
    lb = fw_completion_bound(pre, indices=np.array([0]), remaining=np.array([1, 2]))

    assert np.isfinite(lb)
    assert lb >= 0


# =========================================================
# GREEDY INIT
# =========================================================

def test_greedy_init_basic():
    G = np.eye(4)
    idx = np.array([0, 1, 2, 3])
    pre = Precomputed(G)

    # greedy_init now requires unit_costs; pass zero-cost as the budget-free
    # path is the common case.
    costs = np.zeros(4)
    sel, loss, w = greedy_init(pre, idx, m=2, unit_costs=costs, budget=1e9)

    assert len(sel) == 2
    assert np.isclose(w.sum(), 1.0)
    assert np.isfinite(loss)


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
    clear_cache()

    Q = np.eye(2)
    solve_qp(Q)
    # Use distinct Q values so the warm-start cache does not bypass solve.
    solve_qp(np.eye(2) * 1.0001)

    assert get_qp_call_count() == 2


def test_clear_cache_resets_counter():
    Q = np.eye(2)
    solve_qp(Q)
    assert get_qp_call_count() >= 1

    clear_cache()
    assert get_qp_call_count() == 0


# =========================================================
# NUMERICAL STABILITY
# =========================================================

def test_near_singular_Q():
    Q = np.array([
        [1.0, 0.999999],
        [0.999999, 1.0],
    ])

    loss, w = solve_qp(Q)

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
    pre = Precomputed(G)
    costs = np.zeros(5)

    sel, loss, w = greedy_init(pre, idx, m=2, unit_costs=costs, budget=1e9)

    assert np.isfinite(loss)
    assert np.isclose(w.sum(), 1.0)


# =========================================================
# BRANCH-AND-BOUND vs BRUTE FORCE CONSISTENCY
# =========================================================

import itertools


def brute_force_best(G, idx, m):
    best = float("inf")
    for comb in itertools.combinations(idx, m):
        Q = G[np.ix_(comb, comb)]
        loss, _ = solve_qp(Q)
        best = min(best, loss)
    return best


def test_matches_bruteforce():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 10))
    G = X.T @ X

    idx = np.arange(10)
    res = branch_and_bound_topK(G, idx, m=2, top_K=1)

    best_bnb = res["top_tuples"][0].loss
    best_true = brute_force_best(G, idx, m=2)

    assert np.isclose(best_bnb, best_true, atol=1e-6)
