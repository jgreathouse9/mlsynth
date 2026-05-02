import numpy as np
import pytest

from mlsynth.utils.fast_scm_helpers.fast_scm_bb import branch_and_bound_topK
from mlsynth.utils.fast_scm_helpers.fast_scm_setup import IndexSet


# =========================================================
# FIXTURE
# =========================================================

def make_G():
    """
    Small symmetric PSD-like matrix for deterministic tests.
    """
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
    """
    Main goal: ensure algorithm executes without crashing.
    """
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
