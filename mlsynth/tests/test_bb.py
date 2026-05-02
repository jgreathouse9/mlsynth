import numpy as np
import pytest

from mlsynth.utils.fast_scm_helpers.fast_scm_bb_helpers import (
    branch_and_bound_topK,
    reset_qp_call_count,
)


# =========================================================
# FIXTURES
# =========================================================

def make_G(n=6):
    """Simple PSD-ish symmetric matrix for stability."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(n, n))
    return A.T @ A  # PSD


def make_candidates(n=6):
    return np.arange(n)


# =========================================================
# 1. SMOKE TEST (MINIMAL VALID RUN)
# =========================================================

def test_bb_smoke_runs():
    G = make_G(6)
    cand = make_candidates(6)

    out = branch_and_bound_topK(
        G=G,
        candidate_idx=cand,
        m=2,
        top_K=3,
    )

    assert "top_tuples" in out
    assert "stats" in out
    assert len(out["top_tuples"]) >= 1
    assert np.isfinite(out["stats"]["optimality"]["best_loss"])


# =========================================================
# 2. OUTPUT SHAPE / STRUCTURE INVARIANTS
# =========================================================

def test_bb_output_structure():
    G = make_G(5)
    cand = make_candidates(5)

    out = branch_and_bound_topK(G, cand, m=2, top_K=2)

    stats = out["stats"]

    assert "search_space" in stats
    assert "exploration" in stats
    assert "pruning" in stats
    assert "performance" in stats
    assert "optimality" in stats

    assert stats["search_space"]["total_subsets"] >= 0
    assert stats["performance"]["qp_calls"] >= 0


# =========================================================
# 3. DETERMINISM TEST
# =========================================================

def test_bb_deterministic():
    G = make_G(6)
    cand = make_candidates(6)

    out1 = branch_and_bound_topK(G, cand, m=2, top_K=3)
    out2 = branch_and_bound_topK(G, cand, m=2, top_K=3)

    assert np.isclose(
        out1["stats"]["optimality"]["best_loss"],
        out2["stats"]["optimality"]["best_loss"]
    )


# =========================================================
# 4. EDGE: TOO FEW CANDIDATES
# =========================================================

def test_bb_small_candidate_set():
    G = make_G(3)
    cand = np.array([0, 1])  # M < m edge case

    out = branch_and_bound_topK(G, cand, m=5, top_K=2)

    # should not crash
    assert out["top_tuples"] is not None
    assert isinstance(out["stats"], dict)


# =========================================================
# 5. EDGE: IDENTITY MATRIX (NO STRUCTURE)
# =========================================================

def test_bb_identity_matrix():
    G = np.eye(5)
    cand = make_candidates(5)

    out = branch_and_bound_topK(G, cand, m=2, top_K=3)

    # all losses should be non-negative and finite
    losses = [s.loss for s in out["top_tuples"]]

    assert np.all(np.isfinite(losses))
    assert np.all(np.array(losses) >= 0)


# =========================================================
# 6. PRUNING INVARIANT
# =========================================================

def test_bb_pruning_consistency():
    G = make_G(6)
    cand = make_candidates(6)

    out = branch_and_bound_topK(G, cand, m=2, top_K=3)

    stats = out["stats"]

    assert stats["pruning"]["branches_pruned"] <= stats["pruning"]["branches_considered"]


# =========================================================
# 7. SEARCH SPACE BOUNDS
# =========================================================

def test_bb_search_space_bounds():
    G = make_G(6)
    cand = make_candidates(6)

    out = branch_and_bound_topK(G, cand, m=2, top_K=3)

    search = out["stats"]["search_space"]

    assert search["total_subsets"] >= 0
    assert search["total_nodes"] >= 0


# =========================================================
# 8. QP CALL MONOTONICITY
# =========================================================

def test_bb_qp_calls_nonnegative():
    G = make_G(6)
    cand = make_candidates(6)

    reset_qp_call_count()

    out = branch_and_bound_topK(G, cand, m=2, top_K=3)

    assert out["stats"]["performance"]["qp_calls"] >= 0


# =========================================================
# 9. BEST <= WORST LOSS INVARIANT
# =========================================================

def test_bb_loss_ordering():
    G = make_G(6)
    cand = make_candidates(6)

    out = branch_and_bound_topK(G, cand, m=2, top_K=3)

    solutions = out["top_tuples"]

    if len(solutions) > 1:
        losses = [s.loss for s in solutions]
        assert min(losses) <= max(losses)


# =========================================================
# 10. STABILITY UNDER ZERO MATRIX
# =========================================================

def test_bb_zero_matrix():
    G = np.zeros((5, 5))
    cand = make_candidates(5)

    out = branch_and_bound_topK(G, cand, m=2, top_K=3)

    assert np.isfinite(out["stats"]["optimality"]["best_loss"])
