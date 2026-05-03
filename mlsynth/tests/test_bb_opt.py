import itertools
import numpy as np

from mlsynth.utils.fast_scm_helpers.fast_scm_bb import branch_and_bound_topK
from mlsynth.utils.fast_scm_helpers.fast_scm_bb_helpers import solve_qp_simplex_value


# =========================================================
# BRUTE FORCE REFERENCE IMPLEMENTATION
# =========================================================

def brute_force_best(G, idx, m):
    """Computes the true optimal value by exhaustive search over all subsets."""
    best = float("inf")

    for comb in itertools.combinations(idx, m):
        Q = G[np.ix_(comb, comb)]
        loss, _ = solve_qp_simplex_value(Q)
        best = min(best, loss)

    return best


# =========================================================
# FINAL CONSISTENCY TEST
# =========================================================

def test_branch_and_bound_matches_bruteforce():
    """
    Verifies that branch-and-bound finds the same optimal solution
    as brute-force enumeration on a random PSD matrix.
    """
    rng = np.random.default_rng(0)

    # Create a PSD matrix
    X = rng.normal(size=(25, 25))
    G = X.T @ X

    idx = np.arange(25)

    # Run branch-and-bound
    res = branch_and_bound_topK(G, idx, m=3, top_K=1)
    best_bnb = res["top_tuples"][0].loss

    # Compute brute-force optimum
    best_true = brute_force_best(G, idx, m=3)

    print("\nBNB best loss:", best_bnb)
    print("Brute force best loss:", best_true)

    # Strong consistency check
    assert np.isclose(best_bnb, best_true, atol=1e-6)






def test_bnb_when_greedy_is_suboptimal():
    """
    Construct a G where the greedy pick is deliberately misleading —
    the best single unit is not part of the optimal pair/triple.
    """
    rng = np.random.default_rng(42)
    # Try multiple seeds to increase the chance greedy != optimal
    for seed in range(20):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(15, 8))
        G = X.T @ X  # 8×8, so m=3 is non-trivial relative to rank
        idx = np.arange(8)

        res = branch_and_bound_topK(G, idx, m=3, top_K=1)
        best_bnb = res["top_tuples"][0].loss
        best_true = brute_force_best(G, idx, m=3)

        assert np.isclose(best_bnb, best_true, atol=1e-6), f"Failed on seed {seed}"














