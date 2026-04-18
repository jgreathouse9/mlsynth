import numpy as np
from typing import List, Tuple
from .fast_scm_bb_helpers import compute_search_space_size, expand_tuple, greedy_initial_solution

import heapq
from typing import List, Tuple

def branch_and_bound_topK(
    G: np.ndarray,
    candidate_idx: np.ndarray,
    m: int = 5,
    top_K: int = 20,
    top_P: int = 10
) -> List[Tuple[float, List[int], np.ndarray]]:
    """
    Perform branch-and-bound search to find the top-K m-unit tuples minimizing quadratic loss.

    Parameters
    ----------
    G : np.ndarray, shape (N, N)
        Symmetric loss (Gram) matrix used for computing tuple losses.
    candidate_idx : np.ndarray, shape (M,)
        Sorted array of candidate unit indices to consider.
    m : int, optional
        Number of units per tuple (default 5).
    top_K : int, optional
        Maximum number of top tuples to return (default 20).
    top_P : int, optional
        Number of initial 1-unit seeds to expand (default 10).

    Returns
    -------
    top_tuples : list of tuples
        List of up to `top_K` tuples, each containing:
        - total_loss : float
            Quadratic loss of the tuple, w^T Q w.
        - indices : list of int
            Indices of the units in the tuple.
        - weights : np.ndarray, shape (m,)
            Optimal weights on the simplex for this tuple.

    Raises
    ------
    ValueError
        If the number of candidate units is smaller than `m`.

    Notes
    -----
    - Starts from top-P 1-unit seeds computed from the diagonal of G.
    - Expands tuples recursively using `expand_tuple`.
    - Uses branch-and-bound pruning: partial tuples with lower-bound losses
      exceeding the current worst top-K loss are skipped.
    - Resulting tuples are sorted by increasing loss.
    """
    if len(candidate_idx) < m:
        raise ValueError(f"Not enough candidate units: {len(candidate_idx)} < m={m}")

    # ✅ 1. Sort candidates by diagonal (best-first search)
    diag_vals = np.diag(G)
    candidate_idx = candidate_idx[np.argsort(diag_vals[candidate_idx])]

    top_tuples: List[Tuple[float, List[int], np.ndarray]] = []

    M = len(candidate_idx)
    total_subsets, total_nodes = compute_search_space_size(M, m)

    stats = {
        # empirical
        "nodes_visited": 0,
        "subsets_evaluated": 0,
        "branches_pruned": 0,

        # theoretical
        "total_subsets": total_subsets,
        "total_nodes": total_nodes,
        "branches_considered": 0,
    }

    # ✅ 2. Strong initial solution (tight UB)
    init_loss, init_idx, init_w = greedy_initial_solution(G, candidate_idx, m)
    top_tuples.append((init_loss, init_idx, init_w))

    num_seeds = min(90, max(20, 4 * m))

    # ✅ 3. Expand from top seeds (can use more than 1 if desired)
    for i in range(num_seeds):
        expand_tuple(
            G,
            candidate_idx,
            m,
            top_K,
            top_tuples,
            indices=[candidate_idx[i]],
            stats=stats
        )


        stats["subset_fraction_explored"] = (
        stats["subsets_evaluated"] / total_subsets
        if total_subsets > 0 else 0.0
    )

    stats["node_fraction_explored"] = (
        stats["nodes_visited"] / total_nodes
        if total_nodes > 0 else 0.0
    )

    stats["prune_rate"] = (
        stats["branches_pruned"] / stats["branches_considered"]
        if stats["branches_considered"] > 0 else 0.0
    )

    stats["speedup_factor"] = (
    stats["total_subsets"] / stats["subsets_evaluated"]
    if stats["subsets_evaluated"] > 0 else np.inf
    )

    return {
        "top_tuples": sorted(top_tuples, key=lambda x: x[0]),
        "stats": stats
    }
