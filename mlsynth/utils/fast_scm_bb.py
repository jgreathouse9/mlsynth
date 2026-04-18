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
):
    if len(candidate_idx) < m:
        raise ValueError(f"Not enough candidate units: {len(candidate_idx)} < m={m}")

    # Sort by diagonal (good heuristic)
    diag_vals = np.diag(G)
    candidate_idx = candidate_idx[np.argsort(diag_vals[candidate_idx])]

    top_tuples = []

    M = len(candidate_idx)
    total_subsets, total_nodes = compute_search_space_size(M, m)

    stats = {
        "nodes_visited": 0,
        "subsets_evaluated": 0,
        "branches_pruned": 0,
        "branches_considered": 0,
        "total_subsets": total_subsets,
        "total_nodes": total_nodes,
    }

    # Initial UB
    init_loss, init_idx, init_w = greedy_initial_solution(G, candidate_idx, m)
    top_tuples.append((init_loss, init_idx, init_w))

    num_seeds = min(50, max(20, 4 * m))

    # 🔥 Correct root initialization
    for i in range(num_seeds):
        j = candidate_idx[i]

        Q0 = np.array([[G[j, j]]])

        expand_tuple(
            G,
            candidate_idx,
            m,
            top_K,
            top_tuples,
            indices=[j],
            stats=stats,
            start_pos=i + 1,
            Q_partial=Q0
        )

    # stats
    stats["subset_fraction_explored"] = (
        stats["subsets_evaluated"] / total_subsets if total_subsets else 0
    )
    stats["node_fraction_explored"] = (
        stats["nodes_visited"] / total_nodes if total_nodes else 0
    )
    stats["prune_rate"] = (
        stats["branches_pruned"] / stats["branches_considered"]
        if stats["branches_considered"] else 0
    )
    stats["speedup_factor"] = (
        stats["total_subsets"] / stats["subsets_evaluated"]
        if stats["subsets_evaluated"] else np.inf
    )

    return {
        "top_tuples": sorted(top_tuples, key=lambda x: x[0]),
        "stats": stats
    }
