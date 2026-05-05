import numpy as np
import time
from typing import List, Optional, Dict, Any

from .fast_scm_setup import IndexSet
from .fast_scm_bb_helpers import (
    Precomputed,
    presolve,
    greedy_init,
    expand,
    Solution,
    get_qp_call_count,
    reset_qp_call_count,
    global_lower_bound,
    expand_weights_to_full,
    compute_search_space_size,
)


def branch_and_bound_topK(
    G: np.ndarray,
    candidate_idx: np.ndarray,
    m: int = 5,
    top_K: int = 20,
    unit_index: Optional[IndexSet] = None,
    unit_costs: Optional[np.ndarray] = None,
    budget: Optional[float] = None,
) -> Dict[str, Any]:

    start_time = time.time()
    reset_qp_call_count()

    # =========================================================
    # PRECOMPUTE
    # =========================================================
    pre = Precomputed(G)

    candidate_idx = np.sort(np.asarray(candidate_idx))

    candidate_idx = presolve(
        pre,
        candidate_idx,
        budget=budget,
        unit_costs=unit_costs,
    )

    M = len(candidate_idx)

    if unit_costs is None:
        unit_costs = np.zeros(G.shape[0])

    # search space stats
    total_subsets, total_nodes = compute_search_space_size(M, m)

    # =========================================================
    # GLOBAL LOWER BOUND
    # =========================================================
    global_lb = global_lower_bound(pre, m)

    # =========================================================
    # GREEDY INIT (UB SEED)
    # =========================================================
    greedy_indices, greedy_loss, greedy_w = greedy_init(pre, candidate_idx, m)

    top: List[Solution] = [
        Solution(greedy_loss, greedy_indices, greedy_w)
    ]

    # =========================================================
    # STATS
    # =========================================================
    stats = {
        "nodes_visited": 0,
        "branches_considered": 0,
        "branches_pruned": 0,
        "subsets_evaluated": 0,
        "leaf_nodes": 0,
    }

    # =========================================================
    # ROOT EXPANSION (FIXED — NO PRUNING)
    # =========================================================
    for i in candidate_idx:

        # cost constraint
        cost_i = float(unit_costs[i])
        if budget is not None and cost_i > budget:
            continue

        stats["branches_considered"] += 1

        Q_init = np.array([[pre.G[i, i]]])

        expand(
            pre=pre,
            candidate_idx=candidate_idx,
            m=m,
            top_K=top_K,
            top=top,
            indices=[i],
            stats=stats,
            Q=Q_init,
        )

    # =========================================================
    # FINALIZE
    # =========================================================
    solutions = sorted(top, key=lambda s: s.loss)

    elapsed = time.time() - start_time
    qp_calls = get_qp_call_count()

    best_loss = solutions[0].loss if solutions else np.inf
    worst_loss = solutions[-1].loss if solutions else np.inf

    nodes_visited = stats["nodes_visited"]
    branches_considered = stats["branches_considered"]
    branches_pruned = stats["branches_pruned"]
    subsets_eval = stats["subsets_evaluated"]
    leaf_nodes = stats["leaf_nodes"]

    node_fraction = nodes_visited / total_nodes if total_nodes else 0.0
    subset_fraction = subsets_eval / total_subsets if total_subsets else 0.0

    prune_rate = (
        branches_pruned / branches_considered
        if branches_considered else 0.0
    )

    qp_speedup = total_subsets / qp_calls if qp_calls else np.inf
    node_speedup = total_nodes / nodes_visited if nodes_visited else np.inf

    stats_out = {
        "search_space": {
            "M": M,
            "m": m,
            "total_subsets": total_subsets,
            "total_nodes": total_nodes,
        },

        "exploration": {
            "nodes_visited": nodes_visited,
            "subsets_evaluated": subsets_eval,
            "leaf_nodes": leaf_nodes,
            "node_fraction": node_fraction,
            "subset_fraction": subset_fraction,
        },

        "pruning": {
            "branches_considered": branches_considered,
            "branches_pruned": branches_pruned,
            "prune_rate": prune_rate,
        },

        "speedup": {
            "qp_speedup": round(qp_speedup, 2),
            "node_speedup": round(node_speedup, 2),
        },

        "performance": {
            "runtime_sec": round(elapsed, 4),
            "nodes_per_sec": round(nodes_visited / elapsed, 1) if elapsed else 0,
            "qp_calls": qp_calls,
            "qp_per_node": round(qp_calls / nodes_visited, 6) if nodes_visited else 0.0,
            "qp_per_subset": round(qp_calls / subsets_eval, 6) if subsets_eval else 0.0,
        },

        "optimality": {
            "best_loss": best_loss,
            "worst_in_topK": worst_loss,
            "design_stability": round(
                (worst_loss - best_loss) / (best_loss + 1e-12), 6
            ),
        },
    }

    # =========================================================
    # ATTACH METADATA
    # =========================================================
    total_units = len(unit_index) if unit_index is not None else G.shape[0]

    for i, sol in enumerate(solutions, start=1):
        sol.label = f"Tuple {i}"

        if unit_index is not None:
            sol.full_weights = expand_weights_to_full(
                sol.indices, sol.weights, total_units
            )
            sol.labels = unit_index.get_labels(sol.indices).tolist()
            sol.weight_dict = {
                unit_index.labels[idx]: float(w)
                for idx, w in zip(sol.indices, sol.weights)
            }

    return {
        "top_tuples": solutions,
        "stats": stats_out,
    }
