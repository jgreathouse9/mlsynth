import numpy as np
import time
from typing import List, Optional, Dict, Any
from .fast_scm_setup import IndexSet
from .fast_scm_bb_helpers import (
    compute_search_space_size,
    expand_tuple,
    expand_weights_to_full,
    Solution,
    get_qp_call_count,
    reset_qp_call_count,
)


def branch_and_bound_topK(
    G: np.ndarray,
    candidate_idx: np.ndarray,
    m: int = 5,
    lam: float = 0.0,
    top_K: int = 20,
    unit_index: Optional[IndexSet] = None,
    unit_costs: Optional[np.ndarray] = None,
    budget: Optional[float] = None,
) -> Dict[str, Any]:

    start_time = time.time()
    reset_qp_call_count()

    # Sorted order is required by expand_tuple's position-based combinatorics
    candidate_idx = np.sort(np.asarray(candidate_idx))

    if unit_costs is None:
        unit_costs = np.zeros(G.shape[0])

    M = len(candidate_idx)
    total_subsets, total_nodes = compute_search_space_size(M, m)

    stats = {
        "nodes_visited": 0,
        "subsets_evaluated": 0,
        "branches_pruned": 0,
        "branches_considered": 0,
    }

    # No incumbent — start with an empty top_tuples list.
    # current_ub inside expand_tuple will be inf until top_K leaves are found.
    top_tuples: List[Solution] = []

    print(f"[BnB] Running pure search over {M} candidates, m={m}")

    # Root expansion: seed one unit at a time
    for i in candidate_idx:
        stats["branches_considered"] += 1

        cost_i = float(unit_costs[i])
        if budget is not None and cost_i > budget:
            stats["branches_pruned"] += 1
            continue

        expand_tuple(
            G=G,
            candidate_idx=candidate_idx,
            m=m,
            top_K=top_K,
            top_tuples=top_tuples,
            indices=[i],
            stats=stats,
            Q_partial=np.array([[G[i, i]]]),
            unit_costs=unit_costs,
            budget=budget,
            current_cost=cost_i,
        )

    # ----------------------------------------------------------------
    # Final processing
    # ----------------------------------------------------------------
    solutions = sorted(top_tuples, key=lambda s: s.loss)

    total_units = len(unit_index) if unit_index is not None else G.shape[0]

    for rank, sol in enumerate(solutions, start=1):
        sol.label = f"Tuple {rank}"
        if unit_index is not None:
            sol.full_weights = expand_weights_to_full(sol.indices, sol.weights, total_units)
            sol.labels = unit_index.get_labels(sol.indices).tolist()
            sol.weight_dict = {
                unit_index.labels[idx]: float(w)
                for idx, w in zip(sol.indices, sol.weights)
            }

    # ----------------------------------------------------------------
    # Stats
    # ----------------------------------------------------------------
    elapsed       = time.time() - start_time
    best_loss     = solutions[0].loss if solutions else np.inf
    worst_loss    = solutions[-1].loss if solutions else np.inf
    node_fraction = stats["nodes_visited"] / total_nodes if total_nodes else 0
    subset_fraction = stats["subsets_evaluated"] / total_subsets if total_subsets else 0
    qp_calls      = get_qp_call_count()

    stats_out = {
        "search_space": {
            "total_subsets": total_subsets,
            "total_nodes": total_nodes,
        },
        "exploration": {
            "nodes_visited": stats["nodes_visited"],
            "subsets_evaluated": stats["subsets_evaluated"],
            "node_fraction_explored": node_fraction,
            "subset_fraction_explored": subset_fraction,
        },
        "pruning": {
            "branches_considered": stats["branches_considered"],
            "branches_pruned": stats["branches_pruned"],
        },
        "performance": {
            "runtime_sec": elapsed,
            "nodes_per_sec": stats["nodes_visited"] / elapsed if elapsed else 0,
            "qp_calls": qp_calls,
            "qp_per_node": qp_calls / stats["nodes_visited"] if stats["nodes_visited"] else 0.0,
        },
        "optimality": {
            "best_loss": best_loss,
        },
        "bestvworst": {
            "design_stability": (worst_loss - best_loss) / best_loss if best_loss else 0,
        },
    }

    print(stats_out)

    return {
        "top_tuples": solutions,
        "stats": stats_out,
    }
