import numpy as np
import time
from typing import List, Optional, Dict, Any

from .fast_scm_setup import IndexSet
from .fast_scm_bb_helpers import (
    compute_search_space_size,
    expand_tuple,
    expand_weights_to_full,
    Solution,
    greedy_initial_solution,
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

    # ------------------------------------------------------------
    # PREP
    # ------------------------------------------------------------
    candidate_idx = np.sort(np.asarray(candidate_idx))

    if unit_costs is None:
        unit_costs = np.zeros(G.shape[0])

    M = len(candidate_idx)
    total_subsets, total_nodes = compute_search_space_size(M, m)

    # ------------------------------------------------------------
    # INITIAL UPPER BOUND (greedy)
    # ------------------------------------------------------------
    init_idx, init_loss, init_w = greedy_initial_solution(G, candidate_idx, m)
    top_tuples: List[Solution] = [Solution(init_loss, init_idx, init_w)]

    print(f"[BnB] Running pure search over {M} candidates, m={m}")

    # ------------------------------------------------------------
    # STATS
    # ------------------------------------------------------------
    stats = {
        "nodes_visited": 0,
        "subsets_evaluated": 0,
        "branches_pruned": 0,
        "branches_considered": 0,
    }

    # ------------------------------------------------------------
    # ROOT EXPANSION
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # FINAL SOLUTIONS
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # PERFORMANCE METRICS
    # ------------------------------------------------------------
    elapsed = time.time() - start_time
    best_loss = solutions[0].loss if solutions else np.inf
    worst_loss = solutions[-1].loss if solutions else np.inf

    node_fraction = stats["nodes_visited"] / total_nodes if total_nodes else 0
    subset_fraction = stats["subsets_evaluated"] / total_subsets if total_subsets else 0

    qp_calls = get_qp_call_count()

    # ------------------------------------------------------------
    # NEW METRICS
    # ------------------------------------------------------------

    prune_rate = (
        stats["branches_pruned"] / stats["branches_considered"]
        if stats["branches_considered"] > 0 else 0.0
    )

    exhaustive_qp_calls = total_subsets
    speedup_factor = (
        exhaustive_qp_calls / qp_calls
        if qp_calls > 0 else float("inf")
    )

    qp_per_subset_ratio = (
        qp_calls / total_subsets
        if total_subsets > 0 else 0.0
    )

    # ------------------------------------------------------------
    # OUTPUT STATS
    # ------------------------------------------------------------
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
            "prune_rate": prune_rate,
        },
        "performance": {
            "runtime_sec": elapsed,
            "nodes_per_sec": stats["nodes_visited"] / elapsed if elapsed else 0.0,
            "qp_calls": qp_calls,
            "qp_per_node": qp_calls / stats["nodes_visited"] if stats["nodes_visited"] else 0.0,
            "qp_per_subset_ratio": qp_per_subset_ratio,
            "speedup_factor_vs_bruteforce": speedup_factor,
        },
        "optimality": {
            "best_loss": best_loss,
        },
        "bestvworst": {
            "design_stability": (worst_loss - best_loss) / best_loss if best_loss else 0.0,
        },
    }

    print(stats_out)

    return {
        "top_tuples": solutions,
        "stats": stats_out,
    }
