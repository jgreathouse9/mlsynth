import numpy as np
import time
from typing import List, Optional, Dict, Any
from .fast_scm_setup import IndexSet
from .fast_scm_bb_helpers import (
    compute_search_space_size,
    compute_global_lower_bound,
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

    # Compute once — valid lower bound at every depth and every subset.
    global_lb = compute_global_lower_bound(G, m)

    lam_min_global = float(np.linalg.eigvalsh(G)[0])

    stats: Dict[str, Any] = {
        "nodes_visited":       0,
        "nodes_generated":     0,
        "subsets_evaluated":   0,
        "leaf_nodes":          0,
        "branches_pruned":     0,
        "branches_considered": 0,
        "nodes_pruned":        0,
        "pruned_by_depth":     {},
        "incumbent_updates":   [],
    }

    top_tuples: List[Solution] = []

    print(f"[BnB] M={M} m={m} global_lb={global_lb:.6f}")

    # Root expansion
    for i in candidate_idx:
        stats["branches_considered"] += 1

        cost_i = float(unit_costs[i])
        if budget is not None and cost_i > budget:
            stats["branches_pruned"] += 1
            stats["nodes_pruned"]    += 1
            stats["pruned_by_depth"]["depth_0_budget"] = (
                stats["pruned_by_depth"].get("depth_0_budget", 0) + 1
            )
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
            lam_min_global=lam_min_global,
            global_lb=global_lb,
            unit_costs=unit_costs,
            budget=budget,
            current_cost=cost_i,
        )

    # ----------------------------------------------------------------
    # Final processing
    # ----------------------------------------------------------------
    solutions   = sorted(top_tuples, key=lambda s: s.loss)
    total_units = len(unit_index) if unit_index is not None else G.shape[0]

    for rank, sol in enumerate(solutions, start=1):
        sol.label = f"Tuple {rank}"
        if unit_index is not None:
            sol.full_weights = expand_weights_to_full(sol.indices, sol.weights, total_units)
            sol.labels       = unit_index.get_labels(sol.indices).tolist()
            sol.weight_dict  = {
                unit_index.labels[idx]: float(w)
                for idx, w in zip(sol.indices, sol.weights)
            }

    # ----------------------------------------------------------------
    # Stats
    # ----------------------------------------------------------------
    elapsed         = time.time() - start_time
    best_loss       = solutions[0].loss  if solutions else np.inf
    worst_loss      = solutions[-1].loss if solutions else np.inf
    qp_calls        = get_qp_call_count()

    nodes_visited   = stats["nodes_visited"]
    subsets_eval    = stats["subsets_evaluated"]
    branches_con    = stats["branches_considered"]
    branches_pruned = stats["branches_pruned"]

    node_fraction   = nodes_visited  / total_nodes   if total_nodes   else 0.0
    subset_fraction = subsets_eval   / total_subsets if total_subsets else 0.0
    prune_rate      = branches_pruned / branches_con if branches_con  else 0.0
    qp_speedup      = total_subsets  / qp_calls      if qp_calls      else np.inf
    node_speedup    = total_nodes    / nodes_visited  if nodes_visited else np.inf

    stats_out = {
        "search_space": {
            "M": M, "m": m,
            "total_subsets": total_subsets,
            "total_nodes":   total_nodes,
        },
        "bounds": {
            "global_lb": round(global_lb, 6),
        },
        "exploration": {
            "nodes_visited":     nodes_visited,
            "subsets_evaluated": subsets_eval,
            "leaf_nodes":        stats["leaf_nodes"],
            "node_fraction":     round(node_fraction,   4),
            "subset_fraction":   round(subset_fraction, 4),
        },
        "pruning": {
            "branches_considered": branches_con,
            "branches_pruned":     branches_pruned,
            "prune_rate":          round(prune_rate, 4),
            "pruned_by_depth":     stats["pruned_by_depth"],
        },
        "speedup": {
            "qp_speedup":  round(qp_speedup,   2),
            "node_speedup": round(node_speedup, 2),
        },
        "performance": {
            "runtime_sec":   round(elapsed, 4),
            "nodes_per_sec": round(nodes_visited / elapsed, 1) if elapsed else 0,
            "qp_calls":      qp_calls,
            "qp_per_node":   round(qp_calls / nodes_visited, 4) if nodes_visited else 0.0,
            "qp_per_subset": round(qp_calls / subsets_eval,  4) if subsets_eval  else 0.0,
        },
        "optimality": {
            "best_loss":        best_loss,
            "worst_in_topK":    worst_loss,
            "design_stability": round((worst_loss - best_loss) / best_loss, 4) if best_loss else 0.0,
        },
    }

    print(stats_out)

    return {
        "top_tuples": solutions,
        "stats":      stats_out,
    }
