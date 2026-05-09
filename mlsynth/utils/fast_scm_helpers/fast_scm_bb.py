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
    expand_weights_to_full,
    compute_search_space_size,
    make_stats,
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
    # PRECOMPUTE & COST SETUP
    # =========================================================
    pre = Precomputed(G)

    if unit_costs is None:
        unit_costs = np.zeros(G.shape[0])

    if budget is None:
        budget = np.inf

    candidate_idx = np.sort(np.asarray(candidate_idx))

    # 1. Standard Presolve (Diagonal/Correlation filters)
    candidate_idx = presolve(
        pre,
        candidate_idx,
        m,
        budget=budget,
        unit_costs=unit_costs,
    )

    # 2. Lookahead Presolve (One-shot Budget Filter)
    # If a unit + the (m-1) cheapest other units > budget, it's impossible.
    sorted_all_costs = np.sort(unit_costs[candidate_idx])
    if len(sorted_all_costs) >= m:
        # The absolute minimum cost required to complete any tuple from here
        min_floor_global = np.sum(sorted_all_costs[:m - 1])
        candidate_idx = candidate_idx[unit_costs[candidate_idx] + min_floor_global <= budget]

    # Re-sort costs for ONLY the surviving candidates to pass to the recursive expand
    candidate_costs_sorted = np.sort(unit_costs[candidate_idx])

    M = len(candidate_idx)
    total_subsets, total_nodes = compute_search_space_size(M, m)

    # =========================================================
    # GREEDY INIT (UPPER BOUND SEED)
    # =========================================================
    # Note: Ensure greedy_init logic respects the budget if applicable
    greedy_indices, greedy_loss, greedy_w = greedy_init(
        pre,
        candidate_idx,
        m,
        unit_costs=unit_costs,  # <--- Pass the cost array
        budget=budget  # <--- Pass the budget
    )

    top: List[Solution] = [
        Solution(greedy_loss, greedy_indices, greedy_w)
    ]

    # tighten UB immediately
    ub = greedy_loss

    # =========================================================
    # STATS 
    # =========================================================
    stats = make_stats()

    # =========================================================
    # ROOT EXPANSION
    # =========================================================
    for i in candidate_idx:

        # --- Root Level Lookahead ---
        # Don't even start expanding a root that cannot mathematically succeed
        if budget != np.inf:
            # Cheapest partners for this specific root
            # (Fastest to use global floor, but we could exclude 'i' for more precision)
            if unit_costs[i] + np.sum(candidate_costs_sorted[:m - 1]) > budget:
                continue

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
            unit_costs=unit_costs,
            budget=budget,
            candidate_costs_sorted=candidate_costs_sorted
        )

    # =========================================================
    # FINALIZE SOLUTIONS
    # =========================================================
    solutions = sorted(top, key=lambda x: x.loss)

    elapsed = time.time() - start_time
    qp_calls = get_qp_call_count()

    best_loss = solutions[0].loss if solutions else np.inf
    worst_loss = solutions[-1].loss if solutions else np.inf

    # =========================================================
    # STATS EXTRACTION
    # =========================================================
    nodes_visited = stats["nodes_visited"]
    node_prunes = stats["node_prunes"]
    branch_prunes = stats["branch_prunes"]
    qp_solved = stats["leaves_solved"]

    node_fraction = nodes_visited / total_nodes if total_nodes else 0.0
    bound_hits = stats["bound_hits"]

    pruning_breakdown = {
        name: {
            "node": v["node"],
            "branch": v["branch"],
            "total": v["node"] + v["branch"],
            "fraction": round(
                (v["node"] + v["branch"]) /
                max(node_prunes + branch_prunes, 1),
                6
            ),
        }
        for name, v in bound_hits.items()
        if (v["node"] + v["branch"]) > 0
    }

    # =========================================================
    # OUTPUT
    # =========================================================
    stats_out = {
        "search_space": {
            "M": M,
            "m": m,
            "total_subsets": total_subsets,
            "total_nodes": total_nodes,
        },
        "exploration": {
            "nodes_visited": nodes_visited,
            "node_fraction": node_fraction,
            "qp_solved": qp_solved,
        },
        "pruning": {
            "node_prunes": node_prunes,
            "branch_prunes": branch_prunes,
            "by_bound": pruning_breakdown,
        },
        "performance": {
            "runtime_sec": round(elapsed, 4),
            "nodes_per_sec": round(nodes_visited / elapsed, 1) if elapsed else 0,
            "qp_calls": qp_calls,
            "qp_per_node": round(qp_calls / nodes_visited, 6) if nodes_visited else 0.0,
        },
        "optimality": {
            "best_loss": best_loss,
            "worst_in_topK": worst_loss,
            "design_stability": round(
                (worst_loss - best_loss) / (best_loss + 1e-12),
                6
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

    for i, sol in enumerate(solutions, start=1):
        sol.label = f"Tuple {i}"

        # Core Cost Metrics
        sol.total_cost = float(np.sum(unit_costs[sol.indices]))

        # Handle the 'blah blah' (Remaining & Utilization)
        # We use max(budget, total_cost) for utilization safety if budget is inf
        sol.remaining_budget = budget - sol.total_cost if budget != np.inf else 0.0

        if budget and budget != np.inf:
            sol.utilization_pct = (sol.total_cost / budget) * 100
        else:
            sol.utilization_pct = 0.0

        if unit_index is not None:
            sol.labels = unit_index.get_labels(sol.indices).tolist()

            # Aligned Breakdown
            sol.cost_breakdown = {
                unit_index.labels[idx]: float(unit_costs[idx])
                for idx in sol.indices
            }

            # Weights for plotting/full reconstruction
            sol.full_weights = expand_weights_to_full(
                sol.indices, sol.weights, total_units
            )

    return {
        "top_tuples": solutions,
        "stats": stats_out,
    }
