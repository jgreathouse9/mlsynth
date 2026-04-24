import numpy as np
import time
from typing import List, Optional
from .fast_scm_setup import IndexSet
from .fast_scm_bb_helpers import (
    compute_search_space_size,
    expand_tuple,
    greedy_initial_solution,
    expand_weights_to_full,
    Solution
)

def branch_and_bound_topK(
    G: np.ndarray,
    candidate_idx: np.ndarray,
    m: int = 5,
    lam: float = 0.0,
    top_K: int = 20,
    unit_index: Optional[IndexSet] = None,
    unit_costs: Optional[np.ndarray] = None,
):
    """
    Top-K Branch and Bound using Lagrangian relaxation and Gershgorin bounds.
    """
    start_time = time.time()

    if len(candidate_idx) < m:
        raise ValueError(f"Not enough candidate units: {len(candidate_idx)} < m={m}")

    # Ensure unit_costs is available (default to zeros if not provided)
    if unit_costs is None:
        unit_costs = np.zeros(G.shape[0])

    # --- Heuristic Ordering ---
    # Sorting by diagonal (variance) helps find better solutions earlier, 
    # which leads to more aggressive pruning.
    diag_vals = np.diag(G)
    candidate_idx = candidate_idx[np.argsort(diag_vals[candidate_idx])]

    M = len(candidate_idx)
    total_subsets, total_nodes = compute_search_space_size(M, m)

    raw_stats = {
        "nodes_visited": 0,
        "subsets_evaluated": 0,
        "branches_pruned": 0,
        "branches_considered": 0,
    }

    # --- Greedy Baseline (Initialization) ---
    # This provides a 'score to beat' right at the start.
    init_sol = greedy_initial_solution(G, unit_costs, candidate_idx, m, lam)
    init_loss = init_sol.loss
    top_tuples = [init_sol]

    # --- Search Loop (Seed-based entry) ---
    # We enter the recursion with each possible starting unit.
    num_seeds = min(M, max(1, 4 * m))
    seed_set = candidate_idx[:num_seeds]

    for j in seed_set:
        Q0 = np.array([[G[j, j]]])
        j_pos = np.where(candidate_idx == j)[0][0]

        expand_tuple(
            G=G,
            candidate_idx=candidate_idx,
            unit_costs=unit_costs,
            m=m,
            lam=lam,
            top_K=top_K,
            top_tuples=top_tuples,
            indices=[j],
            stats=raw_stats,
            start_pos=j_pos + 1,
            Q_partial=Q0,
            current_cost=float(unit_costs[j])
        )

    # =========================
    # FINAL SOLUTIONS PROCESSING
    # =========================
    solutions = sorted(top_tuples)
    total_units = len(unit_index) if unit_index is not None else G.shape[0]

    for i, sol in enumerate(solutions, start=1):
        sol.label = f"Tuple {i}"
        
        if unit_index is not None:
            # Map weights back to the full unit dimension
            sol.full_weights = expand_weights_to_full(
                sol.indices,
                sol.weights,
                total_units
            )

            # Assign human-readable labels
            sol.labels = unit_index.get_labels(sol.indices).tolist()

            # Dictionary form for downstream consumption
            sol.weight_dict = {
                unit_index.labels[idx]: float(w)
                for idx, w in zip(sol.indices, sol.weights)
            }

    # =========================
    # STATS & DERIVED VALUES
    # =========================
    best_loss = solutions[0].loss
    worst_loss = solutions[-1].loss
    elapsed = time.time() - start_time

    node_fraction = raw_stats["nodes_visited"] / total_nodes if total_nodes else 0
    subset_fraction = raw_stats["subsets_evaluated"] / total_subsets if total_subsets else 0

    prune_rate = (
        raw_stats["branches_pruned"] / raw_stats["branches_considered"]
        if raw_stats["branches_considered"] else 0
    )

    speedup = (
        total_subsets / raw_stats["subsets_evaluated"]
        if raw_stats["subsets_evaluated"] else np.inf
    )

    improvement = init_loss - best_loss
    ranking_spread = worst_loss - best_loss

    stats = {
        "search_space": {
            "total_subsets": total_subsets,
            "total_nodes": total_nodes,
        },
        "exploration": {
            "nodes_visited": raw_stats["nodes_visited"],
            "subsets_evaluated": raw_stats["subsets_evaluated"],
            "node_fraction_explored": node_fraction,
            "subset_fraction_explored": subset_fraction,
        },
        "pruning": {
            "branches_considered": raw_stats["branches_considered"],
            "branches_pruned": raw_stats["branches_pruned"],
            "prune_rate": prune_rate,
            "pruned_per_node": (
                raw_stats["branches_pruned"] / raw_stats["nodes_visited"]
                if raw_stats["nodes_visited"] else 0
            ),
        },
        "performance": {
            "runtime_sec": elapsed,
            "nodes_per_sec": (
                raw_stats["nodes_visited"] / elapsed if elapsed else 0
            ),
            "speedup_factor": speedup,
        },
        "optimality": {
            "initial_loss": init_loss,
            "best_loss": best_loss,
            "improvement": improvement,
            "improvement_pct": (improvement / init_loss if init_loss else 0),
        },
        "efficiency": {
            "loss_reduction_per_eval": (
                improvement / raw_stats["subsets_evaluated"]
                if raw_stats["subsets_evaluated"] else 0
            ),
            "improvement_per_sec": improvement / elapsed if elapsed else 0,
        },
        "bound_quality": {
            "relative_gap": ranking_spread / best_loss if best_loss else 0
        },
        "branching": {
            "avg_branching_factor": (
                raw_stats["branches_considered"] / raw_stats["nodes_visited"]
                if raw_stats["nodes_visited"] else 0
            )
        }
    }

    return {
        "top_tuples": solutions,
        "stats": stats
    }
```
