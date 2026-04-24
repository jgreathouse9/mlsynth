
## bb


import numpy as np
from typing import List, Optional
from .fast_scm_setup import IndexSet
from .fast_scm_bb_helpers import (
    compute_search_space_size,
    expand_tuple,
    greedy_initial_solution,
    expand_weights_to_full,
    Solution
)

import time


def branch_and_bound_topK(
    G: np.ndarray,
    candidate_idx: np.ndarray,
    m: int = 5,
    top_K: int = 20,
    top_P: int = 10,
    total_units: int = None,
    unit_index: Optional[IndexSet] = None,   # 👈 ADD THIS
    unit_costs: Optional[np.ndarray] = None,
    budget: Optional[float] = None
):
    start_time = time.time()

    if len(candidate_idx) < m:
        raise ValueError(f"Not enough candidate units: {len(candidate_idx)} < m={m}")

    # --- heuristic ordering ---
    diag_vals = np.diag(G)
    candidate_idx = candidate_idx[np.argsort(diag_vals[candidate_idx])]

    top_tuples = []

    M = len(candidate_idx)
    total_subsets, total_nodes = compute_search_space_size(M, m)

    raw_stats = {
        "nodes_visited": 0,
        "subsets_evaluated": 0,
        "branches_pruned": 0,
        "branches_considered": 0,
    }

    # --- greedy baseline ---
    init_loss, init_idx, init_w = greedy_initial_solution(G, candidate_idx, m)
    top_tuples.append(Solution(init_loss, init_idx, init_w))

    M = len(candidate_idx)

    num_seeds = min(M, max(1, 4 * m))

    seed_set = candidate_idx[:num_seeds]  # safe slice, invariant to M

    for j in seed_set:
        Q0 = np.array([[G[j, j]]])

        j_pos = np.where(candidate_idx == j)[0][0]

        expand_tuple(
            G,
            candidate_idx,
            m,
            top_K,
            top_tuples,
            indices=[j],
            stats=raw_stats,
            start_pos=j_pos + 1,
            Q_partial=Q0,
            unit_costs=unit_costs,
            budget=budget,
            current_cost=(unit_costs[j] if unit_costs is not None else 0.0)
        )

    # =========================
    # FINAL SOLUTIONS
    # =========================
    total_units = len(unit_index)

    solutions = sorted(top_tuples)


    if total_units is not None and unit_index is not None:
        for sol in solutions:
            # full vector (index-aligned)
            sol.full_weights = expand_weights_to_full(
                sol.indices,
                sol.weights,
                total_units
            )

            # label list
            sol.labels = unit_index.get_labels(sol.indices).tolist()

            # 🔥 NEW: dictionary form (THIS is what you want downstream)
            sol.weight_dict = {
                unit_index.labels[i]: float(w)
                for i, w in zip(sol.indices, sol.weights)
            }

    # Inside branch_and_bound_topK, near the end:
    for i, sol in enumerate(solutions, start=1):
        sol.label = f"Tuple {i}"

    # =========================
    # DERIVED VALUES
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

    improvement_per_sec = improvement / elapsed if elapsed else 0
    improvement_per_node = improvement / raw_stats["nodes_visited"] if raw_stats["nodes_visited"] else 0

    bound_relative_gap = ranking_spread / best_loss if best_loss else 0

    # =========================
    # STRUCTURED STATS
    # =========================
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

        "results": {
            "best_loss": best_loss,
            "worst_loss": worst_loss,
            "ranking_spread": ranking_spread,
        },

        "optimality": {
            "initial_loss": init_loss,
            "best_loss": best_loss,
            "improvement": improvement,
            "improvement_pct": (
                improvement / init_loss if init_loss else 0
            ),
        },

        "efficiency": {
            "loss_reduction_per_eval": (
                improvement / raw_stats["subsets_evaluated"]
                if raw_stats["subsets_evaluated"] else 0
            ),
            "improvement_per_sec": improvement_per_sec,
            "improvement_per_node": improvement_per_node,
        },

        "bound_quality": {
            "relative_gap": bound_relative_gap
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
