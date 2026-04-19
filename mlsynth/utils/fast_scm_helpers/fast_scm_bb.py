import numpy as np
from typing import List
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
    total_units: int = None
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

    num_seeds = min(50, max(20, 4 * m))

    # --- branch-and-bound ---
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
            stats=raw_stats,
            start_pos=i + 1,
            Q_partial=Q0
        )

    # =========================
    # FINAL SOLUTIONS
    # =========================
    solutions = sorted(top_tuples)

    for i, sol in enumerate(solutions, start=1):
        sol.label = f"Tuple {i}"

        if total_units is not None:
            sol.weights = expand_weights_to_full(
                sol.indices,
                sol.weights,
                total_units
            )

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