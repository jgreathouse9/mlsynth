import numpy as np
import time
from typing import List, Optional, Dict, Any
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
) -> Dict[str, Any]:
    """
    Perform a Top-K Branch and Bound search to find optimal donor subsets.

    This function identifies the best 'm' donors from a candidate pool to minimize
    the quadratic loss (w'Qw) plus a linear cost penalty (lambda * cost). It 
    utilizes a Lagrangian-aware heuristic for tree ordering and aggressive 
    simplex-constrained pruning to bypass large sections of the combinatorial 
    search space.

    Parameters
    ----------
    G : np.ndarray
        The full Gram matrix (MxM) representing the variances and covariances 
        of all potential donor units.
    candidate_idx : np.ndarray
        Array of indices representing the pool of donors available for selection.
    m : int, default=5
        The exact number of donors to be included in each subset (tuple size).
    lam : float, default=0.0
        The penalty parameter (lambda) for the linear cost associated with 
        selecting specific units.
    top_K : int, default=20
        The number of top-ranking unique solutions to return.
    unit_index : IndexSet, optional
        An object containing mapping information (labels) for the units. If 
        provided, the returned solutions will include human-readable labels.
    unit_costs : np.ndarray, optional
        A vector of length M containing the cost associated with each unit. 
        Defaults to zeros if not provided.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing two primary keys:
        - 'top_tuples': A sorted list of the best `Solution` objects found.
        - 'stats': A nested dictionary of performance and search metrics, 
          including:
            * search_space: Total subsets and nodes in the theoretical tree.
            * exploration: Actual nodes visited and subsets fully evaluated.
            * pruning: Total branches bypassed and the calculated prune rate.
            * performance: Runtime, throughput (nodes/sec), and speedup factor.
            * optimality: Comparison between initial greedy and final best loss.
            * bound_quality: The relative gap between top and bottom of the K results.

    Notes
    -----
    The search uses a 'Seed-based entry' to ensure that the branching begins 
    with the most promising individual units, maximizing the early discovery of 
    strong upper bounds.
    """
    start_time = time.time()

    # Ensure unit_costs is available (default to zeros if not provided)
    if unit_costs is None:
        unit_costs = np.zeros(G.shape[0])

    # --- Lagrangian-Aware Heuristic Ordering ---
    # Sorting by marginal score (Variance + Penalty) helps find strong solutions 
    # early, lowering the 'score to beat' and increasing pruning depth.
    marginal_scores = np.diag(G) + (lam * unit_costs)
    candidate_idx = candidate_idx[np.argsort(marginal_scores[candidate_idx])]

    M = len(candidate_idx)
    total_subsets, total_nodes = compute_search_space_size(M, m)

    raw_stats = {
        "nodes_visited": 0,
        "subsets_evaluated": 0,
        "branches_pruned": 0,
        "branches_considered": 0,
    }

    # --- Greedy Baseline (Initialization via SFS) ---
    init_loss, init_idx, init_w = greedy_initial_solution(G, candidate_idx, m)
    top_tuples = []
    top_tuples.append(Solution(init_loss, init_idx, init_w))

    # --- Search Loop (Seed-based entry) ---
    # We seed the search with the top-performing individual units to establish 
    # a strong upper bound immediately.
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
            # Map weights back to the full unit dimension for visualization
            sol.full_weights = expand_weights_to_full(
                sol.indices,
                sol.weights,
                total_units
            )

            # Assign human-readable labels from the IndexSet
            sol.labels = unit_index.get_labels(sol.indices).tolist()

            # Dictionary form for easy JSON export or downstream analysis
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

    # How many times faster was this than evaluating every possible subset?
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
