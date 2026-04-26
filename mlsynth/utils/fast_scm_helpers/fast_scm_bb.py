import numpy as np
import time
from typing import List, Optional, Dict, Any
from .fast_scm_setup import IndexSet
from .fast_scm_bb_helpers import (
    compute_search_space_size,
    expand_tuple,
    greedy_initial_solution,
    expand_weights_to_full,
    Solution,
    get_qp_call_count,
    reset_qp_call_count, presolve_candidates
)


def branch_and_bound_topK(
        G: np.ndarray,
        candidate_idx: np.ndarray,
        m: int = 5,
        lam: float = 0.0,
        top_K: int = 20,
        unit_index: Optional[IndexSet] = None,
        unit_costs: Optional[np.ndarray] = None,
        budget: Optional[float] = None  # Added budget to orchestrator
) -> Dict[str, Any]:
    start_time = time.time()
    reset_qp_call_count()

    if unit_costs is None:
        unit_costs = np.zeros(G.shape[0])

    # ============================================================
    # 1. PRESOLVE & ORDERING
    # ============================================================
    # Step A: Initial Sorting based on marginal impact
    marginal_scores = np.diag(G) + lam * unit_costs
    candidate_idx = candidate_idx[np.argsort(marginal_scores[candidate_idx])]

    # Step B: Presolve (Filter out redundant/impossible units)
    # This reduces M before we calculate search space stats
    candidate_idx = presolve_candidates(
        G=G,
        candidate_idx=candidate_idx,
        budget=budget,
        unit_costs=unit_costs,
        m=m
    )

    M = len(candidate_idx)
    total_subsets, total_nodes = compute_search_space_size(M, m)

    stats = {
        "nodes_visited": 0,
        "subsets_evaluated": 0,
        "branches_pruned": 0,
        "branches_considered": 0,
    }

    # ============================================================
    # 2. INITIAL SOLUTION (On Reduced Set)
    # ============================================================
    init_loss, init_idx, init_w = greedy_initial_solution(G, candidate_idx, m)

    top_tuples: List[Solution] = [
        Solution(init_loss, init_idx, init_w)
    ]

    # ============================================================
    # 3. SEEDING (Synchronized with Pruning Metrics)
    # ============================================================
    # We only seed from the high-quality candidates
    num_seeds = min(M, max(1, 2 * m))
    seed_set = candidate_idx[:num_seeds]

    for j in seed_set:
        # Every seed call is technically 'considering' a branch from the root
        stats["branches_considered"] += 1

        # Quick root-level prune
        if G[j, j] >= top_tuples[-1].loss and len(top_tuples) == top_K:
            stats["branches_pruned"] += 1
            continue

        Q0 = np.array([[G[j, j]]])
        j_pos = np.where(candidate_idx == j)[0][0]

        expand_tuple(
            G=G,
            candidate_idx=candidate_idx,
            m=m,
            top_K=top_K,
            top_tuples=top_tuples,
            indices=[j],
            stats=stats,
            start_pos=j_pos + 1,
            Q_partial=Q0,
            unit_costs=unit_costs,
            budget=budget,
            current_cost=float(unit_costs[j])
        )

    # ============================================================
    # FINAL SORT
    # ============================================================
    solutions = sorted(top_tuples, key=lambda s: s.loss)

    total_units = len(unit_index) if unit_index is not None else G.shape[0]

    for i, sol in enumerate(solutions, start=1):
        sol.label = f"Tuple {i}"

        if unit_index is not None:
            sol.full_weights = expand_weights_to_full(
                sol.indices,
                sol.weights,
                total_units
            )

            sol.labels = unit_index.get_labels(sol.indices).tolist()

            sol.weight_dict = {
                unit_index.labels[idx]: float(w)
                for idx, w in zip(sol.indices, sol.weights)
            }

    # ============================================================
    # METRICS
    # ============================================================
    elapsed = time.time() - start_time

    best_loss = solutions[0].loss if solutions else np.inf
    worst_loss = solutions[-1].loss if solutions else np.inf

    node_fraction = stats["nodes_visited"] / total_nodes if total_nodes else 0
    subset_fraction = stats["subsets_evaluated"] / total_subsets if total_subsets else 0

    prune_rate = (
        stats["branches_pruned"] / stats["branches_considered"]
        if stats["branches_considered"] else 0
    )

    speedup = (
        total_subsets / stats["subsets_evaluated"]
        if stats["subsets_evaluated"] else np.inf
    )

    improvement = init_loss - best_loss

    qp_calls = get_qp_call_count()

    # ============================================================
    # BUILD stats_out FIRST
    # ============================================================
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
            "nodes_per_sec": stats["nodes_visited"] / elapsed if elapsed else 0,
            "speedup_factor": speedup,

            # QP STATS go here
            "qp_calls": qp_calls,
            "qp_per_node": qp_calls / stats["nodes_visited"] if stats["nodes_visited"] else 0.0,
        },
        "optimality": {
            "initial_loss": init_loss,
            "best_loss": best_loss,
            "improvement": improvement,
        },
        "bestvworst": {
            "design_stability": (worst_loss - best_loss) / best_loss if best_loss else 0
        }
    }

    return {
        "top_tuples": solutions,
        "stats": stats_out
    }
