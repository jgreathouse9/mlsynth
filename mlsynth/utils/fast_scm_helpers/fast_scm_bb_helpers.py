
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from math import comb


def presolve_candidates(G, candidate_idx, budget=None, unit_costs=None, m=1):
    """
    Shrinks the search space by removing units that are mathematically
    impossible or strictly redundant.
    """
    # 1. Cost-Impossible Filter
    if budget is not None and unit_costs is not None:
        # A unit is impossible if its cost exceeds the budget
        candidate_idx = np.array([i for i in candidate_idx if unit_costs[i] <= budget])

    # 2. Diagonal Dominance (Variance Filter)
    # If a unit's variance is 100x higher than the average, it's likely just noise.
    # (Be careful with this one, keep it conservative)
    diags = np.diag(G)[candidate_idx]
    threshold = np.median(diags) * 50
    candidate_idx = np.array([i for i in candidate_idx if G[i, i] < threshold])

    # 3. Correlation Redundancy (Collinearity Filter)
    # If two units are 99.9% correlated, we only need to keep the cheaper one.
    to_remove = set()
    for i in range(len(candidate_idx)):
        u_i = candidate_idx[i]
        if u_i in to_remove: continue

        for j in range(i + 1, len(candidate_idx)):
            u_j = candidate_idx[j]
            if u_j in to_remove: continue

            # Correlation calculation
            corr = G[u_i, u_j] / np.sqrt(G[u_i, u_i] * G[u_j, u_j])
            if corr > 0.999:
                # Keep the cheaper one
                if unit_costs is not None:
                    remove_target = u_j if unit_costs[u_j] >= unit_costs[u_i] else u_i
                else:
                    remove_target = u_j
                to_remove.add(remove_target)

    return np.array([c for c in candidate_idx if c not in to_remove])


# ============================================================
# GLOBAL STATE
# ============================================================

_qp_call_count = 0
_warm_start_cache: Dict[Tuple[int, Tuple[int, ...]], np.ndarray] = {}


def get_qp_call_count() -> int:
    return _qp_call_count


def reset_qp_call_count():
    global _qp_call_count
    _qp_call_count = 0


def clear_solver_cache():
    global _qp_call_count, _warm_start_cache
    _qp_call_count = 0
    _warm_start_cache.clear()


# ============================================================
# SOLUTION CONTAINER
# ============================================================

@dataclass(order=True)
class Solution:
    loss: float
    indices: List[int] = field(compare=False)
    weights: np.ndarray = field(compare=False)
    labels: Optional[List[Any]] = field(default=None, compare=False)
    full_weights: Optional[np.ndarray] = field(default=None, compare=False)
    weight_dict: Optional[Dict[Any, float]] = field(default=None, compare=False)
    cost: float = 0.0
    label: Optional[str] = field(default=None, compare=False)


# ============================================================
# SEARCH SPACE SIZE
# ============================================================

def compute_search_space_size(M: int, m: int):
    total_subsets = comb(M, m)
    total_nodes = sum(comb(M, k) for k in range(1, m + 1))
    return total_subsets, total_nodes


def lookahead_lower_bound(G, current_lb, remaining_candidates, slots_left, m):
    """
    Estimates the absolute floor for a branch by assuming the best
    remaining units are added to the current selection.
    """
    if slots_left <= 0:
        return current_lb

    # Best case: the remaining units have the smallest diagonals (variances)
    # in the candidate pool.
    best_remaining_vars = np.sort(np.diag(G)[remaining_candidates])[:slots_left]

    # Heuristic: The loss of a size-m set is roughly a weighted average.
    # A conservative floor: (k/m)*current_lb + ((m-k)/m)*min_possible_var
    k = m - slots_left
    lookahead_floor = (k / m) * current_lb + (slots_left / m) * np.mean(best_remaining_vars)

    # Ensure we don't return a value lower than physically possible (0)
    return max(0.0, lookahead_floor * 0.8)  # 0.8 factor to keep it conservative/safe


# ============================================================
# FAST ACTIVE-SET SIMPLEX QP SOLVER (CORE)
# ============================================================

import cvxpy as cp

# Global cache can still be used if you want, but CVXPY warm-start is limited for small problems

def solve_qp_simplex_value(
    Q: np.ndarray,
    w_init: Optional[np.ndarray] = None,
    indices: Optional[List[int]] = None,
    **kwargs
) -> Tuple[float, np.ndarray]:

    global _qp_call_count
    _qp_call_count += 1

    k = Q.shape[0]

    w = cp.Variable(k, nonneg=True)
    objective = cp.quad_form(w, Q)
    prob = cp.Problem(cp.Minimize(objective), [cp.sum(w) == 1])

    # Optional: warm start (CVXPY supports it modestly)
    if w_init is not None:
        w.value = np.maximum(w_init, 0.0)
        w.value /= np.sum(w.value) + 1e-14

    prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-10, eps_rel=1e-10)

    if prob.status in ["optimal", "optimal_inaccurate"]:
        weights = np.maximum(w.value, 0.0)
        weights /= np.sum(weights) + 1e-14
        loss = float(prob.value)
    else:
        # Fallback: use diagonal minimum if solver fails
        weights = np.zeros(k)
        weights[np.argmin(np.diag(Q))] = 1.0
        loss = float(Q[np.argmin(np.diag(Q)), np.argmin(np.diag(Q))])

    # Optional caching
    if indices is not None:
        _warm_start_cache[tuple(indices)] = weights.copy()

    return loss, weights


# ============================================================
# LOWER BOUND
# ============================================================

def simplex_lower_bound(Q: np.ndarray) -> float:
    """Balanced safe + tighter bound for Gram submatrices (PSD)"""
    k = Q.shape[0]
    if k == 1:
        return float(Q[0, 0])

    diags = np.diag(Q)
    diag_min = float(np.min(diags))
    lambda_min = float(np.min(np.linalg.eigvalsh(Q)))

    # A bit more aggressive but still reasonably safe combination
    lb = max(
        0.0,
        lambda_min,                    # strong and safe
        diag_min * 0.65,               # increased from 0.5
        (diag_min + lambda_min) * 0.5  # gentle blending
    )
    return lb



# ============================================================
# GREEDY INITIAL SOLUTION
# ============================================================

def greedy_initial_solution(G, candidate_idx, m):
    selected = list(candidate_idx[:m])
    Q = G[np.ix_(selected, selected)]
    loss, w = solve_qp_simplex_value(Q)
    return loss, selected, w


# ============================================================
# EXPAND WEIGHTS
# ============================================================

def expand_weights_to_full(indices, weights, total_units):
    w_full = np.zeros(total_units)
    w_full[indices] = weights
    return w_full


# ============================================================
# STRONG BRANCHING SCORE (LIGHTWEIGHT LOOKAHEAD)
# ============================================================

def strong_branch_score(G, Q_partial, candidate_idx, j, indices):

    k = Q_partial.shape[0]

    if k == 0:
        return -G[j, j]

    g = G[j, indices]

    return -G[j, j] - 2.0 * np.sum(g) / (k + 1e-12)


# ============================================================
# BnB CORE EXPANSION
# ============================================================
def expand_tuple(
        G, candidate_idx, m, top_K, top_tuples, indices, stats,
        start_pos, Q_partial, unit_costs=None, budget=None, current_cost=0.0,
):
    # This is a 'Node Visit', not a 'Branch Consideration'
    stats["nodes_visited"] += 1
    k = len(indices)
    slots_left = m - k
    current_ub = top_tuples[-1].loss if len(top_tuples) == top_K else np.inf

    # -----------------------------
    # 1. BASE CASE: REACHED SIZE M
    # -----------------------------
    if k == m:
        stats["subsets_evaluated"] += 1
        loss, w = solve_qp_simplex_value(Q_partial, indices=indices)
        top_tuples.append(Solution(loss, indices[:], w))
        top_tuples.sort(key=lambda s: s.loss)
        if len(top_tuples) > top_K:
            top_tuples.pop()
        return

    # -----------------------------
    # 2. EXPANSION LOOP
    # -----------------------------
    candidates = candidate_idx[start_pos:]

    # Pre-calculate scores for the children of THIS node
    scored = [(j, strong_branch_score(G, Q_partial, candidate_idx, j, indices)) for j in candidates]
    scored.sort(key=lambda x: x[1])

    for j, _ in scored:
        # Step A: Mark that we are LOOKING at this specific unit as a candidate
        stats["branches_considered"] += 1

        # Step B: Check Budget for this specific unit
        new_cost = current_cost + (unit_costs[j] if unit_costs is not None else 0.0)
        if budget is not None and new_cost > budget:
            stats["branches_pruned"] += 1
            continue

        # Step C: Incremental Gram Matrix update
        g = G[j, indices]
        Q_new = np.empty((k + 1, k + 1))
        Q_new[:k, :k] = Q_partial
        Q_new[k, :k] = g
        Q_new[:k, k] = g
        Q_new[k, k] = G[j, j]

        # Step D: Lower Bound Check for the potential new node
        lb_node = simplex_lower_bound(Q_new)

        # Step E: Lookahead Check (Predictive)
        # We do this here so it counts as pruning the branch we just 'considered'
        if k + 1 < m:
            # Remaining pool for the NEXT level
            next_start_pos = np.where(candidate_idx == j)[0][0] + 1
            remaining_pool = candidate_idx[next_start_pos:]

            lb_predicted = lookahead_lower_bound(G, lb_node, remaining_pool, m - (k + 1), m)
            lb_to_check = max(lb_node, lb_predicted)
        else:
            lb_to_check = lb_node

        if lb_to_check >= current_ub:
            stats["branches_pruned"] += 1
            continue

        # Step F: Success! Visit the child
        expand_tuple(
            G, candidate_idx, m, top_K, top_tuples, indices + [j], stats,
                                                    np.where(candidate_idx == j)[0][0] + 1,
            Q_new, unit_costs, budget, new_cost
        )
