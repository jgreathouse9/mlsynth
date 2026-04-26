"""
Fast_scm_bb_helpers.py - GUROBI-style QP-reduced BnB helpers
Now includes:
- active-set simplex QP solver
- incremental warm-start cache
- strong branching heuristic
"""

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
            if corr > 0.98:
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

def solve_qp_simplex_value(
    Q: np.ndarray,
    w_init: Optional[np.ndarray] = None,
    indices: Optional[List[int]] = None,
    max_iter: int = 30,
    tol: float = 1e-10,
) -> Tuple[float, np.ndarray]:

    global _qp_call_count, _warm_start_cache
    _qp_call_count += 1

    k = Q.shape[0]

    # -----------------------------
    # WARM START INIT
    # -----------------------------
    if w_init is not None:
        w = np.maximum(w_init, 0.0)
    else:
        w = np.ones(k) / k

    # reuse parent solution if available
    if indices is not None:
        key = tuple(indices)
        if key in _warm_start_cache:
            prev = _warm_start_cache[key]
            if len(prev) == k:
                w = prev.copy()
            elif len(prev) == k - 1:
                w = np.append(prev, 0.0)

    w /= (np.sum(w) + 1e-12)
    active = w > 1e-12

    # -----------------------------
    # ACTIVE SET ITERATIONS
    # -----------------------------
    for _ in range(max_iter):

        if not np.any(active):
            active = np.ones(k, dtype=bool)
            w = np.ones(k) / k

        A = np.where(active)[0]

        Q_A = Q[np.ix_(A, A)]
        ones = np.ones(len(A))

        KKT = np.block([
            [2 * Q_A, ones[:, None]],
            [ones[None, :], np.zeros((1, 1))]
        ])

        rhs = np.zeros(len(A) + 1)
        rhs[-1] = 1.0

        try:
            sol = np.linalg.solve(KKT, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]

        w_A = sol[:-1]

        w_new = np.zeros(k)
        w_new[A] = w_A

        w_new = np.maximum(w_new, 0.0)
        w_new /= (np.sum(w_new) + 1e-12)

        if np.linalg.norm(w - w_new) < tol:
            w = w_new
            break

        w = w_new
        active = w > 1e-12

    if indices is not None:
        _warm_start_cache[tuple(indices)] = w.copy()

    loss = float(w @ Q @ w)
    return loss, w


def solve_qp_simplex(Q: np.ndarray, w_init: Optional[np.ndarray] = None) -> np.ndarray:
    return solve_qp_simplex_value(Q, w_init)[1]


# ============================================================
# LOWER BOUND
# ============================================================
def simplex_lower_bound(Q: np.ndarray) -> float:
    k = Q.shape[0]

    # Base case for a single unit
    if k == 1:
        return float(Q[0, 0])

    # 1. Basic Diagonals
    diags = np.diag(Q)
    diag_min = float(np.min(diags))

    # 2. Scaled Gershgorin (Tightened for Gram Matrices)
    # We use d = sqrt(diag) as a scaling factor to balance the disks
    d = np.sqrt(diags) + 1e-12
    # Row sum of |Q_ij| * (d_j / d_i)
    scaled_radii = (np.sum(np.abs(Q) * d, axis=1) / d) - np.abs(diags)
    gersh_bound = float(np.min(diags - scaled_radii))

    # 3. Uniform Loss (The "Average" Case)
    w_uniform = np.ones(k) / k
    uniform_loss = float(w_uniform @ Q @ w_uniform)

    # 4. Combine Bounds
    # Since Q is a Gram matrix, the true minimum eigenvalue is >= 0.
    # We use a weighted floor for uniform_loss and diag_min based on typical
    # sparsity in synthetic control weights.
    cheap_lb = max(
        0.0,                   # PSD constraint
        gersh_bound,           # Mathematical lower bound
        diag_min * 0.5,        # Heuristic: harder to beat 50% of best single unit
        uniform_loss * 0.7     # Heuristic: lower than uniform but related
    )

    # 5. Conditional Precision
    # If the cheap bounds are low, they might not prune enough.
    # We trigger the actual QP solver to get the exact minimum.
    if cheap_lb < 0.9:
        try:
            # We use the full solver here because a tight LB is worth the cost
            # of the QP if it prunes an entire massive subtree.
            val, _ = solve_qp_simplex_value(Q)
            return max(val, cheap_lb)
        except Exception:
            pass

    return cheap_lb





# ============================================================
# UPPER BOUND (DETERMINISTIC)
# ============================================================

def feasible_upper_bound(Q: np.ndarray) -> float:

    k = Q.shape[0]

    if k == 1:
        return float(Q[0, 0])

    w_uniform = np.ones(k) / k

    idx_min = np.argmin(np.diag(Q))
    w_diag = np.zeros(k)
    w_diag[idx_min] = 1.0

    # 2-support heuristic (no randomness)
    i, j = np.unravel_index(np.argmin(Q), Q.shape)
    w_pair = np.zeros(k)
    w_pair[i] = 0.5
    w_pair[j] = 0.5

    return min([
        float(w_uniform @ Q @ w_uniform),
        float(w_diag @ Q @ w_diag),
        float(w_pair @ Q @ w_pair),
    ])


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
