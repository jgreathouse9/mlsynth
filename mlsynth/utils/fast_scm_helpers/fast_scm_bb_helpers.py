import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from math import comb
import cvxpy as cp

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
    return comb(M, m), sum(comb(M, k) for k in range(1, m + 1))


# ============================================================
# PRESOLVE
# ============================================================

def presolve_candidates(G, candidate_idx, budget=None, unit_costs=None, m=1):
    candidate_idx = np.array(candidate_idx)

    if budget is not None and unit_costs is not None:
        candidate_idx = candidate_idx[unit_costs[candidate_idx] <= budget]

    if len(candidate_idx) == 0:
        return candidate_idx

    diags = np.diag(G)[candidate_idx]
    threshold = np.median(diags) * 50
    candidate_idx = candidate_idx[np.diag(G)[candidate_idx] < threshold]

    if unit_costs is not None:
        to_remove = set()
        for i in range(len(candidate_idx)):
            u_i = candidate_idx[i]
            if u_i in to_remove:
                continue
            for j in range(i + 1, len(candidate_idx)):
                u_j = candidate_idx[j]
                if u_j in to_remove:
                    continue

                corr = G[u_i, u_j] / (np.sqrt(G[u_i, u_i] * G[u_j, u_j]) + 1e-12)
                if corr > 0.999:
                    remove = u_j if unit_costs[u_j] >= unit_costs[u_i] else u_i
                    to_remove.add(remove)

        candidate_idx = np.array([c for c in candidate_idx if c not in to_remove])

    return candidate_idx


# ============================================================
# LOWER BOUNDS
# ============================================================

def simplex_lower_bound(Q: np.ndarray) -> float:
    k = Q.shape[0]
    if k == 1:
        return float(Q[0, 0])

    eig_min = np.linalg.eigvalsh(Q)[0]
    return float(max(0.0, eig_min / k))


def lookahead_lower_bound(G, current_lb, remaining_candidates, slots_left, m):
    if slots_left <= 0 or len(remaining_candidates) == 0:
        return current_lb

    best = np.sort(np.diag(G)[remaining_candidates])[:max(1, slots_left)]
    avg_best = np.mean(best)

    k = m - slots_left
    return max(current_lb, (k / m) * current_lb + (slots_left / m) * avg_best)


# ============================================================
# QP SOLVER
# ============================================================

def solve_qp_simplex_value(Q: np.ndarray, w_init=None, indices=None):
    global _qp_call_count
    _qp_call_count += 1

    k = Q.shape[0]
    w = cp.Variable(k, nonneg=True)

    objective = cp.quad_form(w, Q)
    prob = cp.Problem(cp.Minimize(objective), [cp.sum(w) == 1])

    if w_init is not None:
        w.value = np.maximum(w_init, 0)
        w.value /= np.sum(w.value) + 1e-12

    prob.solve(solver=cp.OSQP, verbose=False)

    if w.value is None or prob.status not in ["optimal", "optimal_inaccurate"]:
        idx = np.argmin(np.diag(Q))
        w_out = np.zeros(k)
        w_out[idx] = 1.0
        return float(Q[idx, idx]), w_out

    w_out = np.maximum(w.value, 0)
    w_out /= np.sum(w_out) + 1e-12

    if indices is not None:
        _warm_start_cache[tuple(indices)] = w_out.copy()

    return float(prob.value), w_out


# ============================================================
# INIT + UTILS
# ============================================================

def greedy_initial_solution(G, candidate_idx, m):
    selected = list(candidate_idx[:m])
    Q = G[np.ix_(selected, selected)]

    loss, w = solve_qp_simplex_value(Q, indices=selected)

    return loss, selected, w

def expand_weights_to_full(indices, weights, total_units):
    w = np.zeros(total_units)
    w[indices] = weights
    return w


# ============================================================
# SCORING
# ============================================================

def strong_branch_score(G, Q_partial, candidate_idx, j, indices):
    if len(indices) == 0:
        return -G[j, j]

    g = G[j, indices]
    return -G[j, j] - 2.0 * np.mean(g)


def prune_by_correlation(G, candidate_idx, threshold=0.999):
    keep = []
    removed = set()

    for i in candidate_idx:
        if i in removed:
            continue
        keep.append(i)
        for j in candidate_idx:
            if i != j:
                corr = G[i, j] / (np.sqrt(G[i, i] * G[j, j]) + 1e-12)
                if corr > threshold:
                    removed.add(j)

    return np.array(keep)


def prune_by_cost(candidate_idx, unit_costs, budget):
    if budget is None:
        return candidate_idx
    return np.array([i for i in candidate_idx if unit_costs[i] <= budget])


# ============================================================
# BnB CORE (FIXED COMBINATORIAL LOGIC)
# ============================================================

def expand_tuple(
    G,
    candidate_idx,
    m,
    top_K,
    top_tuples,
    indices,
    stats,
    Q_partial,
    unit_costs=None,
    budget=None,
    current_cost=0.0,
):
    stats["nodes_visited"] += 1

    k = len(indices)
    slots_left = m - k

    current_ub = top_tuples[-1].loss if len(top_tuples) == top_K else np.inf

    # -----------------------------
    # BASE CASE
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
    # COMBINATORIAL EXPANSION (FIXED)
    # -----------------------------
    last = indices[-1] if len(indices) > 0 else -1

    for j in candidate_idx:
        if j <= last:
            continue

        stats["branches_considered"] += 1

        new_cost = current_cost + (unit_costs[j] if unit_costs is not None else 0.0)
        if budget is not None and new_cost > budget:
            stats["branches_pruned"] += 1
            continue

        k_new = k + 1
        Q_new = np.empty((k_new, k_new))

        Q_new[:k, :k] = Q_partial

        g = G[j, indices] if k > 0 else np.array([])

        Q_new[k, :k] = g
        Q_new[:k, k] = g
        Q_new[k, k] = G[j, j]

        lb_node = simplex_lower_bound(Q_new)

        if slots_left > 1:
            remaining = np.array([x for x in candidate_idx if x > j])
            lb_pred = lookahead_lower_bound(G, lb_node, remaining, slots_left - 1, m)
            lb = max(lb_node, lb_pred)
        else:
            lb = lb_node

        if lb >= current_ub:
            stats["branches_pruned"] += 1
            continue

        expand_tuple(
            G,
            candidate_idx,
            m,
            top_K,
            top_tuples,
            indices + [j],
            stats,
            Q_new,
            unit_costs,
            budget,
            new_cost,
        )
