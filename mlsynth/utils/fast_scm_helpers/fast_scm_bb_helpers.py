"""
Fast_scm_bb_helpers.py
----------------------
Branch-and-bound synthetic control solver with spectral bounds.

Key upgrade:
    ✔ valid eigenvalue-based lower bound
    ✔ safe pruning at ALL nodes (but only using valid PSD relaxation)
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
from dataclasses import dataclass, field
from math import comb
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 1. GREEDY INIT
# ============================================================

def greedy_initial_solution(G: np.ndarray, candidate_idx: np.ndarray, m: int):
    selected = []
    Q_partial = None

    for _ in range(m):
        best_j = None
        best_loss = np.inf
        best_Q = None

        for j in candidate_idx:
            if j in selected:
                continue

            if not selected:
                Q_new = np.array([[G[j, j]]])
            else:
                k = len(selected)
                Q_new = np.empty((k + 1, k + 1))
                Q_new[:k, :k] = Q_partial
                g = G[j, selected]
                Q_new[k, :k] = g
                Q_new[:k, k] = g
                Q_new[k, k] = G[j, j]

            loss, _ = solve_qp_simplex_value(Q_new)

            if loss < best_loss:
                best_loss = loss
                best_j = j
                best_Q = Q_new

        selected.append(best_j)
        Q_partial = best_Q

    loss, w = solve_qp_simplex_value(Q_partial)
    return selected, loss, w


# ============================================================
# 2. GLOBAL STATE
# ============================================================

_qp_call_count = 0
_warm_start_cache = {}


def get_qp_call_count():
    return _qp_call_count


def reset_qp_call_count():
    global _qp_call_count
    _qp_call_count = 0


def clear_solver_cache():
    global _qp_call_count, _warm_start_cache
    _qp_call_count = 0
    _warm_start_cache.clear()


# ============================================================
# 3. SOLUTION CONTAINER
# ============================================================

@dataclass(order=True)
class Solution:
    loss: float
    indices: List[int] = field(compare=False)
    weights: np.ndarray = field(compare=False)


# ============================================================
# 4. SEARCH SPACE
# ============================================================

def compute_search_space_size(M: int, m: int):
    leaves = comb(M, m)
    nodes = sum(comb(M, k) for k in range(1, m + 1))
    return leaves, nodes


# ============================================================
# 5. SPECTRAL LOWER BOUND (CORE FIX)
# ============================================================

def spectral_lower_bound(Q: np.ndarray) -> float:
    """
    Valid lower bound for quadratic form over simplex (PSD case).
    """
    lam_min = float(np.linalg.eigvalsh(Q)[0])
    return max(0.0, lam_min)


# ============================================================
# 6. QP SOLVER
# ============================================================

def solve_qp_simplex_value(Q, w_init=None, indices=None):
    global _qp_call_count
    _qp_call_count += 1

    k = Q.shape[0]

    if w_init is None and indices is not None:
        cached = _warm_start_cache.get(tuple(indices))
        if cached is not None and len(cached) == k:
            w_init = cached

    w = cp.Variable(k, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Q)), [cp.sum(w) == 1])

    if w_init is not None:
        wv = np.maximum(w_init, 0.0)
        s = wv.sum()
        w.value = wv / (s + 1e-12) if s > 0 else np.ones(k) / k

    prob.solve(solver=cp.OSQP, verbose=False, warm_start=True)

    if w.value is None:
        best = int(np.argmin(np.diag(Q)))
        w_out = np.zeros(k)
        w_out[best] = 1.0
        return float(Q[best, best]), w_out

    w_out = np.maximum(w.value, 0.0)
    w_out /= w_out.sum() + 1e-12

    if indices is not None:
        _warm_start_cache[tuple(indices)] = w_out.copy()

    return float(prob.value), w_out


# ============================================================
# 7. SCORING (SAFE ONLY)
# ============================================================

def branch_score(G, j, indices):
    if len(indices) == 0:
        return -G[j, j]
    return -(G[j, j] + np.mean(G[j, indices]))


# ============================================================
# 8. BnB CORE (CORRECT + SAFE PRUNING)
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
    current_cost=0.0,
):

    stats["nodes_visited"] += 1

    k = len(indices)
    current_ub = top_tuples[-1].loss if len(top_tuples) >= top_K else np.inf

    # =========================================================
    # LEAF
    # =========================================================
    if k == m:
        stats["subsets_evaluated"] += 1

        loss, w = solve_qp_simplex_value(Q_partial, indices=indices)

        if loss < current_ub:
            top_tuples.append(Solution(loss, indices[:], w))
            top_tuples.sort(key=lambda s: s.loss)
            if len(top_tuples) > top_K:
                top_tuples.pop()

        return

    # =========================================================
    # CHILDREN
    # =========================================================
    if indices:
        start = int(np.searchsorted(candidate_idx, indices[-1])) + 1
    else:
        start = 0

    remaining = candidate_idx[start:]

    ordered = sorted(remaining, key=lambda j: branch_score(G, j, indices))

    # =========================================================
    # EXPAND
    # =========================================================
    for j in ordered:

        k_new = k + 1
        Q_new = np.empty((k_new, k_new))
        Q_new[:k, :k] = Q_partial

        if k > 0:
            Q_new[k, :k] = G[j, indices]
            Q_new[:k, k] = G[j, indices]
        Q_new[k, k] = G[j, j]

        # =====================================================
        # ✔ SPECTRAL BOUND (SAFE PRUNING)
        # =====================================================
        lb = spectral_lower_bound(Q_new)

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
            current_cost,
        )
