from __future__ import annotations

import numpy as np
import cvxpy as cp
from dataclasses import dataclass, field
from math import comb
from typing import Any, Dict, List, Optional


# ============================================================
# GLOBAL STATE
# ============================================================

_qp_call_count = 0
_warm_start_cache = {}


def reset_qp_call_count():
    global _qp_call_count
    _qp_call_count = 0


def get_qp_call_count():
    return _qp_call_count


def clear_cache():
    global _qp_call_count, _warm_start_cache
    _qp_call_count = 0
    _warm_start_cache.clear()


def compute_search_space_size(M: int, m: int):
    total_subsets = comb(M, m)
    total_nodes = sum(comb(M, k) for k in range(1, m + 1))
    return total_subsets, total_nodes


def expand_weights_to_full(indices, weights, total_units):
    w_full = np.zeros(total_units)
    w_full[indices] = weights
    return w_full


# ============================================================
# PRECOMPUTED STRUCTURE
# ============================================================

class Precomputed:
    def __init__(self, G: np.ndarray):
        self.G = G
        self.n = G.shape[0]
        self.lam_min_global = float(np.linalg.eigvalsh(G)[0])


# ============================================================
# SOLUTION
# ============================================================

@dataclass(order=True)
class Solution:
    loss: float
    indices: List[int] = field(compare=False)
    weights: np.ndarray = field(compare=False)


# ============================================================
# PRESOLVE
# ============================================================

def presolve(pre: Precomputed, candidate_idx: np.ndarray, budget=None, unit_costs=None):

    idx = candidate_idx.copy()

    if budget is not None and unit_costs is not None:
        idx = idx[unit_costs[idx] <= budget]

    # simple variance filter (kept as heuristic ONLY for search space reduction)
    diag = np.diag(pre.G)[idx]
    med = np.median(diag)
    idx = idx[diag <= med * 30]

    keep = []
    for i in idx:
        if not keep:
            keep.append(i)
            continue

        if all(
            pre.G[i, j] / np.sqrt(pre.G[i, i] * pre.G[j, j]) < 0.9995
            for j in keep[-10:]
        ):
            keep.append(i)

    return np.array(keep)


# ============================================================
# BOUNDS (UPDATED: SIMPLEX-AWARE)
# ============================================================

def global_lower_bound(pre: Precomputed, m: int) -> float:
    return max(0.0, pre.lam_min_global / m)


def diagonal_bound_Q(Q: np.ndarray):
    return float(np.min(np.diag(Q)))


def spectral_lower_bound(Q: np.ndarray):
    return float(np.linalg.eigvalsh(Q)[0])


def affine_relaxation_bound(Q: np.ndarray):
    """
    Tight lower bound using:
        min w^T Q w  s.t. sum(w)=1 (no nonnegativity)

    This captures the simplex structure MUCH better than eigenvalue bounds.
    """
    k = Q.shape[0]

    if k == 1:
        return float(Q[0, 0])

    ones = np.ones(k)

    try:
        # Solve Qx = 1 (more stable than inverse)
        x = np.linalg.solve(Q, ones)
        denom = ones @ x

        if denom <= 1e-12:
            return 0.0

        val = 1.0 / denom
        return float(max(0.0, val))

    except np.linalg.LinAlgError:
        return spectral_lower_bound(Q)


def completion_cross_bound(pre: Precomputed, Q: np.ndarray, remaining: np.ndarray):

    # current subset relaxation
    f_S = affine_relaxation_bound(Q)

    if len(remaining) == 0:
        return f_S

    diag = np.diag(pre.G)
    d_R = float(np.min(diag[remaining]))

    # cross-term approximation
    cross_vals = []
    for i in range(Q.shape[0]):
        cross_vals.append(np.min(pre.G[i, remaining]))

    c_bar = float(np.min(cross_vals))

    # avoid degenerate cases
    if d_R <= 0:
        return 0.0

    den = f_S + d_R - 2 * c_bar
    if abs(den) < 1e-12:
        return min(f_S, d_R)

    alpha = (d_R - c_bar) / den
    alpha = np.clip(alpha, 0.0, 1.0)

    return (
        (f_S + d_R - 2*c_bar) * alpha**2
        + 2*(c_bar - d_R) * alpha
        + d_R
    )


def sdp_bound(Q: np.ndarray):
    """
    (Optional) SDP relaxation.
    Kept for experimentation, but usually unnecessary now.
    """
    k = Q.shape[0]

    X = cp.Variable((k, k), PSD=True)

    constraints = [
        cp.trace(X) == 1,
        cp.sum(X) == 1,
    ]

    prob = cp.Problem(cp.Minimize(cp.trace(Q @ X)), constraints)
    prob.solve()

    if X.value is None:
        return spectral_lower_bound(Q)

    return float(prob.value)


# ============================================================
# QP SOLVER (CACHED)
# ============================================================


def solve_qp(Q: np.ndarray):
    global _qp_call_count, _warm_start_cache

    # Better cache key (faster + safer)
    key = Q.tobytes()

    if key in _warm_start_cache:
        return _warm_start_cache[key]

    _qp_call_count += 1

    k = Q.shape[0]
    w = cp.Variable(k, nonneg=True)

    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, Q)),
        [cp.sum(w) == 1]
    )

    prob.solve(solver=cp.OSQP, verbose=False)

    if w.value is None:
        best = np.argmin(np.diag(Q))
        w_out = np.zeros(k)
        w_out[best] = 1.0
        result = (Q[best, best], w_out)
    else:
        w_out = np.maximum(w.value, 0)
        w_out /= w_out.sum() + 1e-12
        result = (float(prob.value), w_out)

    _warm_start_cache[key] = result
    return result




# ============================================================
# GREEDY INITIALIZATION
# ============================================================

def greedy_init(pre: Precomputed, candidate_idx: np.ndarray, m: int):

    selected = []
    Q = None

    for _ in range(m):

        best_j, best_loss, best_Q = None, np.inf, None
        remaining = candidate_idx[~np.isin(candidate_idx, selected)]

        scores = -np.diag(pre.G)[remaining]
        order = np.argsort(scores)[::-1]

        for idx in order[:min(20, len(order))]:
            j = remaining[idx]

            if not selected:
                Q_new = np.array([[pre.G[j, j]]])
            else:
                k = len(selected)
                Q_new = np.empty((k + 1, k + 1))
                Q_new[:k, :k] = Q
                g = pre.G[j, selected]
                Q_new[k, :k] = g
                Q_new[:k, k] = g
                Q_new[k, k] = pre.G[j, j]

            loss, _ = solve_qp(Q_new)

            if loss < best_loss:
                best_loss, best_j, best_Q = loss, j, Q_new

        selected.append(best_j)
        Q = best_Q

    loss, w = solve_qp(Q)
    return selected, loss, w


# ============================================================
# BRANCH SCORE
# ============================================================

def branch_score(pre: Precomputed, j: int, indices: List[int]):
    if not indices:
        return -pre.G[j, j]

    return -pre.G[j, j] - np.mean([pre.G[j, i] for i in indices])


# ============================================================
# BnB EXPAND (FULL FIXED VERSION)
# ============================================================

def expand(
    pre: Precomputed,
    candidate_idx: np.ndarray,
    m: int,
    top_K: int,
    top: List[Solution],
    indices: List[int],
    stats: Dict,
    Q: np.ndarray,
):

    stats["nodes_visited"] += 1

    k = len(indices)
    ub = top[-1].loss if len(top) == top_K else np.inf

    # ========================================================
    # LEVEL 1: DIAGONAL (SAFE LOCAL BOUND)
    # ========================================================
    if diagonal_bound_Q(Q) >= ub:
        stats["branches_pruned"] += 1
        return

    # ========================================================
    # LEAF
    # ========================================================
    if k == m:
        loss, w = solve_qp(Q)

        top.append(Solution(loss, indices[:], w))
        top.sort(key=lambda x: x.loss)

        if len(top) > top_K:
            top.pop()

        stats["subsets_evaluated"] += 1
        stats["leaf_nodes"] += 1
        return

    # ========================================================
    # CHILDREN
    # ========================================================
    start = (
        np.searchsorted(candidate_idx, indices[-1]) + 1
        if indices else 0
    )

    remaining = candidate_idx[start:]
    ordered = sorted(remaining, key=lambda j: branch_score(pre, j, indices))

    for j in ordered:

        stats["branches_considered"] += 1

        k_new = k + 1
        Q_new = np.empty((k_new, k_new))
        Q_new[:k, :k] = Q

        if k > 0:
            g = pre.G[j, indices]
            Q_new[k, :k] = g
            Q_new[:k, k] = g

        Q_new[k, k] = pre.G[j, j]

        # ====================================================
        # LEVEL 2: SIMPLEX-AWARE BOUND (MAIN PRUNER)
        # ====================================================
        lb = completion_cross_bound(pre, Q_new, remaining)

        if lb >= ub:
            stats["branches_pruned"] += 1
            continue

        lb_sdp = sdp_bound(Q_new)
        if lb_sdp >= ub:
            stats["branches_pruned"] += 1
            continue

        expand(
            pre,
            candidate_idx,
            m,
            top_K,
            top,
            indices + [j],
            stats,
            Q_new,
        )
