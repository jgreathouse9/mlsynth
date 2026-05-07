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


# ============================================================
# UTILITIES
# ============================================================

def compute_search_space_size(M: int, m: int):
    total_subsets = comb(M, m)
    total_nodes = sum(comb(M, k) for k in range(1, m + 1))
    return total_subsets, total_nodes


def expand_weights_to_full(indices, weights, total_units):
    w_full = np.zeros(total_units)
    w_full[indices] = weights
    return w_full


# ============================================================
# STATS (REDESIGNED CLEAN MODEL)
# ============================================================

def make_stats():
    """
    Clean attribution-based stats.

    NO mixed denominators.
    NO internal/leaf ambiguity.
    Every prune is attributed to EXACTLY ONE cause.
    """

    bounds = ["diagonal", "fw", "inverse_rank", "spectral", "global", "sdp"]

    return {
        # ---- ground truth exploration ----
        "nodes_visited": 0,
        "branches_generated": 0,
        "leaves_solved": 0,
        "qp_calls": 0,

        # ---- pruning totals ----
        "node_prunes": 0,
        "branch_prunes": 0,

        # ---- per-bound attribution ----
        "bound_hits": {
            b: {
                "node": 0,
                "branch": 0,
            }
            for b in bounds
        },
    }


def hit(stats: Dict, bound: str, kind: str):
    """kind ∈ {'node', 'branch'}"""
    stats["bound_hits"][bound][kind] += 1

    if kind == "node":
        stats["node_prunes"] += 1
    else:
        stats["branch_prunes"] += 1


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
# PRESOLVE (UNCHANGED LOGIC)
# ============================================================

def presolve(pre: Precomputed, candidate_idx: np.ndarray, budget=None, unit_costs=None):
    idx = candidate_idx.copy()

    if budget is not None and unit_costs is not None:
        idx = idx[unit_costs[idx] <= budget]

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
# BOUNDS
# ============================================================

def diagonal_bound_Q(Q):
    return float(np.min(np.diag(Q)))


def spectral_lower_bound(Q):
    return float(np.linalg.eigvalsh(Q)[0])


def fw_completion_bound(pre, indices, remaining, max_iters=3):
    full_idx = np.concatenate([indices, remaining])
    G = pre.G[np.ix_(full_idx, full_idx)]
    n = len(full_idx)

    w = np.zeros(n)
    if len(indices) > 0:
        k = len(indices)
        w[:k] = 1.0 / k
    else:
        w[:] = 1.0 / n

    for t in range(max_iters):
        grad = 2 * G @ w
        j = np.argmin(grad)
        s = np.zeros(n)
        s[j] = 1.0
        gamma = 2.0 / (t + 2.0)
        w = (1 - gamma) * w + gamma * s

    return float(w @ G @ w)


def inverse_rank_bound(Q):
    k = Q.shape[0]
    if k == 1:
        return float(Q[0, 0])

    d = np.diag(Q)
    return float(np.min(d))


# ============================================================
# QP SOLVER
# ============================================================

def solve_qp(Q):
    global _qp_call_count, _warm_start_cache

    key = Q.tobytes()
    if key in _warm_start_cache:
        return _warm_start_cache[key]

    _qp_call_count += 1

    k = Q.shape[0]
    w = cp.Variable(k, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Q)), [cp.sum(w) == 1])
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
# GREEDY INIT (UNCHANGED LOGIC)
# ============================================================

def greedy_init(pre, candidate_idx, m):
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

def branch_score(pre, j, indices):
    if not indices:
        return -pre.G[j, j]
    return -pre.G[j, j] - np.mean([pre.G[j, i] for i in indices])


# ============================================================
# EXPAND (CLEAN ATTRIBUTION VERSION)
# ============================================================

def expand(pre, candidate_idx, m, top_K, top, indices, stats, Q):

    stats["nodes_visited"] += 1

    k = len(indices)
    ub = top[-1].loss if len(top) == top_K else np.inf

    # -------------------------
    # NODE PRUNE (diagonal)
    # -------------------------
    lb = diagonal_bound_Q(Q)
    if lb >= ub:
        hit(stats, "diagonal", "node")
        return

    # -------------------------
    # LEAF
    # -------------------------
    if k == m:
        loss, w = solve_qp(Q)
        stats["qp_calls"] += 1
        stats["leaves_solved"] += 1

        top.append(Solution(loss, indices[:], w))
        top.sort(key=lambda x: x.loss)
        if len(top) > top_K:
            top.pop()
        return

    # -------------------------
    # CHILDREN
    # -------------------------
    start = (
        np.searchsorted(candidate_idx, indices[-1]) + 1
        if indices else 0
    )

    remaining = candidate_idx[start:]
    ordered = sorted(remaining, key=lambda j: branch_score(pre, j, indices))

    for j in ordered:

        stats["branches_generated"] += 1

        k_new = k + 1
        Q_new = np.empty((k_new, k_new))
        Q_new[:k, :k] = Q

        if k > 0:
            g = pre.G[j, indices]
            Q_new[k, :k] = g
            Q_new[:k, k] = g

        Q_new[k, k] = pre.G[j, j]

        # -------------------------
        # FW BRANCH PRUNE
        # -------------------------
        lb_fw = fw_completion_bound(pre, indices + [j], remaining)
        if lb_fw >= ub:
            hit(stats, "fw", "branch")
            continue

        # -------------------------
        # IRB BRANCH PRUNE
        # -------------------------
        lb_irb = inverse_rank_bound(Q_new)
        if lb_irb >= ub:
            hit(stats, "inverse_rank", "branch")
            continue

        expand(pre, candidate_idx, m, top_K, top, indices + [j], stats, Q_new)
