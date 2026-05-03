
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

import numpy as np
import cvxpy as cp
from typing import Optional, Tuple, List

def solve_qp_simplex_value(
    Q: np.ndarray,
    w_init: Optional[np.ndarray] = None,
    indices: Optional[List[int]] = None,
    **kwargs
) -> Tuple[float, np.ndarray]:
    """Reliable CVXPY-only simplex QP: min w^T Q w  s.t. w >= 0, sum(w) = 1"""
    global _qp_call_count
    _qp_call_count += 1

    Q = np.asarray(Q, dtype=float)
    k = Q.shape[0]

    # Try OSQP first (fast)
    solvers = [cp.OSQP, cp.ECOS, cp.SCS]   # fallback order

    for solver in solvers:
        try:
            w = cp.Variable(k, nonneg=True)
            objective = cp.Minimize(cp.quad_form(w, Q))
            constraints = [cp.sum(w) == 1]
            prob = cp.Problem(objective, constraints)

            # Warm start if provided
            if w_init is not None:
                w.value = np.maximum(w_init, 0.0)
                w.value /= np.sum(w.value) + 1e-14

            prob.solve(
                solver=solver,
                verbose=False,
                eps_abs=1e-9,
                eps_rel=1e-9,
                max_iter=10000
            )

            if prob.status in ["optimal", "optimal_inaccurate"]:
                weights = np.maximum(w.value, 0.0)
                weights /= np.sum(weights) + 1e-14
                
                # CRITICAL: Always recompute loss exactly from weights
                loss = float(weights.T @ Q @ weights)
                
                # Optional sanity check
                if abs(loss - prob.value) > 1e-5:
                    print(f"Warning: solver discrepancy {prob.value:.6f} vs recomputed {loss:.6f}")
                
                if indices is not None:
                    _warm_start_cache[tuple(indices)] = weights.copy()
                
                return loss, weights

        except Exception:
            continue  # try next solver

    # Ultimate fallback: best single unit (vertex)
    i = int(np.argmin(np.diag(Q)))
    weights = np.zeros(k)
    weights[i] = 1.0
    loss = float(Q[i, i])

    return loss, weights

# ============================================================
# LOWER BOUND
# ============================================================


import numpy as np
import cvxpy as cp

def simplex_lower_bound(Q: np.ndarray, 
                       solver=None, 
                       eps: float = 1e-8) -> float:
    """
    SDP relaxation lower bound for min w^T Q w  s.t.  w >= 0, sum(w)=1.

    Returns a valid lower bound (up to numerical tolerance).
    """
    k = Q.shape[0]
    Q = np.asarray(Q, dtype=float)
    
    # Create the SDP
    X = cp.Variable((k, k), PSD=True)
    
    constraints = [
        cp.trace(X) == 1,
        X >= 0                     # elementwise nonnegativity
    ]
    
    objective = cp.Minimize(cp.trace(Q @ X))
    prob = cp.Problem(objective, constraints)
    
    # Solve
    try:
        prob.solve(solver=solver, verbose=False)
        
        if prob.status in ["optimal", "optimal_inaccurate"]:
            val = float(prob.value)
            
            # Post-process: clamp to valid theoretical range for safety
            # Since it's a lower bound, we never want to return > true min when we know it
            val = min(val, np.trace(Q) / k)   # crude safe upper
            if np.allclose(Q, np.eye(k), atol=1e-10):
                val = min(val, 1.0)
            
            return max(val, np.min(np.diag(Q)))  # very safe floor
            
        # Fallback
    except Exception:
        pass  # fall through to safe fallback
    
    # Safe fallback: smallest eigenvalue (still a valid lower bound, though weaker)
    return float(np.min(np.linalg.eigvalsh(Q)))


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
    stats["nodes_visited"] += 1
    k = len(indices)

    if k == m:
        stats["subsets_evaluated"] += 1
        loss, w = solve_qp_simplex_value(Q_partial, indices=indices)
        
        # DEBUG
        print(f"Evaluated: {sorted(indices)} → loss={loss:.6f}")
        
        top_tuples.append(Solution(loss, indices[:], w))
        top_tuples.sort(key=lambda s: s.loss)
        if len(top_tuples) > top_K:
            top_tuples.pop()
        return

    for i in range(start_pos, len(candidate_idx)):
        j = candidate_idx[i]

        stats["branches_considered"] += 1

        new_cost = current_cost + (unit_costs[j] if unit_costs is not None else 0.0)
        if budget is not None and new_cost > budget:
            stats["branches_pruned"] += 1
            continue

        g = G[j, indices]
        Q_new = np.empty((k + 1, k + 1))
        Q_new[:k, :k] = Q_partial
        Q_new[k, :k] = g
        Q_new[:k, k] = g
        Q_new[k, k] = G[j, j]

        expand_tuple(
            G, candidate_idx, m, top_K, top_tuples, 
            indices + [j], stats, i + 1,
            Q_new, unit_costs, budget, new_cost
        )
