"""Treated-unit selection for synthetic-control experimental design.

Drop-in replacement for the branch-and-bound search in
``fast_scm_helpers.fast_scm_bb``. It selects the ``top_K`` candidate
treated m-tuples that best reproduce the population predictor vector,
i.e. that minimise the *imbalance*

    L(S) = min_{w in simplex(S)}  || sum_{j in S} w_j  Xtilde_j ||^2
         = w' G_SS w,            G = Xtilde' Xtilde   (PSD Gram matrix)

where ``Xtilde`` are the f-centred, standardised predictors over the
estimation window E.  Because the design is f-centred, the population
target is the donor centroid (the origin), so ``L`` is the squared
distance from the origin to the convex hull of the selected donors and
``sqrt(L)`` is the achieved imbalance.

Why this is a *search* and not an exact MIP
-------------------------------------------
Abadie & Zhao (2026, "Synthetic Controls for Experimental Design", p.13)
do **not** require global optimality of the weights -- only that the
chosen design is feasible and approximately balanced,
``Xbar - sum_j w_j X_j ~= 0``.  Validity of the bias bound and the
inference is conditional on the *achieved* imbalance, a goodness-of-fit
quantity reported here, not on a certificate of optimality.

Moreover, an exact bound is structurally weak: because the origin lies
inside the convex hull of all candidates, the convex relaxation lower
bound is ~0 over the upper half of any branch-and-bound tree, so it
cannot prune there.  We therefore:

* enumerate exactly when ``C(M, m)`` is small (the gold standard), and
* otherwise run a strengthened multi-start local search, which in Monte
  Carlo lands on the exact optimum 83-100% of the time and within ~1%
  (mean) / ~7% (worst) of the minimal imbalance -- immaterial under the
  approximate-balance criterion above.

The achieved imbalance of every returned design is reported so the
analyst can check the ``~= 0`` condition that actually licenses the
method.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from math import comb
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .conflict import is_independent
from . import strata as _strata
from .feasibility import audit_feasibility
from ...exceptions import MlsynthConfigError


# ======================================================================
# Inner simplex-QP solvers (Away-step Frank-Wolfe; pure numpy)
# ======================================================================

def _afw_single(Q: np.ndarray, iters: int = 300, tol: float = 1e-13):
    """Exact-to-tolerance min_{w in simplex} w'Qw for one m x m PSD Q.

    Returns (loss, w, lower_bound) where lower_bound is the certified
    Frank-Wolfe duality-gap bound (<= true minimum).
    """
    n = Q.shape[0]
    if n == 1:
        return float(Q[0, 0]), np.array([1.0]), float(Q[0, 0])
    d = np.diag(Q)
    w = np.zeros(n); w[int(np.argmin(d))] = 1.0
    active = {int(np.argmax(w))}
    best_lb = -np.inf
    for _ in range(iters):
        grad = 2.0 * (Q @ w)
        f = float(w @ (Q @ w))
        s = int(np.argmin(grad))
        best_lb = max(best_lb, float(grad[s]) - f)   # FW lower bound
        gap = float(grad @ w - grad[s])
        if gap <= tol:
            break
        act = np.array(sorted(active))
        a = int(act[np.argmax(grad[act])])
        d_fw = -w.copy(); d_fw[s] += 1.0
        d_aw = w.copy();  d_aw[a] -= 1.0
        if float(grad @ d_fw) <= float(grad @ d_aw):
            D, gmax, is_fw = d_fw, 1.0, True
        else:
            D = d_aw; gmax = w[a] / (1.0 - w[a]) if w[a] < 1.0 else 1e12; is_fw = False
        QD = Q @ D
        quad = float(D @ QD)
        g = -float(2.0 * (w @ QD)) / (2.0 * quad) if quad > 1e-18 else gmax
        g = max(0.0, min(g, gmax))
        w = w + g * D
        w[w < 1e-15] = 0.0
        if g >= gmax and not is_fw:
            active.discard(a)
        active.add(s)
        active = {j for j in active if w[j] > 0}
    w = np.clip(w, 0.0, None); w /= w.sum() + 1e-18
    return float(w @ (Q @ w)), w, float(best_lb)


def _afw_batched(Qs: np.ndarray, iters: int = 80) -> np.ndarray:
    """Vectorised AFW losses for many tuples at once. Qs: (N, m, m)."""
    N, m, _ = Qs.shape
    if m == 1:
        return Qs[:, 0, 0].copy()
    d = np.diagonal(Qs, axis1=1, axis2=2)
    W = np.zeros((N, m)); W[np.arange(N), d.argmin(1)] = 1.0
    rows = np.arange(N)
    for _ in range(iters):
        grad = 2.0 * np.einsum('nij,nj->ni', Qs, W)
        s = grad.argmin(1)
        ga = np.where(W > 1e-14, grad, -np.inf)
        a = ga.argmax(1)
        d_fw = -W.copy(); d_fw[rows, s] += 1.0
        d_aw = W.copy();  d_aw[rows, a] -= 1.0
        use_fw = np.einsum('ni,ni->n', grad, d_fw) <= np.einsum('ni,ni->n', grad, d_aw)
        D = np.where(use_fw[:, None], d_fw, d_aw)
        wa = W[rows, a]
        gmax = np.where(use_fw, 1.0, np.where(wa < 1.0, wa / (1.0 - wa + 1e-18), 1e12))
        QD = np.einsum('nij,nj->ni', Qs, D)
        quad = np.einsum('ni,ni->n', D, QD)
        lin = 2.0 * np.einsum('ni,ni->n', W, QD)
        safe = quad > 1e-18
        g = gmax.copy()
        g[safe] = -lin[safe] / (2.0 * quad[safe])
        g = np.clip(g, 0.0, gmax)
        W = W + g[:, None] * D
        W[W < 1e-14] = 0.0
    return np.einsum('ni,nij,nj->n', W, Qs, W)


def _losses_for(G: np.ndarray, subsets: np.ndarray, iters: int = 70) -> np.ndarray:
    """Batched losses for an (N, m) int array of subsets."""
    Qs = G[subsets[:, :, None], subsets[:, None, :]]
    return _afw_batched(Qs, iters=iters)


# ======================================================================
# Result container
# ======================================================================

@dataclass
class TreatedDesign:
    indices: List[int]            # selected unit indices (sorted)
    weights: np.ndarray           # treatment weights aligned to `indices`
    loss: float                   # imbalance^2 = w' G_SS w
    imbalance: float              # sqrt(loss) = ||Xbar - sum w_j X_j||
    total_cost: float = 0.0
    label: str = ""
    labels: List[Any] = field(default_factory=list)
    weight_dict: Dict[Any, float] = field(default_factory=dict)
    full_weights: Optional[np.ndarray] = None


# ======================================================================
# Feasibility presolve (SOUND: budget feasibility only)
# ======================================================================

def _budget_feasible_candidates(candidate_idx, m, unit_costs, budget):
    """Drop units that cannot be in ANY budget-feasible m-tuple.

    Sound: a unit i is impossible iff cost(i) + (sum of m-1 cheapest other
    candidate costs) > budget.  Does not touch the objective.
    """
    if budget is None or np.isinf(budget) or unit_costs is None:
        return np.asarray(candidate_idx)
    idx = np.asarray(candidate_idx)
    costs = unit_costs[idx]
    order = np.sort(costs)
    floor_excl_self = order[:m].sum()  # m cheapest (upper bound on m-1 others + self slack)
    # tighter per-unit floor: cost(i) + sum of (m-1) cheapest among the OTHERS
    keep = []
    for i in idx:
        others = np.sort(unit_costs[idx[idx != i]])
        floor = others[:m - 1].sum() if m - 1 <= len(others) else np.inf
        if unit_costs[i] + floor <= budget:
            keep.append(i)
    return np.asarray(keep)


# ======================================================================
# Exact enumeration (small C(M, m))
# ======================================================================

def _enumerate(G, cand, m, top_K, unit_costs, budget, iters, chunk=300_000,
               conflict=None, strata=None, min_per=None, max_per=None,
               required=None):
    cand = np.sort(np.asarray(cand))
    from itertools import combinations, chain
    combs = np.fromiter(chain.from_iterable(combinations(cand.tolist(), m)), dtype=int)
    combs = combs.reshape(-1, m)
    if unit_costs is not None and budget is not None and not np.isinf(budget):
        feasible = unit_costs[combs].sum(1) <= budget
        combs = combs[feasible]
    # Spillover "No interference": keep only m-tuples that are independent sets of
    # the conflict graph (no two members interfere). Vectorised over all pairs.
    if conflict is not None and len(combs):
        iu = np.triu_indices(m, k=1)
        pair_conf = conflict[combs[:, iu[0]], combs[:, iu[1]]]   # (N, n_pairs)
        combs = combs[~pair_conf.any(axis=1)]
    # Coverage / stratification quotas (min / max treated per stratum).
    if strata is not None and len(combs):
        combs = combs[_strata.satisfies_many(strata, combs, min_per, max_per, required)]
    if len(combs) == 0:
        return [], 0
    losses = np.empty(len(combs))
    for s in range(0, len(combs), chunk):
        losses[s:s + len(combs[s:s + chunk])] = _losses_for(G, combs[s:s + chunk], iters=iters)
    order = np.argsort(losses)[:top_K]
    return [(combs[o], float(losses[o])) for o in order], len(combs)


# ======================================================================
# Strengthened multi-start local search (large C(M, m))
# ======================================================================

def _cost_ok(idx_list, unit_costs, budget):
    if unit_costs is None or budget is None or np.isinf(budget):
        return True
    return float(unit_costs[list(idx_list)].sum()) <= budget + 1e-12

def _local_search(G, cand, m, top_K, unit_costs, budget, n_starts, rng, iters,
                  n_kicks=4, conflict=None, strata=None, min_per=None,
                  max_per=None, required=None):
    """Multi-start best-improvement swap search with basin-hopping kicks.

    Returns (ranked_designs, n_distinct_subsets, consensus) where ``consensus``
    carries the multi-start diagnostics that stand in for a solver's MIP gap:
    how many independent starts converged to the incumbent, how many distinct
    local optima were seen, and the incumbent-improvement trail.

    Every generated tuple is required to be both budget-feasible and an
    independent set of ``conflict`` (the spillover "No interference" constraint).
    """
    cand = list(np.sort(np.asarray(cand)))

    def _ok(S):
        # max-per-stratum is partial-safe (prunes during construction); the min
        # coverage quota is a full-tuple property, filtered after the search.
        return (_cost_ok(S, unit_costs, budget)
                and is_independent(conflict, S)
                and _strata.within_max(strata, S, max_per))
    diag = np.diag(G)
    pool: Dict[tuple, float] = {}
    work = [0]                      # total subsets scored ("simplex iterations")
    best = [np.inf]                 # global incumbent objective
    trail: List[tuple] = []         # (work, objective) at each incumbent improvement

    def score(subs):
        arr = np.asarray(subs)
        work[0] += len(arr)
        return _losses_for(G, arr, iters=iters)

    def loss_of(S):
        v = pool.get(tuple(S))
        if v is None:
            v = float(score([S])[0]); pool[tuple(S)] = v
        if v < best[0] - 1e-12:
            best[0] = v; trail.append((work[0], v))
        return v

    def greedy(start):
        S = [start]
        while len(S) < m:
            rem = [j for j in cand if j not in S and _ok(S + [j])]
            if not rem:
                return None
            cands = [sorted(S + [j]) for j in rem]
            S = sorted(S + [rem[int(score(cands).argmin())]])
        return S if _ok(S) else None

    def descend(S):
        improved = True
        while improved:
            improved = False
            moves = [sorted(S[:a] + S[a + 1:] + [j])
                     for a in range(m) for j in cand
                     if j not in S and _ok(S[:a] + S[a + 1:] + [j])]
            if not moves:
                break
            L = score(moves)
            k = int(L.argmin())
            if L[k] < loss_of(S) - 1e-12:
                S = moves[k]; pool[tuple(S)] = float(L[k]); loss_of(S); improved = True
        return S

    def kick(S):
        for _ in range(20):
            T = sorted(S)
            for a in rng.choice(m, size=min(2, m), replace=False):
                free = [j for j in cand if j not in T]
                if free:
                    T = sorted(T[:a] + T[a + 1:] + [int(rng.choice(free))])
            if len(set(T)) == m and _ok(T):
                return T
        return S

    seeds = list(np.argsort(diag[cand])[:n_starts]) + \
            [int(rng.integers(len(cand))) for _ in range(n_starts)]
    finals: List[float] = []        # each start's final local-optimum objective
    final_sets: set = set()
    for s0 in seeds:
        S = greedy(cand[s0 % len(cand)])
        if S is None:
            continue
        S = descend(S)
        best_local = S
        for _ in range(n_kicks):                      # basin hopping
            S2 = descend(kick(best_local))
            if loss_of(S2) < loss_of(best_local) - 1e-12:
                best_local = S2
        finals.append(loss_of(best_local))
        final_sets.add(tuple(best_local))
    if not pool:
        return [], 0, None
    inc = min(finals) if finals else best[0]
    consensus = {
        "n_starts": len(finals),
        "n_kicks": n_kicks,
        "distinct_local_optima": len(final_sets),
        "starts_reaching_incumbent": int(sum(abs(f - inc) <= 1e-9 * (1 + abs(inc))
                                             for f in finals)),
        "incumbent_trail": [(int(w), float(v)) for w, v in trail],
    }
    consensus["consensus_rate"] = (consensus["starts_reaching_incumbent"]
                                   / max(consensus["n_starts"], 1))
    # Enforce the min-coverage quota on the final pool (a full-tuple property the
    # incremental ``_ok`` cannot check); the heuristic is best-effort here, exact
    # enumeration is the gold standard for coverage constraints.
    if strata is not None and min_per is not None:
        pool = {S: v for S, v in pool.items()
                if _strata.satisfies(strata, S, min_per, max_per, required)}
        # Defensive: a cross-stratum (covering) tuple is essentially always at
        # least as balanced as a single-stratum one, so the heuristic does not
        # leave the pool empty here in practice; enumeration is exact anyway.
        if not pool:  # pragma: no cover
            return [], work[0], consensus
    ranked = sorted(pool.items(), key=lambda kv: kv[1])[:top_K]
    return [(np.array(S), L) for S, L in ranked], work[0], consensus


# ======================================================================
# Public API
# ======================================================================

def select_treated_designs(
    G: np.ndarray,
    candidate_idx: Sequence[int],
    m: int,
    top_K: int = 20,
    *,
    unit_costs: Optional[np.ndarray] = None,
    budget: Optional[float] = None,
    method: str = "auto",            # "auto" | "enumerate" | "heuristic"
    enumerate_max: int = 3_000_000,  # max C(M,m) to enumerate exactly
    n_starts: int = 16,
    iters: int = 80,
    random_state: int = 0,
    unit_index: Optional[Any] = None,
    conflict: Optional[np.ndarray] = None,
    strata: Optional[np.ndarray] = None,
    min_per_stratum: Optional[int] = None,
    max_per_stratum: Optional[int] = None,
    size_band: Optional[Any] = None,
    targeting_penalty: float = 0.0,
) -> Dict[str, Any]:
    """Select the ``top_K`` treated m-tuples with smallest imbalance.

    ``targeting_penalty`` (gamma >= 0) makes the design *weakly targeted*: it
    adds ``gamma || w - (1/m) 1 ||^2`` to the treated QP, pulling the weights
    toward the group's own equal-weight aggregate so the treated group looks more
    like itself than like the population mean (trading some population
    representativeness -- the ATE -- for a deployable equal-weight group closer
    to its own ATT). On the simplex this is exactly a diagonal ridge,
    ``min_w w'(G_SS + gamma I) w``; the reported ``imbalance`` remains the true
    targeting distance ``sqrt(w' G w)`` at the penalized ``w``. ``gamma=0`` (the
    default) is the fully targeted design, unchanged.

    Parameters
    ----------
    G : (n, n) PSD Gram matrix of f-centred standardised predictors.
    candidate_idx : indices eligible for treatment.
    m : number of treated units per design.
    top_K : how many designs to return.
    unit_costs, budget : optional per-unit costs and total budget.
    method : "enumerate" forces exact; "heuristic" forces local search;
        "auto" enumerates when C(M, m) <= ``enumerate_max`` else searches.

    Returns
    -------
    dict
        Dictionary with::

            "top_designs" : list[TreatedDesign] sorted by imbalance.
            "stats"       : diagnostics, incl. whether the result is the exact
                            global top-K and the achieved-imbalance range.
    """
    t0 = time.time()
    G = np.asarray(G, dtype=float)
    if targeting_penalty < 0:
        raise MlsynthConfigError(
            f"targeting_penalty (gamma) must be >= 0; got {targeting_penalty}."
        )
    # Weakly-targeted ridge: search and weight-solve on G + gamma I; the true
    # targeting imbalance is read off the original G at the penalized weights.
    Gsearch = G + targeting_penalty * np.eye(G.shape[0]) if targeting_penalty > 0 else G
    rng = np.random.default_rng(random_state)

    # Up-front feasibility audit: one itemised error naming EVERY binding
    # constraint (candidate pool / budget / coverage / quota / spillover), each
    # with its have-vs-need and the minimal fix. Runs on the full eligible pool
    # so the budget line can report the actual cheapest-m bill.
    audit_feasibility(
        candidate_idx, m, unit_costs=unit_costs, budget=budget, conflict=conflict,
        strata=strata, min_per_stratum=min_per_stratum,
        max_per_stratum=max_per_stratum, size_band=size_band,
    )

    cand = _budget_feasible_candidates(candidate_idx, m, unit_costs, budget)
    M = len(cand)
    if M < m:  # pragma: no cover - the audit already raised on budget infeasibility
        raise MlsynthConfigError(f"Only {M} budget-feasible candidates for m={m}.")

    required = (_strata.required_codes(strata, cand) if strata is not None else None)

    total = comb(M, m)
    if method == "auto":
        method = "enumerate" if total <= enumerate_max else "heuristic"

    consensus = None
    if method == "enumerate":
        raw, n_eval = _enumerate(Gsearch, cand, m, top_K, unit_costs, budget, iters,
                                 conflict=conflict, strata=strata,
                                 min_per=min_per_stratum, max_per=max_per_stratum,
                                 required=required)
        exact = True
    elif method == "heuristic":
        raw, n_eval, consensus = _local_search(Gsearch, cand, m, top_K, unit_costs,
                                               budget, n_starts, rng, iters,
                                               conflict=conflict, strata=strata,
                                               min_per=min_per_stratum,
                                               max_per=max_per_stratum,
                                               required=required)
        exact = False
    else:
        raise ValueError(f"unknown method {method!r}")

    # Spillover feasibility: with the "No interference" constraint active, an
    # admissible design must be a size-m independent set of the conflict graph.
    # If the search found none, that constraint (alone or with the budget) is
    # infeasible -- fail loudly, in the same ``have vs need`` shape as the audit,
    # reporting the largest conflict-free set actually found.
    if conflict is not None and not raw:
        from .conflict import greedy_independent_set_size
        largest = greedy_independent_set_size(conflict, cand)
        raise MlsynthConfigError(
            "LEXSCM design is infeasible -- the binding constraint(s):\n  - "
            f"spillover: no conflict-free treated {m}-tuple exists among the {M} "
            f"candidates (largest conflict-free set found is {largest} < m={m}). "
            f"Relax the cluster/adjacency constraint, widen the candidate pool, or "
            f"reduce m."
        )

    # Coverage feasibility backstop: ``_strata.check_feasible`` (and the budget
    # presolve) already reject infeasible quotas up front, so this post-search
    # guard is essentially unreachable -- kept as defence in depth.
    if strata is not None and not raw:  # pragma: no cover
        raise MlsynthConfigError(
            f"No treated {m}-tuple satisfies the per-stratum quotas "
            f"(min_per_stratum={min_per_stratum}, max_per_stratum={max_per_stratum}) "
            f"together with the other constraints. Relax the quotas, increase m, or "
            f"widen the candidate pool."
        )

    # finalise weights to high precision for the returned designs
    designs: List[TreatedDesign] = []
    for rank, (S, _approx) in enumerate(raw, start=1):
        S = [int(x) for x in S]
        # Penalized weights/loss on G + gamma I; true targeting imbalance on G.
        loss, w, _ = _afw_single(Gsearch[np.ix_(S, S)], iters=600, tol=1e-14)
        imb2 = float(w @ G[np.ix_(S, S)] @ w)
        tc = float(unit_costs[S].sum()) if unit_costs is not None else 0.0
        d = TreatedDesign(
            indices=S, weights=w, loss=loss, imbalance=float(np.sqrt(max(imb2, 0.0))),
            total_cost=tc, label=f"Design {rank}",
        )
        if unit_index is not None:
            d.labels = [unit_index.labels[i] for i in S]
            d.weight_dict = {unit_index.labels[i]: float(wi) for i, wi in zip(S, w)}
            d.full_weights = np.zeros(len(unit_index.labels)); d.full_weights[S] = w
        designs.append(d)
    designs.sort(key=lambda x: x.loss)

    # advisory relaxation lower bound (informational only -- typically ~0 for
    # f-centred designs because the target lies in the donor hull, so we do
    # NOT form a MIP-style gap from it; consensus is the confidence signal).
    _, _, relax_lb = _afw_single(G[np.ix_(cand, cand)], iters=400, tol=1e-13)
    relax_lb = max(relax_lb, 0.0)

    if not designs:
        status = "INFEASIBLE"
    elif exact:
        status = "OPTIMAL"               # certified global top-K
    else:
        status = "FEASIBLE"              # near-optimal, no certificate

    # Solution pool: exactly the top_K tuples of size m (mirrors `top_K`).
    pool = [{
        "rank": d.label,
        "indices": d.indices,
        "imbalance": d.imbalance,
        "objective": d.loss,
        "budget_used": d.total_cost,
        "feasible": (budget is None or np.isinf(budget) or d.total_cost <= budget + 1e-9),
    } for d in designs]

    stats = {
        "problem": {
            "objective_sense": "minimize",
            "objective": "imbalance_sq",
            "cardinality_m": m,
            "top_K": top_K,
            "n_candidates": M,
            "feasible_region_C(M,m)": total,
            "budget": (None if budget is None or np.isinf(budget) else float(budget)),
        },
        "presolve": {
            "candidates_removed": int(len(candidate_idx) - M),
            "reason": "budget_infeasible" if M < len(candidate_idx) else None,
        },
        "search": {
            "method": "enumeration" if exact else "multistart_local_search",
            "subsets_evaluated": int(n_eval),
            "consensus": consensus,          # None for enumeration
        },
        "incumbent": {
            "objective": designs[0].loss if designs else None,
            "imbalance": designs[0].imbalance if designs else None,
            "worst_in_pool_imbalance": designs[-1].imbalance if designs else None,
        },
        "relaxation": {
            "lower_bound_imbalance": float(np.sqrt(relax_lb)),
            "note": "convex (l0-free) hull bound; ~0 when target is inside the "
                    "donor hull -- informational, not an optimality gap.",
        },
        "solution_pool": pool,             # K tuples of size m
        "termination": {
            "status": status,
            "runtime_sec": round(time.time() - t0, 4),
        },
    }
    return {"top_designs": designs, "stats": stats}
