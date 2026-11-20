"""Supergeo base problem (Chen et al. 2023, sec. 3): certified by column generation
and reduced-cost enumeration. Scratch prototype."""
import time
from itertools import combinations

import numpy as np
from numba import njit
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix


@njit(cache=True)
def _score(z, idx, k):
    """min over non-empty proper subsets A of (2 Z_A - S)^2."""
    S = 0.0
    for t in range(k):
        S += z[idx[t]]
    best = np.inf
    for mask in range(1, (1 << k) - 1):
        a = 0.0
        for t in range(k):
            if mask >> t & 1:
                a += z[idx[t]]
        d = 2.0 * a - S
        if d * d < best:
            best = d * d
    return best


@njit(cache=True)
def _scan(z, pi, lo, hi, thresh, cap):
    """Every subset of size lo..hi with reduced cost score - pi(G) <= thresh, up to cap,
    as (rows of indices padded with -1, scores, reduced costs)."""
    n = z.shape[0]
    out = np.full((cap, 4), -1, np.int32)
    sc = np.empty(cap); rc = np.empty(cap)
    m = 0
    idx = np.empty(4, np.int64)
    for i in range(n):
        idx[0] = i
        for j in range(i + 1, n):
            idx[1] = j
            if lo <= 2:
                s = _score(z, idx, 2); r = s - pi[i] - pi[j]
                if r <= thresh and m < cap:
                    out[m, 0] = i; out[m, 1] = j; sc[m] = s; rc[m] = r; m += 1
            if hi < 3:
                continue
            for k in range(j + 1, n):
                idx[2] = k
                if lo <= 3:
                    s = _score(z, idx, 3); r = s - pi[i] - pi[j] - pi[k]
                    if r <= thresh and m < cap:
                        out[m, 0] = i; out[m, 1] = j; out[m, 2] = k; sc[m] = s; rc[m] = r; m += 1
                if hi < 4:
                    continue
                for l in range(k + 1, n):
                    idx[3] = l
                    s = _score(z, idx, 4); r = s - pi[i] - pi[j] - pi[k] - pi[l]
                    if r <= thresh and m < cap:
                        out[m, 0] = i; out[m, 1] = j; out[m, 2] = k; out[m, 3] = l
                        sc[m] = s; rc[m] = r; m += 1
    return out[:m], sc[:m], rc[:m], m


def _matrix(cols, n):
    rows, cs = [], []
    for c, members in enumerate(cols):
        for g in members:
            rows.append(g); cs.append(c)
    return csc_matrix((np.ones(len(rows)), (rows, cs)), shape=(n, len(cols)))


def master_lp(cols, costs, n):
    A = _matrix(cols, n)
    r = linprog(costs, A_eq=A, b_eq=np.ones(n), bounds=(0, None), method="highs")
    assert r.status == 0, r.message
    return r.fun, r.eqlin.marginals, r.x


def master_mip(cols, costs, n, time_limit=600):
    A = _matrix(cols, n)
    r = milp(costs, constraints=LinearConstraint(A, 1, 1), integrality=np.ones(len(cols)),
             bounds=Bounds(0, 1), options={"time_limit": time_limit, "mip_rel_gap": 0})
    return r


def initial_pool(z, lo, hi):
    """Pairs of neighbours in sorted order, and their triples, so a cover exists."""
    o = np.argsort(z)
    cols = []
    for w in range(lo, hi + 1):
        for s in range(len(z) - w + 1):
            cols.append(tuple(sorted(o[s:s + w].tolist())))
    return sorted(set(cols))


def certify(z, lo=2, hi=4, add=2000, verbose=True, mip_limit=600):
    t0 = time.perf_counter()
    n = len(z)
    cols = initial_pool(z, lo, hi)
    score = {}
    for c in cols:
        idx = np.array(c + (0,) * (4 - len(c)), np.int64)
        score[c] = float(_score(z, idx, len(c)))
    rounds = 0
    while True:
        rounds += 1
        lb, pi, x = master_lp(cols, np.array([score[c] for c in cols]), n)
        rows, sc, rc, m = _scan(z, pi, lo, hi, -1e-12 * max(1.0, abs(lb)), 50_000_000)
        if m == 0:
            break
        order = np.argsort(rc)[:add]
        new = 0
        for t in order:
            c = tuple(int(v) for v in rows[t] if v >= 0)
            if c not in score:
                score[c] = float(sc[t]); cols.append(c); new += 1
        if verbose:
            print(f"  round {rounds}: LP {lb:.6g}, {m} columns priced out, {new} added, pool {len(cols)}")
        if new == 0:
            break
    t_lp = time.perf_counter() - t0
    r = master_mip(cols, np.array([score[c] for c in cols]), n, mip_limit)
    ub = r.fun
    t_ub = time.perf_counter() - t0
    gap = ub - lb
    rows, sc, rc, m = _scan(z, pi, lo, hi, gap * (1 + 1e-9) + 1e-12, 200_000_000)
    cand = [tuple(int(v) for v in rows[t] if v >= 0) for t in range(m)]
    r2 = master_mip(cand, sc, n, mip_limit)
    t_all = time.perf_counter() - t0
    return dict(n=n, lb=lb, ub_pool=ub, opt=r2.fun, status=r2.status, rounds=rounds,
                pool=len(cols), within_gap=m, t_lp=t_lp, t_ub=t_ub, t_all=t_all,
                design=[cand[i] for i in np.flatnonzero(np.round(r2.x) > 0.5)])


def full_mip(z, lo=2, hi=4, mip_limit=3600):
    rows, sc, rc, m = _scan(z, np.zeros(len(z)), lo, hi, np.inf, 200_000_000)
    cols = [tuple(int(v) for v in rows[t] if v >= 0) for t in range(m)]
    t0 = time.perf_counter()
    r = master_mip(cols, sc, len(z), mip_limit)
    return r.fun, m, time.perf_counter() - t0, r.status


def safe_lb(z, pi, lo, hi):
    """Lagrangian lower bound valid for any dual vector pi (Neumaier-Shcherbina
    form): every design is a partition into at most N/lo columns, so its cost is
    sum(pi) + sum of its columns' reduced costs >= sum(pi) + (N/lo) min(0, min rc).
    Returns (bound, min reduced cost). Rounding is not yet bounded here."""
    rows, sc, rc, m = _scan(z, pi, lo, hi, np.inf, 1)
    # _scan with cap 1 still visits every column but stores one; recompute the min:
    return None


@njit(cache=True)
def _min_rc(z, pi, lo, hi):
    n = z.shape[0]
    idx = np.empty(4, np.int64)
    best = np.inf
    for i in range(n):
        idx[0] = i
        for j in range(i + 1, n):
            idx[1] = j
            if lo <= 2:
                r = _score(z, idx, 2) - pi[i] - pi[j]
                if r < best: best = r
            if hi < 3: continue
            for k in range(j + 1, n):
                idx[2] = k
                if lo <= 3:
                    r = _score(z, idx, 3) - pi[i] - pi[j] - pi[k]
                    if r < best: best = r
                if hi < 4: continue
                for l in range(k + 1, n):
                    idx[3] = l
                    r = _score(z, idx, 4) - pi[i] - pi[j] - pi[k] - pi[l]
                    if r < best: best = r
    return best


def lagrangian_lb(z, pi, lo=2, hi=4):
    m = _min_rc(z, np.asarray(pi, float), lo, hi)
    return float(np.sum(pi) + (len(z) / lo) * min(0.0, m)), float(m)


@njit(cache=True)
def _count(z, pi, lo, hi, thresh):
    """How many subsets of size lo..hi have reduced cost <= thresh."""
    n = z.shape[0]
    idx = np.empty(4, np.int64)
    m = 0
    for i in range(n):
        idx[0] = i
        for j in range(i + 1, n):
            idx[1] = j
            if lo <= 2 and _score(z, idx, 2) - pi[i] - pi[j] <= thresh:
                m += 1
            if hi < 3: continue
            for k in range(j + 1, n):
                idx[2] = k
                if lo <= 3 and _score(z, idx, 3) - pi[i] - pi[j] - pi[k] <= thresh:
                    m += 1
                if hi < 4: continue
                for l in range(k + 1, n):
                    idx[3] = l
                    if _score(z, idx, 4) - pi[i] - pi[j] - pi[k] - pi[l] <= thresh:
                        m += 1
    return m


def certify2(z, lo=2, hi=4, add=2000, verbose=True, mip_limit=1200, max_cols=20_000_000):
    """Column generation on rescaled costs, the Lagrangian bound, a counted
    reduced-cost enumeration and the MIP over it."""
    t0 = time.perf_counter()
    n = len(z); z = np.asarray(z, float).copy(); scale = 1.0
    cols = initial_pool(z, lo, hi)
    def sc_of(c): return float(_score(z, np.array(c + (0,) * (4 - len(c)), np.int64), len(c)))
    score = {c: sc_of(c) for c in cols}
    rounds, best_lb, pi = 0, -np.inf, None
    while True:
        rounds += 1
        lp, pi, x = master_lp(cols, np.array([score[c] for c in cols]), n)
        if 0 < lp < 1e-2 * n:                         # loss too small for absolute tolerances
            f = n / lp
            z *= np.sqrt(f); scale *= f
            score = {c: v * f for c, v in score.items()}
            best_lb *= f
            if verbose: print(f"  rescaled costs by {f:.3g}")
            continue
        lb, mrc = lagrangian_lb(z, pi, lo, hi)
        best_lb = max(best_lb, lb)
        rows, s_, rc, m = _scan(z, pi, lo, hi, -1e-9 * max(1.0, lp), 5_000_000)
        new = 0
        for t in np.argsort(rc)[:add]:
            c = tuple(int(v) for v in rows[t] if v >= 0)
            if c not in score:
                score[c] = float(s_[t]); cols.append(c); new += 1
        if verbose:
            print(f"  round {rounds}: LP {lp:.6g} LagrLB {lb:.6g} minrc {mrc:+.2e} priced {m} added {new} pool {len(cols)} [{time.perf_counter()-t0:.0f}s]")
        if new == 0:
            break
    t_lp = time.perf_counter() - t0
    r = master_mip(cols, np.array([score[c] for c in cols]), n, mip_limit)
    ub = r.fun
    t_ub = time.perf_counter() - t0
    margin = (ub - best_lb) * (1 + 1e-9)
    # columns that can sit in a design below ub: rc <= ub - (sum(pi) + (n/lo - 1) min(0, minrc))
    lb_pi, mrc = lagrangian_lb(z, pi, lo, hi)
    thresh = ub - float(np.sum(pi)) - (n / lo - 1) * min(0.0, mrc)
    cnt = _count(z, pi, lo, hi, thresh)
    if verbose: print(f"  UB {ub:.6g} (pool MIP status {r.status}), LB {best_lb:.6g}, {cnt} columns within the margin [{t_ub:.0f}s]")
    if cnt > max_cols:
        return dict(n=n, lb=best_lb / scale, ub=ub / scale, within=cnt, status="too many", t=time.perf_counter() - t0)
    rows, s_, rc, m = _scan(z, pi, lo, hi, thresh, cnt + 1)
    cand = [tuple(int(v) for v in rows[t] if v >= 0) for t in range(m)]
    r2 = master_mip(cand, s_, n, mip_limit)
    return dict(n=n, lb=best_lb / scale, ub_pool=ub / scale, opt=r2.fun / scale, status=r2.status,
                rounds=rounds, pool=len(cols), within=m, t_lp=t_lp, t_ub=t_ub, t=time.perf_counter() - t0)
