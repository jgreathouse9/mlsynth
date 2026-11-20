"""Subset-row cuts (Jepsen, Petersen, Spoorendonk, Pisinger 2008) for the supergeo
master, with column generation that prices them.

SR3 cut for rows S = {i, j, k}: sum over columns c with |c & S| >= 2 of x_c <= 1.
It is the rank-1 Chvatal-Gomory cut with multipliers 1/2 on the three rows.
For a column c its coefficient is 1 iff c holds two or three of i, j, k, which is
  sum over pairs {a, b} of c of [a, b both in S]  -  2 [S subset of c],
so with Q[a, b] = sum of the duals of cuts holding a and b, and R[i, j, k] the dual
of the cut on {i, j, k}, the cut part of a column's price is
  sum over pairs of c of Q  -  2 * sum over triples of c of R.
Lagrangian bound with cuts, valid for any pi and any sigma <= 0:
  every design's cost >= sum(pi) + sum(sigma) + (N/lo) * min(0, min reduced cost)."""
import time
import numpy as np
from numba import njit
from scipy.optimize import linprog
from scipy.sparse import csc_matrix
from . import sg


@njit(cache=True)
def _cutterm(idx, k, Q, R):
    t = 0.0
    for a in range(k):
        for b in range(a + 1, k):
            t += Q[idx[a], idx[b]]
    if k >= 3:
        for a in range(k):
            for b in range(a + 1, k):
                for c in range(b + 1, k):
                    t -= 2.0 * R[idx[a], idx[b], idx[c]]
    return t


@njit(cache=True)
def _scan_cuts(z, pi, Q, R, lo, hi, thresh, cap):
    """Every subset of size lo..hi with reduced cost <= thresh (up to cap), and the min."""
    n = z.shape[0]
    out = np.full((cap, 4), -1, np.int64); sc = np.empty(cap); rc = np.empty(cap)
    idx = np.empty(4, np.int64); m = 0; best = np.inf
    for i in range(n):
        idx[0] = i
        for j in range(i + 1, n):
            idx[1] = j
            if lo <= 2:
                s = sg._score(z, idx, 2); r = s - pi[i] - pi[j] - _cutterm(idx, 2, Q, R)
                if r < best: best = r
                if r <= thresh and m < cap:
                    out[m, 0] = i; out[m, 1] = j; sc[m] = s; rc[m] = r; m += 1
            if hi < 3: continue
            for k in range(j + 1, n):
                idx[2] = k
                if lo <= 3:
                    s = sg._score(z, idx, 3); r = s - pi[i] - pi[j] - pi[k] - _cutterm(idx, 3, Q, R)
                    if r < best: best = r
                    if r <= thresh and m < cap:
                        out[m, 0] = i; out[m, 1] = j; out[m, 2] = k; sc[m] = s; rc[m] = r; m += 1
                if hi < 4: continue
                for l in range(k + 1, n):
                    idx[3] = l
                    s = sg._score(z, idx, 4)
                    r = s - pi[i] - pi[j] - pi[k] - pi[l] - _cutterm(idx, 4, Q, R)
                    if r < best: best = r
                    if r <= thresh and m < cap:
                        out[m, 0] = i; out[m, 1] = j; out[m, 2] = k; out[m, 3] = l
                        sc[m] = s; rc[m] = r; m += 1
    return out[:m], sc[:m], rc[:m], m, best


@njit(cache=True)
def _separate(n, colrows, collen, x, P, T):
    """Violation of every SR3 triple at the LP point: P_ij + P_ik + P_jk - 2 T_ijk."""
    P[:, :] = 0.0; T[:, :, :] = 0.0
    for c in range(colrows.shape[0]):
        k = collen[c]; w = x[c]
        for a in range(k):
            for b in range(a + 1, k):
                i, j = colrows[c, a], colrows[c, b]
                P[i, j] += w; P[j, i] += w
                for d in range(b + 1, k):
                    T[i, j, colrows[c, d]] += w
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v = P[i, j] + P[i, k] + P[j, k] - 2.0 * T[i, j, k]
                if v > 1.0 + 1e-6: cnt += 1
    out = np.empty((cnt, 4))
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v = P[i, j] + P[i, k] + P[j, k] - 2.0 * T[i, j, k]
                if v > 1.0 + 1e-6:
                    out[cnt, 0] = i; out[cnt, 1] = j; out[cnt, 2] = k; out[cnt, 3] = v; cnt += 1
    return out


def coef(c, S):
    return 1.0 if len(set(c) & set(S)) >= 2 else 0.0


def master(cols, costs, n, cuts):
    rows, cs = [], []
    for j, c in enumerate(cols):
        for g in c: rows.append(g); cs.append(j)
    A = csc_matrix((np.ones(len(rows)), (rows, cs)), shape=(n, len(cols)))
    if cuts:
        bypair = {}
        for t, S in enumerate(cuts):
            i, j, k = sorted(S)
            for p in ((i, j), (i, k), (j, k)):
                bypair.setdefault(p, []).append(t)
        r2, c2 = [], []
        for jj, c in enumerate(cols):
            hit = set()
            for a in range(len(c)):
                for b in range(a + 1, len(c)):
                    hit.update(bypair.get((c[a], c[b]) if c[a] < c[b] else (c[b], c[a]), ()))
            for t in hit: r2.append(t); c2.append(jj)
        Au = csc_matrix((np.ones(len(r2)), (r2, c2)), shape=(len(cuts), len(cols)))
        r = linprog(costs, A_ub=Au, b_ub=np.ones(len(cuts)), A_eq=A, b_eq=np.ones(n), bounds=(0, None), method="highs")
        assert r.status == 0, r.message
        return r.fun, r.eqlin.marginals, r.ineqlin.marginals, r.x
    r = linprog(costs, A_eq=A, b_eq=np.ones(n), bounds=(0, None), method="highs")
    assert r.status == 0, r.message
    return r.fun, r.eqlin.marginals, np.zeros(0), r.x


def duals_to_QR(n, cuts, sigma):
    Q = np.zeros((n, n)); R = np.zeros((n, n, n))
    for S, s in zip(cuts, sigma):
        i, j, k = sorted(S)
        for a, b in ((i, j), (i, k), (j, k)):
            Q[a, b] += s; Q[b, a] += s
        R[i, j, k] += s
    return Q, R


def run(z0, lo=2, hi=4, add=2000, cut_rounds=30, cuts_per_round=100, verbose=True, max_cuts=3000):
    t0 = time.perf_counter()
    n = len(z0); z = np.asarray(z0, float).copy(); scale = 1.0
    cols = sg.initial_pool(z, lo, hi)
    def sc(c): return float(sg._score(z, np.array(tuple(c) + (0,) * (4 - len(c)), np.int64), len(c)))
    score = {c: sc(c) for c in cols}
    cuts = []; hist = []
    P = np.zeros((n, n)); T = np.zeros((n, n, n))
    rnd = 0
    while True:
        # column generation to convergence at the current cuts
        while True:
            lp, pi, sigma, x = master(cols, np.array([score[c] for c in cols]), n, cuts)
            if 0 < lp < 1e-2 * n:
                f = n / lp; z *= np.sqrt(f); scale *= f
                score = {c: v * f for c, v in score.items()}; hist = [(a, b * f, c_) for a, b, c_ in hist]
                continue
            Q, R = duals_to_QR(n, cuts, sigma)
            rows, s_, rc, m, mrc = _scan_cuts(z, pi, Q, R, lo, hi, -1e-9 * max(1.0, lp), 5_000_000)
            new = 0
            for t in np.argsort(rc)[:add]:
                c = tuple(int(v) for v in rows[t] if v >= 0)
                if c not in score:
                    score[c] = float(s_[t]); cols.append(c); new += 1
            if new == 0:
                break
        lb = float(np.sum(pi) + np.sum(sigma) + (n / lo) * min(0.0, mrc))
        hist.append((len(cuts), lb, lp))
        if verbose:
            print(f"  round {rnd}: cuts {len(cuts)}, LP {lp:.6g}, bound {lb:.6g} (orig {lb/scale:.4g}), pool {len(cols)} [{time.perf_counter()-t0:.0f}s]", flush=True)
        if rnd >= cut_rounds or len(cuts) >= max_cuts:
            break
        sup = np.flatnonzero(x > 1e-9)
        cr = np.full((len(sup), 4), 0, np.int64); cl = np.zeros(len(sup), np.int64)
        for a, j in enumerate(sup):
            c = cols[j]; cl[a] = len(c); cr[a, :len(c)] = c
        viol = _separate(n, cr, cl, x[sup], P, T)
        if len(viol) == 0:
            if verbose: print("  no violated SR3 cut", flush=True)
            break
        have = set(cuts)
        order = np.argsort(-viol[:, 3])
        newc = [tuple(int(v) for v in viol[t, :3]) for t in order if tuple(int(v) for v in viol[t, :3]) not in have][:cuts_per_round]
        if not newc:
            break
        cuts.extend(newc); rnd += 1
    return dict(scale=scale, lb=hist[-1][1], hist=hist, cuts=cuts, cols=cols, score=score, z=z, pi=pi, sigma=sigma, t=time.perf_counter() - t0)
