"""Rank-1 Chvatal-Gomory cuts on 3 to 5 rows for the supergeo master.

For rows S and multipliers u >= 0, every design satisfies
    sum over columns c of floor(u . a_c) x_c <= floor(u . 1),
a_c the column's 0-1 incidence on S (Chvatal-Gomory rounding of the partition
rows; Petersen, Pisinger and Spoorendonk 2008; Pecin, Pessoa, Poggi and Uchoa
2017). Multipliers are integer numerators over a common denominator, so every
floor is computed in integers. (3, (1,1,1), 2) is the subset-row cut of :mod:`.sr`.

Bound with cut duals sigma <= 0 (one per cut, right-hand side rhs) and the
parity dual tau >= 0 for odd N:
    sum pi + tau + sum sigma rhs + (N / lo) min(0, min reduced cost).

Pricing enumerates every group of 2 to 4 (fine to N = 40 or so); separation
enumerates every row set of the family's size among the rows that fractional
columns touch, under every distinct arrangement of its multipliers."""
import itertools, math
import numpy as np
from numba import njit
import highspy
from . import sg, dive

FAMILIES = [(3, (1, 1, 1), 2), (4, (1, 1, 1, 1), 3), (4, (2, 1, 1, 1), 3),
            (5, (1, 1, 1, 1, 1), 3), (5, (1, 1, 1, 1, 1), 2)]


class Cut:
    __slots__ = ("S", "num", "den", "rhs")

    def __init__(self, S, num, den):
        self.S = tuple(int(v) for v in S); self.num = tuple(int(v) for v in num); self.den = int(den)
        self.rhs = sum(self.num) // self.den

    def coef(self, c):
        cs = set(c)
        return sum(w for s, w in zip(self.S, self.num) if s in cs) // self.den

    def key(self):
        return (tuple(sorted(zip(self.S, self.num))), self.den)


def _arrays(cuts):
    m = len(cuts)
    cs = np.full((max(m, 1), 5), -1, np.int64); cn = np.zeros((max(m, 1), 5), np.int64); cd = np.ones(max(m, 1), np.int64)
    for t, c in enumerate(cuts):
        cs[t, :len(c.S)] = c.S; cn[t, :len(c.S)] = c.num; cd[t] = c.den
    return cs, cn, cd


@njit(cache=True)
def _cutterm(idx, k, cs, cn, cd, sigma, m):
    term = 0.0
    for t in range(m):
        s = 0
        for q in range(5):
            r = cs[t, q]
            if r < 0: break
            for a in range(k):
                if idx[a] == r:
                    s += cn[t, q]; break
        cf = s // cd[t]
        if cf: term += sigma[t] * cf
    return term


@njit(cache=True)
def _scan(z, pi, tau, cs, cn, cd, sigma, m, lo, hi, thresh, cap):
    n = z.shape[0]
    out = np.full((cap, 4), -1, np.int64); sc = np.empty(cap); rc = np.empty(cap)
    idx = np.empty(4, np.int64); cnt = 0; best = np.inf
    for i in range(n):
        idx[0] = i
        for j in range(i + 1, n):
            idx[1] = j
            if lo <= 2:
                s = sg._score(z, idx, 2); r = s - pi[i] - pi[j] - _cutterm(idx, 2, cs, cn, cd, sigma, m)
                if r < best: best = r
                if r <= thresh and cnt < cap:
                    out[cnt, 0] = i; out[cnt, 1] = j; sc[cnt] = s; rc[cnt] = r; cnt += 1
            if hi < 3: continue
            for c in range(j + 1, n):
                idx[2] = c
                if lo <= 3:
                    s = sg._score(z, idx, 3); r = s - pi[i] - pi[j] - pi[c] - tau - _cutterm(idx, 3, cs, cn, cd, sigma, m)
                    if r < best: best = r
                    if r <= thresh and cnt < cap:
                        out[cnt, 0] = i; out[cnt, 1] = j; out[cnt, 2] = c; sc[cnt] = s; rc[cnt] = r; cnt += 1
                if hi < 4: continue
                for d in range(c + 1, n):
                    idx[3] = d
                    s = sg._score(z, idx, 4); r = s - pi[i] - pi[j] - pi[c] - pi[d] - _cutterm(idx, 4, cs, cn, cd, sigma, m)
                    if r < best: best = r
                    if r <= thresh and cnt < cap:
                        out[cnt, 0] = i; out[cnt, 1] = j; out[cnt, 2] = c; out[cnt, 3] = d; sc[cnt] = s; rc[cnt] = r; cnt += 1
    return out[:cnt], sc[:cnt], rc[:cnt], cnt, best


def price(z, pi, tau, cuts, sigma, lo, hi, thresh, cap=5_000_000):
    cs, cn, cd = _arrays(cuts)
    sig = np.zeros(max(len(cuts), 1)); sig[:len(cuts)] = sigma
    return _scan(np.asarray(z, float), np.asarray(pi, float), float(tau), cs, cn, cd, sig, len(cuts), lo, hi, float(thresh), cap)


@njit(cache=True)
def _separate(masks, xs, R, size, perms, den, rhs, tol, cap):
    """Every row set of `size` from R, under every multiplier arrangement in perms, with violation > tol."""
    nR = R.shape[0]; P = perms.shape[0]
    outS = np.full((cap, 5), -1, np.int64); outP = np.zeros(cap, np.int64); outV = np.zeros(cap); k = 0
    sel = np.zeros(5, np.int64)
    for a in range(nR):
        for b in range(a + 1, nR):
            for c in range(b + 1, nR):
                for d in range(c + 1, nR if size >= 4 else c + 2):
                    for e in range(d + 1, nR if size >= 5 else d + 2):
                        sel[0] = R[a]; sel[1] = R[b]; sel[2] = R[c]
                        if size >= 4: sel[3] = R[d]
                        if size >= 5: sel[4] = R[e]
                        for p in range(P):
                            lhs = 0.0
                            for col in range(masks.shape[0]):
                                s = 0
                                for q in range(size):
                                    if (masks[col] >> sel[q]) & 1:
                                        s += perms[p, q]
                                cf = s // den
                                if cf: lhs += xs[col] * cf
                            if lhs - rhs > tol and k < cap:
                                outS[k, :size] = sel[:size]; outP[k] = p; outV[k] = lhs - rhs; k += 1
    return outS[:k], outP[:k], outV[:k]


def separate(cols, x, n, families, tol=1e-6, cap=200_000):
    sup = [(c, v) for c, v in zip(cols, x) if v > 1e-9]
    frac_rows = sorted({i for c, v in sup if v < 1 - 1e-9 for i in c})
    if len(frac_rows) < 3:
        return []
    masks = np.array([sum(1 << g for g in c) for c, _ in sup], np.int64); xs = np.array([v for _, v in sup])
    R = np.array(frac_rows, np.int64); found = []
    for size, num, den in families:
        if len(R) < size: continue
        perms = np.array(sorted(set(itertools.permutations(num))), np.int64)
        S, P, V = _separate(masks, xs, R, size, perms, den, sum(num) // den, tol, cap)
        for s, p, v in zip(S, P, V):
            found.append((float(v), Cut(tuple(int(t) for t in s[:size]), tuple(int(w) for w in perms[p]), den)))
    found.sort(key=lambda t: -t[0])
    return found


def master(cols, costs, n, odd, cuts):
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
    C = len(cols); nr = n + (1 if odd else 0) + len(cuts); p0 = n + (1 if odd else 0)
    lp = highspy.HighsLp(); lp.num_col_, lp.num_row_ = C, nr
    lp.col_cost_ = np.asarray(costs, float); lp.col_lower_ = np.zeros(C); lp.col_upper_ = np.full(C, np.inf)
    rl = np.ones(nr); ru = np.ones(nr)
    if odd: ru[n] = np.inf
    rl[p0:] = -np.inf; ru[p0:] = [c.rhs for c in cuts]
    lp.row_lower_ = rl; lp.row_upper_ = ru
    start = np.zeros(C + 1, np.int32); idx = []; val = []
    for j, c in enumerate(cols):
        ent = [(g, 1.0) for g in c] + ([(n, 1.0)] if odd and len(c) % 2 else [])
        ent += [(p0 + t, float(cut.coef(c))) for t, cut in enumerate(cuts) if cut.coef(c)]
        for r, v in ent: idx.append(r); val.append(v)
        start[j + 1] = start[j] + len(ent)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise; lp.a_matrix_.start_ = start
    lp.a_matrix_.index_ = np.array(idx, np.int32); lp.a_matrix_.value_ = np.array(val)
    h.passModel(lp); h.run()
    st = h.modelStatusToString(h.getModelStatus())
    if st != "Optimal":
        raise RuntimeError(f"master not solved: {st}")
    sol = h.getSolution(); y = np.array(sol.row_dual)
    tau = max(float(y[n]), 0.0) if odd else 0.0
    return h.getInfo().objective_function_value, y[:n], tau, np.minimum(y[p0:], 0.0), np.array(sol.col_value)


def root_bound(z0, families=FAMILIES, rounds=30, per_round=50, max_cuts=3000, lo=2, hi=4, add=2000):
    """Column generation with rank-1 separation at the root. hist: (cuts, bound in original units) per round."""
    n = len(z0); odd = n % 2 == 1; z = np.asarray(z0, float).copy(); scale = 1.0
    cols = list(sg.initial_pool(z, lo, hi)); score = {c: dive._sc(z, c) for c in cols}
    cuts, keys, hist = [], set(), []
    for rnd in range(rounds + 1):
        while True:
            lp, pi, tau, sigma, x = master(cols, [score[c] for c in cols], n, odd, cuts)
            if 0 < lp < 1e-2 * n:
                k = int(round(math.log(n / lp, 4))); z = z * 2.0 ** k; scale *= 4.0 ** k
                score = {c: dive._sc(z, c) for c in score}; continue
            rows, s_, rc, m, best = price(z, pi, tau, cuts, sigma, lo, hi, -1e-9 * max(1.0, abs(lp)))
            new = 0
            for t in np.argsort(rc)[:add]:
                c = tuple(int(v) for v in rows[t] if v >= 0)
                if c not in score: score[c] = float(s_[t]); cols.append(c); new += 1
            if new == 0: break
        bound = float(np.sum(pi)) + tau + float(sum(s * c.rhs for s, c in zip(sigma, cuts))) + (n / lo) * min(0.0, best)
        hist.append((len(cuts), bound / scale))
        if rnd == rounds or len(cuts) >= max_cuts: break
        viol = [c for v, c in separate(cols, x, n, families) if c.key() not in keys][:per_round]
        if not viol: break
        for c in viol: keys.add(c.key()); cuts.append(c)
    return dict(hist=hist, cuts=cuts, scale=scale, lp=lp / scale)
