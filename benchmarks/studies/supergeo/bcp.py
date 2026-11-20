"""Branch, cut and price for the supergeo base problem.

Combines the Ryan-Foster branch and price of :mod:`.bnp` with the subset-row cuts
of :mod:`.sr`, and adds group branching on the market with the fewest viable
groups.

Node. A matrix T of pair decisions: T[i, j] = 1 (i and j in the same group),
-1 (never in the same group), 0 (free). A column is compatible when it holds no
apart pair and, for each member, every together-partner of that member.

Cuts. SR3 cuts (Jepsen, Petersen, Spoorendonk and Pisinger 2008): for rows
S = {i, j, k}, the columns holding at least two of them sum to at most 1. Each cut
holds for every design, so one global pool serves every node. A column's cut
coefficient is sum over its pairs of Q - 2 * sum over its triples of R, as in
:mod:`.sr`.

Pricing. One kernel applies the partition duals pi, the parity dual tau (odd
sizes, odd N), the cut duals, and the node's pair decisions.

Bound at a node, valid for any duals with sigma <= 0 and tau >= 0: every design
satisfying the node's decisions costs at least
    sum pi + tau [N odd] + sum sigma + (N / lo) * min(0, min reduced cost),
the minimum taken over compatible columns.

Branching. On a fractional node, a market is fractional when more than one
column covering it is positive. Among fractional markets, take the one with the
fewest viable groups (compatible, loss below the incumbent). If that number is at
most GROUP_LIMIT, branch on the market's group: one child per viable group,
fixing it (together within the group, apart from every other market). A design
in the node puts the market in exactly one group, and a group whose loss reaches
the incumbent cannot be in a better design, so the children are complete.
Otherwise branch on the Ryan-Foster pair whose together-weight is closest to 1/2.

Reduced-cost fixing. With the root's duals, a design D costs at least
    S + sum over c in D of rc(c) >= S + rc(c) + (N / lo - 1) m    for each c in D,
S = sum pi + tau [N odd] + sum sigma rhs and m = min(0, min rc), because sigma <= 0
meets each cut's row at most at its right-hand side and an odd N has an odd group.
So a group in a design costing at most the incumbent U has rc <= U - bound + m,
bound = S + (N / lo) m the root bound. After the root, the search keeps those groups
(the universe) and nothing else: the master holds every one of them, so pricing
stops, and the bound of a node is its LP over the universe. Each better incumbent
shrinks the universe again with the same root duals. Above `universe_limit` groups
nothing is fixed and pricing continues.

Floating point throughout. Exactness is the job of :mod:`.certify_exact`, which
can verify the design this search returns."""
import heapq, math, time
import numpy as np
from numba import njit
import highspy
from . import sg, dive, sr, bnp, lns, rank1

BIG = 1e7
GROUP_LIMIT = 64

#: Tuned cut management (tune_cuts.py): every rank-1 family at every depth, a cut purged
#: after 2 LP solves with a zero dual, at most 250 cuts. On 20-market GeoLift subsets it
#: took 3.2 times fewer nodes than subset-row cuts alone (tuning set) and 2.8 times fewer
#: on held-out subsets, in 15% and 8% less time.
def managed():
    return dict(families=rank1.FAMILIES, purge_after=2, deep_depth=None, max_active=250)


@njit(cache=True)
def _scan_bcp(z, pi, tau, Q, R, T, lo, hi, thresh, cap):
    n = z.shape[0]
    out = np.full((cap, 4), -1, np.int64); sc = np.empty(cap); rc = np.empty(cap)
    idx = np.empty(4, np.int64); m = 0; best = np.inf
    for i in range(n):
        idx[0] = i
        for j in range(i + 1, n):
            idx[1] = j
            if lo <= 2 and bnp._compat(T, idx, 2, n):
                s = sg._score(z, idx, 2); r = s - pi[i] - pi[j] - sr._cutterm(idx, 2, Q, R)
                if r < best: best = r
                if r <= thresh and m < cap:
                    out[m, 0] = i; out[m, 1] = j; sc[m] = s; rc[m] = r; m += 1
            if hi < 3: continue
            for c in range(j + 1, n):
                idx[2] = c
                if lo <= 3 and bnp._compat(T, idx, 3, n):
                    s = sg._score(z, idx, 3); r = s - pi[i] - pi[j] - pi[c] - tau - sr._cutterm(idx, 3, Q, R)
                    if r < best: best = r
                    if r <= thresh and m < cap:
                        out[m, 0] = i; out[m, 1] = j; out[m, 2] = c; sc[m] = s; rc[m] = r; m += 1
                if hi < 4: continue
                for d in range(c + 1, n):
                    idx[3] = d
                    if bnp._compat(T, idx, 4, n):
                        s = sg._score(z, idx, 4); r = s - pi[i] - pi[j] - pi[c] - pi[d] - sr._cutterm(idx, 4, Q, R)
                        if r < best: best = r
                        if r <= thresh and m < cap:
                            out[m, 0] = i; out[m, 1] = j; out[m, 2] = c; out[m, 3] = d; sc[m] = s; rc[m] = r; m += 1
    return out[:m], sc[:m], rc[:m], m, best


@njit(cache=True)
def _scan_bcp_r1(z, pi, tau, cs, cn, cd, sigma, mc, T, lo, hi, thresh, cap):
    """As _scan_bcp, with rank-1 cuts (rows cs, numerators cn, denominators cd, duals sigma)."""
    n = z.shape[0]
    out = np.full((cap, 4), -1, np.int64); sc = np.empty(cap); rc = np.empty(cap)
    idx = np.empty(4, np.int64); m = 0; best = np.inf
    for i in range(n):
        idx[0] = i
        for j in range(i + 1, n):
            idx[1] = j
            if lo <= 2 and bnp._compat(T, idx, 2, n):
                s = sg._score(z, idx, 2); r = s - pi[i] - pi[j] - rank1._cutterm(idx, 2, cs, cn, cd, sigma, mc)
                if r < best: best = r
                if r <= thresh and m < cap:
                    out[m, 0] = i; out[m, 1] = j; sc[m] = s; rc[m] = r; m += 1
            if hi < 3: continue
            for c in range(j + 1, n):
                idx[2] = c
                if lo <= 3 and bnp._compat(T, idx, 3, n):
                    s = sg._score(z, idx, 3); r = s - pi[i] - pi[j] - pi[c] - tau - rank1._cutterm(idx, 3, cs, cn, cd, sigma, mc)
                    if r < best: best = r
                    if r <= thresh and m < cap:
                        out[m, 0] = i; out[m, 1] = j; out[m, 2] = c; sc[m] = s; rc[m] = r; m += 1
                if hi < 4: continue
                for d in range(c + 1, n):
                    idx[3] = d
                    if bnp._compat(T, idx, 4, n):
                        s = sg._score(z, idx, 4); r = s - pi[i] - pi[j] - pi[c] - pi[d] - rank1._cutterm(idx, 4, cs, cn, cd, sigma, mc)
                        if r < best: best = r
                        if r <= thresh and m < cap:
                            out[m, 0] = i; out[m, 1] = j; out[m, 2] = c; out[m, 3] = d; sc[m] = s; rc[m] = r; m += 1
    return out[:m], sc[:m], rc[:m], m, best


@njit(cache=True)
def _viable(z, T, i, lo, hi, ub, cap):
    """Groups holding market i, compatible with T, with loss below ub (at most cap of them)."""
    n = z.shape[0]
    out = np.full((cap, 4), -1, np.int64); los = np.empty(cap); m = 0
    others = np.empty(n - 1, np.int64); t = 0
    for j in range(n):
        if j != i:
            others[t] = j; t += 1
    idx = np.empty(4, np.int64); g = np.empty(4, np.int64)
    for a in range(n - 1):
        for b in range(a, n - 1):
            for c in range(b, n - 1):
                # size 2: (a); size 3: (a, b) with b > a; size 4: (a, b, c) with c > b > a
                k = 0
                if b == a and c == a:
                    k = 2; g[0] = i; g[1] = others[a]
                elif c == b and b > a:
                    k = 3; g[0] = i; g[1] = others[a]; g[2] = others[b]
                elif c > b and b > a:
                    k = 4; g[0] = i; g[1] = others[a]; g[2] = others[b]; g[3] = others[c]
                else:
                    continue
                if k < lo or k > hi:
                    continue
                for q in range(k): idx[q] = g[q]
                idx[:k] = np.sort(idx[:k])
                if not bnp._compat(T, idx, k, n):
                    continue
                s = sg._score(z, idx, k)
                if s >= ub:
                    continue
                if m >= cap:
                    return out, los, m + 1
                out[m, :k] = idx[:k]; los[m] = s; m += 1
    return out[:m], los[:m], m


def _master(cols, costs, n, odd, cuts):
    """Master LP: partition rows (= 1), parity row (>= 1, odd N), cut rows (<= 1), artificial singletons."""
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
    C = len(cols); m = C + n; nr = n + (1 if odd else 0) + len(cuts)
    lp = highspy.HighsLp(); lp.num_col_, lp.num_row_ = m, nr
    lp.col_cost_ = np.concatenate([np.asarray(costs, float), np.full(n, BIG)])
    lp.col_lower_ = np.zeros(m); lp.col_upper_ = np.full(m, np.inf)
    rl = np.ones(nr); ru = np.ones(nr)
    p0 = n + (1 if odd else 0)
    if odd: ru[n] = np.inf
    rl[p0:] = -np.inf
    lp.row_lower_ = rl; lp.row_upper_ = ru
    bypair = {}
    for t, S in enumerate(cuts):
        a, b, c = sorted(S)
        for p in ((a, b), (a, c), (b, c)): bypair.setdefault(p, []).append(t)
    start = np.zeros(m + 1, np.int32); idx = []
    for j, c in enumerate(cols):
        rows = list(c)
        if odd and len(c) % 2: rows.append(n)
        hit = set()
        for a in range(len(c)):
            for b in range(a + 1, len(c)):
                hit.update(bypair.get((c[a], c[b]), ()))
        rows.extend(p0 + t for t in sorted(hit))
        idx.extend(rows); start[j + 1] = start[j] + len(rows)
    for i in range(n):
        rows = [i] + ([n] if odd else [])
        idx.extend(rows); start[C + i + 1] = start[C + i] + len(rows)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise; lp.a_matrix_.start_ = start
    lp.a_matrix_.index_ = np.array(idx, np.int32); lp.a_matrix_.value_ = np.ones(len(idx))
    h.passModel(lp); h.run()
    st = h.modelStatusToString(h.getModelStatus())
    if st != "Optimal":
        raise RuntimeError(f"node master not solved: {st}")
    sol = h.getSolution(); y = np.array(sol.row_dual); x = np.array(sol.col_value)
    tau = max(float(y[n]), 0.0) if odd else 0.0
    sigma = np.minimum(y[p0:], 0.0)
    return h.getInfo().objective_function_value, y[:n], tau, sigma, x[:C], float(x[C:].sum())


def _master_r1(cols, costs, n, odd, cuts):
    """As _master, with rank-1 cut rows (coefficient floor(u . a_c), right-hand side floor(u . 1))."""
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
    C = len(cols); m = C + n; p0 = n + (1 if odd else 0); nr = p0 + len(cuts)
    lp = highspy.HighsLp(); lp.num_col_, lp.num_row_ = m, nr
    lp.col_cost_ = np.concatenate([np.asarray(costs, float), np.full(n, BIG)])
    lp.col_lower_ = np.zeros(m); lp.col_upper_ = np.full(m, np.inf)
    rl = np.ones(nr); ru = np.ones(nr)
    if odd: ru[n] = np.inf
    rl[p0:] = -np.inf; ru[p0:] = [c.rhs for c in cuts]
    lp.row_lower_ = rl; lp.row_upper_ = ru
    start = np.zeros(m + 1, np.int32); idx = []; val = []
    for j, c in enumerate(cols):
        ent = [(g, 1.0) for g in c] + ([(n, 1.0)] if odd and len(c) % 2 else [])
        for t, cut in enumerate(cuts):
            cf = cut.coef(c)
            if cf: ent.append((p0 + t, float(cf)))
        for r, v in ent: idx.append(r); val.append(v)
        start[j + 1] = start[j] + len(ent)
    for i in range(n):
        ent = [(i, 1.0)] + ([(n, 1.0)] if odd else [])
        for r, v in ent: idx.append(r); val.append(v)
        start[C + i + 1] = start[C + i] + len(ent)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise; lp.a_matrix_.start_ = start
    lp.a_matrix_.index_ = np.array(idx, np.int32); lp.a_matrix_.value_ = np.array(val)
    h.passModel(lp); h.run()
    st = h.modelStatusToString(h.getModelStatus())
    if st != "Optimal":
        raise RuntimeError(f"node master not solved: {st}")
    sol = h.getSolution(); y = np.array(sol.row_dual); x = np.array(sol.col_value)
    tau = max(float(y[n]), 0.0) if odd else 0.0
    return h.getInfo().objective_function_value, y[:n], tau, np.minimum(y[p0:], 0.0), x[:C], float(x[C:].sum())


class BCP:
    def __init__(self, z0, lo=2, hi=4, add=2000, cuts=True, group_branching=True,
                 cuts_per_round=50, max_cuts=2000, cut_rounds=20, verbose=True, families=None,
                 purge_after=None, deep_depth=None, max_active=None, shallow_families=((3, (1, 1, 1), 2),),
                 fix_columns=True, universe_limit=200_000):
        self.n = n = len(z0); self.lo, self.hi, self.add = lo, hi, add; self.odd = n % 2 == 1
        self.z = np.asarray(z0, float).copy(); self.scale = 1.0; self.rescaled = False
        self.cols = list(sg.initial_pool(self.z, lo, hi)); self.score = {c: dive._sc(self.z, c) for c in self.cols}
        self.cuts = []; self.use_cuts = cuts; self.group_branching = group_branching
        self.cuts_per_round, self.max_cuts, self.cut_rounds = cuts_per_round, max_cuts, cut_rounds
        self.verbose = verbose; self.stats = dict(nodes=0, group_branches=0, pair_branches=0, lps=0)
        self.families = families; self.r1keys = set()          # families: rank-1 cuts from rank1 in place of SR3
        # cut management (rank-1 path): purge a cut after `purge_after` LP solves with a zero dual; separate
        # every family only at depth <= deep_depth and `shallow_families` below; keep at most max_active cuts,
        # the most recently active first. Dropping a cut weakens the relaxation and keeps every bound valid.
        self.purge_after, self.deep_depth, self.max_active = purge_after, deep_depth, max_active
        self.shallow_families = list(shallow_families); self.age = {}
        self.stats.update(purged=0, max_pool=0)
        self.fix_columns, self.universe_limit = fix_columns, universe_limit
        self.universe = None; self._root = None; self._last = None
        self.stats.update(universe=0, universe_at_fix=0, refixes=0, scans_after_fix=0)

    def _QR(self, sigma):
        return sr.duals_to_QR(self.n, self.cuts, sigma) if len(self.cuts) else (np.zeros((self.n, self.n)), np.zeros((1, 1, 1)))

    def _cg(self, T):
        if self.families is not None:
            return self._cg_r1(T)
        n = self.n
        while True:
            cols = [c for c in self.cols if bnp.compat_py(T, c)]
            lp, pi, tau, sigma, x, art = _master(cols, [self.score[c] for c in cols], n, self.odd, self.cuts)
            self.stats["lps"] += 1
            if not self.rescaled and 0 < lp < 1e-2 * n and art < 1e-9:
                k = int(round(math.log(n / lp, 4))); self.z = self.z * 2.0 ** k; self.scale *= 4.0 ** k
                self.rescaled = True; self.score = {c: dive._sc(self.z, c) for c in self.score}; continue
            new = 0; best = 0.0; scan = None
            if self.universe is not None:
                pass                    # every group that can be in a better design is in the master
            else:
                self.stats["scans_after_fix"] += self.universe is not None
                Q, R = self._QR(sigma)
                if R.shape[0] == 1: R = np.zeros((n, n, n))
                rows, s_, rc, m, best = _scan_bcp(self.z, pi, tau, Q, R, T, self.lo, self.hi, -1e-9 * max(1.0, abs(lp)), 5_000_000)
                for t in np.argsort(rc)[:self.add]:
                    c = tuple(int(v) for v in rows[t] if v >= 0)
                    if c not in self.score:
                        self.score[c] = float(s_[t]); self.cols.append(c); new += 1
                z_ = self.z
                scan = lambda th, cap: _scan_bcp(z_, pi, tau, Q, R, T, self.lo, self.hi, th, cap)
            if new == 0:
                bound = float(np.sum(pi)) + tau + float(np.sum(sigma)) + (n / self.lo) * min(0.0, best)
                self._last = (bound, best, scan)
                return bound, lp, dict(zip(cols, x)), art

    def _cg_r1(self, T):
        n = self.n
        while True:
            cols = [c for c in self.cols if bnp.compat_py(T, c)]
            lp, pi, tau, sigma, x, art = _master_r1(cols, [self.score[c] for c in cols], n, self.odd, self.cuts)
            self.stats["lps"] += 1
            for sg_, c in zip(sigma, self.cuts):
                k_ = c.key(); self.age[k_] = 0 if sg_ < -1e-12 else self.age.get(k_, 0) + 1
            if not self.rescaled and 0 < lp < 1e-2 * n and art < 1e-9:
                k = int(round(math.log(n / lp, 4))); self.z = self.z * 2.0 ** k; self.scale *= 4.0 ** k
                self.rescaled = True; self.score = {c: dive._sc(self.z, c) for c in self.score}; continue
            new = 0; best = 0.0; scan = None
            if self.universe is not None:
                pass                    # every group that can be in a better design is in the master
            else:
                self.stats["scans_after_fix"] += self.universe is not None
                cs, cn, cd = rank1._arrays(self.cuts)
                sig = np.zeros(max(len(self.cuts), 1)); sig[:len(self.cuts)] = sigma; mc = len(self.cuts)
                rows, s_, rc, m, best = _scan_bcp_r1(self.z, pi, tau, cs, cn, cd, sig, mc, T, self.lo, self.hi,
                                                     -1e-9 * max(1.0, abs(lp)), 5_000_000)
                for t in np.argsort(rc)[:self.add]:
                    c = tuple(int(v) for v in rows[t] if v >= 0)
                    if c not in self.score:
                        self.score[c] = float(s_[t]); self.cols.append(c); new += 1
                z_ = self.z
                scan = lambda th, cap: _scan_bcp_r1(z_, pi, tau, cs, cn, cd, sig, mc, T, self.lo, self.hi, th, cap)
            if new == 0:
                bound = (float(np.sum(pi)) + tau + float(sum(sg_ * c.rhs for sg_, c in zip(sigma, self.cuts)))
                         + (n / self.lo) * min(0.0, best))
                self._last = (bound, best, scan)
                return bound, lp, dict(zip(cols, x)), art

    def _purge(self):
        if self.families is None:
            return
        drop = set()
        if self.purge_after is not None:
            drop |= {c.key() for c in self.cuts if self.age.get(c.key(), 0) >= self.purge_after}
        keep = [c for c in self.cuts if c.key() not in drop]
        if self.max_active is not None and len(keep) > self.max_active:
            keep.sort(key=lambda c: self.age.get(c.key(), 0))
            drop |= {c.key() for c in keep[self.max_active:]}
            keep = keep[:self.max_active]
        if drop:
            self.cuts = [c for c in self.cuts if c.key() not in drop]
            for k_ in drop: self.r1keys.discard(k_); self.age.pop(k_, None)
            self.stats["purged"] += len(drop)
        self.stats["max_pool"] = max(self.stats["max_pool"], len(self.cuts))

    def _fix(self, ub, design):
        """Keep only the groups that can be in a design costing at most ub (root duals; see the module docstring)."""
        if not self.fix_columns or self._root is False:
            return
        if self._root is None:
            bound, best, scan = self._root_last      # the root's duals: a node's hold only in its subtree
            m_ = min(0.0, best)
            theta = ub - bound + m_ + 1e-9 * max(1.0, abs(ub))
            rows, s_, rc, cnt, _ = scan(theta, self.universe_limit + 1)
            if cnt > self.universe_limit:
                self._root = False
                return
            cols = [tuple(int(v) for v in r if v >= 0) for r in rows]
            for c, sc in zip(cols, s_):
                self.score.setdefault(c, float(sc))
            self._root = dict(bound=bound, m=m_, rc=dict(zip(cols, (float(r) for r in rc))))
            self.stats["universe_at_fix"] = len(cols)
        else:
            self.stats["refixes"] += 1
        r = self._root
        theta = ub - r["bound"] + r["m"] + 1e-9 * max(1.0, abs(ub))
        keep = {c for c, rc in r["rc"].items() if rc <= theta} | {tuple(sorted(int(v) for v in g)) for g in design}
        for c in keep:
            if c not in self.score: self.score[c] = dive._sc(self.z, c)
        self.universe = keep
        self.cols = sorted(keep)
        self.stats["universe"] = len(keep)

    def node(self, T, ub, depth=0):
        """Column generation with cut separation. Returns (bound, x, art)."""
        P = np.zeros((self.n, self.n)); Tt = np.zeros((self.n, self.n, self.n))
        fams = self.families if (self.deep_depth is None or depth <= self.deep_depth) else self.shallow_families
        self._purge()
        for r in range(self.cut_rounds + 1):
            bound, lp, x, art = self._cg(T)
            if art > 1e-6 or bound >= ub * (1 - 1e-9) or not self.use_cuts or len(self.cuts) >= self.max_cuts or r == self.cut_rounds:
                return bound, x, art
            if self.families is not None:
                found = rank1.separate(list(x.keys()), np.array(list(x.values())), self.n, fams)
                new = [c for v, c in found if c.key() not in self.r1keys][:self.cuts_per_round]
                if not new:
                    return bound, x, art
                for c in new: self.r1keys.add(c.key()); self.cuts.append(c); self.age[c.key()] = 0
                self._purge()
                continue
            sup = [(c, v) for c, v in x.items() if v > 1e-9]
            cr = np.zeros((len(sup), 4), np.int64); cl = np.zeros(len(sup), np.int64); xv = np.zeros(len(sup))
            for a, (c, v) in enumerate(sup):
                cl[a] = len(c); cr[a, :len(c)] = c; xv[a] = v
            viol = sr._separate(self.n, cr, cl, xv, P, Tt)
            have = set(self.cuts)
            new = [tuple(int(v) for v in viol[t, :3]) for t in np.argsort(-viol[:, 3])]
            new = [c for c in new if c not in have][:self.cuts_per_round]
            if not new:
                return bound, x, art
            self.cuts.extend(new)

    def _branch(self, T, x, ub):
        n = self.n
        frac = {c: v for c, v in x.items() if v > 1e-9}
        cover = {}
        for c, v in frac.items():
            for i in c: cover.setdefault(i, []).append(v)
        fractional = [i for i, vs in cover.items() if len(vs) > 1 or abs(vs[0] - 1) > 1e-9]
        if self.group_branching and fractional:
            best = None
            for i in fractional:
                g, los, m = _viable(self.z, T, i, self.lo, self.hi, ub, GROUP_LIMIT)
                if m <= GROUP_LIMIT and (best is None or m < best[0]):
                    best = (m, i, g.copy(), los.copy())
            if best is not None:
                m, i, groups, losses = best
                self.stats["group_branches"] += 1
                kids = []
                for g, lo_g in zip(groups, losses):
                    g = [int(v) for v in g if v >= 0]
                    T2 = T.copy()
                    for a in g:
                        for b in range(n):
                            if b == a: continue
                            T2[a, b] = 1 if b in g else -1
                    for a in g:
                        for b in range(n):
                            if b != a: T2[b, a] = T2[a, b]
                    kids.append((float(lo_g), T2))
                return kids
        w = {}
        for c, v in frac.items():
            for a in range(len(c)):
                for b in range(a + 1, len(c)):
                    w[(c[a], c[b])] = w.get((c[a], c[b]), 0.0) + v
        cand = [(p, v) for p, v in w.items() if 1e-6 < v < 1 - 1e-6]
        if not cand:
            raise RuntimeError("fractional master with no fractional pair")
        (i, j), _ = min(cand, key=lambda t: (abs(t[1] - 0.5), -t[1]))
        self.stats["pair_branches"] += 1
        kids = []
        for sign in (1, -1):
            T2 = T.copy(); T2[i, j] = T2[j, i] = sign
            if sign == 1:
                grp = {i, j} | {k for k in range(n) if T[i, k] == 1 or T[j, k] == 1}
                if len(grp) > self.hi or any(T[a, b] == -1 for a in grp for b in grp if a != b):
                    continue
                for a in grp:
                    for b in grp:
                        if a != b: T2[a, b] = 1
                for a in grp:
                    for k in range(n):
                        if T[a, k] == -1:
                            for b in grp: T2[b, k] = T2[k, b] = -1
            kids.append((0.0, T2))
        return kids

    def solve(self, incumbent=None, time_limit=600, node_limit=10**7, heuristic_seconds=0.0, root_mip_seconds=0.0):
        n = self.n; t0 = time.perf_counter()
        T0 = np.zeros((n, n), np.int8)
        bound, x, art = self.node(T0, np.inf)
        root_bound = bound
        self.rescaled = True        # the cost scale is fixed at the root: the incumbent and every bound share it
        self._root_last = self._last
        best_design, ub = None, np.inf
        if incumbent is None and heuristic_seconds > 0:
            o = list(np.argsort(self.z))
            start = [tuple(o[i:i + 2]) for i in range(0, n - 3, 2)] + [tuple(o[-3:])] if self.odd else [tuple(o[i:i + 2]) for i in range(0, n, 2)]
            incumbent, _, _ = lns.lns_dp(self.z, start, seconds=heuristic_seconds, verbose=False)
        if incumbent is not None:
            best_design = [tuple(sorted(int(v) for v in g)) for g in incumbent]
            ub = sum(dive._sc(self.z, g) for g in best_design)
            self._fix(ub, best_design)
        heap = [(bound, 0, T0, (bound, x, art), 0)]; nid = 0; status = "optimal"; last = t0
        while heap:
            if time.perf_counter() - t0 > time_limit: status = "time limit"; break
            if self.stats["nodes"] >= node_limit: status = "node limit"; break
            bnd, _, T, cached, depth = heapq.heappop(heap)
            if bnd >= ub * (1 - 1e-9): continue
            self.stats["nodes"] += 1
            if cached is None:
                bnd, x, art = self.node(T, ub, depth)
            else:
                bnd, x, art = cached
            if art > 1e-6 or bnd >= ub * (1 - 1e-9): continue
            frac = {c: v for c, v in x.items() if v > 1e-9}
            if frac and all(abs(v - 1) < 1e-9 for v in frac.values()):
                val = sum(self.score[c] for c in frac)
                if val < ub:
                    ub, best_design = val, list(frac)
                    self._fix(ub, best_design)
                continue
            for prio, T2 in self._branch(T, x, ub):
                nid += 1
                heapq.heappush(heap, (max(bnd, prio), nid, T2, None, depth + 1))
            now = time.perf_counter()
            if self.verbose and now - last > 30:
                glb = min([h[0] for h in heap], default=ub)
                print(f"    [{now - t0:.0f}s] nodes {self.stats['nodes']}, open {len(heap)}, cuts {len(self.cuts)}, "
                      f"bound {glb / self.scale:.8g}, incumbent {ub / self.scale:.8g}", flush=True)
                last = now
        glb = min([h[0] for h in heap], default=ub) if heap else ub
        return dict(status=status, root_bound=root_bound / self.scale, bound=min(glb, ub) / self.scale, opt=ub / self.scale,
                    design=best_design, cuts=len(self.cuts), t=time.perf_counter() - t0, **self.stats)
