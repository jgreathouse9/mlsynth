"""Branch and price with Ryan-Foster branching for the supergeo base problem.

Node: a set of pair decisions, together (i and j in the same group) or apart (never
in the same group), as a matrix T with T[i,j] = 1 together, -1 apart, 0 free. A
column is compatible when it holds no apart pair and, for every member i, every
together-partner of i. Each node runs column generation over compatible columns;
pricing checks compatibility inside the loop, so a forbidden column is never priced
back in. Artificial single-geo columns at a large cost keep every node's master
feasible; a node whose converged master still uses one has no design.

Bound at a node (Lagrangian, valid for any duals): sum pi + tau + (N/lo) min(0, min
reduced cost over compatible columns). Branching: the pair whose together-weight
w_ij = sum of x over columns holding both is closest to 1/2. Best-first search.
Floating point throughout (a spike); the exact layer can be added as for the root."""
import heapq, math, time
import numpy as np
from numba import njit
import highspy
from . import sg, dive

BIG = 1e7


@njit(cache=True)
def _compat(T, idx, k, n):
    for a in range(k):
        i = idx[a]
        for b in range(a + 1, k):
            if T[i, idx[b]] == -1:
                return False
        for j in range(n):                       # every together-partner of i must be in the column
            if T[i, j] == 1:
                found = False
                for b in range(k):
                    if idx[b] == j:
                        found = True; break
                if not found:
                    return False
    return True


@njit(cache=True)
def _scan_rf(z, pi, tau, T, lo, hi, thresh, cap):
    n = z.shape[0]
    out = np.full((cap, 4), -1, np.int64); sc = np.empty(cap); rc = np.empty(cap)
    idx = np.empty(4, np.int64); m = 0; best = np.inf
    for i in range(n):
        idx[0] = i
        for j in range(i + 1, n):
            idx[1] = j
            if lo <= 2 and _compat(T, idx, 2, n):
                s = sg._score(z, idx, 2); r = s - pi[i] - pi[j]
                if r < best: best = r
                if r <= thresh and m < cap:
                    out[m, 0] = i; out[m, 1] = j; sc[m] = s; rc[m] = r; m += 1
            if hi < 3: continue
            for c in range(j + 1, n):
                idx[2] = c
                if lo <= 3 and _compat(T, idx, 3, n):
                    s = sg._score(z, idx, 3); r = s - pi[i] - pi[j] - pi[c] - tau
                    if r < best: best = r
                    if r <= thresh and m < cap:
                        out[m, 0] = i; out[m, 1] = j; out[m, 2] = c; sc[m] = s; rc[m] = r; m += 1
                if hi < 4: continue
                for d in range(c + 1, n):
                    idx[3] = d
                    if _compat(T, idx, 4, n):
                        s = sg._score(z, idx, 4); r = s - pi[i] - pi[j] - pi[c] - pi[d]
                        if r < best: best = r
                        if r <= thresh and m < cap:
                            out[m, 0] = i; out[m, 1] = j; out[m, 2] = c; out[m, 3] = d; sc[m] = s; rc[m] = r; m += 1
    return out[:m], sc[:m], rc[:m], m, best


def compat_py(T, c):
    idx = np.array(tuple(c) + (0,) * (4 - len(c)), np.int64)
    return _compat(T, idx, len(c), T.shape[0])


def master(cols, costs, n, odd):
    """Master LP with artificial singletons; returns lp, pi, tau, x (x over cols only), artificial use."""
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
    m = len(cols) + n
    lp = highspy.HighsLp(); lp.num_col_ = m; lp.num_row_ = n + (1 if odd else 0)
    lp.col_cost_ = np.concatenate([np.asarray(costs, float), np.full(n, BIG)])
    lp.col_lower_ = np.zeros(m); lp.col_upper_ = np.full(m, np.inf)
    ru = np.ones(lp.num_row_)
    if odd: ru[n] = np.inf                                # parity row: sum over odd columns >= 1
    lp.row_lower_ = np.ones(lp.num_row_); lp.row_upper_ = ru  # (indexing lp.row_upper_ in place edits a copy)
    start = np.zeros(m + 1, np.int32); idx = []; val = []
    for j, c in enumerate(cols):
        rows = list(c) + ([n] if odd and len(c) % 2 else [])
        idx.extend(rows); val.extend([1.0] * len(rows)); start[j + 1] = start[j] + len(rows)
    for i in range(n):                                    # artificial singleton i (odd size: counts for parity)
        rows = [i] + ([n] if odd else [])
        idx.extend(rows); val.extend([1.0] * len(rows)); start[len(cols) + i + 1] = start[len(cols) + i] + len(rows)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise; lp.a_matrix_.start_ = start
    lp.a_matrix_.index_ = np.array(idx, np.int32); lp.a_matrix_.value_ = np.array(val)
    h.passModel(lp); h.run()
    status = h.modelStatusToString(h.getModelStatus())
    if status != "Optimal":
        raise RuntimeError(f"node master not solved: {status}")
    sol = h.getSolution(); y = np.array(sol.row_dual); x = np.array(sol.col_value)
    tau = max(float(y[n]), 0.0) if odd else 0.0
    return h.getInfo().objective_function_value, y[:n], tau, x[:len(cols)], float(x[len(cols):].sum())


class BnP:
    def __init__(self, z0, lo=2, hi=4, add=2000, verbose=True):
        self.n = n = len(z0); self.lo, self.hi, self.add = lo, hi, add; self.odd = n % 2 == 1
        self.z = np.asarray(z0, float).copy(); self.scale = 1.0; self.verbose = verbose
        self.cols = list(sg.initial_pool(self.z, lo, hi)); self.score = {c: dive._sc(self.z, c) for c in self.cols}
        self.rescaled = False

    def node_cg(self, T, max_rounds=500):
        """Column generation at a node. Returns (bound, lp, x by column, artificial use, rounds) in scaled units."""
        n = self.n; rounds = 0
        while True:
            cols = [c for c in self.cols if compat_py(T, c)]
            lp, pi, tau, x, art = master(cols, [self.score[c] for c in cols], n, self.odd)
            if not self.rescaled and 0 < lp < 1e-2 * n and art < 1e-9:
                k = int(round(math.log(n / lp, 4))); f = 4.0 ** k
                self.z = self.z * 2.0 ** k; self.scale *= f; self.rescaled = True
                self.score = {c: dive._sc(self.z, c) for c in self.score}; continue
            rounds += 1
            rows, s_, rc, m, best = _scan_rf(self.z, pi, tau, T, self.lo, self.hi, -1e-9 * max(1.0, abs(lp)), 5_000_000)
            new = 0
            for t in np.argsort(rc)[:self.add]:
                c = tuple(int(v) for v in rows[t] if v >= 0)
                if c not in self.score:
                    self.score[c] = float(s_[t]); self.cols.append(c); new += 1
            if new == 0 or rounds >= max_rounds:
                bound = float(np.sum(pi)) + tau + (n / self.lo) * min(0.0, best)
                return bound, lp, dict(zip(cols, x)), art, rounds

    def solve(self, incumbent=None, time_limit=600, node_limit=10**7):
        n = self.n; t0 = time.perf_counter()
        best_design, ub = None, np.inf
        if incumbent is not None:
            best_design = [tuple(sorted(int(v) for v in g)) for g in incumbent]
        root_T = np.zeros((n, n), np.int8)
        heap = []; nid = 0; nodes = 0; root_bound = None
        bound, lp, x, art, r = self.node_cg(root_T)
        self.rescaled = True        # the cost scale is fixed at the root: the incumbent and every bound share it
        if best_design is not None:
            ub = sum(dive._sc(self.z, g) for g in best_design)          # scaled units, after any rescale
        heapq.heappush(heap, (bound, nid, root_T, (bound, lp, x, art)))
        root_bound = bound; last = t0
        status = "optimal"
        while heap:
            if time.perf_counter() - t0 > time_limit: status = "time limit"; break
            if nodes >= node_limit: status = "node limit"; break
            bound, _, T, cached = heapq.heappop(heap)
            if bound >= ub * (1 - 1e-12): continue
            nodes += 1
            if cached is None:
                bound, lp, x, art = self.node_cg(T)[:4]
                if art > 1e-6 or bound >= ub * (1 - 1e-12): continue
            else:
                bound, lp, x, art = cached
                if art > 1e-6: continue
            frac = {c: v for c, v in x.items() if v > 1e-9}
            if frac and all(abs(v - 1) < 1e-9 for v in frac.values()):  # integral: a design
                val = sum(self.score[c] for c in frac)
                if val < ub: ub, best_design = val, list(frac)
                continue
            w = {}
            for c, v in frac.items():
                for a in range(len(c)):
                    for b in range(a + 1, len(c)):
                        w[(c[a], c[b])] = w.get((c[a], c[b]), 0.0) + v
            cand = [(p, v) for p, v in w.items() if 1e-6 < v < 1 - 1e-6]
            if not cand:                                                # Ryan-Foster: cannot happen for a fractional x
                raise RuntimeError("fractional master with no fractional pair")
            (i, j), wij = min(cand, key=lambda t: (abs(t[1] - 0.5), -t[1]))
            for sign in (1, -1):
                T2 = T.copy(); T2[i, j] = T2[j, i] = sign
                if sign == 1:                                           # together is transitive: close it
                    grp = {i, j} | {k for k in range(n) if T[i, k] == 1 or T[j, k] == 1}
                    if len(grp) > self.hi: continue
                    if any(T[a, b] == -1 for a in grp for b in grp if a != b): continue    # together with an apart pair: empty
                    for a in grp:
                        for b in grp:
                            if a != b: T2[a, b] = 1
                    for a in grp:                                       # apart from one member: apart from all
                        for k in range(n):
                            if T[a, k] == -1:
                                for b in grp: T2[b, k] = T2[k, b] = -1
                nid += 1
                heapq.heappush(heap, (bound, nid, T2, None))
            now = time.perf_counter()
            if self.verbose and now - last > 30:
                glb = min([h[0] for h in heap], default=ub)
                print(f"    [{now - t0:.0f}s] nodes {nodes}, open {len(heap)}, bound {glb / self.scale:.8g}, incumbent {ub / self.scale:.8g}, "
                      f"gap {(ub - glb) / ub * 100 if ub < np.inf else float('nan'):.3g}%", flush=True)
                last = now
        glb = min([h[0] for h in heap], default=ub) if heap else ub
        return dict(status=status, nodes=nodes, root_bound=root_bound / self.scale, bound=min(glb, ub) / self.scale,
                    opt=ub / self.scale, design=best_design, t=time.perf_counter() - t0)
