"""Dual-steered repair for the supergeo base problem (after Maher and Ronnberg 2023).

A move takes a column the LP likes (low reduced cost at the CG duals), frees
every incumbent group that touches it, tops the freed set up to `max_free`
geos with the incumbent groups that share the most low-reduced-cost columns
with it, and repairs the freed geos exactly by the bitmask DP. All costs are in
the rescaled units certify4 works in."""
import time
import numpy as np
from . import sg, dive, lns


def column_generation(z, lo=2, hi=4, add=2000):
    n = len(z); z = np.asarray(z, float).copy(); scale = 1.0
    cols = sg.initial_pool(z, lo, hi)
    score = {c: dive._sc(z, c) for c in cols}
    best_lb = -np.inf
    while True:
        lp, pi, x = sg.master_lp(cols, np.array([score[c] for c in cols]), n)
        if 0 < lp < 1e-2 * n:
            f = n / lp
            z *= np.sqrt(f); scale *= f
            score = {c: v * f for c, v in score.items()}
            best_lb *= f
            continue
        lb, mrc = sg.lagrangian_lb(z, pi, lo, hi)
        best_lb = max(best_lb, lb)
        rows, s_, rc, m = sg._scan(z, pi, lo, hi, -1e-9 * max(1.0, lp), 5_000_000)
        new = 0
        for t in np.argsort(rc)[:add]:
            c = tuple(int(v) for v in rows[t] if v >= 0)
            if c not in score:
                score[c] = float(s_[t]); cols.append(c); new += 1
        if new == 0:
            return z, scale, pi, best_lb, lp


def low_rc_columns(z, pi, lo, hi, target):
    a, b = 1e-9, 1.0
    while sg._count(z, pi, lo, hi, b) < target:
        b *= 10
    for _ in range(12):
        mid = np.sqrt(a * b)
        if sg._count(z, pi, lo, hi, mid) < target: a = mid
        else: b = mid
    rows, s_, rc, m = sg._scan(z, pi, lo, hi, b, int(target * 1.6) + 10)
    return rows[:m].copy(), rc[:m].copy()


def steered(z, pi, design, rows, rc, lo=2, hi=4, max_free=18, seconds=300, seed=0,
            tau=None, p_column=0.8, verbose=True, log_every=60):
    """Dual-steered exact-repair LNS. rows/rc: the low-reduced-cost columns."""
    rng = np.random.default_rng(seed)
    n = len(z)
    tau = tau or float(np.median(rc) - rc.min() + 1e-12)
    w_col = np.exp(-(rc - rc.min()) / tau); p_col = w_col / w_col.sum()
    design = [tuple(g) for g in design]
    loss = [lns.group_loss(z, g) for g in design]

    def rebuild():
        g2 = np.full(n, -1, np.int64)
        for i, g in enumerate(design):
            g2[list(g)] = i
        G = len(design)
        grp = np.where(rows >= 0, g2[np.maximum(rows, 0)], -1)
        M = np.zeros((G, G))
        for a in range(4):
            for b in range(a + 1, 4):
                ok = (grp[:, a] >= 0) & (grp[:, b] >= 0) & (grp[:, a] != grp[:, b])
                np.add.at(M, (grp[ok, a], grp[ok, b]), w_col[ok])
                np.add.at(M, (grp[ok, b], grp[ok, a]), w_col[ok])
        return g2, grp, M

    g2, grp, M = rebuild()
    t0 = time.perf_counter(); last = t0; trace = [(0.0, sum(loss))]
    it = acc = 0
    while time.perf_counter() - t0 < seconds:
        it += 1
        if rng.random() < p_column:                   # seed: a column the LP likes
            t = int(rng.choice(len(rc), p=p_col))
            pick = sorted({int(v) for v in grp[t] if v >= 0})
        else:                                         # seed: the group the LP likes least
            pr = np.array([loss[i] - pi[list(g)].sum() for i, g in enumerate(design)])
            pr = np.exp((pr - pr.max()) / (abs(pr).mean() + 1e-12))
            pick = [int(rng.choice(len(design), p=pr / pr.sum()))]
        size = sum(len(design[i]) for i in pick)
        if size > max_free:
            continue
        aff = M[pick].sum(0) * rng.exponential(1.0, len(design))
        aff[pick] = -1
        for h in np.argsort(-aff):
            if aff[h] <= 0: break
            if size + len(design[h]) <= max_free:
                pick.append(int(h)); size += len(design[h])
        free = np.array(sorted(v for i in pick for v in design[i]), dtype=np.int64)
        cur = sum(loss[i] for i in pick)
        groups, new = lns.repair_dp(z, free, lo, hi)
        if new < cur * (1 - 1e-9):
            acc += 1
            ps = set(pick)
            keep = [i for i in range(len(design)) if i not in ps]
            design = [design[i] for i in keep] + groups
            loss = [loss[i] for i in keep] + [lns.group_loss(z, g) for g in groups]
            g2, grp, M = rebuild()
        now = time.perf_counter()
        trace.append((now - t0, sum(loss)))
        if verbose and now - last > log_every:
            print(f"    [{now - t0:.0f}s] iter {it} accepted {acc} loss {sum(loss):.6g}", flush=True)
            last = now
    return design, sum(loss), trace


import highspy


def solve_partition(cols, costs, nrow, warm=None, time_limit=10.0, return_status=False, gap=None):
    """Set partitioning MIP over `cols` (tuples of local row indices) with HiGHS,
    warm-started from `warm` (indices into cols). Returns (chosen indices, value).
    `gap`, when given, is both the relative and the absolute MIP gap; HiGHS's default
    relative gap of 1e-4 stops at the warm start when it is that close."""
    h = highspy.Highs(); h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit)); h.setOptionValue("threads", 1)
    if gap is not None:
        h.setOptionValue("mip_rel_gap", float(gap)); h.setOptionValue("mip_abs_gap", float(gap))
    lp = highspy.HighsLp()
    m = len(cols)
    lp.num_col_, lp.num_row_ = m, nrow
    lp.col_cost_ = np.asarray(costs, float)
    lp.col_lower_ = np.zeros(m); lp.col_upper_ = np.ones(m)
    lp.row_lower_ = np.ones(nrow); lp.row_upper_ = np.ones(nrow)
    start = np.zeros(m + 1, np.int32); idx = []
    for j, c in enumerate(cols):
        idx.extend(c); start[j + 1] = start[j] + len(c)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = start; lp.a_matrix_.index_ = np.array(idx, np.int32)
    lp.a_matrix_.value_ = np.ones(len(idx))
    lp.integrality_ = [highspy.HighsVarType.kInteger] * m
    h.passModel(lp)
    if warm is not None:
        sol = highspy.HighsSolution(); x = np.zeros(m); x[list(warm)] = 1.0
        sol.col_value = list(x); h.setSolution(sol)
    h.run()
    x = np.array(h.getSolution().col_value)
    status = h.modelStatusToString(h.getModelStatus())
    if x.size == 0:
        return (None, np.inf, status) if return_status else (None, np.inf)
    pick = np.flatnonzero(x > 0.5)
    val = float(np.asarray(costs)[pick].sum())
    return (pick, val, status) if return_status else (pick, val)


def steered_mip(z, pi, design, rows, rc, lo=2, hi=4, free_geos=48, k_cols=20_000, gamma=0.9,
                sub_limit=10.0, seconds=300, seed=0, steer=True, verbose=True, log_every=60):
    """IPColGen-style repair: free about `free_geos` geos (steered by low-rc columns and
    their affinity, or by size rank when steer=False), give the repair MIP the incumbent's
    own groups plus the k_cols columns over the freed geos of lowest damped reduced cost
    score - gamma * pi(c), warm-started from the incumbent."""
    rng = np.random.default_rng(seed)
    n = len(z)
    rank = np.empty(n, int); rank[np.argsort(z)] = np.arange(n)
    tau = float(np.median(rc) - rc.min() + 1e-12)
    w_col = np.exp(-(rc - rc.min()) / tau); p_col = w_col / w_col.sum()
    design = [tuple(sorted(g)) for g in design]
    loss = [lns.group_loss(z, g) for g in design]

    def affinity():
        g2 = np.full(n, -1, np.int64)
        for i, g in enumerate(design): g2[list(g)] = i
        grp = np.where(rows >= 0, g2[np.maximum(rows, 0)], -1)
        G = len(design); M = np.zeros((G, G))
        for a in range(4):
            for b in range(a + 1, 4):
                ok = (grp[:, a] >= 0) & (grp[:, b] >= 0) & (grp[:, a] != grp[:, b])
                np.add.at(M, (grp[ok, a], grp[ok, b]), w_col[ok]); np.add.at(M, (grp[ok, b], grp[ok, a]), w_col[ok])
        return grp, M

    grp, M = affinity()
    t0 = time.perf_counter(); last = t0; trace = [(0.0, sum(loss))]
    it = acc = 0
    while time.perf_counter() - t0 < seconds:
        it += 1
        if steer:
            t = int(rng.choice(len(rc), p=p_col))
            pick = sorted({int(v) for v in grp[t] if v >= 0})
            aff = M[pick].sum(0) * rng.exponential(1.0, len(design)); aff[pick] = -1
            order = [int(h) for h in np.argsort(-aff)]
        else:
            a = int(rng.integers(len(design))); pick = [a]
            centre = np.mean([rank[v] for v in design[a]])
            order = [int(h) for h in np.argsort([abs(np.mean([rank[v] for v in g]) - centre) + rng.exponential(10) for g in design])]
        size = sum(len(design[i]) for i in pick)
        for h in order:
            if h in pick: continue
            if size + len(design[h]) > free_geos: continue
            pick.append(h); size += len(design[h])
            if size >= free_geos - 1: break
        free = np.array(sorted(v for i in pick for v in design[i]), dtype=np.int64)
        loc = {int(g): i for i, g in enumerate(free)}
        cur = sum(loss[i] for i in pick)
        zf, pf = z[free], gamma * pi[free]
        r_, s_, rc_, m_ = sg._scan(zf, pf, lo, hi, np.inf, 5_000_000)
        keep = np.argsort(rc_[:m_])[:k_cols]
        cand = {tuple(int(v) for v in r_[t] if v >= 0): float(s_[t]) for t in keep}
        inc = [tuple(sorted(loc[v] for v in design[i])) for i in pick]
        for c, i in zip(inc, pick): cand.setdefault(c, loss[i])
        cols = list(cand); pos = {c: j for j, c in enumerate(cols)}
        sel, new = solve_partition(cols, [cand[c] for c in cols], len(free),
                                   warm=[pos[c] for c in inc], time_limit=sub_limit)
        if sel is not None and new < cur * (1 - 1e-9):
            acc += 1
            ps = set(pick)
            kept = [i for i in range(len(design)) if i not in ps]
            groups = [tuple(sorted(int(free[v]) for v in cols[j])) for j in sel]
            design = [design[i] for i in kept] + groups
            loss = [loss[i] for i in kept] + [lns.group_loss(z, g) for g in groups]
            grp, M = affinity()
        now = time.perf_counter()
        trace.append((now - t0, sum(loss)))
        if verbose and now - last > log_every:
            print(f"    [{now - t0:.0f}s] iter {it} accepted {acc} loss {sum(loss):.6g}", flush=True)
            last = now
    return design, sum(loss), trace
