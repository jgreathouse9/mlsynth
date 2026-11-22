"""Price-and-dive primal heuristic for the supergeo master, and a certify
driver that uses it for the incumbent."""
import time
import numpy as np
from . import sg


def _sc(z, c):
    return float(sg._score(z, np.array(c + (0,) * (4 - len(c)), np.int64), len(c)))


def cg(z, cols, score, lo, hi, add=2000, max_rounds=200):
    """Column generation on z over the given pool. Returns lp, pi, x."""
    n = len(z)
    for _ in range(max_rounds):
        lp, pi, x = sg.master_lp(cols, np.array([score[c] for c in cols]), n)
        rows, s_, rc, m = sg._scan(z, pi, lo, hi, -1e-9 * max(1.0, lp), 5_000_000)
        new = 0
        for t in np.argsort(rc)[:add]:
            c = tuple(int(v) for v in rows[t] if v >= 0)
            if c not in score:
                score[c] = float(s_[t]); cols.append(c); new += 1
        if new == 0:
            return lp, pi, x
    return lp, pi, x


def dive(z, cols, score, lo=2, hi=4, verbose=False):
    """Fix the LP's largest column (and every column at x > 0.5), re-price on
    the free geos, repeat until every geo is covered. Returns the design (a list
    of global-index tuples) and its loss."""
    n = len(z)
    free = np.arange(n)
    design = []
    while len(free):
        loc = {g: i for i, g in enumerate(free)}
        zf = z[free]
        lcols = [tuple(loc[g] for g in c) for c in cols if all(g in loc for g in c)]
        lcols = sorted(set(lcols) | set(sg.initial_pool(zf, lo, hi)))   # windows always tile
        lscore = {c: _sc(zf, c) for c in lcols}
        lp, pi, x = cg(zf, lcols, lscore, lo, hi)
        order = np.argsort(-x)
        take = [lcols[t] for t in order if x[t] > 0.5]
        if len(free) - sum(len(c) for c in take) == 1:
            take = []
        if not take:
            for t in order:                               # largest x that leaves no lone geo
                if len(free) - len(lcols[t]) != 1:
                    take = [lcols[t]]; break
        for c in take:
            g = tuple(int(free[i]) for i in c)
            design.append(g)
            if g not in score:
                score[g] = _sc(z, g); cols.append(g)
        used = {g for c in take for g in (free[i] for i in c)}
        free = np.array([g for g in free if g not in used], dtype=np.int64)
        if verbose:
            print(f"    dive: fixed {len(take)}, free {len(free)}, sub-LP {lp:.4g}")
    return design, float(sum(score[c] for c in design))


def certify3(z, lo=2, hi=4, add=2000, verbose=True, mip_limit=1200, max_cols=1_500_000):
    t0 = time.perf_counter()
    n = len(z); z = np.asarray(z, float).copy(); scale = 1.0
    cols = sg.initial_pool(z, lo, hi)
    score = {c: _sc(z, c) for c in cols}
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
            break
    t_lp = time.perf_counter() - t0
    if verbose: print(f"  LP {lp:.6g} LB {best_lb:.6g} pool {len(cols)} [{t_lp:.0f}s]")
    design, ub = dive(z, cols, score, lo, hi)
    t_dive = time.perf_counter() - t0
    if verbose: print(f"  dive UB {ub:.6g}, ratio UB/LB {ub / best_lb:.4f} [{t_dive:.0f}s]")
    lb_pi, mrc = sg.lagrangian_lb(z, pi, lo, hi)
    thresh = ub - float(np.sum(pi)) - (n / lo - 1) * min(0.0, mrc)
    cnt = sg._count(z, pi, lo, hi, thresh)
    if verbose: print(f"  {cnt} columns within the margin [{time.perf_counter()-t0:.0f}s]")
    out = dict(n=n, lb=best_lb / scale, ub_dive=ub / scale, within=cnt, t_lp=t_lp, t_dive=t_dive)
    if cnt > max_cols:
        out.update(status="too many", t=time.perf_counter() - t0)
        return out
    rows, s_, rc, m = sg._scan(z, pi, lo, hi, thresh, cnt + 1)
    cand = [tuple(int(v) for v in rows[t] if v >= 0) for t in range(m)]
    r2 = sg.master_mip(cand, s_, n, mip_limit)
    out.update(design=[cand[t] for t in np.flatnonzero(r2.x > 0.5)] if r2.x is not None else None,
               opt=r2.fun / scale, status=r2.status, t=time.perf_counter() - t0)
    return out


def restricted_ub(z, pi, lo, hi, target=300_000, mip_limit=600, verbose=True):
    """MIP over the columns of lowest reduced cost (about `target` of them, plus
    the sorted windows so a cover exists). Returns the loss and the design."""
    n = len(z)
    a, b = 1e-9, 1.0
    while sg._count(z, pi, lo, hi, b) < target:     # bracket the threshold
        b *= 10
    for _ in range(12):                               # geometric bisection
        mid = np.sqrt(a * b)
        if sg._count(z, pi, lo, hi, mid) < target: a = mid
        else: b = mid
    rows, s_, rc, m = sg._scan(z, pi, lo, hi, b, int(target * 1.6) + 10)
    cand = {tuple(int(v) for v in rows[t] if v >= 0): float(s_[t]) for t in range(m)}
    for c in sg.initial_pool(z, lo, hi):
        if c not in cand: cand[c] = _sc(z, c)
    cols = list(cand)
    r = sg.master_mip(cols, np.array([cand[c] for c in cols]), n, mip_limit)
    if verbose: print(f"    restricted master: {len(cols)} columns, rc <= {b:.3g}, MIP status {r.status}, loss {r.fun:.6g}")
    design = [cols[t] for t in np.flatnonzero(r.x > 0.5)]
    return float(r.fun), design


def certify4(z, lo=2, hi=4, add=2000, verbose=True, target=300_000, ub_limit=600, mip_limit=1800, max_cols=1_500_000):
    t0 = time.perf_counter()
    n = len(z); z = np.asarray(z, float).copy(); scale = 1.0
    cols = sg.initial_pool(z, lo, hi)
    score = {c: _sc(z, c) for c in cols}
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
            break
    t_lp = time.perf_counter() - t0
    if verbose: print(f"  LP {lp:.6g} LB {best_lb:.6g} pool {len(cols)} [{t_lp:.0f}s]")
    ub, design = restricted_ub(z, pi, lo, hi, target, ub_limit, verbose)
    t_ub = time.perf_counter() - t0
    if verbose: print(f"  UB {ub:.6g}, UB/LB {ub / best_lb:.4f} [{t_ub:.0f}s]")
    lb_pi, mrc = sg.lagrangian_lb(z, pi, lo, hi)
    thresh = ub - float(np.sum(pi)) - (n / lo - 1) * min(0.0, mrc)
    cnt = sg._count(z, pi, lo, hi, thresh)
    if verbose: print(f"  {cnt} columns within the margin (rc <= {thresh:.4g}) [{time.perf_counter()-t0:.0f}s]")
    out = dict(n=n, scale=scale, lb=best_lb / scale, ub=ub / scale, within=cnt, t_lp=t_lp, t_ub=t_ub)
    if cnt > max_cols:
        out.update(status="too many", t=time.perf_counter() - t0)
        return out
    rows, s_, rc, m = sg._scan(z, pi, lo, hi, thresh, cnt + 1)
    cand = [tuple(int(v) for v in rows[t] if v >= 0) for t in range(m)]
    r2 = sg.master_mip(cand, s_, n, mip_limit)
    out.update(design=[cand[t] for t in np.flatnonzero(r2.x > 0.5)] if r2.x is not None else None,
               opt=r2.fun / scale, status=r2.status, mip_gap=getattr(r2, "mip_gap", None),
               bound=getattr(r2, "mip_dual_bound", None), t=time.perf_counter() - t0)
    if out["bound"] is not None: out["bound"] /= scale
    return out
