"""Parity cut for the supergeo master when N is odd.

Sum over columns of |c| x_c = N. Halving and rounding (a Chvatal-Gomory step) gives
sum floor(|c|/2) x_c <= (N-1)/2, and subtracting twice that from the first,
  sum over odd-size columns x_c >= 1.
Its dual tau >= 0 lowers the reduced cost of every odd column by tau. Bound, valid
for any pi and any tau >= 0: every design has at least one odd column and at most
N/lo columns, so its cost >= sum(pi) + tau + (N/lo) min(0, min reduced cost)."""
import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix
from . import sg, dive, lns, dsr


def master(cols, costs, n):
    rows, cs = [], []
    for j, c in enumerate(cols):
        for g in c: rows.append(g); cs.append(j)
    A = csc_matrix((np.ones(len(rows)), (rows, cs)), shape=(n, len(cols)))
    odd = np.array([-1.0 if len(c) % 2 else 0.0 for c in cols])[None, :]          # -sum odd <= -1
    r = linprog(costs, A_ub=odd, b_ub=[-1.0], A_eq=A, b_eq=np.ones(n), bounds=(0, None), method="highs")
    assert r.status == 0, r.message
    return r.fun, r.eqlin.marginals, -float(r.ineqlin.marginals[0]), r.x


def scan(z, pi, tau, lo, hi, thresh, cap):
    """Columns with reduced cost <= thresh, per size, odd sizes credited tau."""
    out_r, out_s, out_rc, best = [], [], [], np.inf
    for k in range(lo, hi + 1):
        off = tau if k % 2 else 0.0
        r, s, rc, m = sg._scan(z, pi, k, k, thresh + off, cap)
        if m:
            out_r.append(r); out_s.append(s); out_rc.append(rc - off)
        best = min(best, float(sg._min_rc(z, pi, k, k)) - off)
    if not out_r:
        return np.zeros((0, 4), np.int64), np.zeros(0), np.zeros(0), best
    return np.vstack(out_r), np.concatenate(out_s), np.concatenate(out_rc), best


def certify(z0, lo=2, hi=4, add=2000, incumbent=None, mip_limit=600, verbose=True):
    t0 = time.perf_counter()
    n = len(z0); z = np.asarray(z0, float).copy(); scale = 1.0
    assert n % 2 == 1
    cols = sg.initial_pool(z, lo, hi)
    score = {c: dive._sc(z, c) for c in cols}
    while True:
        lp, pi, tau, x = master(cols, np.array([score[c] for c in cols]), n)
        if 0 < lp < 1e-2 * n:
            f = n / lp; z *= np.sqrt(f); scale *= f; score = {c: v * f for c, v in score.items()}
            continue
        rows, s_, rc, mrc = scan(z, pi, tau, lo, hi, -1e-9 * max(1.0, lp), 5_000_000)
        new = 0
        for t in np.argsort(rc)[:add]:
            c = tuple(int(v) for v in rows[t] if v >= 0)
            if c not in score:
                score[c] = float(s_[t]); cols.append(c); new += 1
        if new == 0:
            break
    lb = float(np.sum(pi) + tau + (n / lo) * min(0.0, mrc))
    if verbose: print(f"  LP {lp:.6g}, bound {lb:.6g} (orig {lb/scale:.6g}), tau {tau:.4g} [{time.perf_counter()-t0:.0f}s]", flush=True)
    if incumbent is None:
        o = list(np.argsort(z)); incumbent = [tuple(o[i:i + 2]) for i in range(0, n - 3, 2)] + [tuple(o[-3:])]
        incumbent, _, _ = lns.lns_dp(z, incumbent, seconds=30, verbose=False)
    incumbent = [tuple(sorted(g)) for g in incumbent]
    ub = sum(lns.group_loss(z, g) for g in incumbent)
    thresh = ub - float(np.sum(pi)) - tau - (n / lo - 1) * min(0.0, mrc)
    rows, s_, rc, mrc2 = scan(z, pi, tau, lo, hi, thresh, 3_000_000)
    cand = [tuple(int(v) for v in rows[t] if v >= 0) for t in range(len(rc))]
    if verbose: print(f"  incumbent {ub/scale:.6g} (orig), {len(cand)} columns within the margin [{time.perf_counter()-t0:.0f}s]", flush=True)
    pos = {c: j for j, c in enumerate(cand)}
    for g in incumbent:
        if g not in pos: pos[g] = len(cand); cand.append(g); s_ = np.append(s_, lns.group_loss(z, g))
    sel, val, status = dsr.solve_partition(cand, s_[:len(cand)], n, warm=[pos[g] for g in incumbent], time_limit=mip_limit, return_status=True)
    design = [cand[j] for j in sel] if sel is not None else incumbent
    opt = min(val, ub)
    return dict(n=n, lb=lb / scale, ub=ub / scale, opt=opt / scale, within=len(cand), design=design,
                status=status, certified=(status == "Optimal"), t=time.perf_counter() - t0)
