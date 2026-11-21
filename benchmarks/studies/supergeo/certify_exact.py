"""Supergeo certificate with the bound and the column elimination in exact arithmetic.

Floating point: column generation (any duals give a valid bound), the incumbent,
the window search (widened by the proven margin of mitm_exact), and the final MIP.
Exact: which columns have negative reduced cost (numba error-free kernel), their
minimum (Fraction), the bound sum(pi) + tau + (N/lo) min(0, min rc), the incumbent's
cost, the elimination threshold, and membership in the eliminated set.
The only claim resting on floating point is HiGHS's optimality over the surviving
columns, and it is not needed when the MIP's exact value equals the exact bound."""
import math, time
from fractions import Fraction
import numpy as np
from . import sg, dive, lns, dsr, parity, mitm, mitm_exact as mx, exactnb as ex, verify_nb as vn


def safe_window(z, pi, lo, hi, theta, cap=50_000_000):
    """Every group that can have rc <= theta: mitm_exact's widened window, de-duplicated."""
    hv, ha, hb, hsa, hsb, hw = mitm._halves(z, pi)
    order = np.argsort(hv, kind="stable")
    M = float(np.max(np.abs(z))); P = float(np.max(np.abs(pi)))
    wmax = float(max(pi.max(), 0.0) + max(np.sort(pi)[-2:].sum(), pi.max(), 0.0))
    slack_r2 = 8 * mx.U * (abs(theta) + 6 * P) + 2 * mx.U * abs(wmax); slack_v = 16 * mx.U * 4 * M
    _, k = mx._window_safe(hv, ha, hb, hsa, hsb, hw, order, lo, hi, float(theta), wmax, slack_r2, slack_v, 0)
    if k > cap: raise MemoryError(f"{k} > cap")
    rows, k = mx._window_safe(hv, ha, hb, hsa, hsb, hw, order, lo, hi, float(theta), wmax, slack_r2, slack_v, k)
    n1 = len(z) + 1
    key = (((rows[:, 0] + 1) * n1 + rows[:, 1] + 1) * n1 + rows[:, 2] + 1) * n1 + rows[:, 3] + 1
    return rows[np.unique(key, return_index=True)[1]]


def up(F):
    f = float(F); return f if Fraction(f) >= F else math.nextafter(f, math.inf)


def down(F):
    f = float(F); return f if Fraction(f) <= F else math.nextafter(f, -math.inf)


def exact_rc_tau(z, pi, tau, g):
    return mx.exact_rc(z, pi, g) - (Fraction(tau) if len(g) % 2 else 0)


def _root(z0, lo, hi):
    """Column generation to convergence (floating point; its duals only need to be doubles).
    Returns (z rescaled by a power of 4, the scale, pi, tau)."""
    n = len(z0); odd = n % 2 == 1
    z = np.asarray(z0, float).copy(); scale = 1.0
    cols = sg.initial_pool(z, lo, hi); score = {c: dive._sc(z, c) for c in cols}
    while True:
        if odd: lp, pi, tau, x = parity.master(cols, np.array([score[c] for c in cols]), n)
        else: (lp, pi, x), tau = sg.master_lp(cols, np.array([score[c] for c in cols]), n), 0.0
        if 0 < lp < 1e-2 * n:
            # a power of 4, so z * sqrt(f) = z * 2^k is exact: the certificate is for the original data
            k = int(round(math.log(n / lp, 4))); f = 4.0 ** k
            z = z * 2.0 ** k; scale *= f; score = {c: dive._sc(z, c) for c in score}; continue
        rows, s_, rc, _ = (parity.scan(z, pi, tau, lo, hi, -1e-9 * max(1.0, lp), 5_000_000) if odd
                           else sg._scan(z, pi, lo, hi, -1e-9 * max(1.0, lp), 5_000_000)[:4])
        new = 0
        for t in np.argsort(rc)[:2000]:
            c = tuple(int(v) for v in rows[t] if v >= 0)
            if c not in score: score[c] = float(s_[t]); cols.append(c); new += 1
        if new == 0: break
    return z, scale, pi, max(float(tau), 0.0)


def _exact_bound(z, pi, tau, lo, hi):
    """Every column with rc < 0, exactly, their exact minimum m <= 0, S = sum pi + tau and the exact bound
    LB = S + (N / lo) m. Returns (m, S, LB, the negative columns)."""
    n = len(z)
    W = safe_window(z, pi, lo, hi, 0.0 + tau)                          # superset of rc <= 0 (odd columns credited tau)
    neg = W[ex.members2(z, pi, W, 0.0, tau, True)]
    m = min([exact_rc_tau(z, pi, tau, tuple(int(v) for v in r if v >= 0)) for r in neg], default=Fraction(0))
    m = min(m, Fraction(0))
    S = sum(Fraction(float(p)) for p in pi) + Fraction(tau)
    return m, S, S + Fraction(n, lo) * m, neg


def _survivors(z, pi, tau, lo, hi, S, m, UB):
    """Exact elimination: a column in a design costing <= UB has rc <= UB - S - (N/lo - 1) m. Returns the
    columns that pass, exactly, and the size of the window searched for them."""
    n = len(z)
    th = up(UB - S - (Fraction(n, lo) - 1) * m)
    W2 = safe_window(z, pi, lo, hi, th + tau)
    keep = W2[ex.members2(z, pi, W2, th, tau, False)]
    return [tuple(int(v) for v in r if v >= 0) for r in keep], len(W2)


def certify(z0, lo=2, hi=4, incumbent=None, mip_limit=600, verbose=True):
    t0 = time.perf_counter(); n = len(z0); odd = n % 2 == 1
    # 1. column generation
    z, scale, pi, tau = _root(z0, lo, hi)
    t_cg = time.perf_counter() - t0
    # 2. exact bound
    m, S, LB, neg = _exact_bound(z, pi, tau, lo, hi)
    # 3. exact incumbent cost
    if incumbent is None:
        o = list(np.argsort(z))
        incumbent = ([tuple(o[i:i + 2]) for i in range(0, n - 3, 2)] + [tuple(o[-3:])]) if odd else [tuple(o[i:i + 2]) for i in range(0, n, 2)]
        incumbent, _, _ = lns.lns_dp(z, incumbent, seconds=30, verbose=False)
    incumbent = [tuple(sorted(int(v) for v in g)) for g in incumbent]
    def exact_cost(design):
        return sum(mx.exact_rc(z, np.zeros(n), g) for g in design)       # exact score: rc with zero duals
    UB = exact_cost(incumbent)
    # 4. exact elimination
    cand, nwin = _survivors(z, pi, tau, lo, hi, S, m, UB)
    pos = {c: j for j, c in enumerate(cand)}
    for g in incumbent:
        if g not in pos: pos[g] = len(cand); cand.append(g)
    costs = np.array([dive._sc(z, c) for c in cand])
    # 5. final MIP (floating point), its answer re-costed exactly
    sel, val, status = dsr.solve_partition(cand, costs, n, warm=[pos[g] for g in incumbent], time_limit=mip_limit, return_status=True)
    design = [cand[j] for j in sel] if sel is not None else incumbent
    OPT = min(exact_cost(design), UB)
    exact_tight = OPT == LB
    t_v = time.perf_counter()
    exact_cost_c = {c: mx.exact_rc(z, np.zeros(n), c) for c in cand}
    vstatus, nodes, OPT2, better = vn.verify(n, lo, cand, exact_cost_c, pi, m, tau, OPT)
    t_v = time.perf_counter() - t_v
    proved = vstatus == 0
    cheaper = better is not None
    mip_gap_exact = OPT - OPT2
    if cheaper:
        design, OPT = list(better), OPT2
    exact_tight = OPT == LB
    S_ = Fraction(scale)                                           # Fraction / float would round to a float
    out = dict(n=n, scale=scale, lb=down(LB / S_), opt=float(OPT / S_), lb_exact=LB / S_, opt_exact=OPT / S_,
               negative_columns=len(neg), window=nwin, within=len(cand), mip_status=status,
               proof=("exact: optimum equals the exact bound" if exact_tight else
                      f"exact end to end ({nodes:,} search nodes, {t_v:.1f}s)" if proved else
                      f"exact search stopped (status {vstatus}) after {nodes:,} nodes"),
               highs_excess=float(mip_gap_exact / Fraction(scale)), highs_was_optimal=not cheaper,
               verify_nodes=nodes,
               t_cg=t_cg, t=time.perf_counter() - t0, design=design)
    if verbose:
        print(f"  bound {out['lb']:.10g}  optimum {out['opt']:.12g}  survivors {len(cand)}  HiGHS {status}, "
              f"{'exactly optimal' if out['highs_was_optimal'] else 'above the optimum by ' + format(out['highs_excess'], '.4g')}  -> {out['proof']}  [{out['t']:.0f}s]", flush=True)
    return out


def certify_staged(z0, lo=2, hi=4, incumbent=None, heuristic_seconds=1.0, caps=(300_000, 1_000_000, 3_000_000),
                   verbose=False):
    """The certificate with the elimination redone at every better incumbent and no MIP.

    After the root and the exact bound, each stage runs the exact elimination at the incumbent's exact cost and
    the exact-cover search of :mod:`.verify_nb` over the survivors, from the incumbent, stopping after the stage's
    node cap; its best cover becomes the incumbent, so the next stage's survivors are fewer. A final stage has no
    cap. A stage that completes proves its result optimal: the search covered every exact cover by columns that
    can be in a design costing at most the incumbent it started from. On the 50-state panel, from a 1 s
    heuristic 95 times the optimum, the survivors went 4,629, 941, 549 and the last stage completed after 1.6
    million nodes; the single-threshold certificate searched 3,284 survivors from a 48.9-times design for 39
    million nodes.

    caps: increasing positive node caps for the stages before the final one. Returns the keys of
    :func:`certify` that apply, with `proved`, `proof` and `stages` (one dict per stage)."""
    n = len(z0); odd = n % 2 == 1
    caps = tuple(int(c) for c in caps)
    if any(c < 1 for c in caps) or any(b <= a for a, b in zip(caps, caps[1:])):
        raise ValueError(f"caps must be increasing positive node counts, not {caps}")
    t0 = time.perf_counter()
    z, scale, pi, tau = _root(z0, lo, hi)
    t_cg = time.perf_counter() - t0
    m, S, LB, neg = _exact_bound(z, pi, tau, lo, hi)
    if incumbent is None:
        o = [int(v) for v in np.argsort(z)]
        incumbent = ([tuple(o[i:i + 2]) for i in range(0, n - 3, 2)] + [tuple(o[-3:])]) if odd else [tuple(o[i:i + 2]) for i in range(0, n, 2)]
        if heuristic_seconds > 0:
            incumbent, _, _ = lns.lns_dp(z, incumbent, seconds=heuristic_seconds, verbose=False)
    design = [tuple(sorted(int(v) for v in g)) for g in incumbent]
    zeros = np.zeros(n)
    UB = sum(mx.exact_rc(z, zeros, g) for g in design)
    stages, proved, nodes_all = [], UB == LB, 0
    for cap in caps + (None,):
        if proved:
            break
        cand, nwin = _survivors(z, pi, tau, lo, hi, S, m, UB)
        have = set(cand)
        cand += [g for g in design if g not in have]
        cost = {c: mx.exact_rc(z, zeros, c) for c in cand}
        ts = time.perf_counter()
        status, nodes, val, better = vn.verify(n, lo, cand, cost, pi, m, tau, UB,
                                                node_limit=2_000_000_000 if cap is None else cap)
        nodes_all += nodes
        stages.append(dict(cap=cap, survivors=len(cand), window=nwin, nodes=nodes, status=status,
                           improved=better is not None, t=time.perf_counter() - ts))
        if better is not None:
            design, UB = [tuple(c) for c in better], val
        proved = status == 0 or UB == LB
        if verbose:
            print(f"  stage cap {cap}: {len(cand)} survivors, {nodes:,} nodes, status {status}, "
                  f"incumbent {float(UB / Fraction(scale)):.10g}", flush=True)
    S_ = Fraction(scale)
    return dict(n=n, scale=scale, lb=down(LB / S_), opt=float(UB / S_), lb_exact=LB / S_, opt_exact=UB / S_,
                negative_columns=len(neg), proved=proved, stages=stages, verify_nodes=nodes_all,
                proof=("exact: optimum equals the exact bound" if UB == LB else
                       f"exact end to end, staged ({len(stages)} stages, {nodes_all:,} search nodes)" if proved else
                       f"exact search stopped (status {stages[-1]['status']}) after {nodes_all:,} nodes"),
                t_cg=t_cg, t=time.perf_counter() - t0, design=design)


def verify_over_survivors(z, pi, tau, lo, m, cand, OPT, node_limit=5_000_000):
    """Exact proof that no exact cover by the survivor columns costs less than OPT.
    Depth-first over the lowest uncovered geo; a node is pruned when its exact
    Lagrangian bound, partial + sum_U pi + tau [an odd column still needed] + (|U|/lo) m,
    reaches OPT. Returns (proved, nodes, cheaper design or None)."""
    n = len(z)
    cost = {c: mx.exact_rc(z, np.zeros(n), c) for c in cand}
    PI = [Fraction(float(p)) for p in pi]; T = Fraction(tau); odd_n = n % 2 == 1
    by_geo = [[] for _ in range(n)]
    for c in cand:
        mask = 0
        for g in c: mask |= 1 << g
        by_geo[min(c)].append((mask, cost[c], len(c) % 2 == 1, c))
    for lst in by_geo: lst.sort(key=lambda t: t[1])
    full = (1 << n) - 1; nodes = 0; found = None
    def lb(U, partial, have_odd):
        s = partial + sum(PI[i] for i in range(n) if U >> i & 1) + Fraction(bin(U).count("1"), lo) * m
        if odd_n and not have_odd: s += T
        return s
    stack = [(full, Fraction(0), False, ())]
    while stack:
        U, partial, have_odd, chosen = stack.pop(); nodes += 1
        if nodes > node_limit: return False, nodes, None
        if U == 0:
            if partial < OPT: return False, nodes, chosen        # a cheaper design: the MIP was wrong
            continue
        if lb(U, partial, have_odd) >= OPT: continue
        i = (U & -U).bit_length() - 1
        for mask, cst, is_odd, c in by_geo[i]:
            if mask & U == mask:
                stack.append((U & ~mask, partial + cst, have_odd or is_odd, chosen + (c,)))
    return True, nodes, None
