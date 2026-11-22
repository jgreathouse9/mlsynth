"""Supergeo certificate with the bound and the column elimination in exact arithmetic.

Floating point: column generation (any duals give a valid bound), the incumbent,
the window search (widened by the proven margin of mitm_exact), and the final MIP.
Exact: which columns have negative reduced cost (numba error-free kernel), their
minimum (Fraction), the bound sum(pi) + tau + (N/lo) min(0, min rc), the incumbent's
cost, the elimination threshold, and membership in the eliminated set.
The only claim resting on floating point is HiGHS's optimality over the surviving
columns, and it is not needed when the MIP's exact value equals the exact bound."""
import math, os, time
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
    rows = mx._window_safe_par(hv, ha, hb, hsa, hsb, hw, order, lo, hi, float(theta), wmax, slack_r2, slack_v)
    if len(rows) > cap: raise MemoryError(f"{len(rows)} > cap")
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


def _exact_bound(z, pi, tau, lo, hi, window=None):
    """Every column with rc < 0, exactly, their exact minimum m <= 0, S = sum pi + tau and the exact bound
    LB = S + (N / lo) m. window: a safe window built at a threshold >= 0 (plus tau), used in place of a new one.
    Returns (m, S, LB, the negative columns)."""
    n = len(z)
    W = window if window is not None else safe_window(z, pi, lo, hi, 0.0 + tau)   # superset of rc <= 0
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


def _survivor_rows(z, pi, tau, lo, hi, S, m, UB, window=None):
    """As _survivors, as the window's rows (indices padded with -1) that pass the exact test. window: (rows, thw),
    a safe window built at thw + tau; used when thw is at or above this threshold (a window only grows with its
    threshold), otherwise a new window is built."""
    n = len(z)
    th = up(UB - S - (Fraction(n, lo) - 1) * m)
    if window is not None and th <= window[1]:
        W2 = window[0]
    else:
        W2 = safe_window(z, pi, lo, hi, th + tau)
    return W2[ex.members2(z, pi, W2, th, tau, False)], len(W2)


def _window_threshold(z, pi, tau, lo, hi, S, T):
    """A threshold for one safe window that serves the bound and the first stage. The first stage keeps the groups
    with rc <= T - S - (N / lo - 1) m, m the exact minimum reduced cost, which is known only after the bound's
    window; this computes the threshold with m replaced by the floating-point minimum (sg._min_rc for each group
    size, the parity dual credited to odd sizes) less a cushion. When the exact m is below the guess the first
    stage builds its own window (:func:`_survivor_rows` checks), so the guess sets the time, never the result."""
    n = len(z); pi = np.asarray(pi, float)
    mf = min(float(sg._min_rc(z, pi, k, k)) - (tau if k % 2 else 0.0) for k in range(lo, hi + 1))
    mg = min(0.0, mf)
    mg -= 1e-6 * abs(mg) + 1e-12 * (1.0 + float(np.sum(np.abs(pi))))
    return up(T - S - (Fraction(n, lo) - 1) * Fraction(mg))


def _screen_rows(z, pi, tau, lo, S, m, UB, rows):
    """The rows of an earlier stage's survivors that pass the exact test at the incumbent UB. A survivor is a group
    with exact rc <= th(UB) = UB - S - (N / lo - 1) m, which grows with UB, so at an incumbent no dearer than the
    one the rows were screened at, the survivors are a subset of the rows and the window need not be rebuilt."""
    n = len(z)
    th = up(UB - S - (Fraction(n, lo) - 1) * m)
    return rows[ex.members2(z, pi, rows, th, tau, False)]


def certify_staged(z0, lo=2, hi=4, incumbent=None, heuristic_seconds="auto", caps="auto", keep=0.5, workers="auto",
                   gap=0.0, abs_gap=0.0, max_group_cost=None, verbose=False):
    """The certificate with the elimination redone at every better incumbent and no MIP.

    After the root and the exact bound, each stage runs the exact elimination at the incumbent's exact cost and the
    exact-cover search over the survivors from the incumbent. The first stage screens the safe window; each later
    one screens the last stage's rows (:func:`_screen_rows`), and its `window` is their number. A capped stage searches on floating-point scores
    (:func:`.verify_nb.verify_rows`) up to its node cap and only supplies incumbents: its covers are re-costed
    exactly and it proves nothing. The last stage searches on exact costs with no cap, and its completion is the
    proof: it covered every exact cover by columns that can be in a design costing at most the incumbent it started
    from.

    caps: increasing positive node caps for the capped stages, or "auto": caps of 300,000 times 3^k, and staging
    stops, for the exact stage, once a stage does not improve the incumbent or the next survivors are more than
    half of the last (more capped stages past that point repeat the top of the tree: at 80 markets six of them
    took 688 s against 566 s for three). heuristic_seconds: the pairs-plus-repair budget, "auto" being
    max(1, N / 24) (at 72 markets a 3 s heuristic leaves 62,899 first-stage survivors, a 1 s one 947,976).
    workers: threads for every stage's search (:func:`.verify_nb.verify` and :func:`.verify_nb.verify_rows`), None
    for this thread alone, "auto" for min(4, CPU count).

    gap: a relative optimality gap g in [0, 1). The certificate then proves that no design costs less than
    T = (1 - g) UB, UB the cost of the design returned, so the design is within a factor 1 / (1 - g) of the
    optimum. The run stops once the exact bound LB has UB - LB <= g UB; until then the elimination keeps the groups
    that can be in a design costing at most T, and the exact stage searches for designs cheaper than T. If it finds
    one it finds the optimum, and the proven gap is 0; if not, the proven gap is g. g = 0 is the exact certificate.

    Why a gap: the cost landscape of partitioning problems behaves like a random-cost problem, in which the
    lowest costs of the M feasible solutions are spaced about 1 / (M p(E_1)) apart, p the density of costs at the
    optimum, and no method that examines K << M solutions does better than random search among K (Mertens, "A
    physicist's approach to number partitioning", Theoretical Computer Science 265 (2001) 79-108, section 4; the
    phase transition behind it: Mertens, "Phase transition in the number partitioning problem", Phys. Rev. Lett. 81
    (1998) 4281-4284, doi:10.1103/PhysRevLett.81.4281). A bound proves the exact optimum only once it is finer than
    that spacing, and with M the number of partitions of N markets into groups of 2 to 4 the spacing is far below
    what a bound from the duals resolves. Mallarapu and Sellke ("Strong low degree hardness for the number
    partitioning problem", arXiv:2505.20607, 2025) give evidence that brute force is nearly unimprovable at
    finding such solutions, and Krishnamoorthy, Webb and Moyer ("Lattice-based algorithms for number partitioning
    in the hard phase", Discrete Optimization 9 (2012) 159-171) treat the regime of unique optima. Measured here on
    lognormal panels: the bound is within 4e-8 to 8e-8 of the optimum at 64 to 80 markets, where the exact stage
    takes 3.8 to 80 s on four threads, and within 1.1e-6 of the incumbent at 120 markets, where the exact stage's
    tree is estimated at 7e21 nodes; with g = 1e-6 no search is needed.
    abs_gap: an absolute gap d >= 0 in the units of the objective (squared units of the market totals). With it
    the certificate proves that the design costs at most max(g UB, d) more than the optimum: it stops once
    UB - LB <= max(g UB, d) and otherwise searches below UB - max(g UB, d). A relative gap divides by the optimum,
    and where near-perfect partitions are plentiful (similar-sized markets, the easy phase of Mertens 2001) the
    optimum is near 0 and no useful relative gap is provable, while the design is within its own cost of perfect
    balance. To set d from the noise: if each market's total carries independent noise of standard deviation
    sigma, noise adds |g| sigma^2 to the expected cost of a group g, N sigma^2 to any design's, whatever its
    groups; d = 0.01 N sigma^2 asks for a design within 1% of what noise alone does to the objective.

    max_group_cost: a balance standard t, in the units of the objective: only groups whose imbalance
    (2 Z_A - Z_G)^2 is at most t may be used, so no group of the design is badly matched however good the total.
    Groups above t leave the survivors by the same exact test as the elimination (:func:`.exactnb.members2` with
    zero duals). The bound stays valid (fewer designs can only raise the optimum). A starting design that breaks
    the standard is not used: the search starts from (N // lo) t, above the cost of any design that meets it (at
    most N // lo groups, each at most t), and the gap applies only once a design that meets the standard is known.
    When no partition meets it the certificate proves that and returns no design: balance checks can rule out
    every option, as Chen, Longfils and Remy note of rerandomization ("Trimmed Match Design for randomized paired
    geo experiments", arXiv:2105.07060, section 4.3: "too many balance checks may lead to very few or even no
    options").

    Returns the keys of :func:`certify` that apply, with `proved`, `proof`, `stages` (one dict per stage), `gap`,
    the proven gap `gap_proved` (rounded up), `abs_gap` and the proven absolute gap `abs_gap_proved` (rounded
    up), the proven lower bound `lb_proved` (and `lb_proved_exact`),
    `feasible`, and `standard` (None, or the threshold, whether the design meets it and its largest group cost)."""
    n = len(z0); odd = n % 2 == 1
    if workers == "auto":
        workers = min(4, os.cpu_count() or 1)
    elif workers is not None and (not isinstance(workers, int) or workers < 1):
        raise ValueError(f"workers must be a positive integer, None or 'auto', not {workers!r}")
    auto = isinstance(caps, str)
    if auto and caps != "auto":
        raise ValueError(f"caps must be a tuple of node counts or 'auto', not {caps!r}")
    if not auto:
        caps = tuple(int(c) for c in caps)
        if any(c < 1 for c in caps) or any(b <= a for a, b in zip(caps, caps[1:])):
            raise ValueError(f"caps must be increasing positive node counts, not {caps}")
    if heuristic_seconds == "auto":
        heuristic_seconds = max(1.0, n / 24)
    if isinstance(gap, bool) or not isinstance(gap, (int, float, Fraction)) or not 0 <= gap < 1:
        raise ValueError(f"gap must be a number in [0, 1), not {gap!r}")
    G = Fraction(gap)
    if isinstance(abs_gap, bool) or not isinstance(abs_gap, (int, float)) or not math.isfinite(abs_gap) or abs_gap < 0:
        raise ValueError(f"abs_gap must be a finite number >= 0, not {abs_gap!r}")
    if max_group_cost is not None and (isinstance(max_group_cost, bool) or not isinstance(max_group_cost, (int, float))
                                       or not math.isfinite(max_group_cost) or max_group_cost < 0):
        raise ValueError(f"max_group_cost must be a finite number >= 0 or None, not {max_group_cost!r}")
    t0 = time.perf_counter()
    z, scale, pi, tau = _root(z0, lo, hi)
    t_cg = time.perf_counter() - t0
    if incumbent is None:
        o = [int(v) for v in np.argsort(z)]
        incumbent = ([tuple(o[i:i + 2]) for i in range(0, n - 3, 2)] + [tuple(o[-3:])]) if odd else [tuple(o[i:i + 2]) for i in range(0, n, 2)]
        if heuristic_seconds > 0:
            incumbent, _, _ = lns.lns_dp(z, incumbent, seconds=heuristic_seconds, verbose=False)
    design = [tuple(sorted(int(v) for v in g)) for g in incumbent]
    zeros = np.zeros(n)
    def exact_cost(d):
        return sum(mx.exact_rc(z, zeros, g) for g in d)
    tq = None
    if max_group_cost is not None:
        tq = float(max_group_cost) * scale          # scale is a power of 4: exact
        if Fraction(tq) != Fraction(max_group_cost) * Fraction(scale):  # pragma: no cover - overflow of a power of 4
            raise ValueError(f"max_group_cost {max_group_cost!r} cannot be scaled exactly")
        if any(mx.exact_rc(z, zeros, g) > Fraction(tq) for g in design):
            design = None                           # the start breaks the standard
    if design is not None:
        UB = exact_cost(design)
    else:                                           # above every design that meets the standard
        UB = Fraction(n // lo) * Fraction(tq) * (1 + Fraction(1, 2 ** 30)) + Fraction(1, 2 ** 30)
    D = Fraction(abs_gap) * Fraction(scale)        # in the rescaled units of the costs
    # one window for the bound and the first stage, at a threshold >= 0 and at or above the first stage's
    S0 = sum(Fraction(float(p)) for p in pi) + Fraction(tau)
    T0 = UB - max(G * UB, D) if design is not None else UB
    thw = max(_window_threshold(z, pi, tau, lo, hi, S0, T0), 0.0)
    shared = (safe_window(z, pi, lo, hi, thw + tau), thw)
    m, S, LB, neg = _exact_bound(z, pi, tau, lo, hi, window=shared[0])
    def allowed(ub):                                # the gap in cost units: the larger of g ub and the absolute gap
        return max(G * ub, D)
    def within(ub):                                 # the bound alone proves the gap
        return design is not None and ub - LB <= allowed(ub)
    def standard(rows):
        return rows if tq is None else rows[ex.members2(z, zeros, rows, tq, 0.0, False)]
    stages, proved, nodes_all, k = [], within(UB), 0, 0
    lb_proved = LB
    prev = None
    while not proved:
        T = UB - allowed(UB) if design is not None else UB   # prove that no design costs less than T
        if prev is None:                            # the one window; later stages screen the last stage's rows
            rows, nwin = _survivor_rows(z, pi, tau, lo, hi, S, m, T, window=shared)
            rows = standard(rows)
            shared = None                           # the stages after screen the last stage's rows
        else:
            rows, nwin = _screen_rows(z, pi, tau, lo, S, m, T, prev), len(prev)
        have = {tuple(int(v) for v in r if v >= 0) for r in rows}
        extra = [g for g in design if g not in have] if design is not None else []
        if extra:
            pad = np.full((len(extra), 4), -1, np.int64)
            for j, g in enumerate(extra): pad[j, :len(g)] = g
            rows = np.vstack([rows, pad])
        prev = rows
        last = stages[-1] if stages else None
        if auto:
            capped = last is None or (last["improved"] and len(rows) <= keep * last["survivors"])
            cap = 300_000 * 3 ** k
        else:
            capped = k < len(caps); cap = caps[k] if capped else None
        ts = time.perf_counter()
        if capped:
            sc = sg.scores(z, rows)
            status, nodes, covers = vn.verify_rows(n, lo, rows, sc, pi, float(m), tau, float(UB), node_limit=cap,
                                                   workers=workers)
            better = None
            for r in covers:
                d = [tuple(int(v) for v in rows[j] if v >= 0) for j in r]
                c = exact_cost(d)
                if c < UB and (better is None or c < better[0]):
                    better = (c, d)
            if better is not None:
                UB, design = better
            stages.append(dict(cap=cap, exact=False, workers=workers, survivors=len(rows), window=nwin, nodes=nodes,
                               status=status, improved=better is not None, t=time.perf_counter() - ts))
            proved = within(UB)
        else:
            cand = [tuple(int(v) for v in r if v >= 0) for r in rows]
            cost = {c: mx.exact_rc(z, zeros, c) for c in cand}
            status, nodes, val, found = vn.verify(n, lo, cand, cost, pi, m, tau, T, workers=workers)
            if found is not None:
                design, UB = [tuple(c) for c in found], val
            if status == 0:                         # every design cheaper than T searched: the best of them, or T
                lb_proved = UB if found is not None else T
            stages.append(dict(cap=None, exact=True, workers=workers, survivors=len(rows), window=nwin, nodes=nodes, status=status,
                               improved=found is not None, t=time.perf_counter() - ts))
            proved = status == 0 or within(UB)
            nodes_all += nodes
            if verbose:
                print(f"  exact stage: {len(rows)} survivors, {nodes:,} nodes, status {status}", flush=True)
            break
        nodes_all += nodes; k += 1
        if verbose:
            print(f"  stage cap {cap}: {len(rows)} survivors, {nodes:,} nodes, status {status}, "
                  f"incumbent {float(UB / Fraction(scale)):.10g}", flush=True)
    S_ = Fraction(scale)
    searched = bool(stages) and stages[-1]["exact"] and stages[-1]["status"] == 0
    if design is None:                              # no design meets the standard: proved if the exact stage ended
        return dict(n=n, scale=scale, lb=down(LB / S_), opt=None, lb_exact=LB / S_, opt_exact=None,
                    negative_columns=len(neg), proved=searched, stages=stages, verify_nodes=nodes_all,
                    proof=("no design meets the standard (exact search over the groups within it)" if searched else
                           f"exact search stopped (status {stages[-1]['status']}) before a design met the standard"),
                    gap=float(G), gap_proved=None, abs_gap=abs_gap, abs_gap_proved=None, lb_proved=None,
                    lb_proved_exact=None, feasible=False,
                    standard=dict(max_group_cost=max_group_cost, met=False, max_group=None),
                    t_cg=t_cg, t=time.perf_counter() - t0, design=None)
    if UB == LB:
        proof = "exact: optimum equals the exact bound"
    elif proved and (G or D) and not searched:
        proof = f"within the gap: the bound is within {float(G):g} relative, {abs_gap:g} absolute, of the design"
    elif proved and (G or D):
        proof = (f"within the gap {float(G):g} relative, {abs_gap:g} absolute, staged ({len(stages)} stages, "
                 f"{nodes_all:,} search nodes)")
    elif proved:
        proof = f"exact end to end, staged ({len(stages)} stages, {nodes_all:,} search nodes)"
    else:
        proof = f"exact search stopped (status {stages[-1]['status']}) after {nodes_all:,} nodes"
    if UB == LB:
        lb_proved = LB
    gp = (UB - lb_proved) / UB if UB > 0 else Fraction(0)
    std = None
    if tq is not None:
        worst = max(mx.exact_rc(z, zeros, tuple(g)) for g in design)
        std = dict(max_group_cost=max_group_cost, met=worst <= Fraction(tq), max_group=float(worst / S_))
    return dict(n=n, scale=scale, lb=down(LB / S_), opt=float(UB / S_), lb_exact=LB / S_, opt_exact=UB / S_,
                negative_columns=len(neg), proved=proved, stages=stages, verify_nodes=nodes_all, proof=proof,
                gap=float(G), gap_proved=up(gp), abs_gap=abs_gap, abs_gap_proved=up((UB - lb_proved) / S_),
                lb_proved=down(lb_proved / S_), lb_proved_exact=lb_proved / S_,
                feasible=True, standard=std, t_cg=t_cg, t=time.perf_counter() - t0, design=design)


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
