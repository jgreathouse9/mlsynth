"""An exact-cover tree that bounds every node afresh, by the volume algorithm.

The design is an exact cover of an arm's units by groups (:mod:`.columns`), the cheapest with at least ``k`` groups.
The tree enumerates covers as :mod:`.cover` does: depth first, branching on the uncovered unit with the fewest groups
that still fit, trying those groups in order of cost (Knuth's Algorithm X). Where :mod:`.cover` bounds every node
with the root's duals, which go stale with depth (on a 24-geo Q = 4 seasonal arm the root-dual bound stays 22% below
the optimum at depth 1, where the node's own LP is 0.75% below it), this tree bounds each node by its own problem:
the groups left (those disjoint from the groups chosen and not fixed out), the units left, and the cardinality row
:math:`\\sum_G x_G \\ge k - d` at depth :math:`d`.

Node bound. For multipliers :math:`\\pi` on the units left and :math:`\\sigma \\ge 0` on the cardinality row, the
Lagrangian bound of the node's set-partitioning problem with :math:`x \\in [0, 1]` is

.. math::

   L(\\pi, \\sigma) = \\sum_{i \\in U} \\pi_i + \\sigma (k - d)^+ + \\sum_G \\min(\\bar c_G, 0), \\qquad
   \\bar c_G = c_G - \\sigma - \\sum_{i \\in G} \\pi_i,

a lower bound on every completion for any such multipliers (Geoffrion, "Lagrangean relaxation for integer
programming", Math. Programming Study 2 (1974) 82-114). The multipliers come from the volume algorithm (Barahona and
Anbil, "The volume algorithm: producing primal solutions with a subgradient method", Math. Programming 87 (2000)
385-399, p. 391), started from the parent's multipliers (the root LP's at the root): a direction from the averaged
primal :math:`v = 1 - A\\bar x`, a move only when it raises the bound, and the average updated as
:math:`\\bar x \\leftarrow \\alpha x + (1 - \\alpha) \\bar x`. The step is their equation (7) (p. 388),
:math:`\\lambda (T - \\bar z) / \\lVert v \\rVert^2` with :math:`T` an upper bound on the optimum, here the node's
pruning threshold, the incumbent less the chosen groups' cost; the cardinality row's multiplier is held at zero or
above, as their p. 388 does for an inequality row; and the steps stop as soon as the bound reaches :math:`T`. The
remaining rules are as the same authors state them in "On some difficult linear programs coming from set
partitioning", Discrete Applied Mathematics 118 (2002) 3-11, p. 8: :math:`\\lambda` raised by 1.1 (to at most 2)
after a green step and cut by 0.66 after 20 red steps in a row (unless below 0.0005); :math:`\\alpha` the minimiser
of the unit rows' residual over :math:`[u/10, u]`, :math:`u` starting at 0.1 and halved every 100 steps in which
:math:`\\bar z` rose less than 1%; and, with no incumbent, a target raised to :math:`1.05 \\bar z` each time
:math:`\\bar z` passes :math:`0.95 T`. At most ``iterations`` steps are taken per node. Measured on seasonal arms,
most nodes are pruned at their parent's multipliers before any step.

Pricing. A volume step prices every open group at a trial point :math:`(\\pi, s)`, and pricing is nearly all of a
proof's time. A group enters the step only when its reduced cost there is negative, and most never are. After a pass
over every open group at :math:`(\\pi^f, s^f)`, a group's reduced cost moves by at most
:math:`|G| \\max_i |\\pi_i - \\pi^f_i| + |s - s^f|`, since each of its at most :math:`|G|` multipliers and the
cardinality row's enter it once. Each computed reduced cost is within :math:`\\gamma_{|G|+2}` times its terms'
magnitudes of the exact one (Higham 2002, Lemma 3.1, as for the bound). So a group whose computed reduced cost at
the pass is at least :math:`\\delta`, where :math:`\\delta` exceeds the movement plus both evaluations' rounding
terms, has a computed reduced cost of at least zero at the trial point: it adds nothing to :math:`z` and nothing
to the residual, exactly as if it had been priced. Between passes only the groups below :math:`\\delta` are priced,
in the same order, so every sum is formed from the same terms in the same order and the iterates are those of
pricing every group, bit for bit. A pass is made when the movement reaches :math:`\\delta`, which is then set to
``PRICING_KAPPA`` times the last step's move. Proving from the optimum on seasonal arms, single-threaded, pricing
every group at every step against this pricing, with the same nodes and the same design: 15.9 s against 8.5 s on a
40-geo Q = 3 arm, 0.37 against 0.13 s at 36 geos and Q = 3, 0.53 against 0.28 s at 32 geos and Q = 3, 0.62 against
0.43 s at 50 geos and Q = 2; and 35.8 s against 20.7 s for the same 300 thousand nodes from the start design on a
120-geo Q = 2 arm. Pricing had been 88% and 90% of the 40- and 120-geo trees' time.

Reduced-cost fixing at every node. A group with :math:`\\bar c_G > 0` enters a completion only by raising its cost
to at least :math:`L + \\bar c_G`, so where that reaches the incumbent the group is dropped for the node's subtree
(Nemhauser and Wolsey, "Integer and Combinatorial Optimization", Wiley, 1988, section II.1.4): the root's elimination
of :mod:`.columns`, repeated at every node with the node's multipliers.

Rigour. Groups are bounded on the lower bounds of their exact costs and designs kept on their upper bounds
(:class:`.columns.ColumnPool`). Each reduced cost is a sum of at most ``2Q + 2`` terms, so its computed value lies
within :math:`\\gamma_{2Q+2}` times the sum of the terms' magnitudes of the exact one; the bound subtracts that per
group and :math:`\\gamma_m` of the absolute sum for the outer sum of :math:`m` terms (Higham, "Accuracy and
Stability of Numerical Algorithms", 2nd ed., SIAM, 2002, Lemma 3.1; Rump, Acta Numerica 19 (2010) 287-449, Theorem
2.1), with :math:`\\gamma_k` inflated past its own rounding as :mod:`.cover` does. A node is pruned, and a group fixed
out, only when that lower end reaches the incumbent. ``test_pangeo_vtree.py`` checks the bound against exact
rational arithmetic on inputs with levels to :math:`10^{15}` and differences to :math:`10^{-9}` of them. A tree that
ends under its node cap has seen every design over the groups searched that could beat the incumbent, so the least of
its upper bound and the lowest design lower bound it reached is a rigorous lower bound on the optimum over those
groups.

Measured from the optimum (one arm, 42 estimation periods, single-threaded), against the gated cover search: 24 s
against 573 s on a 40-geo Q = 3 arm (41 thousand nodes against 48 million), 0.98 s against 12.6 s at 50 geos and
Q = 2, 0.50 s against 2.9 s at 36 geos and Q = 3. On the 40-geo arm, without the fixing the tree placed 1.2 million
nodes in 105 s. Units are held in 64-bit words, any number of them.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ...exceptions import MlsynthConfigError
from . import cover
from .cover import CoverSearch, P_DEPTH, P_INCUMBENT, P_NODES, P_ROOT_DONE, P_ROOT_TOTAL, P_TOTAL

try:  # numba is the optional ``fast`` extra, as for :mod:`.cover`
    from numba import njit
    _NUMBA = True
except ImportError:  # pragma: no cover - the suite runs with numba installed
    _NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

#: Compiled code is cached on disk only under this module name (see :mod:`.columns`).
CANONICAL_NAME = "mlsynth.utils.pangeo_helpers.vtree"
_CACHE = __name__ == CANONICAL_NAME

#: Volume steps per node at most. Proving from the optimum on seasonal arms, single-threaded: 15.7, 17.5 and 21.6 s
#: at 20, 50 and 150 steps on a 40-geo Q = 3 arm (165, 97 and 70 thousand nodes), 0.83, 0.71 and 0.82 s on a
#: 50-geo Q = 2 arm, 0.18, 0.25 and 0.33 s on a 36-geo Q = 3 arm. The step rules of COIN-OR Vol (the step factor
#: doubled after a green step, raised by 1.1 after two yellow ones, cut by 0.67 after ten red ones; alpha from 0.01,
#: halved every 80 steps; a moving target) placed 2 to 20 times the nodes on the same arms at every step count
#: (40-geo Q = 3: 37 to 124 s), so the tree keeps the papers' rules.
ITERATIONS = 50

#: Pricing makes a pass over every open group when the multipliers have moved, since the last such pass, by as much as
#: ``PRICING_KAPPA`` times the last step's move; between passes it prices only the groups that can be negative.
#: Proving from the optimum on seasonal arms, single-threaded, at 0 (every group at every step), 2, 4, 8 and 16:
#: 16.8, 9.0, 7.6, 8.2 and 9.2 s on a 40-geo Q = 3 arm; 0.55, 0.26, 0.24, 0.26 and 0.33 s at 32 geos and Q = 3;
#: 0.63, 0.47, 0.46, 0.47 and 0.51 s at 50 geos and Q = 2; 31.9, 21.3, 21.3, 21.6 and 21.3 s for 300 thousand
#: nodes from the start design at 120 geos and Q = 2. Any value gives the same iterates, bit for bit.
PRICING_KAPPA = 4.0

#: The kernel writes the progress array (:func:`.cover.new_progress`) every this many nodes.
PROGRESS_EVERY = 1024

_U = 2.0 ** -53
_ETA = 2.0 ** -1074


@njit(cache=_CACHE)
def _gam(m):  # pragma: no cover - compiled; checked through _bound_low
    """gamma_m = m u / (1 - m u), inflated past its own rounding (:func:`.cover._gamma`)."""
    return m * _U / (1.0 - m * _U) * (1.0 + 4.0 * _U)


def _words(n: int) -> int:
    """The 64-bit words that hold n units' bits."""
    return max(1, -(-int(n) // 64))


def _mask_of(units, words: int) -> np.ndarray:
    """The bit mask of ``units`` in ``words`` 64-bit words."""
    out = np.zeros(words, dtype=np.uint64)
    for u in np.asarray(units, dtype=np.int64):
        out[u >> 6] |= np.uint64(1) << np.uint64(u & 63)
    return out


def _group_masks(members: np.ndarray, words: int) -> np.ndarray:
    """``(C, words)`` bit masks of the groups' units (rows of ``members``, padded with -1)."""
    out = np.zeros((members.shape[0], words), dtype=np.uint64)
    for w in range(members.shape[1]):
        col = members[:, w]
        ok = np.flatnonzero(col >= 0)
        u = col[ok]
        np.bitwise_or.at(out, (ok, u >> 6), np.left_shift(np.uint64(1), (u & 63).astype(np.uint64)))
    return out


@njit(cache=_CACHE)
def _has(m, u):  # pragma: no cover - compiled
    return (m[u >> 6] >> np.uint64(u & 63)) & np.uint64(1) != np.uint64(0)


@njit(cache=_CACHE)
def _price(mem, clo, opn, o0, no, pi, s, uncm, n, kk, xt):  # pragma: no cover - compiled
    """z(pi, s) over the open groups ``opn[o0:o0+no]`` and the box minimiser xt (1 where the reduced cost is
    negative)."""
    z = s * kk
    for u in range(n):
        if _has(uncm, u):
            z += pi[u]
    W = mem.shape[1]
    for i in range(no):
        j = opn[o0 + i]
        rc = clo[j] - s
        for w in range(W):
            u = mem[j, w]
            if u >= 0:
                rc -= pi[u]
        if rc < 0.0:
            z += rc
            xt[i] = 1.0
        else:
            xt[i] = 0.0
    return z


@njit(cache=_CACHE)
def _resid(mem, opn, o0, no, x, uncm, n, kk, r):  # pragma: no cover - compiled
    """r = 1 - A x on the units left (0 elsewhere); returns kk - sum(x)."""
    for u in range(n):
        r[u] = 1.0 if _has(uncm, u) else 0.0
    tot = 0.0
    W = mem.shape[1]
    for i in range(no):
        xi = x[i]
        if xi != 0.0:
            tot += xi
            j = opn[o0 + i]
            for w in range(W):
                u = mem[j, w]
                if u >= 0:
                    r[u] -= xi
    return kk - tot


@njit(cache=_CACHE)
def _price_full(mem, clo, opn, o0, no, pi, s, uncm, n, kk, xt, r, rcv, act, delta, pf):  # pragma: no cover
    """:func:`_price` and :func:`_resid` of its minimiser in one pass over every open group: returns
    ``(z, kk - sum xt, count, cmax, pimax)``, with ``r = 1 - A xt`` on the units left, each group's computed reduced
    cost in ``rcv``, in ``act[:count]`` in order the positions of the groups whose reduced cost is below ``delta``,
    the point priced at copied into ``pf``, and the largest cost and multiplier magnitudes, which the rounding
    margin of :func:`_may_skip` reads."""
    z = s * kk
    pimax = 0.0
    for u in range(n):
        pf[u] = pi[u]
        pimax = max(pimax, abs(pi[u]))
        if _has(uncm, u):
            z += pi[u]
            r[u] = 1.0
        else:
            r[u] = 0.0
    W = mem.shape[1]
    tot = 0.0
    na = 0
    cmax = 0.0
    for i in range(no):
        j = opn[o0 + i]
        cmax = max(cmax, abs(clo[j]))
        rc = clo[j] - s
        for w in range(W):
            u = mem[j, w]
            if u >= 0:
                rc -= pi[u]
        rcv[i] = rc
        if rc < delta:
            act[na] = i
            na += 1
        if rc < 0.0:
            z += rc
            xt[i] = 1.0
            tot += 1.0
            for w in range(W):
                u = mem[j, w]
                if u >= 0:
                    r[u] -= 1.0
        else:
            xt[i] = 0.0
    return z, kk - tot, na, cmax, pimax


@njit(cache=_CACHE)
def _price_act(mem, clo, opn, o0, na, pi, s, uncm, n, kk, xt, r, act):  # pragma: no cover - compiled
    """:func:`_price_full` over the groups at positions ``act[:na]`` only, the others' entries of ``xt`` left as
    they are: ``(z, kk - sum xt)`` and ``r``."""
    z = s * kk
    for u in range(n):
        if _has(uncm, u):
            z += pi[u]
            r[u] = 1.0
        else:
            r[u] = 0.0
    W = mem.shape[1]
    tot = 0.0
    for q in range(na):
        i = act[q]
        j = opn[o0 + i]
        rc = clo[j] - s
        for w in range(W):
            u = mem[j, w]
            if u >= 0:
                rc -= pi[u]
        if rc < 0.0:
            z += rc
            xt[i] = 1.0
            tot += 1.0
            for w in range(W):
                u = mem[j, w]
                if u >= 0:
                    r[u] -= 1.0
        else:
            xt[i] = 0.0
    return z, kk - tot


@njit(cache=_CACHE)
def _may_skip(W, mf, ms, delta, cmax, pfmax, sf, ptmax, st):  # pragma: no cover - compiled; tested directly
    """Whether a group of at most ``W`` units whose computed reduced cost at (pf, sf) is at least ``delta`` has a
    computed reduced cost of at least zero at (pt, st), given ``mf = max |pt - pf|``, ``ms = |st - sf|``, the
    largest cost magnitude ``cmax`` and the largest multiplier magnitudes ``pfmax`` and ``ptmax``. Each reduced cost
    is a sum of at most W + 2 terms, so its computed value is within gamma_{W+2} of its terms' magnitudes of the
    exact one; the exact one moves by at most ``W mf + ms``."""
    g = _gam(W + 2)
    err = g * (cmax + sf + W * pfmax) + g * (cmax + st + W * ptmax)
    return W * mf + ms + err + _ETA < delta


@njit(cache=_CACHE)
def _volume(mem, clo, opn, o0, no, uncm, n, kk, pi0, s0, target, T,  # pragma: no cover - compiled
            pib, xt, xbar, v, r0, r1, pit, rcv, act, pf, kappa):
    """The volume algorithm over the open groups from (pi0, s0), at most T steps; the best multipliers go to pib.
    Returns (sigma, steps, groups priced). Stops once the bound reaches ``target``; an infinite target is replaced
    by the paper's moving one. Pricing skips the groups that cannot be negative (module docstring); with ``kappa``
    0 it prices every group at every step, and the iterates are the same for every ``kappa``."""
    W = mem.shape[1]
    for u in range(n):
        pib[u] = pi0[u] if _has(uncm, u) else 0.0
    sb = s0 if kk > 0 else 0.0
    zb = _price(mem, clo, opn, o0, no, pib, sb, uncm, n, kk, xt)
    priced = no
    fixed_target = np.isfinite(target)
    if fixed_target and zb >= target:
        return sb, 0, priced
    tg = target if fixed_target else zb + 0.05 * abs(zb) + 1e-12
    for i in range(no):
        xbar[i] = xt[i]
    lam, uu, reds = 0.1, 0.1, 0
    z_check = zb
    steps = 0
    delta = 0.0                    # no pass yet: the first trial point gets one
    sf, pfmax, cmax = 0.0, 0.0, 0.0
    na = 0
    sprev = sb
    for u in range(n):
        pf[u] = pib[u]
        r0[u] = pib[u]             # the previous trial point, for the step's move
    for t in range(1, T + 1):
        steps = t
        vs = _resid(mem, opn, o0, no, xbar, uncm, n, kk, v)
        if kk <= 0 or (sb <= 0.0 and vs < 0.0):
            vs = 0.0
        norm = vs * vs
        for u in range(n):
            norm += v[u] * v[u]
        if norm <= 0.0:
            break
        st = lam * (tg - zb) / norm
        for u in range(n):
            pit[u] = pib[u] + st * v[u] if _has(uncm, u) else 0.0
        sgt = max(sb + st * vs, 0.0) if kk > 0 else 0.0
        mv, mf, ptmax = 0.0, 0.0, 0.0
        for u in range(n):
            mv = max(mv, abs(pit[u] - r0[u]))
            mf = max(mf, abs(pit[u] - pf[u]))
            ptmax = max(ptmax, abs(pit[u]))
            r0[u] = pit[u]
        move = W * mv + abs(sgt - sprev)
        sprev = sgt
        if _may_skip(W, mf, abs(sgt - sf), delta, cmax, pfmax, sf, ptmax, sgt):
            zt, vt = _price_act(mem, clo, opn, o0, na, pit, sgt, uncm, n, kk, xt, r1, act)
            priced += na
        else:
            delta = kappa * move
            zt, vt, na, cmax, pfmax = _price_full(mem, clo, opn, o0, no, pit, sgt, uncm, n, kk, xt, r1, rcv, act,
                                                  delta, pf)
            priced += no
            sf = sgt
        num, den = 0.0, 0.0
        for u in range(n):
            dd = r1[u] - v[u]
            num -= v[u] * dd
            den += dd * dd
        a = num / den if den > 0.0 else uu
        a = min(max(a, uu / 10.0), uu)
        for i in range(no):
            xbar[i] = a * xt[i] + (1.0 - a) * xbar[i]
        if zt > zb:
            d = 0.0
            for u in range(n):
                d += v[u] * r1[u]
            if d >= 0.0:
                lam = min(lam * 1.1, 2.0)
            for u in range(n):
                pib[u] = pit[u]
            sb, zb, reds = sgt, zt, 0
            if not fixed_target and zb > 0.95 * tg:
                tg = zb + 0.05 * abs(zb) + 1e-12
        else:
            reds += 1
            if reds >= 20 and lam >= 0.0005:
                lam *= 0.66
                reds = 0
        if t % 100 == 0:
            if zb < z_check + 0.01 * abs(z_check):
                uu = max(uu / 2.0, 1e-5)
            z_check = zb
        if fixed_target and zb >= target:
            break
    return sb, steps, priced


@njit(cache=_CACHE)
def _bound_low(mem, clo, opn, no, pi, s, uncm, n, kk, rcl):  # pragma: no cover - compiled; tested directly
    """A lower bound on L(pi, s) over the groups ``opn[:no]`` in exact arithmetic, and in ``rcl`` the lower end of
    each group's reduced cost."""
    return _bound_at(mem, clo, opn, 0, no, pi, s, uncm, n, kk, rcl)


@njit(cache=_CACHE)
def _bound_at(mem, clo, opn, o0, no, pi, s, uncm, n, kk, rcl):  # pragma: no cover - compiled
    W = mem.shape[1]
    gw = _gam(W + 6)
    s1, a1 = 0.0, 0.0
    for u in range(n):
        if _has(uncm, u):
            s1 += pi[u]
            a1 += abs(pi[u])
    s2, a2 = 0.0, 0.0
    for i in range(no):
        j = opn[o0 + i]
        rc = clo[j] - s
        ab = abs(clo[j]) + s
        for w in range(W):
            u = mem[j, w]
            if u >= 0:
                rc -= pi[u]
                ab += abs(pi[u])
        lo = rc - (gw * ab + _ETA)
        rcl[i] = lo
        if lo < 0.0:
            s2 += lo
            a2 -= lo
    sk = s * kk
    L = s1 + sk + s2
    slack = _gam(n + no + 6) * (a1 + abs(sk) + a2) + _gam(4) * abs(L) + _ETA
    return L - slack


@njit(cache=_CACHE)
def _low(x, ax, m):  # pragma: no cover - compiled
    """A lower bound on a sum of m terms computed as x, with ax the sum of their magnitudes."""
    return x - (_gam(m + 2) * ax + _gam(2) * abs(x) + _ETA)


@njit(cache=_CACHE, nogil=True)
def _tree(mem, gm, clo, cup, n, k, pi0, s0, best0, T,  # pragma: no cover - compiled; checked through search
          fix, node_limit, prog, every, kappa):
    """Depth-first search with a volume bound at every node. Returns (status, best, design positions, nodes,
    lowest leaf lower bound)."""
    C, W = mem.shape
    Wd = gm.shape[1]
    D = n // 2 + 2
    base = prog[P_TOTAL]
    cap = max(4 * C, 1024)
    buf = np.empty(cap, dtype=np.int64)       # the open groups of every depth, depth d's from off[d]
    cbuf = np.empty(cap, dtype=np.int64)      # the branching candidates of every depth, depth d's from coff[d]
    off = np.zeros(D + 1, dtype=np.int64)
    coff = np.zeros(D + 1, dtype=np.int64)
    nopen = np.zeros(D + 1, dtype=np.int64)
    ncand = np.zeros(D + 1, dtype=np.int64)
    cpos = np.zeros(D + 1, dtype=np.int64)
    pis = np.zeros((D + 1, n))
    sig = np.zeros(D + 1)
    unc = np.zeros((D + 1, Wd), dtype=np.uint64)
    flo = np.zeros(D + 1)
    fab = np.zeros(D + 1)
    chosen = np.empty(D + 1, dtype=np.int64)
    design = np.empty(D + 1, dtype=np.int64)
    dsize = 0
    xt = np.empty(C)
    xbar = np.empty(C)
    rcl = np.empty(C)
    keyb = np.empty(C)
    v = np.empty(n)
    r0 = np.empty(n)
    r1 = np.empty(n)
    pit = np.empty(n)
    pib = np.empty(n)
    pin = np.empty(n)
    cnt = np.zeros(n, dtype=np.int64)
    rcv = np.empty(C)
    act = np.empty(C, dtype=np.int64)
    pf = np.empty(n)
    best = best0
    lowest = np.inf
    nodes = 0
    status = 0
    for j in range(C):
        buf[j] = j
    nopen[0] = C
    for u in range(n):
        unc[0, u >> 6] |= np.uint64(1) << np.uint64(u & 63)
    d = 0
    enter = True
    while True:
        if not enter:
            while d >= 0 and cpos[d] >= ncand[d]:
                d -= 1
            if d < 0:
                break
            if nodes >= node_limit:
                status = 1
                break
            a = cbuf[coff[d] + cpos[d]]
            cpos[d] += 1
            nodes += 1
            if nodes % every == 0:
                prog[P_NODES] = nodes
                prog[P_TOTAL] = base + nodes
                prog[P_INCUMBENT] = best
                prog[P_DEPTH] = d
                prog[P_ROOT_DONE] = cpos[0]
                prog[P_ROOT_TOTAL] = ncand[0]
            chosen[d] = a
            left = False
            for w in range(Wd):
                if unc[d, w] & ~gm[a, w] != np.uint64(0):
                    left = True
            if not left:                       # a design
                if d + 1 >= k:
                    us, ua, ls, la = 0.0, 0.0, 0.0, 0.0
                    for e in range(d + 1):
                        us += cup[chosen[e]]
                        ua += abs(cup[chosen[e]])
                        ls += clo[chosen[e]]
                        la += abs(clo[chosen[e]])
                    lw = _low(ls, la, d + 1)
                    if lw < lowest:
                        lowest = lw
                    up = us + (_gam(d + 3) * ua + _gam(2) * abs(us) + _ETA)
                    if up < best:
                        best = up
                        dsize = d + 1
                        for e in range(d + 1):
                            design[e] = chosen[e]
                continue
            need = off[d] + nopen[d] + nopen[d]
            if need > buf.size:                # grow the open-group stack for the child's list
                nb = np.empty(2 * need, dtype=np.int64)
                nb[:buf.size] = buf
                buf = nb
            o1 = off[d] + nopen[d]
            m = 0
            for i in range(nopen[d]):
                j = buf[off[d] + i]
                clash = False
                for w in range(Wd):
                    if gm[j, w] & gm[a, w] != np.uint64(0):
                        clash = True
                        break
                if not clash:
                    buf[o1 + m] = j
                    m += 1
            if m == 0:
                continue
            off[d + 1] = o1
            nopen[d + 1] = m
            for w in range(Wd):
                unc[d + 1, w] = unc[d, w] & ~gm[a, w]
            flo[d + 1] = flo[d] + clo[a]
            fab[d + 1] = fab[d] + abs(clo[a])
            d += 1
        enter = False
        # bound the node at depth d
        kk = k - d if k > d else 0
        if d == 0:
            for u in range(n):
                pin[u] = pi0[u]
            s_in = s0
        else:
            for u in range(n):
                pin[u] = pis[d - 1, u]
            s_in = sig[d - 1]
        f_lo = _low(flo[d], fab[d], d)
        target = best - f_lo
        s_out, _, _ = _volume(mem, clo, buf, off[d], nopen[d], unc[d], n, kk, pin, s_in, target, T,
                              pib, xt, xbar, v, r0, r1, pit, rcv, act, pf, kappa)
        L = _bound_at(mem, clo, buf, off[d], nopen[d], pib, s_out, unc[d], n, kk, rcl)
        tot = L + f_lo
        tot = tot - (_gam(2) * abs(tot) + _ETA)
        ncand[d] = 0
        cpos[d] = 0
        if tot >= best:
            if d == 0:
                break
            d -= 1
            continue
        for u in range(n):
            pis[d, u] = pib[u]
        sig[d] = s_out
        if fix:                                # drop the groups whose reduced cost rules them out below this node
            m = 0
            for i in range(nopen[d]):
                lo = rcl[i]
                if lo > 0.0:
                    tj = L + lo + f_lo
                    tj = tj - (_gam(3) * (abs(L) + lo + abs(f_lo)) + _ETA)
                    if tj >= best:
                        continue
                buf[off[d] + m] = buf[off[d] + i]
                m += 1
            nopen[d] = m
        for u in range(n):
            cnt[u] = 0
        for i in range(nopen[d]):
            j = buf[off[d] + i]
            for w in range(W):
                u = mem[j, w]
                if u >= 0:
                    cnt[u] += 1
        g0, cmin = -1, C + 1
        for u in range(n):
            if _has(unc[d], u) and cnt[u] < cmin:
                cmin, g0 = cnt[u], u
        if cmin == 0:                          # a unit nothing fits
            if d == 0:
                break
            d -= 1
            continue
        coff[d] = coff[d - 1] + ncand[d - 1] if d > 0 else 0   # above the parent's candidates, still in use
        need = coff[d] + nopen[d]
        if need > cbuf.size:                   # grow the candidate stack
            nc = np.empty(2 * need, dtype=np.int64)
            nc[:cbuf.size] = cbuf
            cbuf = nc
        m = 0
        for i in range(nopen[d]):
            j = buf[off[d] + i]
            if _has(gm[j], g0):
                keyb[m] = clo[j]
                cbuf[coff[d] + m] = j
                m += 1
        order = np.argsort(keyb[:m], kind="mergesort")
        tmp = cbuf[coff[d]:coff[d] + m].copy()
        for i in range(m):
            cbuf[coff[d] + i] = tmp[order[i]]
        ncand[d] = m
    return status, best, design[:dsize].copy(), nodes, lowest


def search(pool, root, keep, k: int, best_up: float, node_limit: int, *, iterations: Optional[int] = None,
           fix: bool = True, progress: Optional[np.ndarray] = None, kappa: Optional[float] = None) -> CoverSearch:
    """Search the exact covers of the arm by the pool rows ``keep`` for one with at least ``k`` groups whose upper
    bound is below ``best_up``, bounding every node by the volume algorithm from ``root``'s duals (``pi``,
    ``sigma``), at most ``iterations`` steps a node (``ITERATIONS`` when None), dropping groups by reduced cost at
    every node unless ``fix`` is False, and placing at most ``node_limit`` groups. Returns a
    :class:`.cover.CoverSearch` with ``stage="volume"``. ``progress`` (:func:`.cover.new_progress`) receives the
    tree's counts while it runs and when it ends; the result does not depend on it. ``kappa`` (``PRICING_KAPPA``
    when None) sets how far the multipliers move between passes over every open group; it changes the time taken
    and nothing else, and 0 prices every group at every step."""
    if int(node_limit) < 0:
        raise MlsynthConfigError(f"node_limit must be non-negative; got {node_limit}.")
    steps = ITERATIONS if iterations is None else int(iterations)
    if steps < 0:
        raise MlsynthConfigError(f"iterations must be non-negative; got {iterations}.")
    kap = float(PRICING_KAPPA if kappa is None else kappa)
    if not (np.isfinite(kap) and kap >= 0.0):
        raise MlsynthConfigError(f"kappa must be finite and non-negative; got {kappa}.")
    cover._check_progress(progress)
    prog = cover.new_progress() if progress is None else progress
    base = float(prog[cover.P_TOTAL])
    prog[cover.P_SEARCHES] += 1.0
    prog[cover.P_DONE] = 0.0
    prog[cover.P_NODES] = 0.0
    prog[cover.P_INCUMBENT] = float(best_up)
    prog[cover.P_LOWEST] = np.inf
    prog[cover.P_DEPTH] = prog[cover.P_ROOT_DONE] = prog[cover.P_ROOT_TOTAL] = 0.0
    keep = np.asarray(keep, dtype=np.int64)
    if keep.size == 0:
        res = CoverSearch("proved", keep, float(best_up), float(best_up), 0, stage="volume")
    else:
        n = int(pool.unit_indices.size)
        mem = np.ascontiguousarray(pool.members[keep], dtype=np.int64)
        st, upper, pos, nodes, lowest = _tree(
            mem, _group_masks(mem, _words(n)), np.ascontiguousarray(pool.cost_lo[keep], dtype=float),
            np.ascontiguousarray(pool.cost_up[keep], dtype=float), n, int(k),
            np.ascontiguousarray(root.pi, dtype=float), float(max(root.sigma, 0.0)), float(best_up), steps,
            bool(fix), int(node_limit), prog, max(int(PROGRESS_EVERY), 1), kap)
        res = CoverSearch("node_limit" if st else "proved", keep[pos], float(upper),
                          float(min(upper, lowest)), int(nodes), stage="volume")
    prog[cover.P_NODES] = float(res.nodes)
    prog[cover.P_TOTAL] = base + float(res.nodes)
    prog[cover.P_INCUMBENT] = res.upper
    prog[cover.P_LOWEST] = res.lower
    prog[cover.P_DONE] = 1.0
    return res
