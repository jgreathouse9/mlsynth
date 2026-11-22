"""Minimum-weight matching of markets into pairs by Edmonds' blossom algorithm, with the dual certificate that
proves the pairing optimal.

The cost of a design of pairs is a sum over its pairs, so the best pairing is a minimum-weight perfect matching on
the complete graph of markets (Edmonds, "Maximum matching and a polyhedron with 0,1-vertices", J. Res. NBS 69B
(1965) 125-130; "Paths, trees, and flowers", Canad. J. Math. 17 (1965) 449-467). This module is the primal-dual
algorithm in its O(n^3) form, written from Lawler, "Combinatorial Optimization: Networks and Matroids", Holt,
Rinehart and Winston, 1976, chapter 6, sections 8 to 10, and Galil, "Efficient algorithms for finding maximum
matching in graphs", ACM Computing Surveys 18 (1986) 23-38, section 3.

With an odd number of markets one is left out. A dummy market joined to every market at cost 0 makes the count
even; the market matched to it is the one left out, and the pairing of the rest is the least-cost among those of
(n - 1) / 2 pairs. That is the cardinality-optimality of Edmonds' intermediate matchings (Kaibel and Walter, "A note
on matchings constructed during Edmonds' weighted perfect matching algorithm", arXiv:1703.09505, 2017; Lawler,
chapter 6, problem 8.2).

The result carries the duals the algorithm ends with: one per market and one per odd set (a blossom). They prove the
pairing optimal by linear programming duality on Edmonds' matching polytope: the duals are feasible and their
objective equals the cost of the pairing. :func:`min_weight_matching` checks this in exact arithmetic before it
returns. Costs that are integers, small enough that every sum of them is exact in floating point, give an exact
certificate; other costs give one exact to rounding, with the tolerance stated on the result.
"""
from fractions import Fraction
import numpy as np
from numba import njit

# rows of the integer state
MATE, INB, BPAR, BBASE, LABEL, LABU, LABV, NCH, BESTU, BESTV, BLN, BLOK = range(12)
# planes of the per-blossom arrays: children, the edge from child k to child k + 1 (ex in child k, ey in child k + 1),
# and the blossom's list of least-slack edges to other S-blossoms
CH, EX, EY, BLU, BLV = range(5)


@njit(cache=True)
def _slack(W, dual, i, j):
    return dual[i] + dual[j] - 2.0 * W[i, j]


@njit(cache=True)
def _leaves(b, m, A, I, out, stack):
    k = 0
    top = 1
    stack[0] = b
    while top > 0:
        top -= 1
        x = stack[top]
        if x < m:
            out[k] = x
            k += 1
        else:
            for i in range(I[NCH, x]):
                stack[top] = A[CH, x, i]
                top += 1
    return k


@njit(cache=True)
def _assign(w, t, p, m, A, I, Q, CNT, buf, stack):
    """Label the top-level blossom of vertex w with t (1 = S, 2 = T) through the edge (p, w), p outside. A T label
    labels the mate of the blossom's base S in turn. S blossoms' vertices go on the scan queue."""
    while True:
        b = I[INB, w]
        I[LABEL, w] = t
        I[LABEL, b] = t
        I[LABU, w] = p
        I[LABV, w] = w
        I[LABU, b] = p
        I[LABV, b] = w
        I[BESTU, w] = -1
        I[BESTU, b] = -1
        if t == 1:
            k = _leaves(b, m, A, I, buf, stack)
            for i in range(k):
                Q[CNT[0]] = buf[i]
                CNT[0] += 1
            return
        base = I[BBASE, b]
        w, t, p = I[MATE, base], 1, base


@njit(cache=True)
def _scan(v, w, I, path):
    """Trace from the S vertices v and w back toward their trees' roots, alternately. Returns the base vertex of
    the first blossom both paths reach (a new blossom closes there), or -1 (different trees: an augmenting path)."""
    n = 0
    base = -1
    while v != -1:
        b = I[INB, v]
        if I[LABEL, b] & 4:
            base = I[BBASE, b]
            break
        I[LABEL, b] |= 4
        path[n] = b
        n += 1
        if I[LABU, b] == -1:
            v = -1
        else:
            v = I[LABU, I[INB, I[LABU, b]]]
        if w != -1:
            v, w = w, v
    for i in range(n):
        I[LABEL, path[i]] &= ~4
    return base


@njit(cache=True)
def _add_blossom(base, v, w, W, m, A, I, dual, Q, CNT, FREE, buf, stack, side, bt):
    """Close the blossom formed by the tight edge (v, w) between two S blossoms of one tree, whose paths meet at
    the blossom of vertex base. Its children run from the base's blossom down the v side, across (v, w), and up the
    w side; it is labelled S, and the T vertices it absorbs go on the scan queue."""
    bb = I[INB, base]
    bv = I[INB, v]
    bw = I[INB, w]
    CNT[1] -= 1
    b = FREE[CNT[1]]
    I[BBASE, b] = base
    I[BPAR, b] = -1
    I[BPAR, bb] = b
    k1 = 0
    x = bv
    while x != bb:
        I[BPAR, x] = b
        side[k1] = x
        k1 += 1
        x = I[INB, I[LABU, x]]
    nc = 0
    A[CH, b, nc] = bb
    nc += 1
    for i in range(k1 - 1, -1, -1):
        A[CH, b, nc] = side[i]
        nc += 1
    x = bw
    while x != bb:
        I[BPAR, x] = b
        A[CH, b, nc] = x
        nc += 1
        x = I[INB, I[LABU, x]]
    I[NCH, b] = nc
    for i in range(nc):
        j = i + 1
        if j <= k1:
            c = A[CH, b, j]
            A[EX, b, i] = I[LABU, c]
            A[EY, b, i] = I[LABV, c]
        elif j == k1 + 1:
            A[EX, b, i] = v
            A[EY, b, i] = w
        else:
            c = A[CH, b, i]
            A[EX, b, i] = I[LABV, c]
            A[EY, b, i] = I[LABU, c]
    I[LABEL, b] = 1
    I[LABU, b] = I[LABU, bb]
    I[LABV, b] = I[LABV, bb]
    dual[b] = 0.0
    k = _leaves(b, m, A, I, buf, stack)
    for i in range(k):
        x = buf[i]
        if I[LABEL, I[INB, x]] == 2:
            Q[CNT[0]] = x
            CNT[0] += 1
        I[INB, x] = b
    # the new blossom's least-slack edges to each other S blossom, from its children's lists (or, for a child with
    # none, from all edges of its vertices)
    for i in range(2 * m):
        bt[0, i] = -1
    for ci in range(nc):
        c = A[CH, b, ci]
        if I[BLOK, c] == 1:
            for e in range(I[BLN, c]):
                ii = A[BLU, c, e]
                jj = A[BLV, c, e]
                bj = I[INB, jj]
                if bj != b and I[LABEL, bj] == 1:
                    if bt[0, bj] == -1 or _slack(W, dual, ii, jj) < _slack(W, dual, bt[0, bj], bt[1, bj]):
                        bt[0, bj] = ii
                        bt[1, bj] = jj
        else:
            k = _leaves(c, m, A, I, buf, stack)
            for li in range(k):
                ii = buf[li]
                for jj in range(m):
                    if jj == ii:
                        continue
                    bj = I[INB, jj]
                    if bj != b and I[LABEL, bj] == 1:
                        if bt[0, bj] == -1 or _slack(W, dual, ii, jj) < _slack(W, dual, bt[0, bj], bt[1, bj]):
                            bt[0, bj] = ii
                            bt[1, bj] = jj
        I[BLOK, c] = 0
        I[BESTU, c] = -1
    n = 0
    I[BESTU, b] = -1
    for bj in range(2 * m):
        if bt[0, bj] != -1:
            A[BLU, b, n] = bt[0, bj]
            A[BLV, b, n] = bt[1, bj]
            n += 1
            if I[BESTU, b] == -1 or (_slack(W, dual, bt[0, bj], bt[1, bj])
                                     < _slack(W, dual, I[BESTU, b], I[BESTV, b])):
                I[BESTU, b] = bt[0, bj]
                I[BESTV, b] = bt[1, bj]
    I[BLN, b] = n
    I[BLOK, b] = 1


@njit(cache=True)
def _free_blossom(b, I, dual, FREE, CNT):
    I[LABEL, b] = 0
    I[BBASE, b] = -1
    I[BPAR, b] = -1
    I[NCH, b] = 0
    I[BLOK, b] = 0
    I[BLN, b] = 0
    I[BESTU, b] = -1
    I[LABU, b] = -1
    I[LABV, b] = -1
    dual[b] = 0.0
    FREE[CNT[1]] = b
    CNT[1] += 1


@njit(cache=True)
def _expand(b0, endstage, m, A, I, dual, Q, CNT, FREE, buf, stack, tasks, eps):
    """Dissolve blossom b0 into its children. At the end of a stage, children that are blossoms with zero dual are
    dissolved too. Inside a stage b0 is a T blossom whose dual reached zero: the children on the even path from the
    one it was entered through to the base are relabelled T, S, ..., T, and the rest are relabelled where an S
    vertex already reached one of their vertices."""
    top = 1
    tasks[0] = b0
    while top > 0:
        top -= 1
        b = tasks[top]
        nc = I[NCH, b]
        for i in range(nc):
            s = A[CH, b, i]
            I[BPAR, s] = -1
            if s < m:
                I[INB, s] = s
            elif endstage and dual[s] <= eps:
                tasks[top] = s
                top += 1
            else:
                k = _leaves(s, m, A, I, buf, stack)
                for li in range(k):
                    I[INB, buf[li]] = s
        if (not endstage) and b == b0 and I[LABEL, b] == 2:
            entry = I[INB, I[LABV, b]]
            j = 0
            while A[CH, b, j] != entry:
                j += 1
            if j % 2 == 1:
                j -= nc
                jstep = 1
            else:
                jstep = -1
            p = I[LABU, b]
            q = I[LABV, b]
            while j != 0:
                I[LABEL, q] = 0
                I[LABEL, I[MATE, I[BBASE, I[INB, q]]]] = 0
                _assign(q, 2, p, m, A, I, Q, CNT, buf, stack)
                j += jstep
                if jstep == 1:
                    x = A[EX, b, j % nc]
                    y = A[EY, b, j % nc]
                else:
                    x = A[EY, b, (j - 1) % nc]
                    y = A[EX, b, (j - 1) % nc]
                p, q = x, y
                j += jstep
            bv = A[CH, b, 0]
            I[LABEL, q] = 2
            I[LABEL, bv] = 2
            I[LABU, q] = p
            I[LABV, q] = q
            I[LABU, bv] = p
            I[LABV, bv] = q
            I[BESTU, bv] = -1
            j += jstep
            while A[CH, b, j % nc] != entry:
                c = A[CH, b, j % nc]
                if I[LABEL, c] == 1:
                    j += jstep
                    continue
                k = _leaves(c, m, A, I, buf, stack)
                hit = -1
                for li in range(k):
                    if I[LABEL, buf[li]] != 0:
                        hit = buf[li]
                        break
                if hit != -1:
                    I[LABEL, hit] = 0
                    I[LABEL, I[MATE, I[BBASE, c]]] = 0
                    _assign(hit, 2, I[LABU, hit], m, A, I, Q, CNT, buf, stack)
                j += jstep
        _free_blossom(b, I, dual, FREE, CNT)


@njit(cache=True)
def _augment_blossom(b0, v0, m, A, I, tasks):
    """Make vertex v0 the base of blossom b0: flip the matching along the even path inside it from v0's child to
    the base, recursively in the children it crosses, and rotate the children so v0's child comes first."""
    top = 1
    tasks[0, 0] = b0
    tasks[1, 0] = v0
    while top > 0:
        top -= 1
        b = tasks[0, top]
        v = tasks[1, top]
        t = v
        while I[BPAR, t] != b:
            t = I[BPAR, t]
        if t >= m:
            tasks[0, top] = t
            tasks[1, top] = v
            top += 1
        nc = I[NCH, b]
        i = 0
        while A[CH, b, i] != t:
            i += 1
        if i % 2 == 1:
            jstep = 1
            end = nc
        else:
            jstep = -1
            end = 0
        pos = i
        while pos != end:
            pos += jstep
            if jstep == 1:
                x = A[EX, b, pos % nc]
                y = A[EY, b, pos % nc]
            else:
                x = A[EY, b, (pos - 1) % nc]
                y = A[EX, b, (pos - 1) % nc]
            cx = A[CH, b, pos % nc]
            cy = A[CH, b, (pos + jstep) % nc]
            if cx >= m:
                tasks[0, top] = cx
                tasks[1, top] = x
                top += 1
            if cy >= m:
                tasks[0, top] = cy
                tasks[1, top] = y
                top += 1
            I[MATE, x] = y
            I[MATE, y] = x
            pos += jstep
        if i > 0:
            for plane in (CH, EX, EY):
                row = A[plane, b, :nc].copy()
                for k in range(nc):
                    A[plane, b, k] = row[(k + i) % nc]
        I[BBASE, b] = v


@njit(cache=True)
def _augment(v, w, m, A, I, tasks):
    """Augment along the path through the tight edge (v, w) between the roots of two trees."""
    for side in range(2):
        s, j = (v, w) if side == 0 else (w, v)
        while True:
            bs = I[INB, s]
            if bs >= m:
                _augment_blossom(bs, s, m, A, I, tasks)
            I[MATE, s] = j
            if I[LABU, bs] == -1:
                break
            t = I[LABU, bs]
            bt = I[INB, t]
            s2 = I[LABU, bt]
            j2 = I[LABV, bt]
            if bt >= m:
                _augment_blossom(bt, j2, m, A, I, tasks)
            I[MATE, j2] = s2
            s, j = s2, j2


@njit(cache=True)
def _solve(W, eps):
    """Maximum-weight perfect matching on the complete graph of an even number m of vertices, weights W (the
    diagonal is not read). Duals are kept doubled: the constraint on an edge (i, j) is y_i + y_j + sum of z_B over
    the blossoms B containing both >= 2 W_ij, so with integer weights every quantity stays an integer. A step of
    size delta lowers S vertices' y by delta, raises T vertices' by delta, raises S blossoms' z by 2 delta and lowers
    T blossoms' by 2 delta, which keeps every tight edge in the forest tight. Returns (mate, dual over 2m slots,
    the integer state, the per-blossom arrays)."""
    m = W.shape[0]
    nb = 2 * m
    I = np.full((12, nb), -1, np.int64)
    A = np.zeros((5, nb, max(m, 1)), np.int64)
    dual = np.zeros(nb)
    I[INB, :m] = np.arange(m)
    I[BBASE, :m] = np.arange(m)
    I[NCH, :] = 0
    I[BLN, :] = 0
    I[BLOK, :] = 0
    I[LABEL, :] = 0
    wmax = 0.0
    for i in range(m):
        for j in range(m):
            if i != j and W[i, j] > wmax:
                wmax = W[i, j]
    for i in range(m):
        dual[i] = wmax
    FREE = np.empty(m, np.int64)
    for k in range(m):
        FREE[k] = nb - 1 - k
    CNT = np.zeros(2, np.int64)                         # queue length, free blossom ids
    CNT[1] = m
    Q = np.empty(4 * m * m + 16, np.int64)
    buf = np.empty(m, np.int64)
    stack = np.empty(nb + 4, np.int64)
    side = np.empty(nb, np.int64)
    path = np.empty(nb, np.int64)
    bt = np.empty((2, nb), np.int64)
    tasks = np.empty((2, nb * 2 + 4), np.int64)
    tasks1 = np.empty(nb + 4, np.int64)
    for stage in range(m + 1):
        for b in range(nb):
            if b < m or I[BBASE, b] >= 0:
                I[LABEL, b] = 0
            I[BESTU, b] = -1
            I[BLOK, b] = 0
        CNT[0] = 0
        for v in range(m):
            if I[MATE, v] == -1 and I[LABEL, I[INB, v]] == 0:
                _assign(v, 1, -1, m, A, I, Q, CNT, buf, stack)
        augmented = False
        while True:
            while CNT[0] > 0 and not augmented:
                CNT[0] -= 1
                v = Q[CNT[0]]
                for w in range(m):
                    if w == v:
                        continue
                    bv = I[INB, v]
                    bw = I[INB, w]
                    if bv == bw:
                        continue
                    ks = _slack(W, dual, v, w)
                    if ks <= eps:
                        if I[LABEL, bw] == 0:
                            _assign(w, 2, v, m, A, I, Q, CNT, buf, stack)
                        elif I[LABEL, bw] == 1:
                            base = _scan(v, w, I, path)
                            if base >= 0:
                                _add_blossom(base, v, w, W, m, A, I, dual, Q, CNT, FREE, buf, stack, side, bt)
                            else:
                                _augment(v, w, m, A, I, tasks)
                                augmented = True
                                break
                        elif I[LABEL, w] == 0:
                            I[LABEL, w] = 2
                            I[LABU, w] = v
                            I[LABV, w] = w
                    elif I[LABEL, bw] == 1:
                        if I[BESTU, bv] == -1 or ks < _slack(W, dual, I[BESTU, bv], I[BESTV, bv]):
                            I[BESTU, bv] = v
                            I[BESTV, bv] = w
                    elif I[LABEL, w] == 0:
                        if I[BESTU, w] == -1 or ks < _slack(W, dual, I[BESTU, w], I[BESTV, w]):
                            I[BESTU, w] = v
                            I[BESTV, w] = w
            if augmented:
                break
            dt = 0
            delta = np.inf
            du = -1
            dv = -1
            db = -1
            for v in range(m):
                if I[LABEL, I[INB, v]] == 0 and I[BESTU, v] != -1:
                    d = _slack(W, dual, I[BESTU, v], I[BESTV, v])
                    if d < delta:
                        delta, dt, du, dv = d, 2, I[BESTU, v], I[BESTV, v]
            for b in range(nb):
                if (b < m or I[BBASE, b] >= 0) and I[BPAR, b] == -1 and I[LABEL, b] == 1 and I[BESTU, b] != -1:
                    d = _slack(W, dual, I[BESTU, b], I[BESTV, b]) / 2.0
                    if d < delta:
                        delta, dt, du, dv = d, 3, I[BESTU, b], I[BESTV, b]
            for b in range(m, nb):
                if I[BBASE, b] >= 0 and I[BPAR, b] == -1 and I[LABEL, b] == 2 and dual[b] / 2.0 < delta:
                    delta, dt, db = dual[b] / 2.0, 4, b
            if dt == 0:
                break
            for v in range(m):
                lb = I[LABEL, I[INB, v]]
                if lb == 1:
                    dual[v] -= delta
                elif lb == 2:
                    dual[v] += delta
            for b in range(m, nb):
                if I[BBASE, b] >= 0 and I[BPAR, b] == -1:
                    if I[LABEL, b] == 1:
                        dual[b] += 2.0 * delta
                    elif I[LABEL, b] == 2:
                        dual[b] -= 2.0 * delta
            if dt == 2 or dt == 3:
                Q[CNT[0]] = du
                CNT[0] += 1
            else:
                _expand(db, False, m, A, I, dual, Q, CNT, FREE, buf, stack, tasks1, eps)
        if not augmented:
            break
        for b in range(m, nb):
            if I[BBASE, b] >= 0 and I[BPAR, b] == -1 and I[LABEL, b] == 1 and dual[b] <= eps:
                _expand(b, True, m, A, I, dual, Q, CNT, FREE, buf, stack, tasks1, eps)
    return I[MATE, :m].copy(), dual, I, A


def _validate(W):
    try:
        W = np.asarray(W, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"the cost matrix must hold numbers: {exc}") from exc
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"the cost matrix must be square, not of shape {W.shape}")
    off = ~np.eye(len(W), dtype=bool)
    if not np.isfinite(W[off]).all():
        raise ValueError("the cost matrix must be finite off the diagonal")
    if not np.array_equal(W[off], W.T[off]):
        raise ValueError("the cost matrix must be symmetric")
    return W


def _integral(W):
    """True when the off-diagonal costs are integers whose sums over any matching are exact in floating point."""
    n = len(W)
    if n < 2:
        return True
    off = W[~np.eye(n, dtype=bool)]
    return bool(np.all(off == np.round(off)) and np.abs(off).max() * 4 * n < 2.0 ** 52)


def min_weight_matching(W):
    """The minimum-cost pairing of the n markets indexed by the symmetric cost matrix W (the diagonal is not read):
    a perfect matching when n is even; when n is odd, the least-cost (n - 1) / 2 pairs and the one market left out.

    Returns a dict: pairs (sorted (i, j), i < j), unmatched (the market left out, or None), cost (the sum of W over
    the pairs), p (one dual per market, and for odd n a last one for the dummy market n), blossoms (a list of
    (sorted members, q) for every odd set with a positive dual q; members may include the dummy n), and tolerance
    (0 when the certificate is exact). The duals satisfy p_i + p_j - sum of q over the odd sets containing i and j
    <= W_ij for every pair of markets, q >= 0, and sum p - sum floor(|B| / 2) q_B = cost: no pairing costs less.
    Raises ValueError for a cost matrix that is not square, symmetric and finite off the diagonal, and RuntimeError
    if the certificate does not hold (a defect in the solver, never in the data)."""
    W = _validate(W)
    n = len(W)
    if n == 0:
        return dict(pairs=[], unmatched=None, cost=0.0, p=np.zeros(0), blossoms=[], tolerance=0.0)
    m = n + n % 2
    Wp = np.zeros((m, m))
    Wp[:n, :n] = -W
    np.fill_diagonal(Wp, 0.0)
    exact = _integral(W)
    scale = float(np.abs(W[~np.eye(n, dtype=bool)]).max()) if n > 1 else 0.0
    eps = 0.0 if exact else 1e-13 * max(scale, 1e-300)
    mate, dual, I, A = _solve(Wp, eps)
    pairs = sorted((int(i), int(mate[i])) for i in range(n) if i < mate[i] < n)
    unmatched = None
    if n % 2:
        unmatched = int(mate[n])
    p = -dual[:m] / 2.0
    blossoms = []
    for b in range(m, 2 * m):
        if I[BBASE, b] >= 0 and dual[b] != 0.0:
            out = np.empty(m, np.int64)
            k = _leaves(b, m, A, I, out, np.empty(2 * m + 4, np.int64))
            blossoms.append((tuple(sorted(int(x) for x in out[:k])), float(dual[b] / 2.0)))
    cost = float(sum(W[i, j] for i, j in pairs))
    result = dict(pairs=pairs, unmatched=unmatched, cost=cost, p=p, blossoms=blossoms,
                  tolerance=0.0 if exact else 1e-9 * max(scale, 1.0) * m)
    gap = check_certificate(W, result)
    if gap > result["tolerance"]:
        raise RuntimeError(f"the matching's dual certificate fails by {gap:g} (tolerance {result['tolerance']:g}): "
                           "the solver returned a pairing it cannot prove optimal")
    return result


def _interval_order(family, m):
    """An order of the vertices 0..m-1 in which every set of family is a run of consecutive positions, as
    (order, lo, hi) with set b at positions lo[b] .. hi[b] - 1; None when no such order is found.

    A laminar family (any two sets nested or disjoint), such as the odd sets of Edmonds' algorithm, is a tree, and
    reading its leaves depth first puts each set's members together (Korte and Vygen, "Combinatorial Optimization:
    Theory and Algorithms", 6th ed., Springer 2018, section 2.2, the tree representation of a laminar family). Sorting
    the vertices lexicographically by membership, the largest set the most significant key, gives that order: the
    members of a set agree on every larger set, which contains them all or none of them. Crossing sets get None."""
    k = len(family)
    if k == 0:
        return np.arange(m), np.zeros(0, np.int64), np.zeros(0, np.int64)
    M = np.zeros((k, m), dtype=bool)
    for b, members in enumerate(family):
        M[b, list(members)] = True
    size = M.sum(axis=1)
    rank = np.argsort(-size, kind="stable")
    order = np.lexsort(~M[rank][::-1])
    pos = np.empty(m, np.int64)
    pos[order] = np.arange(m)
    lo = np.where(M, pos[None, :], m).min(axis=1)
    hi = np.where(M, pos[None, :], -1).max(axis=1) + 1
    if np.any(hi - lo != size):
        return None
    return order, lo, hi


def _near_tight(Wf, p, qs, order, lo, hi):
    """The pairs (i, j), i < j, of the m vertices whose dual slack W_ij - p_i - p_j + sum of q over the sets holding
    both cannot be shown positive in floating point, as two index arrays: the pairs the certificate must settle exactly.

    With the sets laid out as intervals of order, the sets holding the vertices at positions a < b are those with
    lo <= a and hi - 1 >= b, so the sum of their q is a two-dimensional cumulative sum of q placed at (lo, hi - 1).
    Each slack is a floating sum of at most 2m + len(qs) + 3 nonzero terms through additions, whose error is at most
    gamma_k times the sum of the terms' magnitudes, gamma_k = k u / (1 - k u), u = 2^-53 (Higham, "Accuracy and
    Stability of Numerical Algorithms", 2nd ed., SIAM 2002, section 4.2); the screen doubles that bound and adds
    k times the smallest subnormal for underflow. A pair whose computed slack exceeds the bound has positive exact
    slack; every other pair, a NaN included, is returned. That is a floating-point filter in front of an exact
    test (Shewchuk, "Adaptive precision floating-point arithmetic and fast robust geometric predicates", Discrete
    and Computational Geometry 18 (1997) 305-363)."""
    m = len(p)
    q = np.asarray(qs, dtype=float)
    G = np.zeros((m, m))
    Ga = np.zeros((m, m))
    np.add.at(G, (lo, hi - 1), q)
    np.add.at(Ga, (lo, hi - 1), np.abs(q))
    C = np.cumsum(np.cumsum(G, axis=0)[:, ::-1], axis=1)[:, ::-1]
    Ca = np.cumsum(np.cumsum(Ga, axis=0)[:, ::-1], axis=1)[:, ::-1]
    P = p[order]
    Wp = Wf[np.ix_(order, order)]
    with np.errstate(invalid="ignore", over="ignore"):
        S = Wp - P[:, None] - P[None, :] + C
        A = np.abs(Wp) + np.abs(P)[:, None] + np.abs(P)[None, :] + Ca
        k = 2 * m + len(q) + 8
        u = 2.0 ** -53
        bound = 2.0 * (k * u / (1.0 - k * u)) * A + k * 2.0 ** -1074
        uncertain = ~(S > bound)
    a, b = np.nonzero(np.triu(uncertain, 1))
    return order[a], order[b]


def check_certificate(W, result):
    """The largest violation, in exact arithmetic, of the conditions that prove result's pairing a minimum-cost one:
    the pairing covers every market once (all but one when n is odd); every dual of an odd set is >= 0; every pair of
    markets (and of a market and the dummy, for odd n, at cost 0) has p_i + p_j - sum of q_B over the odd sets
    containing both <= W_ij; and sum p - sum floor(|B| / 2) q_B equals the pairing's cost. Returns 0 for a proof,
    a positive number otherwise, and infinity when the pairing is not a matching of the required size or an odd set
    is not a set of distinct vertices.

    The pair conditions are screened in floating point (:func:`_near_tight`) and only the pairs the screen cannot
    clear are evaluated in rational arithmetic, so the result is the exact one at a cost near one floating pass over
    the pairs. When the odd sets cannot be laid out as intervals (they cross), every pair is evaluated exactly."""
    W = np.asarray(W, dtype=float)
    n = len(W)
    m = n + n % 2
    pairs = list(result["pairs"])
    covered = [v for pr in pairs for v in pr]
    if len(set(covered)) != len(covered) or len(pairs) != n // 2 or any(not 0 <= v < n for v in covered):
        return float("inf")
    if n % 2 and (result["unmatched"] in covered or not 0 <= result["unmatched"] < n):
        return float("inf")
    p = np.asarray(result["p"], dtype=float)
    if p.shape != (m,):
        return float("inf")
    family = [tuple(int(v) for v in members) for members, _ in result["blossoms"]]
    qs = [float(q) for _, q in result["blossoms"]]
    if any(len(set(mem)) != len(mem) or any(not 0 <= v < m for v in mem) for mem in family):
        return float("inf")
    F = Fraction
    worst = max([F(0)] + [-F(q) for q in qs])
    obj = sum((F(float(x)) for x in p), F(0)) - sum(((len(mem) // 2) * F(q) for mem, q in zip(family, qs)), F(0))
    cost = sum((F(float(W[i, j])) for i, j in pairs), F(0))
    worst = max(worst, abs(obj - cost))
    Wf = np.zeros((m, m))
    Wf[:n, :n] = W
    np.fill_diagonal(Wf, 0.0)
    layout = _interval_order(family, m)
    if layout is None:
        ii, jj = np.triu_indices(m, 1)
    else:
        ii, jj = _near_tight(Wf, p, qs, *layout)
    M = np.zeros((len(family), m), dtype=bool)
    for b, mem in enumerate(family):
        M[b, list(mem)] = True
    for i, j in zip(ii.tolist(), jj.tolist()):
        cross = sum((F(qs[b]) for b in np.nonzero(M[:, i] & M[:, j])[0].tolist()), F(0))
        worst = max(worst, F(p[i]) + F(p[j]) - cross - F(Wf[i, j]))
    return float(worst)
