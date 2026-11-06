"""The search over treatment groups, certified (#965, step 9).

The reference key ranks a design by (corr_test, aa_test, bb_test, dw_test,
round(corr, 2), 1 / required_impact) (mlsynth.utils.tbr_helpers.design.objective.
score_split). Within one treatment group the least-RSS passing design has the key's
greatest value (gated.py), so the best design of size k is the best of the groups'
own optima, compared on the rounded correlation and then on RSS: the required
impact is a constant times sqrt(RSS / (n - 2)) for every group, the window being
the same. The search therefore needs, for each treatment group, either its own
passing optimum (gated.certify_gated) or a proof that none of its designs can match
the incumbent.

The proof is a floor on the RSS of every positive-slope design of the group. With w
= b u, every such design is a point of {0 <= w <= cap}, cap the slope cap, so the
group's nonnegative least squares, min ||y - X w||^2 over w >= 0 and w <= cap, is at
most its RSS. Its value is bounded below exactly at the NNLS point by the float hull
certificate over that box (certificate.hull_bound_float, with lo = 0 and the size at
the number of controls, so that the size constraint is slack). The cap is that of
certificate.ExactData.slope_cap, ||y|| / sqrt(lambda_min(G)), with one lower bound on
lambda_min for every group: the controls' Gram matrix is a principal submatrix of the
full one, and a principal submatrix's smallest eigenvalue is at least the full
matrix's (the interlacing inequalities, Hogben 2007, Sect. 8.2, Fact 5, applied
once per deleted row and column).

Each design of the group has corr^2 = 1 - RSS / y'y, so the floor bounds its
correlation too, and can_compete reads the rounding: a group whose correlation bound
is below the incumbent's rounded class cannot match it, and one that cannot round
above the class has to beat the incumbent's RSS to replace it. The class edges are
half a rounding step either side, less a slack of 1e-9 for the key's correlation
being a double; the slack is a tolerance, not derived.

Every group that the floor does not rule out is certified, with the incumbent
replaced whenever a group's optimum has a larger key, and the floors are read again
against the new incumbent before each certification. A group is certified against
its ceiling, the RSS below which one of its designs could beat the incumbent: each
size's relaxation is tested at y'y - ceiling instead of at the group's own
optimum, a weaker claim that closes sooner, and the group's gated search ends,
certified, at the first round whose RSS-best design is not below the ceiling.
Certifying each group's own optimum instead leaves groups open on GeoLift: one at
k = 2 branched to the depth limit at one size, and others ran to the round limit on
designs that kept failing the gates, though none of them could reach the
incumbent's class.

A group whose certificate does not close against the ceiling of its turn is
certified again once every group has been read, against the final incumbent's
ceiling, which is lower whenever the incumbent improved after its turn: a weaker
claim than the one that failed. The certificates of the groups that closed on
their turn still hold, since they were proved against ceilings at least as
high. On GeoLift at k = 4 one group of 93 is left open on its turn and closes on
its retry.

With ``n_jobs`` above one the floors are computed in chunks over a pool of forked
processes, and the pass and the certifications run in waves of ``n_jobs`` groups.
A wave's groups are screened and given their ceilings from the incumbent as the
wave is filled, and their results are applied in queue order, a group replacing
the incumbent only with a strictly larger key. A group certified in a wave against
a ceiling the wave's later results lower holds against the lower one too (no
design below a ceiling means none below a lower one), and the certified design is
the one the sequential search finds, ties included: n_jobs = 1 is that search.

Hogben, L., ed. (2007). Handbook of Linear Algebra. Chapman & Hall/CRC.
"""
from __future__ import annotations

import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from fractions import Fraction as Q
from itertools import combinations
from math import comb
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import nnls

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.marex_helpers.branch import _from_fractions, lambda_min_lower

from .certificate import hull_bound_float, qarr
from .gated import certify_gated, gates_of

#: Half a step of round(corr, 2), and the slack for the key's correlation being a double.
EDGE = Q(5, 1000)
SLACK = Q(1, 10 ** 9)


def gram_floor(M) -> Optional[Q]:
    """An exact lower bound on the smallest eigenvalue of M'M, or None when none
    positive is found; it bounds every group's control Gram, a principal submatrix
    (module docstring)."""
    Mq = qarr(np.asarray(M, dtype=float))
    lam = lambda_min_lower(_from_fractions((Mq.T @ Mq).tolist()))
    return lam if lam is not None and lam > 0 else None


def _split(M, T):
    T = sorted(T)
    rest = [j for j in range(M.shape[1]) if j not in T]
    return M[:, T].sum(axis=1), M[:, rest], rest


def _integers(v, what="y'y"):
    """The doubles ``v`` as integers over one power of two, D: a finite double is an
    integer over a power of two, n / d, and with D the largest of the d every D / d
    is an integer. Returns the integers and D (1 for no entry)."""
    try:
        parts = [float(x).as_integer_ratio() for x in np.asarray(v, dtype=float).ravel()]
    except (OverflowError, ValueError) as exc:
        raise MlsynthDataError(f"{what} needs finite values: the treatment group's series "
                               "has an infinite or missing entry.") from exc
    if not parts:
        return [], 1
    D = max(d for _, d in parts)
    return [n * (D // d) for n, d in parts], D


def square_sum(v) -> Q:
    """The exact sum of squares of the doubles ``v``: sum n_t^2 / D^2 over _integers,
    one division of Python integers. The sum of Fractions it replaces reduces every
    partial sum by a gcd; on the GeoLift window (90 periods) it took 0.19 ms a
    group, this 0.03 ms."""
    ints, D = _integers(v)
    return Q(sum(n * n for n in ints), D * D)


def design_moments(y, X, C):
    """c'u, u'Gu and y'y exactly at the control group C (columns of X), with
    c = X'y and G = X'X: u'Gu = ||X u||^2 and c'u = y'X u, X u summed over C in
    integers. The pass scores one design per group, and the exact Gram it built
    for that (certificate.ExactData) took 0.17 s of the 0.5 s a group on GeoLift;
    this takes 0.4 ms."""
    C = sorted(C)
    if not C:
        raise MlsynthDataError("the control group is empty: a design with no control has no slope.")
    yi, Dy = _integers(y)
    cols = [_integers(X[:, j], "a control's series") for j in C]
    D = max(d for _, d in cols)
    xu = [0] * len(yi)
    for vals, d in cols:
        f = D // d
        for t, v in enumerate(vals):
            xu[t] += v * f
    cu = Q(sum(a * b for a, b in zip(yi, xu)), Dy * D)
    return cu, Q(sum(v * v for v in xu), D * D), Q(sum(v * v for v in yi), Dy * Dy)


def rss_floor(M, T: Sequence[int], lam: Optional[Q], yy: Optional[Q] = None) -> Q:
    """An exact lower bound on the RSS of every positive-slope design whose treatment
    group is T (module docstring); 0 when lam is None. ``yy`` is the group's y'y,
    computed here when not given."""
    if lam is None:
        return Q(0)
    y, X, _ = _split(M, T)
    if yy is None:
        yy = square_sum(y)
    cap = Q(math.sqrt(float(yy / lam)) * (1 + 1e-9))
    while cap * cap * lam < yy:           # pragma: no cover - rounding guard
        cap *= Q(1000001, 1000000)
    w, _ = nnls(X, y)
    fl = hull_bound_float(y, X, np.minimum(w, float(cap)), 0.0, float(cap), X.shape[1])
    return max(Q(0), fl) if fl != -np.inf else Q(0)


def ceiling(yy: Q, r_star: Optional[Q], rss_star: Optional[Q]) -> Optional[Q]:
    """The RSS below which a design of a group with y'y = ``yy`` could have a key
    above the incumbent's, whose rounded correlation is ``r_star`` and RSS
    ``rss_star``; None when there is no incumbent.

    A design's corr^2 is 1 - RSS / y'y. At or below yy (1 - up^2) its correlation
    could round above the incumbent's class, so it competes whatever its RSS; at or
    above yy (1 - low^2) it cannot reach the class; between them it is in the class
    and competes when its RSS is below the incumbent's."""
    if r_star is None:
        return None
    up = r_star + EDGE - SLACK            # below up, round(corr, 2) <= r_star
    low = r_star - EDGE - SLACK           # below low, round(corr, 2) < r_star
    return max(yy * (1 - up * up), min(rss_star, yy * (1 - low * low)))


def can_compete(yy: Q, floor: Q, r_star: Optional[Q], rss_star: Optional[Q]) -> bool:
    """Whether a group whose designs have RSS at least ``floor`` (and y'y = ``yy``)
    could have a design whose key is above the incumbent's (see ceiling)."""
    c = ceiling(yy, r_star, rss_star)
    return c is None or floor < c


@dataclass
class TreatmentCertificate:
    """The outcome of :func:`certify_treatment` for one treatment size."""

    k: int
    treatment: Optional[List[int]]          # None when no design of the size passes
    controls: Optional[List[int]]
    key: Optional[tuple]
    rss: Optional[Q]
    passing: bool
    certified: bool                         # every group not ruled out was certified
    groups: int
    screened_out: int
    replaced: int = 0                       # times a survivor replaced the incumbent
    searched: int = 0                       # groups given a local search before certification
    pass_key: Optional[tuple] = None        # the incumbent's key when the pass ends
    retried: int = 0                        # open groups certified again at the final ceiling
    survivors: List[dict] = field(default_factory=list)
    seconds: float = 0.0


def _start(M, k, n_test, restarts):
    """The library's greedy search and local search (#964), every geo eligible for
    every role: the first incumbent's treatment and control groups at size k."""
    from mlsynth.utils.tbr_helpers.design.local import local_search
    from mlsynth.utils.tbr_helpers.design.search import greedy_search

    roles = [frozenset({"treatment", "control", "unassigned"})] * M.shape[1]
    out = greedy_search(M, roles, max_treatment_size=k, n_test=n_test, objective="reference")
    best = local_search(M, roles, out[-1], n_test=n_test, restarts=restarts, seed=0)
    return sorted(best.treatment), sorted(best.control)


def _design(M, T, C, n_test):
    """The design (T, C), C in geo indices, as _local reports one: C in the group's
    control indices, the reference key when it has a positive slope and passes the
    four gates (else None), its exact RSS, and the group's controls."""
    y, X, rest = _split(M, T)
    Cg = sorted(rest.index(j) for j in C)
    cu, uGu, yy = design_moments(y, X, Cg)
    ratio = cu * cu / uGu if cu > 0 else Q(0)
    gates, key = gates_of(y, X, Cg, n_test)
    return Cg, (key if cu > 0 and all(gates) else None), yy - ratio, rest


def _yy(M, T):
    y, _, _ = _split(M, T)
    return square_sum(y)


def _local(M, T, n_test, restarts):
    """The study's RSS descent in group T from ``restarts`` random starts: its design
    (indices into the group's controls), the reference key when the design has a
    positive slope and passes the four gates (else None), and its exact RSS."""
    from .search import incumbent

    y, X, rest = _split(M, T)
    _, C = incumbent(y, X, restarts=restarts)
    cu, uGu, yy = design_moments(y, X, C)
    ratio = cu * cu / uGu if cu > 0 else Q(0)
    gates, key = gates_of(y, X, C, n_test)
    return sorted(C), (key if cu > 0 and all(gates) else None), yy - ratio, rest


def _pool(n_jobs):
    """A pool of ``n_jobs`` forked workers. Forked, not spawned, so a worker starts
    with the parent's modules (and BLAS settings) already loaded."""
    return ProcessPoolExecutor(max_workers=n_jobs, mp_context=multiprocessing.get_context("fork"))


def _star(fn, args):
    return fn(*args)


def _map(pool, fn, argss):
    """fn over argss, in order: in this process when pool is None, else on the pool,
    whose errors reach the caller as they were raised."""
    if pool is None:
        return [fn(*a) for a in argss]
    return list(pool.map(_star, [fn] * len(argss), argss))


def _floor_chunk(M, lam, groups):
    out = []
    for T in groups:
        yy = _yy(M, T)
        out.append((rss_floor(M, T, lam, yy=yy), yy, T))
    return out


def _floor_rows(M, groups, lam, pool, n_jobs):
    """(floor, y'y, T) for each group, in the order given: in chunks over the pool,
    four per worker, so a slow chunk does not hold the others."""
    if pool is None or len(groups) < 2:
        return _floor_chunk(M, lam, groups)
    n = min(len(groups), 4 * n_jobs)
    size = -(-len(groups) // n)
    chunks = [groups[i:i + size] for i in range(0, len(groups), size)]
    return [r for part in _map(pool, _floor_chunk, [(M, lam, c) for c in chunks]) for r in part]


def _waves(rows, n_jobs, admit):
    """The rows in waves of up to n_jobs admitted ones: ``admit(row)`` is read as each
    wave is filled, against the incumbent as it stands then, and a row it refuses
    takes no place in the wave."""
    i = 0
    while i < len(rows):
        wave = []
        while i < len(rows) and len(wave) < n_jobs:
            row = rows[i]; i += 1
            if admit(row):
                wave.append(row)
        if wave:
            yield wave


def _optimum_timed(M, T, n_test, restarts, ceiling=None, start=None):
    t0 = time.perf_counter()
    res, C = _optimum(M, T, n_test, restarts, ceiling=ceiling, start=start)
    return res, C, time.perf_counter() - t0


def _optimum(M, T, n_test, restarts, ceiling=None, start=None):
    y, X, rest = _split(M, T)
    kw = {} if start is None else {"start": set(start)}
    res = certify_gated(y, X, n_test=n_test, restarts=restarts, ceiling=ceiling, **kw)
    C = [rest[j] for j in res.controls] if res.passing else None
    return res, C


def certify_treatment(M, k: int, *, n_test: int, start: Optional[Sequence[int]] = None,
                      restarts: int = 50, pass_restarts: int = 10,
                      n_jobs: int = 1) -> TreatmentCertificate:
    """The reference key's best design with ``k`` treatment geos over the columns of
    ``M`` (centred, periods by geos, every geo eligible for every role), certified
    (module docstring). ``pass_restarts`` random starts for the local search each
    surviving group gets before any is certified; 0 skips the pass. ``n_jobs``
    processes compute the floors, the pass and the certifications (module
    docstring); 1 runs them in this process."""
    t0 = time.perf_counter()
    M = np.asarray(M, dtype=float)
    p = M.shape[1]
    if not 1 <= k <= p - 1:
        raise MlsynthDataError(f"a treatment group of {k} geos leaves no control among {p} geos; "
                               f"k has to be between 1 and {p - 1}.")
    if pass_restarts < 0:
        raise MlsynthDataError(f"pass_restarts is {pass_restarts}; it cannot be negative.")
    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise MlsynthDataError(f"n_jobs is {n_jobs!r}; it has to be a positive integer.")
    with (_pool(n_jobs) if n_jobs > 1 else nullcontext(None)) as pool:
        return _certify_treatment(M, k, p, n_test, start, restarts, pass_restarts, n_jobs, pool, t0)


def _certify_treatment(M, k, p, n_test, start, restarts, pass_restarts, n_jobs, pool, t0):
    if start is not None and len(set(start)) != k:
        raise MlsynthDataError(f"the start treatment group has {len(set(start))} geos, not {k}.")
    if start is not None:
        T0 = sorted(start)
        C0, key0, rss0, rest0 = _local(M, T0, n_test, restarts)
    else:                                   # the library's design, which its own search scored
        T0, Cs = _start(M, k, n_test, restarts)
        C0, key0, rss0, rest0 = _design(M, T0, Cs, n_test)
    out = TreatmentCertificate(k=k, treatment=None, controls=None, key=None, rss=None, passing=False,
                               certified=True, groups=comb(p, k), screened_out=0)

    def take(T, res, C):
        out.treatment, out.controls, out.rss, out.passing = list(T), sorted(C), res.rss, True
        out.key = res.key

    local = {tuple(T0): (C0, key0)}         # the first incumbent, not yet certified
    if key0 is not None:
        out.treatment, out.controls, out.rss, out.key = list(T0), sorted(rest0[j] for j in C0), rss0, key0
        out.passing = True
    left_open = []
    lam = gram_floor(M)
    rows = _floor_rows(M, list(combinations(range(p), k)), lam, pool, n_jobs)
    rows.sort(key=lambda r: r[0] / r[1])  # the highest correlation bound first

    def class_of():
        return (None, None) if out.key is None else (Q(out.key[4]), out.rss)

    if pass_restarts:                       # the pass: group -> (design, key or None)
        for wave in _waves(rows, n_jobs, lambda r: r[2] not in local and can_compete(r[1], r[0], *class_of())):
            found = _map(pool, _local, [(M, T, n_test, pass_restarts) for _, _, T in wave])
            for (_, _, T), (C, key, rss, rest) in zip(wave, found):   # in queue order
                local[T] = (C, key)
                out.searched += 1
                if key is not None and (out.key is None or key > out.key):
                    out.treatment, out.controls, out.rss, out.key = list(T), sorted(rest[j] for j in C), rss, key
                    out.passing = True
        out.pass_key = out.key
        # best local design first; groups with no passing design after, in correlation-bound order
        rows.sort(key=lambda r: (local.get(r[2], (None, None))[1] is None,
                                 tuple(-v for v in local[r[2]][1]) if local.get(r[2], (None, None))[1] else ()))

    def admit(row):                         # screened against the incumbent as the wave fills
        floor, yy, T = row
        if can_compete(yy, floor, *class_of()):
            return True
        out.screened_out += 1
        return False

    for wave in _waves(rows, n_jobs, admit):
        jobs = [(T, yy, ceiling(yy, *class_of()), *local.get(T, (None, None))) for _, yy, T in wave]
        done = _map(pool, _optimum_timed, [(M, T, n_test, restarts, c, seed) for T, _, c, seed, _ in jobs])
        for (T, yy, c, seed, pass_key), (res, C, secs) in zip(jobs, done):   # in queue order
            if not res.certified:
                left_open.append((list(T), yy, c))
            out.survivors.append(dict(treatment=list(T), certified=res.certified, passing=res.passing,
                                      key=res.key, rounds=len(res.rounds), stopped=res.stopped,
                                      pass_key=pass_key, ceiling=c, seconds=secs, start=list(T) == T0))
            if res.passing and (out.key is None or res.key > out.key):
                take(T, res, C)
                out.replaced += 1
    for T, yy, c_then in left_open:         # the incumbent only improves, so its ceiling only falls
        c = ceiling(yy, *class_of())
        if c is not None and (c_then is None or c < c_then):
            out.retried += 1
            res, C = _optimum(M, T, n_test, restarts, ceiling=c)
            if res.passing and res.key > out.key:
                take(T, res, C)
                out.replaced += 1
            if res.certified:
                continue
        out.certified = False
    out.seconds = time.perf_counter() - t0
    return out
