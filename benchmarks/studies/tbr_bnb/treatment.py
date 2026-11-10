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
certificate over that box (certificate.box_bound_float: the hull's with lo = 0 and the
size at the number of controls, the size constraint slack and the knapsack at
multiplier 0). The cap (slope_cap, in doubles where its rounding is relative) is that of
certificate.ExactData.slope_cap, ||y|| / sqrt(L), with one L for every group, the
full Gram matrix M'M's (certificate.gram_lower): the controls' Gram matrix is a
principal submatrix of M'M, so when no entry of M'M is negative none of its entries
is and its smallest diagonal entry is at least M'M's, and otherwise its smallest
eigenvalue is at least M'M's (the interlacing inequalities, Hogben 2007, Sect. 8.2,
Fact 5, applied once per deleted row and column). With more geos than periods M'M
is singular, and only the diagonal bound is above zero.

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

The gap. With ``gap`` g above zero every screen and every ceiling reads a target
RSS, the incumbent's RSS over (1 + g)^2, in place of the incumbent's own: a group is
admitted and certified against ceiling(y'y, r_star, target). The ceiling's first
term does not read the RSS, so a design that could round above the incumbent's class
is still excluded exactly. A certified search then proves that every design whose
key is above the reported design's is in its class with an RSS at least the target,
so with a required impact at least the reported design's over 1 + g, the required
impact being a constant times sqrt(RSS / (n - 2)): the reported design needs at most
1 + g times the optimum's required impact (derived here). A group whose optimum is
below the target replaces the incumbent and the target falls with it; what was
proved against a higher target still holds. g = 0 is the exact search.

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

import functools
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from fractions import Fraction as Q
from itertools import combinations
from math import comb
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import nnls

from mlsynth.exceptions import MlsynthDataError

from . import fastfloor
from .certificate import box_bound_float, gram_lower, qarr
from .gated import certify_gated, gates_of
from .progress import emit, reporting, tagged

#: Half a step of round(corr, 2), and the slack for the key's correlation being a double.
EDGE = Q(5, 1000)
SLACK = Q(1, 10 ** 9)


def gram_floor(M) -> Optional[Q]:
    """certificate.gram_lower of M'M, exact: at most x'x for every control group of
    every treatment group, or None when no positive bound is found (module
    docstring)."""
    Mq = qarr(np.asarray(M, dtype=float))
    return gram_lower(Mq.T @ Mq)


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


#: The range in which the float slope cap's relative argument holds: a quotient
#: y'y / lambda and its root are normal doubles, so each rounding is relative.
_CAP_RANGE = (2.0 ** -1000, 2.0 ** 1000)


#: The scale of the exact cap's integer root: the cap exceeds sqrt(y'y / lambda) by
#: at most a relative 2^-64.
_ROOT_SCALE = 2 ** 64


def root_above(q: Q) -> Q:
    """A rational r > 0 with r^2 > q, for a rational q > 0: with q = a / b in lowest
    terms and K = _ROOT_SCALE, r = (isqrt(a b K^2) + 1) / (b K). Then (b K r)^2 >
    a b K^2, so r^2 > a / b, and b K r - 1 <= sqrt(a b) K, so r exceeds sqrt(q) by
    at most 1 / (b K) <= sqrt(q) / K since a b >= 1 (derived here)."""
    a, b = q.numerator, q.denominator
    return Q(math.isqrt(a * b * _ROOT_SCALE * _ROOT_SCALE) + 1, b * _ROOT_SCALE)


def _cap_exact(yy: Q, lam: Q) -> float:
    """The slope cap from rationals, for any y'y and lambda: root_above(y'y / lambda)
    rounded up to a double, or inf past the doubles, a cap whose bound clears
    nothing."""
    q = Q(yy) / lam
    if q == 0:
        return 0.0
    r = root_above(q)
    try:
        c = float(r)
    except OverflowError:
        return math.inf
    return c if Q(c) >= r else math.nextafter(c, math.inf)


def slope_cap(yy: Union[float, Q], lam: Q) -> float:
    """A double cap with cap^2 lambda >= y'y, for y'y at most ``yy``.

    In doubles when ``yy`` is one and the quotient is in _CAP_RANGE: float(lambda)
    is within a relative u = 2^-53 of lambda, the quotient and its root each round
    by u more, and the product with 1 + 1e-9 by u, so the cap is at least
    sqrt(y'y / lambda) (1 + 1e-9)(1 - u)^4 / sqrt(1 + u) > sqrt(y'y / lambda), the
    margin being 10^7 times the roundings (derived here). Zero for a zero y'y.
    Otherwise, a rational y'y or a quotient off the range, the exact construction."""
    if isinstance(yy, float):
        if yy == 0.0:
            return 0.0
        lf = float(lam)
        if lf > 0.0:
            q = yy / lf
            if _CAP_RANGE[0] <= q <= _CAP_RANGE[1]:
                return math.sqrt(q) * (1 + 1e-9)
        yy = Q(yy)
    return _cap_exact(yy, lam)


def rss_floor(M, T: Sequence[int], lam: Optional[Q], yy: Union[None, float, Q] = None) -> Union[float, Q]:
    """A lower bound on the RSS of every positive-slope design whose treatment group
    is T (module docstring), a double or 0; 0 when lam is None. ``yy`` is the
    group's y'y or a double above it (all the cap needs), computed here when not
    given. The bound is the hull's at the NNLS point over the box
    {0 <= x <= cap}, box_bound_float."""
    if lam is None:
        return Q(0)
    y, X, _ = _split(M, T)
    if yy is None:
        yy = square_sum(y)
    cap = slope_cap(yy, lam)
    w, _ = nnls(X, y)
    fl = box_bound_float(y, X, np.minimum(w, cap), cap)
    return fl if fl > 0 else Q(0)


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


#: The relative margin of the float bracket and the float screen: 2^-48, 32 units of
#: rounding, against the at most four roundings of relative size 2^-53 that either
#: side carries (derived here).
_MARGIN = 2.0 ** -48
#: An entry between these, or zero, has a normal square; a sum of _MAX_PERIODS such
#: squares is below 2^1021 and does not overflow.
_TINY, _HUGE, _MAX_PERIODS = 2.0 ** -500, 2.0 ** 500, 2 ** 20
#: A product at or above this is a normal double, whose rounding is relative.
_NORMAL = 2.0 ** -1000


def yy_bracket(y) -> Optional[Tuple[float, float]]:
    """lo <= y'y <= hi in doubles, or None when the bound is not shown to hold.

    Each square of an entry that is zero or between 2^-500 and 2^500 is a normal
    double rounded by a relative 2^-53 at most, so their sum is within that of y'y,
    every term being nonnegative, and math.fsum rounds it once more (it returns the
    correctly rounded sum of its doubles). With s that value, s (1 -/+ 2^-48), each
    rounded once, bracket y'y: the four roundings together move s by less than
    5 * 2^-53 (derived here). An entry outside that range, a non-finite one, or more
    than 2^20 entries leaves no bracket, and the caller takes the exact sum."""
    y = np.asarray(y, dtype=float)
    a = np.abs(y)
    if (a.size > _MAX_PERIODS or not np.isfinite(a).all() or (a > _HUGE).any()
            or ((a > 0) & (a < _TINY)).any()):
        return None
    s = math.fsum((y * y).tolist())
    return s * (1 - _MARGIN), s * (1 + _MARGIN)


@functools.lru_cache(maxsize=16)
def _class_doubles(r_star: Q, rss_star: Q):
    """1 - up^2, 1 - low^2 and the incumbent's RSS (ceiling's terms) as doubles, each
    rounded once; None in place of 1 - up^2 when it is not positive. 1 - low^2 is
    not positive only for a class near -1, where 1 - up^2 is (up - low is 2 EDGE),
    so the ceiling there is yy (1 - up^2) in doubles and in rationals alike."""
    up = r_star + EDGE - SLACK
    low = r_star - EDGE - SLACK
    a, b = 1 - up * up, 1 - low * low
    return (float(a) if a > 0 else None), float(b), float(rss_star)


def compete_bracket(floor: Q, lo: float, hi: float, r_star: Optional[Q],
                    rss_star: Optional[Q]) -> Optional[bool]:
    """can_compete(yy, floor, r_star, rss_star) for every yy in [lo, hi] when the
    doubles show one answer for all of them, else None.

    ceiling(yy) = max(yy (1 - up^2), min(rss_star, yy (1 - low^2))) is nondecreasing
    in yy, its second term being nonnegative and growing, and its first at most zero
    whenever 1 - up^2 is: so the group competes at every yy in the bracket when the
    floor is below ceiling(lo), and at none when it is at least ceiling(hi). Each
    ceiling is computed in doubles from terms rounded once, by products rounded once,
    and compared with the floor rounded once across a relative margin of 2^-48:
    from normal positive operands the roundings move each side by under 5 * 2^-53
    (derived here). A ceiling below 2^-1000, where rounding is not relative, is left
    undecided; the floor needs no such guard, a floor rounded to a subnormal double
    being within 2^-1074 of it, far below a ceiling of 2^-1000 times the margin."""
    if r_star is None:
        return True
    a, b, rss = _class_doubles(r_star, rss_star)

    def ceil(yy):
        second = min(rss, yy * b)
        return second if a is None else max(yy * a, second)

    f = float(floor)
    c_lo, c_hi = ceil(lo), ceil(hi)
    if c_lo >= _NORMAL and f < c_lo * (1 - _MARGIN):
        return True
    if c_hi >= _NORMAL and f * (1 - _MARGIN) >= c_hi:
        return False
    return None


YYBound = Union[Tuple[float, float], Q]


def _competes(floor: Q, yy: YYBound, exact: Callable[[], Q], r_star: Optional[Q],
              rss_star: Optional[Q]) -> bool:
    """can_compete for a row whose y'y is a float bracket (decided in doubles where
    compete_bracket can, else at the exact y'y that ``exact`` returns) or exact."""
    if isinstance(yy, tuple):
        got = compete_bracket(floor, yy[0], yy[1], r_star, rss_star)
        if got is not None:
            return got
        yy = exact()
    return can_compete(yy, floor, r_star, rss_star)


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
    gap: Q = Q(0)                           # the relative gap in required impact the certificate holds at
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


def _floor_row(M, lam, T):
    y = _split(M, T)[0]
    b = yy_bracket(y)
    yy = b if b is not None else square_sum(y)
    return rss_floor(M, T, lam, yy=b[1] if b is not None else yy), yy, T


def _floor_chunk(M, lam, groups):
    """(floor, y'y, T) for each group, the y'y a float bracket (yy_bracket) or, for a
    series that leaves none, exact; the floor's cap reads the bracket's top, a bound
    on y'y from above being all the cap needs. With numba installed, fastfloor
    computes the chunk in one compiled loop, and a group it leaves (a nan floor) takes
    this path."""
    if lam is None or not fastfloor.AVAILABLE or not groups:
        return [_floor_row(M, lam, T) for T in groups]
    Mc = np.ascontiguousarray(M, dtype=float)
    fl, lo, hi = fastfloor.floors(Mc, np.ascontiguousarray(Mc.T @ Mc), np.asarray(groups, dtype=np.int64),
                                  float(lam))
    out = []
    for f, a, b, T in zip(fl, lo, hi, groups):
        if math.isnan(f):
            out.append(_floor_row(M, lam, T))
        else:
            out.append((f if f > 0 else Q(0), (float(a), float(b)), T))
    return out


def _ratio(row) -> float:
    """The row's floor over its y'y in doubles: the queue's order, one minus the
    group's squared-correlation bound."""
    floor, yy, _ = row
    top = yy[1] if isinstance(yy, tuple) else float(yy)
    return float(floor) / top if top > 0 else math.inf


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
    with tagged(treatment=tuple(T)):
        emit("group", treatment=tuple(T), ceiling=ceiling)
        y, X, rest = _split(M, T)
        kw = {} if start is None else {"start": set(start)}
        res = certify_gated(y, X, n_test=n_test, restarts=restarts, ceiling=ceiling, **kw)
        C = [rest[j] for j in res.controls] if res.passing else None
        emit("group-done", treatment=tuple(T), certified=res.certified, passing=res.passing,
             rounds=len(res.rounds), seconds=res.seconds)
        return res, C


def _checked_gap(gap) -> Q:
    """``gap`` as an exact rational: a finite nonnegative int, Fraction or float (a
    float is read at its exact binary value), else MlsynthDataError."""
    if isinstance(gap, bool) or not isinstance(gap, (int, float, Q)):
        raise MlsynthDataError(f"gap is a nonnegative number, not {gap!r}.")
    if isinstance(gap, float) and not math.isfinite(gap):
        raise MlsynthDataError(f"gap is a finite number, not {gap!r}.")
    if gap < 0:
        raise MlsynthDataError(f"gap is {gap!r}; it cannot be negative.")
    return Q(gap)


def certify_treatment(M, k: int, *, n_test: int, start: Optional[Sequence[int]] = None,
                      restarts: int = 50, pass_restarts: int = 10, gap=0,
                      n_jobs: int = 1, verbose: int = 0, progress=None, log=None,
                      log_level: int = 3) -> TreatmentCertificate:
    """The reference key's best design with ``k`` treatment geos over the columns of
    ``M`` (centred, periods by geos, every geo eligible for every role), certified
    (module docstring). ``pass_restarts`` random starts for the local search each
    surviving group gets before any is certified; 0 skips the pass. With ``gap`` g
    above zero the certificate holds within g (module docstring, "The gap"): every
    design that beats the reported one is in its correlation class and needs a
    required impact above 1 / (1 + g) of the reported design's. ``n_jobs``
    processes compute the floors, the pass and the certifications (module
    docstring); 1 runs them in this process. ``verbose``, ``progress`` and ``log``
    report the search's progress (progress.reporting), each group's events tagged
    with the group, from the workers too."""
    with reporting(verbose=verbose, progress=progress, log=log, log_level=log_level):
        out = _certify_treatment_entry(M, k, n_test=n_test, start=start, restarts=restarts,
                                       pass_restarts=pass_restarts, gap=gap, n_jobs=n_jobs)
        emit("treatment-done", certified=out.certified, treatment=out.treatment, rss=out.rss,
             passing=out.passing, survivors=len(out.survivors), seconds=out.seconds)
        return out


def _certify_treatment_entry(M, k, *, n_test, start, restarts, pass_restarts, gap, n_jobs) -> TreatmentCertificate:
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
    g = _checked_gap(gap)
    emit("treatment", k=k, p=p, groups=comb(p, k), n_jobs=n_jobs)
    with (_pool(n_jobs) if n_jobs > 1 else nullcontext(None)) as pool:
        return _certify_treatment(M, k, p, n_test, start, restarts, pass_restarts, g, n_jobs, pool, t0)


def _certify_treatment(M, k, p, n_test, start, restarts, pass_restarts, g, n_jobs, pool, t0):
    if start is not None and len(set(start)) != k:
        raise MlsynthDataError(f"the start treatment group has {len(set(start))} geos, not {k}.")
    if start is not None:
        T0 = sorted(start)
        C0, key0, rss0, rest0 = _local(M, T0, n_test, restarts)
    else:                                   # the library's design, which its own search scored
        T0, Cs = _start(M, k, n_test, restarts)
        C0, key0, rss0, rest0 = _design(M, T0, Cs, n_test)
    out = TreatmentCertificate(k=k, treatment=None, controls=None, key=None, rss=None, passing=False,
                               certified=True, groups=comb(p, k), screened_out=0, gap=g)
    shrink = (1 + g) ** 2                   # the target RSS is the incumbent's over this

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
    rows.sort(key=_ratio)                  # the highest correlation bound first
    exact = {}

    def exact_yy(T, yy):                    # a row's exact y'y, computed once a group
        if not isinstance(yy, tuple):
            return yy
        if T not in exact:
            exact[T] = _yy(M, T)
        return exact[T]

    held = [None, None, (None, None)]

    def class_of():                         # the incumbent's class and target RSS, rebuilt when it changes
        if out.key is None:
            return (None, None)
        if held[0] is not out.key or held[1] is not out.rss:
            held[0], held[1], held[2] = out.key, out.rss, (Q(out.key[4]), out.rss / shrink if g else out.rss)
        return held[2]

    def competes(row):
        floor, yy, T = row
        return _competes(floor, yy, lambda: exact_yy(T, yy), *class_of())

    if pass_restarts:                       # the pass: group -> (design, key or None)
        for wave in _waves(rows, n_jobs, lambda r: r[2] not in local and competes(r)):
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
        if competes(row):
            return True
        out.screened_out += 1
        return False

    for wave in _waves(rows, n_jobs, admit):
        exacts = [(T, exact_yy(T, yy)) for _, yy, T in wave]
        jobs = [(T, yy, ceiling(yy, *class_of()), *local.get(T, (None, None))) for T, yy in exacts]
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
