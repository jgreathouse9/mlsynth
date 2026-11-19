"""The search over treatment groups, certified, under the relative key (#965).

The key. A design (T, C) is ranked by the library's relative key, level_T /
required_impact (mlsynth.utils.tbr_helpers.design.objective.score_split,
objective="relative"): level_T is the mean of the treated aggregate over the raw
window and required_impact the smallest effect the experiment could detect. Power is
the only criterion: no assumption gates and no correlation term. The four gates of
the reference key are read at the certified design and reported (gates.py); they
rank nothing. The required impact is a constant times sqrt(RSS / (n - 2)), the
constant the same for every design since the window is (score_split's
required_impact), so designs compare on level_T^2 / RSS: (T, C) beats (T*, C*)
exactly when level_T^2 RSS* > level*^2 RSS (beats). A design the library scores -inf
(its regression fits exactly, #1000) is never reported.

The panel. certify_treatment takes the raw panel, periods by geos, every geo eligible
for every role, and refuses a geo whose mean is not positive, the relative key
needing a positive level. The RSS is the intercept regression's, computed on the
panel centred in doubles; the levels are exact means of the raw doubles (geo_sums).

Within a treatment group the best design is the least-RSS one, of either slope sign.
A design with a negative slope for y is one with a positive slope for -y and the
same RSS, so a group is certified as two searches of the slope-free certificate
(slopefree.certify_slope_free), on y and on -y, each over its positive-slope designs.

The proof that a group cannot compete is a floor on the RSS of its designs, one per
sign. With w = b u, every positive-slope design of the series s y (s = +1 or -1) is
a point of {0 <= w <= cap}, cap the slope cap, so the nonnegative least squares
min ||s y - X w||^2 over w >= 0 and w <= cap is at most its RSS. Its value is bounded
below exactly at the NNLS point by the float hull certificate over that box
(certificate.box_bound_float: the hull's with lo = 0 and the size at the number of
controls, the size constraint slack and the knapsack at multiplier 0). The cap
(slope_cap, in doubles where its rounding is relative) is that of
certificate.ExactData.slope_cap, ||y|| / sqrt(L), with one L for every group, the
full Gram matrix M'M's (certificate.gram_lower): the controls' Gram matrix is a
principal submatrix of M'M, so when no entry of M'M is negative none of its entries
is and its smallest diagonal entry is at least M'M's, and otherwise its smallest
eigenvalue is at least M'M's (the interlacing inequalities, Hogben 2007, Sect. 8.2,
Fact 5, applied once per deleted row and column). With more geos than periods M'M
is singular, and only the diagonal bound is above zero. The cap reads only y'y and L,
so it is the same for both signs.

The ceiling. Against an incumbent with level* and RSS*, a design of group T beats it
only when its RSS is below RSS* (level_T / level*)^2, the group's ceiling
(relative_ceiling, from the bar RSS* / level*^2). A sign whose floor is at least the
ceiling cannot compete, and a group neither of whose signs can is ruled out; each
sign that can is certified against the ceiling (certify_slope_free's ``ceiling``),
which ends at its least-RSS design below it or proves none is.

The gap. With ``gap`` g above zero the bar is RSS* / (level*^2 (1 + g)^2): a group
is admitted and certified against that lower ceiling, so a certified search proves
that no design's key exceeds (1 + g) times the reported design's, the key being
level / (constant sqrt(RSS)) (derived here). A group whose optimum is below the
lower ceiling beats the incumbent and replaces it, and the bar falls with it; what
was proved against a higher bar still holds. g = 0 is the exact search.

The order. The groups are read best key bound first, level_T^2 over the smaller of
their two floors. A start design (the library's greedy and local searches under the
relative key, or the caller's treatment group) is the first incumbent, and a pass
gives each group that can compete the study's RSS descent on both signs before any
is certified, so the queue starts from the groups with the best designs found.

A group whose certificate does not close against the ceiling of its turn is
certified again once every group has been read, against the final incumbent's
ceiling, which is lower whenever the incumbent improved after its turn: a weaker
claim than the one that failed. The certificates of the groups that closed on their
turn still hold, since they were proved against ceilings at least as high.

With ``n_jobs`` above one the floors are computed in chunks over a pool of forked
processes, the pass runs in waves of ``n_jobs`` groups, and the certifications run
over a sliding window (_window, #1069): up to ``n_jobs`` groups at once, the next
group submitted as soon as a worker is free. A group is screened and given its
ceiling from the incumbent as it stands when the group is submitted, and results are
applied in queue order, a group replacing the incumbent only with a strictly larger
key. A group's ceiling comes from a committed prefix of the queue, so it is at least
the one the sequential search gives it; a group certified against it holds against
the lower one too, and a group whose optimum beats the sequential incumbent at its
place is found and compared with exactly that incumbent. At g = 0 the certified
design is the one the sequential search finds, ties included: n_jobs = 1 is that
search. At g > 0 a higher ceiling can find a design within the gap that a lower one
closes unseen, so the reported design can depend on n_jobs and on the order the
workers finish; it is within the gap either way.

The time limit. Past ``time_limit`` seconds (deadline.py) the floors stop between
blocks, the pass between waves, the window admits no further group, a group's branch
and bound leaves each node it enters open, and no group is retried. The groups left
neither ruled out nor certified bound the reported design (proven_gap): a design of a
group with level L and floor f has a key at most sqrt(L^2 RSS* / (level*^2 f)) times
the incumbent's, so no design's key exceeds 1 + proven_gap times the reported one.

Hogben, L., ed. (2007). Handbook of Linear Algebra. Chapman & Hall/CRC.
"""
from __future__ import annotations

import math
import multiprocessing
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from fractions import Fraction as Q
from math import comb
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import nnls

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.tbr_helpers.calibration import resolve_blank_periods
from mlsynth.utils.tbr_helpers.design.objective import (MIN_HOLDOUT_PRETEST, aa_test_ok, check_positive_levels,
                                                        score_split)

from . import deadline, fastfloor
from .certificate import box_bound_float, gram_lower, qarr
from .gates import gates_of
from .progress import emit, reporting, tagged

#: The slope signs a group is certified on: +1 searches y, -1 searches -y.
SIGNS = (1, -1)


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


def _checked_panel(M) -> np.ndarray:
    """The panel as a periods-by-geos array of doubles, refused with MlsynthDataError
    when it is not two-dimensional or has a missing or infinite entry, named by its
    geo and period (the first in period order) with the count of such entries. The
    library's least squares otherwise fail on it with numpy's LinAlgError."""
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise MlsynthDataError(f"the panel has {M.ndim} dimensions; it has to be periods by geos.")
    bad = ~np.isfinite(M)
    if bad.any():
        t, j = (int(i) for i in np.argwhere(bad)[0])
        raise MlsynthDataError(f"the panel needs finite values: geo {j} has a missing or infinite value "
                               f"at period {t} ({int(bad.sum())} entries in all).")
    return M


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


def geo_sums(M) -> List[Q]:
    """Each geo's sum over the window, exact: sum n_t / D over _integers, so a
    group's level is a sum of these over n (group_level)."""
    out = []
    for j in range(np.asarray(M).shape[1]):
        ints, D = _integers(np.asarray(M, dtype=float)[:, j], "a geo's series")
        out.append(Q(sum(ints), D))
    return out


def group_level(sums: Sequence[Q], T: Sequence[int], n: int) -> Q:
    """The treated aggregate's mean over the window, exact: the sum of its geos'
    sums (geo_sums) over the n periods."""
    return sum((sums[j] for j in T), Q(0)) / n


def relative_bar(rss_star: Q, level_star: Q, g: Q) -> Q:
    """RSS* / (level*^2 (1 + g)^2): the group ceiling's factor, so that a group with
    level L has ceiling bar L^2 (relative_ceiling)."""
    return Q(rss_star) / (Q(level_star) ** 2 * (1 + Q(g)) ** 2)


def relative_ceiling(level: Q, bar: Optional[Q]) -> Optional[Q]:
    """The RSS below which a design of a group with this level could have a key above
    the incumbent's (at g = 0, exactly the designs that beat it): bar level^2; None
    when there is no incumbent."""
    if bar is None:
        return None
    return bar * Q(level) ** 2


def beats(level: Q, rss: Q, level_star: Q, rss_star: Q) -> bool:
    """Whether a design with this level and RSS has a strictly larger relative key
    than the incumbent's: level^2 RSS* > level*^2 RSS, the key being level over a
    constant times sqrt(RSS) (module docstring)."""
    return Q(level) ** 2 * Q(rss_star) > Q(level_star) ** 2 * Q(rss)


def _check_levels(M) -> None:
    """Refuse a geo whose mean is not positive (MlsynthDataError, naming each),
    every geo being eligible for the treatment group."""
    check_positive_levels(M, range(M.shape[1]), list(range(M.shape[1])),
                          need="the certified search ranks designs by relative MDE, a required impact "
                               "as a share of the treated group's level, which needs a positive level",
                          remedy="Pass the raw panel, not a centred one, and drop those geos.")


def rss_floor(M, T: Sequence[int], lam: Optional[Q], yy: Union[None, float, Q] = None,
              sign: int = 1) -> Union[float, Q]:
    """A lower bound on the RSS of every design whose treatment group is T and whose
    slope has the sign ``sign`` (module docstring), a double or 0; 0 when lam is None.
    ``yy`` is the group's y'y or a double above it (all the cap needs), computed here
    when not given. The bound is the hull's at the NNLS point over the box
    {0 <= x <= cap} for the series sign y, box_bound_float."""
    if lam is None:
        return Q(0)
    y, X, _ = _split(M, T)
    y = sign * y
    if yy is None:
        yy = square_sum(y)
    cap = slope_cap(yy, lam)
    w, _ = nnls(X, y)
    fl = box_bound_float(y, X, np.minimum(w, cap), cap)
    return fl if fl > 0 else Q(0)


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


#: The unit roundoff of a double.
_U = 2.0 ** -53


def level_doubles(sums: Sequence[Q], n: int) -> np.ndarray:
    """Each geo's level, sum / n, rounded once to a double (a Fraction's float is the
    correctly rounded value): the float screen's terms."""
    return np.array([float(s / n) for s in sums], dtype=float)


def _screen_terms(bar: Optional[Q], ell: np.ndarray, k: int):
    """(bar as a double, the levels as doubles, the margin) for the float screen, or
    None when it is not shown to hold: no incumbent, or a bar or a level below 2^-1000,
    where rounding is not relative.

    The margin is (8 k + 16) 2^-53. A group's level in doubles, the sum of its k
    rounded levels in any order, is within a relative gamma_k of its level, every term
    being positive (gamma_m = m u / (1 - m u), Higham 2002, Sect. 3.1); its square and
    the product with the bar, each rounded once, put the computed ceiling within
    gamma_{2k+3} of bar L^2, and the product with 1 + margin rounds once more. A floor
    at or above the computed bar L^2 (1 + margin) is therefore at or above bar L^2
    whenever the margin is at least twice gamma_{2k+4}, which (8 k + 16) u is while
    (2k + 4) u <= 1/2 (derived here)."""
    if bar is None:
        return None
    try:
        b = float(bar)
    except OverflowError:                # a bar past the doubles: nothing is dropped in doubles
        return None
    if not (b >= _NORMAL and math.isfinite(b)) or not bool(np.all(ell >= _NORMAL)):
        return None
    return b, ell, (8 * k + 16) * _U


def _drop_side(fl, Lhat, b, margin) -> np.ndarray:
    """Whether a sign's floor ``fl`` is shown in doubles to be at or above its group's
    ceiling (_screen_terms): the ceiling b L^2 computed as b (L L), a normal finite
    double, and the floor at or above it times 1 + margin. A nan floor (left to
    rss_floor) or a ceiling the doubles do not carry is never dropped."""
    fl = np.asarray(fl, dtype=float); Lhat = np.asarray(Lhat, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        sq = Lhat * Lhat
        c = b * sq
        return (np.isfinite(fl) & np.isfinite(c) & (sq >= _NORMAL) & (c >= _NORMAL)
                & (fl >= c * (1 + margin)))


def _kernel_floors(M, lam, groups):
    """The compiled floors of each group on y and on -y (fastfloor.floors)."""
    Mc = np.ascontiguousarray(M, dtype=float)
    G = np.ascontiguousarray(Mc.T @ Mc)
    g = np.asarray(groups, dtype=np.int64)
    return (fastfloor.floors(Mc, G, g, float(lam), 1.0)[0],
            fastfloor.floors(Mc, G, g, float(lam), -1.0)[0])


def _side(M, lam, T, f, sign):
    """A kernel floor as a row carries it: rss_floor's where the kernel left the
    group (nan), else the floor clipped at zero."""
    if math.isnan(f):
        return rss_floor(M, T, lam, sign=sign)
    return f if f > 0 else Q(0)


def _floor_row(M, lam, T):
    """(floor on y, floor on -y, T) for the group T, in Python."""
    y = _split(M, T)[0]
    b = yy_bracket(y)
    yy = b[1] if b is not None else square_sum(y)
    return rss_floor(M, T, lam, yy=yy, sign=1), rss_floor(M, T, lam, yy=yy, sign=-1), T


def _floor_chunk(M, lam, groups):
    """(floor on y, floor on -y, T) for each group. With numba installed, fastfloor
    computes the chunk in one compiled loop per sign, and a group it leaves (a nan
    floor) takes rss_floor for that sign."""
    if lam is None or not fastfloor.AVAILABLE or len(groups) == 0:
        return [_floor_row(M, lam, _group(T)) for T in groups]
    fp, fm = _kernel_floors(M, lam, groups)
    out = []
    for a, b, T in zip(fp, fm, groups):
        T = _group(T)
        out.append((_side(M, lam, T, a, 1), _side(M, lam, T, b, -1), T))
    return out


def _floor_chunk_screened(M, lam, groups, scr):
    """_floor_chunk's rows less those the float screen shows cannot compete, both
    signs' floors being at or above the group's ceiling (_drop_side), and how many it
    dropped. ``scr`` is _screen_terms' (bar, levels, margin), or None for no screen."""
    if scr is None:
        return _floor_chunk(M, lam, groups), 0
    b, ell, margin = scr
    if lam is None or not fastfloor.AVAILABLE or len(groups) == 0:
        rows = _floor_chunk(M, lam, groups)
        keep = []
        for row in rows:
            L = np.array([float(ell[list(row[2])].sum())])
            if not (_drop_side([float(row[0])], L, b, margin)[0] and _drop_side([float(row[1])], L, b, margin)[0]):
                keep.append(row)
        return keep, len(rows) - len(keep)
    g = np.asarray(groups, dtype=np.int64)
    fp, fm = _kernel_floors(M, lam, g)
    Lhat = ell[g].sum(axis=1)
    drop = _drop_side(fp, Lhat, b, margin) & _drop_side(fm, Lhat, b, margin)
    out = []
    for i in np.flatnonzero(~drop):
        T = _group(g[i])
        out.append((_side(M, lam, T, fp[i], 1), _side(M, lam, T, fm[i], -1), T))
    return out, int(drop.sum())


def _screened_floor_rows(M, groups, lam, pool, n_jobs, scr):
    """_floor_rows screened where the floors are computed (_floor_chunk_screened):
    the rows that may compete, in the order given, and how many were dropped. The
    search calls _streamed_floor_rows, which builds the groups in the workers; this
    form over a list of groups is its reference in the tests. The incumbent only
    improves, so a row that cannot compete at this point cannot compete later."""
    if pool is None or len(groups) < 2:
        return _floor_chunk_screened(M, lam, groups, scr)
    n = min(len(groups), 4 * n_jobs)
    size = -(-len(groups) // n)
    chunks = [groups[i:i + size] for i in range(0, len(groups), size)]
    parts = _map(pool, _floor_chunk_screened, [(M, lam, c, scr) for c in chunks])
    return [r for rows, _ in parts for r in rows], sum(d for _, d in parts)


@dataclass
class TreatmentCertificate:
    """The outcome of :func:`certify_treatment` for one treatment size."""

    k: int
    treatment: Optional[List[int]]          # None only when every design fits exactly
    controls: Optional[List[int]]
    key: Optional[float]                    # the library's relative key at the design: level / required impact
    rss: Optional[Q]                        # its RSS, on the panel centred in doubles
    level: Optional[Q]                      # its treated aggregate's mean over the raw window, exact
    slope: Optional[int]                    # the sign of its slope, +1 or -1
    certified: bool                         # every group not ruled out was certified
    groups: int
    screened_out: int
    replaced: int = 0                       # times a survivor replaced the incumbent
    searched: int = 0                       # groups given a local search before certification
    pass_key: Optional[float] = None        # the incumbent's key when the pass ends
    retried: int = 0                        # open groups certified again at the final ceiling
    gap: Q = Q(0)                           # the relative gap in the key the certificate holds at
    gates: Optional[Dict[str, bool]] = None  # the reference key's four gates at the design; they rank nothing
    blank_periods: int = 0                  # the last periods held back from the search
    design_periods: int = 0                 # the periods the search certified on, before them
    blank_aa: List[bool] = field(default_factory=list)  # the library's A/A test at each blank origin
    time_limit: Optional[float] = None      # the seconds the search was given, None for no limit
    timed_out: bool = False                 # the limit passed before every group was resolved
    unresolved: int = 0                     # groups neither ruled out nor certified
    proven_gap: Optional[Q] = None          # no design's key exceeds 1 + this times the reported one's
    survivors: List[dict] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class _Design:
    """A design of one treatment group: its controls (geo indices), exact RSS and
    slope sign."""
    controls: List[int]
    rss: Q
    slope: int


@dataclass
class _GroupOutcome:
    """A group's certification: its least-RSS design below the ceiling (None when no
    design is), whether every sign run closed, and the signs run."""
    design: Optional[_Design]
    certified: bool
    signs: Tuple[int, ...]


def _start(M, k, n_test, restarts, pool=None):
    """The library's greedy search and local search (#964) under the relative key,
    every geo eligible for every role: the first incumbent's treatment and control
    groups at size k, M the raw panel.

    With a pool, local_search's restarts run on it. Its draws come from one generator
    seeded 0 and each descent depends only on its draw, so the draws are made here in
    local_search's order, descended on the pool (_descend), and the best taken in
    draw order with the greedy's descent first, an earlier one keeping a tie as
    local_search's strict comparison does: local_search's design."""
    from mlsynth.utils.tbr_helpers.design import local as lib_local
    from mlsynth.utils.tbr_helpers.design.search import greedy_search

    roles = [frozenset({"treatment", "control", "unassigned"})] * M.shape[1]
    out = greedy_search(M, roles, max_treatment_size=k, n_test=n_test, objective="relative")
    if pool is None:
        best = lib_local.local_search(M, roles, out[-1], n_test=n_test, restarts=restarts, seed=0,
                                      objective="relative")
        return sorted(best.treatment), sorted(best.control)
    if restarts < 0:
        raise MlsynthDataError(f"restarts is {restarts}; it cannot be negative.")
    first = lib_local.local_search(M, roles, out[-1], n_test=n_test, restarts=0, seed=0, objective="relative")
    best = (first.score.key, first.treatment, first.control)
    rng = np.random.default_rng(0)
    draws = [d for d in (lib_local._random_design(rng, roles, out[-1].k) for _ in range(restarts))
             if d is not None]
    if draws:
        for key, T, C in _map(pool, _descend, [(M, roles, T, sorted(C), n_test) for T, C in draws]):
            if key > best[0]:
                best = (key, T, C)
    return sorted(best[1]), sorted(best[2])


def _descend(M, roles, T, C, n_test):
    """local_search's descent from the design (T, C) under the relative key, as one of
    its restarts runs it: (key, treatment, control) of the design it ends at."""
    from mlsynth.utils.tbr_helpers.design.local import local_search
    from mlsynth.utils.tbr_helpers.design.search import SearchOutcome

    score = score_split(M[:, sorted(T)].sum(axis=1), M[:, sorted(C)].sum(axis=1),
                        objective="relative", n_test=n_test)
    start = SearchOutcome(k=len(T), treatment=sorted(T), control=sorted(C), unassigned=[], score=score,
                          trace=[], converged=True, evaluations=0)
    best = local_search(M, roles, start, n_test=n_test, restarts=0, seed=0, objective="relative")
    return best.score.key, best.treatment, best.control


def _fits(M, T, C, n_test) -> bool:
    """Whether the library scores the design (T, C) of the raw panel M at all: False
    for an exact fit, which it scores -inf (#1000)."""
    return bool(np.isfinite(score_split(M[:, sorted(T)].sum(axis=1), M[:, sorted(C)].sum(axis=1),
                                        objective="relative", n_test=n_test).key[0]))


def _design(Mc, M, T, C, n_test) -> Optional[_Design]:
    """The design (T, C), C in geo indices: its exact RSS on the centred panel Mc,
    y'y - (c'u)^2 / u'Gu at either slope sign, and the sign of c'u; None for an exact
    fit."""
    y, X, rest = _split(Mc, T)
    Cg = sorted(rest.index(j) for j in C)
    cu, uGu, yy = design_moments(y, X, Cg)
    if not _fits(M, T, C, n_test):
        return None
    return _Design(sorted(C), yy - cu * cu / uGu, 1 if cu >= 0 else -1)


def _local(Mc, M, T, n_test, restarts):
    """The study's RSS descent in group T on y and on -y from ``restarts`` random
    starts: the design each sign's descent ends at (control indices of the group, by
    sign, for that sign's certificate to start from), and the better of the two as a
    _Design (None when both fit exactly)."""
    from .search import incumbent

    y, X, rest = _split(Mc, T)
    starts, best = {}, None
    for s in SIGNS:
        _, C = incumbent(s * y, X, restarts=restarts)
        starts[s] = sorted(C)
        d = _design(Mc, M, T, [rest[j] for j in C], n_test)
        if d is not None and (best is None or d.rss < best.rss):
            best = d
    return starts, best


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


def _group(T) -> tuple:
    """A group as the search keys it: a tuple of Python ints, from a tuple or from a row
    of an int64 block."""
    return tuple(int(v) for v in T)


#: Groups built and floored at a time in _floor_range_screened: 2^18 rows, 12.6 MB of
#: int64 at six treated, and nothing of the range's Python rows but the kept ones.
FLOOR_BLOCK = 1 << 18


def _unrank(r: int, p: int, k: int) -> tuple:
    """The k-subset of range(p) at rank r in lexicographic order (the r-th of
    itertools.combinations(range(p), k)), in exact integers."""
    if not 0 <= r < comb(p, k):
        raise ValueError(f"rank {r} is outside the {comb(p, k)} subsets of {k} of {p}.")
    T, x = [], 0
    for i in range(k):
        while True:
            here = comb(p - x - 1, k - i - 1)    # the subsets with x at position i
            if r < here:
                break
            r -= here
            x += 1
        T.append(x)
        x += 1
    return tuple(T)



def _floor_range_screened(M, lam, p, k, lo, hi, scr, block=None, until=None):
    """_floor_chunk_screened over the k-subsets of range(p) of rank lo to hi - 1, in
    order: built ``block`` at a time from the first subset of each block
    (fastfloor.combinations_from), so only a block and the kept rows are held. Past
    the deadline ``until`` no further block is read, and the subsets from there on are
    neither rows nor dropped."""
    block = block or FLOOR_BLOCK
    rows, dropped, r = [], 0, lo
    while r < hi:
        if deadline.passed(until):
            break
        n = min(block, hi - r)
        groups = fastfloor.combinations_from(np.array(_unrank(r, p, k), dtype=np.int64), n, p)
        part, d = _floor_chunk_screened(M, lam, groups, scr)
        rows += part
        dropped += d
        r += n
    return rows, dropped


def _streamed_floor_rows(M, p, k, lam, pool, n_jobs, scr, until=None):
    """_screened_floor_rows over every k-subset of range(p), without the list: the
    ranks in chunks over the pool, four per worker as there, each worker building its
    own range. The rows, their order and the dropped count are the list's. Past the
    deadline ``until`` each range stops between blocks, and the subsets it did not
    read are the comb(p, k) less the rows and the dropped."""
    total = comb(p, k)
    if pool is None or total < 2:
        return _floor_range_screened(M, lam, p, k, 0, total, scr, until=until)
    n = min(total, 4 * n_jobs)
    size = -(-total // n)
    cuts = [(lo, min(lo + size, total)) for lo in range(0, total, size)]
    parts = _map(pool, _floor_range_screened, [(M, lam, p, k, lo, hi, scr, None, until) for lo, hi in cuts])
    return [r for rows, _ in parts for r in rows], sum(d for _, d in parts)


def _bound_order(row, ell) -> float:
    """The queue's order: the smaller of the row's two floors over its level squared,
    in doubles, the reciprocal of a bound on its best key squared up to the common
    constant; best bound first."""
    fp, fm, T = row
    L = float(ell[list(T)].sum())
    return min(float(fp), float(fm)) / (L * L)


def _floor_rows(M, groups, lam, pool, n_jobs):
    """(floor on y, floor on -y, T) for each group, in the order given: in chunks over
    the pool, four per worker, so a slow chunk does not hold the others."""
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


def _window(pool, n_jobs, rows, admit, job, commit, occupancy=None):
    """Each row ``admit(row)`` accepts, run as fn(*args) for (fn, args, ctx) =
    ``job(row)``, then ``commit(row, ctx, result)``, in queue order.

    Without a pool this is the sequential loop: each row is admitted after the one
    before it is committed. On a pool up to n_jobs rows run at once, and the next row
    is read as soon as a worker is free, so a slow row holds one worker and not the
    others. ``admit`` and ``job`` are read at submission, against every result
    committed by then; a result is committed once every row ahead of it has been, so
    the commits are the sequential loop's order. A worker's error reaches the caller
    when its row's turn to commit comes, and nothing after that row is committed.
    ``occupancy``, when given, is told how many rows run each time the window has
    filled what it can, and 0 when it is done."""
    if pool is None:
        for row in rows:
            if admit(row):
                fn, args, ctx = job(row)
                commit(row, ctx, fn(*args))
        return
    ahead = deque()                         # (row, ctx, future), in queue order, not yet committed
    running = set()
    it = iter(rows)
    left = True
    while True:
        while ahead and ahead[0][2].done():
            row, ctx, fut = ahead.popleft()
            commit(row, ctx, fut.result())
        while left and len(running) < n_jobs:
            row = next(it, it)
            if row is it:
                left = False
            elif admit(row):
                fn, args, ctx = job(row)
                fut = pool.submit(_star, fn, args)
                ahead.append((row, ctx, fut))
                running.add(fut)
        if occupancy is not None:
            occupancy(len(running))
        if not running:                     # a row not committed above has a head still running
            return
        running = wait(running, return_when=FIRST_COMPLETED).not_done


class _Lender:
    """The workers of the search's pool that have nothing to run, lent to a group's
    size loop (slopefree.SPARE): the window sets how many are free, and take(want)
    lends up to ``want`` of those not already lent. Two shared integers, made before
    the pool forks, so every worker reads and takes from the same ones."""

    def __init__(self):
        ctx = multiprocessing.get_context("fork")
        self.free = ctx.Value("i", 0)
        self.lent = ctx.Value("i", 0)

    def set_free(self, n: int) -> None:
        self.free.value = n

    def take(self, want: int) -> int:
        with self.lent.get_lock():
            got = max(0, min(want, self.free.value - self.lent.value))
            self.lent.value += got
            return got

    def give(self, n: int) -> None:
        with self.lent.get_lock():
            self.lent.value -= n


def _optimum_timed(Mc, M, T, n_test, restarts, ceiling, floors, starts, until=None):
    t0 = time.perf_counter()
    res = _optimum(Mc, M, T, n_test, restarts, ceiling, floors, starts, until=until)
    return res, time.perf_counter() - t0


def _optimum(Mc, M, T, n_test, restarts, ceiling, floors, starts, until=None) -> _GroupOutcome:
    """Group T certified against ``ceiling`` (None: its own optimum): each sign whose
    floor is below the ceiling is a certify_slope_free on that sign's series, from its
    pass design in ``starts`` when there is one. A design the library scores -inf is
    excluded and the sign certified again. The group's design is the lower-RSS of the
    signs' designs below the ceiling. Past the deadline ``until`` a sign's branch and
    bound leaves its nodes open (slopefree._branch), and the group is not certified."""
    from .slopefree import NoDesignLeft, certify_slope_free

    with tagged(treatment=tuple(T)):
        t0 = time.perf_counter()
        emit("group", treatment=tuple(T), ceiling=ceiling)
        y, X, rest = _split(Mc, T)
        best, certified, run = None, True, []
        for s, f in zip(SIGNS, floors):
            if ceiling is not None and not Q(f) < ceiling:
                continue
            run.append(s)
            excluded: List[set] = []
            while True:
                kw = {} if starts.get(s) is None else {"start": set(starts[s])}
                try:
                    r = certify_slope_free(s * y, X, restarts=restarts, exclude=excluded, ceiling=ceiling,
                                           deadline=until, **kw)
                except NoDesignLeft:             # no design of this sign is left
                    break
                if not r.certified:
                    certified = False
                    break
                if r.ceiling_held:               # no design of this sign is below the ceiling
                    break
                C = [rest[j] for j in r.controls]
                if not _fits(M, T, C, n_test):   # an exact fit: the library never ranks it
                    excluded.append(set(r.controls))
                    continue
                rss = Q(r.incumbent)
                if best is None or rss < best.rss:
                    best = _Design(sorted(C), rss, s)
                break
        emit("group-done", treatment=tuple(T), certified=certified, found=best is not None, signs=len(run),
             seconds=time.perf_counter() - t0)
        return _GroupOutcome(best, certified, tuple(run))


#: The seconds the search is given by default: two minutes, the v1 default the author
#: set (#965). Past them the search reports its best design with the proven gap.
TIME_LIMIT = 120.0


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
                      n_jobs: int = 1, blank_periods: Optional[int] = 0,
                      time_limit: Optional[float] = TIME_LIMIT, verbose: int = 0,
                      progress=None, log=None, log_level: int = 3) -> TreatmentCertificate:
    """The relative key's best design with ``k`` treatment geos over the columns of
    the raw panel ``M`` (periods by geos, every geo eligible for every role, every
    geo's mean positive), certified (module docstring). ``start`` is a treatment group
    to search first in place of the library's design. ``pass_restarts`` random starts
    for the RSS descent each group that can compete gets on each slope sign before any
    is certified; 0 skips the pass. With ``gap`` g above zero the certificate holds
    within g: no design's key exceeds 1 + g times the reported design's. ``n_jobs``
    processes compute the floors, the pass and the certifications (module docstring);
    1 runs them in this process. The last ``blank_periods`` periods of ``M`` are held
    back from the search, none by default and BLANK_TESTS test lengths when None (see
    BLANK_TESTS); the result reports the A/A test at each of their origins in
    ``blank_aa`` and the four gates on the whole panel. The search stops at
    ``time_limit`` seconds (TIME_LIMIT; None for no limit) and reports the best design
    found, not certified, with ``proven_gap``: no design's key exceeds 1 + proven_gap
    times the reported design's (proven_gap; None when the floors were not all
    computed). ``verbose``, ``progress`` and
    ``log`` report the search's progress (progress.reporting), each group's events
    tagged with the group, from the workers too."""
    with reporting(verbose=verbose, progress=progress, log=log, log_level=log_level):
        out = _certify_treatment_entry(M, k, n_test=n_test, start=start, restarts=restarts,
                                       pass_restarts=pass_restarts, gap=gap, n_jobs=n_jobs,
                                       blank_periods=blank_periods, time_limit=time_limit)
        emit("treatment-done", certified=out.certified, treatment=out.treatment, key=out.key,
             survivors=len(out.survivors), seconds=out.seconds)
        return out


def _certify_treatment_entry(M, k, *, n_test, start, restarts, pass_restarts, gap, n_jobs,
                             blank_periods=0, time_limit=None) -> TreatmentCertificate:
    t0 = time.perf_counter()
    whole = _checked_panel(M)
    B = _checked_blank(blank_periods, n_test, whole.shape[0])
    M = whole[:whole.shape[0] - B]       # the design window: the search never reads a blank period
    _check_levels(M)
    p = M.shape[1]
    if not 1 <= k <= p - 1:
        raise MlsynthDataError(f"a treatment group of {k} geos leaves no control among {p} geos; "
                               f"k has to be between 1 and {p - 1}.")
    if pass_restarts < 0:
        raise MlsynthDataError(f"pass_restarts is {pass_restarts}; it cannot be negative.")
    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise MlsynthDataError(f"n_jobs is {n_jobs!r}; it has to be a positive integer.")
    g = _checked_gap(gap)
    limit = _checked_time_limit(time_limit)
    until = deadline.at(limit)
    emit("treatment", k=k, p=p, groups=comb(p, k), n_jobs=n_jobs)
    if n_jobs == 1:
        out = _certify_treatment(M, k, p, n_test, start, restarts, pass_restarts, g, n_jobs, None, t0, until=until)
    else:
        from . import slopefree
        lender, before = _Lender(), slopefree.SPARE
        slopefree.SPARE = (lender.take, lender.give)       # read in the workers, which fork with it
        try:
            with _pool(n_jobs) as pool:
                out = _certify_treatment(M, k, p, n_test, start, restarts, pass_restarts, g, n_jobs, pool, t0,
                                         lender=lender, until=until)
        finally:
            slopefree.SPARE = before
    out.blank_periods, out.design_periods, out.time_limit = B, M.shape[0], limit
    if out.treatment is not None:
        y = whole[:, out.treatment].sum(axis=1)
        out.gates = gates_of(y, whole, out.controls, n_test)
        out.blank_aa = _blank_aa(y, whole[:, out.controls].sum(axis=1), M.shape[0], n_test)
    out.seconds = time.perf_counter() - t0
    return out


def _checked_time_limit(limit) -> Optional[float]:
    """``limit`` as seconds: None, or a positive finite int or float (not a bool),
    else MlsynthDataError."""
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
        raise MlsynthDataError(f"time_limit is {limit!r}; it has to be a positive, finite number of seconds, "
                               f"or None for no limit.")
    return float(limit)


def proven_gap(bounds, rss_star, level_star, g) -> Optional[Q]:
    """The gap a search proves when the groups in ``bounds``, each (floor on y, floor
    on -y, level), are not resolved and every other group is certified within g. A
    design of a group with level L and floor f has RSS at least f, so its key is at
    most sqrt(q) times the incumbent's, q = L^2 RSS* / (level*^2 f) (the key being
    level over a constant times sqrt(RSS), module docstring). The gap is the largest
    sqrt(q) - 1, rounded up to a rational (root_above), and g where that is smaller;
    None when a floor is not above zero, which bounds nothing."""
    worst = Q(g)
    for fp, fm, L in bounds:
        f = min(Q(fp), Q(fm))
        if not f > 0:
            return None
        q = Q(L) ** 2 * Q(rss_star) / (Q(level_star) ** 2 * f)
        if q > (1 + worst) ** 2:
            worst = root_above(q) - 1
    return worst


#: Blank periods held back from the search when ``blank_periods`` is None, in test
#: lengths: the library's calibrated inference holds back three (mlsynth.utils.
#: tbr_helpers.calibration.resolve_blank_periods), and refuses fewer than two, so the
#: A/A test has whole windows the design never saw. The default holds back none. A
#: design chosen on the design window alone detected a planted 3% lift less often
#: than one chosen on the whole panel (on simulated panels of 40 pre-periods, 43%
#: against 54%; within noise at 104), at the same false-positive rate, since TBR's
#: test fits on the whole pre-period, design window included (README, "Blank periods").
BLANK_TESTS = 3


def _checked_blank(blank_periods, n_test: int, n: int) -> int:
    """B: BLANK_TESTS test lengths when None, none when 0, else at least two test
    lengths. The design window left has to hold the A/A test the search's start
    scores designs with: one test length and MIN_HOLDOUT_PRETEST periods to fit on
    before it, more than the library's MIN_DESIGN_PERIODS (calibration.design_end)."""
    if blank_periods == 0 and type(blank_periods) is int:
        return 0
    if blank_periods is not None and (type(blank_periods) is not int or blank_periods < 0):
        raise MlsynthDataError(f"blank_periods is {blank_periods!r}; it has to be a count of periods, "
                               f"0 for none or None for {BLANK_TESTS} test lengths.")
    B = resolve_blank_periods(BLANK_TESTS * n_test if blank_periods is None else blank_periods, n_test)
    if n - B < n_test + MIN_HOLDOUT_PRETEST:
        raise MlsynthDataError(f"blank_periods={B} leaves {n - B} of {n} periods to design on; the search scores "
                               f"designs with an A/A test that needs {n_test + MIN_HOLDOUT_PRETEST} "
                               f"(a test of {n_test} and {MIN_HOLDOUT_PRETEST} to fit on). Hold back fewer "
                               f"periods, or pass blank_periods=0.")
    return B


def _blank_aa(y, x, design_periods: int, n_test: int) -> List[bool]:
    """The library's A/A test at each blank origin o = D0, D0 + n_test, ...: fit on
    the periods before o, the cumulative effect over [o, o + n_test) against zero.
    Only whole windows are tested."""
    return [aa_test_ok(y[:o + n_test], x[:o + n_test], n_test)
            for o in range(design_periods, y.size - n_test + 1, n_test)]


def _certify_treatment(M, k, p, n_test, start, restarts, pass_restarts, g, n_jobs, pool, t0, lender=None,
                       until=None):
    if start is not None and len(set(start)) != k:
        raise MlsynthDataError(f"the start treatment group has {len(set(start))} geos, not {k}.")
    n = M.shape[0]
    Mc = M - M.mean(axis=0)
    sums = geo_sums(M)
    if not all(s > 0 for s in sums):    # pragma: no cover - _check_levels refuses a mean the doubles show is not positive
        raise MlsynthDataError("every geo's level has to be positive.")
    ell = level_doubles(sums, n)
    levels = {}

    def level(T):
        T = tuple(T)
        if T not in levels:
            levels[T] = group_level(sums, T, n)
        return levels[T]

    out = TreatmentCertificate(k=k, treatment=None, controls=None, key=None, rss=None, level=None, slope=None,
                               certified=True, groups=comb(p, k), screened_out=0, gap=g)
    inc = {}

    def take(T, d: _Design):
        inc.update(T=list(T), C=list(d.controls), rss=d.rss, level=level(T), slope=d.slope)

    def bar():
        return None if not inc else relative_bar(inc["rss"], inc["level"], g)

    if start is not None:
        T0 = sorted(start)
        starts0, d0 = _local(Mc, M, tuple(T0), n_test, restarts)
    else:
        T0, C0 = _start(M, k, n_test, restarts, pool)
        starts0, d0 = {}, _design(Mc, M, T0, C0, n_test)
    if d0 is not None:
        take(T0, d0)
    local = {tuple(T0): starts0}
    found = {} if d0 is None else {tuple(T0): d0}
    left_open = []
    late = []                               # (floor on y, floor on -y, T): neither ruled out nor certified

    def competes(row):
        fp, fm, T = row
        c = relative_ceiling(level(T), bar())
        return c is None or Q(fp) < c or Q(fm) < c

    lam = gram_floor(Mc)                 # rows that cannot compete with the start are dropped where computed
    rows, dropped = _streamed_floor_rows(Mc, p, k, lam, pool, n_jobs, _screen_terms(bar(), ell, k), until=until)
    out.screened_out += dropped
    unread = comb(p, k) - len(rows) - dropped   # floors never computed: out of time (deadline.py)
    rows.sort(key=lambda r: _bound_order(r, ell))      # the best key bound first

    if pass_restarts:                       # the pass: each group's RSS descent on both signs
        for wave in _waves(rows, n_jobs, lambda r: r[2] not in local and competes(r)):
            if deadline.passed(until):
                break
            got = _map(pool, _local, [(Mc, M, T, n_test, pass_restarts) for _, _, T in wave])
            for (_, _, T), (starts, d) in zip(wave, got):   # in queue order
                local[T] = starts
                out.searched += 1
                if d is not None:
                    found[T] = d
                    if not inc or beats(level(T), d.rss, inc["level"], inc["rss"]):
                        take(T, d)
        if inc:
            out.pass_key = _key(M, inc, n_test)
        # the best design found first, then the groups with none, in bound order
        rows.sort(key=lambda r: (0, -float(level(r[2]) ** 2 / found[r[2]].rss)) if r[2] in found and found[r[2]].rss > 0
                  else (1, _bound_order(r, ell)))

    def admit(row):                         # screened against the incumbent as it stands
        if competes(row):
            if deadline.passed(until):      # out of time: bounded by its floors
                late.append(row)
                return False
            return True
        out.screened_out += 1
        return False

    def job(row):                           # given its ceiling as it is submitted
        fp, fm, T = row
        c = relative_ceiling(level(T), bar())
        return _optimum_timed, (Mc, M, T, n_test, restarts, c, (fp, fm), local.get(T, {}), until), (c, (fp, fm))

    def commit(row, ctx, result):           # in queue order
        T, (c, fl), (res, secs) = row[2], ctx, result
        if not res.certified:
            left_open.append((T, c, fl))
        out.survivors.append(dict(treatment=list(T), certified=res.certified, found=res.design is not None,
                                  signs=list(res.signs), ceiling=c, seconds=secs, start=list(T) == T0))
        d = res.design
        if d is not None and (not inc or beats(level(T), d.rss, inc["level"], inc["rss"])):
            take(T, d)
            out.replaced += 1

    occupancy = None if lender is None else (lambda running: lender.set_free(n_jobs - running))
    _window(pool, n_jobs, rows, admit, job, commit, occupancy=occupancy)
    if lender is not None:
        lender.set_free(n_jobs - 1)          # the retries run here, with the pool idle
    for T, c_then, fl in left_open:         # the incumbent only improves, so its ceiling only falls
        c = relative_ceiling(level(T), bar())
        if c is not None and (c_then is None or c < c_then) and not deadline.passed(until):
            out.retried += 1
            res = _optimum(Mc, M, T, n_test, restarts, c, fl, local.get(T, {}), until=until)
            d = res.design
            if d is not None and beats(level(T), d.rss, inc["level"], inc["rss"]):
                take(T, d)
                out.replaced += 1
            if res.certified:
                continue
        elif not competes((fl[0], fl[1], T)):   # not retried, and the incumbent since rules it out
            continue
        late.append((fl[0], fl[1], T))
    out.unresolved = len(late) + unread
    out.certified = out.unresolved == 0
    if inc:
        out.treatment, out.controls = sorted(inc["T"]), sorted(inc["C"])
        out.rss, out.level, out.slope = inc["rss"], inc["level"], inc["slope"]
        out.key = _key(M, inc, n_test)
        out.proven_gap = None if unread else proven_gap([(fp, fm, level(T)) for fp, fm, T in late],
                                                        inc["rss"], inc["level"], g)
    out.timed_out = not out.certified and deadline.passed(until)
    out.seconds = time.perf_counter() - t0
    return out


def _key(M, inc, n_test) -> float:
    """The library's relative key at the incumbent, on the raw panel."""
    return float(score_split(M[:, sorted(inc["T"])].sum(axis=1), M[:, sorted(inc["C"])].sum(axis=1),
                             objective="relative", n_test=n_test).key[0])
