"""The floor-guided search over treatment groups, uncertified (#965).

certify_treatment (treatment.py) proves its design optimal; most of its time on
large panels goes to the certificates of the few groups the floors leave. This
search keeps everything before the certificates and reports how far its design can
be from the optimum, in place of a proof:

1. The start. The library's greedy and local search (treatment._start), or the
   group the caller gives, is searched first, from the library's design when there
   is one, and its best design is the first incumbent.
2. The floors. Every treatment group gets the exact floor on the RSS of its
   designs (treatment._floor_rows), and the groups are read in floor order, best
   first. A group whose floor shows that none of its designs can beat the incumbent
   (treatment.can_compete) is ruled out. The incumbent only improves, so a group
   ruled out stays ruled out.
3. The group search. Every other group gets group_search: the study's RSS descent
   (treatment._local) and, when the RSS descent's design fails one of the key's
   gates, a descent on the key itself from that design, the library's local search
   with the treatment group fixed. The RSS descent's design is the group's least
   RSS design it reaches; when that design fails a gate the group's best passing
   design is a different one, and the key's descent moves to it. A start design,
   when given, is descended from as well.

The bound. A group searched whose floor can still beat the reported design is
open: its own optimum may be better than what the search found in it. Every other
group either cannot beat the design or holds it. So no design of the size is better
than the reported one unless it lies in an open group, and every design of an open
group has an RSS at least the group's floor. When no open group's correlation bound
can round above the design's class (round(corr, 2)), the optimum is in that class,
the key compares designs of one class by RSS alone (the required impact is a
constant times sqrt(RSS / (n - 2)), the window being the same; treatment.py's
docstring), and so the optimum's RSS is at least ``rss_bound``, the least floor of
the open groups and the design's own RSS, and its required impact at least
``impact_bound``, the design's impact times sqrt(rss_bound / rss) (derived here).
When some open group could round above the class, no impact bound follows and
``above_class`` says so. With no open group the design is the optimum.

The floors, the screen and the class edges are treatment.py's, exact; the impact
bound is computed in doubles from the exact ratio, a report, not a prune.

On GeoLift, on one process with 50 restarts, the search took 4.6, 8.0 and 32.6 s at
three, four and five treated geos and reported each size's certified optimum
(README, "The floor-guided search"; results/guided.log).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from fractions import Fraction as Q
from itertools import combinations
from math import comb
from typing import List, Optional, Sequence, Tuple

import numpy as np

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.tbr_helpers.design.local import local_search
from mlsynth.utils.tbr_helpers.design.objective import score_split
from mlsynth.utils.tbr_helpers.design.search import SearchOutcome

from . import treatment
from .progress import emit, reporting, tagged
from .treatment import EDGE, SLACK, _competes, _design, _local, _ratio, _start, _yy, can_compete, gram_floor


@dataclass
class GroupDesign:
    """The best passing design group_search found in one treatment group."""

    controls: Optional[List[int]]          # geo indices; None when no design found passes
    key: Optional[tuple]
    rss: Optional[Q]


@dataclass
class GuidedResult:
    """The outcome of :func:`guided_treatment` for one treatment size."""

    k: int
    treatment: Optional[List[int]]         # None when no design found passes
    controls: Optional[List[int]]
    key: Optional[tuple]
    rss: Optional[Q]
    passing: bool
    groups: int
    screened_out: int                       # groups the floors ruled out
    searched: int                           # groups given group_search
    searched_groups: List[Tuple[int, ...]] = field(default_factory=list)
    open: List[Tuple[int, ...]] = field(default_factory=list)   # searched groups that can still beat the design
    rss_bound: Optional[Q] = None           # no design in the design's class has a smaller RSS
    above_class: bool = True                # an open group could round above the design's class
    impact_bound: Optional[float] = None    # no design has a smaller required impact (when not above_class)
    proven: bool = False                    # no group is open: the design is the optimum
    seconds: float = 0.0


def _floor_rows(M, groups, lam):
    """(floor, y'y, T) for each group, by certify_treatment's floor function."""
    return treatment._floor_rows(M, groups, lam, None, 1)


def _key_descent(M, T, C, n_test):
    """The library's local search on the key, the treatment group fixed at T, from
    the controls C (geo indices), no restarts: its design as treatment._design
    reports one."""
    p = M.shape[1]
    roles = [frozenset({"treatment"}) if j in T else frozenset({"control", "unassigned"}) for j in range(p)]
    y = M[:, sorted(T)].sum(axis=1)
    score = score_split(y, M[:, sorted(C)].sum(axis=1), objective="reference", n_test=n_test)
    start = SearchOutcome(k=len(T), treatment=sorted(T), control=sorted(C),
                          unassigned=sorted(set(range(p)) - set(T) - set(C)), score=score,
                          trace=[], converged=True, evaluations=0)
    out = local_search(M, roles, start, n_test=n_test, restarts=0, seed=0)
    Cg, key, rss, rest = _design(M, T, out.control, n_test)
    return [rest[j] for j in Cg], key, rss


def group_search(M, T: Sequence[int], n_test: int, restarts: int,
                 start: Optional[Sequence[int]] = None) -> GroupDesign:
    """The best passing design found in treatment group T: the RSS descent's from
    ``restarts`` random starts, and, when that design fails a gate, the key's descent
    from it; the key's descent from ``start`` (geo indices) too, when given (module
    docstring)."""
    T = sorted(T)
    C, key, rss, rest = _local(M, T, n_test, restarts)
    Cg = [rest[j] for j in C]
    best = GroupDesign(Cg, key, rss) if key is not None else GroupDesign(None, None, None)
    froms = [] if key is not None else [Cg]
    if start:
        froms.append(sorted(start))
    for C0 in froms:
        C2, key2, rss2 = _key_descent(M, T, C0, n_test)
        if key2 is not None and (best.key is None or key2 > best.key):
            best = GroupDesign(sorted(C2), key2, rss2)
    return best


def guided_treatment(M, k: int, *, n_test: int, restarts: int = 50,
                     start: Optional[Sequence[int]] = None, verbose: int = 0, progress=None,
                     log=None, log_level: int = 3) -> GuidedResult:
    """The floor-guided search's design with ``k`` treatment geos over the columns of
    ``M`` (centred, periods by geos, every geo eligible for every role), and the bound
    the floors give on its distance from the optimum (module docstring). ``verbose``,
    ``progress`` and ``log`` report each group searched (progress.reporting)."""
    with reporting(verbose=verbose, progress=progress, log=log, log_level=log_level):
        out = _guided_treatment(M, k, n_test=n_test, restarts=restarts, start=start)
        emit("treatment-done", certified=out.proven, treatment=out.treatment, rss=out.rss,
             passing=out.passing, survivors=out.searched, seconds=out.seconds)
        return out


def _guided_treatment(M, k, *, n_test, restarts, start) -> GuidedResult:
    t0 = time.perf_counter()
    M = treatment._checked_panel(M)
    p = M.shape[1]
    if not 1 <= k <= p - 1:
        raise MlsynthDataError(f"a treatment group of {k} geos leaves no control among {p} geos; "
                               f"k has to be between 1 and {p - 1}.")
    if restarts < 1:
        raise MlsynthDataError(f"restarts is {restarts}; the RSS descent needs at least one start.")
    if start is not None and len(set(start)) != k:
        raise MlsynthDataError(f"the start treatment group has {len(set(start))} geos, not {k}.")
    out = GuidedResult(k=k, treatment=None, controls=None, key=None, rss=None, passing=False,
                       groups=comb(p, k), screened_out=0, searched=0)

    emit("treatment", k=k, p=p, groups=comb(p, k), n_jobs=1)

    def search(T, C=None):
        with tagged(treatment=tuple(T)):
            emit("group", treatment=tuple(T), ceiling=None)
            t1 = time.perf_counter()
            g = group_search(M, T, n_test, restarts, start=C)
            emit("group-done", treatment=tuple(T), certified=None, passing=g.key is not None, rounds=None,
                 seconds=time.perf_counter() - t1)
        out.searched += 1
        out.searched_groups.append(tuple(T))
        if g.key is not None and (out.key is None or g.key > out.key):
            out.treatment, out.controls, out.key, out.rss, out.passing = list(T), g.controls, g.key, g.rss, True

    if start is not None:
        T0, C0 = sorted(start), None
    else:
        T0, C0 = _start(M, k, n_test, restarts)
    search(T0, C0)

    rows = _floor_rows(M, list(combinations(range(p), k)), gram_floor(M))
    rows.sort(key=_ratio)
    yys = {}

    def exact(T):
        if T not in yys:
            yys[T] = _yy(M, T)
        return yys[T]

    def competes(floor, yy, T):
        r = Q(out.key[4]) if out.key is not None else None
        return _competes(floor, yy, lambda: exact(T), r, out.rss)

    floors = {}
    for floor, yy, T in rows:
        floors[T] = floor
        if T == tuple(T0):
            continue
        if competes(floor, yy, T):
            search(T)
        else:
            out.screened_out += 1

    r = Q(out.key[4]) if out.key is not None else None
    out.open = [T for T in out.searched_groups if can_compete(exact(T), floors[T], r, out.rss)]
    out.rss_bound = min([floors[T] for T in out.open] + ([out.rss] if out.rss is not None else []))
    if out.passing:
        up = r + EDGE - SLACK
        out.above_class = any(floors[T] < exact(T) * (1 - up * up) for T in out.open)
        if not out.above_class:
            out.impact_bound = (1 / out.key[5]) * math.sqrt(float(out.rss_bound / out.rss))
        out.proven = not out.open
    out.seconds = time.perf_counter() - t0
    return out
