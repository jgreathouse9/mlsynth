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
matrix's: it is the least Rayleigh quotient over vectors supported on the
submatrix's indices, a subset of the vectors the full matrix's is taken over
(derived here).

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

"""
from __future__ import annotations

import math
import time
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
from .gated import certify_gated

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


def rss_floor(M, T: Sequence[int], lam: Optional[Q]) -> Q:
    """An exact lower bound on the RSS of every positive-slope design whose treatment
    group is T (module docstring); 0 when lam is None."""
    if lam is None:
        return Q(0)
    y, X, _ = _split(M, T)
    yy = sum(Q(v) ** 2 for v in y.tolist())
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
    survivors: List[dict] = field(default_factory=list)
    seconds: float = 0.0


def _start(M, k, n_test, restarts):
    """The library's greedy search and local search (#964), every geo eligible for
    every role: the incumbent's treatment group at size k."""
    from mlsynth.utils.tbr_helpers.design.local import local_search
    from mlsynth.utils.tbr_helpers.design.search import greedy_search

    roles = [frozenset({"treatment", "control", "unassigned"})] * M.shape[1]
    out = greedy_search(M, roles, max_treatment_size=k, n_test=n_test, objective="reference")
    best = local_search(M, roles, out[-1], n_test=n_test, restarts=restarts, seed=0)
    return best.treatment


def _optimum(M, T, n_test, restarts, ceiling=None):
    y, X, rest = _split(M, T)
    res = certify_gated(y, X, n_test=n_test, restarts=restarts, ceiling=ceiling)
    C = [rest[j] for j in res.controls] if res.passing else None
    return res, C


def certify_treatment(M, k: int, *, n_test: int, start: Optional[Sequence[int]] = None,
                      restarts: int = 50) -> TreatmentCertificate:
    """The reference key's best design with ``k`` treatment geos over the columns of
    ``M`` (centred, periods by geos, every geo eligible for every role), certified
    (module docstring)."""
    t0 = time.perf_counter()
    M = np.asarray(M, dtype=float)
    p = M.shape[1]
    if not 1 <= k <= p - 1:
        raise MlsynthDataError(f"a treatment group of {k} geos leaves no control among {p} geos; "
                               f"k has to be between 1 and {p - 1}.")
    if start is not None and len(set(start)) != k:
        raise MlsynthDataError(f"the start treatment group has {len(set(start))} geos, not {k}.")
    T0 = sorted(start) if start is not None else sorted(_start(M, k, n_test, restarts))
    out = TreatmentCertificate(k=k, treatment=None, controls=None, key=None, rss=None, passing=False,
                               certified=True, groups=comb(p, k), screened_out=0)

    def take(T, res, C):
        out.treatment, out.controls, out.rss, out.passing = list(T), sorted(C), res.rss, True
        out.key = res.key

    t1 = time.perf_counter()
    res, C = _optimum(M, T0, n_test, restarts)
    out.certified &= res.certified
    out.survivors.append(dict(treatment=list(T0), certified=res.certified, passing=res.passing,
                              key=res.key, rounds=len(res.rounds), stopped=res.stopped,
                              seconds=time.perf_counter() - t1, start=True))
    if res.passing:
        take(T0, res, C)
    lam = gram_floor(M)
    rows = []
    for T in combinations(range(p), k):
        if list(T) == T0:
            continue
        y, _, _ = _split(M, T)
        yy = sum(Q(v) ** 2 for v in y.tolist())
        rows.append((rss_floor(M, T, lam), yy, T))
    rows.sort(key=lambda r: r[0] / r[1])  # the highest correlation bound first

    def class_of():
        return (None, None) if out.key is None else (Q(out.key[4]), out.rss)

    for floor, yy, T in rows:
        if not can_compete(yy, floor, *class_of()):
            out.screened_out += 1
            continue
        t1 = time.perf_counter()
        res, C = _optimum(M, T, n_test, restarts, ceiling=ceiling(yy, *class_of()))
        out.certified &= res.certified
        out.survivors.append(dict(treatment=list(T), certified=res.certified, passing=res.passing,
                                  key=res.key, rounds=len(res.rounds), stopped=res.stopped,
                                  seconds=time.perf_counter() - t1))
        if res.passing and (out.key is None or res.key > out.key):
            take(T, res, C)
            out.replaced += 1
    out.seconds = time.perf_counter() - t0
    return out
