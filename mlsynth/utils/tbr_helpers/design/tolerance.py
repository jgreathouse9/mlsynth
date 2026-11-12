"""A treatment group that stays close to the population (``population_tolerance``).

TBR estimates the effect on the treatment group it was handed. A search that
ranks groups by how well they can be measured -- the relative MDE above all --
prefers groups whose sales move least with the market-wide trend, since those
are the easiest to forecast. When a campaign's lift varies with that same
sensitivity, the chosen group's effect is then a biased stand-in for the
effect the campaign would have had everywhere. The tolerance keeps the group
close to the population on the one thing the pre-period shows about it: the
shape of its path.

The anchor distance of a treatment group ``T`` over the scoring window is

    d(T) = sqrt(mean_t (y_T[t] / mean(y_T) - p[t] / mean(p))^2),

``y_T`` the treated aggregate and ``p`` the sum of every geo in the panel, so
it is the RMS gap between the group's path and the population's, each scaled
to its own mean. It is MAREX's treated-side fit to the population mean with
0/1 weights. Under ``population_tolerance = C`` a group of size ``k`` is
admissible only when ``d(T) <= C d_min(k)``, ``d_min(k)`` the smallest distance
over the groups of size ``k`` that some design can use: the business rules
admit them and a control group is left. The bound is relative to what is
achievable at each size, not a fixed share of the path, because the closest
achievable distance differs by an order of magnitude across panels.

``d_min(k)`` is a search of its own. When the candidate groups number at most
:data:`ENUMERATION_LIMIT` every one is scored, in vectorised blocks. Above it a
one-swap descent runs from a forward-greedy start and from
:data:`SEARCH_STARTS` seeded random starts and keeps the closest group it
reaches, which is an upper bound on ``d_min(k)``: the threshold is then at
least as loose as the exact one, never tighter. Each size records which method
produced it. The group that attains ``d_min(k)`` is the witness: it is
admitted at every ``C >= 1``, which is what lets the design search always
return a design at that size.

Scoring many groups uses one quadratic form. With ``mu_j`` geo ``j``'s mean and
``r_j = y_j - mu_j p / mean(p)`` its residual from its own mean times the
population's path, ``y_T - mean(y_T) p / mean(p)`` is the sum of the ``r_j``
over ``T``, so ``d(T)^2 = u' H u / (mu' u)^2`` with ``H = R'R / n`` and ``u``
the group's indicator. No term is a difference of large numbers. The witness's
distance is then recomputed from the definition, which is the number every
comparison reads.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import chain, combinations, islice
from typing import (Any, Dict, FrozenSet, List, Optional, Sequence, Tuple,
                    TYPE_CHECKING)

import numpy as np

from ....exceptions import MlsynthConfigError, MlsynthDataError
from .objective import check_positive_levels
from .rules import CONTROL, TREATMENT

if TYPE_CHECKING:  # pragma: no cover - read by type checkers only
    from .rules import DesignRules

#: Candidate groups of one size are enumerated when they number at most this
#: many, and searched otherwise. The 3.9 million groups of four among 100 geos
#: score in about 1.4 seconds.
ENUMERATION_LIMIT = 2_000_000

#: Groups scored per vectorised block of the enumeration.
BLOCK = 100_000

#: A group is admitted when its distance is within this relative slack of the
#: threshold. Two groups equally close in exact arithmetic, a geo and a scaled
#: copy of it, land one rounding apart, and both are the closest at C = 1.
TIE_RTOL = 1e-9

#: Random starts the searched path descends from, besides the forward greedy.
#: Over the GeoLift panel and six simulated panels of 60 geos at sizes three
#: and four, 8 starts reached the enumerated minimum in 5 of 14 cases (largest
#: gap 12%), 32 in 12 of 14 (2%) and 64 in all 14, at about 0.07 s a size.
SEARCH_STARTS = 64

#: Draws a random start makes for a usable group before it gives up.
DRAWS_PER_START = 50

#: Seed of the random starts; size k draws from (SEARCH_SEED, k).
SEARCH_SEED = 0


# --------------------------------------------------------------------------- #
# the distance
# --------------------------------------------------------------------------- #
def scaled_gap(treated: np.ndarray, population: np.ndarray) -> float:
    """RMS gap between two paths, each scaled to its own mean."""
    y = np.asarray(treated, dtype=float)
    p = np.asarray(population, dtype=float)
    return float(np.sqrt(np.mean((y / y.mean() - p / p.mean()) ** 2)))


def population_path(window: np.ndarray) -> np.ndarray:
    """The population's path: the sum of every geo, per period."""
    return np.asarray(window, dtype=float).sum(axis=1)


def anchor_distance(window: np.ndarray, group: Sequence[int]) -> float:
    """``d(T)`` for the geos ``group`` index, over ``window`` (periods by geos)."""
    window = np.asarray(window, dtype=float)
    return scaled_gap(window[:, sorted(group)].sum(axis=1),
                      population_path(window))


def population_distance(window: np.ndarray, group: Sequence[int]) -> Optional[float]:
    """``d(T)`` when both levels are positive, ``None`` when either is not.

    Each design reports it whether or not the tolerance is on, and a panel the
    tolerance would refuse has no meaningful distance to report.
    """
    window = np.asarray(window, dtype=float)
    level = window[:, sorted(group)].sum(axis=1).mean()
    if not (level > 0.0 and population_path(window).mean() > 0.0):
        return None
    return anchor_distance(window, group)


def check_population_levels(window: np.ndarray,
                            eligibility: Sequence[FrozenSet[str]],
                            labels: Sequence[Any]) -> None:
    """Refuse a non-positive level the distance would divide by.

    Every treatment-eligible geo needs a positive mean, which makes every
    candidate group's level positive, and so does the population.
    """
    window = np.asarray(window, dtype=float)
    check_positive_levels(
        window, [j for j, r in enumerate(eligibility) if TREATMENT in r], labels,
        need="population_tolerance compares each treatment group's path, "
             "scaled to its mean, with the population's, which needs a "
             "positive level",
        remedy="Exclude them from the treatment group or drop "
               "population_tolerance.")
    if not population_path(window).mean() > 0.0:
        means = window.mean(axis=0)
        bad = [labels[j] for j in range(window.shape[1]) if not means[j] > 0.0]
        raise MlsynthDataError(
            f"population_tolerance scales the population's path, the sum of "
            f"every geo, by its mean, and that mean is not positive on the "
            f"scoring window; geo(s) {bad} have a mean outcome that is not "
            f"positive. Drop them from the panel or drop population_tolerance.")


# --------------------------------------------------------------------------- #
# d_min
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Anchor:
    """The closest usable group of one size, and how it was found."""

    k: int
    d_min: float
    witness: Tuple[int, ...]
    method: str                 # "enumerated" or "searched"
    n_candidate_groups: int


class _Problem:
    """The quadratic form and the group sets one window and eligibility give."""

    def __init__(self, window: np.ndarray, eligibility: Sequence[FrozenSet[str]],
                 rules: Optional["DesignRules"]) -> None:
        Y = np.asarray(window, dtype=float)
        p = population_path(Y)
        shape = p / p.mean()
        self.mu = Y.mean(axis=0)
        R = Y - np.outer(shape, self.mu)
        self.H = R.T @ R / Y.shape[0]
        self.rules = rules
        self.forced = [j for j, r in enumerate(eligibility)
                       if r == frozenset({TREATMENT})]
        self.free = [j for j, r in enumerate(eligibility)
                     if TREATMENT in r and j not in self.forced]
        control = np.zeros(Y.shape[1], dtype=bool)
        control[[j for j, r in enumerate(eligibility) if CONTROL in r]] = True
        self.control = control

    def d2(self, groups: np.ndarray) -> np.ndarray:
        """``d^2`` of every row of ``groups`` (m by k column indices)."""
        k = groups.shape[1]
        q = np.zeros(groups.shape[0])
        for a in range(k):
            for b in range(k):
                q += self.H[groups[:, a], groups[:, b]]
        s = self.mu[groups].sum(axis=1)
        return np.maximum(q, 0.0) / (s * s)

    def leaves_control(self, groups: np.ndarray) -> np.ndarray:
        """Rows that leave some control-eligible geo outside the group."""
        return self.control[groups].sum(axis=1) < self.control.sum()

    def usable(self, group: Sequence[int]) -> bool:
        """The rules admit the group and some design can still use it."""
        if self.rules is None:
            return bool(self.leaves_control(np.asarray([list(group)]))[0])
        return self.rules.admits(group) and self.rules.extendable(group, len(group))

    def first_usable(self, groups: np.ndarray, d2: np.ndarray,
                     below: float) -> Optional[int]:
        """The usable row of smallest ``d2`` under ``below``, or None."""
        d2 = np.where(self.leaves_control(groups), d2, np.inf)
        if self.rules is None:
            i = int(np.argmin(d2))
            return i if d2[i] < below else None
        for i in np.argsort(d2, kind="stable"):
            if not d2[i] < below:
                return None
            if self.usable(groups[i].tolist()):
                return int(i)
        return None


def closest_group(window: np.ndarray, eligibility: Sequence[FrozenSet[str]],
                  rules: Optional["DesignRules"], k: int, *,
                  limit: int = ENUMERATION_LIMIT, starts: int = SEARCH_STARTS,
                  seed: int = SEARCH_SEED, block: int = BLOCK) -> Optional[Anchor]:
    """``d_min(k)`` and its witness, or None when no group of size ``k`` is usable.

    A group holds every forced geo (eligible for treatment alone) and draws the
    rest from the treatment-eligible geos; it is usable when ``rules`` admit it
    (None admits every group) and a control group is left. The candidates are
    enumerated when they number at most ``limit``, and searched otherwise.
    """
    P = _Problem(window, eligibility, rules)
    r = k - len(P.forced)
    if r < 0 or r > len(P.free):
        return None
    total = math.comb(len(P.free), r)
    if total <= limit:
        found = _enumerate(P, r, block)
        method = "enumerated"
    else:
        found = _search(P, r, k, starts, seed)
        method = "searched"
    if found is None:
        return None
    witness = tuple(sorted(found))
    return Anchor(k=k, d_min=anchor_distance(window, witness), witness=witness,
                  method=method, n_candidate_groups=total)


def _with_forced(P: _Problem, rows: np.ndarray) -> np.ndarray:
    F = np.asarray(P.forced, dtype=np.intp)
    return np.hstack([np.broadcast_to(F, (rows.shape[0], F.size)), rows])


def _enumerate(P: _Problem, r: int, block: int) -> Optional[List[int]]:
    """The usable group of smallest distance, over every candidate."""
    best, best_d2 = None, np.inf
    combos = combinations(P.free, r)
    while True:
        flat = np.fromiter(chain.from_iterable(islice(combos, block)),
                           dtype=np.intp)
        rows = flat.reshape(-1, r) if r else np.empty((1, 0), dtype=np.intp)
        if rows.shape[0] == 0:
            break
        groups = _with_forced(P, rows)
        d2 = P.d2(groups)
        i = P.first_usable(groups, d2, best_d2)
        if i is not None:
            best, best_d2 = groups[i].tolist(), float(d2[i])
        if r == 0:
            break
    return best


def _search(P: _Problem, r: int, k: int, starts: int,
            seed: int) -> Optional[List[int]]:
    """The closest group a one-swap descent reaches from several starts."""
    rng = np.random.default_rng([seed, k])
    origins = []
    greedy = _forward_greedy(P, r, k)
    if greedy is not None:
        origins.append(greedy)
    for _ in range(starts):
        for _ in range(DRAWS_PER_START):
            draw = sorted(P.forced + rng.choice(P.free, size=r,
                                                replace=False).tolist())
            if P.usable(draw):
                origins.append(draw)
                break
    best, best_d2 = None, np.inf
    for origin in origins:
        group, d2 = _descend(P, origin)
        if d2 < best_d2:
            best, best_d2 = group, d2
    return best


def _forward_greedy(P: _Problem, r: int, k: int) -> Optional[List[int]]:
    """Add, one at a time, the geo that leaves the group closest."""
    group = list(P.forced)
    for _ in range(r):
        options = [g for g in P.free if g not in group
                   and (P.rules is None or P.rules.extendable(group + [g], k))]
        if not options:
            return None
        rows = np.asarray([sorted(group + [g]) for g in options], dtype=np.intp)
        group = rows[int(np.argmin(P.d2(rows)))].tolist()
    return group if P.usable(group) else None


def _descend(P: _Problem, group: List[int]) -> Tuple[List[int], float]:
    """Take the closest usable one-swap neighbour until none is closer."""
    current = sorted(group)
    here = float(P.d2(np.asarray([current], dtype=np.intp))[0])
    forced = set(P.forced)
    while True:
        inside = [g for g in current if g not in forced]
        outside = [g for g in P.free if g not in current]
        if not inside or not outside:
            return current, here
        rows = np.asarray([sorted(set(current) - {a} | {b})
                           for a in inside for b in outside], dtype=np.intp)
        d2 = P.d2(rows)
        i = P.first_usable(rows, d2, here)
        if i is None:
            return current, here
        current, here = rows[i].tolist(), float(d2[i])


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #
class PopulationTolerance:
    """``d(T) <= C d_min(|T|)`` over one scoring window.

    ``rules`` are the business rules ``d_min`` is taken under; None admits
    every group. A tolerance they already carry is set aside, since ``d_min``
    is the closest group the business rules admit and reading the tolerance
    there would ask for ``d_min`` inside its own computation. ``d_min`` is
    computed for a size the first time it is asked for and kept, and so is each
    group's distance.
    """

    def __init__(self, window: np.ndarray, eligibility: Sequence[FrozenSet[str]],
                 rules: Optional["DesignRules"], factor: float, *,
                 labels: Optional[Sequence[Any]] = None,
                 limit: int = ENUMERATION_LIMIT) -> None:
        if not (math.isfinite(factor) and factor >= 1.0):
            raise MlsynthConfigError(
                f"population_tolerance must be a finite number of at least 1, "
                f"the closest achievable distance itself; got {factor!r}.")
        window = np.asarray(window, dtype=float)
        if window.ndim != 2:
            raise MlsynthDataError(
                f"the scoring window has to be periods by geos; got shape "
                f"{window.shape}.")
        if len(eligibility) != window.shape[1]:
            raise MlsynthDataError(
                f"eligibility covers {len(eligibility)} geo(s) but the panel "
                f"has {window.shape[1]} geo(s).")
        check_population_levels(window, eligibility,
                                list(range(window.shape[1])) if labels is None
                                else list(labels))
        self.window = window
        self.eligibility = list(eligibility)
        self.rules = (rules if rules is None or rules.tolerance is None
                      else replace(rules, tolerance=None))
        self.factor = float(factor)
        self.limit = int(limit)
        self._anchors: Dict[int, Optional[Anchor]] = {}
        self._distances: Dict[FrozenSet[int], float] = {}

    def distance(self, group: Sequence[int]) -> float:
        """``d(T)``, kept per group."""
        key = frozenset(int(j) for j in group)
        hit = self._distances.get(key)
        if hit is None:
            hit = anchor_distance(self.window, sorted(key))
            self._distances[key] = hit
        return hit

    def anchor(self, k: int) -> Optional[Anchor]:
        """The closest usable group of size ``k``, or None when there is none."""
        if k not in self._anchors:
            self._anchors[k] = closest_group(self.window, self.eligibility,
                                             self.rules, k, limit=self.limit)
        return self._anchors[k]

    def threshold(self, k: int) -> Optional[float]:
        """``C d_min(k)``, or None when no group of size ``k`` is usable."""
        a = self.anchor(k)
        return None if a is None else self.factor * a.d_min

    def admits(self, group: Sequence[int]) -> bool:
        """``d(T)`` within the threshold of its own size, up to :data:`TIE_RTOL`."""
        bound = self.threshold(len(group))
        return bound is not None and \
            self.distance(group) <= bound * (1.0 + TIE_RTOL)
