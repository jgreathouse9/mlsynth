"""A local search after Algorithm 1's greedy, scored in batch (#962).

The greedy grows the treatment group one geo at a time and matches each size's
control group by single toggles, stopping at the first design no single toggle
improves. Two moves it never makes decide how far that stops from the best
design: swapping one control geo for another, which a pair of toggles reaches
only through a worse design between them, and replacing a treated geo once it
has been added. On the GeoLift markets, from the greedy's own designs, those two
moves alone shrink the smallest detectable impact by 17% at one treated geo.

This search starts from the greedy's design at one treatment size and from
``restarts`` random designs of the same size, and descends from each. Every step
scores the start's whole neighbourhood -- each control add, drop and swap, and
each treatment swap -- in one call to
:func:`~mlsynth.utils.tbr_helpers.design.batched.score_designs`, takes the best
neighbour, and keeps it only when the direct fit
(:func:`~mlsynth.utils.tbr_helpers.design.objective.score_split`) confirms that
it improves the key. The descent stops at the first design whose best neighbour
does not, which is a local optimum under these moves. The best descent wins, so
the result is never worse than the greedy's design.

Eligibility is Au's ``A_i`` and binds every move. A geo eligible for treatment
alone is in every treatment group and is never swapped out. A geo outside the
treatment group sits in the control group when control is its only other role,
outside the experiment when control is not among its roles, and either way when
both are. A geo leaving the treatment group takes whichever of those it allows.

This is a heuristic like the greedy, with a larger move set: it carries no
optimality guarantee. On a ten-geo subpanel of the GeoLift markets, where every
design can be scored, it reached the best design at three of four treatment
sizes.
"""
from __future__ import annotations

from typing import FrozenSet, List, Sequence, Set, Tuple

import numpy as np

from ....exceptions import MlsynthDataError
from .batched import design_cache, score_designs
from .objective import score_split
from .search import CONTROL, TREATMENT, UNASSIGNED, SearchOutcome

Design = Tuple[List[int], Set[int]]


def _sides(roles: Sequence[FrozenSet[str]]):
    """The role classes the moves read."""
    forced = {j for j, r in enumerate(roles) if r == frozenset({TREATMENT})}
    treatable = {j for j, r in enumerate(roles) if TREATMENT in r}
    pinned = {j for j, r in enumerate(roles)
              if CONTROL in r and UNASSIGNED not in r}
    free = {j for j, r in enumerate(roles) if CONTROL in r and UNASSIGNED in r}
    return forced, treatable, pinned, free


def neighbourhood(treatment: Sequence[int], control: Set[int],
                  roles: Sequence[FrozenSet[str]]) -> List[Design]:
    """Every design one move from ``(treatment, control)`` that ``roles`` allow.

    The moves are a free control geo toggled in or out, a free control geo
    swapped for a free geo outside the experiment, and a treated geo that is not
    forced swapped for any other treatment-eligible geo. A geo entering the
    treatment group leaves the control group; a geo leaving it goes to the
    control group when it must, outside when it must, and to each in turn when
    it may. No move empties the control group.
    """
    forced, treatable, pinned, free = _sides(roles)
    T = sorted(treatment)
    C = set(control)
    inT = set(T)
    out: List[Design] = []
    outside_free = [g for g in sorted(free) if g not in inT and g not in C]
    for g in sorted(free):
        if g in inT:
            continue
        C2 = C ^ {g}
        if C2:
            out.append((T, C2))
    for b in sorted(C & free):
        for a in outside_free:
            out.append((T, (C - {b}) | {a}))
    for t in T:
        if t in forced:
            continue
        r = roles[t]
        landings = []
        if t in pinned:
            landings.append(True)
        else:
            if UNASSIGNED in r:
                landings.append(False)
            if CONTROL in r:
                landings.append(True)
        for g in sorted(treatable - inT):
            T2 = sorted((inT - {t}) | {g})
            for to_control in landings:
                C2 = (C - {g}) | ({t} if to_control else set())
                if C2:
                    out.append((T2, C2))
    return out


def _random_design(rng: np.random.Generator, roles, k: int) -> Design | None:
    """A uniformly drawn design of size ``k`` that ``roles`` allow, or None."""
    forced, treatable, pinned, free = _sides(roles)
    pool = sorted(treatable - forced)
    need = k - len(forced)
    if need < 0 or need > len(pool):  # pragma: no cover - the greedy's own size
        return None                   # is feasible, so a start of it is too
    T = sorted(forced | set(rng.choice(pool, size=need, replace=False).tolist())
               if need else forced)
    inT = set(T)
    C = {g for g in pinned if g not in inT}
    choices = [g for g in sorted(free) if g not in inT]
    if choices:
        size = int(rng.integers(0 if C else 1, len(choices) + 1))
        C |= set(rng.choice(choices, size=size, replace=False).tolist())
    return (T, C) if C else None


def _rows(N: int, designs: Sequence[Design]):
    UT = np.zeros((len(designs), N))
    UC = np.zeros((len(designs), N))
    for i, (T, C) in enumerate(designs):
        UT[i, list(T)] = 1.0
        UC[i, list(C)] = 1.0
    return UT, UC


def local_search(y_matrix: np.ndarray, eligibility: Sequence[FrozenSet[str]],
                 start: SearchOutcome, *, n_test: int, restarts: int,
                 seed: int) -> SearchOutcome:
    """Descend from ``start`` and from ``restarts`` random designs of its size.

    Returns the best design reached, as a :class:`SearchOutcome` whose trace
    opens at ``start``'s key and records every accepted step of the winning
    descent, and whose ``evaluations`` counts every design scored, batched or
    direct.
    """
    if restarts < 0:
        raise MlsynthDataError(f"restarts is {restarts}; it cannot be negative.")
    y = np.asarray(y_matrix, dtype=float)
    N = y.shape[1]
    roles = list(eligibility)
    cache = design_cache(y, n_test=n_test)
    spent = 0

    def direct(T, C):
        nonlocal spent
        spent += 1
        return score_split(y[:, sorted(T)].sum(axis=1), y[:, sorted(C)].sum(axis=1),
                           objective="reference", n_test=n_test)

    def descend(T, C):
        nonlocal spent
        current = direct(T, C)
        trace = [current.key]
        while True:
            moves = neighbourhood(T, C, roles)
            if not moves:  # pragma: no cover - a size the greedy reached
                return T, C, current, trace  # always has a neighbour
            keys = score_designs(cache, *_rows(N, moves))
            spent += len(moves)
            best = int(np.lexsort(keys.T[::-1])[-1])
            T2, C2 = moves[best]
            candidate = direct(T2, C2)
            if not candidate.key > current.key:
                return T, C, current, trace
            T, C, current = list(T2), set(C2), candidate
            trace.append(current.key)

    best = descend(list(start.treatment), set(start.control))
    best_trace = [start.score.key] + best[3][1:]
    rng = np.random.default_rng(seed)
    for _ in range(restarts):
        drawn = _random_design(rng, roles, start.k)
        if drawn is None:
            continue
        run = descend(*drawn)
        if run[2].key > best[2].key:
            # A restart begins below the start as often as not; its steps that
            # clear the start's key are the climb the result reports.
            best = run
            best_trace = [start.score.key] + [k for k in run[3]
                                              if k > start.score.key]
    T, C, score, _ = best
    assigned = set(T) | set(C)
    return SearchOutcome(
        k=start.k, treatment=sorted(T), control=sorted(C),
        unassigned=sorted(set(range(N)) - assigned), score=score,
        trace=best_trace, converged=True, evaluations=spent)
