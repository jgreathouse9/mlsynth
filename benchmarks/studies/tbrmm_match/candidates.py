"""One step of the climb, enumerated and scored by whichever engine is passed.

Separating enumeration from scoring is what lets the study answer the two
questions apart: given the same candidates, do the engines score them the same,
and given the same scores, do they pick the same one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Set, Tuple

Key = Tuple[float, ...]
Scorer = Callable[[Sequence[str], Sequence[str]], Key]


@dataclass(frozen=True)
class Candidate:
    """One neighbouring split, and the geo whose move produced it."""

    geo: str
    treatment: frozenset
    control: frozenset


@dataclass(frozen=True)
class Scored:
    candidate: Candidate
    key: Key


def augmentation_candidates(treatment: Set[str], control: Set[str],
                            treatment_pool: Set[str]) -> List[Candidate]:
    """Every split reachable by moving one geo into the treatment group.

    A candidate already in the control group leaves it in the same step, which
    is how both engines resolve a geo belonging to two groups at once.
    """
    out = []
    for geo in sorted(treatment_pool - treatment):
        remaining = control - {geo}
        if not remaining:
            continue
        out.append(Candidate(geo, frozenset(treatment | {geo}), frozenset(remaining)))
    return out


def matching_candidates(treatment: Set[str], control: Set[str],
                        toggleable: Set[str]) -> List[Candidate]:
    """Every split reachable by one symmetric difference on the control group."""
    out = []
    for geo in sorted(toggleable):
        flipped = control ^ {geo}
        if not flipped:
            continue
        out.append(Candidate(geo, frozenset(treatment), frozenset(flipped)))
    return out


def score_all(candidates: Sequence[Candidate], scorer: Scorer) -> List[Scored]:
    """Score every candidate with one engine, in the order given."""
    return [Scored(c, scorer(sorted(c.treatment), sorted(c.control)))
            for c in candidates]


def best(scored: Sequence[Scored]) -> Scored:
    """The candidate a greedy step would take: the first strict maximum.

    First and not last, so the walk is a function of the candidate order and
    nothing else.
    """
    winner = scored[0]
    for item in scored[1:]:
        if item.key > winner.key:
            winner = item
    return winner


def score_disagreements(left: Sequence[Scored], right: Sequence[Scored],
                        tol: float) -> List[Tuple[str, Key, Key, float]]:
    """Candidates the two engines score differently, with the relative gap.

    The gap is measured on the last element, the inverse detectable impact,
    because the four gates ahead of it are integers and the correlation is
    rounded: those either match exactly or name the disagreement themselves.
    """
    out = []
    for a, b in zip(left, right):
        assert a.candidate.geo == b.candidate.geo, "candidate lists are misaligned"
        gates_differ = a.key[:-1] != b.key[:-1]
        denom = abs(a.key[-1]) or 1.0
        gap = abs(a.key[-1] - b.key[-1]) / denom
        if gates_differ or gap > tol:
            out.append((a.candidate.geo, a.key, b.key, gap))
    return out
