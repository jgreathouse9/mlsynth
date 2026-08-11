"""The restricted space of treated sets: how many there are, and how to walk it.

The exact two-way backend searches treated sets, and how many it must visit
decides whether it can finish. Generating every ``C(N, K)`` subset and testing
each against the restrictions makes that count a property of the unrestricted
problem, so an instance with two hundred units and all but ten of them forbidden
costs what two hundred free units cost, though only ``C(10, K)`` designs exist.

Three restrictions are structural, meaning they can be honoured while a candidate
is built:

``forced``
    units in every design, so they leave the choice entirely;
``forbidden``
    units in no design, so they leave the ground set;
``strata``
    a lower and upper quota on how many members of a group a design may hold.

:func:`design_restrictions` keys each unit to one stratum, so the groups are
disjoint and the admissible designs are a product of independent per-group
choices. That product is countable by a short dynamic program and walkable in the
same order, which is what this module provides: :attr:`SearchSpace.size` is the
exact number of subsets :meth:`SearchSpace.iter_subsets` will emit, computed
without emitting any of them.

The count is what the enumeration limit should be measured against. Conflict
pairs, a cost budget and the pool's no-good sets stay predicates on a finished
candidate, since none of them decomposes over a group. They only remove designs,
so :attr:`SearchSpace.size` bounds the admissible count from above and the gate
stays safe.

With no restrictions the walk is ``combinations(range(N), K)`` term for term,
including order, so the search meets designs in the order it always did.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

#: A stratum as :mod:`restrictions` states it: members, then optional bounds.
Stratum = Tuple[Sequence[int], Optional[int], Optional[int]]


@dataclass(frozen=True)
class Group:
    """Selectable units sharing a quota, with the quota resolved to integers.

    ``lo`` and ``hi`` are the bounds after the forced units inside the group have
    been credited against them and after the forbidden ones have left
    ``members``, so a walk reads them directly.
    """

    members: Tuple[int, ...]
    lo: int
    hi: int


@dataclass(frozen=True)
class SearchSpace:
    """The treated sets a restricted enumeration will visit.

    Attributes
    ----------
    forced : tuple of int
        Units present in every emitted design.
    groups : tuple of Group
        Disjoint blocks of selectable units, each with a resolved quota.
    k_free : int
        How many units the walk still chooses, once ``forced`` is accounted for.
    size : int
        Exactly how many subsets :meth:`iter_subsets` emits. Zero when the
        restrictions are jointly unsatisfiable.
    """

    forced: Tuple[int, ...]
    groups: Tuple[Group, ...]
    k_free: int
    size: int
    _ways: Tuple[Tuple[int, ...], ...] = ()
    _presorted: bool = False

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        N: int,
        K: int,
        *,
        forced: Iterable[int] = (),
        forbidden: Iterable[int] = (),
        strata: Sequence[Stratum] = (),
    ) -> "SearchSpace":
        """Resolve the structural restrictions into groups and count the result.

        An unsatisfiable combination gives a space of size zero with an empty
        walk. The search reads that as "no admissible design" and raises with the
        message it already had, so an impossible instance keeps one explanation.
        """
        forced_set = frozenset(int(i) for i in forced)
        forbidden_set = frozenset(int(i) for i in forbidden)
        k_free = int(K) - len(forced_set)

        if forced_set & forbidden_set or k_free < 0:
            return cls._empty(forced_set, k_free)

        free = [i for i in range(int(N))
                if i not in forced_set and i not in forbidden_set]
        groups = cls._groups(free, forced_set, strata, K=int(K))
        if groups is None:
            return cls._empty(forced_set, k_free)

        ways = _count_ways(groups, k_free)
        return cls(forced=tuple(sorted(forced_set)), groups=groups,
                   k_free=k_free, size=ways[0][k_free],
                   _ways=tuple(tuple(row) for row in ways),
                   _presorted=not forced_set and _ascending(groups))

    @classmethod
    def _empty(cls, forced: frozenset, k_free: int) -> "SearchSpace":
        return cls(forced=tuple(sorted(forced)), groups=(),
                   k_free=max(k_free, 0), size=0, _ways=())

    @staticmethod
    def _groups(
        free: List[int],
        forced: frozenset,
        strata: Sequence[Stratum],
        *,
        K: int,
    ) -> Optional[Tuple[Group, ...]]:
        """Partition the free units into quota groups, or ``None`` if impossible.

        Falls back to a single unquotaed group when the strata overlap, which
        keeps the walk correct for a caller that supplies its own: the emitted
        designs then cover the admissible ones, and the quota is applied as a
        candidate test downstream.
        """
        if not strata or not _disjoint(strata):
            return (Group(tuple(free), 0, len(free)),)

        groups: List[Group] = []
        claimed: set = set()
        for members, lo, hi in strata:
            member_set = frozenset(int(m) for m in members)
            claimed |= member_set
            selectable = tuple(i for i in free if i in member_set)
            held = len(forced & member_set)
            low = max(0, (0 if lo is None else int(lo)) - held)
            high = min(len(selectable),
                       (K if hi is None else int(hi)) - held)
            if high < low:
                return None
            groups.append(Group(selectable, low, high))

        leftover = tuple(i for i in free if i not in claimed)
        if leftover:
            groups.append(Group(leftover, 0, len(leftover)))
        return tuple(groups)

    # ------------------------------------------------------------------
    # the walk
    # ------------------------------------------------------------------

    def iter_subsets(self) -> Iterator[Tuple[int, ...]]:
        """Yield every admissible treated set once, as a sorted tuple.

        Emits exactly :attr:`size` subsets. Branches that no completion can
        finish are skipped on the same table the count was read from, so the walk
        does no work per design it does not emit.

        One group and no forced units is the unrestricted case, and there the
        general walk would pay a frame and a sort per design for a merge it does
        not need: a single group's members are ascending, so its combinations are
        already sorted treated sets. That case defers to
        :func:`itertools.combinations`, which is what the search used before
        candidates were built inside the restrictions. A quota can still make the
        one group unreachable, so the size check above is what admits this path.
        """
        if not self.size:
            return
        if not self.forced and len(self.groups) == 1:
            yield from combinations(self.groups[0].members, self.k_free)
            return
        yield from self._walk(0, self.k_free, [])

    def _walk(self, g: int, remaining: int,
              picked: List[int]) -> Iterator[Tuple[int, ...]]:
        if g == len(self.groups):
            # Picks arrive group by group, ascending inside each. When the groups
            # are themselves ascending and nothing has to be merged in, that is
            # already the sorted treated set, and a restriction trimming the
            # space by a third does not earn a sort per design.
            yield (tuple(picked) if self._presorted
                   else tuple(sorted(self.forced + tuple(picked))))
            return
        group = self.groups[g]
        for take in range(group.lo, min(group.hi, remaining) + 1):
            if not self._ways[g + 1][remaining - take]:
                continue
            for pick in combinations(group.members, take):
                yield from self._walk(g + 1, remaining - take,
                                      picked + list(pick))


def _ascending(groups: Tuple[Group, ...]) -> bool:
    """Whether concatenating one pick per group, in order, comes out sorted.

    Members inside a group are ascending because the ground set is. This asks the
    same of the groups: every member of one below every member of the next. A
    caller is free to list strata in any order, and the leftover group lands last
    whatever it holds, so this is checked and not assumed.
    """
    highest = -1
    for group in groups:
        if not group.members:
            continue
        if group.members[0] <= highest:
            return False
        highest = group.members[-1]
    return True


def _disjoint(strata: Sequence[Stratum]) -> bool:
    """Whether no unit belongs to two strata."""
    seen: set = set()
    for members, _, _ in strata:
        member_set = {int(m) for m in members}
        if seen & member_set:
            return False
        seen |= member_set
    return True


def _count_ways(groups: Tuple[Group, ...], k_free: int) -> List[List[int]]:
    """``ways[g][j]``: how many ways groups ``g`` onward supply ``j`` units.

    Read backwards so ``ways[0][k_free]`` is the size of the whole space and
    ``ways[g + 1][.]`` tells the walk whether a branch can still be completed.
    """
    n_groups = len(groups)
    ways = [[0] * (k_free + 1) for _ in range(n_groups + 1)]
    ways[n_groups][0] = 1
    for g in range(n_groups - 1, -1, -1):
        group = groups[g]
        for j in range(k_free + 1):
            total = 0
            for take in range(group.lo, min(group.hi, j) + 1):
                total += comb(len(group.members), take) * ways[g + 1][j - take]
            ways[g][j] = total
    return ways
