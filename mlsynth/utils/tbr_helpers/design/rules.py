"""The business rules on a TBR design, over geo indices (#1046).

Au's ``A_i`` decides which of the three groups each geo may join, one geo at a
time. The rules SYNDES, MAREX and LEXSCM offer on top of that are statements
about the groups as a whole, and a role set cannot express them:

* no two treated geos may conflict (``cluster_col`` / ``adjacency``), because a
  campaign in one bleeds into its neighbour and the two are then not separate
  treated units;
* the treatment group meets coverage quotas (``stratum_col`` with
  ``min_per_stratum`` / ``max_per_stratum``). A minimum binds only in a stratum
  with a treatment-eligible geo, which is how SYNDES reads it;
* the treated geos' market costs sum to at most ``max_total_cost``;
* with ``exclude_bordering_donors``, a treated geo's conflict neighbours stay out
  of the control group, since spillover would put part of the effect in the
  counterfactual;
* the control group has between ``min_controls`` and ``max_controls`` geos.

Forced geos, forbidden geos and the size band are statements about one geo, so
they are folded into the role sets before the search and are not rules here.

The search reads four things off a :class:`DesignRules`. ``admits`` is the final
check on a treatment group. ``extendable`` is what the augmentation filters on:
the conflict, stratum-maximum and budget rules only tighten as the group grows,
so a group that breaks one cannot be repaired, and a stratum minimum can only be
met by geos that can still be added. Both conditions are necessary for some
admissible group of at most ``K`` geos to contain the current one, so discarding
a geo that fails them never discards a design. They are not sufficient: a
conflict graph and a quota together can close every completion in a way no
count detects, which is why the search still records a size only when
``admits`` holds. ``banned_controls`` and ``control_band`` bound the control
group. ``audit`` refuses, before any climbing, a problem no size admits, naming
the rule that binds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import FrozenSet, Iterable, Optional, Sequence, Tuple

from ....exceptions import MlsynthDataError

TREATMENT = "treatment"
CONTROL = "control"
UNASSIGNED = "unassigned"

#: Costs are summed in floating point, so a group whose costs add to the budget
#: exactly can land one rounding step above it.
_BUDGET_SLACK = 1e-9

#: A stratum: its member indices, its minimum and its maximum.
Stratum = Tuple[FrozenSet[int], Optional[int], Optional[int]]


@dataclass(frozen=True)
class DesignRules:
    """Index-level rules on a TBR design. Build with :meth:`build`."""

    treatable: FrozenSet[int]
    controllable: FrozenSet[int]
    must_control: FrozenSet[int]
    neighbours: Tuple[FrozenSet[int], ...]
    strata: Tuple[Stratum, ...] = ()
    costs: Optional[Tuple[float, ...]] = None
    budget: Optional[float] = None
    exclude_bordering_donors: bool = False
    min_controls: Optional[int] = None
    max_controls: Optional[int] = None

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, eligibility: Sequence[FrozenSet[str]], *,
              conflict_pairs: Iterable[Tuple[int, int]] = (),
              strata: Iterable[Tuple[Iterable[int], Optional[int], Optional[int]]] = (),
              costs: Optional[Sequence[float]] = None,
              budget: Optional[float] = None,
              exclude_bordering_donors: bool = False,
              min_controls: Optional[int] = None,
              max_controls: Optional[int] = None) -> "DesignRules":
        """Rules over the geos ``eligibility`` describes, in its column order.

        ``eligibility`` is each geo's ``A_i`` after forced and forbidden geos
        have been folded in. A stratum's minimum is dropped when none of its
        members may be treated.
        """
        n = len(eligibility)
        if n == 0:
            raise MlsynthDataError("rules need at least one geo to apply to.")
        treatable = frozenset(j for j, r in enumerate(eligibility) if TREATMENT in r)
        controllable = frozenset(j for j, r in enumerate(eligibility) if CONTROL in r)
        must = frozenset(j for j in controllable if UNASSIGNED not in eligibility[j])

        near = [set() for _ in range(n)]
        for i, j in conflict_pairs:
            i, j = int(i), int(j)
            if not (0 <= i < n and 0 <= j < n) or i == j:
                raise MlsynthDataError(
                    f"conflict pair ({i}, {j}) has to name two different geos "
                    f"among the {n}.")
            near[i].add(j)
            near[j].add(i)

        seen: set = set()
        parsed = []
        for members, lo, hi in strata:
            members = frozenset(int(m) for m in members)
            if not members or min(members) < 0 or max(members) >= n:
                raise MlsynthDataError(
                    f"stratum {sorted(members)} has to name geos among the {n}.")
            if members & seen:
                raise MlsynthDataError(
                    f"stratum {sorted(members)} overlaps another; strata come "
                    f"from one column, so each geo belongs to one.")
            if lo is not None and hi is not None and lo > hi:
                raise MlsynthDataError(
                    f"stratum {sorted(members)} asks for at least {lo} and at "
                    f"most {hi} treated geos.")
            seen |= members
            parsed.append((members, lo if members & treatable else None, hi))

        if (costs is None) != (budget is None):
            raise MlsynthDataError(
                "market costs and a budget come together: costs without max_total_cost "
                "bound nothing, and a budget without costs has nothing to sum.")
        if costs is not None:
            costs = tuple(float(c) for c in costs)
            if len(costs) != n:
                raise MlsynthDataError(
                    f"costs has {len(costs)} entries for {n} geos.")
            if not all(math.isfinite(c) and c >= 0 for c in costs):
                raise MlsynthDataError(
                    "every market cost has to be finite and non-negative.")
            budget = float(budget)

        if min_controls is not None and min_controls < 1:
            raise MlsynthDataError(f"min_controls is {min_controls}; it has to be at least 1.")
        if min_controls is not None and max_controls is not None \
                and min_controls > max_controls:
            raise MlsynthDataError(
                f"min_controls ({min_controls}) is above max_controls ({max_controls}).")

        return cls(treatable=treatable, controllable=controllable, must_control=must,
                   neighbours=tuple(frozenset(s) for s in near),
                   strata=tuple(parsed), costs=costs, budget=budget,
                   exclude_bordering_donors=bool(exclude_bordering_donors),
                   min_controls=min_controls, max_controls=max_controls)

    # ------------------------------------------------------------- reading
    @property
    def n_units(self) -> int:
        """How many geos the rules cover."""
        return len(self.neighbours)

    @property
    def active(self) -> Tuple[str, ...]:
        """The config fields whose rules bind, for messages and the record."""
        out = []
        if any(self.neighbours):
            out.append("cluster_col/adjacency")
        if any(lo is not None for _, lo, _ in self.strata):
            out.append("min_per_stratum")
        if any(hi is not None for _, _, hi in self.strata):
            out.append("max_per_stratum")
        if self.budget is not None:
            out.append("max_total_cost")
        if self.exclude_bordering_donors:
            out.append("exclude_bordering_donors")
        if self.min_controls is not None:
            out.append("min_controls")
        if self.max_controls is not None:
            out.append("max_controls")
        return tuple(out)

    @property
    def control_band(self) -> Tuple[int, int]:
        """The smallest and largest control group a design may have."""
        lo = 1 if self.min_controls is None else max(1, self.min_controls)
        hi = self.n_units if self.max_controls is None else self.max_controls
        return lo, hi

    def _blocked(self, treatment: Iterable[int]) -> FrozenSet[int]:
        out: set = set()
        for t in treatment:
            out |= self.neighbours[t]
        return frozenset(out)

    def _cost(self, treatment: Iterable[int]) -> float:
        return float(sum(self.costs[t] for t in treatment)) if self.costs else 0.0

    def _over(self, total: float) -> bool:
        return self.budget is not None and \
            total > self.budget + _BUDGET_SLACK * max(1.0, abs(self.budget))

    def _conflicted(self, treatment: FrozenSet[int]) -> bool:
        return any(self.neighbours[t] & treatment for t in treatment)

    def _over_maximum(self, treatment: FrozenSet[int]) -> bool:
        return any(hi is not None and len(members & treatment) > hi
                   for members, _, hi in self.strata)

    def banned_controls(self, treatment: Iterable[int]) -> FrozenSet[int]:
        """Geos the control group may not hold while ``treatment`` is treated."""
        if not self.exclude_bordering_donors:
            return frozenset()
        T = frozenset(treatment)
        return self._blocked(T) - T

    def admits(self, treatment: Iterable[int]) -> bool:
        """Every rule on the treatment group holds, stratum minimums included."""
        T = frozenset(treatment)
        if self._conflicted(T) or self._over_maximum(T):
            return False
        if any(lo is not None and len(members & T) < lo
               for members, lo, _ in self.strata):
            return False
        return not self._over(self._cost(T))

    def admits_design(self, treatment: Iterable[int], control: Iterable[int]) -> bool:
        """``admits`` plus the two rules on the control group."""
        T, C = frozenset(treatment), frozenset(control)
        lo, hi = self.control_band
        return (self.admits(T) and not (C & self.banned_controls(T))
                and lo <= len(C) <= hi)

    def extendable(self, treatment: Iterable[int], K: int) -> bool:
        """Some admissible group of at most ``K`` geos could still contain this one.

        A necessary condition, not a sufficient one. It holds when the rules
        that only tighten as the group grows hold now, when the stratum
        minimums can be met by geos that can still be added -- treatment
        eligible, outside the group, in conflict with none of it, and within the
        budget left -- in the room ``K`` leaves, at a cost the budget covers,
        and when the controls the group leaves can still form a control group:
        no geo that has to be a control is banned, enough geos remain for
        ``min_controls``, and the geos that have to be controls fit under
        ``max_controls`` once the room left treats as many of them as it can.
        """
        T = frozenset(treatment)
        if len(T) > K or self._conflicted(T) or self._over_maximum(T):
            return False
        spent = self._cost(T)
        if self._over(spent):
            return False
        left = (math.inf if self.budget is None
                else self.budget + _BUDGET_SLACK * max(1.0, abs(self.budget)) - spent)
        addable = {g for g in self.treatable - T - self._blocked(T)
                   if self.costs is None or self.costs[g] <= left}
        need, cheapest = 0, 0.0
        for members, lo, _ in self.strata:
            if lo is None:
                continue
            deficit = lo - len(members & T)
            if deficit <= 0:
                continue
            pool = members & addable
            if len(pool) < deficit:
                return False
            need += deficit
            if self.costs is not None:
                cheapest += sum(sorted(self.costs[g] for g in pool)[:deficit])
        if need > K - len(T) or cheapest > left:
            return False
        banned = self.banned_controls(T)
        must = self.must_control - T
        if must & banned:
            return False
        lo, hi = self.control_band
        # A geo that has to be a control leaves that duty only by being treated,
        # and the room left caps how many can be.
        if len(must) - min(K - len(T), len(must & self.treatable)) > hi:
            return False
        return len(self.controllable - T - banned) >= lo

    # ---------------------------------------------------------------- audit
    def audit(self, forced: Sequence[int], K: int) -> None:
        """Refuse a problem no treatment size up to ``K`` can satisfy.

        Each check is a condition every admissible design meets, so a refusal
        is never wrong; the message names the rule that binds. A problem that
        passes can still have no design, and the search says so when it finds
        none.
        """
        F = frozenset(forced)
        for t in sorted(F):
            clash = sorted(self.neighbours[t] & F)
            if clash:
                raise MlsynthDataError(
                    f"forced geos at columns {t} and {clash[0]} conflict under "
                    f"cluster_col/adjacency, and every treatment group contains "
                    f"both.")
        for members, lo, hi in self.strata:
            if hi is not None and len(members & F) > hi:
                raise MlsynthDataError(
                    f"{len(members & F)} forced geos fall in one stratum and "
                    f"max_per_stratum is {hi}.")
            if lo is not None and len(members & self.treatable) < lo:
                raise MlsynthDataError(
                    f"min_per_stratum is {lo} but a stratum has "
                    f"{len(members & self.treatable)} treatment-eligible geo(s).")
        deficits = [(members, lo - len(members & F)) for members, lo, _ in self.strata
                    if lo is not None and lo > len(members & F)]
        need = len(F) + sum(d for _, d in deficits)
        if need > K:
            raise MlsynthDataError(
                f"the min_per_stratum quotas need at least {need} treated geos "
                f"but max_treatment_size is {K}.")
        if self.budget is not None:
            spent = self._cost(F)
            if self._over(spent):
                raise MlsynthDataError(
                    f"the forced geos cost {spent:g} together and max_total_cost "
                    f"is {self.budget:g}.")
            floor = spent + sum(
                sum(sorted(self.costs[g] for g in (members & self.treatable) - F)[:d])
                for members, d in deficits)
            if deficits and self._over(floor):
                raise MlsynthDataError(
                    f"meeting min_per_stratum costs at least {floor:g} and "
                    f"max_total_cost is {self.budget:g}.")
            if not F and not deficits and self._over(
                    min(self.costs[g] for g in self.treatable)):
                raise MlsynthDataError(
                    f"every treatment-eligible geo costs more than max_total_cost "
                    f"({self.budget:g}).")
        banned = self.banned_controls(F)
        stuck = sorted((self.must_control - F) & banned)
        if stuck:
            raise MlsynthDataError(
                f"the geo at column {stuck[0]} has to be in the control group but "
                f"borders a forced geo, and exclude_bordering_donors keeps it out.")
        lo, hi = self.control_band
        pool = self.controllable - F
        # Every design treats at least one geo and meets the quotas, so at least
        # this many geos join the forced ones; those that could have been
        # controls leave the pool.
        extra = max(need, 1) - len(F)
        drain = max(0, extra - len(self.treatable - self.controllable - F))
        if len(pool) - drain < lo:
            raise MlsynthDataError(
                f"a design needs {lo} control geo(s) (min_controls) but at most "
                f"{max(len(pool) - drain, 0)} remain once the treatment group "
                f"is filled.")
        if len(pool - banned) < lo:
            raise MlsynthDataError(
                f"exclude_bordering_donors leaves {len(pool - banned)} control "
                f"geo(s) beside the forced geos and a design needs {lo}.")
        must = self.must_control - F
        relief = min(K - len(F), len(must & self.treatable))
        if len(must) - relief > hi:
            raise MlsynthDataError(
                f"max_controls is {hi} but at least {len(must) - relief} geos have "
                f"to be controls: their roles allow neither being held out nor, "
                f"within max_treatment_size, being treated.")
        if not self.extendable(F, K):
            raise MlsynthDataError(
                f"the rules leave no completion of the forced geos up to "
                f"max_treatment_size {K}: {', '.join(self.active)} together "
                f"close every treatment group.")
