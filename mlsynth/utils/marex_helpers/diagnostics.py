"""What MAREX's exact search did, as a fixed-size summary per cluster (#774).

The exact backend proves its design optimal by discarding every other treated
set. This module records how: how many candidates each stage decided, how far
the discarded ones cleared the incumbent, whether any bound fell back, and where
the time went. The summary has the same size whatever the number of candidates,
and every field merges by addition, minimum or maximum, so summaries combine
across clusters (and, later, across worker processes) without rereading any
candidate.

Each admissible candidate gets exactly one fate, and each fate is counted where
it is decided:

* ``cleared_by_bound``: its closed-form bound cleared the incumbent, so it was
  never fitted. Of these, ``cleared_outside_buffer`` were never walked: their
  bound sat above the buffer of best-bounded candidates, and the buffer was
  enough to certify the result;
* ``cleared_by_coupling``: the weakly-targeted design's hull-distance bound
  cleared it;
* ``cleared_by_treated_fit``: its treated-side fit alone cleared the incumbent;
* ``without_control``: its donor exclusions left no control market;
* ``unweighted``: its treated fit gave a market no weight, so the set treats
  fewer markets than it holds and is refused;
* ``cleared_by_fit_cutoff``: the weakly-targeted coupled fit stopped once its
  certified bound cleared the incumbent;
* ``fitted``: fitted to completion.

so the seven sum to ``candidates``. The branch and bound above the candidate limit
reports its own counts (``bnb_bounds``, ``bnb_priced``), the designs it fitted,
its work in the cost model's seconds (``bnb_work``) and the nodes its work limit
left open (``bnb_open``, zero when it proved its design optimal).

Some fields are facts about the problem and some about the run. The candidate
count, the search kind and the bound fallbacks of the first pass do not depend
on the order in which candidates are visited; every other count and the
clearance histogram do, and the times depend on the machine.
:meth:`ExactSearchDiagnostics.order_independent` returns the first group.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

#: Clearance bin edges: four a decade from 1e-12 to 1e6. With an underflow and
#: an overflow bin the histogram holds 74 counts, 592 bytes, whatever the
#: candidate count.
CLEARANCE_EDGES = 10.0 ** (np.arange(73) / 4.0 - 12.0)

_BINS = len(CLEARANCE_EDGES) + 1


class MAREXSearchWarning(UserWarning):
    """Something about MAREX's exact search a user should act on: a bound that
    fell back to zero, or a certificate that could not resolve a design."""


@dataclass(frozen=True)
class LogHistogram:
    """Counts of non-negative values in :data:`CLEARANCE_EDGES`' bins.

    ``counts[0]`` holds values under 1e-12, ``counts[-1]`` values at or past 1e6
    and any that are not finite, and ``counts[i]`` the values in
    ``[edges[i - 1], edges[i])``.
    """

    counts: Tuple[int, ...]

    @classmethod
    def empty(cls) -> "LogHistogram":
        return cls(counts=(0,) * _BINS)

    @classmethod
    def of(cls, values: np.ndarray, weights: Optional[np.ndarray] = None) -> "LogHistogram":
        values = np.asarray(values, dtype=float).reshape(-1)
        idx = np.searchsorted(CLEARANCE_EDGES, values, side="right")
        idx[~np.isfinite(values)] = _BINS - 1
        counts = np.bincount(idx, weights=weights, minlength=_BINS)
        return cls(counts=tuple(int(round(c)) for c in counts))

    @property
    def total(self) -> int:
        return int(sum(self.counts))

    def merge(self, other: "LogHistogram") -> "LogHistogram":
        return LogHistogram(counts=tuple(a + b for a, b in zip(self.counts, other.counts)))

    def quantile(self, q: float) -> float:
        """The upper edge of the bin holding the ``q`` quantile: the clearance is
        at most this. ``nan`` for an empty histogram, ``inf`` in the overflow."""
        total = self.total
        if total == 0:
            return float("nan")
        k = int(np.searchsorted(np.cumsum(self.counts), q * total, side="left"))
        k = min(k, _BINS - 1)
        return float(CLEARANCE_EDGES[k]) if k < _BINS - 1 else float("inf")


def _merge(a: Any, b: Any) -> Any:
    """Field-wise merge of two frozen summaries: booleans or, ``*_max`` fields
    take the maximum, every other number adds."""
    out = {}
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, bool):
            out[f.name] = x or y
        elif f.name.endswith("_max_gap"):
            out[f.name] = max(x, y)
        else:
            out[f.name] = x + y
    return type(a)(**out)


@dataclass(frozen=True)
class SearchCounts:
    """How many candidates each stage decided; see the module docstring."""

    candidates: int = 0
    priced: int = 0
    cleared_by_bound: int = 0
    cleared_outside_buffer: int = 0
    cleared_by_coupling: int = 0
    cleared_by_treated_fit: int = 0
    without_control: int = 0
    unweighted: int = 0
    cleared_by_fit_cutoff: int = 0
    fitted: int = 0
    second_pass: bool = False
    bnb_bounds: int = 0
    bnb_priced: int = 0
    bnb_work: float = 0.0
    bnb_open: int = 0

    def merge(self, other: "SearchCounts") -> "SearchCounts":
        return _merge(self, other)


@dataclass(frozen=True)
class SearchNumerics:
    """Whether the numbers can be trusted.

    ``bound_fallbacks`` counts the bounds of the first pass that fell back to 0
    because the bordered update broke down (a prefix with no inverse, or a
    Schur complement that is not positive): each such candidate was searched
    on no bound at all. The coupled-fit fields cover the weakly-targeted design's fits that ran
    to completion: their Newton steps and sweeps, and the largest Frank-Wolfe
    gap at termination relative to ``1 + value``.
    """

    bound_fallbacks: int = 0
    coupled_fits: int = 0
    coupled_steps: int = 0
    coupled_sweeps: int = 0
    coupled_max_gap: float = 0.0

    def merge(self, other: "SearchNumerics") -> "SearchNumerics":
        return _merge(self, other)


@dataclass(frozen=True)
class SearchTiming:
    """Seconds by phase, on the machine that ran the search: pricing the
    closed-form bounds, the fits, and the branch and bound when it ran."""

    pricing: float = 0.0
    fitting: float = 0.0
    branch_and_bound: float = 0.0

    def merge(self, other: "SearchTiming") -> "SearchTiming":
        return _merge(self, other)


@dataclass(frozen=True)
class ExactSearchDiagnostics:
    """One cluster's exact search, or the merge of several.

    Attributes
    ----------
    search : str
        ``"enumeration"`` (every candidate priced), ``"branch_and_bound"``
        (above the candidate limit), or ``"mixed"`` once two kinds merge.
    counts : SearchCounts
    clearance : LogHistogram
        ``(bound - cutoff) / |cutoff|`` for each candidate the walk discarded on
        its bound, the cutoff being the incumbent when it was discarded. The
        candidates discarded outside the buffer are not in it; their count is
        ``counts.cleared_outside_buffer`` and their least clearance is at least
        ``outside_buffer_clearance``.
    outside_buffer_clearance : float
        A lower bound on the clearance of every candidate discarded outside the
        buffer (``inf`` when there were none).
    numerics : SearchNumerics
    timing : SearchTiming
    """

    search: str
    counts: SearchCounts = field(default_factory=SearchCounts)
    clearance: LogHistogram = field(default_factory=LogHistogram.empty)
    outside_buffer_clearance: float = float("inf")
    numerics: SearchNumerics = field(default_factory=SearchNumerics)
    timing: SearchTiming = field(default_factory=SearchTiming)

    #: The fields that do not depend on the order candidates were visited in.
    ORDER_INDEPENDENT = ("search", "counts.candidates", "numerics.bound_fallbacks")

    @classmethod
    def empty(cls, search: str) -> "ExactSearchDiagnostics":
        return cls(search=search)

    def merge(self, other: "ExactSearchDiagnostics") -> "ExactSearchDiagnostics":
        return ExactSearchDiagnostics(
            search=self.search if self.search == other.search else "mixed",
            counts=self.counts.merge(other.counts),
            clearance=self.clearance.merge(other.clearance),
            outside_buffer_clearance=min(self.outside_buffer_clearance,
                                         other.outside_buffer_clearance),
            numerics=self.numerics.merge(other.numerics),
            timing=self.timing.merge(other.timing))

    def order_independent(self) -> Dict[str, Any]:
        """The fields that are facts about the problem, by dotted name."""
        out = {}
        for name in self.ORDER_INDEPENDENT:
            value: Any = self
            for part in name.split("."):
                value = getattr(value, part)
            out[name] = value
        return out


class SearchReport:
    """The mutable record a search fills as it runs; :meth:`freeze` turns it
    into an :class:`ExactSearchDiagnostics`. Counters are plain attributes so
    the search's inner loop pays one addition per event."""

    def __init__(self) -> None:
        self.counts: Dict[str, Any] = {f.name: f.default for f in dataclasses.fields(SearchCounts)}
        self.numerics: Dict[str, Any] = {f.name: f.default
                                         for f in dataclasses.fields(SearchNumerics)}
        self.timing: Dict[str, float] = {f.name: f.default for f in dataclasses.fields(SearchTiming)}
        self.clearance = LogHistogram.empty()
        self.outside_buffer_clearance = float("inf")

    def cleared(self, bounds: np.ndarray, cutoff: float) -> None:
        """Candidates the walk discarded on their bounds against ``cutoff``."""
        bounds = np.asarray(bounds, dtype=float)
        if bounds.size:
            self.counts["cleared_by_bound"] += int(bounds.size)
            scale = max(abs(float(cutoff)), np.finfo(float).tiny)
            self.clearance = self.clearance.merge(
                LogHistogram.of((bounds - cutoff) / scale))

    def cleared_outside(self, count: int, least_bound: float, cutoff: float) -> None:
        """``count`` candidates outside the buffer, every bound at least
        ``least_bound``, discarded against ``cutoff``."""
        if count > 0:
            self.counts["cleared_by_bound"] += int(count)
            self.counts["cleared_outside_buffer"] += int(count)
            scale = max(abs(float(cutoff)), np.finfo(float).tiny)
            self.outside_buffer_clearance = min(self.outside_buffer_clearance,
                                                (float(least_bound) - cutoff) / scale)

    def coupled(self, info: Dict[str, Any], value: float) -> None:
        """A coupled fit that ran to completion, with its ``return_info``."""
        self.numerics["coupled_fits"] += 1
        self.numerics["coupled_steps"] += int(info["newton_steps"])
        self.numerics["coupled_sweeps"] += int(info["sweeps"])
        self.numerics["coupled_max_gap"] = max(
            self.numerics["coupled_max_gap"], float(info["gap"]) / (1.0 + abs(value)))

    def freeze(self, search: str) -> ExactSearchDiagnostics:
        return ExactSearchDiagnostics(
            search=search, counts=SearchCounts(**self.counts), clearance=self.clearance,
            outside_buffer_clearance=self.outside_buffer_clearance,
            numerics=SearchNumerics(**self.numerics), timing=SearchTiming(**self.timing))
