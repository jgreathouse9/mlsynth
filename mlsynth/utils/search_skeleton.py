"""The loop the certified branch and bounds share (#846).

MAREX's exact backend (:mod:`~mlsynth.utils.marex_helpers.branch`) and SYNDES's
two-way and one-way backends (:mod:`~mlsynth.utils.syndes_helpers.branch`)
search treated sets the same way. A node fixes some units into the treated set,
some out of it, and leaves the rest free. Each node is priced set by set or
bounded and split, best first by the bound its parent reached. The work is
charged in a cost model's seconds and capped by a limit, and a search the limit
stops reports a bound over the nodes it left open. This module holds what they
share:

- :class:`Ledger`, the work charged against the limit;
- :class:`NodeQueue`, the order nodes are taken in;
- :func:`run`, the loop: it asks the search's ``plan`` whether to drop a node,
  price it (:class:`Price`) or bound it (:class:`Bound`), charges the pricing,
  and keeps a node open when the limit is in the way;
- :func:`cheaper_to_price`, the rule both use to choose pricing over a bound;
- :func:`least_open_bound`, the bound over the nodes a stopped search left open;
- :func:`check_work_limit`, the validation every search's limit gets.

Each search keeps its bounds, its pricing and its certificates; ``plan`` and the
steps it returns are closures over them. Two choices differ between the searches
and are parameters here, so neither search's work changes:

- ties between equal keys: SYNDES takes the node pushed last first, so a split
  continues into the child it pushed last; MAREX takes the first pushed
  (``last_first``);
- where the limit is read: SYNDES reads it before each node, and stops once its
  work reaches the limit (``check_before_pop``). Both read it before pricing a
  subtree (SYNDES prices a node whose units are all decided regardless), and
  MAREX before each bound; a step that would pass it is not started.

SYNDES also dives depth first until it has a design (``found``); MAREX does not.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from heapq import heapify, heapreplace
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..exceptions import MlsynthConfigError

#: A queue entry: the key, a tie-break, the node.
Entry = Tuple[Any, int, Any]


def check_work_limit(work_limit) -> None:
    """Refuse a work limit that is not a positive finite number of seconds."""
    if (isinstance(work_limit, bool) or not isinstance(work_limit, (int, float, np.number))
            or not np.isfinite(work_limit) or work_limit <= 0):
        raise MlsynthConfigError(
            f"work_limit must be a positive finite number of seconds; got {work_limit!r}.")


def cheaper_to_price(size: float, price_seconds: float, bound_seconds: float,
                     rate: float = 1.0) -> bool:
    """Whether pricing a subtree's ``size`` sets costs no more than bounding it.

    ``rate`` is the chance a bound discards the node: a bound that discards one
    node in ten saves a tenth of the pricing it could, so the pricing it is
    weighed against is ``size * price_seconds * rate``.
    """
    return size * price_seconds * rate <= bound_seconds


class Ledger:
    """The work a search has charged, in its cost model's seconds, against its
    limit."""

    def __init__(self, limit: float):
        self.limit = limit
        self.spent = 0.0

    def charge(self, seconds: float) -> None:
        self.spent += seconds

    def exhausted(self) -> bool:
        """Whether the work has reached the limit."""
        return self.spent >= self.limit

    def passes(self, seconds: float) -> bool:
        """Whether charging ``seconds`` would take the work past the limit."""
        return self.spent + seconds > self.limit

    def left(self) -> float:
        """What the limit leaves, none once the work has passed it."""
        return max(self.limit - self.spent, 0.0)


class NodeQueue:
    """Open nodes, best first by key.

    Equal keys go last pushed first (``last_first``) or first pushed first.
    With ``found``, a node pushed while ``found()`` is false goes on a stack
    taken last-in first, whatever its key: the search dives until it has a
    design, since best first can spend its limit near the root and reach none.
    Once ``found()`` holds, the next pop moves the stack onto the heap.
    """

    def __init__(self, *, last_first: bool, found: Optional[Callable[[], bool]] = None):
        self.heap: List[Entry] = []
        self.dive: List[Entry] = []
        self.deferred: List[Entry] = []
        self._sign = -1 if last_first else 1
        self._found = found
        self._pushed = 0

    def _has_design(self) -> bool:
        return self._found is None or self._found()

    def push(self, key, node) -> None:
        self._pushed += 1
        entry = (key, self._sign * self._pushed, node)
        if self._has_design():
            heapq.heappush(self.heap, entry)
        else:
            self.dive.append(entry)

    def pop(self) -> Entry:
        if self.dive and self._has_design():
            for waiting in self.dive:
                heapq.heappush(self.heap, waiting)
            self.dive.clear()
        return self.dive.pop() if self.dive else heapq.heappop(self.heap)

    def put_back(self, entry: Entry, key=None) -> None:
        """Keep a popped node open, under ``key`` if given."""
        self.deferred.append(entry if key is None else (key,) + tuple(entry[1:]))

    def open(self) -> List[Entry]:
        """Every node still open: the heap's, the stack's, then those put back."""
        return self.heap + self.dive + self.deferred

    def __len__(self) -> int:
        return len(self.heap) + len(self.dive)


@dataclass(frozen=True)
class Price:
    """Price the node's sets: ``run()`` does it, ``cost`` is what it is
    charged. A ``deferrable`` pricing that would pass the limit is not started."""

    cost: float
    deferrable: bool
    run: Callable[[], None]


@dataclass(frozen=True)
class Bound:
    """Bound the node: ``run(ledger)`` charges its own work and returns the
    children to push as ``(key, node)`` pairs, ``None`` when the node is
    discarded, or :class:`Deferred` when the limit is in the way."""

    run: Callable[[Ledger], Any]


@dataclass(frozen=True)
class Deferred:
    """The node stays open, under ``key``."""

    key: Any


def run(queue: NodeQueue, ledger: Ledger, *,
        plan: Callable[[Any, Any], Optional[Any]], check_before_pop: bool) -> bool:
    """Take nodes off ``queue`` until it is empty or the limit stops the search.

    ``plan(key, node)`` returns ``None`` to drop the node, a :class:`Price` or a
    :class:`Bound`. Returns whether the search closed: ``False`` when the limit
    stopped it, with the nodes it left in ``queue.open()``.
    """
    while queue:
        if check_before_pop and ledger.exhausted():
            return False
        entry = queue.pop()
        key, _, node = entry
        step = plan(key, node)
        if step is None:
            continue
        if isinstance(step, Price):
            if step.deferrable and ledger.passes(step.cost):
                queue.put_back(entry)
                return False
            ledger.charge(step.cost)
            step.run()
            continue
        out = step.run(ledger)
        if isinstance(out, Deferred):
            queue.put_back(entry, out.key)
            return False
        for child_key, child in out or ():
            queue.push(child_key, child)
    return True


def least_open_bound(
    items: Sequence[Tuple[float, object]],
    evaluate: Callable[[float, object], Tuple[float, float]],
    budget: float,
) -> Tuple[float, float, int]:
    """The smallest certified bound over the nodes a stopped search left open.

    ``items`` are ``(inherited, node)`` pairs, a node's inherited certified bound
    and what ``evaluate`` needs to bound it; ``evaluate(inherited, node)``
    returns the node's own certified bound and its cost in modelled seconds. A
    node's bound is the higher of the two. Only the smallest bound decides the
    bracket, so the nodes are bounded lowest first, and the search stops when
    the smallest bound left is one already computed: every other node's
    inherited bound is at least that, and its own bound can only raise it. A
    node is bounded only while ``budget`` is not spent; past it the nodes keep
    their inherited bounds, which are still bounds.

    Returns the bound (infinite with no node), the cost spent and the number of
    nodes bounded. The cost exceeds ``budget`` by at most one node's.
    """
    heap = [(float(b), 1, i) for i, (b, _) in enumerate(items)]
    heapify(heap)
    spent, evaluated = 0.0, 0
    while heap:
        bound, unbounded, i = heap[0]
        if not unbounded or spent >= budget:
            return bound, spent, evaluated
        own, cost = evaluate(*items[i])
        spent += cost
        evaluated += 1
        heapreplace(heap, (max(bound, own), 0, i))
    return np.inf, spent, evaluated
