"""SYNDES's two-way branch and bound, measured with what it does not report.

:func:`measure` runs :func:`mlsynth.utils.syndes_helpers.branch.branch_and_bound`
on one panel and size, and returns its own counts beside four it does not keep:
the lift's dual evaluations (one eigendecomposition each), the exact
certificates, the largest open queue, and the swap search's seconds with whether
its seeds already held the design. Each is read by wrapping the module's own
functions for the length of one call; the wrappers count and pass through, so
the search makes the same decisions, and the module is restored on the way out
whether the search returns or raises. The queue is the shared search loop's
(:mod:`mlsynth.utils.search_skeleton`, #846), so its ``heapq`` is the one watched.

The open queue is best first's heap. The search also keeps a depth-first stack
until its first incumbent, and the swap search supplies one before the first
node, so the heap is the queue.
"""
from __future__ import annotations

import heapq
import time
from contextlib import contextmanager

import numpy as np
from threadpoolctl import threadpool_limits

from mlsynth.utils import search_skeleton
from mlsynth.utils.syndes_helpers import branch
from mlsynth.utils.syndes_helpers.exact import WORK_LIMIT, SubsetFilter
from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem

#: Costs that make every node a bound and a lift, for small checks.
EAGER = dict(price_seconds=1.0, bound_seconds=lambda n: 0.0,
             lift_seconds=lambda n: 0.0)

_ORIGINAL = {"lift": branch.lift, "certify": branch.ExactR.certify,
             "heapq": search_skeleton.heapq, "maxcut": branch.maxcut_candidates}


class _QueueWatch:
    """``heapq`` as the search uses it, keeping the largest heap seen."""

    def __init__(self):
        self.largest = 0

    def heappush(self, heap, item):
        heapq.heappush(heap, item)
        self.largest = max(self.largest, len(heap))

    def heappop(self, heap):
        return heapq.heappop(heap)


@contextmanager
def instrumented():
    tally = {"evaluations": 0, "certificates": 0, "swap_seconds": 0.0, "seeds": []}
    watch = _QueueWatch()

    def lift(*args, **kwargs):
        U, m, evaluations = _ORIGINAL["lift"](*args, **kwargs)
        tally["evaluations"] += int(evaluations)
        return U, m, evaluations

    def certify(self, *args, **kwargs):
        tally["certificates"] += 1
        return _ORIGINAL["certify"](self, *args, **kwargs)

    def maxcut(*args, **kwargs):
        t0 = time.process_time()
        out = _ORIGINAL["maxcut"](*args, **kwargs)
        tally["swap_seconds"] += time.process_time() - t0
        tally["seeds"].append(np.asarray(out))
        return out

    branch.lift, branch.ExactR.certify = lift, certify
    search_skeleton.heapq, branch.maxcut_candidates = watch, maxcut
    try:
        yield tally, watch
    finally:
        branch.lift, branch.ExactR.certify = _ORIGINAL["lift"], _ORIGINAL["certify"]
        search_skeleton.heapq, branch.maxcut_candidates = (_ORIGINAL["heapq"],
                                                           _ORIGINAL["maxcut"])


def _with_complements(seeds, N):
    out = set()
    for block in seeds:
        for S in np.atleast_2d(block):
            t = tuple(sorted(int(i) for i in S))
            out.add(t)
            out.add(tuple(sorted(set(range(N)) - set(t))))
    return out


def measure(Y: np.ndarray, K: int, *, work_limit: float = WORK_LIMIT, **costs) -> dict:
    """One search at size ``K``, with the study's columns."""
    N = Y.shape[1]
    t0 = time.process_time()
    problem = TwoWayProblem.from_panel(Y)
    engine = BoundEngine.from_problem(problem)
    setup = time.process_time() - t0
    with instrumented() as (tally, watch), threadpool_limits(1):
        c0, w0 = time.process_time(), time.perf_counter()
        o = branch.branch_and_bound(problem, engine, K, SubsetFilter(),
                                    work_limit=work_limit, **costs)
        cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    treated = sorted(int(i) for i in o.treated)
    return {
        "N": N, "K": K, "work_limit": work_limit,
        "nodes": o.nodes, "lifts": o.lifts, "evaluations": tally["evaluations"],
        "certificates": tally["certificates"], "pruned": o.pruned,
        "priced": o.priced, "work": o.work,
        "cpu_seconds": cpu, "wall_seconds": wall, "setup_seconds": setup,
        "cpu_per_modelled": cpu / o.work if o.work else float("nan"),
        "max_queue": watch.largest, "swap_seconds": tally["swap_seconds"],
        "swap_found": tuple(treated) in _with_complements(tally["seeds"], N),
        "closed": o.closed, "certified": o.certified,
        "value": o.value, "lower": o.lower,
        "gap": (o.value - o.lower) / o.value, "treated": treated,
    }


def profile_lift(Y: np.ndarray, K: int) -> dict:
    """One lift at the root, every unit free: timed, then profiled.

    The lift is deterministic, so the profiled run repeats the timed one. Its
    seconds are inflated by the profiler; the shares are read from it, on its
    own clock, and the seconds from the timed run, which is CPU time. Both runs
    hold BLAS to one thread whatever the caller's setting: unpinned under four
    test workers, SciPy's L-BFGS-B core took 10.7 s of an 11.8 s lift at 20
    units (#857). ``eigh`` is ``numpy.linalg.eigh``, one call per
    evaluation of the dual; ``_theta`` is the whole evaluation around it;
    ``separate`` is the triangle separation; ``solver`` is SciPy's L-BFGS-B, its
    Fortran core and Python driver, counted by own time so the dual it calls
    back is not counted twice. The three are disjoint; what is left is the
    lift's own bookkeeping and the profiler's.
    """
    import cProfile
    import pstats

    problem = TwoWayProblem.from_panel(Y)
    engine = BoundEngine.from_problem(problem)
    N = Y.shape[1]
    parts = branch.node_parts(engine.R, (), (), tuple(range(N)))
    s = 2 * K - N
    with threadpool_limits(1):
        t0 = time.process_time()
        _, _, evaluations = branch.lift(*parts, s)
        seconds = time.process_time() - t0

        prof = cProfile.Profile()
        prof.enable()
        _, _, profiled_evaluations = branch.lift(*parts, s)
        prof.disable()
    stats = pstats.Stats(prof).stats

    def cumulative(match) -> float:
        return sum(ct for (path, _, name), (_, _, _, ct, _) in stats.items()
                   if match(path, name))

    def own(match) -> float:
        return sum(tt for (path, _, name), (_, _, tt, _, _) in stats.items()
                   if match(path, name))

    # every share on the profiler's own clock: process_time sums the CPU of
    # every BLAS thread, so dividing the profiler's seconds by it understates
    # each share by the thread count
    profiled = cumulative(lambda path, name: name == "lift"
                          and path.endswith("branch.py"))
    eigh = cumulative(lambda path, name: name == "eigh" and "linalg" in path)
    theta = cumulative(lambda path, name: name == "_theta")
    separate = cumulative(lambda path, name: name == "separate"
                          and path.endswith("branch.py"))
    solver = own(lambda path, name: "optimize" in path or "_lbfgsb" in name)
    return {
        "N": N, "K": K, "evaluations": evaluations, "seconds": seconds,
        "profiled_evaluations": profiled_evaluations, "profiled_seconds": profiled,
        "eigh_seconds": eigh, "theta_seconds": theta,
        "separate_seconds": separate, "solver_seconds": solver,
        "eigh_share": eigh / profiled, "theta_share": theta / profiled,
        "separate_share": separate / profiled, "solver_share": solver / profiled,
        "seconds_per_evaluation": seconds / max(evaluations, 1),
    }
