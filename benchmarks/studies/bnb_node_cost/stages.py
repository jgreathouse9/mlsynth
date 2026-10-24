"""Where the library's two-way search spends its time (#844).

:func:`profile` runs ``branch_and_bound`` with its stages wrapped by a
:class:`Ledger`, which charges each wrapped call its own seconds: its time
less the time of the wrapped calls it makes. The search decides what it
decides unwrapped; the wrappers only read the clock.

A stage that also runs inside another (the trust region inside the lift, an
eigendecomposition inside either, the closed form inside the swap search) is
labelled by its caller, ``"lift > trust region"``, so the tree's own trust
regions stay apart from the lift's fallback. Beside each stage's seconds is
the cost model's charge for what it did at the top level, the charge the
search spends its work limit against: a set priced at
``PRICE_SECONDS_PER_UNIT * N``, a trust region at ``trust_region_seconds``,
a lift at its evaluations times ``evaluation_seconds``, a certificate at
``certificate_seconds``. What no wrapper claims is ``rest``: the tree loop,
subtree sizes, the heap.

From the repository root::

    python -m benchmarks.studies.bnb_node_cost.stages geolift 40 20
    python -m benchmarks.studies.bnb_node_cost.stages bls 40 20 limit
"""
from __future__ import annotations

import inspect
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from mlsynth.utils import persp_bound
from mlsynth.utils.marex_helpers.certify import Dyadic
from mlsynth.utils.syndes_helpers import branch
from mlsynth.utils.syndes_helpers.gram import BoundEngine

#: Where the runner appends its rows.
RESULTS = Path(__file__).resolve().parent / "results"


class Ledger:
    """Own seconds and calls per label. A wrapped call is charged its time
    less the wrapped calls it makes, and adds its whole time to its caller's
    nested total, so every second is charged once."""

    def __init__(self):
        self.own = defaultdict(float)
        self.calls = defaultdict(int)
        self.modelled = defaultdict(float)
        self._stack = []        # [label, nested seconds] per open wrapped call

    def wrap(self, label: str, fn: Callable, *, under_caller: bool = False,
             model: Optional[Callable] = None) -> Callable:
        """``fn`` charged to ``label``, or to ``"<caller> > label"`` inside
        another wrapped call when ``under_caller``. ``model(args, result)``
        is the cost model's charge for a call made outside every wrapper."""

        def timed(step):
            caller = self._stack[-1][0] if self._stack else None
            name = f"{caller} > {label}" if (under_caller and caller) else label
            frame = [name, 0.0]
            self._stack.append(frame)
            t0 = time.perf_counter()
            try:
                return caller, name, step()
            finally:
                spent = time.perf_counter() - t0
                self._stack.pop()
                self.own[name] += spent - frame[1]
                self.calls[name] += 1
                if self._stack:
                    self._stack[-1][1] += spent

        def inner(*args, **kwargs):
            caller, name, out = timed(lambda: fn(*args, **kwargs))
            if model is not None and caller is None:
                self.modelled[name] += model(args, out)
            return out

        def stream(*args, **kwargs):
            # a generator's work is done in next(), not in the call
            it = fn(*args, **kwargs)
            done = object()
            while True:
                item = timed(lambda: next(it, done))[2]
                if item is done:
                    return
                yield item
        return stream if inspect.isgeneratorfunction(fn) else inner


def _price_model(N: int) -> Callable:
    return lambda args, out: len(out.lower) * branch.PRICE_SECONDS_PER_UNIT * N


#: ``(owner, attribute, label, under_caller, model)``; ``model`` takes ``N``
#: and returns the charge per call, or is None for a stage the model does not
#: charge. ``BoundEngine.bound``'s first argument is the engine.
STAGES = (
    (BoundEngine, "bound", "pricing: closed form", True, _price_model),
    (branch, "signs_from_subsets", "pricing: sign matrices", True, None),
    (branch, "_stream_from_space", "pricing: enumeration", True, None),
    (branch, "trust_region", "trust region", True,
     lambda N: lambda args, out: branch.trust_region_seconds(len(args[1]))),
    (branch, "lift", "lift", False,
     lambda N: lambda args, out: out[2] * branch.evaluation_seconds(len(args[1]))),
    (branch, "minimize", "lift: L-BFGS-B", False, None),
    (branch, "_theta", "lift: dual evaluation", False, None),
    (branch, "_cut_matrix", "lift: cut matrix", False, None),
    (branch, "separate", "lift: separation", False, None),
    (np.linalg, "eigh", "eigh", True, None),
    (branch.ExactR, "certify", "certificate: assembly", False,
     lambda N: lambda args, out: branch.certificate_seconds(len(args[3]))),
    (branch, "lambda_min_lower", "certificate: lambda_min_lower", False, None),
    (persp_bound, "_float", "certificate: to floating point", False, None),
    (Dyadic, "__matmul__", "certificate: exact product", False, None),
    (branch, "maxcut_candidates", "seed: swap search", False, None),
    (branch, "_settle", "settle", True, None),
    (branch, "_resolve_open", "resolve open sets", False, None),
)


def profile(Y: np.ndarray, K: int, *, work_limit: float = float("inf"),
            costs: Optional[dict] = None) -> dict:
    """The library's branch and bound on ``Y`` at size ``K`` on one BLAS
    thread, under the library's cost model or ``costs`` (its
    ``price_seconds``, ``bound_seconds``, ``lift_seconds``), its stages on a
    :class:`Ledger`: its counts, value and bound, the run's seconds, each
    stage's own seconds, calls and modelled charge, and ``rest``. Every
    wrapper comes off on the way out."""
    from threadpoolctl import threadpool_limits

    from mlsynth.utils.syndes_helpers.exact import SubsetFilter
    from mlsynth.utils.syndes_helpers.gram import TwoWayProblem

    N = Y.shape[1]
    ledger = Ledger()
    saved = [(owner, attr, getattr(owner, attr)) for owner, attr, *_ in STAGES]
    for owner, attr, label, under, model in STAGES:
        setattr(owner, attr, ledger.wrap(label, getattr(owner, attr), under_caller=under,
                                         model=None if model is None else model(N)))
    try:
        problem = TwoWayProblem.from_panel(Y)
        engine = BoundEngine.from_problem(problem)
        with threadpool_limits(1):
            t0 = time.perf_counter()
            o = branch.branch_and_bound(problem, engine, K, SubsetFilter(),
                                        work_limit=work_limit, **(costs or {}))
            seconds = time.perf_counter() - t0
    finally:
        for owner, attr, fn in saved:
            setattr(owner, attr, fn)
    table = {name: {"seconds": ledger.own[name], "calls": ledger.calls[name],
                    "modelled": ledger.modelled.get(name, 0.0)}
             for name in ledger.own}
    return {"N": N, "K": K, "work_limit": work_limit, "nodes": o.nodes,
            "lifts": o.lifts, "pruned": o.pruned, "priced": o.priced, "work": o.work,
            "closed": o.closed, "value": o.value, "lower": o.lower, "seconds": seconds,
            "stages": table, "rest": seconds - sum(v["seconds"] for v in table.values())}


def _panel(name: str, N: int) -> np.ndarray:
    from benchmarks.studies.syndes_global_bnb.run import PANELS
    if name in PANELS:
        return PANELS[name](N)
    from benchmarks.studies.bnb_scale import panels
    return panels.panel(name, N)


def main(argv=None) -> int:
    """``PANEL N K [limit]``: one JSON line appended to ``results/stages.jsonl``;
    ``limit`` runs under the library's default work limit. A panel is
    ``syndes_global_bnb``'s (factor, noise, geolift) or ``bnb_scale``'s
    (bls)."""
    import json
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    name, N, K = args[0], int(args[1]), int(args[2])
    limit = branch.WORK_LIMIT if "limit" in args[3:] else float("inf")
    row = dict(profile(_panel(name, N), K, work_limit=limit), panel=name)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "stages.jsonl", "a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - the command line's entry; main is tested directly
    raise SystemExit(main())
