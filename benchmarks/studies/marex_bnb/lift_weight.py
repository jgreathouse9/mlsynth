"""The library's branch and bound with its lift weighted by its prune rate (#832).

The search bounds a node with the persp+N relaxation when pricing the node's
sets would cost more than the bound, times the rate at which bounds have
pruned at the node's subtree size (``branch.PruneRates``). This runs the
search three ways on the same panel: as the library has it (``weighted``),
with the rate held at one (``unweighted``, the search before #832), and with
an ``optimistic`` estimate whose pooled rate starts at one, not one half. Each
row has the nodes bounded, the bounds computed, the sets priced, the seconds,
and the design with its certificate, so the three can be checked to agree.

From the repository root::

    python -m benchmarks.studies.marex_bnb.lift_weight walmart 6 weighted
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

from mlsynth.utils.marex_helpers import branch

#: Where the runner appends its rows.
RESULTS = Path(__file__).resolve().parent / "results"


class _Unweighted:
    """Every bound prunes: the search's rule before the weight."""

    def rate(self, size: int) -> float:
        return 1.0

    def record(self, size: int, pruned: bool) -> None:
        pass


class _Optimistic(branch.PruneRates):
    """The pooled rate starts at one and falls with each bound that does not
    prune: ``(all hits + 1) / (all tries + 1)``."""

    def rate(self, size: int) -> float:
        pooled = (self.all_hits + 1.0) / (self.all_tries + 1.0)
        return ((self.hits.get(size, 0) + self.weight * pooled)
                / (self.tries.get(size, 0) + self.weight))


VARIANTS = {"weighted": branch.PruneRates, "unweighted": _Unweighted,
            "optimistic": _Optimistic}


def panel(name: str):
    """``walmart``, ``geolift``, or ``factor`` / ``noise`` on 40 markets and 40
    periods (``panels.factor(40, 40, seed=1)``, ``panels.noise(40, 40, seed=1)``)."""
    from benchmarks.studies.marex_bnb import panels
    if name == "walmart":
        return panels.walmart()
    if name == "geolift":
        return panels.geolift()
    return getattr(panels, name)(40, 40, seed=1)


def run(B, A, m: int, variant: str, **costs) -> dict:
    """One search of ``m`` treated under ``variant``'s prune rate."""
    from threadpoolctl import threadpool_limits

    from mlsynth.utils.syndes_helpers.exact import SubsetFilter

    saved = branch.PruneRates
    branch.PruneRates = VARIANTS[variant]
    try:
        with warnings.catch_warnings(), threadpool_limits(1):
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            res = branch.branch_and_bound(B, A, [m], filt=SubsetFilter(), **costs)
            seconds = time.perf_counter() - t0
    finally:
        branch.PruneRates = saved
    d, c = res.designs[0], res.certificate
    return {"m": m, "variant": variant, "nodes": res.nodes, "lifts": res.lifts,
            "priced": res.priced, "seconds": seconds, "value": d.value,
            "treated": list(d.treated), "lower": float(c.lower),
            "upper": float(c.upper), "unresolved": len(c.unresolved)}


def main(argv=None) -> int:
    """``PANEL M VARIANT...``: one JSON line per variant appended to
    ``results/lift_weight.jsonl``."""
    import os
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    name, m, variants = args[0], int(args[1]), args[2:] or list(VARIANTS)
    B, A = panel(name)
    RESULTS.mkdir(exist_ok=True)
    for variant in variants:
        row = dict(run(B, A, m, variant), panel=name, load=os.getloadavg()[0])
        with open(RESULTS / "lift_weight.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - the command line's entry; main is tested directly
    raise SystemExit(main())
