"""Which bounds prune at their own node in the unweighted search (#832).

One JSON line per panel to stdout: the bounds computed and, for each in order,
the bit length of its node's set count and whether the node was discarded
right after it. From the repository root::

    python -m benchmarks.studies.marex_bnb.lift_weight_prunes geolift 8
"""
import json
import sys
import warnings

from threadpoolctl import threadpool_limits

from benchmarks.studies.marex_bnb import lift_weight as lw
from mlsynth.utils.marex_helpers import branch
from mlsynth.utils.syndes_helpers.exact import SubsetFilter


class _Recorded(lw._Unweighted):
    log = []

    def record(self, size, pruned):
        _Recorded.log.append((size, bool(pruned)))


def main(name: str, m: int) -> dict:
    B, A = lw.panel(name)
    saved = branch.PruneRates
    branch.PruneRates = _Recorded
    _Recorded.log = []
    try:
        with warnings.catch_warnings(), threadpool_limits(1):
            warnings.simplefilter("ignore")
            res = branch.branch_and_bound(B, A, [m], filt=SubsetFilter())
    finally:
        branch.PruneRates = saved
    return {"panel": name, "m": m, "lifts": res.lifts, "seq": _Recorded.log}


if __name__ == "__main__":  # pragma: no cover - the command line's entry
    print(json.dumps(main(sys.argv[1], int(sys.argv[2]))), flush=True)
