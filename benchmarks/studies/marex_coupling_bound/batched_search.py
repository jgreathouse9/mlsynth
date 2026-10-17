"""The weakly-targeted GeoLift search three ways, in one process.

* ``batched``: :func:`search_batched` below, a search driven by
  :func:`batched.wolfe_bound`;
* ``per_candidate``: the shipped search with the per-candidate bound in front
  of each fit (``search.run`` with the guard);
* ``shipped``: the search as it ships.

:func:`search_batched` follows the shipped search's logic for one cluster with
no restrictions: candidates in order of the closed-form bound ``lb``, a
candidate discarded once a bound clears the incumbent plus the search's margin,
and the coupled fit, with that cutoff, for the rest. The first 8 candidates are
fitted to set an incumbent. After that, each batch of 512 is bounded by
``lb + beta * dist_lb^2`` against the cutoff at the batch's start, and the
candidates it does not clear are fitted in ``lb`` order, the treated fit first
as the shipped search does. A cutoff that improves inside a batch only costs
extra fits. The loop stops at the first batch whose ``lb`` all clear the cutoff,
since ``lb`` is sorted.

Single-threaded, after one untimed batched search at ``beta = 1``. From the
repository root, after ``search.py``::

    python benchmarks/studies/marex_coupling_bound/batched_search.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402

from benchmarks.studies.marex_coupling_bound import batched, bound, panel  # noqa: E402
from benchmarks.studies.marex_coupling_bound.search import run  # noqa: E402
from mlsynth.utils.marex_helpers import exact  # noqa: E402

SEED = 8
BATCH = 512


def search_batched(B, A, beta, m=panel.M):
    t0 = time.perf_counter()
    n = B.shape[1]
    Bc = B - A[:, None]
    G = Bc.T @ Bc
    d2 = bound.squared_pair_distances(B)
    margin = lambda c: exact.PRUNE_MARGIN * (1.0 + abs(c))
    sets = np.array(list(itertools.combinations(range(n), m)))
    lb = exact.treated_fit_bound(B, A, sets)
    order = np.argsort(lb, kind="stable")
    best, best_S = np.inf, None
    count = {"bounded": 0, "discarded_by_bound": 0, "treated_fits": 0, "coupled_fits": 0}
    pos, size = 0, SEED
    while pos < len(order):
        cut = best + margin(best) if np.isfinite(best) else np.inf
        block = order[pos:pos + size]
        pos, size = pos + size, BATCH
        block = block[lb[block] < cut]
        if not block.size:
            break
        if np.isfinite(cut):
            count["bounded"] += len(block)
            status = batched.wolfe_bound(G, sets[block], (cut - lb[block]) / beta, d2)[1]
            count["discarded_by_bound"] += int((status == 1).sum())
            block = block[status != 1]
        for k in block:
            cut = best + margin(best) if np.isfinite(best) else np.inf
            if lb[k] >= cut:
                continue
            S = [int(s) for s in sets[k]]
            count["treated_fits"] += 1
            g, w = exact.side_fit(B, A, S)
            if g >= best:
                continue
            C = [j for j in range(n) if j not in S]
            count["coupled_fits"] += 1
            fit = exact.coupled_fit(B, A, S, C, beta, w0=w,
                                    cutoff=cut if np.isfinite(cut) else None)
            if fit is not None and (fit[0] < best or (fit[0] == best and tuple(S) < best_S)):
                best, best_S = fit[0], tuple(S)
    return dict(count, value=best, treated=list(best_S),
                seconds=round(time.perf_counter() - t0, 2))


def main():
    B, A = panel.geolift()
    search_batched(B, A, panel.BETAS[0])
    out = {}
    for beta in panel.BETAS:
        r = {"batched": search_batched(B, A, beta),
             "per_candidate": run(B, A, beta, guard=True),
             "shipped": run(B, A, beta, guard=False)}
        r["same_design"] = all(r[k]["treated"] == r["shipped"]["treated"]
                               and r[k]["value"] == r["shipped"]["value"]
                               for k in ("batched", "per_candidate"))
        out[str(beta)] = r
        print(f"beta={beta:g}: " + ", ".join(
            f"{k} {r[k]['seconds']} s" for k in ("batched", "per_candidate", "shipped"))
            + f"; same design {r['same_design']}", flush=True)
    (panel.RESULTS / "batched_search.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
