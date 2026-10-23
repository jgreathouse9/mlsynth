"""The batched bound across worker processes, issue #768's pool.

Two measurements at 1, 2 and 4 workers, on a machine with 4 cores:

* the bound pass alone: every candidate the closed-form bound leaves under the
  optimum, in batches of 512, cutoff at the optimum; the statuses must equal the
  single-process run's;
* the whole search: ``batched_search.search_batched``'s logic with
  :func:`contiguous.wolfe_bound`, where each round bounds ``w`` batches of 512 in
  parallel against the cutoff at the round's start, then fits what they leave in
  ``lb`` order in the parent. The committed batched search runs beside it.

Workers fork, each single-threaded, and get the Gram matrix and pair distances
once through the pool's initializer. Each pool is started before the timer.

From the repository root, after ``search.py``::

    python benchmarks/studies/marex_coupling_bound/pool.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import itertools
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402

from benchmarks.studies.marex_coupling_bound import bound, contiguous, panel  # noqa: E402
from benchmarks.studies.marex_coupling_bound.batched_search import search_batched  # noqa: E402
from mlsynth.utils.marex_helpers import exact  # noqa: E402

BATCH = 512
SEED = 8
WORKERS = (1, 2, 4)


def search(B, A, G, d2, sets, lb, beta, pool, w):
    """The batched search with ``w`` batches bounded per round, by ``pool`` if given."""
    t0 = time.perf_counter()
    n = B.shape[1]
    margin = lambda c: exact.PRUNE_MARGIN * (1.0 + abs(c))
    order = np.argsort(lb, kind="stable")
    best, best_S = np.inf, None
    fits = 0
    pos = SEED
    rounds = [[order[:SEED]]]
    while True:
        blocks = rounds.pop() if rounds else None
        if blocks is None:
            cut = best + margin(best)
            blocks = []
            for _ in range(w):
                blk = order[pos:pos + BATCH]
                pos += BATCH
                blk = blk[lb[blk] < cut]
                if blk.size:
                    blocks.append(blk)
            if not blocks:
                break
            args = [(sets[b_], (cut - lb[b_]) / beta) for b_ in blocks]
            status = (list(pool.map(contiguous.bound_batch, args)) if pool is not None
                      else [contiguous.wolfe_bound(G, a[0], a[1], d2)[1] for a in args])
            blocks = [b_[st != 1] for b_, st in zip(blocks, status)]
        for blk in blocks:
            for k in blk:
                cut = best + margin(best) if np.isfinite(best) else np.inf
                if lb[k] >= cut:
                    continue
                S = [int(s) for s in sets[k]]
                g, wv = exact.side_fit(B, A, S)
                if g >= best:
                    continue
                C = [j for j in range(n) if j not in S]
                fits += 1
                fit = exact.coupled_fit(B, A, S, C, beta, w0=wv,
                                        cutoff=cut if np.isfinite(cut) else None)
                if fit is not None and (fit[0] < best or (fit[0] == best and tuple(S) < best_S)):
                    best, best_S = fit[0], tuple(S)
    return {"seconds": round(time.perf_counter() - t0, 2), "coupled_fits": fits,
            "value": best, "treated": list(best_S)}


def main():
    B, A = panel.geolift()
    Bc = B - A[:, None]
    G = Bc.T @ Bc
    d2 = bound.squared_pair_distances(B)
    sets = np.array(list(itertools.combinations(range(B.shape[1]), panel.M)))
    lb = exact.treated_fit_bound(B, A, sets)
    ctx = mp.get_context("fork")
    out = {"cores": os.cpu_count()}
    for beta in panel.BETAS:
        fstar = panel.optimum(beta)
        cut = fstar + exact.PRUNE_MARGIN * (1.0 + abs(fstar))
        live = np.flatnonzero(lb < cut)
        chunks = [(sets[live[lo:lo + BATCH]], (cut - lb[live[lo:lo + BATCH]]) / beta)
                  for lo in range(0, len(live), BATCH)]
        res = {"bound_pass": {"candidates": int(len(live))}, "search": {}}
        t0 = time.perf_counter()
        ref = np.concatenate([contiguous.wolfe_bound(G, c[0], c[1], d2)[1] for c in chunks])
        t1 = time.perf_counter() - t0
        res["bound_pass"]["workers_1_seconds"] = round(t1, 3)
        res["search"]["committed_batched"] = search_batched(B, A, beta)
        res["search"]["workers_1"] = search(B, A, G, d2, sets, lb, beta, None, 1)
        for w in WORKERS[1:]:
            with ProcessPoolExecutor(max_workers=w, mp_context=ctx,
                                     initializer=contiguous.init_worker, initargs=(G, d2)) as p:
                list(p.map(contiguous.bound_batch, chunks[:w]))
                t0 = time.perf_counter()
                par = np.concatenate(list(p.map(contiguous.bound_batch, chunks)))
                tw = time.perf_counter() - t0
                res["bound_pass"][f"workers_{w}_seconds"] = round(tw, 3)
                res["bound_pass"][f"workers_{w}_speedup"] = round(t1 / tw, 2)
                res["bound_pass"][f"workers_{w}_same_status"] = bool((par == ref).all())
                res["search"][f"workers_{w}"] = search(B, A, G, d2, sets, lb, beta, p, w)
        shipped = res["search"]["committed_batched"]
        for r in res["search"].values():
            r["same_design"] = r["treated"] == shipped["treated"] and r["value"] == shipped["value"]
        out[str(beta)] = res
        print(f"beta={beta:g}:", json.dumps(res["bound_pass"]), flush=True)
        for k, r in res["search"].items():
            print(f"  {k}: {r['seconds']} s, {r['coupled_fits']} fits, same design {r['same_design']}",
                  flush=True)
    (panel.RESULTS / "pool.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
