"""The batched search with the fits in the workers too, issue #768's proposal.

Each worker runs :func:`contiguous.bound_and_fit` on a batch of 512 candidates
in ``lb`` order: it bounds them against the best value known when the batch was
sent, fits what the bound leaves, and returns the batch's best design. The
parent fits the first 8 candidates to set an incumbent, merges the batches'
best designs with the search's tie rule (value, then treated set), and sends
each new batch with the best value it knows. Two schedules:

* rounds: as many batches as there are workers, sent together; the next round
  waits for all of them;
* streaming: a fixed number of batches in flight; when one returns, its design
  is merged and the next batch is sent with the current best.

A stale best only lets more candidates through to a fit, so the design must not
depend on the workers or the schedule; the script records that it does not. The
baseline is the streaming logic run inline in one process, one batch at a time.

Workers fork, run single-threaded, and get the panel once through the pool's
initializer; each pool is started before the timer. From the repository root,
after ``search.py``::

    python benchmarks/studies/marex_coupling_bound/parallel.py
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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402

from benchmarks.studies.marex_coupling_bound import bound, contiguous, panel  # noqa: E402
from mlsynth.utils.marex_helpers import exact  # noqa: E402

BATCH = 512
SEED = 8
WORKERS = (2, 4)


class Inline:
    """A stand-in for a pool that runs each task when it is submitted."""

    class _Done:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    def submit(self, fn, arg):
        return self._Done(fn(arg))


def better(v, S, best, best_S):
    return S is not None and (v < best or (v == best and (best_S is None or S < best_S)))


def seed(B, A, sets, order, beta):
    n = B.shape[1]
    best, best_S = np.inf, None
    for k in order[:SEED]:
        S = [int(s) for s in sets[k]]
        g, w = exact.side_fit(B, A, S)
        if g >= best:
            continue
        C = [j for j in range(n) if j not in S]
        cut = best + contiguous.margin(best) if np.isfinite(best) else None
        fit = exact.coupled_fit(B, A, S, C, beta, w0=w, cutoff=cut)
        if fit is not None and better(fit[0], tuple(S), best, best_S):
            best, best_S = fit[0], tuple(S)
    return best, best_S


def search(B, A, sets, lb, beta, pool, in_flight, rounds):
    t0 = time.perf_counter()
    order = np.argsort(lb, kind="stable")
    best, best_S = seed(B, A, sets, order, beta)
    fits, discarded, sent, pos = SEED, 0, 0, SEED

    def next_batch():
        nonlocal pos, sent
        while pos < len(order):
            cut = best + contiguous.margin(best)
            blk = order[pos:pos + BATCH]
            pos += BATCH
            blk = blk[lb[blk] < cut]
            if blk.size:
                sent += 1
                return pool.submit(contiguous.bound_and_fit, (sets[blk], lb[blk], best, beta))
        return None

    def merge(result):
        nonlocal best, best_S, fits, discarded
        v, S, f, d = result
        fits += f
        discarded += d
        if better(v, S, best, best_S):
            best, best_S = v, S

    if rounds:
        while True:
            futures = [f for f in (next_batch() for _ in range(in_flight)) if f is not None]
            if not futures:
                break
            for f in futures:
                merge(f.result())
    else:
        pending = {f for f in (next_batch() for _ in range(in_flight)) if f is not None}
        while pending:
            done, pending = (wait(pending, return_when=FIRST_COMPLETED)
                             if not isinstance(pool, Inline) else (pending, set()))
            for f in done:
                merge(f.result())
            for _ in range(len(done)):
                f = next_batch()
                if f is not None:
                    pending.add(f)
    return {"seconds": round(time.perf_counter() - t0, 2), "coupled_fits": fits,
            "discarded_by_bound": discarded, "batches": sent,
            "value": best, "treated": list(best_S)}


def main():
    B, A = panel.geolift()
    Bc = B - A[:, None]
    G = Bc.T @ Bc
    d2 = bound.squared_pair_distances(B)
    sets = np.array(list(itertools.combinations(range(B.shape[1]), panel.M)))
    lb = exact.treated_fit_bound(B, A, sets)
    ctx = mp.get_context("fork")
    contiguous.init_fit_worker(B, A, G, d2)
    out = {"cores": os.cpu_count()}
    for beta in panel.BETAS:
        runs = {"one_process": search(B, A, sets, lb, beta, Inline(), 1, rounds=False)}
        for w in WORKERS:
            with ProcessPoolExecutor(max_workers=w, mp_context=ctx,
                                     initializer=contiguous.init_fit_worker,
                                     initargs=(B, A, G, d2)) as p:
                warm = [(sets[:4], lb[:4], np.inf, beta)] * w
                list(p.map(contiguous.bound_and_fit, warm))
                runs[f"rounds_{w}_workers"] = search(B, A, sets, lb, beta, p, w, rounds=True)
                runs[f"streaming_{w}_workers_{w}_in_flight"] = search(
                    B, A, sets, lb, beta, p, w, rounds=False)
                runs[f"streaming_{w}_workers_{2 * w}_in_flight"] = search(
                    B, A, sets, lb, beta, p, 2 * w, rounds=False)
        ref = runs["one_process"]
        for r in runs.values():
            r["same_design"] = r["treated"] == ref["treated"] and r["value"] == ref["value"]
            r["speedup"] = round(ref["seconds"] / r["seconds"], 2)
        out[str(beta)] = runs
        for k, r in runs.items():
            print(f"beta={beta:g} {k}: {r['seconds']} s, x{r['speedup']}, "
                  f"{r['coupled_fits']} fits, same design {r['same_design']}", flush=True)
    (panel.RESULTS / "parallel.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
