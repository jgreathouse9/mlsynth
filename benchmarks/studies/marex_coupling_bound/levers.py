"""Two levers on the batched bound's cost: fewer copies, and warm starts.

Every candidate the closed-form bound leaves under the optimum ``search.py``
recorded, bounded by ``lb + beta * dist_lb^2`` as in ``batched_curve.py``, in
lexicographic order.

Fewer copies: :func:`contiguous.wolfe_bound` against
:func:`batched.wolfe_bound` at three batch sizes. The algorithm is the same, so
the statuses must agree everywhere; the script records that they do.

Warm starts: candidates sharing their first ``m - 1`` markets form a chain,
consecutive in lexicographic order. Round ``r`` bounds the ``r``-th candidate of
every chain at once, in batches of 512, each started from its predecessor's
final corral with the points that stay valid once one treated market is swapped
(``i`` still treated, ``j`` still a control): all of them, the 3 heaviest, or
the heaviest, weights renormalised. A row with none left starts cold. The cold
run bounds the same candidates in lexicographic batches of 512.

Single-threaded. From the repository root, after ``search.py``::

    python benchmarks/studies/marex_coupling_bound/levers.py
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

from benchmarks.studies.marex_coupling_bound import batched, bound, contiguous, panel  # noqa: E402
from mlsynth.utils.marex_helpers import exact  # noqa: E402

BATCH = 512
SIZES = (512, 2048, 8192)
STARTS = ("cold", "all", "top3", "top1")


def candidates(B, A, beta):
    sets = np.array(list(itertools.combinations(range(B.shape[1]), panel.M)))
    lb = exact.treated_fit_bound(B, A, sets)
    fstar = panel.optimum(beta)
    cut = fstar + exact.PRUNE_MARGIN * (1.0 + abs(fstar))
    keep = np.flatnonzero(lb < cut)
    return sets[keep], (cut - lb[keep]) / beta


def fewer_copies(G, d2, S, need):
    out = {}
    ref = np.empty(len(S), dtype=int)
    t0 = time.perf_counter()
    for lo in range(0, len(S), BATCH):
        ref[lo:lo + BATCH] = batched.wolfe_bound(G, S[lo:lo + BATCH], need[lo:lo + BATCH], d2)[1]
    out["committed_batch_512_us"] = round((time.perf_counter() - t0) / len(S) * 1e6, 1)
    for size in SIZES:
        st = np.empty(len(S), dtype=int)
        t0 = time.perf_counter()
        for lo in range(0, len(S), size):
            st[lo:lo + size] = contiguous.wolfe_bound(G, S[lo:lo + size], need[lo:lo + size], d2)[1]
        out[f"contiguous_batch_{size}_us"] = round((time.perf_counter() - t0) / len(S) * 1e6, 1)
        out[f"contiguous_batch_{size}_same_status"] = bool((st == ref).all())
    out["cleared"] = int((ref == 1).sum())
    return out


def inherit(S_new, prev_state, G, d2, how):
    """The valid points of the predecessors' corrals, packed to the front."""
    I0, J0, V0, L0 = (x.copy() for x in prev_state)
    b, n = len(S_new), G.shape[0]
    ar = np.arange(b)[:, None]
    M = np.zeros((b, n), dtype=bool)
    M[ar, S_new] = True
    keep = V0 & M[ar, I0] & ~M[ar, J0]
    Lk = np.where(keep, L0, 0.0)
    if how != "all":
        top = 3 if how == "top3" else 1
        place = np.argsort(np.argsort(-Lk, axis=1, kind="stable"), axis=1, kind="stable")
        keep &= place < top
        Lk = np.where(keep, Lk, 0.0)
    tot = Lk.sum(axis=1)
    empty = ~keep.any(axis=1) | (tot <= 0)
    keep[empty] = False
    Lk = np.where(keep, Lk / np.where(tot > 0, tot, 1.0)[:, None], 0.0)
    if empty.any():
        e = np.flatnonzero(empty)
        i0, j0 = contiguous.closest_pairs(S_new[e], M[e], d2)
        I0[e, 0], J0[e, 0], keep[e, 0], Lk[e, 0] = i0, j0, True, 1.0
    order = np.argsort(~keep, axis=1, kind="stable")
    packed = tuple(np.take_along_axis(x, order, 1) for x in (I0, J0, keep, Lk))
    return packed, int(empty.sum())


def warm_starts(G, d2, S, need):
    N = len(S)
    pre = S[:, :-1]
    first = np.ones(N, dtype=bool)
    first[1:] = np.any(pre[1:] != pre[:-1], axis=1)
    chain = np.cumsum(first) - 1
    rank = np.arange(N) - np.flatnonzero(first)[chain]
    out = {"chains": int(first.sum()), "longest_chain": int(rank.max()) + 1}
    for how in STARTS:
        status = np.empty(N, dtype=int)
        cycles = np.empty(N, dtype=np.int64)
        K = 24
        Is = np.zeros((N, K), dtype=np.int64)
        Js = np.zeros((N, K), dtype=np.int64)
        Vs = np.zeros((N, K), dtype=bool)
        Ls = np.zeros((N, K))
        sizes, restarted = [], 0
        rounds = ([np.arange(N)] if how == "cold"
                  else [np.flatnonzero(rank == r) for r in range(int(rank.max()) + 1)])
        t0 = time.perf_counter()
        for r, rows in enumerate(rounds):
            for lo in range(0, len(rows), BATCH):
                rw = rows[lo:lo + BATCH]
                init = None
                if how != "cold" and r > 0:
                    prev = rw - 1
                    init, k = inherit(S[rw], (Is[prev], Js[prev], Vs[prev], Ls[prev]), G, d2, how)
                    restarted += k
                    sizes.append(init[2].sum(axis=1))
                _, st, corral, cyc = contiguous.wolfe_bound(G, S[rw], need[rw], d2, init=init)
                status[rw], cycles[rw] = st, cyc
                Is[rw], Js[rw], Vs[rw], Ls[rw] = corral
        secs = time.perf_counter() - t0
        out[how] = {"us": round(secs / N * 1e6, 1), "cleared": int((status == 1).sum()),
                    "major_steps_mean": round(float(cycles.mean()), 2),
                    "inherited_points_mean": (round(float(np.concatenate(sizes).mean()), 2)
                                              if sizes else None),
                    "rows_restarted_cold": restarted}
    return out


def main():
    B, A = panel.geolift()
    Bc = B - A[:, None]
    G = Bc.T @ Bc
    d2 = bound.squared_pair_distances(B)
    out = {}
    for beta in panel.BETAS:
        S, need = candidates(B, A, beta)
        out[str(beta)] = {"candidates": int(len(S)),
                          "fewer_copies": fewer_copies(G, d2, S, need),
                          "warm_starts": warm_starts(G, d2, S, need)}
        print(f"beta={beta:g}:", json.dumps(out[str(beta)]), flush=True)
    (panel.RESULTS / "levers.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
