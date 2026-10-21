"""The batched bounds per candidate: how many they clear, at what cost, how valid.

With the cutoff at the optimum ``search.py`` recorded, every candidate the
closed-form bound leaves under it is bounded by ``lb + beta * dist_lb^2``, the
closed-form treated bound in place of the treated fit, so no candidate needs a
simplex fit to be bounded. Recorded per ``beta``:

* the ceiling: on 1,000 sampled candidates, the exact distance (Wolfe's method
  run to convergence) with ``lb`` and with the treated fit ``g``, and the share
  of the exact distance a bound must reach;
* Frank-Wolfe, plain and pairwise, 200 iterations at most: how many candidates
  clear within 1 to 200;
* batched Wolfe at three batch sizes: how many clear, and the cost;
* validity: batched Wolfe run to convergence on 300 sampled candidates against
  the per-candidate exact distance.

From the repository root, after ``search.py``::

    python benchmarks/studies/marex_coupling_bound/batched_curve.py
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
from mlsynth.utils.marex_helpers import exact  # noqa: E402

CEILING_SAMPLE = 1000
VALIDITY_SAMPLE = 300
BATCHES = (512, 2048, 8192)
FW_CHECKPOINTS = (1, 2, 5, 10, 20, 50, 100, 200)


def measure(B, A, beta):
    n = B.shape[1]
    Bc = B - A[:, None]
    G = Bc.T @ Bc
    d2 = bound.squared_pair_distances(B)
    fstar = panel.optimum(beta)
    cut = fstar + exact.PRUNE_MARGIN * (1.0 + abs(fstar))
    sets = np.array(list(itertools.combinations(range(n), panel.M)))
    lb = exact.treated_fit_bound(B, A, sets)
    live = np.flatnonzero(lb < cut)
    S, need = sets[live], (cut - lb[live]) / beta
    out = {"optimum": fstar, "under_optimum_closed_form": int(len(live))}

    rng = np.random.default_rng(4)
    pick = rng.choice(len(live), min(CEILING_SAMPLE, len(live)), replace=False)
    left_lb = left_g = 0
    frac = []
    for p in pick:
        s = [int(x) for x in S[p]]
        c = [j for j in range(n) if j not in s]
        d = bound.hull_distance_bound(B, s, c, np.inf, pair_d2=d2)[0]
        g = exact.side_fit(B, A, s)[0]
        left_lb += lb[live[p]] + beta * d * d < cut
        left_g += g + beta * d * d < cut
        frac.append(np.sqrt(need[p]) / d)
    scale = len(live) / len(pick)
    out["ceiling"] = {"sampled": int(len(pick)),
                      "left_with_lb": int(left_lb), "left_with_g": int(left_g),
                      "left_with_lb_projected": int(round(left_lb * scale)),
                      "required_share_of_distance_q10_q50_q90_q99":
                          [round(float(q), 3) for q in np.quantile(frac, [.1, .5, .9, .99])]}

    for pairwise in (False, True):
        t0 = time.perf_counter()
        at = np.empty(len(S), dtype=int)
        for lo in range(0, len(S), 8192):
            at[lo:lo + 8192] = batched.frank_wolfe_bound(
                B, S[lo:lo + 8192], need[lo:lo + 8192], d2, pairwise=pairwise)[1]
        secs = time.perf_counter() - t0
        out["frank_wolfe_" + ("pairwise" if pairwise else "plain")] = {
            "cleared_within": {str(k): int(((at >= 0) & (at < k)).sum()) for k in FW_CHECKPOINTS},
            "left_after_200": int((at < 0).sum()),
            "us_per_candidate": round(secs / len(S) * 1e6, 1)}

    for size in BATCHES:
        t0 = time.perf_counter()
        st = np.empty(len(S), dtype=int)
        for lo in range(0, len(S), size):
            st[lo:lo + size] = batched.wolfe_bound(G, S[lo:lo + size], need[lo:lo + size], d2)[1]
        secs = time.perf_counter() - t0
        out[f"wolfe_batch_{size}"] = {
            "cleared": int((st == 1).sum()), "left": int((st != 1).sum()),
            "status_counts": {str(k): int(v) for k, v in zip(*np.unique(st, return_counts=True))},
            "us_per_candidate": round(secs / len(S) * 1e6, 1)}

    pick = rng.choice(len(live), min(VALIDITY_SAMPLE, len(live)), replace=False)
    bb, st = batched.wolfe_bound(G, S[pick], np.full(len(pick), np.inf), d2, max_iter=300)
    ratio = []
    for k, p in enumerate(pick):
        s = [int(x) for x in S[p]]
        c = [j for j in range(n) if j not in s]
        ratio.append(bb[k] / bound.hull_distance_bound(B, s, c, np.inf, pair_d2=d2)[0])
    ratio = np.array(ratio)
    out["validity"] = {"sampled": int(len(pick)), "max_bound_over_exact": float(ratio.max()),
                       "median_abs_relative_gap": float(np.median(np.abs(ratio - 1.0))),
                       "status_counts": {str(k): int(v) for k, v in zip(*np.unique(st, return_counts=True))}}
    return out


def main():
    B, A = panel.geolift()
    out = {}
    for beta in panel.BETAS:
        out[str(beta)] = measure(B, A, beta)
        print(f"beta={beta:g}:", json.dumps(out[str(beta)]), flush=True)
    (panel.RESULTS / "batched_curve.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
