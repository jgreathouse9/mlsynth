"""Where the per-candidate bound's time goes, at beta = 100.

Runs :func:`bound.hull_distance_bound` on 600 candidates the closed-form bound
leaves under the optimum (those whose treated fit is under it too), under
cProfile, and times the two operations a step is made of on their own: one
least-squares solve on a 90-by-8 corral and one product ``B'x``.

From the repository root, after ``search.py``::

    python benchmarks/studies/marex_coupling_bound/profile_bound.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import cProfile
import itertools
import json
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402

from benchmarks.studies.marex_coupling_bound import bound, panel  # noqa: E402
from mlsynth.utils.marex_helpers import exact  # noqa: E402

BETA = 100.0


def main():
    B, A = panel.geolift()
    n = B.shape[1]
    fstar = panel.optimum(BETA)
    cut = fstar + exact.PRUNE_MARGIN * (1.0 + fstar)
    sets = np.array(list(itertools.combinations(range(n), panel.M)))
    lb = exact.treated_fit_bound(B, A, sets)
    pick = np.random.default_rng(5).choice(np.flatnonzero(lb < cut), 600, replace=False)
    d2 = bound.squared_pair_distances(B)
    work, t_g = [], 0.0
    for k in pick:
        S = [int(s) for s in sets[k]]
        t0 = time.perf_counter()
        g = exact.side_fit(B, A, S)[0]
        t_g += time.perf_counter() - t0
        if g < cut:
            work.append((S, [j for j in range(n) if j not in S], (cut - g) / BETA))
    cycles = []
    t0 = time.perf_counter()
    for S, C, need in work:
        cycles.append(bound.hull_distance_bound(B, S, C, need, pair_d2=d2)[1])
    t_bound = time.perf_counter() - t0
    prof = cProfile.Profile()
    prof.enable()
    for S, C, need in work:
        bound.hull_distance_bound(B, S, C, need, pair_d2=d2)
    prof.disable()
    stats = pstats.Stats(prof).stats
    total = sum(v[2] for v in stats.values())          # total internal time
    lstsq = [v for k, v in stats.items() if k[2] == "lstsq"][0]
    body = [v for k, v in stats.items() if k[2] == "hull_distance_bound"][0]
    Z = np.random.default_rng(0).normal(size=(B.shape[0], 8))
    t0 = time.perf_counter()
    for _ in range(2000):
        np.linalg.lstsq(Z, Z[:, 0], rcond=None)
    one_lstsq = (time.perf_counter() - t0) / 2000
    x = np.ones(B.shape[0])
    t0 = time.perf_counter()
    for _ in range(20000):
        B.T @ x
    one_proj = (time.perf_counter() - t0) / 20000
    out = {"candidates": len(work),
           "ms_treated_fit": round(t_g / len(pick) * 1e3, 3),
           "ms_bound": round(t_bound / len(work) * 1e3, 3),
           "major_cycles_median_mean": [int(np.median(cycles)), round(float(np.mean(cycles)), 1)],
           "lstsq_calls_per_candidate": round(lstsq[1] / len(work), 1),
           "share_in_lstsq": round(lstsq[3] / total, 3),
           "share_in_bound_body": round(body[2] / total, 3),
           "us_one_lstsq_90x8": round(one_lstsq * 1e6, 1),
           "us_one_projection": round(one_proj * 1e6, 2)}
    print(json.dumps(out), flush=True)
    (panel.RESULTS / "profile_bound.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
