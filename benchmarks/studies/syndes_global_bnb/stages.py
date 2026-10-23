"""Where the search's time goes: seconds per stage, and the lift's cost model.

Each stage function of ``search.py`` is wrapped and charged its own time minus
the time of the wrapped stages it calls; calibration is charged everything it
calls. The rest of the run is the tree loop and building blocks. The lift's
calibrated cost (a power law fitted at 10 and 20 free units) is printed beside
its measured cost at each size it ran.

From the repository root::

    python -m benchmarks.studies.syndes_global_bnb.stages geolift 40 10
    python -m benchmarks.studies.syndes_global_bnb.stages factor 60 6 nolift
    python -m benchmarks.studies.syndes_global_bnb.stages geolift 40 10 dual

``dual`` lifts by the regularised dual of ``dual.py``.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
from collections import defaultdict
from math import comb

import numpy as np

from mlsynth.utils.syndes_helpers import gram

from . import search as S
from .run import PANELS

STAGES = (("_price_seconds", "calibrate: price"),
          ("_bound_seconds", "calibrate: bound"),
          ("_lift_seconds", "calibrate: lift"),
          ("maxcut_candidates", "seed: swap search"),
          ("_settle", "settle (seeds + final)"),
          ("trs_upper", "root order (TRS)"),
          ("node_upper", "node bound (TRS + spectral)"),
          ("lift_upper", "node lift"),
          ("dual_upper", "node lift"),
          ("signs_from_subsets", "price: signs"),
          ("_combos", "price: combination tables"),
          ("solve_partition_batch", "final projected gradient"))


def main(argv):
    name, N, K = argv[0], int(argv[1]), int(argv[2])
    own, calls, stack = defaultdict(float), defaultdict(int), []
    calibrating = [False]

    def wrap(label, fn):
        calib = label.startswith("calibrate")

        def inner(*a, **k):
            if calibrating[0]:
                return fn(*a, **k)
            calibrating[0] = calib
            t = time.perf_counter()
            stack.append(0.0)
            try:
                return fn(*a, **k)
            finally:
                dt = time.perf_counter() - t
                calibrating[0] = False
                own[label] += dt - stack.pop()
                calls[label] += 1
                if stack:
                    stack[-1] += dt
        return inner

    saved = {attr: getattr(S, attr) for attr, _ in STAGES}
    saved_bound = gram.BoundEngine.bound
    model, measured = {}, defaultdict(list)
    try:
        for attr, label in STAGES:
            setattr(S, attr, wrap(label, getattr(S, attr)))
        gram.BoundEngine.bound = wrap("price: closed form", saved_bound)
        name_lift = "dual_upper" if "dual" in argv else "lift_upper"
        lift_seconds, lift_fn = S._lift_seconds, getattr(S, name_lift)

        def fitted(R, K, lift=None):
            model["lift"] = lift_seconds(R, K, lift) if lift is not None else lift_seconds(R, K)
            return model["lift"]

        def timed(R, I, O, F, K, *a, **k):
            t = time.perf_counter()
            out = lift_fn(R, I, O, F, K, *a, **k)
            if not calibrating[0]:
                measured[len(F)].append(time.perf_counter() - t)
            return out

        S._lift_seconds = fitted
        setattr(S, name_lift, timed)
        kw = {"lift_cost": np.inf} if "nolift" in argv else {}
        if "dual" in argv:
            kw["lift"] = "dual"
        Y = PANELS[name](N)
        t = time.perf_counter()
        r = S.search(Y, K, **kw)
        total = time.perf_counter() - t
    finally:
        for attr, fn in saved.items():
            setattr(S, attr, fn)
        gram.BoundEngine.bound = saved_bound

    print(f"{name} N={N} K={K}{' nolift' if 'nolift' in argv else ''}{' dual' if 'dual' in argv else ''}: "
          f"{total:.1f}s value={r.value:.10g} "
          f"bounded={r.nodes_bounded} lifted={r.nodes_lifted} pruned={r.nodes_pruned} "
          f"priced={r.sets_priced:,} of {comb(N, K):,} ({r.sets_priced / comb(N, K):.2%})")
    rows = sorted(own.items(), key=lambda kv: -kv[1])
    rows.append(("tree loop + block building", total - sum(own.values())))
    for label, s in rows:
        print(f"  {label:30s} {s:8.2f}s {s / total:6.1%}  calls={calls.get(label, '')}")
    for n in sorted(measured):
        print(f"  lift at n={n}: {len(measured[n])} calls, {np.mean(measured[n]):.2f}s each, "
              f"calibrated {model['lift'](n):.2f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
