"""Race every candidate on every family, and score them on the right things.

    python -m benchmarks.studies.simplex_shootout.run
    python -m benchmarks.studies.simplex_shootout.run --reps 9 --budget 4.0

This is a study, not a benchmark case: timings are machine-dependent and nothing
here is pinned. What it produces is a recommendation about which method belongs
behind ``mlsynth.utils.weights.solve_weights`` for which regime.

Scoring. The objective is the only quantity every candidate can be compared on:
where the minimiser is a face -- the whole ``degenerate`` family, and much of
``gaussian`` -- which point comes back is a property of the method, so weights are
reported and never scored. The reference objective is the best any candidate
attains, so a candidate that beats the oracle is visible instead of hidden.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from .algorithms import ALGORITHMS, _obj
from .panels import by_family

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def _timed(fn, B, A, reps, budget):
    """Best-of-``reps`` wall clock, stopping early once ``budget`` is spent."""
    try:
        first = fn(B, A)
    except Exception as exc:                       # a candidate that cannot run
        return None, float("nan"), f"error:{type(exc).__name__}"
    best = float("inf")
    spent = 0.0
    out = first
    for _ in range(reps):
        t = time.perf_counter()
        out = fn(B, A)
        dt = time.perf_counter() - t
        best = min(best, dt)
        spent += dt
        if spent > budget:
            break
    return out, best * 1e6, out.status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--budget", type=float, default=3.0,
                    help="seconds per candidate per panel")
    ap.add_argument("--only", default=None, help="comma-separated family filter")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    fams = by_family()
    if args.only:
        keep = set(args.only.split(","))
        fams = {k: v for k, v in fams.items() if k in keep}

    names = list(ALGORITHMS)
    rows = []
    for fam, panels in fams.items():
        for p in panels:
            got = {}
            for n in names:
                cand, us, status = _timed(ALGORITHMS[n], p.B, p.A,
                                          args.reps, args.budget)
                got[n] = (cand, us, status)
            objs = {n: (_obj(p.B, p.A, c.w) if c is not None else float("inf"))
                    for n, (c, _, _) in got.items()}
            ref = min(objs.values())
            scale = max(float(p.A @ p.A), 1e-300)
            for n in names:
                cand, us, status = got[n]
                w = cand.w if cand is not None else None
                rows.append({
                    "family": fam, "panel": p.name,
                    "m": int(p.B.shape[0]), "J": int(p.B.shape[1]),
                    "algorithm": n, "us": us,
                    "iterations": int(cand.iterations) if cand else -1,
                    "status": status,
                    # scale-free objective excess over the best any method found
                    "excess": (objs[n] - ref) / scale,
                    "feasible": bool(
                        w is not None and w.min() >= -1e-9
                        and abs(w.sum() - 1.0) <= 1e-7 and np.all(np.isfinite(w))),
                    "support": int((w > 1e-9).sum()) if w is not None else -1,
                })
            print(f"  {fam}/{p.name} ({p.B.shape[0]}x{p.B.shape[1]}) done")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "raw.json").write_text(json.dumps(rows, indent=1))

    # ---- per-family summary -------------------------------------------- #
    print()
    hdr = f"{'algorithm':<18s}" + "".join(f"{f[:9]:>11s}" for f in fams) + f"{'TOTAL':>11s}"
    print("total microseconds per family (lower is better)")
    print(hdr)
    tot = {}
    for n in names:
        cells = []
        grand = 0.0
        for fam in fams:
            s = sum(r["us"] for r in rows if r["algorithm"] == n and r["family"] == fam)
            cells.append(f"{s:11.0f}" if np.isfinite(s) else f"{'nan':>11s}")
            grand += s if np.isfinite(s) else 0.0
        tot[n] = grand
        print(f"{n:<18s}" + "".join(cells) + f"{grand:11.0f}")
    print()
    print("correctness: panels where the objective is above the best found, "
          "by more than 1e-9 of ||A||^2")
    print(f"{'algorithm':<18s} {'wrong':>6s} {'infeasible':>11s} {'unconverged':>12s} "
          f"{'worst excess':>13s}")
    for n in names:
        rs = [r for r in rows if r["algorithm"] == n]
        wrong = sum(1 for r in rs if r["excess"] > 1e-9)
        infeas = sum(1 for r in rs if not r["feasible"])
        unconv = sum(1 for r in rs if r["status"] in ("maxiter", "failed")
                     or r["status"].startswith("error"))
        worst = max((r["excess"] for r in rs), default=float("nan"))
        print(f"{n:<18s} {wrong:6d} {infeas:11d} {unconv:12d} {worst:13.2e}")
    print()
    print(f"raw rows -> {RESULTS / 'raw.json'}")


if __name__ == "__main__":
    main()
