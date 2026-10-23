"""Run the scale study's grid and append one JSON line per measurement.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python -m benchmarks.studies.bnb_scale.run syndes --kinds bls noise flat --sizes 40 60 80 100
    python -m benchmarks.studies.bnb_scale.run marex-bound --sizes 45 60 80 100
    python -m benchmarks.studies.bnb_scale.run marex-search --sizes 60 --m 6
    python -m benchmarks.studies.bnb_scale.run root --kinds bls noise flat --sizes 40 60 100
    python -m benchmarks.studies.bnb_scale.run trace --kinds bls --sizes 40 60
    python -m benchmarks.studies.bnb_scale.run profile --kinds bls --sizes 40 60 80 100
    python -m benchmarks.studies.bnb_scale.run depth --kinds bls noise flat --sizes 60 80 100 --closure-at 20 40 --projection

``root``, ``trace`` and ``depth`` read each panel's best design from
``results/syndes.jsonl``, so they run after ``syndes`` has. ``trace`` runs the
root lift three ways (shipped, ``root_gap.LONG`` and ``root_gap.NO_STALL``) and
records the rule that ended each; ``profile`` splits one root lift's time;
``depth`` bounds nodes on the path to the design by their count of fixed units
(``depth.depth_table``), with the triangle closure at the counts in
``--closure-at`` and, with ``--projection``, the projected fit where the
least-norm one is not semidefinite.

Each line carries the load average before and after, since the CPU seconds are
only comparable on an unloaded machine (#840 reported its load the same way).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def _append(name: str, row: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"{name}.jsonl", "a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k not in ("treated", "stopped")},
                     default=float), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["syndes", "marex-bound", "marex-search", "root",
                                     "trace", "profile", "depth"])
    ap.add_argument("--kinds", nargs="+", default=["bls"])
    ap.add_argument("--sizes", nargs="+", type=int, required=True)
    ap.add_argument("--K", type=int, default=None, help="SYNDES size; default N/2")
    ap.add_argument("--m", type=int, default=6, help="MAREX treated count")
    ap.add_argument("--work-limit", type=float, default=None)
    ap.add_argument("--closure", action="store_true",
                    help="root: also solve the triangle relaxation by interior point")
    ap.add_argument("--fixed", nargs="+", type=int,
                    default=[0, 5, 10, 20, 30, 40, 50, 60, 70],
                    help="depth: the counts of fixed units")
    ap.add_argument("--reps", type=int, default=3, help="depth: nodes per count")
    ap.add_argument("--closure-at", nargs="*", type=int, default=[],
                    help="depth: the counts whose first node gets the triangle closure")
    ap.add_argument("--projection", action="store_true",
                    help="depth: also fit by projection where the least-norm fit is not semidefinite")
    args = ap.parse_args(argv)

    from benchmarks.studies.bnb_scale import depth, marex_scale, panels, root_gap, syndes_scale
    from mlsynth.utils.syndes_helpers.exact import WORK_LIMIT

    warnings.simplefilter("ignore")
    for kind in args.kinds:
        for N in args.sizes:
            Y = panels.panel(kind, N)
            load0 = os.getloadavg()[0]
            if args.what in ("root", "trace", "depth"):
                K = args.K or N // 2
                known = [json.loads(l) for l in open(RESULTS / "syndes.jsonl")]
                best = min((r for r in known
                            if r["kind"] == kind and r["N"] == N and r["K"] == K),
                           key=lambda r: r["value"])
                incumbent = best["value"]
            if args.what == "depth":
                for row in depth.depth_table(Y, K, best["treated"], fixed=args.fixed,
                                             reps=args.reps, closure_at=args.closure_at,
                                             projection=args.projection):
                    row.update(kind=kind, load_before=load0, load_after=os.getloadavg()[0])
                    _append("depth", row)
                continue
            if args.what == "trace":
                for name, values in (("shipped", {}), ("long", root_gap.LONG),
                                     ("no_stall", root_gap.NO_STALL)):
                    t0 = time.process_time()
                    row = root_gap.trace_lift(Y, K, **values)
                    row.update(run=name, seconds=time.process_time() - t0,
                               incumbent=incumbent,
                               gap=(incumbent - row["lower"]) / incumbent,
                               rounds=len(row["rounds"]), kind=kind,
                               load_before=load0, load_after=os.getloadavg()[0])
                    _append("trace", row)
                continue
            if args.what == "profile":
                row = syndes_scale.profile_lift(Y, args.K or N // 2)
            elif args.what == "syndes":
                K = args.K or N // 2
                row = syndes_scale.measure(Y, K, work_limit=args.work_limit or WORK_LIMIT)
            elif args.what == "root":
                row = root_gap.root_bounds(Y, K, incumbent=incumbent)
                if args.closure:
                    tri = root_gap.triangle_closure(Y, K)
                    row.update({f"closure_{k}": v for k, v in tri.items()
                                if k not in ("N", "K")})
                    row["closure_gap"] = (row["incumbent"] - tri["lower"]) / row["incumbent"]
            elif args.what == "marex-bound":
                row = marex_scale.root_bound(Y, args.m)
            else:
                row = marex_scale.search(Y, args.m, **(
                    {} if args.work_limit is None else {"work_limit": args.work_limit}))
            row.update(kind=kind, load_before=load0, load_after=os.getloadavg()[0])
            _append("lift_profile" if args.what == "profile" else args.what, row)
    return 0


if __name__ == "__main__":  # pragma: no cover - the command line's entry; main is tested directly
    raise SystemExit(main())
