"""Run the scale study's grid and append one JSON line per measurement.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python -m benchmarks.studies.bnb_scale.run syndes --kinds bls noise flat --sizes 40 60 80 100
    python -m benchmarks.studies.bnb_scale.run marex-bound --sizes 45 60 80 100
    python -m benchmarks.studies.bnb_scale.run marex-search --sizes 60 --m 6 --max-lifts 40

Each line carries the load average before and after, since the CPU seconds are
only comparable on an unloaded machine (#840 reported its load the same way).
"""
from __future__ import annotations

import argparse
import json
import os
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
    ap.add_argument("what", choices=["syndes", "marex-bound", "marex-search"])
    ap.add_argument("--kinds", nargs="+", default=["bls"])
    ap.add_argument("--sizes", nargs="+", type=int, required=True)
    ap.add_argument("--K", type=int, default=None, help="SYNDES size; default N/2")
    ap.add_argument("--m", type=int, default=6, help="MAREX treated count")
    ap.add_argument("--max-lifts", type=int, default=40)
    ap.add_argument("--work-limit", type=float, default=None)
    args = ap.parse_args(argv)

    from benchmarks.studies.bnb_scale import marex_scale, panels, syndes_scale
    from mlsynth.utils.syndes_helpers.exact import WORK_LIMIT

    warnings.simplefilter("ignore")
    for kind in args.kinds:
        for N in args.sizes:
            Y = panels.panel(kind, N)
            load0 = os.getloadavg()[0]
            if args.what == "syndes":
                K = args.K or N // 2
                row = syndes_scale.measure(Y, K, work_limit=args.work_limit or WORK_LIMIT)
            elif args.what == "marex-bound":
                row = marex_scale.root_bound(Y, args.m)
            else:
                row = marex_scale.search(Y, args.m, max_lifts=args.max_lifts)
            row.update(kind=kind, load_before=load0, load_after=os.getloadavg()[0])
            _append(args.what, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
