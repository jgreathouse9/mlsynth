#!/usr/bin/env python3
"""Driver: run registered benchmark cases and report pass/fail vs tolerance.

    python benchmarks/run_benchmarks.py --all
    python benchmarks/run_benchmarks.py --case fdid_table5
    python benchmarks/run_benchmarks.py --with-reference   # include R cross-checks
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# allow "python benchmarks/run_benchmarks.py" from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import registry            # noqa: E402
from benchmarks.compare import compare     # noqa: E402


def run_one(name: str) -> bool:
    mod = registry.load(name)
    t0 = time.time()
    got = mod.run()
    ok, report = compare(got, mod.EXPECTED)
    print(f"\n[{'PASS' if ok else 'FAIL'}] {name}   ({time.time()-t0:.0f}s)")
    print(report)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--case", default=None)
    ap.add_argument("--with-reference", action="store_true",
                    help="also run cases that require an R reference implementation")
    args = ap.parse_args()

    names = ([args.case] if args.case else list(registry.CASES))
    if not args.with_reference:
        names = [n for n in names if n not in registry.NEEDS_REFERENCE]

    results = {n: run_one(n) for n in names}
    n_pass = sum(results.values())
    print(f"\n==== {n_pass}/{len(results)} benchmarks passed ====")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
