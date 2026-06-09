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

from benchmarks import registry                          # noqa: E402
from benchmarks.compare import compare, BenchmarkSkipped  # noqa: E402


def run_one(name: str):
    """Run a single case. Returns True (pass), False (fail), or None (skipped)."""
    mod = registry.load(name)
    t0 = time.time()
    try:
        got = mod.run()
    except BenchmarkSkipped as exc:
        print(f"\n[SKIP] {name}   ({exc})")
        return None
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
    n_pass = sum(v is True for v in results.values())
    n_skip = sum(v is None for v in results.values())
    n_fail = sum(v is False for v in results.values())
    tail = f" ({n_skip} skipped)" if n_skip else ""
    print(f"\n==== {n_pass}/{n_pass + n_fail} benchmarks passed{tail} ====")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
