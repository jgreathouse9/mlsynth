"""Measure which files each benchmark case executes, and write the manifest.

    python tools/benchmark_deps.py --case conformal_window_count
    python tools/benchmark_deps.py --all --merge

``pr-suite`` runs the whole registry on every pull request and its wall clock is
set by a few Monte Carlo and MIP cases most changes cannot touch. Selecting by
dependency needs a map, and the map has to be measured: cases import inside their
functions, so a static scan sees entry points and not reach, and tracing imports
returns the whole library because ``mlsynth/__init__.py`` loads every estimator.

So this runs the case and records which files execute. The package is imported
before measurement starts, which is what separates executing a module from
loading it -- without that every case measures all 834 library files.

File reads are captured too, through an audit hook, so a case is selected when
the panel it reads from ``basedata`` changes.

``--merge`` keeps entries for cases this invocation did not run, so a sharded
CI job can contribute its slice without dropping everyone else's.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _trace(name: str) -> tuple[list[str], float]:
    """The repository-relative files ``name``'s ``run()`` executes or reads."""
    import coverage

    from benchmarks import registry
    from benchmarks.compare import BenchmarkSkipped

    read: set[str] = set()

    def audit(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            p = args[0].decode() if isinstance(args[0], bytes) else args[0]
            try:
                rel = Path(p).resolve().relative_to(ROOT)
            except (ValueError, OSError):
                return
            read.add(str(rel))

    import mlsynth  # noqa: F401 -- warm the package, so module bodies are not measured
    mod = registry.load(name)

    cov = coverage.Coverage(source=["mlsynth"], timid=True, data_file=None)
    sys.addaudithook(audit)
    t0 = time.time()
    cov.start()
    try:
        mod.run()
    except BenchmarkSkipped:
        cov.stop()
        raise
    finally:
        try:
            cov.stop()
        except Exception:                      # pragma: no cover - stop after stop
            pass
    secs = time.time() - t0

    data = cov.get_data()
    executed = {str(Path(f).resolve().relative_to(ROOT))
                for f in data.measured_files() if data.lines(f)}
    # Keep only what a diff can name: library sources and shipped panels. The
    # case's own module is implied by its name and is not stored.
    keep = {p for p in executed | read
            if (p.startswith("mlsynth/") or p.startswith("basedata/"))
            and "__pycache__" not in p and not p.endswith((".pyc", ".pyo"))}
    keep.discard(f"benchmarks/cases/{name}.py")
    return sorted(keep), secs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", action="append", default=[],
                    help="trace this case (repeatable)")
    ap.add_argument("--all", action="store_true", help="trace every registered case")
    ap.add_argument("--merge", action="store_true",
                    help="keep manifest entries for cases not traced here")
    ap.add_argument("--out", default=None, help="write somewhere other than the default")
    args = ap.parse_args()

    from benchmarks import case_deps, registry

    names = list(registry.CASES) if args.all else args.case
    if not names:
        ap.error("give --case NAME or --all")

    out = dict(case_deps.load()) if args.merge else {}
    from benchmarks.compare import BenchmarkSkipped
    skipped = []
    for name in names:
        try:
            deps, secs = _trace(name)
        except BenchmarkSkipped as exc:
            # A case that cannot run here was not measured. Dropping its entry
            # leaves it unmapped, which makes it always run -- the safe side.
            out.pop(name, None)
            skipped.append(name)
            print(f"  skip {name}: {exc}", flush=True)
            continue
        out[name] = deps
        print(f"  {name}: {len(deps)} file(s), {secs:.0f}s", flush=True)

    path = Path(args.out) if args.out else None
    case_deps.save(out, path)
    print(f"\n{len(out)} case(s) mapped"
          + (f", {len(skipped)} left unmapped (they will always run)" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
