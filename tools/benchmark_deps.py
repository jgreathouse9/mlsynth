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

Measuring the whole registry does not fit one job. At 199 cases the single
``case-deps`` job reached its 350-minute cap on each of the four days after it
was added and was cancelled every time, which left the manifest at the five
entries that shipped with it -- so the selection it feeds had nothing to select
with and every pull request ran the whole registry. ``--shard i --num-shards n``
splits the work round-robin, the same slice ``run_benchmarks.py`` takes, so the
heavy cases spread; each shard writes its own slice with ``--out`` and one
``--combine`` step merges them onto the shipped map and commits once.
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


def shard_of(names, shard: int, num_shards: int) -> list:
    """The round-robin slice ``shard`` of ``names``.

    Same slice ``run_benchmarks.select_cases`` takes, because cost is not
    uniform across the registry: contiguous blocks would put the Monte Carlo and
    MIP cases in one shard and leave the rest idle.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard must be in [0, {num_shards}), got {shard}")
    return list(names)[shard::num_shards] if num_shards > 1 else list(names)


def combine(slices, base: dict) -> dict:
    """``base`` updated by each slice, later slices winning.

    A case no slice measured keeps its base entry, so a shard that died leaves
    everyone else's cases mapped instead of un-mapping them. A case a slice did
    measure takes the new value, including an empty list, which is a finding.
    """
    import json as _json
    from pathlib import Path as _Path

    out = dict(base)
    for path in slices:
        out.update(_json.loads(_Path(path).read_text()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", action="append", default=[],
                    help="trace this case (repeatable)")
    ap.add_argument("--all", action="store_true", help="trace every registered case")
    ap.add_argument("--merge", action="store_true",
                    help="keep manifest entries for cases not traced here")
    ap.add_argument("--out", default=None, help="write somewhere other than the default")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the cases into this many shards (round-robin)")
    ap.add_argument("--shard", type=int, default=0,
                    help="0-based index of the shard to measure")
    ap.add_argument("--combine", nargs="+", default=None, metavar="SLICE",
                    help="merge these shard slices onto the shipped map and "
                         "write it, measuring nothing")
    args = ap.parse_args()

    from benchmarks import case_deps, registry

    if args.combine:
        merged = combine(args.combine, base=case_deps.load())
        case_deps.save(merged, Path(args.out) if args.out else None)
        print(f"{len(merged)} case(s) mapped from {len(args.combine)} slice(s)")
        return 0

    names = list(registry.CASES) if args.all else args.case
    if not names:
        ap.error("give --case NAME or --all")
    try:
        names = shard_of(names, args.shard, args.num_shards)
    except ValueError as exc:
        ap.error(str(exc))

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
