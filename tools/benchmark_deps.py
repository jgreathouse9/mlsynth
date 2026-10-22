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

What is measured is the runner's own call for the case, ``run_one``: the case's
import (its ``EXPECTED`` and the reference bundle it reads), ``run()``, and
``compare()``. Both Python trees a diff can name are measured, ``mlsynth`` and
``benchmarks``. File reads are captured too, through an audit hook, so a case is
selected when the panel it reads from ``basedata`` or the reference it compares
against changes. A case that skips or raises is left unmapped, and always runs.

On Python 3.12 and later coverage runs on its ``sysmon`` core. Under the
pure-Python tracer the four shards ran 4.8 times the untraced time and one or
two of them reached the 350-minute cap every night.

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


class Unmeasured(Exception):
    """A case the runner skipped or that raised, so what it reaches is unknown.

    Its entry is dropped, which leaves it unmapped and always run. A partial
    measurement would claim less than the case reaches.
    """


def _trace(name: str) -> tuple[list[str], float]:
    """The repository-relative files the runner executes or reads for ``name``.

    What is measured is ``run_benchmarks.run_one(name)``, the call the PR suite
    makes: importing the case, which builds ``EXPECTED`` and reads the case's
    reference bundle; ``run()``; and ``compare()``. Measuring ``run()`` alone
    left a reference read at import, ``compare.py`` and the runner itself
    unclaimed, and an unclaimed file is a hole without an inventory and covered
    by nothing with one.
    """
    from benchmarks import run_benchmarks

    read: set[str] = set()

    def audit(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            p = args[0].decode() if isinstance(args[0], bytes) else args[0]
            try:
                rel = Path(p).resolve().relative_to(ROOT)
            except (ValueError, OSError):
                return
            read.add(rel.as_posix())

    import mlsynth  # noqa: F401 -- warm the package, so module bodies are not measured

    cov = measurement()
    sys.addaudithook(audit)
    t0 = time.time()
    cov.start()
    try:
        result = run_benchmarks.run_one(name)
    finally:
        cov.stop()
    secs = time.time() - t0
    if result["status"] not in ("pass", "fail"):
        raise Unmeasured(f"{result['status']}: {result.get('reason', '')}".rstrip(": "))

    data = cov.get_data()
    executed = {Path(f).resolve().relative_to(ROOT).as_posix()
                for f in data.measured_files() if data.lines(f)}
    return claimable(executed | read, name), secs


def _python() -> tuple:
    """The interpreter's version, read through one name so a test can take the
    path an interpreter without ``sys.monitoring`` takes."""
    return tuple(sys.version_info[:2])


def _tracer(python: tuple, outer) -> tuple:
    """``(timid, core)`` for the map's coverage session, given the interpreter
    and the coverage session already running in the process, if any.

    On Python 3.12 and later the core is ``sysmon``, which records a code
    object's lines once and then stops paying for them. The pure-Python tracer
    pays on every line: the 209 cases measured on 2026-10-21 took 1,149
    minutes under it against 240 untraced, up to 17 times longer for a case
    whose time is spent in Python, and the shards were cancelled at their
    350-minute cap. On ``wan_pda_vs_scm`` the ``sysmon`` core recorded the same
    23 files in 100.0 s against 99.2 s untraced.

    Before 3.12 there is no ``sys.monitoring``, and the pure-Python tracer
    stays, except inside a C tracer. A pure-Python tracer nested in a C tracer
    records no line at all, so under pytest-cov (the daily coverage job) the
    map claimed only the files the audit hook saw opened, and main's coverage
    run failed on 2026-10-22. A C tracer nested in either records normally.
    """
    if python >= (3, 12):
        return False, "sysmon"
    if outer is not None and not outer.config.timid:
        return False, None
    return True, None


def measurement():
    """The coverage session a case runs under: both Python trees a diff can
    name, measured and not saved, with the tracer :func:`_tracer` picks."""
    import coverage

    timid, core = _tracer(_python(), coverage.Coverage.current())
    cov = coverage.Coverage(
        source=[str(ROOT / "mlsynth"), str(ROOT / "benchmarks")],
        timid=timid, data_file=None)
    if core:
        cov.set_option("run:core", core)
    return cov


def claimable(paths, name: str) -> list:
    """The measured paths a diff can name, which is what the map keeps.

    That is every path under :data:`benchmarks.case_deps.REACHABLE` outside the
    test trees, compiled files excepted. The case's own module is implied by
    its name and is not stored. ``benchmarks/`` belongs here as much as
    ``mlsynth/``: a case executes ``compare.py`` and reads its reference's
    committed outputs, and a path the map can never claim is either a hole or,
    once an inventory exists, a file covered by nothing.
    """
    from benchmarks import case_deps

    own = f"benchmarks/cases/{name}.py"
    return sorted({p for p in paths
                   if p.startswith(case_deps.REACHABLE)
                   and not p.startswith(case_deps.NEVER_REACHABLE)
                   and "__pycache__" not in p
                   and not p.endswith((".pyc", ".pyo"))
                   and p != own})


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


def should_record_inventory(from_slices, merged) -> tuple[bool, str]:
    """Whether this combine may settle what nothing covers.

    The inventory turns an unclaimed reachable file from doubt into an answer:
    the sweep looked and no case ran it. Two things make that sound, and only
    one of them needs checking here.

    A case with no entry at all is always selected (``select`` runs an unmeasured
    case unconditionally), so a case that skipped or died cannot be silenced by
    an inventory. Its absence is safe.

    What is not safe is an entry older than the tree. A file added after a case
    was measured does not appear in that case's entry whether or not the case
    executes it, so recording the file as covered-by-nothing would rest on a
    measurement taken before it existed. The inventory is therefore written only
    when every entry in the merged map came from this run's slices, which makes
    each entry exactly as old as the inventory beside it.
    """
    stale = sorted(set(merged) - set(from_slices))
    if stale:
        return False, (
            f"{len(stale)} entr(y/ies) predate this run's slices, so the tree "
            f"may have changed under them: {', '.join(stale[:5])}"
            + (" ..." if len(stale) > 5 else "")
        )
    return True, f"every one of {len(set(merged))} entr(y/ies) measured in this run"


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
        import json as _json
        from_slices = set()
        for sl in args.combine:
            from_slices.update(_json.loads(Path(sl).read_text()))
        merged = combine(args.combine, base=case_deps.load())
        case_deps.save(merged, Path(args.out) if args.out else None)
        print(f"{len(merged)} case(s) mapped from {len(args.combine)} slice(s)")

        ok, why = should_record_inventory(from_slices, merged)
        if ok:
            files = case_deps.reachable_inventory()
            case_deps.save_swept(files, cases=len(merged))
            print(f"inventory: {len(files)} reachable file(s) recorded "
                  f"({why}); unclaimed ones now narrow the selection")
        else:
            print(f"inventory: not recorded -- {why}")
        return 0

    names = list(registry.CASES) if args.all else args.case
    if not names:
        ap.error("give --case NAME or --all")
    try:
        names = shard_of(names, args.shard, args.num_shards)
    except ValueError as exc:
        ap.error(str(exc))

    out = dict(case_deps.load()) if args.merge else {}
    skipped = []
    for name in names:
        try:
            deps, secs = _trace(name)
        except Unmeasured as exc:
            # A case that skipped or raised here was not measured. Dropping its
            # entry leaves it unmapped, which makes it always run -- the safe side.
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
