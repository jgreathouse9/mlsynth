"""Regenerate ``mlsynth/tests/_shard_durations.json``.

The suite is split across CI jobs by packing modules longest-first, which needs
to know how long each module takes. This measures that and writes the map the
packer reads.

Run it from the repository root::

    python tools/measure_shard_durations.py

It runs the whole suite once under ``-n auto`` -- the way CI runs it, so the
numbers describe the thing being balanced -- and sums setup, call and teardown
per module. That takes as long as an unsharded run, which is why the result is
committed: CI reads the file and never measures.

The numbers are advisory. They decide how work is distributed, never what is
run, so a stale map costs balance and cannot cost correctness -- a module that
has become slower since the last regeneration lands in a shard that finishes
later, and nothing is skipped. Regenerate after adding a slow test file, or
when one shard starts finishing well after the others.

Measure on a quiet machine. Durations taken while something else is competing
for the cores are noise, and noise here is the thing being corrected for.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import platform
import re
import subprocess
import sys

_LINE = re.compile(r"^([\d.]+)s\s+(setup|call|teardown)\s+(\S+?)::")
_OUT = pathlib.Path("mlsynth/tests/_shard_durations.json")
_ROOTS = ("mlsynth/tests", "benchmarks/tests")
_NOTE = ("Seconds per test module, summed over setup/call/teardown. "
         "Advisory: this decides how the suite is split across CI shards, "
         "never what runs. Regenerate with tools/measure_shard_durations.py.")


def parse(text: str) -> dict:
    """Sum setup + call + teardown per module from ``--durations=0`` output."""
    totals: dict = collections.defaultdict(float)
    for line in text.splitlines():
        m = _LINE.match(line.strip())
        if m:
            totals[m.group(3)] += float(m.group(1))
    return dict(totals)


def load_existing(path: "pathlib.Path | None" = None) -> dict:
    """The committed payload, or an empty one when there is nothing to read.

    A map that is absent or unreadable is not an error here for the same
    reason it is not one in :func:`mlsynth.tests._shard.load_durations`: the
    numbers are advisory, so the fallback costs balance and cannot cost
    correctness.
    """
    p = _OUT if path is None else pathlib.Path(path)
    try:
        payload = json.loads(p.read_text())
    except (ValueError, OSError):
        return {"modules": {}}
    if not isinstance(payload, dict):
        return {"modules": {}}
    payload.setdefault("modules", {})
    return payload


def missing_modules(durations, roots=_ROOTS, base=None) -> list:
    """The test modules on disk that the map has not seen, sorted.

    An unmeasured module is charged the median of those present, which on the
    committed map is 0.67s against a slowest module of 456s. That charge is a
    guess, and for a slow file it is wrong by two orders of magnitude, so the
    packer places it as if it were free and the shard holding it finishes
    long after the others.
    """
    root_base = pathlib.Path(".") if base is None else pathlib.Path(base)
    found = []
    for root in roots:
        directory = root_base / root
        if not directory.is_dir():
            continue
        for path in directory.glob("test_*.py"):
            found.append(f"{root}/{path.name}")
    return sorted(set(found) - set(durations))


def merge_modules(existing, measured) -> dict:
    """``existing`` updated by ``measured``, sorted by module path.

    A module in both takes its measured value, including when that value is
    0.0: a module that runs in no measurable time is a measurement, not a
    missing one, and dropping it would send it back to the median charge.
    """
    merged = dict(existing)
    merged.update(measured)
    return {k: merged[k] for k in sorted(merged)}


def provenance_matches(payload) -> tuple:
    """Whether the map was measured on this interpreter and this machine.

    Durations from two machines are not comparable, and a merge is the only
    path that can mix them -- a full regeneration replaces every entry at
    once, so it is internally consistent whatever it runs on. A map with no
    recorded provenance predates the fields and is accepted; refusing it
    would strand it with no way to add a module.

    Returns
    -------
    tuple of (bool, str)
        ``(True, "")`` when the merge may proceed, otherwise ``False`` and
        the field that disagrees with both values.
    """
    current = {"python": platform.python_version(),
               "platform": platform.platform()}
    for field, now in current.items():
        was = payload.get(field)
        if was is not None and was != now:
            return (False, f"{field}: the map records {was!r} and this "
                           f"machine is {now!r}")
    return (True, "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-file", metavar="PATH",
                    help="parse an existing --durations=0 log instead of "
                         "running the suite again")
    ap.add_argument("--module", metavar="PATH", action="append", default=[],
                    help="measure only this test module; repeatable. Requires "
                         "--merge, since a subset run knows nothing about the "
                         "entries it did not measure.")
    ap.add_argument("--missing", action="store_true",
                    help="measure the modules the committed map has not seen. "
                         "Requires --merge.")
    ap.add_argument("--merge", action="store_true",
                    help="update the committed map in place, keeping every "
                         "entry this run did not measure. Without it the run "
                         "owns the whole file and replaces it.")
    args = ap.parse_args()

    existing = load_existing()
    subset = list(args.module)
    if args.missing:
        subset += missing_modules(existing["modules"])
    subset = sorted(set(subset))

    if subset and not args.merge:
        print("a subset measurement needs --merge: writing only the modules "
              f"named here would discard the other {len(existing['modules'])} "
              "entries, and the map is what CI packs its shards against.",
              file=sys.stderr)
        return 2

    if args.merge:
        ok, why = provenance_matches(existing)
        if not ok:
            print(f"refusing to merge across environments -- {why}. Durations "
                  "from two machines are not comparable; regenerate the whole "
                  "map here instead, which replaces every entry at once.",
                  file=sys.stderr)
            return 2

    if args.missing and not subset and not args.from_file:
        print("nothing missing from the map; it already covers every test "
              "module on disk.")
        return 0

    if subset:
        print("measuring %d module(s): %s" % (len(subset), ", ".join(subset)))

    if args.from_file:
        text = pathlib.Path(args.from_file).read_text()
    else:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
             "-n", "auto", *subset,
             # --durations=0 alone hides anything under 0.005s, which silently
             # drops the modules that are entirely fast -- 38 of 473 here. A
             # module missing from the map is charged the median, so those
             # would be over-estimated by orders of magnitude, which is the
             # imbalance this map exists to remove. --durations-min=0 reports
             # all of them.
             "--durations=0", "--durations-min=0", "--tb=no", "-rN"],
            capture_output=True, text=True)
        text = proc.stdout + proc.stderr

    modules = parse(text)
    if not modules:
        print("no durations parsed; was the run a --durations=0 run?",
              file=sys.stderr)
        return 1

    measured = {k: round(v, 3) for k, v in modules.items()}
    written = (merge_modules(existing["modules"], measured) if args.merge
               else {k: measured[k] for k in sorted(measured)})
    payload = {
        "note": _NOTE,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "modules": written,
    }
    # A module that skips at collection -- an optional solver absent, say --
    # reports no setup/call/teardown, so a subset run can select it and come
    # back with nothing. Recording 0.0 would claim it is free where the
    # dependency is installed, so it stays out of the map and takes the median
    # charge. Naming it here is what stops the gap surviving the next pass.
    unmeasured = sorted(set(subset) - set(measured))
    if unmeasured:
        payload["unmeasured"] = unmeasured
    _OUT.write_text(json.dumps(payload, indent=2) + "\n")
    total = sum(written.values())
    kept = len(written) - len(measured)
    how = f", {len(measured)} measured and {kept} kept" if args.merge else ""
    print(f"wrote {_OUT}: {len(written)} modules{how}, {total:.0f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
