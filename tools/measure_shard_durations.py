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


def parse(text: str) -> dict:
    """Sum setup + call + teardown per module from ``--durations=0`` output."""
    totals: dict = collections.defaultdict(float)
    for line in text.splitlines():
        m = _LINE.match(line.strip())
        if m:
            totals[m.group(3)] += float(m.group(1))
    return dict(totals)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-file", metavar="PATH",
                    help="parse an existing --durations=0 log instead of "
                         "running the suite again")
    args = ap.parse_args()

    if args.from_file:
        text = pathlib.Path(args.from_file).read_text()
    else:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
             "-n", "auto",
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

    payload = {
        "note": "Seconds per test module, summed over setup/call/teardown. "
                "Advisory: this decides how the suite is split across CI "
                "shards, never what runs. Regenerate with "
                "tools/measure_shard_durations.py.",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "modules": {k: round(v, 3) for k, v in sorted(modules.items())},
    }
    _OUT.write_text(json.dumps(payload, indent=2) + "\n")
    total = sum(modules.values())
    print(f"wrote {_OUT}: {len(modules)} modules, {total:.0f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
