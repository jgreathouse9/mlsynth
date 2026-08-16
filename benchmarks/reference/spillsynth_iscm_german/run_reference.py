#!/usr/bin/env python3
"""Driver for the Melnychuk spillover-SCM reference capture.

``reference.R`` runs the authors' unedited functions, which live in a repo that
is cloned on demand at a pinned commit instead of vendored. This driver ensures
the clone is present and hands its path to Rscript, so the whole capture is one
command for ``benchmarks/reference/generate.py``.

    python benchmarks/reference/spillsynth_iscm_german/run_reference.py

Needs Rscript with kernlab, MASS, Synth and foreign. Exits non-zero (and prints
nothing parseable) when any of that is missing, which ``generate.py`` reports as
a skip.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmarks.reference.clone_iscm import _ensure_clone  # noqa: E402

_HERE = Path(__file__).resolve().parent
_DTA = ROOT / "basedata" / "repgermany.dta"


def main() -> int:
    clone = _ensure_clone()
    proc = subprocess.run(
        ["Rscript", str(_HERE / "reference.R"), str(clone), str(_DTA)],
        capture_output=True, text=True,
    )
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
