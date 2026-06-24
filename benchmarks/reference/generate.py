#!/usr/bin/env python3
"""Generate (or refresh) a captured reference bundle for a benchmark case.

A reference bundle makes a cross-validation case's pinned numbers auditable: it
runs the genuine reference implementation, captures its verbatim output, parses
the values the Python case pins against, and records full provenance (tool
versions, OS, git SHA, and a checksum of every input data file). The result is
a directory under ``benchmarks/reference/<case>/`` that a reader -- or a JSS
reviewer -- can open, inspect, and regenerate.

Layout produced::

    benchmarks/reference/<case>/
        manifest.json     # hand-written: case metadata + data dependencies (input)
        reference.R       # hand-written: the exact reference script (input)
        reference.out     # captured stdout of the reference run (generated)
        reference.json    # parsed {values, weights} the case pins against (generated)
        provenance.json   # versions, OS, git SHA, data checksums, timestamp (generated)

Usage::

    python benchmarks/reference/generate.py synth_prop99
    python benchmarks/reference/generate.py --all

Skips cleanly (non-zero exit avoided) when the reference toolchain is absent, so
running it without R does not fail a pipeline -- it reports and moves on.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF_DIR = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_values(out: str) -> dict:
    """Parse the ``== REFERENCE VALUES ==`` block: tab-separated key/value lines
    and ``weight\\t<label>\\t<value>`` rows."""
    values, weights = {}, {}
    in_block = False
    for line in out.splitlines():
        if line.startswith("== REFERENCE VALUES =="):
            in_block = True
            continue
        if line.startswith("== ") and in_block:
            break
        if not in_block:
            continue
        parts = line.split("\t")
        if parts[0] == "weight" and len(parts) == 3:
            weights[parts[1]] = float(parts[2])
        elif len(parts) == 2:
            values[parts[0]] = float(parts[1])
    return {"values": values, "weights": weights}


def _r_provenance(out: str) -> dict:
    """Pull R and key-package versions out of the captured sessionInfo()."""
    prov = {}
    m = re.search(r"R version [^\n]+", out)
    if m:
        prov["r_version"] = m.group(0).strip()
    # package_version tokens from sessionInfo, e.g. "Synth_1.1-10"; the version
    # must look like a version (digits with . or - separators) to avoid matching
    # the "x86_64-linux-gnu" platform string.
    for name, ver in re.findall(r"\b([A-Za-z][A-Za-z0-9.]{2,})_(\d+(?:[.\-]\d+)+)\b", out):
        prov.setdefault("packages", {})[name] = ver
    return prov


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # pragma: no cover - git always present in this repo
        return "unknown"


def generate(case: str) -> bool:
    """Run a case's reference and write its bundle. Returns True on success."""
    bundle = REF_DIR / case
    manifest = json.loads((bundle / "manifest.json").read_text())
    cmd = manifest["command"]
    rscript = shutil.which(cmd.split()[0])
    if rscript is None:
        print(f"[skip] {case}: '{cmd.split()[0]}' not on PATH")
        return False
    proc = subprocess.run(cmd.split(), cwd=ROOT, capture_output=True, text=True)
    out = proc.stdout
    if proc.returncode != 0 or "== REFERENCE VALUES ==" not in out:
        print(f"[skip] {case}: reference run failed: {proc.stderr.strip()[-300:]}")
        return False

    (bundle / "reference.out").write_text(out)
    parsed = _parse_values(out)
    (bundle / "reference.json").write_text(json.dumps(parsed, indent=2) + "\n")
    provenance = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "command": cmd,
        "data": [{"path": p, "sha256": _sha256(ROOT / p)} for p in manifest["data"]],
        **_r_provenance(out),
    }
    (bundle / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    vals = ", ".join(f"{k}={v:.5g}" for k, v in parsed["values"].items())
    print(f"[ok] {case}: {vals}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", help="case name (a directory under benchmarks/reference/)")
    ap.add_argument("--all", action="store_true", help="regenerate every bundle")
    args = ap.parse_args()
    if args.all:
        cases = sorted(p.name for p in REF_DIR.iterdir()
                       if (p / "manifest.json").exists())
    elif args.case:
        cases = [args.case]
    else:
        ap.error("give a case name or --all")
    any_ok = any(generate(c) for c in cases)
    return 0 if any_ok else 0      # absence of a toolchain is not a failure


if __name__ == "__main__":
    raise SystemExit(main())
