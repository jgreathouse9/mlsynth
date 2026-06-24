#!/usr/bin/env python3
"""Export side-by-side mlsynth-vs-reference comparison tables.

A cross-validation case may define ``comparison() -> list[dict]`` returning rows
``{quantity, mlsynth, reference}`` that pair the library's number against the
captured reference value, quantity by quantity. This writes those rows to an
easily inspectable, committed artifact per case::

    benchmarks/reference/<case>/comparison.csv     # quantity, mlsynth, reference, abs_diff

and, when ``openpyxl`` is installed, a combined workbook with one sheet per
case::

    benchmarks/reference/comparisons.xlsx

so a reviewer can open one file and read the two implementations side by side.

Usage::

    python benchmarks/reference/export_comparison.py synth_prop99
    python benchmarks/reference/export_comparison.py --all
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from benchmarks import registry              # noqa: E402

_FIELDS = ["quantity", "mlsynth", "reference", "abs_diff"]


def _cases_with_comparison() -> list:
    out = []
    for name in registry.CASES:
        try:
            mod = registry.load(name)
        except Exception:                     # pragma: no cover - import-time issues
            continue
        if hasattr(mod, "comparison"):
            out.append(name)
    return out


def export_case(name: str) -> list:
    """Run a case's comparison() and write its comparison.csv. Returns the rows."""
    mod = registry.load(name)
    rows = mod.comparison()
    for r in rows:
        r["abs_diff"] = round(abs(float(r["mlsynth"]) - float(r["reference"])), 6)
    bundle = REF_DIR / name
    bundle.mkdir(exist_ok=True)
    with open(bundle / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] {name}: wrote {bundle / 'comparison.csv'} ({len(rows)} rows)")
    return rows


def _write_workbook(per_case: dict) -> None:
    """Write a combined Excel workbook (one sheet per case) if openpyxl is present."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError:
        print("[note] openpyxl not installed; skipping comparisons.xlsx (CSVs written)")
        return
    wb = Workbook()
    wb.remove(wb.active)
    for name, rows in per_case.items():
        ws = wb.create_sheet(name[:31])       # Excel sheet-name limit
        ws.append(_FIELDS)
        for c in ws[1]:
            c.font = Font(bold=True)
        for r in rows:
            ws.append([r[f] for f in _FIELDS])
    out = REF_DIR / "comparisons.xlsx"
    wb.save(out)
    print(f"[ok] wrote {out} ({len(per_case)} sheet(s))")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    cases = _cases_with_comparison() if args.all else ([args.case] if args.case else [])
    if not cases:
        ap.error("give a case name or --all (a case must define comparison())")
    per_case = {name: export_case(name) for name in cases}
    _write_workbook(per_case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
