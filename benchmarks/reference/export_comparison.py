#!/usr/bin/env python3
"""Export side-by-side mlsynth-vs-reference comparison tables.

A cross-validation case defines ``comparison() -> {"rows": [...],
"mlsynth_call": {...}}``: the rows pair the library's number against the captured
reference value quantity by quantity, and ``mlsynth_call`` records the estimator
and config the mlsynth numbers came from. This writes, per case, a committed,
metadata-stamped artifact::

    benchmarks/reference/<case>/comparison.csv   # metadata header + quantity, mlsynth, reference, abs_diff

and a combined workbook -- the headline a reviewer opens -- with a summary sheet
plus one detail sheet per case::

    benchmarks/reference/comparisons.xlsx

Each artifact carries provenance: when it was written, the mlsynth version and
the exact call (estimator + config), and the reference implementation and its
version.

Usage::

    python benchmarks/reference/export_comparison.py synth_prop99
    python benchmarks/reference/export_comparison.py --all
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import mlsynth                                  # noqa: E402
from benchmarks import registry                # noqa: E402

_FIELDS = ["quantity", "mlsynth", "reference", "abs_diff"]


def _cases_with_comparison() -> list:
    out = []
    for name in registry.CASES:
        try:
            mod = registry.load(name)
        except Exception:                        # pragma: no cover - import-time issues
            continue
        if hasattr(mod, "comparison"):
            out.append(name)
    return out


def _metadata(name: str, result: dict) -> dict:
    """Provenance for one case's comparison: when, which mlsynth call, which
    reference (and version). The reference descriptor comes from the case's
    ``comparison()`` (for live, vendored references) or from the captured bundle's
    manifest/provenance when present."""
    call = result.get("mlsynth_call", {})
    meta = {
        "case": name,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mlsynth_version": getattr(mlsynth, "__version__", "unknown"),
        "estimator": call.get("estimator", ""),
        "config": json.dumps(call.get("config", {}), sort_keys=True),
    }
    ref = result.get("reference", {})
    if ref:                                       # case-supplied (live reference)
        meta["reference_impl"] = ref.get("impl", "")
        meta["reference_version"] = ref.get("version", "")
        return meta
    bundle = REF_DIR / name                       # captured-bundle reference
    if (bundle / "manifest.json").exists():
        meta["reference_impl"] = json.loads((bundle / "manifest.json").read_text()).get("reference_impl", "")
    if (bundle / "provenance.json").exists():
        prov = json.loads((bundle / "provenance.json").read_text())
        ver = prov.get("r_version") or prov.get("python_version") or ""
        pkgs = prov.get("packages", {})
        key = next((p for p in ("Synth", "augsynth", "scpi", "causaltensor") if p in pkgs), None)
        if key:
            ver = f"{ver}; {key} {pkgs[key]}".strip("; ")
        meta["reference_version"] = ver
        meta["reference_generated_at"] = prov.get("generated_at", "")
    return meta


def export_case(name: str) -> dict:
    """Run a case's comparison(), write its comparison.csv, return rows + metadata."""
    result = registry.load(name).comparison()
    rows = result["rows"]
    for r in rows:
        r["abs_diff"] = round(abs(float(r["mlsynth"]) - float(r["reference"])), 6)
    meta = _metadata(name, result)
    bundle = REF_DIR / name
    bundle.mkdir(exist_ok=True)
    with open(bundle / "comparison.csv", "w", newline="") as fh:
        for k, v in meta.items():                # metadata as leading comment lines
            fh.write(f"# {k}: {v}\n")
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] {name}: wrote comparison.csv ({len(rows)} rows, max |Δ|="
          f"{max(r['abs_diff'] for r in rows):.4g})")
    return {"rows": rows, "meta": meta}


def _write_workbook(per_case: dict) -> None:
    """Combined workbook: a summary sheet plus a metadata-stamped sheet per case."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError:                          # pragma: no cover - openpyxl present here
        print("[note] openpyxl not installed; skipping comparisons.xlsx (CSVs written)")
        return
    wb = Workbook()
    summ = wb.active
    summ.title = "summary"
    summ.append(["case", "quantities", "max_abs_diff", "generated_at",
                 "mlsynth_version", "reference_impl"])
    for c in summ[1]:
        c.font = Font(bold=True)
    for name, payload in per_case.items():
        rows, meta = payload["rows"], payload["meta"]
        summ.append([name, len(rows), max(r["abs_diff"] for r in rows),
                     meta["generated_at"], meta["mlsynth_version"],
                     meta.get("reference_impl", "")])
        ws = wb.create_sheet(name[:31])
        for k, v in meta.items():                # metadata block atop the sheet
            ws.append([k, v])
            ws.cell(ws.max_row, 1).font = Font(bold=True)
        ws.append([])
        ws.append(_FIELDS)
        for c in ws[ws.max_row]:
            c.font = Font(bold=True)
        for r in rows:
            ws.append([r[f] for f in _FIELDS])
    out = REF_DIR / "comparisons.xlsx"
    wb.save(out)
    print(f"[ok] wrote {out} (summary + {len(per_case)} case sheet(s))")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    cases = _cases_with_comparison() if args.all else ([args.case] if args.case else [])
    if not cases:
        ap.error("give a case name or --all (a case must define comparison())")
    from benchmarks.compare import BenchmarkSkipped
    per_case = {}
    for name in cases:
        try:
            per_case[name] = export_case(name)
        except BenchmarkSkipped as exc:
            print(f"[skip] {name}: {exc}")
        except Exception as exc:                  # reference toolchain absent / draft error
            print(f"[skip] {name}: {type(exc).__name__}: {str(exc)[:160]}")
    _write_workbook(per_case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
