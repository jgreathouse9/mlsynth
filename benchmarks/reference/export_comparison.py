#!/usr/bin/env python3
"""Export side-by-side mlsynth-vs-reference comparison tables.

A cross-validation case defines ``comparison() -> {"rows": [...],
"mlsynth_call": {...}}``: the rows pair the library's number against the captured
reference value quantity by quantity, and ``mlsynth_call`` records the estimator
and config the mlsynth numbers came from. This writes, per case, a committed,
metadata-stamped artifact::

    benchmarks/reference/<case>/comparison.csv   # metadata header + quantity, mlsynth, reference, abs_diff

Each carries provenance: when it was written, the mlsynth version and the exact
call (estimator + config), and the reference implementation and its version.
The public, web-native rollup of these CSVs is the Validation dashboard
(``docs/validation.rst``), built by ``build_validation.py``.

Usage::

    python benchmarks/reference/export_comparison.py synth_prop99
    python benchmarks/reference/export_comparison.py --all
    python benchmarks/reference/export_comparison.py --missing
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


def _missing_cases() -> list:
    """Comparison cases that do not yet have a committed comparison.csv.

    The daily action generates only these -- re-running an existing (often
    stochastic) case would churn Monte-Carlo jitter into the committed workbook.
    """
    return [n for n in _cases_with_comparison()
            if not (REF_DIR / n / "comparison.csv").exists()]


def _read_comparison_csv(path: Path) -> dict:
    """Parse a committed comparison.csv back into ``{"rows", "meta"}``.

    The metadata header is ``# key: value`` comment lines; the body is the
    ``quantity, mlsynth, reference, abs_diff`` table. Numeric columns are coerced
    to float so the assembled workbook matches a live export.
    """
    meta: dict = {}
    body: list = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("# "):
            key, _, val = line[2:].partition(": ")
            meta[key] = val
        elif line.strip():
            body.append(line)
    rows = []
    for r in csv.DictReader(body):
        rows.append({"quantity": r["quantity"],
                     "mlsynth": float(r["mlsynth"]),
                     "reference": float(r["reference"]),
                     "abs_diff": float(r["abs_diff"])})
    return {"rows": rows, "meta": meta}


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
        "config": json.dumps(call.get("config", {}), sort_keys=True, default=str),
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
        key = next((p for p in ("Synth", "augsynth", "scpi", "synthdid", "MCPanel") if p in pkgs), None)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?")
    ap.add_argument("--all", action="store_true",
                    help="(re)run every case that defines comparison()")
    ap.add_argument("--missing", action="store_true",
                    help="run only cases that have no committed comparison.csv yet")
    args = ap.parse_args()

    if args.missing:
        cases = _missing_cases()
    elif args.all:
        cases = _cases_with_comparison()
    else:
        cases = [args.case] if args.case else []
    if not cases:
        if args.missing:                          # nothing to do is success, not an error
            print("[ok] no missing comparison entries")
            return 0
        ap.error("give a case name, --all, or --missing "
                 "(a case must define comparison())")
    from benchmarks.compare import BenchmarkSkipped
    for name in cases:
        try:
            export_case(name)
        except BenchmarkSkipped as exc:
            print(f"[skip] {name}: {exc}")
        except Exception as exc:                  # reference toolchain absent / draft error
            print(f"[skip] {name}: {type(exc).__name__}: {str(exc)[:160]}")
    print("[note] rebuild the public dashboard with: "
          "python benchmarks/reference/build_validation.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
