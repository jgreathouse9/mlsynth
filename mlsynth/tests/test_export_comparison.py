"""Tests for the comparison-workbook exporter's new-entry + assemble modes.

The daily benchmark action creates comparison-workbook entries for cases that do
not yet have one (``--missing``) and rebuilds ``comparisons.xlsx`` from the
committed per-case CSVs (``--assemble``) -- rather than re-running every
(stochastic) case daily. These pin that behaviour without a live reference run.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from benchmarks.reference import export_comparison as ec


def _write_csv(bundle: Path, meta: dict, rows: list) -> None:
    bundle.mkdir(parents=True, exist_ok=True)
    with open(bundle / "comparison.csv", "w", newline="") as fh:
        for k, v in meta.items():
            fh.write(f"# {k}: {v}\n")
        w = csv.DictWriter(fh, fieldnames=ec._FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_read_comparison_csv_round_trips(tmp_path):
    meta = {"case": "demo", "generated_at": "2026-07-11T00:00:00Z",
            "mlsynth_version": "9.9.9", "estimator": "BFSC", "config": "{}",
            "reference_impl": "author Stan (live)", "reference_version": "v1"}
    rows = [{"quantity": "ATT", "mlsynth": -16.0, "reference": -16.2, "abs_diff": 0.2}]
    _write_csv(tmp_path / "demo", meta, rows)

    got = ec._read_comparison_csv(tmp_path / "demo" / "comparison.csv")
    assert got["meta"]["estimator"] == "BFSC"
    assert got["meta"]["reference_impl"] == "author Stan (live)"
    assert len(got["rows"]) == 1
    r = got["rows"][0]
    assert r["quantity"] == "ATT"
    assert float(r["mlsynth"]) == pytest.approx(-16.0)
    assert float(r["abs_diff"]) == pytest.approx(0.2)


def test_missing_cases_flags_uncaptured_only():
    # _missing_cases() = comparison cases lacking a committed CSV; a case that
    # ships a bundle CSV (synth_prop99) is never flagged.
    missing = set(ec._missing_cases())
    have = set(ec._cases_with_comparison())
    assert missing <= have
    assert "synth_prop99" in have and "synth_prop99" not in missing
    assert (ec.REF_DIR / "synth_prop99" / "comparison.csv").exists()
    # the flag is exactly "has no committed comparison.csv", both directions
    for name in missing:
        assert not (ec.REF_DIR / name / "comparison.csv").exists()
    for name in have:
        if (ec.REF_DIR / name / "comparison.csv").exists():
            assert name not in missing


def test_assemble_from_csv_builds_workbook(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    for name, att in (("case_a", -16.0), ("case_b", -3.2)):
        _write_csv(tmp_path / name,
                   {"case": name, "generated_at": "2026-07-11T00:00:00Z",
                    "mlsynth_version": "9.9.9", "estimator": "X", "config": "{}",
                    "reference_impl": "ref", "reference_version": "v"},
                   [{"quantity": "ATT", "mlsynth": att, "reference": att + 0.1,
                     "abs_diff": 0.1}])
    per_case = ec.assemble_workbook_from_csv(ref_dir=tmp_path)
    assert set(per_case) == {"case_a", "case_b"}
    out = tmp_path / "comparisons.xlsx"
    assert out.exists()
    wb = openpyxl.load_workbook(out)
    assert "summary" in wb.sheetnames
    assert "case_a" in wb.sheetnames and "case_b" in wb.sheetnames


def test_assemble_is_byte_deterministic(tmp_path):
    # Identical committed CSVs must yield a byte-identical workbook, so the daily
    # "commit only if changed" gate stays quiet when nothing new was added.
    pytest.importorskip("openpyxl")
    _write_csv(tmp_path / "case_a",
               {"case": "case_a", "generated_at": "2026-07-11T00:00:00Z",
                "mlsynth_version": "9.9.9", "estimator": "X", "config": "{}"},
               [{"quantity": "ATT", "mlsynth": -1.0, "reference": -1.1, "abs_diff": 0.1}])
    ec.assemble_workbook_from_csv(ref_dir=tmp_path)
    first = (tmp_path / "comparisons.xlsx").read_bytes()
    ec.assemble_workbook_from_csv(ref_dir=tmp_path)
    second = (tmp_path / "comparisons.xlsx").read_bytes()
    assert first == second


def test_assemble_ignores_bundles_without_comparison_csv(tmp_path):
    pytest.importorskip("openpyxl")
    (tmp_path / "no_csv").mkdir()                       # a bundle dir with no CSV
    _write_csv(tmp_path / "has_csv",
               {"case": "has_csv", "generated_at": "t", "mlsynth_version": "9",
                "estimator": "X", "config": "{}"},
               [{"quantity": "ATT", "mlsynth": 1.0, "reference": 1.0, "abs_diff": 0.0}])
    per_case = ec.assemble_workbook_from_csv(ref_dir=tmp_path)
    assert set(per_case) == {"has_csv"}
