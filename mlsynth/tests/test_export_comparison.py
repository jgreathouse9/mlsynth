"""Tests for the comparison exporter and the Validation-dashboard generator.

The daily benchmark action creates a per-case ``comparison.csv`` for any
cross-validation case that lacks one (``--missing``), then rebuilds the public
web-native dashboard ``docs/validation.rst`` from the committed CSVs -- rather
than re-running every (stochastic) case daily. These pin that behaviour without
a live reference run.
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


def test_no_workbook_machinery_remains():
    # The binary .xlsx workbook was retired in favour of the generated docs page;
    # its openpyxl assembly must be gone so it cannot silently come back.
    assert not hasattr(ec, "assemble_workbook_from_csv")
    assert not hasattr(ec, "_write_workbook")
    assert not (ec.REF_DIR / "comparisons.xlsx").exists()


def _bv():
    from benchmarks.reference import build_validation as bv
    return bv


def test_validation_verdict_bands():
    bv = _bv()
    # exact: display-precision match; tight: <=2% worst relative; documented: loose.
    assert bv._verdict([{"reference": -15.6, "abs_diff": 1e-6}])[0] == "exact"
    assert bv._verdict([{"reference": -19.5, "abs_diff": 0.02}])[0] == "tight"
    assert bv._verdict([{"reference": -20.0, "abs_diff": 5.0}])[0] == "documented"


def test_validation_canonicalises_estimator_labels():
    bv = _bv()
    assert bv._canon("ClusterSC/PCR (run_pcr OLS, fixed rank)") == "ClusterSC"
    assert bv._canon("ridge_augment_weights") == "VanillaSC"   # remapped
    assert bv._canon("VanillaSC") == "VanillaSC"


def test_validation_page_builds_from_real_corpus():
    bv = _bv()
    by_est = bv.collect()
    assert by_est, "no committed comparison.csv found"
    rst = bv.to_rst(by_est)
    assert rst.startswith(".. _validation:")
    assert "Validation dashboard" in rst and "Coverage:" in rst
    # every estimator with a committed cross-validation gets a linkable section
    assert "VanillaSC" in by_est
    assert f".. _{bv._slug('VanillaSC')}:" in rst
