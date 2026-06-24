"""Tests for the captured benchmark-reference corpus.

These validate the committed reference bundles -- the artifacts a reviewer
inspects -- without re-running the reference toolchain: the loader works, the
parsed ``reference.json`` is consistent with the verbatim ``reference.out``, and
the recorded input-data checksums still match the shipped data (so a silent data
change is caught as a stale reference).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from benchmarks.reference import load_reference, reference_value
from benchmarks.reference.generate import _parse_values

_ROOT = Path(__file__).resolve().parents[2]
_BUNDLE = _ROOT / "benchmarks" / "reference" / "synth_prop99"


def test_loader_returns_values_and_weights():
    ref = load_reference("synth_prop99")
    assert set(ref) >= {"values", "weights"}
    assert "synth_pre_ssr" in ref["values"]
    assert ref["weights"]["Utah"] == pytest.approx(0.396, abs=0.01)


def test_reference_value_accessor():
    assert reference_value("synth_prop99", "synth_pre_ssr") == pytest.approx(52.136, abs=0.01)


def test_missing_bundle_raises():
    with pytest.raises(FileNotFoundError):
        load_reference("does_not_exist")


def test_reference_json_matches_captured_output():
    # The parsed JSON must reproduce a re-parse of the verbatim reference.out,
    # so the committed artifacts cannot silently disagree.
    out = (_BUNDLE / "reference.out").read_text()
    reparsed = _parse_values(out)
    stored = json.loads((_BUNDLE / "reference.json").read_text())
    assert reparsed == stored


def test_provenance_data_checksums_current():
    # Each recorded input-data checksum must match the shipped data file; a
    # mismatch means the reference is stale and must be regenerated.
    prov = json.loads((_BUNDLE / "provenance.json").read_text())
    assert prov["git_sha"] and prov["generated_at"]
    for entry in prov["data"]:
        h = hashlib.sha256((_ROOT / entry["path"]).read_bytes()).hexdigest()
        assert h == entry["sha256"], f"stale reference: {entry['path']} changed"


def test_provenance_records_reference_versions():
    prov = json.loads((_BUNDLE / "provenance.json").read_text())
    assert "Synth" in prov.get("packages", {})
    assert re.match(r"R version", prov.get("r_version", ""))
