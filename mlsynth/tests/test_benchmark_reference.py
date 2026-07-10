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
from benchmarks.reference import _fetch
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


def test_codeload_skips_on_unwritable_cache(tmp_path, monkeypatch):
    # A filesystem error at the cache-dir creation (e.g. the .cache dir left
    # root-owned by the sudo R provisioners) must degrade to a graceful skip --
    # _codeload returns False so fetch_pinned_repo raises BenchmarkSkipped rather
    # than the raw OSError crashing the whole benchmark shard.
    def fake_run(cmd, **kw):
        proc = type("P", (), {"stdout": "", "returncode": 0})()
        if cmd[0] == "curl":                                  # fake a 200 + non-empty tarball
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"tar-bytes")
            proc.stdout = "200"
        elif cmd[0] == "tar":                                 # extract a single <repo>-<sha>/ dir
            (Path(cmd[cmd.index("-C") + 1]) / "repo-deadbee").mkdir()
        return proc

    monkeypatch.setattr(_fetch.subprocess, "run", fake_run)
    # dest's parent is a regular file, so dest.mkdir(parents=True) raises
    # NotADirectoryError (an OSError even root cannot bypass) at the exact line
    # that failed in CI.
    (tmp_path / "cache").write_text("not a directory")
    dest = tmp_path / "cache" / "ClusterSC"
    assert _fetch._codeload("https://github.com/owner/repo.git", "deadbeef", dest) is False


def test_comparison_csv_is_self_consistent():
    # The committed side-by-side table: a metadata header (timestamp, mlsynth
    # version + call) then rows whose abs_diff equals |mlsynth - reference| and
    # whose reference column matches the captured bundle.
    import csv

    ref = load_reference("synth_prop99")
    text = (_BUNDLE / "comparison.csv").read_text().splitlines()
    meta = dict(re.match(r"# (\w+): (.*)", ln).groups() for ln in text if ln.startswith("#"))
    assert meta["generated_at"] and meta["mlsynth_version"]
    assert meta["estimator"] == "VanillaSC" and "backend" in meta["config"]
    rows = list(csv.DictReader([ln for ln in text if not ln.startswith("#")]))
    assert rows, "comparison.csv is empty"
    for r in rows:
        ml, rf = float(r["mlsynth"]), float(r["reference"])
        assert abs(ml - rf) == pytest.approx(float(r["abs_diff"]), abs=1e-6)
        if r["quantity"].startswith("weight["):
            donor = r["quantity"][len("weight["):-1]
            assert rf == pytest.approx(ref["weights"].get(donor, 0.0), abs=1e-6)
        elif r["quantity"] == "pre_period_SSR":
            assert rf == pytest.approx(ref["values"]["synth_pre_ssr"], abs=1e-6)
