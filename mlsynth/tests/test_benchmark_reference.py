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
import os
import re
from pathlib import Path

import pytest

from benchmarks.reference import load_reference, reference_value
from benchmarks.reference import _fetch
from benchmarks.reference.generate import _parse_values

_ROOT = Path(__file__).resolve().parents[2]
_BUNDLE = _ROOT / "benchmarks" / "reference" / "synth_prop99"


class TestCIRunsWhenTheSuiteCanBreak:
    """The CI paths filter must cover every directory the test suite reads.

    ``.github/workflows/build.yml`` skips the (slow) suite for changes that
    cannot affect it. That is a sound optimisation and a dangerous one: a check
    that is skipped reports success, and a skipped success is indistinguishable
    from a real one on the pull request page.

    The filter was originally correct. It stopped being correct when tests
    started reading outside ``mlsynth/``: ``TestRegistryCoversEveryCase`` below
    reads ``benchmarks/registry.py`` and ``benchmarks/cases/``, and 75 test files
    load fixtures from ``basedata/``. A pull request touching only those
    directories could therefore break the suite and still go green -- which is
    exactly what happened to the pull request that added the Song benchmark: the
    one test that validates a new case against the registry was the one CI
    skipped.

    So the coupling is asserted here rather than left to whoever next edits the
    workflow. If the suite stops reading one of these directories, delete the
    entry; do not narrow the filter while tests still depend on it.
    """

    @staticmethod
    def _filter_blocks():
        """Every ``filters:`` block in the workflow, not just the first.

        ``build.yml`` gates two jobs (``build`` and the ``pyversions`` matrix)
        and each carries its own copy of the list. Checking only the first would
        have passed while the matrix jobs still skipped -- which is the same
        class of partial check this whole class exists to prevent.
        """
        import re
        wf = (_ROOT / ".github" / "workflows" / "build.yml").read_text()
        blocks = re.findall(r"filters:\s*\|\n((?:\s+(?:code:|-\s*'[^']+')\n)+)", wf)
        assert blocks, "could not locate any paths-filter block in build.yml"
        return [set(re.findall(r"-\s*'([^']+)'", b)) for b in blocks]

    def test_the_workflow_still_has_the_two_filter_copies(self):
        """If a copy is added or removed, re-read the test below before trusting it."""
        assert len(self._filter_blocks()) == 2

    @pytest.mark.parametrize("directory", ["mlsynth", "benchmarks", "basedata"])
    def test_directories_the_tests_read_trigger_a_run(self, directory):
        for i, paths in enumerate(self._filter_blocks()):
            assert f"{directory}/**" in paths, (
                f"tests read from {directory}/ but CI filter block {i} does not "
                "list it, so a change there skips the suite and still reports "
                "success")

    def test_the_tests_really_do_read_those_directories(self):
        """The premise of the test above, asserted rather than assumed.

        Without this, the filter entries could outlive the dependency that
        justifies them and nobody would know it was safe to drop them.
        """
        srcs = [p.read_text() for p in (_ROOT / "mlsynth" / "tests").glob("*.py")]
        assert sum("basedata" in s for s in srcs) > 10
        assert sum("benchmarks" in s for s in srcs) > 5


class TestRegistryCoversEveryCase:
    """Every case module is reachable from ``benchmarks/registry.py``.

    A case that is written, committed and reviewed but never added to ``CASES``
    is invisible to ``run_benchmarks.py --all``: it does not run in the routine
    sweep, so it can rot indefinitely without a single check going red. That is
    a worse failure than a case that fails, because nothing announces it -- and
    it is easy to do, since the case file and the registry are separate edits
    and only the case file is the interesting one to write.
    """

    @staticmethod
    def _modules():
        d = _ROOT / "benchmarks" / "cases"
        return {p.stem for p in d.glob("*.py") if p.stem != "__init__"}

    @staticmethod
    def _registered():
        from benchmarks import registry
        return {v.rsplit(".", 1)[1] for v in registry.CASES.values()}

    def test_every_case_module_is_registered(self):
        missing = sorted(self._modules() - self._registered())
        assert not missing, (
            "case module(s) not in benchmarks/registry.py CASES, so they never "
            f"run under --all: {missing}")

    def test_every_registered_case_module_exists(self):
        """The other direction: a rename that misses the registry."""
        stale = sorted(self._registered() - self._modules())
        assert not stale, f"CASES names a module that does not exist: {stale}"

    def test_registry_keys_match_their_module_names(self):
        """The short name and the module stem are the same word.

        ``run_benchmarks.py --case <name>`` takes the key while every other
        reference to a case -- docs, the reference bundle directory, this test --
        uses the module name. Letting them drift apart makes a case findable by
        one name and not the other.
        """
        from benchmarks import registry
        mismatched = sorted(
            (k, v) for k, v in registry.CASES.items()
            if v.rsplit(".", 1)[1] != k)
        assert not mismatched, f"key != module stem: {mismatched}"


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


def test_mscmt_basque_reference_prepped_to_capture_dependent_loss():
    # The mscmt_basque dependent-loss pin is "prepped for regen": reference.R
    # must emit the loss_v (and rmspe) lines so a bundle regeneration captures
    # MSCMT's loss.v, and generate.py's parser must pick them up. This guards the
    # regen contract -- a future edit cannot silently drop the dependent loss and
    # leave the pin permanently dormant.
    rscript = (_ROOT / "benchmarks" / "reference" / "mscmt_basque"
               / "reference.R").read_text()
    assert "res$loss.v" in rscript, "reference.R no longer emits MSCMT's loss.v"
    assert "sqrt(res$loss.v)" in rscript, "reference.R no longer emits the RMSPE"
    assert "loss_v" in rscript and "rmspe" in rscript
    sample = (
        "== REFERENCE VALUES ==\n"
        "weight\tCataluna\t0.632794\n"
        "avg_post_gap\t-0.770963\n"
        "loss_v\t0.004286071\n"
        "rmspe\t0.065468095\n"
        "== SESSION INFO ==\n"
    )
    parsed = _parse_values(sample)
    assert parsed["values"]["loss_v"] == pytest.approx(0.004286071)
    assert parsed["values"]["rmspe"] == pytest.approx(0.065468095)


def test_mscmt_basque_dependent_loss_pin_dormant_until_bundle_has_it():
    # Until the bundle carries loss_v the pin stays dormant, so the case remains
    # green on the current bundle; when a regen adds loss_v the case's guard
    # (``.get("loss_v")``) is what flips it on. Assert the guard reflects the
    # bundle's actual state, both ways.
    from benchmarks.cases import mscmt_basque as case

    has_loss = load_reference("mscmt_basque")["values"].get("loss_v") is not None
    assert ("dependent_mspe" in case.EXPECTED) == has_loss


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


class TestTheSbcMonteCarloPanelsAreCommitted:
    """``sbc_mc`` estimates on R-drawn panels, and they ship with the repo.

    The panels are captured from the authors' own DGP (``reference.R``, base R)
    so that the simulator sits on the reference side of the comparison. That
    only holds while the case can find them: a broken path would make the case
    fall back to a skip, and ``--all`` reports a skip as success. Same failure
    mode the Brabander cases hit below, asserted the same way.
    """

    def test_the_panels_are_reachable_from_the_case(self):
        from benchmarks import registry
        from benchmarks.cases.sbc_mc import PANELS

        assert os.path.exists(PANELS), (
            f"sbc_mc reads committed panels that are not where it looks for "
            f"them ({PANELS}); the case would skip, not fail.")
        assert "sbc_mc" in registry.CASES

    def test_the_capture_script_ships_with_the_panels(self):
        """A fixture nobody can regenerate is a number with no provenance."""
        bundle = _ROOT / "benchmarks" / "reference" / "sbc_mc"
        assert (bundle / "reference.R").exists()
        assert (bundle / "versions.txt").exists()


class TestTheBrabanderCasesActuallyRun:
    """A case whose data is committed must not skip.

    ``BenchmarkSkipped`` is the right answer when an optional toolchain is
    missing, and the driver prints it as ``[SKIP]`` rather than a failure. That
    makes it a silent hole for a case whose data ships with the repository: a
    wrong path inside the case reads exactly like an absent dependency, and
    ``--all`` still reports success. This happened while these cases were being
    written -- moving their shared helper changed ``__file__``, the panel path
    broke, and both cases skipped without anything going red.

    So for the cases backed by ``benchmarks/reference/brabander_sdid_match/``,
    which is committed, a skip is a defect and is asserted to be one.
    """

    CASES = ("brabander_brexit_table1", "brabander_brexit_insample")

    @pytest.mark.parametrize("name", CASES)
    def test_the_panel_is_reachable_from_the_case(self, name):
        """Cheap surrogate for the whole case: the data it needs resolves.

        Deliberately not a full run -- these fit twenty-odd synthetic controls
        each and belong in the benchmark suite, not the unit tests. What is
        asserted here is only the thing that silently broke.
        """
        from benchmarks import registry
        from benchmarks.brabander_common import _PANEL

        assert os.path.exists(_PANEL), (
            f"{name} reads a committed panel that is not where it looks for "
            f"it ({_PANEL}); the case would skip rather than fail.")
        assert name in registry.CASES

    def test_the_helper_is_not_a_case_module(self):
        """It lives outside ``benchmarks/cases/`` on purpose.

        Every module in that directory has to be registered (see
        ``TestRegistryCoversEveryCase``), and a shared helper is not a case, so
        it would be permanently unregisterable there.
        """
        cases_dir = _ROOT / "benchmarks" / "cases"
        assert not (cases_dir / "_brabander_brexit.py").exists()
        assert (_ROOT / "benchmarks" / "brabander_common.py").exists()
