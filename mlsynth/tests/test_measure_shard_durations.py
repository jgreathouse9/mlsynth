"""Tests for ``tools/measure_shard_durations.py``.

The tool writes ``mlsynth/tests/_shard_durations.json``, the map
:mod:`mlsynth.tests._shard` packs CI's shards against. The map is advisory --
it decides how work is distributed, never what runs -- so these tests are
about keeping it honest, not about correctness of the suite.

The measurement itself takes as long as an unsharded run, which is why the map
is committed. That is also why a subset measurement exists: a map missing
thirteen modules charges each of them the median, and the median here is 0.67s
against a slowest module of 456s, so an unmeasured slow file lands wherever the
packer happens to put it. Re-measuring the whole suite to add one file is the
cost that keeps the map stale.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_TOOL = _ROOT / "tools" / "measure_shard_durations.py"


def _load():
    spec = importlib.util.spec_from_file_location("measure_shard_durations",
                                                  _TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["measure_shard_durations"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return _load()


_LOG = """\
============================= slowest durations ==============================
2.50s call     mlsynth/tests/test_alpha.py::TestOne::test_a
0.30s setup    mlsynth/tests/test_alpha.py::TestOne::test_a
0.20s teardown mlsynth/tests/test_alpha.py::TestOne::test_a
1.00s call     benchmarks/tests/test_beta.py::test_b
0.00s call     mlsynth/tests/test_gamma.py::test_c
======================== 4 passed in 12.34s ==========================
"""


# ----------------------------------------------------------------------
# parse
# ----------------------------------------------------------------------

class TestParse:
    def test_it_sums_setup_call_and_teardown_per_module(self, tool):
        got = tool.parse(_LOG)
        assert got["mlsynth/tests/test_alpha.py"] == pytest.approx(3.0)

    def test_it_keeps_a_module_measured_at_zero(self, tool):
        """Dropping it would charge it the median, which is the bug."""
        got = tool.parse(_LOG)
        assert "mlsynth/tests/test_gamma.py" in got
        assert got["mlsynth/tests/test_gamma.py"] == 0.0

    def test_it_spans_both_test_roots(self, tool):
        got = tool.parse(_LOG)
        assert got["benchmarks/tests/test_beta.py"] == pytest.approx(1.0)

    def test_it_ignores_everything_that_is_not_a_duration_line(self, tool):
        assert tool.parse("4 passed\nnothing here\n") == {}

    def test_an_empty_log_parses_to_an_empty_map(self, tool):
        assert tool.parse("") == {}


# ----------------------------------------------------------------------
# The modules the map has not seen
# ----------------------------------------------------------------------

class TestMissingModules:
    def test_it_finds_the_module_absent_from_the_map(self, tool, tmp_path):
        root = tmp_path / "mlsynth" / "tests"
        root.mkdir(parents=True)
        (root / "test_seen.py").write_text("")
        (root / "test_unseen.py").write_text("")
        got = tool.missing_modules({"mlsynth/tests/test_seen.py": 1.0},
                                   roots=("mlsynth/tests",), base=tmp_path)
        assert got == ["mlsynth/tests/test_unseen.py"]

    def test_it_is_sorted_and_posix_spelled(self, tool, tmp_path):
        root = tmp_path / "mlsynth" / "tests"
        root.mkdir(parents=True)
        for name in ("test_b.py", "test_a.py", "test_c.py"):
            (root / name).write_text("")
        got = tool.missing_modules({}, roots=("mlsynth/tests",), base=tmp_path)
        assert got == ["mlsynth/tests/test_a.py", "mlsynth/tests/test_b.py",
                       "mlsynth/tests/test_c.py"]
        assert all("\\" not in m for m in got)

    def test_a_complete_map_has_nothing_missing(self, tool, tmp_path):
        root = tmp_path / "mlsynth" / "tests"
        root.mkdir(parents=True)
        (root / "test_only.py").write_text("")
        assert tool.missing_modules({"mlsynth/tests/test_only.py": 0.0},
                                    roots=("mlsynth/tests",),
                                    base=tmp_path) == []

    def test_a_root_that_does_not_exist_contributes_nothing(self, tool,
                                                            tmp_path):
        assert tool.missing_modules({}, roots=("nope",), base=tmp_path) == []

    def test_the_committed_map_and_the_repository_agree_on_the_roots(self, tool):
        """A real call, so the roots in the tool are the roots on disk.

        This does not assert the map is complete -- that is the thing the tool
        exists to fix, and asserting it here would make adding a test file a
        failing build. It asserts the reverse: every key in the map is a file
        that exists, so a rename cannot leave a duration behind pointing at
        nothing.
        """
        existing = tool.load_existing(_ROOT / "mlsynth" / "tests"
                                      / "_shard_durations.json")
        for module in existing["modules"]:
            assert (_ROOT / module).exists(), module


# ----------------------------------------------------------------------
# Merging a subset into the committed map
# ----------------------------------------------------------------------

class TestMerge:
    def test_measured_modules_replace_their_entries(self, tool):
        got = tool.merge_modules({"a.py": 1.0, "b.py": 2.0}, {"b.py": 9.0})
        assert got == {"a.py": 1.0, "b.py": 9.0}

    def test_unmeasured_modules_are_kept(self, tool):
        got = tool.merge_modules({"a.py": 1.0}, {"b.py": 2.0})
        assert got == {"a.py": 1.0, "b.py": 2.0}

    def test_the_result_is_sorted_by_module(self, tool):
        got = tool.merge_modules({"c.py": 1.0}, {"a.py": 2.0, "b.py": 3.0})
        assert list(got) == ["a.py", "b.py", "c.py"]

    def test_merging_into_an_empty_map_is_the_measured_map(self, tool):
        assert tool.merge_modules({}, {"a.py": 1.0}) == {"a.py": 1.0}

    def test_a_zero_measurement_overwrites_a_stale_nonzero_one(self, tool):
        """0.0 is a measurement, not a missing value."""
        assert tool.merge_modules({"a.py": 5.0}, {"a.py": 0.0}) == {"a.py": 0.0}


class TestProvenance:
    """Durations from two machines are not comparable, so a merge checks.

    The tool's own docstring says to measure on a quiet machine because noise
    here is the thing being corrected for. Timings from a different
    interpreter or a different container are the same problem one step larger,
    and a merge is the only path that can mix them.
    """

    def test_the_current_environment_matches_itself(self, tool):
        ok, why = tool.provenance_matches({
            "python": platform.python_version(),
            "platform": platform.platform(),
        })
        assert ok and why == ""

    @pytest.mark.parametrize("field", ["python", "platform"])
    def test_a_different_environment_is_refused_and_says_which(self, tool,
                                                               field):
        payload = {"python": platform.python_version(),
                   "platform": platform.platform()}
        payload[field] = "something-else"
        ok, why = tool.provenance_matches(payload)
        assert not ok
        assert field in why
        assert "something-else" in why

    def test_a_map_with_no_recorded_provenance_is_accepted(self, tool):
        """An older map predates the fields; refusing it would strand it."""
        ok, why = tool.provenance_matches({})
        assert ok and why == ""


# ----------------------------------------------------------------------
# The command line
# ----------------------------------------------------------------------

def _run(*args, cwd):
    return subprocess.run([sys.executable, str(_TOOL), *args],
                          capture_output=True, text=True, cwd=str(cwd))


def _seed_map(base: Path, modules: dict, *, provenance: bool = True) -> Path:
    out = base / "mlsynth" / "tests" / "_shard_durations.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"note": "seeded", "modules": modules}
    if provenance:
        payload["python"] = platform.python_version()
        payload["platform"] = platform.platform()
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


class TestTheCommandLine:
    def test_a_subset_without_merge_is_refused(self, tmp_path):
        """Otherwise measuring one module silently discards 484 entries."""
        out = _seed_map(tmp_path, {"mlsynth/tests/test_keep.py": 7.0})
        before = out.read_text()
        log = tmp_path / "log.txt"
        log.write_text(_LOG)
        proc = _run("--from-file", str(log), "--module",
                    "mlsynth/tests/test_alpha.py", cwd=tmp_path)
        assert proc.returncode != 0
        assert "--merge" in proc.stdout + proc.stderr
        assert out.read_text() == before

    def test_merge_keeps_the_entries_it_did_not_measure(self, tmp_path):
        out = _seed_map(tmp_path, {"mlsynth/tests/test_keep.py": 7.0,
                                   "mlsynth/tests/test_alpha.py": 99.0})
        log = tmp_path / "log.txt"
        log.write_text(_LOG)
        proc = _run("--from-file", str(log), "--merge", cwd=tmp_path)
        assert proc.returncode == 0, proc.stderr
        got = json.loads(out.read_text())["modules"]
        assert got["mlsynth/tests/test_keep.py"] == 7.0
        assert got["mlsynth/tests/test_alpha.py"] == 3.0
        assert got["mlsynth/tests/test_gamma.py"] == 0.0

    def test_without_merge_the_map_is_replaced(self, tmp_path):
        """The full-run path is unchanged: it owns the whole file."""
        out = _seed_map(tmp_path, {"mlsynth/tests/test_keep.py": 7.0})
        log = tmp_path / "log.txt"
        log.write_text(_LOG)
        proc = _run("--from-file", str(log), cwd=tmp_path)
        assert proc.returncode == 0, proc.stderr
        got = json.loads(out.read_text())["modules"]
        assert "mlsynth/tests/test_keep.py" not in got
        assert set(got) == {"mlsynth/tests/test_alpha.py",
                            "benchmarks/tests/test_beta.py",
                            "mlsynth/tests/test_gamma.py"}

    def test_a_merge_across_environments_is_refused_and_writes_nothing(
        self, tmp_path
    ):
        out = _seed_map(tmp_path, {"mlsynth/tests/test_keep.py": 7.0})
        payload = json.loads(out.read_text())
        payload["python"] = "2.7.0"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        before = out.read_text()
        log = tmp_path / "log.txt"
        log.write_text(_LOG)
        proc = _run("--from-file", str(log), "--merge", cwd=tmp_path)
        assert proc.returncode != 0
        assert "2.7.0" in proc.stdout + proc.stderr
        assert out.read_text() == before

    def test_a_log_with_no_durations_is_refused(self, tmp_path):
        _seed_map(tmp_path, {})
        log = tmp_path / "log.txt"
        log.write_text("4 passed in 1.00s\n")
        proc = _run("--from-file", str(log), "--merge", cwd=tmp_path)
        assert proc.returncode != 0

    def test_missing_needs_no_module_list(self, tmp_path):
        """--missing selects the modules the map has not seen."""
        root = tmp_path / "mlsynth" / "tests"
        root.mkdir(parents=True, exist_ok=True)
        (root / "test_alpha.py").write_text("")
        (root / "test_keep.py").write_text("")
        out = _seed_map(tmp_path, {"mlsynth/tests/test_keep.py": 7.0})
        log = tmp_path / "log.txt"
        log.write_text(_LOG)
        proc = _run("--from-file", str(log), "--merge", "--missing",
                    cwd=tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert "test_alpha.py" in proc.stdout
        got = json.loads(out.read_text())["modules"]
        assert got["mlsynth/tests/test_keep.py"] == 7.0

    def test_missing_with_nothing_missing_is_a_pass_that_changes_nothing(
        self, tmp_path
    ):
        root = tmp_path / "mlsynth" / "tests"
        root.mkdir(parents=True, exist_ok=True)
        (root / "test_keep.py").write_text("")
        out = _seed_map(tmp_path, {"mlsynth/tests/test_keep.py": 7.0})
        before = out.read_text()
        proc = _run("--merge", "--missing", cwd=tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert out.read_text() == before
