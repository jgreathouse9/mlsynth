"""Selecting the benchmark cases a change can actually reach.

``pr-suite`` runs the whole registry on every pull request, and the wall clock is
set by a handful of Monte Carlo and MIP cases -- the SYNDES and MAREX ones -- that
most changes cannot touch. Selecting by dependency turns those into the ones that
run when they are implicated and not otherwise.

Two approaches do not work here, and the manifest exists because of them. Cases
import inside their functions, so a static scan sees entry points and not reach;
and ``mlsynth/__init__.py`` imports every estimator, so tracing imports gives all
834 library files for a case that executes one. What the manifest records is
which files a case's ``run()`` actually *executes*, measured with coverage.

The selection rule is deliberately lopsided. A case with no manifest entry always
runs, and a changed file no entry mentions selects every case. Both directions
cost time and neither can silence a case, which is the failure mode that matters:
a skipped check reports success, and a skipped success is indistinguishable from
a real one.

Levels: smoke, unit invariants, edge, failure.
"""
import json

import pytest

from benchmarks import case_deps, registry


# --------------------------------------------------------------------------- smoke
def test_the_manifest_loads():
    m = case_deps.load()
    assert isinstance(m, dict)


def test_selecting_on_no_changes_selects_nothing_mapped():
    """An empty diff implicates nothing, so only unmapped cases run."""
    m = {"a": ["mlsynth/x.py"], "b": ["mlsynth/y.py"]}
    assert case_deps.select([], ["a", "b"], m) == []


# ------------------------------------------------------------------ unit invariants
def test_a_case_runs_when_a_file_it_executes_changes():
    m = {"a": ["mlsynth/x.py"], "b": ["mlsynth/y.py"]}
    assert case_deps.select(["mlsynth/x.py"], ["a", "b"], m) == ["a"]


def test_a_case_does_not_run_when_its_files_are_untouched():
    m = {"a": ["mlsynth/x.py"], "b": ["mlsynth/y.py"]}
    assert "b" not in case_deps.select(["mlsynth/x.py"], ["a", "b"], m)


def test_an_unmapped_case_always_runs():
    """No entry means no evidence, and no evidence means run it."""
    m = {"a": ["mlsynth/x.py"]}
    assert case_deps.select(["mlsynth/x.py"], ["a", "b"], m) == ["a", "b"]
    assert case_deps.select(["docs/z.rst"], ["a", "b"], m) == ["b"]


def test_a_changed_file_no_entry_mentions_selects_everything():
    """An unrecognised dependency is a hole in the map, so widen, never narrow."""
    m = {"a": ["mlsynth/x.py"], "b": ["mlsynth/y.py"]}
    assert case_deps.select(["mlsynth/brand_new.py"], ["a", "b"], m) == ["a", "b"]


def test_a_doc_change_selects_nothing():
    """Prose reaches no case, and is not a hole in the map either."""
    m = {"a": ["mlsynth/x.py"]}
    assert case_deps.select(["docs/ppscm.rst", "README.md"], ["a"], m) == []


def test_the_case_file_itself_selects_it():
    m = {"a": ["mlsynth/x.py"]}
    assert case_deps.select(["benchmarks/cases/a.py"], ["a"], m) == ["a"]


def test_selection_preserves_the_given_order():
    m = {"a": ["mlsynth/x.py"], "b": ["mlsynth/x.py"]}
    assert case_deps.select(["mlsynth/x.py"], ["b", "a"], m) == ["b", "a"]


def test_a_data_file_selects_the_cases_that_read_it():
    m = {"a": ["basedata/p99.csv"], "b": ["mlsynth/y.py"]}
    assert case_deps.select(["basedata/p99.csv"], ["a", "b"], m) == ["a"]


# ------------------------------------------------------------------------- edge
def test_an_empty_manifest_runs_everything():
    assert case_deps.select(["mlsynth/x.py"], ["a", "b"], {}) == ["a", "b"]


def test_a_case_measured_to_touch_nothing_is_mapped_not_unmapped():
    """An empty list is a finding -- the case executed no library file -- and it
    is distinct from a missing entry, which means nobody looked.

    The changed file has to be claimed by some other entry for the distinction to
    show. If no entry claims it the map has a hole, and the hole widens the
    selection before the empty list can narrow it, which is the right order.
    """
    m = {"a": [], "b": ["mlsynth/x.py"]}
    assert case_deps.select(["mlsynth/x.py"], ["a", "b"], m) == ["b"]


def test_an_empty_entry_does_not_shield_a_case_from_a_hole_in_the_map():
    m = {"a": []}
    assert case_deps.select(["mlsynth/unclaimed.py"], ["a"], m) == ["a"]


# ---------------------------------------------------------------------- failure
@pytest.mark.parametrize("bad", [None, "mlsynth/x.py", 3])
def test_changed_must_be_a_sequence_of_paths(bad):
    with pytest.raises((TypeError, ValueError)):
        case_deps.select(bad, ["a"], {"a": []})


# ------------------------------------------------------- the manifest as shipped
class TestTheShippedManifest:
    @staticmethod
    @pytest.fixture(scope="class")
    def manifest():
        return case_deps.load()

    def test_every_mapped_case_is_registered(self, manifest):
        """A stale entry names a case that no longer exists."""
        unknown = set(manifest) - set(registry.CASES)
        assert not unknown, unknown

    def test_every_path_it_names_exists(self, manifest):
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        missing = {p for deps in manifest.values() for p in deps
                   if not (root / p).exists()}
        assert not missing, missing

    def test_it_is_sorted_and_deduplicated(self, manifest):
        """So a regenerated manifest diffs as content, not as ordering."""
        for name, deps in manifest.items():
            assert deps == sorted(set(deps)), name

    def test_it_names_no_build_artifact(self, manifest):
        """The audit hook sees bytecode reads; a diff never names one."""
        junk = {p for deps in manifest.values() for p in deps
                if "__pycache__" in p or p.endswith((".pyc", ".pyo"))}
        assert not junk, junk

    def test_every_entry_is_reachable_by_a_diff(self, manifest):
        """Every path must sit under a prefix the selection rule recognises."""
        from benchmarks.case_deps import REACHABLE
        stray = {p for deps in manifest.values() for p in deps
                 if not p.startswith(REACHABLE)}
        assert not stray, stray

    def test_it_is_json_and_stable_on_disk(self):
        raw = case_deps.PATH.read_text()
        assert json.loads(raw) == case_deps.load()
        assert raw.endswith("\n")


class TestTheDriverAcceptsADiff:
    """``run_benchmarks.py --changed-from`` narrows the run to what a diff reaches."""

    @staticmethod
    def _argv(*extra):
        return ["run_benchmarks.py", "--all", *extra]

    def test_changed_from_reads_one_path_per_line(self, tmp_path):
        from benchmarks import run_benchmarks

        f = tmp_path / "diff.txt"
        f.write_text("mlsynth/a.py\n\nmlsynth/b.py\n")
        assert run_benchmarks._read_changed(f) == ["mlsynth/a.py", "mlsynth/b.py"]

    def test_an_empty_diff_file_is_not_the_same_as_no_flag(self, tmp_path):
        """An empty diff means nothing changed, which selects only unmapped cases."""
        from benchmarks import run_benchmarks

        f = tmp_path / "diff.txt"
        f.write_text("")
        assert run_benchmarks._read_changed(f) == []

    def test_a_missing_diff_file_is_refused(self, tmp_path):
        from benchmarks import run_benchmarks

        with pytest.raises((FileNotFoundError, OSError)):
            run_benchmarks._read_changed(tmp_path / "nope.txt")

    def test_selection_composes_with_sharding(self):
        """Selecting first and sharding after keeps every shard non-empty work."""
        m = {"a": ["mlsynth/x.py"], "b": ["mlsynth/y.py"], "c": ["mlsynth/x.py"]}
        picked = case_deps.select(["mlsynth/x.py"], ["a", "b", "c"], m)
        assert picked == ["a", "c"]
