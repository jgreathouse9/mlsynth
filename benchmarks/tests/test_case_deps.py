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


# ------------------------------------------------- holes the map can never fill
class TestPathsNoManifestCanEverClaim:
    """Three kinds of path are unclaimable by construction, so treating them as
    holes in the map is a permanent false positive.

    The widen-on-doubt rule is right when doubt is real: an unrecognised
    dependency might be one the map has not measured yet. These three cannot be.
    The registry loads only ``benchmarks.cases``, so no case executes a test
    module. A file added by the diff did not exist when any pre-existing case
    ran, so no case can have executed it. And ``mlsynth/__init__.py`` runs at
    import, before measurement starts, which is the separation that keeps a case
    from measuring all 834 library files -- so it is invisible to coverage no
    matter how many times the map is refreshed.

    Each therefore widened the selection to the whole registry on every run, and
    an estimator pull request tripped all three at once.
    """

    MAP = {"a": ["mlsynth/x.py"], "b": ["mlsynth/y.py"]}

    @pytest.mark.parametrize("path", [
        "mlsynth/tests/test_fdid.py",
        "benchmarks/tests/test_case_deps.py",
    ])
    def test_a_test_module_is_not_a_hole(self, path):
        assert case_deps.select([path], ["a", "b"], self.MAP) == []

    def test_the_package_init_is_not_a_hole(self):
        assert case_deps.select(["mlsynth/__init__.py"], ["a", "b"], self.MAP) == []

    def test_an_added_file_is_not_a_hole(self):
        """It did not exist when any pre-existing case ran."""
        picked = case_deps.select(["mlsynth/new.py"], ["a", "b"], self.MAP,
                                  added=["mlsynth/new.py"])
        assert picked == []

    def test_a_modified_file_the_map_does_not_claim_still_widens(self):
        """The rule narrows only where doubt is impossible, never where it is
        merely unlikely."""
        assert case_deps.select(["mlsynth/new.py"], ["a", "b"], self.MAP) == ["a", "b"]

    def test_one_modified_hole_widens_past_any_number_of_added_files(self):
        changed = ["mlsynth/added_one.py", "mlsynth/added_two.py", "mlsynth/edited.py"]
        picked = case_deps.select(changed, ["a", "b"], self.MAP,
                                  added=changed[:2])
        assert picked == ["a", "b"]

    def test_an_added_case_module_still_selects_its_own_case(self):
        """Unclaimable is not unselectable: the case's own file always runs it."""
        picked = case_deps.select(["benchmarks/cases/a.py"], ["a", "b"], self.MAP,
                                  added=["benchmarks/cases/a.py"])
        assert "a" in picked

    def test_an_added_file_a_manifest_entry_names_still_selects_its_case(self):
        """`added` suppresses the hole test, never a positive match."""
        picked = case_deps.select(["mlsynth/x.py"], ["a", "b"], self.MAP,
                                  added=["mlsynth/x.py"])
        assert picked == ["a"]

    def test_an_estimator_shaped_diff_selects_no_mapped_case(self):
        """The shape this rule exists for: a new estimator package, its tests,
        its export line, and a parametrize entry in a shared test."""
        changed = [
            "mlsynth/estimators/mosc.py",
            "mlsynth/utils/mosc_helpers/pipeline.py",
            "mlsynth/tests/test_mosc.py",
            "mlsynth/tests/test_result_contract.py",
            "mlsynth/__init__.py",
            "docs/mosc.rst",
        ]
        added = changed[:3]
        assert case_deps.select(changed, ["a", "b"], self.MAP, added=added) == []


# ---------------------------------------------------------------------- failure
@pytest.mark.parametrize("bad", [None, "mlsynth/x.py", 3])
def test_added_must_be_a_sequence_of_paths(bad):
    with pytest.raises((TypeError, ValueError)):
        case_deps.select(["mlsynth/x.py"], ["a"], {"a": []}, added=bad)


def test_the_shipped_manifest_names_nothing_unclaimable():
    """A measured entry naming one of these would mean the measurement is wrong."""
    manifest = case_deps.load()
    stray = {p for deps in manifest.values() for p in deps
             if p.startswith(case_deps.NEVER_REACHABLE)
             or p in case_deps.RE_EXPORT_ONLY}
    assert not stray, stray


def test_the_package_init_really_is_re_export_only():
    """`RE_EXPORT_ONLY` is a claim about the file, so check the file.

    A change to a module of imports and an ``__all__`` can add or remove a name;
    it cannot alter what any estimator computes. Put a statement with behaviour
    in here and that stops being true, so this fails and the entry comes out.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for rel in case_deps.RE_EXPORT_ONLY:
        tree = ast.parse((root / rel).read_text())
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, ast.Try):      # the __version__ lookup
                continue
            assert isinstance(node, ast.Assign), (rel, ast.dump(node)[:80])
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            assert targets <= {"__all__", "__version__"}, (rel, targets)


class TestTheDriverAcceptsTheAddedSubset:
    """``--added-from`` names the files the diff creates, which cannot be holes."""

    def test_added_from_is_optional(self):
        from benchmarks import run_benchmarks

        ap = run_benchmarks.build_parser()
        assert ap.parse_args(["--all"]).added_from is None

    def test_added_from_reads_one_path_per_line(self, tmp_path):
        from benchmarks import run_benchmarks

        f = tmp_path / "added.txt"
        f.write_text("mlsynth/a.py\n\nmlsynth/b.py\n")
        assert run_benchmarks._read_changed(f) == ["mlsynth/a.py", "mlsynth/b.py"]

    def test_added_without_changed_is_refused(self):
        """The added set is a subset of the diff; alone it says nothing."""
        from benchmarks import run_benchmarks

        ap = run_benchmarks.build_parser()
        args = ap.parse_args(["--all", "--added-from", "x.txt"])
        with pytest.raises(SystemExit):
            run_benchmarks._check_args(ap, args)


class TestTheMapBuilderShards:
    """The map is measured by running every case under coverage, which does not
    fit one job: at 199 cases the single job hit its 350-minute cap on each of
    the four days after it was added and was cancelled every time, so the
    manifest stayed at the five entries that shipped with it and the selection
    it exists to feed had nothing to select with.
    """

    def test_shard_zero_of_one_is_every_case(self):
        from tools import benchmark_deps

        names = ["a", "b", "c", "d", "e"]
        assert benchmark_deps.shard_of(names, 0, 1) == names

    def test_the_shards_partition_the_cases(self):
        from tools import benchmark_deps

        names = [f"c{i}" for i in range(23)]
        seen = []
        for i in range(4):
            seen += benchmark_deps.shard_of(names, i, 4)
        assert sorted(seen) == sorted(names)
        assert len(seen) == len(names)

    def test_the_shards_are_round_robin_like_the_runner(self):
        """Cost is not uniform across the registry, so slice the same way
        ``run_benchmarks.select_cases`` does and the heavy cases spread."""
        from tools import benchmark_deps

        names = [f"c{i}" for i in range(10)]
        assert benchmark_deps.shard_of(names, 1, 3) == names[1::3]

    @pytest.mark.parametrize("shard,num", [(-1, 2), (2, 2), (0, 0)])
    def test_an_out_of_range_shard_is_refused(self, shard, num):
        from tools import benchmark_deps

        with pytest.raises(ValueError):
            benchmark_deps.shard_of(["a", "b"], shard, num)


class TestTheMapBuilderCombinesSlices:
    """Each shard writes only what it measured; one job merges and commits."""

    def test_combining_unions_the_slices(self, tmp_path):
        from tools import benchmark_deps

        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_text(json.dumps({"one": ["mlsynth/x.py"]}))
        b.write_text(json.dumps({"two": ["mlsynth/y.py"]}))
        assert benchmark_deps.combine([a, b], base={}) == {
            "one": ["mlsynth/x.py"], "two": ["mlsynth/y.py"]}

    def test_a_slice_overrides_the_base_for_the_cases_it_measured(self, tmp_path):
        from tools import benchmark_deps

        a = tmp_path / "a.json"
        a.write_text(json.dumps({"one": ["mlsynth/new.py"]}))
        out = benchmark_deps.combine([a], base={"one": ["mlsynth/old.py"],
                                                "two": ["mlsynth/y.py"]})
        assert out == {"one": ["mlsynth/new.py"], "two": ["mlsynth/y.py"]}

    def test_a_case_no_slice_measured_keeps_its_base_entry(self, tmp_path):
        """A shard that died must not un-map everyone else's cases."""
        from tools import benchmark_deps

        a = tmp_path / "a.json"
        a.write_text(json.dumps({"one": ["mlsynth/x.py"]}))
        out = benchmark_deps.combine([a], base={"two": ["mlsynth/y.py"]})
        assert out["two"] == ["mlsynth/y.py"]

    def test_combining_nothing_returns_the_base_unchanged(self):
        from tools import benchmark_deps

        base = {"one": ["mlsynth/x.py"]}
        assert benchmark_deps.combine([], base=base) == base

    def test_a_missing_slice_is_refused(self, tmp_path):
        from tools import benchmark_deps

        with pytest.raises((FileNotFoundError, OSError)):
            benchmark_deps.combine([tmp_path / "nope.json"], base={})


class TestTheMapBuilderCommandLine:
    """``main`` is the part CI actually calls, so the two paths CI uses are
    exercised here: combining slices, and refusing a shard that does not exist.
    """

    def test_combine_writes_the_merged_map_and_measures_nothing(self, tmp_path, monkeypatch):
        from tools import benchmark_deps

        slice_ = tmp_path / "slice.json"
        slice_.write_text(json.dumps({"only": ["mlsynth/x.py"]}))
        out = tmp_path / "out.json"

        def _never(*a, **k):                      # measuring would take hours
            raise AssertionError("--combine must not trace a case")

        monkeypatch.setattr(benchmark_deps, "_trace", _never)
        monkeypatch.setattr("sys.argv", ["benchmark_deps.py", "--combine",
                                         str(slice_), "--out", str(out)])
        assert benchmark_deps.main() == 0
        assert json.loads(out.read_text())["only"] == ["mlsynth/x.py"]

    def test_a_shard_outside_the_split_is_refused(self, monkeypatch):
        from tools import benchmark_deps

        monkeypatch.setattr("sys.argv", ["benchmark_deps.py", "--all",
                                         "--shard", "4", "--num-shards", "4"])
        with pytest.raises(SystemExit):
            benchmark_deps.main()
