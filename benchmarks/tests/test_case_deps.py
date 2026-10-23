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
from pathlib import Path

import pytest

from benchmarks import case_deps, registry


#: The inventory as shipped. The nightly ``case-deps-commit`` job rewrites it, so
#: a test that reads it by default asserts whatever the last sweep found.
SHIPPED_SWEPT = case_deps.SWEPT_PATH


@pytest.fixture(autouse=True, scope="module")
def _no_shipped_inventory(tmp_path_factory):
    """Select against no inventory unless a test passes one.

    Every ``select`` here that omits ``swept`` was written against the rule
    without an inventory. The first inventory to reach ``main`` (b50e1395, a
    ``[skip ci]`` commit) recorded ``benchmarks/constraints.txt`` as read by no
    case, and the prose rule's negative control went red on ``main`` six hours
    before a pull request ran it. A test that means the shipped inventory passes
    ``swept=case_deps.load_swept(SHIPPED_SWEPT)``. Module scope, so the
    ``hypothesis`` properties below can run under it.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(case_deps, "SWEPT_PATH",
                   tmp_path_factory.mktemp("inventory") / "none.json")
        yield


def test_no_test_here_reads_the_shipped_inventory_by_default():
    """The fixture above is in force: the default inventory is empty, whatever
    the nightly job last committed."""
    assert case_deps.load_swept(SHIPPED_SWEPT), \
        "no shipped inventory, so this check has no power"
    assert case_deps.load_swept() == frozenset()


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


def test_every_untracked_prefix_is_gitignored():
    """``UNTRACKED`` is a claim about git, so check git's own list: a prefix
    that stopped being ignored could be named by a diff again."""
    root = Path(__file__).resolve().parents[2]
    ignored = {line.strip() for line in (root / ".gitignore").read_text().splitlines()}
    assert case_deps.UNTRACKED
    for prefix in case_deps.UNTRACKED:
        assert prefix.startswith(case_deps.REACHABLE)
        assert prefix in ignored, prefix


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


class TestTheMapClaimsWhatADiffCanName:
    """The map keeps every reachable path a case executed or read.

    It used to keep ``mlsynth/`` and ``basedata/`` only, and measure only
    ``mlsynth``, so nothing under ``benchmarks/`` -- ``compare.py``, a
    reference's committed CSV, a study module a case imports -- could ever be
    claimed. Without an inventory each was a hole and ran the whole registry;
    with one each would read as covered by nothing and run none of the cases
    that use it.
    """

    @staticmethod
    def _keep(paths, name="c"):
        from tools.benchmark_deps import claimable
        return claimable(paths, name)

    def test_the_benchmarks_tree_is_claimed(self):
        paths = ["benchmarks/compare.py",
                 "benchmarks/reference/c/expected.csv",
                 "benchmarks/studies/s/search.py"]
        assert self._keep(paths) == sorted(paths)

    def test_the_library_and_the_panels_still_are(self):
        paths = ["mlsynth/utils/x.py", "basedata/p.csv"]
        assert self._keep(paths) == sorted(paths)

    def test_the_cases_own_module_is_implied_not_stored(self):
        assert self._keep(["benchmarks/cases/c.py", "benchmarks/cases/d.py"],
                          name="c") == ["benchmarks/cases/d.py"]

    @pytest.mark.parametrize("path", [
        "mlsynth/tests/conftest.py", "benchmarks/tests/test_x.py",
        "mlsynth/__pycache__/x.cpython-311.pyc", "docs/x.rst", "setup.py",
        "mlsynth/utils/x.pyc",
        "benchmarks/reference/.cache/dpsc/src/matrix.py",
        "benchmarks/reference/.cache/CAST_panel/sample_data/expansion.csv",
    ])
    def test_what_no_diff_can_reach_is_dropped(self, path):
        assert self._keep([path]) == []

    def test_the_tracer_measures_both_python_trees(self):
        pytest.importorskip("coverage")           # the build job installs it
        from tools.benchmark_deps import ROOT, measurement

        cov = measurement()
        roots = {Path(s).resolve() for s in cov.config.source}
        assert {ROOT / "mlsynth", ROOT / "benchmarks"} <= roots

    def test_the_tracer_is_the_fast_one_where_it_exists(self):
        """Under the pure-Python tracer the 209 cases measured on 10-21 took
        1,149 minutes against 240 untraced, and the shards were cancelled at
        their cap. Coverage's ``sysmon`` core recorded the same 23 files on
        ``wan_pda_vs_scm`` in 100.0 s against 99.2 s untraced. Before 3.12
        there is no ``sys.monitoring``, and the pure-Python tracer stays."""
        import sys

        pytest.importorskip("coverage")
        from tools.benchmark_deps import measurement

        import coverage

        config = measurement().config
        outer = coverage.Coverage.current()
        if sys.version_info >= (3, 12):
            assert config.core == "sysmon" and not config.timid
        else:                                  # the build job's 3.11
            # a C tracer when one is already running: see TestTheTracerNests
            assert config.timid == (outer is None or outer.config.timid)


class TestTheTracerNests:
    """The map's coverage session may start inside another one: the daily
    coverage job runs this suite under pytest-cov, a C tracer. A pure-Python
    tracer nested inside a C tracer records no line at all, so on Python 3.11
    the map claimed only the files the audit hook saw opened, and main's
    coverage run failed on 2026-10-22. A C tracer or ``sysmon`` nested
    inside one records normally."""

    @staticmethod
    def _outer(timid):
        from types import SimpleNamespace
        return SimpleNamespace(config=SimpleNamespace(timid=timid))

    def test_sys_monitoring_is_used_where_it_exists(self):
        from tools.benchmark_deps import _tracer
        assert _tracer((3, 12), None) == (False, "sysmon")
        assert _tracer((3, 13), self._outer(False)) == (False, "sysmon")

    def test_alone_before_3_12_the_python_tracer_stays(self):
        from tools.benchmark_deps import _tracer
        assert _tracer((3, 11), None) == (True, None)

    def test_inside_a_c_tracer_it_nests_a_c_tracer(self):
        from tools.benchmark_deps import _tracer
        assert _tracer((3, 11), self._outer(False)) == (False, None)

    def test_inside_a_python_tracer_it_nests_a_python_tracer(self):
        from tools.benchmark_deps import _tracer
        assert _tracer((3, 11), self._outer(True)) == (True, None)


class TestTheMapMeasuresWhatTheRunnerRuns:
    """The runner does three things for a case: it imports the module, which
    builds ``EXPECTED`` and reads the case's reference bundle; it calls
    ``run()``; and it ``compare()``s the two. The map measured ``run()`` alone,
    so a reference read at import, ``compare.py`` and the runner itself were
    never claimed. Without an inventory each was a hole; with one, each would
    read as covered by nothing, and a changed reference value would run no case.
    """

    @pytest.fixture
    def case(self, tmp_path, monkeypatch):
        import sys
        import textwrap

        pytest.importorskip("coverage")           # the build job installs it
        from tools.benchmark_deps import ROOT

        panel = sorted((ROOT / "basedata").glob("*.csv"))[0]

        def make(body):
            name = f"fake_case_{len(list(tmp_path.iterdir()))}"
            (tmp_path / f"{name}.py").write_text(textwrap.dedent(body).format(
                root=str(ROOT), panel=str(panel)))
            monkeypatch.syspath_prepend(str(tmp_path))
            monkeypatch.setitem(registry.CASES, name, name)
            sys.modules.pop(name, None)
            return name, panel.relative_to(ROOT).as_posix()
        return make

    def test_import_run_and_compare_are_all_claimed(self, case):
        from tools.benchmark_deps import _trace

        name, panel = case('''
            from pathlib import Path
            _ROOT = Path({root!r})
            (_ROOT / "benchmarks" / "reference" / "__init__.py").read_text()
            EXPECTED = {{"x": (1.0, 0.1)}}

            def run():
                Path({panel!r}).read_text()
                return {{"x": 1.0}}
            ''')
        deps, _ = _trace(name)
        assert {"benchmarks/reference/__init__.py",       # read at import
                panel,                                    # read in run()
                "benchmarks/compare.py",                  # compared after it
                "benchmarks/registry.py",                 # loaded through it
                "benchmarks/run_benchmarks.py"} <= set(deps), deps

    def test_a_run_under_another_coverage_is_still_measured(self, case, monkeypatch):
        """The 3.11 path inside a C tracer, as the daily coverage job runs it:
        the case's executed files are claimed, not only the files it read."""
        import coverage
        from tools import benchmark_deps

        monkeypatch.setattr(benchmark_deps, "_python", lambda: (3, 11))
        name, panel = case('''
            EXPECTED = {{"x": (1.0, 0.1)}}

            def run():
                return {{"x": 1.0}}
            ''')
        outer = coverage.Coverage(data_file=None, timid=False)
        outer.set_option("run:core", "ctrace")
        outer.start()
        try:
            deps, _ = benchmark_deps._trace(name)
        finally:
            outer.stop()
        assert {"benchmarks/compare.py", "benchmarks/registry.py",
                "benchmarks/run_benchmarks.py"} <= set(deps), deps

    @pytest.mark.parametrize("body", [
        '''
        from benchmarks.compare import BenchmarkSkipped
        EXPECTED = {{}}
        def run():
            raise BenchmarkSkipped("needs R")
        ''',
        '''
        EXPECTED = {{}}
        def run():
            raise RuntimeError("broke")
        ''',
    ], ids=["skipped", "raised"])
    def test_a_case_that_does_not_finish_is_left_unmeasured(self, case, body):
        """A partial measurement would under-claim, so the case stays unmapped
        and always runs; a raising case used to take the whole shard down."""
        from tools.benchmark_deps import Unmeasured, _trace

        name, _ = case(body)
        with pytest.raises(Unmeasured):
            _trace(name)


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

    def test_a_case_its_slice_could_not_measure_loses_its_base_entry(self, tmp_path):
        """The shard drops an unmeasured case so that it always runs. Restoring
        its old entry from the base undid that, and one such entry vetoed the
        inventory: on 22 October two scpi cases raised under pandas 3.1 and no
        inventory was written, so every pull request touching a module no case
        runs ran the whole registry (/rca)."""
        from tools import benchmark_deps

        a = tmp_path / "a.json"
        a.write_text(json.dumps({"one": ["mlsynth/x.py"],
                                 benchmark_deps.UNMEASURED: ["two"]}))
        out = benchmark_deps.combine([a], base={"two": ["mlsynth/old.py"],
                                                "three": ["mlsynth/y.py"]})
        assert out == {"one": ["mlsynth/x.py"], "three": ["mlsynth/y.py"]}

    def test_the_unmeasured_list_never_becomes_a_case(self, tmp_path):
        from tools import benchmark_deps

        a = tmp_path / "a.json"
        a.write_text(json.dumps({benchmark_deps.UNMEASURED: ["two"]}))
        out = benchmark_deps.combine([a], base={})
        assert benchmark_deps.UNMEASURED not in out and out == {}
        assert benchmark_deps.measured_in([a]) == set()

    def test_a_case_another_slice_measured_is_kept(self, tmp_path):
        """Shards are disjoint, but a measurement outranks a failure to make
        one whatever order the slices come in."""
        from tools import benchmark_deps

        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_text(json.dumps({"one": ["mlsynth/x.py"]}))
        b.write_text(json.dumps({benchmark_deps.UNMEASURED: ["one"]}))
        for order in ([a, b], [b, a]):
            assert benchmark_deps.combine(order, base={"one": ["mlsynth/old.py"]}) \
                == {"one": ["mlsynth/x.py"]}
            assert benchmark_deps.measured_in(order) == {"one"}

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

    def test_a_case_that_does_not_finish_is_dropped_and_the_rest_written(
            self, tmp_path, monkeypatch):
        """A case that skipped or raised leaves no entry, so it always runs;
        before, a raising case took the shard and its whole slice with it."""
        from tools import benchmark_deps

        def fake(name):
            if name == "stops":
                raise benchmark_deps.Unmeasured("error: RuntimeError: broke")
            return ["mlsynth/x.py"], 0.0

        out = tmp_path / "slice.json"
        monkeypatch.setattr(benchmark_deps, "_trace", fake)
        monkeypatch.setattr("sys.argv", ["benchmark_deps.py", "--case", "runs",
                                         "--case", "stops", "--out", str(out)])
        assert benchmark_deps.main() == 0
        assert json.loads(out.read_text()) == {
            "runs": ["mlsynth/x.py"], benchmark_deps.UNMEASURED: ["stops"]}

    @staticmethod
    def _nightly(tmp_path, monkeypatch, base, slices):
        """``--combine`` as the nightly job runs it, on a shipped map and an
        inventory path of the test's own."""
        from tools import benchmark_deps

        shipped = tmp_path / "case_deps.json"
        shipped.write_text(json.dumps(base))
        swept = tmp_path / "case_deps_swept.json"
        monkeypatch.setattr(case_deps, "PATH", shipped)
        monkeypatch.setattr(case_deps, "SWEPT_PATH", swept)
        monkeypatch.setattr(case_deps, "reachable_inventory",
                            lambda root=None: ["mlsynth/utils/lonely.py"])
        paths = []
        for i, body in enumerate(slices):
            p = tmp_path / f"case_deps.shard-{i}.json"
            p.write_text(json.dumps(body))
            paths.append(str(p))
        monkeypatch.setattr("sys.argv", ["benchmark_deps.py", "--combine", *paths])
        assert benchmark_deps.main() == 0
        return json.loads(shipped.read_text()), swept

    def test_a_case_that_errors_in_the_sweep_no_longer_vetoes_the_inventory(
            self, tmp_path, monkeypatch):
        """22 October in miniature: ``stops`` had an entry and raised tonight.
        It comes out of the map, so it always runs, and the inventory is
        written, so a module no case runs narrows the selection."""
        from tools import benchmark_deps

        merged, swept = self._nightly(
            tmp_path, monkeypatch,
            base={"runs": ["mlsynth/old.py"], "stops": ["mlsynth/y.py"]},
            slices=[{"runs": ["mlsynth/x.py"]},
                    {benchmark_deps.UNMEASURED: ["stops"]}])
        assert merged == {"runs": ["mlsynth/x.py"]}
        assert case_deps.load_swept(swept) == {"mlsynth/utils/lonely.py"}
        assert case_deps.select(["mlsynth/utils/lonely.py"], ["runs", "stops"],
                                merged, swept=case_deps.load_swept(swept)) \
            == ["stops"]

    def test_a_dead_shard_still_withholds_the_inventory_and_says_so(
            self, tmp_path, monkeypatch, capsys):
        """A case no slice mentions keeps its entry and blocks the inventory,
        as before; the refusal is a warning on the run, not a line in a green
        job's log that nobody reads."""
        merged, swept = self._nightly(
            tmp_path, monkeypatch,
            base={"runs": ["mlsynth/old.py"], "lost": ["mlsynth/y.py"]},
            slices=[{"runs": ["mlsynth/x.py"]}])
        assert merged["lost"] == ["mlsynth/y.py"] and not swept.exists()
        out = capsys.readouterr().out
        assert "::warning" in out and "not recorded" in out and "lost" in out

    def test_a_shard_outside_the_split_is_refused(self, monkeypatch):
        from tools import benchmark_deps

        monkeypatch.setattr("sys.argv", ["benchmark_deps.py", "--all",
                                         "--shard", "4", "--num-shards", "4"])
        with pytest.raises(SystemExit):
            benchmark_deps.main()


class TestProseIsProseWhereverItSits:
    """``docs/ppscm.rst`` reaches no case because ``docs/`` is outside
    :data:`REACHABLE`. ``benchmarks/README.md`` is the same prose one directory
    over, and was a hole in the map on every diff that touched it.

    The existing rule already says prose reaches nothing; it said it by
    directory. Saying it by kind as well is the same claim, applied where the
    directory test cannot reach.
    """

    MAP = {"a": ["mlsynth/x.py"], "b": ["mlsynth/y.py"]}

    @pytest.mark.parametrize("path", [
        "benchmarks/README.md",
        "benchmarks/reference/mosc_spike/README.md",
        "mlsynth/utils/NOTES.rst",
    ])
    def test_prose_under_a_reachable_prefix_is_not_a_hole(self, path):
        assert case_deps.select([path], ["a", "b"], self.MAP) == []

    @pytest.mark.parametrize("path", [
        "benchmarks/data/donors.txt",
        "basedata/prop99.csv",
        "benchmarks/R/sc_references.R",
    ])
    def test_a_file_a_case_could_read_is_still_a_hole(self, path):
        """The reads a case makes are measured, so these are claimable and,
        with no inventory to settle them, their absence from the map is real
        doubt. The rule is by suffix: ``.txt`` and ``.R`` are not prose.

        This test named ``benchmarks/constraints.txt`` and
        ``benchmarks/R/requirements.R`` until the first inventory reached
        ``main``. No case reads either: the install step does, before any case
        runs (:class:`TestTheInstallReachesEveryCase`).
        """
        assert case_deps.select([path], ["a", "b"], self.MAP,
                                swept=frozenset()) == ["a", "b"]


class TestTheInstallReachesEveryCase:
    """A file the pull-request suite's install reads changes what every case
    runs on.

    ``pip install -e '.[all]' -c benchmarks/constraints.txt`` reads
    ``pyproject.toml`` and the constraints before any case runs, so no
    measurement of ``run()`` claims either, and a complete sweep records the
    constraints as read by nothing. Through the inventory a change to the pins
    ran the 26 unmapped cases of 243; ``pyproject.toml`` sits outside
    :data:`~benchmarks.case_deps.REACHABLE` and had run the same 26 since the map
    began. Both now run every case.
    """

    MAP = {"a": ["mlsynth/x.py"], "b": ["mlsynth/y.py"]}

    @pytest.mark.parametrize("path", ["benchmarks/constraints.txt", "pyproject.toml"])
    def test_it_selects_every_case(self, path):
        assert case_deps.select([path], ["a", "b"], self.MAP) == ["a", "b"]

    @pytest.mark.parametrize("path", ["benchmarks/constraints.txt", "pyproject.toml"])
    def test_an_inventory_that_records_it_does_not_settle_it(self, path):
        """The incident: the sweep found nothing that reads it."""
        assert case_deps.select([path], ["a", "b"], self.MAP,
                                swept=frozenset({path})) == ["a", "b"]

    @pytest.mark.parametrize("path", ["benchmarks/constraints.txt", "pyproject.toml"])
    def test_a_diff_that_creates_it_still_reaches_every_case(self, path):
        """A created file is not a hole because no earlier run read it. A
        created pin file is read by this run's install all the same."""
        assert case_deps.select([path], ["a", "b"], self.MAP,
                                added=[path]) == ["a", "b"]

    def test_the_shipped_map_and_inventory_run_the_registry_on_a_pin_change(self):
        """On the data that produced the incident: a change to the constraints
        alone selected 26 of 243 cases."""
        cases = list(registry.CASES)
        picked = case_deps.select(["benchmarks/constraints.txt"], cases,
                                  swept=case_deps.load_swept(SHIPPED_SWEPT))
        assert picked == cases

    def test_the_order_is_the_candidates_order(self):
        assert case_deps.select(["pyproject.toml", "mlsynth/x.py"], ["b", "c", "a"],
                                self.MAP) == ["b", "c", "a"]

    def test_no_candidates_select_nothing(self):
        assert case_deps.select(["benchmarks/constraints.txt"], [], self.MAP) == []

    @pytest.mark.parametrize("path", ["requirements.txt",
                                      "benchmarks/R/requirements.R"])
    def test_a_file_this_install_does_not_read_keeps_the_map_rule(self, path):
        """``build.yml`` installs ``requirements.txt`` and the R jobs run
        ``requirements.R``; ``pr-suite`` does neither, so a change to one cannot
        move a case it runs."""
        assert case_deps.select([path], ["a", "b"], self.MAP,
                                swept=frozenset({path})) == []

    def test_the_list_is_not_prose_and_not_unreachable(self):
        for path in case_deps.ENVIRONMENT:
            assert not path.endswith(case_deps.PROSE_SUFFIXES)
            assert not path.startswith(case_deps.NEVER_REACHABLE)


# --------------------------------------------------------------------------
# The sweep inventory: "never measured" and "measured and covered by nothing"
# are different facts, and the map used to store only the first.
# --------------------------------------------------------------------------
class TestSweepInventory:
    """A complete sweep settles a file's status even when nothing claims it.

    The manifest holds positive claims, so an unclaimed file under a reachable
    prefix has two readings: the sweep never ran on the case that would have
    claimed it, or the sweep ran on everything and no case executes it. The
    first is doubt and widens the selection; the second is an answer and does
    not. Without the distinction every uncovered file widened the selection to
    the whole registry, which is the state 780 of ``mlsynth``'s Python files
    were in.
    """

    MAP = {"a": ["mlsynth/x.py"], "b": ["mlsynth/y.py"]}
    UNCLAIMED = "mlsynth/utils/lonely_helpers/engine.py"

    def test_an_unclaimed_file_is_a_hole_without_an_inventory(self):
        """Today's behaviour, which has to survive the change."""
        assert case_deps.select([self.UNCLAIMED], ["a", "b"], self.MAP,
                                swept=frozenset()) == ["a", "b"]

    def test_a_swept_unclaimed_file_is_not_a_hole(self):
        """The sweep looked at it and no case ran it, so nothing can reach it."""
        assert case_deps.select([self.UNCLAIMED], ["a", "b"], self.MAP,
                                swept=frozenset({self.UNCLAIMED})) == []

    def test_a_file_absent_from_the_inventory_is_still_a_hole(self):
        """Added after the sweep, so the sweep says nothing about it."""
        assert case_deps.select(["mlsynth/utils/brand_new.py"], ["a", "b"],
                                self.MAP,
                                swept=frozenset({self.UNCLAIMED})) == ["a", "b"]

    def test_the_inventory_does_not_suppress_a_positive_match(self):
        """Being swept is not an exemption: a claimed file still selects."""
        assert case_deps.select(["mlsynth/x.py"], ["a", "b"], self.MAP,
                                swept=frozenset({"mlsynth/x.py"})) == ["a"]

    def test_a_case_with_no_entry_still_runs_whatever_the_inventory_says(self):
        """An unmeasured case is unmeasured; the inventory is about files."""
        assert case_deps.select([self.UNCLAIMED], ["a", "b", "c"], self.MAP,
                                swept=frozenset({self.UNCLAIMED})) == ["c"]

    def test_an_absent_inventory_file_loads_as_empty(self, tmp_path):
        assert case_deps.load_swept(tmp_path / "nope.json") == frozenset()

    def test_a_malformed_inventory_is_refused(self, tmp_path):
        p = tmp_path / "swept.json"
        p.write_text('{"files": "mlsynth/x.py"}')      # a string, not a list
        with pytest.raises(ValueError, match="list of repository-relative"):
            case_deps.load_swept(p)

    def test_a_round_trip_through_the_writer(self, tmp_path):
        p = tmp_path / "swept.json"
        case_deps.save_swept(["mlsynth/b.py", "mlsynth/a.py", "mlsynth/a.py"],
                             cases=7, path=p)
        assert case_deps.load_swept(p) == frozenset({"mlsynth/a.py",
                                                     "mlsynth/b.py"})
        import json as _json
        body = _json.loads(p.read_text())
        assert body["files"] == ["mlsynth/a.py", "mlsynth/b.py"]  # sorted, unique
        assert body["cases"] == 7 and body["generated"]


class TestInventoryWalk:
    """``reachable_inventory`` mirrors the structural filters in ``_is_hole``.

    Anything it leaves out is already not a hole for a reason of its own, so
    recording it would add nothing; anything it includes is a path whose
    unclaimed status would otherwise be read as doubt.
    """

    def test_it_mirrors_the_structural_filters(self, tmp_path):
        for rel in ("mlsynth/utils/thing.py",
                    "mlsynth/__init__.py",
                    "mlsynth/tests/test_thing.py",
                    "mlsynth/utils/NOTES.rst",
                    "benchmarks/cases/x.py",
                    "benchmarks/tests/test_x.py",
                    "basedata/panel.csv",
                    "benchmarks/reference/.cache/pkg/src/a.py",
                    "docs/thing.rst"):
            f = tmp_path / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("x")
        got = set(case_deps.reachable_inventory(tmp_path))
        assert got == {"mlsynth/utils/thing.py", "benchmarks/cases/x.py",
                       "basedata/panel.csv"}, got

    def test_a_swept_inventory_closes_the_hole_it_records(self, tmp_path):
        (tmp_path / "mlsynth/utils").mkdir(parents=True)
        (tmp_path / "mlsynth/utils/lonely.py").write_text("x")
        inv = frozenset(case_deps.reachable_inventory(tmp_path))
        m = {"a": ["mlsynth/x.py"]}
        assert case_deps.select(["mlsynth/utils/lonely.py"], ["a"], m,
                                swept=frozenset()) == ["a"]
        assert case_deps.select(["mlsynth/utils/lonely.py"], ["a"], m,
                                swept=inv) == []


class TestInventoryIsOnlyWrittenForAFreshSweep:
    """An entry older than the tree cannot settle what nothing covers.

    A case with no entry is always selected, so a skipped or dead case is safe.
    A *stale* entry is not: a file added after that case was measured is absent
    from its entry whether or not the case executes it.
    """

    @staticmethod
    def _decide(from_slices, merged):
        from tools.benchmark_deps import should_record_inventory
        return should_record_inventory(from_slices, merged)

    def test_every_entry_measured_in_this_run_is_recorded(self):
        ok, why = self._decide({"a", "b"}, {"a": [], "b": []})
        assert ok and "measured in this run" in why

    def test_an_entry_carried_from_the_base_blocks_it(self):
        ok, why = self._decide({"a"}, {"a": [], "b": []})
        assert not ok and "b" in why and "predate" in why

    def test_slices_may_cover_more_than_the_merged_map(self):
        assert self._decide({"a", "b", "c"}, {"a": [], "b": []})[0]



# The invariant the inventory promises, over the input domain: it can only
# narrow. Recording what a sweep found can never add a case to the selection,
# because `swept` is consulted in exactly one place -- the hole test -- and a
# hole is the thing that widens.
from hypothesis import given, settings
from hypothesis import strategies as st

_PATHS = st.sampled_from([
    "mlsynth/x.py", "mlsynth/y.py", "mlsynth/utils/unclaimed.py",
    "mlsynth/utils/other.py", "mlsynth/tests/test_a.py", "docs/a.rst",
    "basedata/p.csv", "benchmarks/cases/a.py", "requirements.txt",
])


@given(changed=st.lists(_PATHS, max_size=6),
       swept=st.lists(_PATHS, max_size=6),
       added=st.lists(_PATHS, max_size=3))
@settings(max_examples=250, deadline=None)
def test_an_inventory_can_only_narrow_the_selection(changed, swept, added):
    m = {"a": ["mlsynth/x.py"], "b": ["mlsynth/y.py"]}
    cases = ["a", "b", "c"]
    wide = case_deps.select(changed, cases, m, added=added, swept=frozenset())
    narrow = case_deps.select(changed, cases, m, added=added,
                              swept=frozenset(swept))
    assert set(narrow) <= set(wide)


_INSTALLED = st.sampled_from(["benchmarks/constraints.txt", "pyproject.toml"])


@given(changed=st.lists(_PATHS, max_size=5), at=st.integers(0, 5),
       env=_INSTALLED, created=st.booleans(),
       added=st.lists(_PATHS, max_size=3),
       swept=st.lists(st.one_of(_PATHS, _INSTALLED), max_size=6),
       entries=st.dictionaries(st.sampled_from("abcd"),
                               st.lists(st.one_of(_PATHS, _INSTALLED), max_size=3),
                               max_size=4),
       cases=st.permutations(list("abcde")))
@settings(max_examples=250, deadline=None)
def test_a_change_to_what_the_install_reads_selects_every_case(
        changed, at, env, created, added, swept, entries, cases):
    """Whatever else the diff holds, whatever the map claims, whatever the
    inventory recorded and whether or not the diff creates it, a change to a
    file the install reads runs every candidate, in the candidates' order."""
    diff = changed[:at] + [env] + changed[at:]
    picked = case_deps.select(diff, cases, entries,
                              added=added + ([env] if created else []),
                              swept=frozenset(swept), imports={})
    assert picked == list(cases)


_FATES =st.sampled_from(["measured-0", "measured-1", "unmeasured-0",
                          "unmeasured-1", "not-attempted"])


@given(fates=st.dictionaries(st.sampled_from("abcdef"), _FATES, min_size=1),
       based=st.sets(st.sampled_from("abcdef")),
       files=st.lists(_PATHS, max_size=2))
@settings(max_examples=200, deadline=None)
def test_combining_keeps_measurements_drops_failures_and_carries_the_rest(
        tmp_path_factory, fates, based, files):
    """Over any split of cases into measured, attempted-but-unmeasured and
    not attempted, and any shipped map: a measured case takes its new entry, an
    unmeasured one has none (so it always runs), an unattempted one keeps its
    shipped entry, and the inventory is written exactly when no shipped entry
    was carried over unmeasured."""
    from tools import benchmark_deps

    tmp = tmp_path_factory.mktemp("slices")
    base = {c: ["mlsynth/old.py"] for c in based}
    bodies = [{}, {}]
    for c, fate in fates.items():
        kind, _, shard = fate.partition("-")
        if kind == "measured":
            bodies[int(shard)][c] = list(files)
        elif kind == "unmeasured":
            bodies[int(shard)].setdefault(benchmark_deps.UNMEASURED, []).append(c)
    paths = []
    for i, body in enumerate(bodies):
        p = tmp / f"s{i}.json"
        p.write_text(json.dumps(body))
        paths.append(p)
    merged = benchmark_deps.combine(paths, base=base)
    assert benchmark_deps.UNMEASURED not in merged
    for c in "abcdef":
        fate = fates.get(c, "not-attempted")
        if fate.startswith("measured"):
            assert merged[c] == list(files)
        elif fate.startswith("unmeasured"):
            assert c not in merged
        else:
            assert merged.get(c) == base.get(c)
    carried = {c for c in based if fates.get(c, "not-attempted") == "not-attempted"}
    ok, _ = benchmark_deps.should_record_inventory(
        benchmark_deps.measured_in(paths), merged)
    assert ok == (not carried)
