"""A module no case executes still reaches the cases that import it.

The map records what each case's ``run()`` executes, measured with the package
already imported. That separation is what keeps a case from claiming the whole
library, and it has a blind side: code that runs only at import is never
attributed to anyone. A module of class definitions -- ``structures.py`` in
every estimator's helpers -- runs nothing after import, so no case claims it,
however many cases build its classes.

Before this rule such a module was either a hole, which runs the whole registry,
or, once a sweep inventory exists, a file "covered by nothing", which runs none
of the cases that use it. The first is the incident: #855 changed
``mlsynth/utils/marex_helpers/structures.py`` and its PR suite ran 243 of 243
cases. The second is the same defect pointed the dangerous way.

The import graph answers the question the map cannot. A change to a module's
import-time code reaches only modules that import it, so the cases claiming any
of its importers are the ones to run. The graph is read from the source at the
diff's head, so it is never older than the change.

Levels: smoke, unit, edge, failure, and properties over generated graphs.
"""
from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from benchmarks import case_deps, registry


def _tree(root: Path, files: dict) -> dict | None:
    for rel, text in files.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    return case_deps.import_graph(root)


# --------------------------------------------------------------------------- smoke
@pytest.fixture(scope="module")
def graph():
    return case_deps.import_graph()


class TestTheShippedTree:

    def test_it_parses(self, graph):
        assert graph is not None
        assert "mlsynth/utils/marex_helpers/structures.py" in graph

    def test_it_holds_no_test_module(self, graph):
        assert not [f for f in graph if f.startswith(case_deps.NEVER_REACHABLE)]

    def test_every_module_that_loads_cases_in_bulk_is_not_followed(self, graph):
        """A module with edges into more than one case module dispatches to
        them; the walk must stop there, as it does at the package root. Two
        modules reach one case each -- a case reusing another's helpers, a
        study reusing a case's panel -- and those are followed."""
        bulk = {src for src, deps in graph.items()
                if sum(d.startswith("benchmarks/cases/")
                       and not d.endswith("__init__.py") for d in deps) > 1}
        assert bulk <= set(case_deps.NOT_FOLLOWED), bulk - set(case_deps.NOT_FOLLOWED)

    def test_every_edge_lands_on_a_node(self, graph):
        stray = {d for deps in graph.values() for d in deps} - set(graph)
        assert not stray, sorted(stray)[:5]

    def test_the_incident_selects_the_marex_cases_and_not_the_registry(self):
        """#855's structures.py change, against the shipped map, with an
        inventory that records the file as covered by nothing.

        Every case that runs the MAREX estimator builds ``MAREXResults``, so
        each must be selected; the inventory must not silence them, and the
        registry must not run whole.
        """
        path = "mlsynth/utils/marex_helpers/structures.py"
        manifest = case_deps.load()
        cases = list(registry.CASES)
        picked = case_deps.select([path], cases, manifest,
                                  swept=frozenset({path}))
        # orchestration.py builds MAREXResults; every case that runs it is owed.
        marex = {c for c, deps in manifest.items()
                 if "mlsynth/utils/marex_helpers/orchestration.py" in deps}
        assert marex, "no mapped case builds MAREXResults; the check has no power"
        assert marex <= set(picked)
        assert len(picked) <= len(cases) // 4, len(picked)


# --------------------------------------------------------------- unit: the graph
class TestTheGraphReadsEverySpelling:
    """Each way of naming a module produces the same edge."""

    def test_an_absolute_import_names_its_module_and_not_its_parents(self, tmp_path):
        """Importing ``mlsynth.pkg.b`` runs the two ``__init__`` files above it,
        but the importer reads ``b``'s names. Counting what runs made one
        config import an importer of every structure its package re-exports."""
        g = _tree(tmp_path, {
            "mlsynth/__init__.py": "",
            "mlsynth/pkg/__init__.py": "",
            "mlsynth/pkg/b.py": "",
            "mlsynth/a.py": "import mlsynth.pkg.b\n",
        })
        assert g["mlsynth/a.py"] == {"mlsynth/pkg/b.py"}

    def test_a_name_from_a_module(self, tmp_path):
        g = _tree(tmp_path, {
            "mlsynth/pkg/m.py": "X = 1\n",
            "mlsynth/c.py": "from mlsynth.pkg.m import X\n",
        })
        assert "mlsynth/pkg/m.py" in g["mlsynth/c.py"]

    def test_a_submodule_from_its_package(self, tmp_path):
        g = _tree(tmp_path, {
            "mlsynth/pkg/__init__.py": "",
            "mlsynth/pkg/m.py": "",
            "mlsynth/c.py": "from mlsynth.pkg import m\n",
        })
        assert g["mlsynth/c.py"] == {"mlsynth/pkg/m.py"}

    def test_a_name_bound_in_a_package_init(self, tmp_path):
        """``from pkg import X`` reads ``pkg/__init__.py``, which is how a
        re-exported structure reaches the modules using it."""
        g = _tree(tmp_path, {
            "mlsynth/pkg/__init__.py": "from .s import X\n",
            "mlsynth/pkg/s.py": "X = 1\n",
            "mlsynth/c.py": "from mlsynth.pkg import X\n",
        })
        assert g["mlsynth/c.py"] == {"mlsynth/pkg/__init__.py"}
        assert case_deps.importers_of("mlsynth/pkg/s.py", g) == {
            "mlsynth/pkg/__init__.py", "mlsynth/c.py"}

    def test_relative_imports_at_every_level(self, tmp_path):
        g = _tree(tmp_path, {
            "mlsynth/top.py": "Y = 1\n",
            "mlsynth/pkg/__init__.py": "from .m import X\n",
            "mlsynth/pkg/m.py": "X = 1\n",
            "mlsynth/pkg/n.py": "",
            "mlsynth/pkg/r.py": ("from .m import X\n"
                                 "from . import n\n"
                                 "from ..top import Y\n"),
        })
        assert {"mlsynth/pkg/m.py", "mlsynth/pkg/n.py",
                "mlsynth/top.py"} <= g["mlsynth/pkg/r.py"]
        assert "mlsynth/pkg/m.py" in g["mlsynth/pkg/__init__.py"]

    def test_an_import_inside_a_function(self, tmp_path):
        """Cases import inside ``run()``; so do lazy helpers."""
        g = _tree(tmp_path, {
            "mlsynth/pkg/m.py": "",
            "benchmarks/cases/c.py": ("def run():\n"
                                      "    from mlsynth.pkg.m import X\n"),
        })
        assert "mlsynth/pkg/m.py" in g["benchmarks/cases/c.py"]

    def test_a_string_that_names_a_module(self, tmp_path):
        """``config_models`` resolves relocated configs from a table of module
        paths, and one helper calls ``__import__`` on a literal."""
        g = _tree(tmp_path, {
            "mlsynth/pkg/m.py": "",
            "mlsynth/table.py": '_TABLE = {"X": "mlsynth.pkg.m"}\n',
        })
        assert "mlsynth/pkg/m.py" in g["mlsynth/table.py"]

    def test_a_name_the_package_re_exports(self, tmp_path):
        """201 case imports are ``from mlsynth import X``. The package itself is
        not followed, so the name is resolved to the module that defines it."""
        g = _tree(tmp_path, {
            "mlsynth/__init__.py": ("from .estimators.est import EST\n"
                                    "__all__ = ['EST']\n"),
            "mlsynth/estimators/est.py": "class EST: ...\n",
            "benchmarks/cases/c.py": ("def run():\n"
                                      "    from mlsynth import EST\n"),
        })
        assert "mlsynth/estimators/est.py" in g["benchmarks/cases/c.py"]

    @pytest.mark.parametrize("src", [
        "import mlsynth\ndef run():\n    mlsynth.EST()\n",
        "import mlsynth as ms\ndef run():\n    ms.EST()\n",
    ])
    def test_a_re_exported_name_read_as_an_attribute(self, tmp_path, src):
        g = _tree(tmp_path, {
            "mlsynth/__init__.py": "from .estimators.est import EST\n",
            "mlsynth/estimators/est.py": "class EST: ...\n",
            "benchmarks/cases/c.py": src,
        })
        assert "mlsynth/estimators/est.py" in g["benchmarks/cases/c.py"]

    def test_a_sibling_imported_by_a_script(self, tmp_path):
        """A study script puts its own directory on ``sys.path`` and imports
        its neighbours by bare name."""
        g = _tree(tmp_path, {
            "benchmarks/studies/s/search.py": "",
            "benchmarks/studies/s/run.py": "import search\n",
        })
        assert "benchmarks/studies/s/search.py" in g["benchmarks/studies/s/run.py"]

    def test_a_third_party_import_is_not_an_edge(self, tmp_path):
        g = _tree(tmp_path, {"mlsynth/a.py": "import numpy\nfrom scipy import linalg\n"})
        assert g["mlsynth/a.py"] == frozenset()


# ------------------------------------------------------------------- edge cases
class TestWhatTheGraphHolds:

    def test_test_modules_are_not_nodes(self, tmp_path):
        g = _tree(tmp_path, {
            "mlsynth/a.py": "",
            "mlsynth/tests/test_a.py": "import mlsynth.a\n",
            "benchmarks/tests/test_b.py": "import mlsynth.a\n",
        })
        assert set(g) == {"mlsynth/a.py"}

    def test_a_module_does_not_import_itself(self, tmp_path):
        g = _tree(tmp_path, {"mlsynth/a.py": "import mlsynth.a\n"})
        assert g["mlsynth/a.py"] == frozenset()


class TestImportersOf:
    """The reverse closure, which is what a change to a module can reach."""

    def test_it_is_transitive(self):
        g = {"a": {"b"}, "b": {"c"}, "c": set()}
        assert case_deps.importers_of("c", g) == {"a", "b"}

    def test_it_does_not_pass_through_the_package_root(self):
        """``mlsynth/__init__.py`` imports every estimator and every module
        imports it as a parent package; following it would make every module an
        importer of every other."""
        g = {"mlsynth/__init__.py": {"mlsynth/c.py"},
             "mlsynth/x.py": {"mlsynth/__init__.py"},
             "mlsynth/c.py": set()}
        assert case_deps.importers_of("mlsynth/c.py", g) == frozenset()

    def test_it_passes_through_a_subpackage_init(self):
        """Importing ``pkg.d`` executes ``pkg/__init__.py``, which binds ``c``'s
        names, so ``y`` sees them."""
        g = {"mlsynth/pkg/__init__.py": {"mlsynth/pkg/c.py"},
             "mlsynth/y.py": {"mlsynth/pkg/__init__.py", "mlsynth/pkg/d.py"},
             "mlsynth/pkg/c.py": set(), "mlsynth/pkg/d.py": set()}
        assert case_deps.importers_of("mlsynth/pkg/c.py", g) == {
            "mlsynth/pkg/__init__.py", "mlsynth/y.py"}

    def test_it_does_not_pass_through_the_case_loader(self):
        """The registry names every case module, and every case runs through
        it, so it is claimed by every case. Following it made any module one
        case imports select the whole registry; each load touches one case."""
        g = {"benchmarks/registry.py": {"benchmarks/cases/a.py",
                                        "benchmarks/cases/b.py"},
             "benchmarks/run_benchmarks.py": {"benchmarks/registry.py"},
             "benchmarks/cases/a.py": {"mlsynth/m.py"},
             "benchmarks/cases/b.py": set(), "mlsynth/m.py": set()}
        assert case_deps.importers_of("mlsynth/m.py", g) == {"benchmarks/cases/a.py"}

    def test_a_cycle_terminates(self):
        g = {"a": {"b"}, "b": {"a"}}
        assert case_deps.importers_of("a", g) == {"b"}

    def test_a_module_nothing_imports(self):
        assert case_deps.importers_of("z", {"a": set()}) == frozenset()


# ------------------------------------------------- unit: the selection it drives
STRUCT = "mlsynth/h/structures.py"
EST = "mlsynth/est/marex.py"
MAP = {"marex": [EST], "other": ["mlsynth/y.py"]}
GRAPH = {EST: {STRUCT}, STRUCT: set(), "mlsynth/y.py": set(),
         "mlsynth/h/lonely.py": set()}
CASES = ["marex", "other"]


class TestImportClaims:

    def test_an_unclaimed_module_selects_the_cases_claiming_its_importers(self):
        assert case_deps.select([STRUCT], CASES, MAP, swept=frozenset(),
                                imports=GRAPH) == ["marex"]

    def test_the_inventory_cannot_silence_a_module_its_importers_use(self):
        """Rung 4 of the ladder. Swept and unclaimed is true of
        ``structures.py`` and says nothing about the classes it defines."""
        assert case_deps.select([STRUCT], CASES, MAP, swept=frozenset({STRUCT}),
                                imports=GRAPH) == ["marex"]

    def test_through_an_unclaimed_importer(self):
        g = {EST: {"mlsynth/h/mid.py"}, "mlsynth/h/mid.py": {STRUCT},
             STRUCT: set(), "mlsynth/y.py": set()}
        assert case_deps.select([STRUCT], CASES, MAP, swept=frozenset({STRUCT}),
                                imports=g) == ["marex"]

    def test_a_case_whose_own_module_imports_it(self):
        g = dict(GRAPH, **{"benchmarks/cases/other.py": {STRUCT}})
        assert case_deps.select([STRUCT], CASES, MAP, swept=frozenset(),
                                imports=g) == ["marex", "other"]

    def test_with_no_claimed_importer_the_inventory_decides(self):
        lonely = "mlsynth/h/lonely.py"
        assert case_deps.select([lonely], CASES, MAP, swept=frozenset({lonely}),
                                imports=GRAPH) == []
        assert case_deps.select([lonely], CASES, MAP, swept=frozenset(),
                                imports=GRAPH) == CASES

    def test_a_data_file_is_left_to_the_map_and_the_inventory(self):
        panel = "basedata/unread.csv"
        assert case_deps.select([panel], CASES, MAP, swept=frozenset({panel}),
                                imports=GRAPH) == []
        assert case_deps.select([panel], CASES, MAP, swept=frozenset(),
                                imports=GRAPH) == CASES

    def test_the_runner_claimed_by_every_case_does_not_widen(self):
        """Once the map measures the runner's own call, the registry and the
        runner are claimed by every case. A module one case imports must
        still select that case alone."""
        harness = ["benchmarks/registry.py", "benchmarks/run_benchmarks.py"]
        m = {"a": harness, "b": harness}
        g = {"benchmarks/registry.py": {"benchmarks/cases/a.py",
                                        "benchmarks/cases/b.py"},
             "benchmarks/run_benchmarks.py": {"benchmarks/registry.py"},
             "benchmarks/cases/a.py": {"mlsynth/m.py"},
             "benchmarks/cases/b.py": set(), "mlsynth/m.py": set()}
        assert case_deps.select(["mlsynth/m.py"], ["a", "b"], m,
                                swept=frozenset({"mlsynth/m.py"}),
                                imports=g) == ["a"]

    def test_a_claimed_module_selects_its_claimers_only(self):
        """The rule is for modules the map cannot claim. A claimed module's
        importers are not added: on the shipped tree that would add 124 cases
        to a change to ``accelerate.py``, and the cases that run a module are
        the ones it reaches at run time."""
        g = dict(GRAPH, **{"mlsynth/y.py": {EST}})
        assert case_deps.select([EST], CASES, MAP, swept=frozenset(),
                                imports=g) == ["marex"]


class TestThePremise:
    """The importer rule follows names, not execution. That is sound only while
    a library module's import-time code changes nothing outside the names it
    defines, so check the modules."""

    ALLOWED = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
               ast.ClassDef, ast.Assign, ast.AnnAssign)

    def _flags(self, rel, tree):
        out = []
        defined = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
        for node in tree.body:
            if isinstance(node, self.ALLOWED):
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue                                    # a docstring
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) \
                    and isinstance(node.value.func, ast.Attribute) \
                    and node.value.func.attr == "model_rebuild" \
                    and isinstance(node.value.func.value, ast.Name) \
                    and node.value.func.value.id in defined:
                continue                     # pydantic finishing its own class
            if isinstance(node, ast.If) and ast.unparse(node.test) in (
                    "TYPE_CHECKING", "__name__ == '__main__'"):
                continue
            if isinstance(node, ast.Try) and all(
                    isinstance(n, (*self.ALLOWED, ast.Pass))
                    for part in (node.body, node.orelse, node.finalbody,
                                 *[h.body for h in node.handlers])
                    for n in part):
                continue                                 # an optional import
            out.append(f"{rel}:{node.lineno} {ast.unparse(node).splitlines()[0]}")
        return out

    def test_library_modules_have_no_import_time_side_effects(self):
        root = Path(case_deps.__file__).resolve().parents[1]
        flags = []
        for f in sorted((root / "mlsynth").rglob("*.py")):
            rel = f.relative_to(root).as_posix()
            if rel.startswith(case_deps.NEVER_REACHABLE):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                flags += self._flags(rel, ast.parse(f.read_text()))
        assert not flags, flags

    def test_the_check_sees_a_side_effect(self):
        tree = ast.parse("import warnings\nwarnings.filterwarnings('ignore')\n")
        assert self._flags("mlsynth/x.py", tree)


# --------------------------------------------------------------------- failure
class TestAnUnreadableTree:

    def test_a_file_that_does_not_parse_leaves_the_graph_unknown(self, tmp_path):
        assert _tree(tmp_path, {"mlsynth/a.py": "",
                                "mlsynth/bad.py": "def (:\n"}) is None

    def test_an_unknown_graph_widens_on_an_unclaimed_module(self, monkeypatch):
        """Without the graph nothing can vouch for what the module's
        definitions reach, so the inventory does not get the last word."""
        monkeypatch.setattr(case_deps, "import_graph", lambda root=None: None)
        assert case_deps.select([STRUCT], CASES, MAP,
                                swept=frozenset({STRUCT})) == CASES

    def test_an_unknown_graph_still_lets_a_data_file_narrow(self, monkeypatch):
        monkeypatch.setattr(case_deps, "import_graph", lambda root=None: None)
        panel = "basedata/unread.csv"
        assert case_deps.select([panel], CASES, MAP,
                                swept=frozenset({panel})) == []

    @pytest.mark.parametrize("bad", ["mlsynth/a.py", 3])
    def test_imports_must_be_a_mapping(self, bad):
        with pytest.raises(TypeError, match="imports"):
            case_deps.select([STRUCT], CASES, MAP, imports=bad)


# ------------------------------------------------------------------ properties
_NODES = ["mlsynth/a.py", "mlsynth/b.py", "mlsynth/c.py", "mlsynth/d.py",
          "mlsynth/e.py", "benchmarks/cases/k.py", "basedata/p.csv"]
_PY = [n for n in _NODES if n.endswith(".py")]


@st.composite
def _worlds(draw):
    graph = {n: frozenset(draw(st.sets(st.sampled_from(_PY), max_size=3)))
             - {n} for n in _PY}
    manifest = {name: draw(st.lists(st.sampled_from(_NODES), max_size=3,
                                    unique=True))
                for name in ("j", "k", "l")}
    changed = draw(st.lists(st.sampled_from(_NODES), max_size=4, unique=True))
    added = draw(st.lists(st.sampled_from(changed), unique=True)) if changed else []
    swept = frozenset(draw(st.sets(st.sampled_from(_NODES))))
    return graph, manifest, changed, added, swept


@given(_worlds())
@settings(max_examples=300, deadline=None)
def test_no_case_that_claims_an_importer_is_silenced(world):
    """For every changed module the map cannot claim, every case claiming one
    of its importers runs, whatever the inventory says."""
    graph, manifest, changed, added, swept = world
    cases = ["j", "k", "l"]
    picked = set(case_deps.select(changed, cases, manifest, added=added,
                                  swept=swept, imports=graph))
    known = {p for deps in manifest.values() for p in deps}
    known |= {f"benchmarks/cases/{c}.py" for c in manifest}
    for c in changed:
        if c in known or c in added or not c.endswith(".py"):
            continue
        via = case_deps.importers_of(c, graph) & known
        owed = {n for n in cases
                if via & (set(manifest[n]) | {f"benchmarks/cases/{n}.py"})}
        assert owed <= picked, (c, via, owed, picked)


@given(_worlds())
@settings(max_examples=300, deadline=None)
def test_the_inventory_still_only_narrows(world):
    graph, manifest, changed, added, swept = world
    cases = ["j", "k", "l", "m"]
    wide = case_deps.select(changed, cases, manifest, added=added,
                            swept=frozenset(), imports=graph)
    narrow = case_deps.select(changed, cases, manifest, added=added,
                              swept=swept, imports=graph)
    assert set(narrow) <= set(wide)


@given(_worlds())
@settings(max_examples=300, deadline=None)
def test_import_claims_never_select_past_the_hole_rule(world):
    """What the graph adds is bounded by what treating every unclaimed file as
    a hole selected, so the rule cannot cost more time than it replaced."""
    graph, manifest, changed, added, swept = world
    cases = ["j", "k", "l", "m"]
    before = case_deps.select(changed, cases, manifest, added=added,
                              swept=frozenset(), imports={})
    after = case_deps.select(changed, cases, manifest, added=added,
                             swept=swept, imports=graph)
    assert set(after) <= set(before)


_LAYOUT = ["mlsynth/p/__init__.py", "mlsynth/p/q/__init__.py",
           "mlsynth/p/q/r/__init__.py"]
_HOMES = ["mlsynth/p/i.py", "mlsynth/p/q/i.py", "mlsynth/p/q/r/i.py",
          "mlsynth/p/q/__init__.py"]
_TARGETS = ["mlsynth/p/t.py", "mlsynth/p/q/t.py", "mlsynth/p/q/r/t.py"]


def _relative(home: str, target: str) -> str:
    here = home.split("/")[:-1]
    there = target[:-3].split("/")
    common = 0
    while common < min(len(here), len(there) - 1) and here[common] == there[common]:
        common += 1
    level = len(here) - common + 1
    rest = ".".join(there[common:-1])
    return f"from {'.' * level}{rest} import {there[-1]}\n"


@given(home=st.sampled_from(_HOMES), target=st.sampled_from(_TARGETS))
@settings(max_examples=60, deadline=None)
def test_a_relative_spelling_reaches_what_the_absolute_one_does(tmp_path_factory,
                                                                 home, target):
    absolute = f"from {target[:-3].rsplit('/', 1)[0].replace('/', '.')} " \
               f"import {target[:-3].rsplit('/', 1)[1]}\n"
    edges = []
    for src in (absolute, _relative(home, target)):
        root = tmp_path_factory.mktemp("tree")
        files = {p: "" for p in _LAYOUT + _TARGETS}
        files[home] = src
        edges.append(_tree(root, files)[home])
    assert edges[0] == edges[1]
    assert target in edges[0]
