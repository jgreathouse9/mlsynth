"""Which benchmark cases a change can actually reach.

``pr-suite`` runs the whole registry on every pull request, and the wall clock is
set by a handful of Monte Carlo and MIP cases that most changes cannot touch.
This module holds the map from a case to the files its ``run()`` executes, and
the rule for selecting cases from a diff.

Why the map is measured, not written
-----------------------------------
Two cheaper approaches fail on this repository. Cases import inside their
functions, so a static scan of imports sees entry points and never reach --
``from mlsynth import PPSCM`` says nothing about PPSCM using
``conformal/resample.py``. And tracing imports at runtime is worse than useless:
``mlsynth/__init__.py`` imports every estimator, so a case that executes one file
has an import closure of 834.

So the manifest records what a case's ``run()`` *executes*, measured with
coverage while the package is already imported, which separates running a module
from loading it. Measured that way ``conformal_window_count`` executes exactly
``mlsynth/utils/conformal/resample.py``.

Regenerate with ``python tools/benchmark_deps.py --all``. The daily workflow does
it and commits the result, so the map is at most a day behind the code.

Why the selection rule is lopsided
----------------------------------
A case with no entry always runs, and a changed file that no entry mentions
selects every case unless one of the rules below settles it. Both directions
spend time; neither can silence a case. That
asymmetry is the point: a skipped check reports success, and a skipped success is
indistinguishable from a real one, so the map is allowed to cost time and is not
allowed to hide anything.

Three paths the map can never claim
-----------------------------------
Widening on an unclaimed file is right when the map might have claimed it and
did not. Three kinds of path cannot be claimed however often the map is
refreshed, so treating them as holes widened every selection to the whole
registry and the rule never narrowed anything.

The registry loads only ``benchmarks.cases``, so no case's ``run()`` executes a
test module -- :data:`NEVER_REACHABLE`. A file the diff creates did not exist
when any pre-existing case ran -- the ``added`` argument. ``__init__.py``
executes at import, before measurement starts, which is the separation that
stops a case from measuring all 834 library files -- :data:`RE_EXPORT_ONLY`,
whose one member is checked against the file itself in the tests, since the
claim holds only while that file is imports and an ``__all__``. And a README is
prose, which the rule already answers for by directory: ``docs/ppscm.rst`` is
outside :data:`REACHABLE` and so reaches nothing. :data:`PROSE_SUFFIXES` says
the same thing by kind, for the prose that sits under a reachable prefix.

An estimator pull request trips all three at once: new helper modules, new test
modules, and one export line.

Modules that run only at import
-------------------------------
Measuring after import has a blind side. A module whose code runs only at import
-- class definitions, a table of constants, ``structures.py`` in every
estimator's helpers -- executes nothing during any ``run()``, so no case claims
it however many cases build its classes. Read as a hole it ran the whole
registry: #855 changed ``marex_helpers/structures.py`` and its suite ran 243 of
243 cases. Read through the sweep inventory it would run none of the cases that
use it, which is the same defect pointed the dangerous way.

A change to a module's import-time code reaches only the modules that import
it, so for a module no entry claims, :func:`select` runs the cases that claim
any of its importers (:func:`importers_of`, over :func:`import_graph`). The
graph is read from the source at the diff's head, so it is never older than the
change, and it settles the question before the inventory is consulted. It is
not used for claimed modules: the cases that run a module are the ones it
reaches at run time, and adding importers there would add 124 cases to a change
to ``accelerate.py``.
"""
from __future__ import annotations

import ast
import json
import warnings
from functools import lru_cache
from pathlib import Path
from typing import AbstractSet, Dict, FrozenSet, Iterable, List, Mapping, Sequence

#: Prefixes a case can execute. A change outside all of them reaches no case, so
#: it is neither a selection nor a hole in the map -- docs and prose land here.
REACHABLE = ("mlsynth/", "basedata/", "benchmarks/")

#: Prefixes under :data:`REACHABLE` that no measurement can ever claim. The
#: registry loads only ``benchmarks.cases``, so no case's ``run()`` executes a
#: test module; the references to test files in case docstrings are prose.
NEVER_REACHABLE = ("mlsynth/tests/", "benchmarks/tests/")

#: Prefixes under :data:`REACHABLE` that git does not track. The reference
#: cache holds what the references download: a case reads it and the
#: measurement sees the reads, but no diff can name it, and it exists only
#: where the references were fetched.
UNTRACKED = ("benchmarks/reference/.cache/",)

#: Suffixes that are prose. A change outside :data:`REACHABLE` already reaches no
#: case -- that is how ``docs/ppscm.rst`` is neither a selection nor a hole. A
#: README one directory over is the same prose, so it gets the same answer.
PROSE_SUFFIXES = (".md", ".rst")

#: Modules that re-export and do nothing else. These execute at import, before
#: measurement starts, so coverage never attributes them to a case -- and a file
#: of imports and an ``__all__`` can add or remove a name but cannot change what
#: any estimator computes. The claim is checked against the file itself in
#: ``benchmarks/tests/test_case_deps.py``; break it and the entry comes out.
RE_EXPORT_ONLY = ("mlsynth/__init__.py",)

#: Modules :func:`importers_of` does not walk through. Each reaches every case
#: and each use of it touches one: the package root imports every estimator
#: and nearly every case imports a name from it, and the registry names every
#: case module and every case is loaded through it. Following either makes a
#: module one case uses an importer of all of them. A test checks the shipped
#: tree for any other module that loads more than one case.
NOT_FOLLOWED = RE_EXPORT_ONLY + ("benchmarks/registry.py",)

#: The trees whose Python modules :func:`import_graph` reads. Test modules are
#: left out by :data:`NEVER_REACHABLE`, as everywhere else.
IMPORT_ROOTS = ("mlsynth/", "benchmarks/")

PATH = Path(__file__).resolve().parent / "case_deps.json"

#: Where a complete sweep records the reachable files that existed when it ran.
#: The manifest holds positive claims only, so an unclaimed file under
#: :data:`REACHABLE` is ambiguous: either nothing measured it, or everything did
#: and no case executes it. The first is doubt and widens the selection; the
#: second is an answer. This file is what separates them, and it is written only
#: by a sweep that produced an entry for every registered case -- a case that
#: raised contributes no entry, and its files would otherwise read as uncovered
#: when they were merely unmeasured.
SWEPT_PATH = Path(__file__).resolve().parent / "case_deps_swept.json"


def load(path: Path | None = None) -> Dict[str, List[str]]:
    """The manifest as shipped, or an empty map when it has not been built."""
    p = PATH if path is None else Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def save(manifest: Mapping[str, Iterable[str]], path: Path | None = None) -> None:
    """Write the manifest sorted and deduplicated, so it diffs as content."""
    p = PATH if path is None else Path(path)
    tidy = {k: sorted(set(v)) for k, v in sorted(manifest.items())}
    p.write_text(json.dumps(tidy, indent=1) + "\n")


def load_swept(path: Path | None = None) -> frozenset:
    """The files a complete sweep measured, or an empty set when none has run.

    An empty set is the conservative answer and reproduces the behaviour from
    before the inventory existed: every unclaimed reachable file is a hole.
    """
    p = SWEPT_PATH if path is None else Path(path)
    if not p.exists():
        return frozenset()
    body = json.loads(p.read_text())
    files = body.get("files") if isinstance(body, dict) else body
    if not isinstance(files, list) or any(not isinstance(f, str) for f in files):
        raise ValueError(
            f"{p} must hold a list of repository-relative paths under "
            f'"files"; got {type(files).__name__}. Regenerate it with '
            f"`python tools/benchmark_deps.py --combine <slices>` over a sweep "
            f"that covered every registered case."
        )
    return frozenset(files)


def save_swept(files: Iterable[str], *, cases: int,
               path: Path | None = None) -> None:
    """Record the sweep's inventory, sorted and deduplicated.

    ``cases`` is how many cases the sweep produced an entry for, kept so a
    reader can tell the inventory came from a complete run.
    """
    import datetime as _dt

    p = SWEPT_PATH if path is None else Path(path)
    body = {
        "generated": _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "cases": int(cases),
        "files": sorted(set(files)),
    }
    p.write_text(json.dumps(body, indent=1) + "\n")


def reachable_inventory(root: Path | None = None) -> List[str]:
    """Every file a case could have claimed, as a sweep would see the tree.

    The filters mirror :func:`_is_hole`'s structural ones, so the inventory
    holds exactly the paths whose unclaimed status is otherwise ambiguous.
    Anything this leaves out is not a hole for a reason of its own and needs no
    record.
    """
    base = Path(root) if root is not None else PATH.parent.parent
    out = []
    for prefix in REACHABLE:
        for f in sorted((base / prefix).rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(base).as_posix()
            if "__pycache__" in rel:
                continue
            if rel.startswith(NEVER_REACHABLE + UNTRACKED) or rel in RE_EXPORT_ONLY:
                continue
            if rel.endswith(PROSE_SUFFIXES):
                continue
            out.append(rel)
    return out


def _case_source(name: str) -> str:
    """The case's own module, which always selects it."""
    return f"benchmarks/cases/{name}.py"


def _module_name(rel: str) -> str:
    """``mlsynth/pkg/m.py`` -> ``mlsynth.pkg.m``; a package's ``__init__`` is
    the package."""
    parts = rel[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _package_of(rel: str) -> str:
    """The package a module's relative imports start from."""
    name = _module_name(rel)
    return name if rel.endswith("/__init__.py") else name.rpartition(".")[0]


def _from_base(rel: str, node: ast.ImportFrom) -> str:
    """The module an ``ImportFrom`` names, with its dots resolved."""
    if not node.level:
        return node.module or ""
    parts = _package_of(rel).split(".")
    parts = parts[: len(parts) - (node.level - 1)]
    return ".".join(parts + ([node.module] if node.module else []))


def _resolve(rel: str, dotted: str, by_name: Mapping[str, str]) -> str | None:
    """``dotted`` as an absolute name, or failing that as a script's sibling.

    A study script puts its own directory on ``sys.path`` and imports its
    neighbours by bare name; reading such a name as a sibling can only add
    edges, which can only add cases.
    """
    if dotted in by_name:
        return dotted
    here = rel.rpartition("/")[0].replace("/", ".")
    sibling = f"{here}.{dotted}"
    return sibling if dotted and sibling in by_name else None


def _exports(rel: str, tree: ast.Module, by_name: Mapping[str, str]) -> Dict[str, set]:
    """Each name a re-export module binds, mapped to the file defining it."""
    out: Dict[str, set] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        base = _from_base(rel, node)
        for alias in node.names:
            sub = f"{base}.{alias.name}"
            name = sub if sub in by_name else base
            if name in by_name and by_name[name] not in RE_EXPORT_ONLY:
                out.setdefault(alias.asname or alias.name, set()).add(by_name[name])
    return out


def _edges(rel: str, tree: ast.Module, by_name: Mapping[str, str],
           exports: Mapping[str, Mapping[str, set]]) -> set:
    """The files whose names module ``rel`` reads, however it spells the import.

    An edge is a module whose definitions ``rel`` can see, not every module
    importing it executes. ``import a.b.c`` also runs ``a/__init__.py`` and
    ``a/b/__init__.py``, but ``rel`` reads ``c``'s names, and a package's
    ``__init__`` reaches it only through a name bound there -- ``from a.b import
    X`` -- which is an edge of its own. Counting execution instead made a module
    that imports one config an importer of every structure its package
    re-exports; on the shipped tree a change to ``marex_helpers/structures.py``
    then selected 230 of 243 cases. Execution matters only through import-time
    side effects, and ``test_case_deps_imports`` pins that library modules have
    none.
    """
    out: set = set()
    bound: Dict[str, str] = {}       # a local name -> the re-export module it is
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found = _resolve(rel, alias.name, by_name)
                if found:
                    out.add(by_name[found])
                top = alias.name if alias.asname else alias.name.split(".")[0]
                if by_name.get(top) in exports:
                    bound[alias.asname or top] = by_name[top]
        elif isinstance(node, ast.ImportFrom):
            base = _from_base(rel, node)
            if not node.level:
                base = _resolve(rel, base, by_name) or base
            names = exports.get(by_name.get(base, ""), {})
            for alias in node.names:
                sub = f"{base}.{alias.name}"
                if sub in by_name:                  # a submodule
                    out.add(by_name[sub])
                elif base in by_name:               # a name bound in ``base``
                    out.add(by_name[base])
                if alias.name == "*":
                    out.update(*names.values())
                out |= names.get(alias.name, set())
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value in by_name:
            out.add(by_name[node.value])
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in bound:
            out |= exports[bound[node.value.id]].get(node.attr, set())
    out.discard(rel)
    return out


def import_graph(root: Path | None = None) -> Dict[str, FrozenSet[str]] | None:
    """Which reachable modules each module imports, read from the source.

    Keys are every Python file under :data:`IMPORT_ROOTS` outside
    :data:`NEVER_REACHABLE`; each value is the set of those files whose names
    the module imports, at any depth of its body. Absolute, relative and
    in-function imports count, so do a string literal that names a module (the
    relocated-config table, ``__import__``) and a script's bare-name sibling. A
    name imported from a :data:`RE_EXPORT_ONLY` module is resolved to the module
    that defines it, since that module is never followed.

    Returns ``None`` when any file fails to parse: a graph with a module
    missing would read that module's imports as none.
    """
    if root is None:
        return _shipped_graph()
    base = Path(root)
    files = sorted(
        f.relative_to(base).as_posix()
        for prefix in IMPORT_ROOTS
        for f in (base / prefix).rglob("*.py")
        if f.is_file() and "__pycache__" not in f.parts
        and not f.relative_to(base).as_posix().startswith(NEVER_REACHABLE))
    by_name = {_module_name(f): f for f in files}
    trees = {}
    for rel in files:
        try:
            with warnings.catch_warnings():    # a docstring's "\m" is not ours
                warnings.simplefilter("ignore", SyntaxWarning)
                trees[rel] = ast.parse((base / rel).read_text(encoding="utf-8"),
                                       filename=rel)
        except (SyntaxError, UnicodeDecodeError, ValueError):
            return None
    exports = {rel: _exports(rel, trees[rel], by_name)
               for rel in RE_EXPORT_ONLY if rel in trees}
    return {rel: frozenset(_edges(rel, tree, by_name, exports))
            for rel, tree in trees.items()}


@lru_cache(maxsize=1)
def _shipped_graph() -> Dict[str, FrozenSet[str]] | None:
    """The graph of the checked-out tree, read once per process."""
    return import_graph(PATH.parent.parent)


def importers_of(path: str,
                 graph: Mapping[str, AbstractSet[str]]) -> FrozenSet[str]:
    """Every module that imports ``path``, directly or through others.

    The walk does not pass through a :data:`NOT_FOLLOWED` module.
    ``mlsynth/__init__.py`` imports every estimator and nearly every case
    imports a name from it, so following it would make every case an importer
    of every module; :func:`import_graph` resolves the names it re-exports
    instead. The registry is the same shape from the other side: it names every
    case module, and every case runs through it.
    """
    reverse: Dict[str, set] = {}
    for src, deps in graph.items():
        for dep in deps:
            reverse.setdefault(dep, set()).add(src)
    seen: set = set()
    todo = [path]
    while todo:
        for src in reverse.get(todo.pop(), ()):
            if src in seen or src in NOT_FOLLOWED:
                continue
            seen.add(src)
            todo.append(src)
    seen.discard(path)
    return frozenset(seen)


def _is_hole(path: str, known: set, created: set,
             swept: frozenset = frozenset()) -> bool:
    """Whether ``path`` is a dependency the map failed to record.

    A hole widens the selection to the whole registry, so the question is not
    "did the map claim this?" but "could it have?". Four kinds of path answer no
    by construction and are not holes; everything else under :data:`REACHABLE`
    that no entry names is.

    The fourth is ``swept``: a file a complete sweep measured and no case
    executed is covered by nothing, which is an answer and not a gap. Without
    it, library code no benchmark reaches widened every selection that touched
    it -- the common case, since the manifest claims 673 files and the reachable
    tree holds far more.
    """
    if not path.startswith(REACHABLE):
        return False
    if path.startswith(NEVER_REACHABLE) or path in RE_EXPORT_ONLY:
        return False
    if path.endswith(PROSE_SUFFIXES):
        return False
    if path in created:
        return False
    if path in swept:
        return False
    return path not in known


def select(
    changed: Sequence[str],
    cases: Sequence[str],
    manifest: Mapping[str, Sequence[str]] | None = None,
    *,
    added: Sequence[str] = (),
    swept: frozenset | None = None,
    imports: Mapping[str, AbstractSet[str]] | None = None,
) -> List[str]:
    """The cases a diff can reach, in the order ``cases`` gives them.

    Parameters
    ----------
    changed : sequence of str
        Repository-relative paths the diff touches.
    cases : sequence of str
        Candidate case names, already filtered for references and sharding.
    manifest : mapping, optional
        Case to executed files. Defaults to the shipped manifest.
    added : sequence of str, optional
        The subset of ``changed`` the diff creates. A file that did not exist
        cannot have been executed by a pre-existing case, so its absence from
        the map is not a hole. It still selects on a positive match, which is
        what runs a newly added case.
    swept : frozenset of str, optional
        The files a complete sweep measured, from :func:`load_swept`. One of
        these that no entry claims is covered by nothing, so it is not a hole.
        Pass ``frozenset()`` to treat every unclaimed file as doubt.
    imports : mapping, optional
        Module to the modules it imports, from :func:`import_graph`. Defaults to
        the graph of the checked-out tree. A changed module no entry claims
        selects the cases that claim any of its importers, before the inventory
        is consulted. When the tree does not parse the graph is unknown, and such
        a module widens the selection.

    Returns
    -------
    list of str
        A case is selected when it has no manifest entry, when its entry names a
        changed file, when its own module changed, when its entry or its module
        imports a changed module no entry claims, or when some changed file
        under :data:`REACHABLE` is unclaimed with nothing to settle it -- that
        last being a hole in the map, which widens the selection instead of
        narrowing it.
    """
    if isinstance(changed, (str, bytes)) or not isinstance(changed, Sequence):
        raise TypeError(
            f"changed must be a sequence of repository-relative paths, not "
            f"{type(changed).__name__}. A single path must be wrapped in a list, "
            f"since a bare string would iterate character by character."
        )
    if isinstance(added, (str, bytes)) or not isinstance(added, Sequence):
        raise TypeError(
            f"added must be a sequence of repository-relative paths, not "
            f"{type(added).__name__}. It names the subset of `changed` the diff "
            f"creates; pass () when the diff adds nothing."
        )
    if imports is not None and not isinstance(imports, Mapping):
        raise TypeError(
            f"imports must map each module to the modules it imports, as "
            f"import_graph() returns, not {type(imports).__name__}. Omit it to "
            f"read the checked-out tree."
        )
    m = load() if manifest is None else manifest
    seen = load_swept() if swept is None else frozenset(swept)
    changed = [str(c) for c in changed]
    created = {str(a) for a in added}

    known = {p for deps in m.values() for p in deps}
    known.update(_case_source(name) for name in m)
    graph = import_graph() if imports is None else imports
    via: set = set()
    for c in changed:
        if not _is_hole(c, known, created):    # claimed, created or structural
            continue
        if c.endswith(".py"):
            if graph is None:                  # nothing can vouch for its reach
                return list(cases)
            claimed = importers_of(c, graph) & known
            if claimed:
                via |= claimed
                continue
        if _is_hole(c, known, created, seen):
            return list(cases)

    touched = set(changed) | via
    out = []
    for name in cases:
        deps = m.get(name)
        if deps is None:                       # never measured -> always run
            out.append(name)
        elif _case_source(name) in touched or touched.intersection(deps):
            out.append(name)
    return out
