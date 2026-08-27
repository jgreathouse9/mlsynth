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
selects every case. Both directions spend time; neither can silence a case. That
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
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

#: Prefixes a case can execute. A change outside all of them reaches no case, so
#: it is neither a selection nor a hole in the map -- docs and prose land here.
REACHABLE = ("mlsynth/", "basedata/", "benchmarks/")

#: Prefixes under :data:`REACHABLE` that no measurement can ever claim. The
#: registry loads only ``benchmarks.cases``, so no case's ``run()`` executes a
#: test module; the references to test files in case docstrings are prose.
NEVER_REACHABLE = ("mlsynth/tests/", "benchmarks/tests/")

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

PATH = Path(__file__).resolve().parent / "case_deps.json"


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


def _case_source(name: str) -> str:
    """The case's own module, which always selects it."""
    return f"benchmarks/cases/{name}.py"


def _is_hole(path: str, known: set, created: set) -> bool:
    """Whether ``path`` is a dependency the map failed to record.

    A hole widens the selection to the whole registry, so the question is not
    "did the map claim this?" but "could it have?". Three kinds of path answer
    no by construction and are not holes; everything else under
    :data:`REACHABLE` that no entry names is.
    """
    if not path.startswith(REACHABLE):
        return False
    if path.startswith(NEVER_REACHABLE) or path in RE_EXPORT_ONLY:
        return False
    if path.endswith(PROSE_SUFFIXES):
        return False
    if path in created:
        return False
    return path not in known


def select(
    changed: Sequence[str],
    cases: Sequence[str],
    manifest: Mapping[str, Sequence[str]] | None = None,
    *,
    added: Sequence[str] = (),
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

    Returns
    -------
    list of str
        A case is selected when it has no manifest entry, when its entry names a
        changed file, when its own module changed, or when some changed file
        under :data:`REACHABLE` appears in no entry at all -- that last being a
        hole in the map, which widens the selection instead of narrowing it.
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
    m = load() if manifest is None else manifest
    changed = [str(c) for c in changed]
    created = {str(a) for a in added}

    known = {p for deps in m.values() for p in deps}
    known.update(_case_source(name) for name in m)
    unmapped_change = [c for c in changed if _is_hole(c, known, created)]
    if unmapped_change:
        return list(cases)

    touched = set(changed)
    out = []
    for name in cases:
        deps = m.get(name)
        if deps is None:                       # never measured -> always run
            out.append(name)
        elif _case_source(name) in touched or touched.intersection(deps):
            out.append(name)
    return out
