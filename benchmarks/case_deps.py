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
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

#: Prefixes a case can execute. A change outside all of them reaches no case, so
#: it is neither a selection nor a hole in the map -- docs and prose land here.
REACHABLE = ("mlsynth/", "basedata/", "benchmarks/")

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


def select(
    changed: Sequence[str],
    cases: Sequence[str],
    manifest: Mapping[str, Sequence[str]] | None = None,
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
    m = load() if manifest is None else manifest
    changed = [str(c) for c in changed]

    known = {p for deps in m.values() for p in deps}
    known.update(_case_source(name) for name in m)
    unmapped_change = [c for c in changed
                       if c.startswith(REACHABLE) and c not in known]
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
