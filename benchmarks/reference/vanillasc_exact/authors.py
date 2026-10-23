"""The authors' code at the pinned commits, fetched on demand and imported from a cache.

``fetch`` places lychemon/synth-AS-replication and lychemon/synthx_package (MIT) in the
gitignored ``benchmarks/reference/.cache`` at the commits ``reference.py`` pins, and
``load`` imports the replication's solver modules and the synthx package from there.
Nothing of theirs is vendored. The design capture (``capture_designs.py``) and the live
test in ``mlsynth/tests/test_exact_search.py`` use them.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.reference._fetch import fetch_pinned_repo  # noqa: E402

REPOS = {
    "synth-AS-replication": ("https://github.com/lychemon/synth-AS-replication.git",
                             "791a3865fe3e9fd0b82a8121c0b338791b7ffa39"),
    "synthx_package": ("https://github.com/lychemon/synthx_package.git",
                       "a10a28c571a906ee76574bad9c5446c22c3a184f"),
}
CACHE = ROOT / "benchmarks" / "reference" / ".cache"


def fetch() -> dict:
    """Both repositories at the pinned commits: {name: (path, commit)}."""
    out = {}
    for name, (url, sha) in REPOS.items():
        path = CACHE / f"lychemon-{name}-{sha[:12]}"
        if not (path / "README.md").exists():
            CACHE.mkdir(parents=True, exist_ok=True)
            fetch_pinned_repo(url, sha, path)
        out[name] = (path, sha)
    return out


def load(paths: dict) -> dict:
    """The replication's data preparation, solvers and predictor weights, and the synthx package."""
    support = str(paths["synth-AS-replication"][0] / "code" / "support")
    python = str(paths["synthx_package"][0] / "code" / "python")
    for p in (support, python):
        if p not in sys.path:
            sys.path.insert(0, p)
    import data_preparation                       # noqa: E402  (theirs)
    import predictor_weights                      # noqa: E402  (theirs)
    import visible_facet_solver                   # noqa: E402  (theirs)
    import synthx                                 # noqa: E402  (theirs)
    return dict(data_preparation=data_preparation, predictor_weights=predictor_weights,
                visible_facet_solver=visible_facet_solver, synthx=synthx)
