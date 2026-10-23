"""The authors' code, fetched from GitHub at pinned commits and imported from a cache directory.

Nothing of theirs is vendored. ``fetch`` clones the two repositories (MIT licensed)
into ``MLSYNTH_AS_EXACT_CACHE`` (default ``~/.cache/mlsynth/as_exact``) and checks
out the commits below, the current heads when this study was run:

  synth-AS-replication  791a3865fe3e9fd0b82a8121c0b338791b7ffa39  (v1.0.0, 2026-10-19)
  synthx_package        a10a28c571a906ee76574bad9c5446c22c3a184f  (v0.9.1, 2026-10-20)

``latest=True`` uses the default branch heads instead.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

REPOS = {
    "synth-AS-replication": "791a3865fe3e9fd0b82a8121c0b338791b7ffa39",
    "synthx_package": "a10a28c571a906ee76574bad9c5446c22c3a184f",
}
URL = "https://github.com/lychemon/{}"


def cache_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("MLSYNTH_AS_EXACT_CACHE",
                                       pathlib.Path.home() / ".cache" / "mlsynth" / "as_exact"))


def fetch(latest: bool = False) -> dict:
    """Clone (or reuse) both repositories; return {name: (path, commit)}."""
    root = cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, sha in REPOS.items():
        path = root / name
        if not (path / ".git").exists():
            subprocess.run(["git", "clone", "--quiet", URL.format(name), str(path)], check=True)
        else:
            subprocess.run(["git", "-C", str(path), "fetch", "--quiet", "origin"], check=True)
        target = "origin/HEAD" if latest else sha
        subprocess.run(["git", "-C", str(path), "checkout", "--quiet", "--detach", target], check=True)
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
        out[name] = (path, head)
    return out


def load(paths: dict):
    """Import the replication's solver modules and the synthx package from fetched paths."""
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
