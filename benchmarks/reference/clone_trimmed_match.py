"""On-demand clone of Google's ``trimmed_match`` reference repo.

The Trimmed Match Design package (https://github.com/google/trimmed_match,
Apache 2.0) pairs geos for a randomized paired geo experiment
(``TrimmedMatchGeoXDesign.create_geo_pairs``) and ships the example data its
design colab runs on. The ``pangeo_trimmed_match_pairs`` cross-validation
clones the repo at a pinned commit into a local cache and runs its pairing code
live in its own environment (``trimmed_match/install.sh``). If git or the
network is unavailable the benchmark skips.

The pinned commit (``_COMMIT``) freezes the reference; bump it deliberately,
never silently.
"""
from __future__ import annotations

from pathlib import Path

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference._fetch import fetch_pinned_repo

_REPO = "https://github.com/google/trimmed_match.git"
# trimmed_match master @ 2023-06-01.
_COMMIT = "0be855234ee3a589eaddc8b945e2a0d924ad8791"
_CACHE = Path(__file__).resolve().parent / ".cache" / "trimmed_match"


def ensure_clone() -> Path:
    """Clone (or reuse) the reference repo pinned at ``_COMMIT``. Returns its path."""
    design = _CACHE / "trimmed_match" / "design" / "trimmed_match_design.py"
    if design.exists():
        return _CACHE
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    try:
        fetch_pinned_repo(_REPO, _COMMIT, _CACHE)
    except Exception as exc:                       # network or git unavailable
        raise BenchmarkSkipped(f"could not fetch google/trimmed_match: {exc}") from exc
    if not design.exists():  # pragma: no cover - defensive
        raise BenchmarkSkipped("reference clone missing trimmed_match_design.py")
    return _CACHE
