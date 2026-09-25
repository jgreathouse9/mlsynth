"""On-demand fetch of Shen, Ding, Sekhon & Yu's ``deshen24/panel-data-regressions``.

The replication repository for *Same Root Different Leaves: Time Series and
Cross-Sectional Methods in Panel Data* (https://arxiv.org/abs/2207.14481). It
carries the three canonical case-study panels -- Basque terrorism, West German
reunification, California Proposition 99 -- already split into
``pre_outcomes.csv`` / ``post_outcomes.csv``, and the authors' own principal-
component-regression and rank-selection routines (``regr.pcr``,
``rank.spectral_rank``).

The ``snn_nesting`` case needs both halves: the authors' data with the authors'
pre/post split, and the authors' PCR as the reference estimator. The repository
carries no licence file, so this helper fetches it at a pinned commit into the
gitignored ``benchmarks/reference/.cache`` instead of vendoring it, mirroring
``clone_syntheticnn``. If git and codeload are both unreachable the benchmark
skips.

``regr.py`` imports ``toolz`` for its simplex regression. The PCR path does not
need it, so :func:`import_panel_data_regressions` installs a stub module when
``toolz`` is absent: the reference PCR runs unmodified, and only the convex
solver this case never calls is stubbed out.

The pinned commit (``_COMMIT``) freezes the reference; bump it deliberately.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Tuple

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference._fetch import fetch_pinned_repo

_REPO = "https://github.com/deshen24/panel-data-regressions.git"
_COMMIT = "51e2170d33463bbf403f23fe8a72cbf66bcc34ef"
_CACHE = Path(__file__).resolve().parent / ".cache" / "panel-data-regressions"

#: Treated unit per case study, as ``case_study.py`` names it.
TREATED = {
    "prop99": "California",
    "basque": "Basque",
    "germany": "West Germany",
}


def _install_toolz_stub() -> None:
    """Satisfy ``regr.py``'s import of ``toolz`` when the package is absent.

    ``regr.py`` uses ``toolz.reduce``/``toolz.partial`` only inside
    ``cvx_regr``, the simplex solver. This case calls ``pcr`` alone, so a stub
    exposing the two names from the standard library is enough to import the
    module, and nothing this case runs touches the stub. Idempotent, and a real
    ``toolz`` install takes precedence.
    """
    if "toolz" in sys.modules:
        return
    try:
        importlib.import_module("toolz")
        return
    except ImportError:
        pass
    import functools

    stub = ModuleType("toolz")
    stub.reduce = functools.reduce          # type: ignore[attr-defined]
    stub.partial = functools.partial        # type: ignore[attr-defined]
    stub._mlsynth_stub = True               # type: ignore[attr-defined]
    sys.modules["toolz"] = stub


def _ensure_clone() -> Path:
    """Fetch (or reuse) the reference repo pinned at ``_COMMIT``. Returns its path."""
    marker = _CACHE / "regr.py"
    if marker.exists():
        return _CACHE
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    fetch_pinned_repo(_REPO, _COMMIT, _CACHE)    # git clone, else codeload tarball
    if not marker.exists():  # pragma: no cover - defensive
        raise BenchmarkSkipped("reference clone missing regr.py module")
    return _CACHE


def import_panel_data_regressions() -> Tuple[ModuleType, ModuleType, Path]:
    """Import the authors' ``regr`` and ``rank`` modules from the pinned clone.

    Returns ``(regr, rank, data_dir)``. ``regr.pcr(X, y, max_rank=k)`` is the
    reference principal-component regression; ``rank.spectral_rank(s, t)`` is
    the reference rank rule; ``data_dir`` holds one subdirectory per case study.
    """
    path = _ensure_clone()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    try:
        _install_toolz_stub()
        regr = importlib.import_module("regr")
        rank = importlib.import_module("rank")
    except ImportError as exc:  # pragma: no cover - e.g. scipy missing
        raise BenchmarkSkipped(
            f"reference panel-data-regressions import failed ({exc})"
        ) from exc
    data_dir = path / "data"
    if not data_dir.is_dir():  # pragma: no cover - defensive
        raise BenchmarkSkipped("reference clone missing data/ directory")
    return regr, rank, data_dir
