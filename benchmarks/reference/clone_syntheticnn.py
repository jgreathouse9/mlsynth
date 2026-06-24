"""On-demand fetch of Dwivedi/Shah/Shen's ``deshen24/syntheticNN`` reference.

The canonical Synthetic Nearest Neighbors implementation
(https://github.com/deshen24/syntheticNN) is a single ``snn.py`` module. It
carries no licence file, so rather than vendoring it into the repository this
helper fetches it at a pinned commit into the gitignored
``benchmarks/reference/.cache`` and imports ``SyntheticNearestNeighbors`` from
there (mirroring ``clone_clustersc`` / ``clone_proximal``). If git or the
network is unavailable the benchmark skips gracefully.

The reference predates the ``check_array`` kwarg rename in scikit-learn 1.6
(``force_all_finite`` -> ``ensure_all_finite``). :func:`import_syntheticnn`
installs a thin translation shim so the upstream source runs unmodified on
current scikit-learn; the SNN algorithm itself is untouched.

The pinned commit (``_COMMIT``) freezes the reference; bump it deliberately.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference._fetch import fetch_pinned_repo

_REPO = "https://github.com/deshen24/syntheticNN.git"
_COMMIT = "a95b511385d93717d535bd0342d0d6150634ff30"
_CACHE = Path(__file__).resolve().parent / ".cache" / "syntheticNN"


def _install_check_array_shim() -> None:
    """Translate the legacy ``force_all_finite`` kwarg for scikit-learn >=1.6.

    deshen24/syntheticNN calls ``check_array(X, force_all_finite=False)``;
    scikit-learn 1.6 renamed that to ``ensure_all_finite``. We wrap the symbol
    in-place so the vendored algorithm needs no edit. Idempotent.
    """
    import sklearn.utils as sku

    orig = sku.check_array
    if getattr(orig, "_mlsynth_shimmed", False):
        return

    def check_array(*args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return orig(*args, **kwargs)

    check_array._mlsynth_shimmed = True  # type: ignore[attr-defined]
    sku.check_array = check_array


def _ensure_clone() -> Path:
    """Fetch (or reuse) the reference repo pinned at ``_COMMIT``. Returns its path."""
    marker = _CACHE / "snn.py"
    if marker.exists():
        return _CACHE
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    fetch_pinned_repo(_REPO, _COMMIT, _CACHE)    # git clone, else codeload tarball
    if not marker.exists():  # pragma: no cover - defensive
        raise BenchmarkSkipped("reference clone missing snn.py module")
    return _CACHE


def import_syntheticnn() -> ModuleType:
    """Import the authors' ``snn`` module from the pinned clone.

    Returns the module; ``snn.SyntheticNearestNeighbors`` is the estimator.
    """
    path = _ensure_clone()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    try:
        _install_check_array_shim()
        snn = importlib.import_module("snn")          # needs networkx, sklearn
        # snn.py did ``from sklearn.utils import check_array`` -- patch the
        # name it bound directly, too.
        import sklearn.utils as sku
        snn.check_array = sku.check_array
    except ImportError as exc:  # pragma: no cover - e.g. networkx missing
        raise BenchmarkSkipped(
            f"reference syntheticNN import failed ({exc}); "
            f"install its deps (`pip install networkx scikit-learn`)"
        ) from exc
    return snn
