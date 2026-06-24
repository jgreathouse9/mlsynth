"""On-demand fetch of Amjad/Shah/Shin's ``jehangiramjad/tslib`` reference.

The original Robust Synthetic Control implementation
(https://github.com/jehangiramjad/tslib) -- the ``RobustSyntheticControl`` of
Amjad, Shah & Shin, *Robust Synthetic Control*, JMLR 19(22):1-51, 2018 -- ships
without a licence file, so rather than vendoring it into the repository this
helper fetches the library at a pinned commit into the gitignored
``benchmarks/reference/.cache`` and imports the genuine algorithm from there
(mirroring ``clone_syntheticnn`` / ``clone_proximal``). If git or the network is
unavailable the benchmark skips gracefully.

The package's internal imports are absolute (``from tslib.src...``), so the
fetched tree is placed at ``.cache/tslib`` and ``.cache`` is put on ``sys.path``;
``import tslib.src...`` then resolves to the pinned clone.

Compatibility note: ``tslib`` targets Python 2.7 / early 3, and
``src/synthcontrol/syntheticControl.py`` mixes tabs and spaces in its
``predict`` / ``getControl`` methods, which raises ``TabError`` on import under
Python 3. Robust Synthetic Control's *fit* path does not touch those methods --
``RobustSyntheticControl.fit`` delegates straight to
``tslib.src.models.tsSVDModel.SVDModel.fit`` (N=1, ``includePastDataOnly=False``,
SVD/numpy) -- so :func:`import_rsc` imports the genuine ``SVDModel`` (which is
clean) and :func:`build_rsc_weights` constructs and fits it exactly as the
``RobustSyntheticControl`` constructor does. The algorithm itself is the
upstream source, unmodified.

The pinned commit (``_COMMIT``) freezes the reference; bump it deliberately.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference._fetch import fetch_pinned_repo

_REPO = "https://github.com/jehangiramjad/tslib.git"
_COMMIT = "3e50bc1fbe0178bf2c1b21b2ce9fd0f0ca2d5f76"  # master @ 2021-12-14
_CACHE = Path(__file__).resolve().parent / ".cache" / "tslib"


def _ensure_clone() -> Path:
    """Fetch (or reuse) the tslib tree pinned at ``_COMMIT``; return its parent.

    The clone lands at ``.cache/tslib`` so that ``import tslib`` resolves there.
    Returns the directory to add to ``sys.path`` (the parent of ``tslib``).
    """
    marker = _CACHE / "src" / "models" / "tsSVDModel.py"
    if not marker.exists():
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        fetch_pinned_repo(_REPO, _COMMIT, _CACHE)    # git clone, else codeload tarball
        if not marker.exists():  # pragma: no cover - defensive
            raise BenchmarkSkipped("tslib clone missing src/models/tsSVDModel.py")
    return _CACHE.parent


def import_rsc() -> ModuleType:
    """Import the genuine ``tslib`` SVD model used by ``RobustSyntheticControl``.

    Returns the ``tslib.src.models.tsSVDModel`` module; its ``SVDModel`` is the
    engine ``RobustSyntheticControl`` fits.
    """
    parent = _ensure_clone()
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    try:
        return importlib.import_module("tslib.src.models.tsSVDModel")
    except ImportError as exc:  # pragma: no cover - missing numpy/pandas/sklearn
        raise BenchmarkSkipped(
            f"reference tslib import failed ({exc}); "
            f"install its deps (`pip install numpy pandas scikit-learn`)"
        ) from exc


def build_rsc_weights(
    target_pre: np.ndarray,
    donor_pre: np.ndarray,
    donor_names,
    k: int,
) -> np.ndarray:
    """Fit genuine tslib Robust Synthetic Control and return its donor weights.

    Reproduces ``RobustSyntheticControl(seriesToPredictKey, k, M=T0, N=1,
    modelType='svd', svdMethod='numpy', otherSeriesKeysArray=donors).fit(df)``
    by constructing the same ``SVDModel`` (the ``.fit`` path the public class
    delegates to) and reading ``model.weights``. The OG RSC: hard
    singular-value thresholding of the stacked donor+treated pre-matrix at rank
    ``k``, then a pseudo-inverse (OLS) of the de-noised treated row on the
    de-noised donor rows -- unconstrained weights, no intercept.

    Parameters
    ----------
    target_pre : np.ndarray
        Treated unit's pre-period outcomes, shape ``(T0,)``.
    donor_pre : np.ndarray
        Donor pre-period outcomes, columns = donors, shape ``(T0, J)``.
    donor_names : sequence of str
        Length-``J`` donor labels (column order of ``donor_pre``).
    k : int
        Number of singular values to retain (``kSingularValuesToKeep``).

    Returns
    -------
    np.ndarray
        Donor weight vector aligned to ``donor_names``, shape ``(J,)``.
    """
    import pandas as pd

    mod = import_rsc()
    SVDModel = mod.SVDModel

    donor_names = [str(n) for n in donor_names]
    target_pre = np.asarray(target_pre, dtype=float).ravel()
    donor_pre = np.asarray(donor_pre, dtype=float)
    T0 = target_pre.shape[0]
    treated_key = "__treated__"

    data = {name: donor_pre[:, j] for j, name in enumerate(donor_names)}
    data[treated_key] = target_pre
    train_df = pd.DataFrame(data=data)

    # RobustSyntheticControl: N=1 (each series its own row), M=T0,
    # includePastDataOnly=False, modelType='svd', svdMethod='numpy'.
    model = SVDModel(
        treated_key, k, 1, T0,
        probObservation=1.0, svdMethod="numpy",
        otherSeriesKeysArray=donor_names, includePastDataOnly=False,
    )
    model.fit(train_df)
    return np.asarray(model.weights, dtype=float).ravel()
