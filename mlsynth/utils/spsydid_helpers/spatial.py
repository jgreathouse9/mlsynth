"""Spatial weight matrix utilities for SpSyDiD.

The estimator accepts a row-standardised :math:`N \\times N` spatial weight
matrix :math:`W` directly. These helpers cover the common ways one builds
:math:`W` in practice, so users can either plug in their own matrix or
construct one from coordinates / adjacency information without an external
dependency on ``libpysal``.
"""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Sequence

import numpy as np

from ...exceptions import MlsynthDataError


def _require_finite_coords(coords: np.ndarray) -> None:
    """Reject coordinate arrays containing NaN / Inf entries.

    Non-finite coordinates silently propagate into the distance matrix
    (producing ``nan`` / ``inf`` distances and a degenerate, often
    all-zero, weight matrix), so guard at the boundary instead.
    """
    if not np.all(np.isfinite(coords)):
        raise MlsynthDataError("coords must not contain NaN or Inf entries.")


def validate_spatial_matrix(W: np.ndarray, n_units: int) -> np.ndarray:
    """Sanity-check ``W`` and return a float-array copy.

    Checks:
        * shape is ``(n_units, n_units)``;
        * entries are finite and non-negative;
        * diagonal is zero (a unit is not its own spatial neighbour).

    The matrix is *not* automatically row-standardised here -- pass it
    through :func:`row_standardize` first if needed.
    """
    if not isinstance(W, np.ndarray):
        W = np.asarray(W, dtype=float)
    else:
        W = W.astype(float, copy=True)
    if W.ndim != 2 or W.shape != (n_units, n_units):
        raise MlsynthDataError(
            f"Spatial matrix W has shape {W.shape}; expected ({n_units}, {n_units})."
        )
    if not np.all(np.isfinite(W)):
        raise MlsynthDataError("Spatial matrix W contains NaN or Inf entries.")
    if (W < 0).any():
        raise MlsynthDataError("Spatial matrix W must be non-negative.")
    if np.any(np.diag(W) != 0):
        raise MlsynthDataError("Spatial matrix W must have a zero diagonal.")
    return W


def row_standardize(W: np.ndarray, warn_isolated: bool = False) -> np.ndarray:
    """Divide each row of ``W`` by its row sum.

    Rows with zero sum (units with no neighbours) are left as zero. The
    paper's algorithm assumes row-standardised :math:`W` so the
    spillover term :math:`(WD)_{it} = \\sum_j w_{ij} D_{jt}` lies in
    :math:`[0, 1]`.

    Parameters
    ----------
    W : np.ndarray
        Non-negative weight matrix.
    warn_isolated : bool
        If True, emit a :class:`RuntimeWarning` when one or more rows
        sum to zero (units with no spatial neighbours). Such units can
        never be classified as spillover-exposed and contribute a
        constant-zero exposure column, which is easy to miss. Off by
        default to keep the low-level helper quiet; enabled at the
        :func:`mlsynth.utils.spsydid_helpers.setup.prepare_spsydid_inputs`
        boundary.
    """
    row_sums = W.sum(axis=1, keepdims=True)
    isolated = row_sums.flatten() == 0
    if warn_isolated and isolated.any():
        n_iso = int(isolated.sum())
        warnings.warn(
            f"Spatial weight matrix has {n_iso} unit(s) with no neighbours "
            "(zero row sum); these units have zero spillover exposure and "
            "cannot be classified as indirectly treated.",
            RuntimeWarning,
            stacklevel=2,
        )
    safe = np.where(row_sums > 0, row_sums, 1.0)
    out = W / safe
    out[isolated] = 0.0
    return out


def knn_weights(
    coords: np.ndarray,
    k: int,
    row_standardized: bool = True,
) -> np.ndarray:
    """Build a :math:`k`-nearest-neighbour spatial weight matrix from coords.

    Parameters
    ----------
    coords : np.ndarray
        Shape ``(N, d)`` of unit coordinates in some metric space
        (e.g. ``(lat, lon)`` or projected ``(x, y)``). Euclidean
        distance is used; project to a metric CRS for geographic data.
    k : int
        Number of neighbours per unit (excluding self).
    row_standardized : bool
        If True (default), divide each row by ``k`` so weights sum to 1.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2:
        raise MlsynthDataError("coords must be 2-D (N, d).")
    _require_finite_coords(coords)
    N = coords.shape[0]
    if k < 1 or k >= N:
        raise MlsynthDataError(f"k must lie in [1, N-1]; got k={k} for N={N}.")
    # Squared-distance matrix; cheap for N <= a few thousand.
    diffs = coords[:, None, :] - coords[None, :, :]
    sqd = (diffs * diffs).sum(axis=2)
    np.fill_diagonal(sqd, np.inf)
    nearest = np.argsort(sqd, axis=1)[:, :k]
    W = np.zeros((N, N), dtype=float)
    rows = np.repeat(np.arange(N), k)
    cols = nearest.flatten()
    W[rows, cols] = 1.0
    return row_standardize(W) if row_standardized else W


def inverse_distance_weights(
    coords: np.ndarray,
    cutoff: float | None = None,
    power: float = 1.0,
    row_standardized: bool = True,
) -> np.ndarray:
    """Build an inverse-distance spatial weight matrix.

    :math:`w_{ij} = 1 / d(i, j)^{\\text{power}}` for ``i != j``, zero
    elsewhere. Entries beyond ``cutoff`` (Euclidean distance) are set
    to zero.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2:
        raise MlsynthDataError("coords must be 2-D (N, d).")
    _require_finite_coords(coords)
    if power <= 0:
        raise MlsynthDataError("power must be positive.")
    N = coords.shape[0]
    diffs = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diffs * diffs).sum(axis=2))
    with np.errstate(divide="ignore"):
        W = np.where(d > 0, 1.0 / np.power(d, power), 0.0)
    if cutoff is not None:
        if cutoff <= 0:
            raise MlsynthDataError("cutoff must be positive.")
        W[d > cutoff] = 0.0
    np.fill_diagonal(W, 0.0)
    return row_standardize(W) if row_standardized else W


def contiguity_weights(
    adjacency: Dict[int, Iterable[int]],
    unit_order: Sequence,
    row_standardized: bool = True,
) -> np.ndarray:
    """Build a contiguity (queen / rook) spatial weight matrix.

    Parameters
    ----------
    adjacency : dict
        ``{unit_id: iterable of neighbour unit_ids}``.
    unit_order : sequence
        Length-``N`` canonical ordering of unit ids matching the panel.
    row_standardized : bool
        Divide each row by its row sum (i.e., uniform weight 1/k_i
        across neighbours).
    """
    unit_order = list(unit_order)
    N = len(unit_order)
    pos = {u: i for i, u in enumerate(unit_order)}
    W = np.zeros((N, N), dtype=float)
    for u, neighbours in adjacency.items():
        if u not in pos:
            continue
        i = pos[u]
        for v in neighbours:
            if v in pos and v != u:
                W[i, pos[v]] = 1.0
    return row_standardize(W) if row_standardized else W
