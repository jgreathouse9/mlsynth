"""Spectral rank-selection — the single source of truth shared by every
PCR-based estimator in mlsynth (ClusterSC, SI, SNN).

Historically the Donoho-Gavish (2014) universal threshold was reimplemented in
three places, and they drifted: SNN's copy had the ``omega`` coefficients
swapped (``1.43 b + 1.82`` instead of the published ``1.82 b + 1.43``), which
mis-selected the rank. Centralising the formula here removes that drift class.

Three rules are exposed:

* :func:`usvt_rank` -- Universal Singular Value Thresholding
  (Chatterjee 2015; Donoho & Gavish 2014): keep singular values above
  ``omega(ratio) * median``.
* :func:`spectral_rank` -- smallest rank whose cumulative spectral *energy*
  reaches a threshold (operates on raw singular values).
* :func:`select_rank` -- the ClusterSC entry point that consumes a *matrix*
  and dispatches over ``{"cumvar", "fixed", "usvt"}`` (standardising the matrix
  for the data-driven rules).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthEstimationError


def donoho_gavish_omega(ratio: float) -> float:
    """Donoho & Gavish (2014) optimal-hard-threshold coefficient ``omega(beta)``.

    ``omega(beta) = 0.56 beta^3 - 0.95 beta^2 + 1.82 beta + 1.43`` (their
    median-based approximation; ``omega(1) = 2.86``). ``ratio`` is the aspect
    ratio at which the approximation is evaluated -- the canonical choice is
    ``min(m, n) / max(m, n)``, though Agarwal-Shah-Shen evaluate it at ``m / n``.
    """
    return 0.56 * ratio ** 3 - 0.95 * ratio ** 2 + 1.82 * ratio + 1.43


def usvt_rank(singular_values: np.ndarray, ratio: float) -> int:
    """USVT rank: count singular values exceeding ``omega(ratio) * median``.

    Returns at least 1. Callers that need an upper cap (``min(m, n)``) apply it.
    """
    s = np.asarray(singular_values, dtype=float)
    if s.size == 0:
        return 1
    threshold = donoho_gavish_omega(ratio) * np.median(s)
    return max(1, int(np.sum(s > threshold)))


def spectral_rank(singular_values: np.ndarray, energy: float = 0.95) -> int:
    """Smallest rank whose singular values capture ``energy`` of the spectrum."""
    s = np.asarray(singular_values, dtype=float)
    if s.size == 0:
        return 0
    total = float(np.sum(s ** 2))
    if total <= 0.0:
        return 1
    cum = np.cumsum(s ** 2) / total
    return int(np.searchsorted(cum, energy) + 1)


def _standardise(X: np.ndarray) -> np.ndarray:
    """Per-column z-score with zero-variance columns left at zero.

    Used inside :func:`select_rank` for data-driven rules (``"cumvar"``,
    ``"usvt"``) so the spectral-energy comparison is not dominated by uncentered
    level information. The standardised matrix is *only* used for rank picking --
    the HSVT step itself still consumes the raw matrix.
    """
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    safe = np.where(stds == 0, 1.0, stds)
    out = (X - means) / safe
    out[:, stds == 0] = 0.0
    return out


def select_rank(
    X: np.ndarray,
    method: str = "cumvar",
    cumvar_threshold: float = 0.95,
    r: Optional[int] = None,
    standardize: bool = True,
) -> int:
    """Pick a truncation rank for matrix ``X`` under the chosen rule.

    Parameters
    ----------
    X : np.ndarray
        Matrix to decompose, shape ``(m, n)``.
    method : {"cumvar", "fixed", "usvt"}
        Rank-selection rule.
    cumvar_threshold : float
        Cumulative-variance target in ``(0, 1]`` for ``method="cumvar"``.
    r : int, optional
        Explicit rank for ``method="fixed"``.
    standardize : bool
        When True (default), the data-driven rules (``"cumvar"`` / ``"usvt"``)
        operate on the column-standardised version of ``X`` so the leading
        singular value is not dominated by uncentered level information.
        Ignored for ``method="fixed"``.

    Returns
    -------
    int
        Selected rank :math:`r \\in [1, \\min(m, n)]`.
    """
    if X.ndim != 2:
        raise MlsynthEstimationError("HSVT input must be a 2D matrix.")
    m, n = X.shape
    rank_cap = max(1, min(m, n))

    if method == "fixed":
        if r is None or r < 1:
            raise MlsynthConfigError(
                "rank_method='fixed' requires a positive integer `rank`."
            )
        return int(min(r, rank_cap))

    X_for_rank = _standardise(X) if standardize else X

    if method == "cumvar":
        if not (0.0 < cumvar_threshold <= 1.0):
            raise MlsynthConfigError("cumvar_threshold must lie in (0, 1].")
        s = np.linalg.svd(X_for_rank, compute_uv=False)
        return min(spectral_rank(s, cumvar_threshold), rank_cap)

    if method == "usvt":
        s = np.linalg.svd(X_for_rank, compute_uv=False)
        return min(usvt_rank(s, min(m, n) / max(m, n)), rank_cap)

    raise MlsynthConfigError(
        f"Unknown rank_method {method!r}; expected 'cumvar', 'fixed', or 'usvt'."
    )
