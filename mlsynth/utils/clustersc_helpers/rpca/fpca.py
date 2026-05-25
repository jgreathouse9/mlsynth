"""Functional Principal Component Analysis (FPCA) for RPCA-SC.

Implements Step 1 of Bayani (2021), *Robust PCA Synthetic Control*:

* Smooth each unit's pre-period outcome trajectory with a cubic
  B-spline (Li, Wang & Carroll 2010).
* Apply PCA to the smoothed trajectories.
* Truncate to the leading components that explain at least
  ``cumvar_threshold`` of the variance (paper Section 3 recommends
  :math:`\\geq 95\\%`).
* Standardise the surviving scores so they are ready for k-means.

A fallback to plain PCA is used when the pre-period is too short for a
cubic spline. Edge cases (no units / no scores after truncation) are
handled by returning a degenerate one-cluster partition upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.interpolate import make_interp_spline
from sklearn.decomposition import PCA

from ....exceptions import MlsynthDataError, MlsynthEstimationError


@dataclass(frozen=True)
class FPCAFeatures:
    """Output of :func:`compute_fpca_features`."""

    scores: np.ndarray  # (n_units, rank), standardised FPC scores
    rank: int           # number of components retained
    smoothing: str      # "bspline" or "fallback_pca"


def _spectral_rank(singular_values: np.ndarray, energy_threshold: float) -> int:
    """Smallest :math:`r` whose cumulative spectral energy meets the threshold."""
    if singular_values.size == 0:
        return 0
    energy = singular_values ** 2
    total = float(energy.sum())
    if total <= 0.0:
        return 0
    cum = np.cumsum(energy) / total
    return int(np.searchsorted(cum, energy_threshold) + 1)


def _standardise(scores: np.ndarray) -> np.ndarray:
    """Per-column z-score, with zero-variance columns left at zero."""
    means = scores.mean(axis=0)
    stds = scores.std(axis=0)
    safe = np.where(stds == 0, 1.0, stds)
    out = (scores - means) / safe
    out[:, stds == 0] = 0.0
    return out


def compute_fpca_features(
    pre_outcomes: np.ndarray,
    cumvar_threshold: float = 0.95,
    spline_degree: int = 3,
) -> FPCAFeatures:
    """Compute standardised FPC scores from a panel of pre-period trajectories.

    Parameters
    ----------
    pre_outcomes : np.ndarray
        Pre-period outcomes, shape ``(n_units, T0)``. Rows are units
        (the treated unit included, by paper convention), columns are
        time points.
    cumvar_threshold : float
        Cumulative-variance target for truncating the FPC expansion.
        Paper default is 0.95.
    spline_degree : int
        B-spline degree. Default 3 (cubic, Bayani 2021).
    """
    if not isinstance(pre_outcomes, np.ndarray):
        raise MlsynthDataError("pre_outcomes must be a NumPy array.")
    if pre_outcomes.ndim != 2:
        raise MlsynthDataError("pre_outcomes must be 2D (n_units, T0).")
    if pre_outcomes.shape[1] == 0:
        raise MlsynthDataError("FPCA requires at least one pre-period time point.")
    if not (0.0 < cumvar_threshold <= 1.0):
        raise MlsynthDataError("cumvar_threshold must lie in (0, 1].")

    n_units, n_time = pre_outcomes.shape
    if n_units == 0:
        return FPCAFeatures(
            scores=np.zeros((0, 0)), rank=0, smoothing="bspline",
        )

    smoothing_tag = "bspline"
    if n_time <= spline_degree:
        # Cubic spline needs T0 > degree; fall back to plain PCA on raw data.
        smoothed = pre_outcomes.copy()
        smoothing_tag = "fallback_pca"
    else:
        grid = np.linspace(0.0, 1.0, num=n_time)
        try:
            spline = make_interp_spline(grid, pre_outcomes.T, k=spline_degree)
            smoothed = spline(grid).T
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise MlsynthEstimationError(
                f"B-spline smoothing failed in FPCA: {exc}"
            ) from exc

    try:
        pca = PCA()
        scores = pca.fit_transform(smoothed)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise MlsynthEstimationError(f"PCA failed in FPCA: {exc}") from exc

    # Select the rank from the *centered* spectrum that produced the scores.
    # Using the raw (uncentered) SVD here would let the cross-sectional level
    # swamp the leading component on level-dominated panels (e.g. GDP) and
    # collapse the FPC expansion to rank 1.
    rank = _spectral_rank(pca.singular_values_, cumvar_threshold)
    if rank == 0:
        return FPCAFeatures(
            scores=np.zeros((n_units, 0)),
            rank=0,
            smoothing=smoothing_tag,
        )
    truncated = scores[:, :rank]
    return FPCAFeatures(
        scores=_standardise(truncated),
        rank=rank,
        smoothing=smoothing_tag,
    )
