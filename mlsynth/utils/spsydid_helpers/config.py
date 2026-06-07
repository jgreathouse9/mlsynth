"""Configuration for the SpSyDiD estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import Field
from ...config_models import BaseEstimatorConfig


class SpSyDiDConfig(BaseEstimatorConfig):
    """Configuration for the Spatial Synthetic Difference-in-Differences estimator.

    Serenini & Masek (2024). *"Spatial Synthetic
    Difference-in-Differences,"* SSRN 4736857. Extends SDID
    (Arkhangelsky et al. 2021) with a spatial spillover term so the
    estimator separates the direct ATT from the indirect (spillover)
    effect on units exposed via the spatial weight matrix :math:`W`.

    Parameters
    ----------
    spatial_matrix : np.ndarray
        Square :math:`N \\times N` spatial weight matrix. Rows /
        columns must align with ``unit_order`` (or
        ``sorted(df[unitid].unique())`` if ``unit_order`` is None).
        Use the helpers in
        :mod:`mlsynth.utils.spsydid_helpers.spatial` to build ``W``
        from coordinates (k-NN, inverse distance) or from an adjacency
        list (queen / rook contiguity).
    unit_order : list, optional
        Canonical ordering of unit ids matching the rows / columns of
        ``spatial_matrix``. If ``None`` (default), units are ordered
        by ``sorted(df[unitid].unique())``.
    row_standardize_spatial : bool
        Row-standardise ``W`` internally before computing exposure.
        Default True. Skip when the caller has already standardised.
    """

    spatial_matrix: Any = Field(
        ...,
        description="Square (N, N) spatial weight matrix as a numpy array.",
    )
    unit_order: Optional[List[Any]] = Field(
        default=None,
        description="Canonical ordering of unit ids matching the rows / columns "
                    "of spatial_matrix. Defaults to sorted(unique unit ids).",
    )
    row_standardize_spatial: bool = Field(
        default=True,
        description="Row-standardise the spatial matrix internally so each row "
                    "of W sums to 1.",
    )
