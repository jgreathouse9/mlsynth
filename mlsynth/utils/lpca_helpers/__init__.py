"""Helpers for the LPCA estimator (Feng 2024).

Local principal component analysis: match each unit to its ``K`` nearest
neighbours on one block of periods, take a truncated SVD of the neighbour
submatrix on the complementary block, and read the treated unit's row off the
reconstruction. Because the approximation is local, the untreated-outcome
matrix has to be low-rank only in a neighbourhood, which is what admits a
factor structure that is nonlinear in the latent variables.

Module layout:

* :mod:`.core` -- the two numerical steps (:func:`pseudo_max_distance`,
  :func:`neighbour_index`, :func:`select_rank`, :func:`local_pca_row`).
* :mod:`.config` -- :class:`LPCAConfig`.
* :mod:`.structures` -- :class:`LPCAInputs`, :class:`LPCAResults`.
* :mod:`.setup` -- panel ingestion via ``dataprep``.
* :mod:`.pipeline` -- :func:`run_lpca` causal dispatcher.
* :mod:`.plotter` -- the local-spectrum diagnostic.
"""

from __future__ import annotations

from .config import LPCAConfig
from .core import (
    METRICS,
    local_pca_row,
    neighbour_index,
    pseudo_max_distance,
    select_rank,
)
from .pipeline import run_lpca
from .plotter import plot_local_spectrum
from .setup import prepare_lpca_inputs, resolve_match_periods, resolve_neighbours
from .structures import LPCAInputs, LPCAResults

__all__ = [
    "METRICS",
    "LPCAConfig",
    "LPCAInputs",
    "LPCAResults",
    "local_pca_row",
    "neighbour_index",
    "plot_local_spectrum",
    "prepare_lpca_inputs",
    "pseudo_max_distance",
    "resolve_match_periods",
    "resolve_neighbours",
    "run_lpca",
    "select_rank",
]
