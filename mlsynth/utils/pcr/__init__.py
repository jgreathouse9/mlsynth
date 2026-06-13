"""Shared principal-component-regression kernel.

The single home for the spectral rank-selection rules and the PCR linear algebra
used across mlsynth's PCR-based estimators (ClusterSC, Synthetic Interventions,
Synthetic Nearest Neighbors). Centralising these primitives keeps the
Donoho-Gavish threshold and the truncated-SVD weights from drifting between
estimators.
"""

from __future__ import annotations

from .core import hsvt, pcr_weights
from .rank import (
    donoho_gavish_omega,
    select_rank,
    spectral_rank,
    usvt_rank,
)

__all__ = [
    "donoho_gavish_omega",
    "hsvt",
    "pcr_weights",
    "select_rank",
    "spectral_rank",
    "usvt_rank",
]
