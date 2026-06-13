"""HSVT primitives for PCR-SC — back-compat re-export of the shared kernel.

The implementations now live in :mod:`mlsynth.utils.pcr` (the single source of
truth shared by ClusterSC, SI, and SNN). This module is preserved so existing
imports (``mlsynth.utils.clustersc_helpers.pcr.hsvt``) keep working unchanged;
``hsvt`` and ``select_rank`` are byte-for-byte the same objects.

See Rho, Tang, Bergam, Cummings & Misra (2025), *ClusterSC*, Algorithm 2 for the
HSVT step and the three rank-selection modes (``"cumvar"``, ``"fixed"``,
``"usvt"``).
"""

from __future__ import annotations

from ...pcr.core import hsvt
from ...pcr.rank import _standardise, select_rank

__all__ = ["hsvt", "select_rank", "_standardise"]
