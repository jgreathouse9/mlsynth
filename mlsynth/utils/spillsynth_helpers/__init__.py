"""Spillover-aware synthetic-control helper package.

Currently houses the Cao & Dowd (2023) Sp-adjusted SCM machinery under
:mod:`.cd`. Additional methods (e.g.\ iSCM Di Stefano-Mellace, distance-
weighted spillover variants) will be added as sibling subpackages and
dispatched through ``SPILLSYNTHConfig.method``.
"""

from __future__ import annotations

from .cd import run_cd
from .plotter import plot_spillsynth
from .setup import build_A_example3, prepare_spillsynth_inputs
from .structures import CDFit, SpillSynthInputs, SpillSynthResults

__all__ = [
    "CDFit",
    "SpillSynthInputs",
    "SpillSynthResults",
    "build_A_example3",
    "plot_spillsynth",
    "prepare_spillsynth_inputs",
    "run_cd",
]
