"""SPILLCORRECT: correct known contaminated controls in a fitted design (#932).

* :mod:`.config` -- :class:`SPILLCORRECTConfig`.
* :mod:`.setup` -- the realized panel on the design's units and periods.
* :mod:`.correct` -- the iterative and inclusive corrections, on arrays.
* :mod:`.pipeline` -- correct, recompute the design's inference, report.
* :mod:`.structures` -- inputs, corrections and the result.
* :mod:`.plotter` -- the effect path before and after the correction.
"""
from __future__ import annotations

from .config import SPILLCORRECTConfig
from .correct import inclusive_correction, iterative_correction
from .pipeline import run_spillcorrect
from .plotter import plot_spillcorrect
from .structures import (Correction, CorrectionInputs, SpillCorrectEstimate,
                         SpillCorrectResults)

__all__ = [
    "Correction",
    "CorrectionInputs",
    "SPILLCORRECTConfig",
    "SpillCorrectEstimate",
    "SpillCorrectResults",
    "inclusive_correction",
    "iterative_correction",
    "plot_spillcorrect",
    "run_spillcorrect",
]
