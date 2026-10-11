"""Helper package for EASC, the exposure-augmented synthetic control (Spruk 2026).

``setup`` ingests the panel through ``dataprep`` and reads the exposure index
and the observed shock; the pure kernels are ``diagnostics`` (leverage and the
silent-factor diagnostic), ``correction`` (the cross-donor exposure regression),
``frontier`` (the exposure-fit frontier) and ``inference`` (bias-aware intervals
and breakdown values); ``pipeline`` runs them; ``structures`` holds the frozen
results; ``plotter`` draws them.
"""

from .config import EASCConfig
from .structures import (
    EASCDiagnostics,
    EASCFit,
    EASCFrontier,
    EASCInference,
    EASCInputs,
    EASCPhaseInference,
    EASCResults,
)
from .setup import prepare_easc_inputs
from .pipeline import EASCEstimates, run_easc
from .plotter import plot_easc

__all__ = [
    "EASCConfig",
    "EASCInputs", "EASCFit", "EASCDiagnostics", "EASCFrontier", "EASCPhaseInference", "EASCInference",
    "EASCResults", "EASCEstimates",
    "prepare_easc_inputs", "run_easc", "plot_easc",
]
