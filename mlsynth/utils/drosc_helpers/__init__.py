"""Helper package for :class:`mlsynth.DROSC` (Distributionally Robust Synthetic
Control, Koo & Guo 2026)."""
from __future__ import annotations

from .config import DROSCConfig
from .estimation import drosc_point, sc_weights
from .inference import drosc_union_ci, hac_cov
from .pipeline import run_drosc
from .plotter import plot_drosc
from .setup import prepare_drosc_inputs
from .structures import DROSCInputs

__all__ = [
    "DROSCConfig",
    "DROSCInputs",
    "drosc_point",
    "sc_weights",
    "drosc_union_ci",
    "hac_cov",
    "prepare_drosc_inputs",
    "run_drosc",
    "plot_drosc",
]
