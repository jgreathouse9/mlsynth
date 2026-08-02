"""Helpers for STACKEDSC (Wiltshire 2023)."""

from .config import STACKEDSCConfig
from .inference import placebo_inference
from .pipeline import run_stackedsc
from .plotter import plot_stackedsc
from .structures import (STACKEDSCResults, StackedDesign, StackedEventStudy,
                         StackedPlacebo, StackedUnitFit)

__all__ = [
    "STACKEDSCConfig",
    "STACKEDSCResults",
    "StackedDesign",
    "StackedEventStudy",
    "StackedPlacebo",
    "StackedUnitFit",
    "placebo_inference",
    "plot_stackedsc",
    "run_stackedsc",
]
