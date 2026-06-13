"""Helper package for ROLLDID (rolling-transformation DiD)."""

from .config import ROLLDIDConfig
from .setup import rolldid_setup
from .pipeline import estimate
from .structures import ROLLDIDResults
from .plotter import plot_rolldid

__all__ = [
    "ROLLDIDConfig",
    "rolldid_setup",
    "estimate",
    "ROLLDIDResults",
    "plot_rolldid",
]
