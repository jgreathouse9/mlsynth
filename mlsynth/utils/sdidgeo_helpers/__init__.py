"""Helpers for SDIDGEO: synthetic difference-in-differences geo experiment design."""

from .config import SDIDGEOConfig
from .orchestration import run_design
from .structures import CandidateDesign, MarketSearch, SDIDGEOResults

__all__ = [
    "SDIDGEOConfig",
    "run_design",
    "CandidateDesign",
    "MarketSearch",
    "SDIDGEOResults",
]
