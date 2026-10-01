"""Helpers for the TBRMM design estimator (Au 2018).

TBRMM searches for a geo split. :mod:`.engine` is the TBR analysis model of
Kerman, Wang and Vaver (2017) that every candidate split is scored against, and
that the exported ``TBR`` estimator runs directly when the groups were fixed by
someone else. The dependency runs one way: the engine knows nothing about the
search.
"""

from .config import TBRMMConfig
from .engine import (
    CumulativeEffect, IROASResult, TBRConfig, TBRFit, TBRResults,
)
from .structures import TBRMMDesign, TBRMMResults

__all__ = ["TBRMMConfig", "TBRMMDesign", "TBRMMResults",
           "TBRConfig", "TBRFit", "CumulativeEffect", "IROASResult",
           "TBRResults"]
