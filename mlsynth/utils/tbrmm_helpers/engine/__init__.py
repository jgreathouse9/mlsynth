"""TBR, the analysis model a TBRMM design is scored against.

Kerman, Wang and Vaver (2017) regress one group aggregate on the other over the
pretest and read the cumulative effect off a shifted, scaled t posterior. Au
(2018) turns that posterior around: fixing a significance level and a power
target and solving for the smallest effect that would clear the threshold gives
a candidate split's minimum detectable effect, computable before the experiment
runs. So every split TBRMM's hill climb evaluates is one fit from this package,
and nothing outside the TBRMM family fits one.

The ``TBR`` estimator remains exported from ``mlsynth`` and documented at
``docs/tbr.rst``, for an advertiser whose groups were fixed by someone else and
who runs no search.
"""

from .config import TBRConfig
from .structures import CumulativeEffect, IROASResult, TBRFit, TBRResults

__all__ = ["TBRConfig", "TBRFit", "CumulativeEffect", "IROASResult",
           "TBRResults"]
