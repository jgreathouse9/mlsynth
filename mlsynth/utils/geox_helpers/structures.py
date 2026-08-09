"""Result containers for GEOX.

GEOX produces a *design*, not a treatment effect, so its result is a
:class:`~mlsynth.config_models.DesignResult`: which markets to treat, the donor
weights the analysis will use, the power table behind the choice, and a
``report`` slot that stays ``None`` until the experiment has actually run.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ...config_models import DesignResult, WeightsResults


@dataclass
class CandidateDesign:
    """One candidate test region's deployable design.

    Attributes
    ----------
    units : list
        The markets in the candidate region.
    weights : WeightsResults
        Both SDID weight vectors: ``donor_weights`` over markets and
        ``time_weights`` over the pre-period dates.
    observed : np.ndarray
        The test region's aggregated outcome over the panel.
    counterfactual : np.ndarray
        Synthetic path over the panel.
    n_pre : int
        Pre-period length the design was fit on.
    pre_rmspe, scaled_l2 : float
        Pre-period fit quality.
    mde, power, rank : float or None
        Stitched in from the shortlist: the minimum detectable effect, the power
        at that effect, and the composite rank. ``None`` when the candidate
        cleared no effect size.
    mde_planning : float or None
        The winner's MDE re-scored on backtests that took no part in choosing
        it. ``mde`` is the smallest in a field of candidates and so is
        optimistic; this one is not selected on. Set only on the winner, and
        ``None`` when ``n_validation_backtests`` is 0 or the panel is too short.
    """

    units: List
    weights: WeightsResults
    observed: np.ndarray
    counterfactual: np.ndarray
    n_pre: int
    pre_rmspe: float
    scaled_l2: float
    mde: Optional[float] = None
    power: Optional[float] = None
    rank: Optional[float] = None
    mde_planning: Optional[float] = None


@dataclass
class MarketSearch:
    """The design search: every candidate region and the winner.

    Attributes
    ----------
    shortlist : pd.DataFrame
        Ranked table, one row per (candidate, duration) that cleared the power
        threshold.
    power_table : pd.DataFrame
        Power at every (candidate, duration, effect size), before the MDE
        collapses it. This is what the power curves are drawn from.
    candidates : list of CandidateDesign
        Every candidate's deployable design.
    winner : CandidateDesign or None
        The top-ranked design, or ``None`` when nothing cleared the threshold.
    """

    shortlist: pd.DataFrame
    power_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    candidates: List[CandidateDesign] = field(default_factory=list)
    winner: Optional[CandidateDesign] = None


class GEOXResults(DesignResult):
    """Top-level result of the GEOX design.

    A :class:`~mlsynth.config_models.DesignResult` front door
    (``selected_units`` / ``assignment`` / ``design_weights`` / ``power`` /
    ``metadata`` / ``report``) plus the grouped :class:`MarketSearch` detail.
    """

    search: Optional[MarketSearch] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
