"""Result container for ROLLDID."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from mlsynth.config_models import BaseEstimatorResults


class ROLLDIDResults(BaseEstimatorResults):
    """Rolling-DiD result: a standardized :class:`BaseEstimatorResults`
    (``effects.att`` = the aggregated ATT, ``inference`` = SE / p / CI / method,
    ``time_series`` = the event-study path for common timing) plus the
    rolling-DiD specifics.

    Attributes
    ----------
    transformation : str
        ``"demean"`` (Procedure 2.1) or ``"detrend"`` (Procedure 3.1).
    inference_type : str
        ``"exact"`` / ``"hc3"`` / ``"ri"`` as requested.
    design : str
        ``"common"`` (single cohort) or ``"staggered"``.
    n_treated, n_control : int
        Eventually-treated and never-treated unit counts.
    per_period : pandas.DataFrame or None
        Per-period ATTs (common timing): ``time`` / ``att`` / ``se`` / ``t`` /
        ``p_value`` / ``ci_lower`` / ``ci_upper``.
    per_cohort : pandas.DataFrame or None
        Per-cohort ATTs (staggered): ``cohort`` / ``n_treated`` / ``att`` / ...
    """

    transformation: Optional[str] = None
    inference_type: Optional[str] = None
    design: Optional[str] = None
    n_treated: Optional[int] = None
    n_control: Optional[int] = None
    per_period: Optional[pd.DataFrame] = None
    per_cohort: Optional[pd.DataFrame] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
