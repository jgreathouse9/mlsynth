"""Data-shaping helpers for GeoLift market-selection scoring.

Given the canonical wide panel (``geoex_dataprep(...)["Ywide"]``) and a candidate
test-market set, produce the two arrays one SCM fit needs: the aggregated
treated series and the donor pool. Each helper does a single trivial reshaping
step — and nothing else — so they stay easy to reason about, debug, and test.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError


def _validate_candidate(Ywide: pd.DataFrame, candidate: Iterable) -> List:
    """Return the candidate's members; reject empty or unknown-unit candidates."""
    members = list(candidate)
    if len(members) == 0:
        raise MlsynthConfigError("candidate must contain at least one unit.")
    missing = [unit for unit in members if unit not in Ywide.columns]
    if missing:
        raise MlsynthDataError(f"Candidate units not found in the panel: {missing}.")
    return members


def aggregate_treated(
    Ywide: pd.DataFrame, candidate: Iterable, how: str = "sum"
) -> pd.Series:
    """Collapse the candidate's units into a single treated series.

    ``how="sum"`` (faithful to GeoLift) totals the units — the right aggregation
    for cost/total-lift reporting. ``how="mean"`` averages them, keeping the
    target at donor scale (in the donor convex hull) for the SCM fit. Returns a
    time-indexed Series named ``"treated"``.

    Raises
    ------
    MlsynthConfigError
        If the candidate is empty, or ``how`` is not ``"sum"``/``"mean"``.
    MlsynthDataError
        If any candidate unit is absent from ``Ywide``.
    """
    members = _validate_candidate(Ywide, candidate)
    if how == "sum":
        treated = Ywide[members].sum(axis=1)
    elif how == "mean":
        treated = Ywide[members].mean(axis=1)
    else:
        raise MlsynthConfigError(f"how must be 'sum' or 'mean'; got {how!r}.")
    treated.name = "treated"
    return treated


def donor_matrix(
    Ywide: pd.DataFrame, candidate: Iterable, exclude: Optional[Iterable] = None
) -> pd.DataFrame:
    """The donor pool: every unit NOT in the candidate, in panel-column order.

    ``exclude`` drops additional units beyond the candidate — the spillover
    "exclusion restriction": a treated geo's conflict-neighbours
    (:func:`~mlsynth.utils.geolift_helpers.marketselect.helpers.constraints.conflict_neighbors`)
    must not enter its synthetic control. ``None`` (the default) leaves the donor
    pool exactly as the candidate complement.

    Raises
    ------
    MlsynthConfigError
        If the candidate is empty.
    MlsynthDataError
        If any candidate unit is absent from ``Ywide``, or if removing the
        candidate (and any excluded units) leaves no donors.
    """
    _validate_candidate(Ywide, candidate)
    removed = set(candidate) | set(exclude or ())
    donors = [unit for unit in Ywide.columns if unit not in removed]
    if len(donors) == 0:
        raise MlsynthDataError(
            "No donor units remain after removing the candidate from the panel."
        )
    return Ywide[donors]
