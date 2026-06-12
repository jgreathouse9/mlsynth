"""Result container for the multi-cell GeoLift analysis."""

from __future__ import annotations

from typing import Dict, List, Optional

from mlsynth.config_models import BaseEstimatorResults, DesignResult


class MultiCellResults(DesignResult):
    """A multi-cell GeoLift analysis: a per-cell effect report plus the
    cross-cell comparison.

    Respects the two-family contract as a :class:`DesignResult` whose ``report``
    is a representative cell's :class:`EffectResult` (the winner if one is
    declared, else the largest-magnitude effect). The full breakdown lives in:

    Attributes
    ----------
    cells : dict of str -> BaseEstimatorResults
        Each treatment cell's realized effect report (ATT, lift, conformal
        inference, weights, cost), measured against the **shared control** pool.
    comparison : list of dict
        Pairwise cross-cell rows: ``cell_a`` / ``cell_b`` / ``att_a`` / ``att_b``
        / ``att_diff`` / ``winner`` (the cell whose ATT confidence interval lies
        strictly above the other's, GeoLift's non-overlapping-CI rule; ``None``
        when the intervals overlap).
    winner : str or None
        The cell that wins every pairwise comparison, else ``None``.
    """

    cells: Optional[Dict[str, BaseEstimatorResults]] = None
    comparison: Optional[List[dict]] = None
    winner: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
