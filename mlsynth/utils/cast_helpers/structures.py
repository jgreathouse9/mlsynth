"""Frozen structures for the CAST estimator (Xia, Yan & Wainwright 2025).

CAST returns a confidence interval for every treated unit-period, so its
distinctive output is a *panel* of estimates, standard errors and intervals
rather than a single counterfactual path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import ConfigDict, Field

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class CASTInputs:
    """Preprocessed staggered panel for CAST.

    Parameters
    ----------
    Y : np.ndarray
        Outcome panel, shape ``(N, T)``.
    A : np.ndarray
        Observation indicator, shape ``(N, T)``: 1 where the outcome is
        observed under control, 0 at the treated (counterfactual) cells.
    unit_names : np.ndarray
        Unit labels, length ``N``.
    time_labels : np.ndarray
        Period labels, length ``T``.
    adoption_index : np.ndarray
        Per-unit index of first treatment; ``T`` for never-treated units.
    """

    Y: np.ndarray
    A: np.ndarray
    unit_names: np.ndarray
    time_labels: np.ndarray
    adoption_index: np.ndarray

    @property
    def N(self) -> int:
        """Number of units."""
        return int(self.Y.shape[0])

    @property
    def T(self) -> int:
        """Number of periods."""
        return int(self.Y.shape[1])

    @property
    def treated_mask(self) -> np.ndarray:
        """Boolean mask of the treated (counterfactual) cells."""
        return ~self.A.astype(bool)

    @property
    def treated_units(self) -> np.ndarray:
        """Labels of the ever-treated units."""
        return self.unit_names[self.adoption_index < self.T]

    @property
    def never_treated(self) -> np.ndarray:
        """Labels of the never-treated units."""
        return self.unit_names[self.adoption_index >= self.T]


class CASTResults(BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.CAST.fit`.

    An :class:`~mlsynth.config_models.EffectResult`: the standardized
    sub-models describe the average treated unit (so ``att`` / ``att_ci`` /
    ``counterfactual`` / ``gap`` resolve through the base contract), while the
    entrywise panel below carries the per-unit-per-period inference that is the
    method's reason for existing.

    Parameters
    ----------
    inputs : CASTInputs
        Preprocessed panel.
    rank : int
        Rank used for the factor structure.
    effects_panel : np.ndarray
        ``(N, T)`` individual treatment effects; ``nan`` at untreated cells.
    std_errors : np.ndarray
        ``(N, T)`` entrywise standard errors; ``nan`` at untreated cells.
    ci_lower_panel, ci_upper_panel : np.ndarray
        ``(N, T)`` entrywise confidence bounds; ``nan`` at untreated cells.
    counterfactual_panel : np.ndarray
        ``(N, T)`` estimated counterfactual outcomes (observed values retained
        at untreated cells).
    period_effects : dict
        ``{period label: (effect, standard error)}`` -- the aggregate treated
        effect per period, with the weighting requested in the config.
    significance : dict
        ``{period label: {"positive": int, "negative": int, "null": int,
        "n_treated": int}}`` -- the per-period sign breakdown at ``alpha``.
    metadata : dict
        Free-form pipeline diagnostics (rank selection, weighting, ...).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: CASTInputs
    rank: int
    effects_panel: np.ndarray
    std_errors: np.ndarray
    ci_lower_panel: np.ndarray
    ci_upper_panel: np.ndarray
    counterfactual_panel: np.ndarray
    period_effects: Dict[Any, Any]
    significance: Dict[Any, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def effect_table(self) -> List[Dict[str, Any]]:
        """Tidy ``(unit, period, effect, se, ci_lower, ci_upper)`` rows for the
        treated cells -- the entrywise interval panel, long-format."""
        rows: List[Dict[str, Any]] = []
        mask = self.inputs.treated_mask
        for i, j in zip(*np.where(mask)):
            rows.append({
                "unit": self.inputs.unit_names[i],
                "period": self.inputs.time_labels[j],
                "effect": float(self.effects_panel[i, j]),
                "se": float(self.std_errors[i, j]),
                "ci_lower": float(self.ci_lower_panel[i, j]),
                "ci_upper": float(self.ci_upper_panel[i, j]),
            })
        return rows

    def significant_units(self, period: Any, level: float = 0.05) -> Dict[Any, float]:
        """``{unit: effect}`` for units whose effect at ``period`` excludes zero.

        Uses the stored entrywise intervals when ``level`` matches the fitted
        ``alpha``; otherwise rescales the interval to ``level``.
        """
        from scipy.stats import norm

        j = int(np.where(self.inputs.time_labels == period)[0][0])
        z = norm.ppf(1 - level / 2)
        out: Dict[Any, float] = {}
        for i in np.where(self.inputs.treated_mask[:, j])[0]:
            eff, se = self.effects_panel[i, j], self.std_errors[i, j]
            if np.isfinite(eff) and np.isfinite(se) and abs(eff) > z * se:
                out[self.inputs.unit_names[i]] = float(eff)
        return out


CASTResults.model_rebuild()
