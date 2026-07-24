"""Frozen structures for the RRSC estimator (He, Li, Shi & Miao 2026).

RRSC estimates, under an untreated linear factor model, the direct effect on
the treated unit and the interference effect on every control unit -- without
prespecifying which controls are interfered. The distinctive output is a
per-unit *interference map*: an average effect, a confidence interval and a
p-value for each unit (the treated unit's entry is the direct effect).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from pydantic import ConfigDict, Field

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class RRSCInputs:
    """Preprocessed panel for RRSC estimation.

    Parameters
    ----------
    Y : np.ndarray
        Full outcome panel, shape ``(N, T)`` with the treated unit in row 0.
    X : np.ndarray
        Intervention indicator, shape ``(T,)`` (0 pre, 1 post).
    unit_names : np.ndarray
        Unit labels, length ``N``, treated first.
    treated_name : Any
        Label of the treated unit.
    T, T0 : int
        Total and pre-intervention period counts.
    time_labels : np.ndarray
        Period labels, length ``T``.
    regime : str
        Resolved regime, ``"fixed_n"`` or ``"large_n"``.
    """

    Y: np.ndarray
    X: np.ndarray
    unit_names: np.ndarray
    treated_name: Any
    T: int
    T0: int
    time_labels: np.ndarray
    regime: str

    @property
    def N(self) -> int:
        """Number of units (treated + controls)."""
        return int(self.Y.shape[0])

    @property
    def n_post(self) -> int:
        """Number of post-intervention periods."""
        return self.T - self.T0

    @property
    def treated_outcome(self) -> np.ndarray:
        """Observed outcome path of the treated unit."""
        return self.Y[0]

    @property
    def donor_names(self) -> np.ndarray:
        """Labels of the control units."""
        return self.unit_names[1:]


class RRSCResults(BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.RRSC.fit`.

    An :class:`~mlsynth.config_models.EffectResult`: the standardized sub-models
    are populated for the treated unit (so ``att`` / ``att_ci`` /
    ``counterfactual`` / ``gap`` / ``pre_rmse`` resolve through the base
    contract, the treated ``att`` being the direct effect). The interference
    map -- an average effect, CI and p-value for *every* unit -- lives on the
    typed fields below.

    Parameters
    ----------
    inputs : RRSCInputs
        Preprocessed panel.
    regime : str
        Regime used (``"fixed_n"`` / ``"large_n"``).
    n_factors : int
        Number of common factors.
    interference_effects : dict
        ``{unit label: average effect}`` for every unit (treated = direct).
    interference_ci : dict
        ``{unit label: (lower, upper)}`` confidence intervals.
    interference_pvalue : dict
        ``{unit label: p-value}``.
    counterfactual_panel : np.ndarray
        Per-period counterfactual proxy ``mu_i + lambda_i' f_t``, shape
        ``(N, T)``.
    communality : dict
        ``{unit label: factor communality}`` -- the fraction of the unit's
        pre-period variance the common factors explain. A near-zero value for
        the treated unit warns that its counterfactual is close to a raw
        before-after comparison.
    metadata : dict
        Free-form pipeline diagnostics (block length, loss, selection set, ...).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: RRSCInputs
    regime: str
    n_factors: int
    interference_effects: Dict[str, float]
    interference_ci: Dict[str, Tuple[Optional[float], Optional[float]]]
    interference_pvalue: Dict[str, float]
    counterfactual_panel: np.ndarray
    communality: Dict[str, float]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def direct_effect(self) -> float:
        """Average direct effect on the treated unit (alias of ``att``)."""
        return self.interference_effects[str(self.inputs.treated_name)]

    @property
    def interference_map(self):
        """A tidy ``(unit, effect, ci_lower, ci_upper, pvalue)`` table as a list
        of dicts -- the per-unit direct/interference summary, treated first."""
        rows = []
        for u in self.interference_effects:
            lo, hi = self.interference_ci.get(u, (None, None))
            rows.append(dict(unit=u, effect=self.interference_effects[u],
                             ci_lower=lo, ci_upper=hi,
                             pvalue=self.interference_pvalue.get(u)))
        return rows

    def significant_units(self, level: float = 0.05) -> Dict[str, float]:
        """``{unit: effect}`` for units whose interference/direct effect is
        significant at ``level`` (excluding the treated unit)."""
        treated = str(self.inputs.treated_name)
        return {u: e for u, e in self.interference_effects.items()
                if u != treated and self.interference_pvalue.get(u, 1.0) < level}


RRSCResults.model_rebuild()
