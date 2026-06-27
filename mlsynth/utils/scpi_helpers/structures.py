"""Result containers for the CFPT/scpi prediction intervals.

Cattaneo, Feng, Palomba & Titiunik (2025), *"Uncertainty Quantification in
Synthetic Controls with Staggered Treatment Adoption"* (arXiv:2210.05026).

The method quantifies the uncertainty of a synthetic-control prediction
``tau_hat`` by separately bounding two error sources and combining them by a
union bound,

    I(tau) = [ tau_hat - Mbar_in - Mbar_out ,  tau_hat - M_in - M_out ],

where ``[M_in, Mbar_in]`` covers the *in-sample* error (SC-weight estimation)
and ``[M_out, Mbar_out]`` covers the *out-of-sample* error (post-treatment
sampling noise). Four causal predictands are supported -- TSUS (unit x period),
TAUS (unit, time-averaged), TSUA (period, unit-averaged) and TAUA (overall) --
plus simultaneous (uniform-over-periods) bands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SCPIBand:
    """One prediction interval for a single predictand value.

    Attributes
    ----------
    predictand : str
        One of ``"TSUS"``, ``"TAUS"``, ``"TSUA"``, ``"TAUA"``.
    label : Any
        Identifier of the value: a unit (TAUS), a post-period (TSUA), a
        ``(unit, period)`` pair (TSUS), or ``None`` (TAUA).
    point : float
        The point prediction ``tau_hat``.
    lower, upper : float
        Prediction-interval endpoints.
    out_sample : (float, float)
        The out-of-sample contribution ``(M_out, Mbar_out)``.
    in_sample : (float, float)
        The in-sample contribution ``(M_in, Mbar_in)`` (``(0, 0)`` when the
        in-sample error is not modelled, e.g. for MSQRT).
    """

    predictand: str
    label: Any
    point: float
    lower: float
    upper: float
    out_sample: Tuple[float, float]
    in_sample: Tuple[float, float] = (0.0, 0.0)

    @property
    def significant(self) -> bool:
        """True when the interval excludes zero."""
        return (self.lower > 0.0) or (self.upper < 0.0)

    @property
    def ci(self) -> Tuple[float, float]:
        return (self.lower, self.upper)


@dataclass(frozen=True)
class SCPIResults:
    """Full CFPT/scpi uncertainty quantification for a set of predictands.

    Attributes
    ----------
    method : str
        ``"cfpt_scpi"``.
    alpha, alpha_in, alpha_out : float
        Total target miscoverage and its in/out split. When the in-sample
        error is omitted, ``alpha_out == alpha`` and ``alpha_in == 0``.
    in_sample_included : bool
        Whether the in-sample error was modelled.
    taua : SCPIBand
        Overall average effect (the ATT) with its band.
    tsua : dict
        ``{period_label: SCPIBand}`` -- unit-averaged effect each post-period.
    taus : dict
        ``{unit: SCPIBand}`` -- time-averaged effect for each treated unit.
    tsus : dict
        ``{(unit, period_label): SCPIBand}`` -- per unit, per period.
    simultaneous : dict
        ``{unit: [SCPIBand, ...]}`` -- TSUS bands widened for joint coverage
        across all post-periods of that unit.
    sigma : dict
        ``{unit: sigma}`` -- per-unit sub-Gaussian variance-proxy estimate.
    time_dependence : str
        ``"iid"`` or ``"general"`` (controls the time-averaging bound).
    """

    method: str
    alpha: float
    alpha_in: float
    alpha_out: float
    in_sample_included: bool
    taua: SCPIBand
    tsua: Dict[Any, SCPIBand]
    taus: Dict[Any, SCPIBand]
    tsus: Dict[Tuple[Any, Any], SCPIBand]
    simultaneous: Dict[Any, List[SCPIBand]]
    sigma: Dict[Any, float]
    time_dependence: str = "iid"
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Size-weighted unit aggregation (when ``unit_weights`` are supplied): the
    # same TSUA/TAUA predictands but with a convex combination of treated units
    # by user weights (e.g. market size) instead of the 1/m equal weights. The
    # in-sample and out-of-sample components combine with the same weights, per
    # CFPT's predictand decomposition. ``None`` when no weights were given.
    taua_weighted: Optional[SCPIBand] = None
    tsua_weighted: Optional[Dict[Any, SCPIBand]] = None

    # Convenience accessors mirroring the old single-CI surface.
    @property
    def ci(self) -> Tuple[float, float]:
        """The overall-ATT (TAUA) interval."""
        return self.taua.ci

    @property
    def att(self) -> float:
        return self.taua.point
