"""Frozen containers for EASC, the exposure-augmented synthetic control (Spruk 2026).

Everything is NumPy; the only DataFrame touchpoint is
:func:`mlsynth.utils.easc_helpers.setup.prepare_easc_inputs`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from ...config_models import (
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
    MethodDetailsResults,
    TimeSeriesResults,
    WeightsResults,
)


@dataclass(frozen=True)
class EASCInputs:
    """The panel, the exposure index and the observed shock, as arrays.

    Parameters
    ----------
    treated_name : Any
        Label of the treated unit.
    donor_names : Sequence
        Length-``J`` donor labels, in the column order of ``donor_matrix``.
    y : np.ndarray
        Treated outcome path, shape ``(T,)``.
    donor_matrix : np.ndarray
        Donor outcomes, shape ``(T, J)``.
    T0 : int
        Number of pre-treatment periods.
    time_labels : Sequence
        Length-``T`` period labels.
    exposure_treated : float
        The treated unit's exposure index ``s_1``.
    exposure_donors : np.ndarray
        The donors' exposure indices ``s_j``, shape ``(J,)``.
    shock : np.ndarray
        The observed common shock ``x_t``, shape ``(T,)``.
    base_index : int
        Position of the base period ``b`` of the exposure regression.
    metadata : dict
        Provenance: the column names and the exposure window.
    """

    treated_name: Any
    donor_names: Sequence
    y: np.ndarray
    donor_matrix: np.ndarray
    T0: int
    time_labels: Sequence
    exposure_treated: float
    exposure_donors: np.ndarray
    shock: np.ndarray
    base_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_donors(self) -> int:
        return int(self.donor_matrix.shape[1])

    @property
    def pre(self) -> np.ndarray:
        """Boolean mask of the pre-treatment periods."""
        return np.arange(self.T) < self.T0


@dataclass(frozen=True)
class EASCFit:
    """One of the four estimators on the same panel.

    ``pre_rmse`` is the pre-treatment RMSPE of the weights, ``y_1t - sum_j w_j
    Y_jt`` over the pre-period: the quantity the bias bound scales. For a
    corrected fit the correction's pre-period values are visible in ``gap``.
    """

    name: str
    weights: np.ndarray
    counterfactual: np.ndarray
    gap: np.ndarray
    att: float
    pre_rmse: float
    exposure_imbalance: float
    corrected: bool


@dataclass(frozen=True)
class EASCDiagnostics:
    """The diagnostics of Section 4.1, none of which uses post-treatment outcomes.

    Parameters
    ----------
    n_factors : int
        Number ``k`` of exposure-orthogonal latent factors estimated from the donors.
    leverage, latent_leverage : np.ndarray
        ``kappa_t`` for ``g_t = (1, lambda_t, x_t)`` and ``kappa^lambda_t`` for
        ``(1, lambda_t)``, shape ``(T,)`` (Proposition 1).
    partial_leverage : np.ndarray
        ``ell_t`` (Corollary 1), shape ``(T,)``; ``leverage^2 = latent_leverage^2
        + partial_leverage^2``. NaN when the shock is an exact linear function of
        the factors before treatment.
    shock_r2 : float
        Pre-treatment R^2 of the shock on the latent factors.
    max_partial_leverage_pre, max_partial_leverage_post : float
        Largest ``|ell_t|`` before and after treatment.
    d_min : float
        Distance from the treated unit's exposure to the donors' exposure range.
    inside_hull : bool
        Whether ``d_min`` is zero: simplex weights can balance exposure, and the
        correction interpolates.
    exposure_effect, exposure_effect_se : np.ndarray
        ``phi_t`` and its delete-one-donor jackknife standard error, shape ``(T,)``.
    hat_values : np.ndarray
        Each donor's leverage in the exposure regression, shape ``(J,)``.
    phase_leverage : dict
        Phase name to ``(kappa_SC, kappa_EASC)``: the leverage of the phase
        average for synthetic control and the EASC bound of Proposition 4(d).
    """

    n_factors: int
    leverage: np.ndarray
    latent_leverage: np.ndarray
    partial_leverage: np.ndarray
    shock_r2: float
    max_partial_leverage_pre: float
    max_partial_leverage_post: float
    d_min: float
    inside_hull: bool
    exposure_effect: np.ndarray
    exposure_effect_se: np.ndarray
    hat_values: np.ndarray
    phase_leverage: Dict[str, Tuple[float, float]]


@dataclass(frozen=True)
class EASCFrontier:
    """The exposure-fit frontier and the EASC* selection (Proposition 3(b), equation 12).

    Row ``i`` is the best-fitting simplex weights with ``|EI(w)| <= tolerance[i]``;
    row 0 is EBSC-min and the last row the synthetic-control weights.
    ``selected`` is the row EASC* corrects.
    """

    tolerance: np.ndarray
    exposure_imbalance: np.ndarray
    rmspe: np.ndarray
    weights: np.ndarray
    criterion: np.ndarray
    selected: int


@dataclass(frozen=True)
class EASCPhaseInference:
    """Inference for one estimator over one post-treatment phase (Section 4.3).

    ``intervals`` maps each ``M`` to its bias-aware interval. ``breakdown`` is
    the largest ``M`` at which the interval excludes zero (capped at 5), and the
    ``_robust`` fields use the median-absolute-deviation noise scale.
    """

    estimator: str
    phase: str
    first: Any
    last: Any
    estimate: float
    noise_sd: float
    noise_sd_robust: float
    leverage: float
    placebo_p_value: float
    intervals: Dict[float, Tuple[float, float]]
    breakdown: float
    breakdown_robust: float


@dataclass(frozen=True)
class EASCInference:
    """Bias-aware inference for SC and EASC, phase by phase."""

    alpha: float
    m_grid: Tuple[float, ...]
    headline_m: float
    correction_se: str
    rows: Tuple[EASCPhaseInference, ...]
    headline_interval: Tuple[float, float]

    def row(self, estimator: str, phase: str) -> EASCPhaseInference:
        """The row for ``estimator`` (``"SC"`` or ``"EASC"``) and ``phase``."""
        for r in self.rows:
            if r.estimator == estimator and r.phase == phase:
                return r
        raise KeyError(f"no inference row for estimator {estimator!r} and phase {phase!r}")


class EASCResults(BaseEstimatorResults):
    """Results of :meth:`mlsynth.EASC.fit`.

    An :class:`~mlsynth.config_models.EffectResult`: the headline fit (chosen by
    the ``method`` setting) is lifted into the standardized sub-models, and the
    estimator-specific outputs are typed fields.

    Parameters
    ----------
    inputs : EASCInputs
        The panel as arrays.
    method : str
        The headline estimator: ``"easc"``, ``"ebsc_min"`` or ``"easc_tuned"``.
    fits : dict of EASCFit
        ``"sc"``, ``"easc"``, ``"ebsc_min"`` and ``"easc_tuned"``.
    diagnostics : EASCDiagnostics
        Leverage, the silent-factor diagnostic, the exposure geometry and the
        exposure regression.
    frontier : EASCFrontier
        The exposure-fit frontier.
    bias_aware : EASCInference, optional
        Placebos, bias-aware intervals and breakdown values for SC and EASC;
        ``None`` when inference is switched off.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: EASCInputs
    method: str
    fits: Dict[str, EASCFit]
    diagnostics: EASCDiagnostics
    frontier: EASCFrontier
    bias_aware: Optional[EASCInference] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _populate_standard_submodels(self) -> "EASCResults":
        if self.effects is not None:  # pragma: no cover - idempotency guard; effects is None at construction
            return self
        head = self.fits[self.method]
        labels = np.asarray(self.inputs.time_labels)
        T0, T = self.inputs.T0, self.inputs.T
        object.__setattr__(self, "effects", EffectsResults(
            att=float(head.att),
            additional_effects={name: float(fit.att) for name, fit in self.fits.items()},
        ))
        object.__setattr__(self, "time_series", TimeSeriesResults(
            observed_outcome=np.asarray(self.inputs.y, dtype=float),
            counterfactual_outcome=np.asarray(head.counterfactual, dtype=float),
            estimated_gap=np.asarray(head.gap, dtype=float),
            time_periods=labels,
            intervention_time=labels[T0] if T0 < T else None,
        ))
        object.__setattr__(self, "weights", WeightsResults(
            donor_weights={str(k): float(v) for k, v in zip(self.inputs.donor_names, head.weights)}))
        object.__setattr__(self, "fit_diagnostics", FitDiagnosticsResults(rmse_pre=float(head.pre_rmse)))
        if self.bias_aware is not None and self.method == "easc":
            ba = self.bias_aware
            row = ba.row("EASC", "post")
            lo, hi = ba.headline_interval
            object.__setattr__(self, "inference", InferenceResults(
                p_value=float(row.placebo_p_value),
                ci_lower=float(lo),
                ci_upper=float(hi),
                standard_error=float(row.noise_sd),
                confidence_level=1.0 - ba.alpha,
                method=f"bias-aware interval at M={ba.headline_m:g}, placebo noise scale",
                details={"breakdown": row.breakdown, "leverage": row.leverage},
            ))
        dg = self.diagnostics
        object.__setattr__(self, "method_details", MethodDetailsResults(
            method_name="EASC",
            is_recommended=True,
            parameters_used={
                "method": self.method,
                "n_factors": int(dg.n_factors),
                "n_subperiods": self.metadata.get("n_subperiods"),
                "base_period": self.inputs.time_labels[self.inputs.base_index],
                "exposure_treated": float(self.inputs.exposure_treated),
                "d_min": float(dg.d_min),
                "inside_hull": bool(dg.inside_hull),
                "frontier_points": int(self.frontier.tolerance.size),
            },
        ))
        return self


EASCResults.model_rebuild()
