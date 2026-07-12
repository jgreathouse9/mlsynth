"""Frozen dataclasses for the Proximal Inference (PROXIMAL) estimator.

PROXIMAL bundles up to three proximal causal-inference estimators that
all run on the same prepared panel:

* **PI** -- Proximal Inference with negative-control donor outcomes
  ``W`` and donor proxies ``Z0`` (Shi, Li, Miao, Hu and Tchetgen
  Tchetgen 2023, arXiv:2108.13935). A pre-period IV fit imputes the
  post-period counterfactual.

* **PIS** -- Proximal Inference with Surrogates. Adds a second stage
  projecting the treatment effect onto surrogate outcomes ``X``
  instrumented by surrogate proxies ``Z1`` (Liu, Tchetgen Tchetgen and
  Varjao 2023, arXiv:2308.09527), estimated on the full sample.

* **PIPost** -- the post-treatment-only surrogate variant of PIS.

PIS and PIPost run only when surrogate units are configured, so the user
can compare the available estimates side by side. Every method closes
with a GMM sandwich variance for the ATT (HAC/Bartlett middle), validated
value-for-value against the authors' reference code.

The three layers below (inputs, per-method fit, top-level results) keep
the pipeline pluggable and mirror the CLUSTERSC container design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


# Public method names.
PI = "PI"
PIS = "PIS"
PIPOST = "PIPost"
SPSC = "SPSC"
DR = "DR"
PIPW = "PIPW"
DR_OID = "DR-OID"
PIOID = "PIOID"


@dataclass(frozen=True)
class PROXIMALInputs:
    """Preprocessed panel data for the proximal pipeline.

    Parameters
    ----------
    y : np.ndarray
        Treated-unit outcome over all ``T`` periods, shape ``(T,)``.
    donor_outcomes : np.ndarray
        Donor outcomes ``W``, shape ``(T, n_donors)``.
    donor_proxies : np.ndarray
        Donor proxies ``Z0`` (instruments for ``W``), shape
        ``(T, n_donors)``.
    surrogate_outcomes : np.ndarray or None
        Cleaned surrogate outcomes ``X``, shape ``(T, n_surrogate_vars)``;
        ``None`` when no surrogates are configured.
    surrogate_proxies : np.ndarray or None
        Surrogate proxies ``Z1`` (instruments for ``X``); ``None`` when no
        surrogates are configured.
    T : int
        Total number of periods.
    T0 : int
        Number of pre-treatment periods.
    bandwidth : int
        Bartlett HAC truncation lag used for all GMM standard errors.
    time_labels : np.ndarray
        Length-``T`` time labels.
    treated_unit_name : Any
        Identifier of the treated unit.
    donor_names : Sequence
        Length-``n_donors`` donor labels (column order of
        ``donor_outcomes``).
    methods : Sequence of str
        Which estimators to run: any of ``"PI"``, ``"PIS"``, ``"PIPost"``,
        ``"SPSC"``.
    spsc_detrend : bool
        Whether SPSC detrends the treated outcome against a B-spline time
        trend (SPSC-DT vs SPSC-NoDT).
    spsc_lambda : float or None
        log10 ridge penalty for SPSC; ``None`` selects it by LOO-CV.
    spsc_spline_df : int
        Degrees of freedom of the SPSC detrend B-spline basis.
    spsc_basis_degree : int
        Degree of the polynomial sieve on the SPSC treated-outcome instrument
        (1 = linear single proxy; >=2 = nonparametric / series SPSC).
    spsc_conformal : bool
        Whether to compute SPSC conformal prediction intervals.
    spsc_conformal_periods : Sequence of int or None
        Absolute post-period indices to cover with conformal intervals;
        ``None`` covers every post-treatment period.
    """

    y: np.ndarray
    donor_outcomes: np.ndarray
    donor_proxies: Optional[np.ndarray]
    surrogate_outcomes: Optional[np.ndarray]
    surrogate_proxies: Optional[np.ndarray]
    T: int
    T0: int
    bandwidth: int
    time_labels: np.ndarray
    treated_unit_name: Any
    donor_names: Sequence
    methods: Sequence[str] = (PI,)
    spsc_detrend: bool = True
    spsc_lambda: Optional[float] = None
    spsc_spline_df: int = 5
    spsc_basis_degree: int = 1
    spsc_att_degree: int = 0
    spsc_detrend_basis: str = "bspline"
    spsc_detrend_degree: int = 1
    spsc_conformal: bool = False
    spsc_conformal_periods: Optional[Sequence[int]] = None
    # Over-identified DR (DR-OID): instrument *units* (not a separate variable).
    # outcome_instruments = full proxy pool for the outcome bridge h; the
    # treatment bridge q uses only the smaller treatment_instruments subset.
    outcome_instruments: Optional[np.ndarray] = None
    treatment_instruments: Optional[np.ndarray] = None
    dr_oid_ridge: float = 0.0
    dr_oid_n_starts: int = 8
    pioid_hac_lag: int = 10
    pioid_simplex: bool = False
    pioid_band: bool = False
    pioid_band_method: str = "gmm"
    pioid_band_level: float = 0.90

    @property
    def has_surrogates(self) -> bool:
        """True when surrogate outcomes and proxies are both available."""
        return self.surrogate_outcomes is not None and self.surrogate_proxies is not None

    @property
    def n_donors(self) -> int:
        """Number of donor units."""
        return int(self.donor_outcomes.shape[1])

    @property
    def n_post(self) -> int:
        """Number of post-treatment periods."""
        return self.T - self.T0


@dataclass(frozen=True)
class ProximalMethodFit:
    """Single proximal-method (PI / PIS / PIPost) fit output.

    Parameters
    ----------
    name : str
        Method identifier (``"PI"``, ``"PIS"``, or ``"PIPost"``).
    counterfactual : np.ndarray
        Estimated counterfactual outcome path, shape ``(T,)``.
    gap : np.ndarray
        Observed treated minus counterfactual, shape ``(T,)``.
    time_varying_effect : np.ndarray
        Estimated time-varying treatment effect, shape ``(T,)``. For PI
        this equals ``gap``; for the surrogate methods it is the fitted
        ``X gamma`` series.
    att : float
        Mean post-treatment gap.
    att_se : float or None
        GMM/HAC standard error of the ATT (``None`` if inference failed).
    pre_rmse : float
        Root-mean-squared pre-treatment gap.
    post_rmse : float
        Root-mean-squared post-treatment gap.
    alpha_weights : np.ndarray
        Estimated donor coefficients ``alpha``.
    donor_weights : dict
        Mapping ``{donor_name: coefficient}``.
    metadata : dict
        Free-form per-method diagnostics.
    """

    name: str
    counterfactual: np.ndarray
    gap: np.ndarray
    time_varying_effect: np.ndarray
    att: float
    att_se: Optional[float]
    pre_rmse: float
    post_rmse: float
    alpha_weights: np.ndarray
    donor_weights: Dict[Any, float]
    counterfactual_lower: Optional[np.ndarray] = None   # per-period band (T,)
    counterfactual_upper: Optional[np.ndarray] = None
    band_level: Optional[float] = None
    band_kind: Optional[str] = None                     # e.g. "gmm" / "conformal"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ci(self) -> Tuple[float, float]:
        """Two-sided 95% Wald CI for the ATT from the GMM standard error."""
        if self.att_se is None or not np.isfinite(self.att_se):
            return (float("nan"), float("nan"))
        half = 1.959963985 * self.att_se
        return (self.att - half, self.att + half)


@dataclass(frozen=True)
class PROXIMALResults:
    """Top-level container returned by :meth:`mlsynth.PROXIMAL.fit`.

    Parameters
    ----------
    inputs : PROXIMALInputs
        Preprocessed panel.
    pi : ProximalMethodFit or None
        Proximal Inference fit (always populated).
    pis : ProximalMethodFit or None
        Proximal-with-surrogates fit (populated only when surrogates are
        configured).
    pipost : ProximalMethodFit or None
        Post-treatment surrogate fit (populated only when surrogates are
        configured).
    selected_variant : str
        Which fit is exposed via the convenience aliases ``att``,
        ``att_se``, ``counterfactual``, ``gap``, ``donor_weights`` --
        one of ``"PI"``, ``"PIS"``, ``"PIPost"``. Defaults to ``"PI"``.
    metadata : dict
        Free-form pipeline diagnostics.
    """

    inputs: PROXIMALInputs
    pi: Optional[ProximalMethodFit]
    pis: Optional[ProximalMethodFit]
    pipost: Optional[ProximalMethodFit]
    spsc: Optional[ProximalMethodFit] = None
    dr: Optional[ProximalMethodFit] = None
    pipw: Optional[ProximalMethodFit] = None
    dr_oid: Optional[ProximalMethodFit] = None
    pioid: Optional[ProximalMethodFit] = None
    selected_variant: str = PI
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def mode(self) -> str:
        """Solver mode reported to downstream consumers."""
        return "proximal"

    @property
    def methods(self) -> Dict[str, ProximalMethodFit]:
        """``{method_name: fit}`` for the methods that were run, in order."""
        out: Dict[str, ProximalMethodFit] = {}
        if self.pi is not None:
            out[PI] = self.pi
        if self.pis is not None:
            out[PIS] = self.pis
        if self.pipost is not None:
            out[PIPOST] = self.pipost
        if self.spsc is not None:
            out[SPSC] = self.spsc
        if self.dr is not None:
            out[DR] = self.dr
        if self.pipw is not None:
            out[PIPW] = self.pipw
        if self.dr_oid is not None:
            out[DR_OID] = self.dr_oid
        if self.pioid is not None:
            out[PIOID] = self.pioid
        return out

    @property
    def _primary(self) -> Optional[ProximalMethodFit]:
        methods = self.methods
        if not methods:
            return None
        return methods.get(self.selected_variant, next(iter(methods.values())))

    @property
    def att(self) -> float:
        """ATT of the primary variant."""
        fit = self._primary
        return float("nan") if fit is None else fit.att

    @property
    def att_se(self) -> Optional[float]:
        """ATT standard error of the primary variant."""
        fit = self._primary
        return None if fit is None else fit.att_se

    @property
    def counterfactual(self) -> np.ndarray:
        """Counterfactual of the primary variant."""
        fit = self._primary
        return np.full(self.inputs.T, np.nan) if fit is None else fit.counterfactual

    @property
    def gap(self) -> np.ndarray:
        """Gap of the primary variant."""
        fit = self._primary
        return np.full(self.inputs.T, np.nan) if fit is None else fit.gap

    @property
    def donor_weights(self) -> Dict[Any, float]:
        """Donor weights of the primary variant."""
        fit = self._primary
        return {} if fit is None else fit.donor_weights

    @property
    def pre_rmse(self) -> float:
        """Pre-treatment RMSE of the primary variant."""
        fit = self._primary
        return float("nan") if fit is None else fit.pre_rmse

    @property
    def counterfactual_band(self):
        """``(lower, upper)`` per-period counterfactual band of the primary
        variant, or ``None`` when the variant carries no per-period band. Read by
        the cross-method comparison (this dispatcher has no standardized
        ``time_series`` to hold the canonical band)."""
        fit = self._primary
        if fit is None or fit.counterfactual_lower is None:
            return None
        return fit.counterfactual_lower, fit.counterfactual_upper

    def att_by_method(self) -> Dict[str, float]:
        """``{method: ATT}`` across the methods that were run."""
        return {name: fit.att for name, fit in self.methods.items()}

    def se_by_method(self) -> Dict[str, Optional[float]]:
        """``{method: ATT standard error}`` across the methods that were run."""
        return {name: fit.att_se for name, fit in self.methods.items()}

    def ci_by_method(self) -> Dict[str, Tuple[float, float]]:
        """``{method: (lower, upper)}`` Wald CIs from the GMM standard errors."""
        return {name: fit.ci for name, fit in self.methods.items()}
