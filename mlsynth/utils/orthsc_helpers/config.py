"""Configuration for the Orthogonalized Synthetic Control (ORTHSC) estimator.

Fry, J. (2026). "Orthogonalized Synthetic Controls." ORTHSC is an IV synthetic
control whose ATT estimate is Neyman-orthogonalized with respect to the
(partially identified, simplex-constrained) control weights, with a
fixed-smoothing Series-HAC variance and a Sun (2013) bandwidth giving a t-test
that controls size without a consistent variance.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class OrthSCConfig(BaseEstimatorConfig):
    """Configuration for ORTHSC and its dispatched GMM-SCE method.

    Beyond the standard panel fields (``df``, ``outcome``, ``treat``,
    ``unitid``, ``time``), the estimators in this family use the outcomes of
    units excluded from the control pool as instruments for the control weights
    (Fry). ``method`` selects between the two estimators in the family:

    - ``"orthogonalized"`` (default) -- the Orthogonalized Synthetic Control of
      Fry (2026): a Neyman-orthogonalized ATT with a fixed-smoothing Series-HAC
      t-test. Requires an explicit ``instruments`` set.
    - ``"gmm_sce"`` -- the GMM Synthetic Control Estimator of Fry (2024): simplex
      weights estimated by one-step GMM with the instrument units as
      instruments, optionally with the Andrews--Lu downward-testing procedure
      (``model_selection=True``) that decides which donors are controls vs
      instruments.
    """

    method: Literal["orthogonalized", "gmm_sce"] = Field(
        default="orthogonalized",
        description="Which estimator in the Fry family to run.")
    instruments: List[str] = Field(
        default_factory=list,
        description="Untreated unit labels to use as instruments (outcomes of "
        "units excluded from the control pool, per Fry). Required for "
        "'orthogonalized' and for 'gmm_sce' without model selection; ignored "
        "when 'gmm_sce' chooses the split itself (model_selection=True).")
    controls: Optional[List[str]] = Field(
        default=None, description="Control-pool unit labels (the synthetic "
        "control donors). If omitted, every donor not named as an instrument is "
        "used as a control.")
    alpha: float = Field(default=0.05,
                         description="Significance level for the t-test CI.")
    beta0: float = Field(default=0.0,
                         description="Null value of the ATT for the t-test.")
    include_constant: bool = Field(
        default=True, description="Add a constant as an extra instrument "
        "(mean-matching moment), as in the reference.")
    model_selection: bool = Field(
        default=False,
        description="GMM-SCE only: run the Andrews--Lu downward-testing "
        "procedure to assign donors to controls vs instruments, rather than "
        "using the supplied 'controls'/'instruments'.")
    guaranteed_instruments: Optional[List[str]] = Field(
        default=None,
        description="GMM-SCE model selection only: donor labels that are always "
        "instruments and never controls (e.g. neighbours of the treated unit).")
    weight_tol: float = Field(
        default=1e-5,
        description="GMM-SCE model selection only: a donor enters the starting "
        "control set if its initial GMM weight exceeds this.")

    @model_validator(mode="after")
    def _check(self):
        needs_instruments = self.method == "orthogonalized" or (
            self.method == "gmm_sce" and not self.model_selection)
        if needs_instruments and not self.instruments:
            raise MlsynthConfigError(
                f"method='{self.method}' requires at least one instrument unit "
                "(or set model_selection=True for gmm_sce to choose the split).")
        if len(set(self.instruments)) != len(self.instruments):
            raise MlsynthConfigError("instruments contains duplicate labels.")
        if self.controls is not None:
            overlap = set(self.controls) & set(self.instruments)
            if overlap:
                raise MlsynthConfigError(
                    f"units cannot be both control and instrument: {sorted(overlap)}.")
        if self.guaranteed_instruments is not None:
            if len(set(self.guaranteed_instruments)) != len(self.guaranteed_instruments):
                raise MlsynthConfigError(
                    "guaranteed_instruments contains duplicate labels.")
            if self.controls is not None:
                gi_overlap = set(self.guaranteed_instruments) & set(self.controls)
                if gi_overlap:
                    raise MlsynthConfigError(
                        "guaranteed_instruments overlap supplied controls: "
                        f"{sorted(gi_overlap)}.")
        if self.method == "orthogonalized" and self.model_selection:
            raise MlsynthConfigError(
                "model_selection is a gmm_sce option; it does not apply to "
                "method='orthogonalized'.")
        if not 0.0 < self.alpha < 1.0:
            raise MlsynthConfigError(f"alpha must be in (0, 1); got {self.alpha}.")
        if self.weight_tol <= 0.0:
            raise MlsynthConfigError(f"weight_tol must be > 0; got {self.weight_tol}.")
        return self
