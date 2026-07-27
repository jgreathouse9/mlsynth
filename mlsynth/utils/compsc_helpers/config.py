"""Configuration for the COMPSC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class COMPSCConfig(BaseEstimatorConfig):
    """Configuration for the compositional (Aitchison-geometry) synthetic control.

    Beyond the common fields (``df``, ``treat``, ``unitid``, ``time``,
    ``display_graphs``, ``save``, colors), COMPSC reads ``outcomes`` (the ``K``
    share columns forming the composition), ``geometry`` (which log-ratio map
    defines the fitting objective), ``baseline`` (the reference category, used
    only by ``geometry="alr"``), ``target`` (which share drives the flat
    accessors), ``inference``, ``placebo_screen`` and ``zero_replacement``.
    """

    outcomes: List[str] = Field(
        ...,
        description="The K share columns forming the composition. Rows are "
        "renormalized to sum to one, so unnormalized counts are accepted. Must "
        "list at least two columns; one weight vector is fit across all of them.",
        min_length=2,
    )
    geometry: Literal["clr", "alr"] = Field(
        default="clr",
        description="Log-ratio map defining the fitting objective. 'clr' "
        "(centered log-ratio) is the isometry for the Aitchison metric, so the "
        "fit minimizes pre-treatment Aitchison distance and does not depend on "
        "any reference category -- the default. 'alr' (additive log-ratio) "
        "reproduces Boussim (2026) Algorithm 1 exactly but is baseline-"
        "dependent; combine with `baseline` for paper fidelity.",
    )
    baseline: Optional[str] = Field(
        default=None,
        description="Reference category for geometry='alr'. Defaults to the "
        "last entry of `outcomes`. Ignored when geometry='clr'.",
    )
    target: Optional[str] = Field(
        default=None,
        description="Which share in `outcomes` drives the flat accessors "
        "(att/counterfactual/gap). Defaults to the first outcome.",
    )
    inference: Literal["placebo", "none"] = Field(
        default="placebo",
        description="'placebo' runs the Abadie permutation test on the post/pre "
        "Aitchison RMSPE ratio (Algorithm 1, Step 6); 'none' skips inference.",
    )
    placebo_screen: float = Field(
        default=5.0,
        gt=0.0,
        description="Retain a placebo unit only if its pre-treatment Aitchison "
        "RMSPE is at most this multiple of the treated unit's. The paper uses 5.",
    )
    zero_replacement: float = Field(
        default=1e-6,
        ge=0.0,
        description="Constant added to zero shares before the log-ratio "
        "transform (Assumption 1 is relaxed the standard CoDA way; Algorithm 1, "
        "Step 1). Set to 0 to reject zeros instead.",
    )

    @model_validator(mode="before")
    @classmethod
    def _default_outcome(cls, data):
        # BaseEstimatorConfig requires a scalar ``outcome``; default it to the
        # first composition column so COMPSC users need only supply ``outcomes``.
        if isinstance(data, dict):
            outs = data.get("outcomes")
            if outs and not data.get("outcome"):
                data["outcome"] = outs[0]
        return data

    @model_validator(mode="after")
    def _validate_composition(self):
        cols = set(self.df.columns)
        missing = [c for c in self.outcomes if c not in cols]
        if missing:
            raise MlsynthConfigError(
                f"outcomes not found in df.columns: {missing}")
        if len(set(self.outcomes)) != len(self.outcomes):
            raise MlsynthConfigError("outcomes must be unique.")
        if self.target is not None and self.target not in self.outcomes:
            raise MlsynthConfigError(
                f"target {self.target!r} must be one of outcomes "
                f"{list(self.outcomes)}.")
        if self.baseline is not None and self.baseline not in self.outcomes:
            raise MlsynthConfigError(
                f"baseline {self.baseline!r} must be one of outcomes "
                f"{list(self.outcomes)}.")
        return self
