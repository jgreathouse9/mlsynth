"""Configuration for the PROPSC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class PROPSCConfig(BaseEstimatorConfig):
    """Configuration for the compositional common-weights SC/SDID estimator.

    Beyond the common fields (``df``, ``treat``, ``unitid``, ``time``,
    ``display_graphs``, ``save``, colors), PROPSC reads ``outcomes`` (the
    ``K`` proportion columns forming the composition), ``method``
    (``sdid``/``sc``/``did``), ``target`` (which proportion drives the flat
    accessors), and ``inference``.
    """

    outcomes: List[str] = Field(
        ...,
        description="The K proportion columns forming the composition (each "
        "unit-time row of these columns sums to the whole). Must list at least "
        "two outcomes; a common set of unit/time weights is fit across all of "
        "them so the estimated ATTs sum to zero.",
        min_length=2,
    )
    method: Literal["sdid", "sc", "did"] = Field(
        default="sdid",
        description="Weighting scheme: 'sdid' (unit + time weights, intercept "
        "shift), 'sc' (unit weights only, no intercept), or 'did' (uniform "
        "weights).",
    )
    target: Optional[str] = Field(
        default=None,
        description="Which proportion in `outcomes` drives the flat accessors "
        "(att/counterfactual/gap). Defaults to the first outcome.",
    )
    inference: Literal["jackknife", "none"] = Field(
        default="jackknife",
        description="Variance estimator for the per-proportion ATTs. "
        "'jackknife' is the fixed-weights leave-one-unit-out jackknife (the "
        "paper's default); 'none' skips inference.",
    )

    @model_validator(mode="before")
    @classmethod
    def _default_outcome(cls, data):
        # BaseEstimatorConfig requires a scalar ``outcome``; default it to the
        # target (or the first composition column) so PROPSC users need only
        # supply ``outcomes``.
        if isinstance(data, dict):
            outs = data.get("outcomes")
            if outs and not data.get("outcome"):
                # Default the base scalar ``outcome`` to the first composition
                # column (a real column, so base validation passes). The target
                # is validated separately and drives the flat accessors.
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
        return self
