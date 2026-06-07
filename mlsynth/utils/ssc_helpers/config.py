"""Configuration for the SSC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from pydantic import Field
from ...config_models import BaseEstimatorConfig


class SSCConfig(BaseEstimatorConfig):
    """Configuration for the SSC (Staggered Synthetic Control) estimator.

    Cao, Lu & Wu (2026), *"Synthetic Control Inference for Staggered
    Adoption"* (The Econometrics Journal). Models each unit's untreated
    outcome as an intercept plus a simplex synthetic control on all other
    units (not-yet-treated units are valid donors), jointly estimates every
    unit x time effect by GLS, and reports event-time / overall ATT with
    Andrews (2003) end-of-sample stability inference. Targets staggered
    adoption with a long pre-period (large ``T``, moderate ``N``, small
    ``S``). Inherits the standard ``df`` / ``outcome`` / ``treat`` /
    ``unitid`` / ``time`` interface.

    Parameters
    ----------
    inference : bool
        Attach Andrews end-of-sample bands and p-values to the event-time and
        overall ATT. Default True.
    alpha : float
        Two-sided level for the bands (default 0.1 -> 90% band).
    """

    inference: bool = Field(
        default=True,
        description="Attach Andrews end-of-sample bands/p-values to the ATTs.")
    alpha: float = Field(
        default=0.1, gt=0.0, lt=1.0,
        description="Two-sided level for the end-of-sample bands.")
