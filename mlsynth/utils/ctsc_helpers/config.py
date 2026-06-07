"""Configuration for the CTSC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import Field
from ...config_models import BaseEstimatorConfig


class CTSCConfig(BaseEstimatorConfig):
    """Configuration for the Continuous-Treatment Synthetic Control (CTSC).

    Powell, D. (2022). *"Synthetic Control Estimation Beyond Comparative
    Case Studies,"* Journal of Business & Economic Statistics. Generalises
    synthetic control to continuous / multi-valued treatments with no clean
    treated/never-treated split; jointly estimates unit-specific treatment
    slopes and synthetic controls for all units. (The paper calls it "GSC";
    mlsynth uses CTSC to avoid collision with Xu (2017)'s GSC.)

    Parameters
    ----------
    treatment_vars : list of str
        The ``K >= 1`` treatment / explanatory columns (continuous or
        discrete). CTSC estimates an average marginal effect for each.
    population_col : str, optional
        Time-invariant per-unit weight column for the average effect
        (e.g. population). Defaults to uniform weights.
    use_fit_weights : bool
        Use the two-step per-unit fit weights ``Omega_i`` (paper eq. 6).
        Default True.
    inference : bool
        Run the sign-flip Wald test of ``H0: alpha^AE = 0``. Default True.
    n_draws : int
        Rademacher draws for the randomization test.
    random_state : int
        Seed for the randomization-test RNG.

    Notes
    -----
    The base ``treat`` field is unused by CTSC; provide the continuous /
    discrete treatment column(s) via ``treatment_vars`` instead. Pass any
    existing column name for ``treat`` to satisfy the base config.
    """

    treatment_vars: List[str] = Field(
        ...,
        description="The K >= 1 continuous/discrete treatment columns.",
    )
    population_col: Optional[str] = Field(
        default=None,
        description="Per-unit weight column for the average effect "
                    "(default uniform).",
    )
    use_fit_weights: bool = Field(
        default=True,
        description="Use the two-step per-unit fit weights Omega_i.",
    )
    inference: bool = Field(
        default=True,
        description="Run the sign-flip Wald test of H0: alpha^AE = 0.",
    )
    n_draws: int = Field(
        default=2000, ge=100,
        description="Rademacher draws for the randomization test.",
    )
    random_state: int = Field(
        default=0,
        description="Seed for the randomization-test RNG.",
    )
