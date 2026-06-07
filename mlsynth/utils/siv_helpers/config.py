"""Configuration for the SIV estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import Field, model_validator
from ...exceptions import MlsynthConfigError
from ...config_models import BaseEstimatorConfig


class SIVConfig(BaseEstimatorConfig):
    """Configuration for the Synthetic Instrumental Variables (SIV) estimator.

    Implements Gulek and Vives-i-Bastida (2024), "Synthetic IV
    Estimation in Panels". SIV is a two-step procedure for panels
    with an instrumental variable: a per-unit synthetic-control fit
    on the pre-period builds debiased outcome / treatment / instrument
    series, and a just-identified 2SLS on those debiased series in the
    post-period delivers a causal effect estimate that is robust to
    both unobserved factor structure and treatment endogeneity given
    a partially-valid instrument.

    Parameters
    ----------
    instrument : str
        Name of the instrument column in ``df``.
    T0 : int or None
        Number of pre-treatment periods. Either ``T0`` or ``post_col``
        must be supplied.
    post_col : str or None
        Optional 0/1 column identifying post-treatment periods.
    T0_train : int or None
        Optional end of the training block inside the pre-period
        (exclusive); the remaining pre-periods form the "blank" block
        used by the ensemble CV and the split-conformal inference.
        Defaults to ``floor(0.75 * T0)``.
    weight_constraint : {"simplex", "l1_ball"}
        SC weight constraint per unit. ``"simplex"`` (default) matches
        the paper's empirical applications; ``"l1_ball"`` is the
        regularised relaxation analysed in Section 3.
    l1_C : float
        L1-ball radius; ignored when ``weight_constraint == "simplex"``.
    mode : {"siv", "projected", "ensemble"}
        Which estimator the orchestrator reports as the primary
        ``theta_hat``. The other variants are always computed and
        returned in ``results.estimates`` for diagnostics.
    ensemble_alpha : float or None
        Override the CV-selected blend weight in ``ensemble`` mode.
        ``None`` (default) triggers the validation-block CV from
        Section 5.1.
    inference_method : {"asymptotic", "conformal", "none"}
        ``"asymptotic"`` uses the IV sandwich SE (valid under
        Theorem 4); ``"conformal"`` runs the split-conformal
        permutation test of Section 5.2.
    alpha : float
        Two-sided significance level for the CI.
    n_permutations : int
        Maximum number of permutations enumerated when building the
        conformal distribution. Ignored under ``asymptotic``.
    """

    instrument: str = Field(
        ..., description="Instrument column name."
    )
    T0: Optional[int] = Field(
        default=None, gt=0,
        description="Number of pre-treatment periods (alternative to post_col).",
    )
    post_col: Optional[str] = Field(
        default=None,
        description="Optional 0/1 column identifying post-treatment periods.",
    )
    T0_train: Optional[int] = Field(
        default=None, ge=2,
        description="End of the training block inside the pre-period.",
    )
    weight_constraint: Literal["simplex", "l1_ball"] = Field(
        default="simplex",
        description="SC weight constraint per unit.",
    )
    l1_C: float = Field(
        default=1.0, gt=0.0,
        description="L1-ball radius for weight_constraint='l1_ball'.",
    )
    mode: Literal["siv", "projected", "ensemble"] = Field(
        default="siv",
        description="Primary estimator variant reported in results.",
    )
    ensemble_alpha: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Override CV-selected ensemble blend weight.",
    )
    inference_method: Literal["asymptotic", "conformal", "none"] = Field(
        default="conformal",
        description="Post-estimation inference procedure.",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided significance level for the CI.",
    )
    n_permutations: int = Field(
        default=5000, ge=100,
        description="Max permutations for the split-conformal test.",
    )
    seed: int = Field(
        default=1400,
        description="Random seed for conformal permutation sampling.",
    )
    display_graph: bool = Field(
        default=False,
        description="Whether to plot the event-study coefficients.",
    )

    @model_validator(mode="after")
    def _check_columns_and_periods(cls, values: Any) -> Any:
        df = values.df
        if values.instrument not in df.columns:
            raise MlsynthConfigError(
                f"instrument '{values.instrument}' is not in df."
            )
        if values.T0 is None and values.post_col is None:
            raise MlsynthConfigError(
                "Either T0 or post_col must be supplied."
            )
        if values.post_col is not None and values.post_col not in df.columns:
            raise MlsynthConfigError(
                f"post_col '{values.post_col}' is not in df."
            )
        return values
