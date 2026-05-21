"""Time-Aware Synthetic Control (TASC) estimator.

Implements:

    Rho, S., Illick, C., Narasipura, S., Abadie, A., Hsu, D., & Misra, V. (2026).
    "Time-Aware Synthetic Control." arXiv:2601.03099.

TASC embeds the standard SC panel inside a linear-Gaussian state-space model

    x_t = A x_{t-1} + q_{t-1},   q ~ N(0, Q)
    y_t = H x_t + r_t,            r ~ N(0, R)

and learns the parameters theta = {A, H, Q, R, m_0, P_0} from the pre-treatment
data via EM (Kalman filter + RTS smoother E-step, closed-form MLE M-step).
Counterfactual inference runs a Kalman filter pass with the target's
observation-noise variance R_{1, 1} set to infinity over the post-treatment
window, followed by RTS smoothing. The counterfactual is then
``y_hat_{0, t} = h_1^T m_t^s``, with posterior CIs derived from
``h_1^T P_t^s h_1 + R_{1, 1}``.

Algorithm flow (see ``mlsynth.utils.tasc_helpers``):

    setup.py         : Algorithm 3 line 0     prepare_tasc_inputs, init theta_0
    em.py            : Algorithm 2            EM_pre
    filtering.py     : Algorithms 4 and 5     Kalman filter passes
    smoothing.py     : Algorithm 6            RTS smoother
    mstep.py         : Algorithm 7            closed-form MLE M-step
    orchestration.py : Algorithm 3            run_tasc + summarize_effects
    inference.py     : Algorithm 3 footer     h_1^T m^s, posterior CIs
"""

from __future__ import annotations

from typing import Any, Optional, Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import TASCConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.datautils import balance
from ..utils.tasc_helpers.orchestration import run_tasc, summarize_effects
from ..utils.tasc_helpers.plotter import plot_tasc
from ..utils.tasc_helpers.setup import prepare_tasc_inputs
from ..utils.tasc_helpers.structures import TASCResults


class TASC:
    """Time-Aware Synthetic Control (TASC) estimator.

    Fits a linear-Gaussian state-space model to a single-treated-unit
    synthetic control panel via EM, and forms counterfactual estimates and
    posterior confidence intervals from a Kalman / RTS pass that treats the
    target's post-treatment observations as missing.

    Parameters
    ----------
    config : TASCConfig or dict
        Configuration object specifying the panel inputs and the EM / inference
        hyperparameters. See :class:`mlsynth.config_models.TASCConfig`.

    Returns
    -------
    TASCResults
        Container with the learned model, EM diagnostics, smoothed states,
        counterfactual path, and posterior confidence intervals. See
        :class:`mlsynth.utils.tasc_helpers.structures.TASCResults`.

    Notes
    -----
    - Each column of the input panel is a unit, each row is a time period
      (``datautils.dataprep`` convention). Internally, TASC reshapes the
      data to the paper's ``Y in R^{N x T}`` orientation.
    - The hidden state dimension ``d`` should be small relative to
      ``min(n_donors, T)`` to preserve the low-rank structure of the signal.
    - EM is sensitive to initialization. Defaults use a spectral
      (top-``d`` SVD) start which generally produces stable fits.

    References
    ----------
    Rho, S., Illick, C., Narasipura, S., Abadie, A., Hsu, D., & Misra, V.
    (2026). "Time-Aware Synthetic Control." arXiv:2601.03099.

    Examples
    --------
    >>> from mlsynth import TASC                                       # doctest: +SKIP
    >>> config = {                                                     # doctest: +SKIP
    ...     "df": panel,
    ...     "outcome": "sales",
    ...     "unitid": "state",
    ...     "time": "year",
    ...     "treat": "treated",
    ...     "d": 2,
    ... }
    >>> results = TASC(config).fit()                                   # doctest: +SKIP
    """

    def __init__(self, config: Union[TASCConfig, dict]) -> None:
        """Initialize TASC from a Pydantic config or compatible dictionary."""

        if isinstance(config, dict):
            try:
                config = TASCConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid TASC configuration: {exc}"
                ) from exc

        self.config: TASCConfig = config

        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.treat: str = config.treat

        self.d: int = config.d
        self.n_em_iter: int = config.n_em_iter
        self.em_tol: Optional[float] = config.em_tol
        self.diagonal_Q: bool = config.diagonal_Q
        self.diagonal_R: bool = config.diagonal_R
        self.alpha: float = config.alpha
        self.seed: Optional[int] = config.seed

        self.display_graphs: bool = config.display_graphs
        self.save: Any = config.save
        self.counterfactual_color = config.counterfactual_color
        self.treated_color: str = config.treated_color

    def fit(self) -> TASCResults:
        """Run the TASC pipeline and return the learned design.

        Returns
        -------
        TASCResults
            Final design, inputs, posterior inference, and summary effects.
        """

        try:
            balance(self.df, self.unitid, self.time)
        except MlsynthDataError:
            raise
        except Exception as exc:
            raise MlsynthDataError(
                f"Error balancing panel data: {exc}"
            ) from exc

        try:
            inputs = prepare_tasc_inputs(
                df=self.df,
                outcome=self.outcome,
                unitid=self.unitid,
                time=self.time,
                treat=self.treat,
            )
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except Exception as exc:
            raise MlsynthDataError(
                f"Error preparing TASC inputs: {exc}"
            ) from exc

        try:
            design, inference = run_tasc(
                inputs=inputs,
                d=self.d,
                n_em_iter=self.n_em_iter,
                em_tol=self.em_tol,
                diagonal_Q=self.diagonal_Q,
                diagonal_R=self.diagonal_R,
                alpha=self.alpha,
                seed=self.seed,
            )
            att, pre_rmse = summarize_effects(inputs=inputs, inference=inference)
        except (
            MlsynthConfigError,
            MlsynthDataError,
            MlsynthEstimationError,
        ):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"TASC estimation failed: {exc}"
            ) from exc

        results = TASCResults(
            inputs=inputs,
            design=design,
            inference=inference,
            att=att,
            pre_rmse=pre_rmse,
        )

        if self.display_graphs:
            try:
                plot_tasc(
                    results=results,
                    treated_color=self.treated_color,
                    counterfactual_color=self.counterfactual_color,
                    save=self.save,
                    time_axis_label=self.time,
                    outcome_label=self.outcome,
                    treatment_label=self.treat,
                    unit_label=self.unitid,
                )
            except MlsynthPlottingError:
                raise
            except Exception as exc:
                raise MlsynthPlottingError(
                    f"TASC plotting failed: {exc}"
                ) from exc

        return results
