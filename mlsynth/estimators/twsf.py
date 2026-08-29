"""Two-Way Synthetic Forecasting (TWSF) estimator.

Shen, D. (2026). *"Causal Forecasting in Panel Data: A Two-Way Synthetic
Forecasting Approach."* arXiv:2606.18512.

Every other estimator in mlsynth imputes a missing cell *inside* the observed
panel. TWSF forecasts one *outside* it: the treated potential outcome of a unit
that has never had the intervention, at dates past the end of the data.

It does that by learning two things separately and combining them:

1. the cross-unit relationship, from the pre-adoption window when every unit is
   still under control -- who the target resembles;
2. the treated regime's own dynamics, from the post-adoption trajectories of
   donors already exposed to it -- how that regime evolves.

Neither literature supplies both. Synthetic control cannot extrapolate, because
at ``T + 1`` there are no donor outcomes left to borrow; time-series forecasting
cannot counterfactualise, because the target has no treated history to continue.
The donors' treated period is what closes the gap.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import (
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
    MethodDetailsResults,
    TimeSeriesResults,
    TWSFConfig,
    WeightsResults,
)
from ..config_models import BaseEstimatorResults
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from ..utils.twsf_helpers.pipeline import fit_twsf
from ..utils.twsf_helpers.plotter import plot_twsf
from ..utils.twsf_helpers.setup import prepare_twsf_inputs


class TWSF:
    """Forecast a never-treated unit's treated outcome past the end of the panel.

    Parameters
    ----------
    config : TWSFConfig or dict
        Validated configuration. See
        :class:`~mlsynth.config_models.TWSFConfig`.

    Returns
    -------
    BaseEstimatorResults
        ``time_series.observed_outcome`` is the target's realised (control)
        path over the panel; ``counterfactual_outcome`` is the forecast treated
        path over the ``horizon`` dates after it, with
        ``counterfactual_lower`` / ``counterfactual_upper`` its pointwise band.
        ``weights.donor_weights`` are the unit-side weights and
        ``weights.time_weights`` the one-step temporal rule -- the two halves of
        the method.

    Notes
    -----
    ``estimated_gap`` is the forecast treated path minus the last observed
    control level, so a positive gap means the intervention is predicted to
    raise the outcome. The contrast is against a level rather than a
    contemporaneous counterfactual because, past the end of the panel, there is
    no observed control path to difference against.

    Examples
    --------
    >>> from mlsynth import TWSF
    >>> from mlsynth.config_models import TWSFConfig
    >>> res = TWSF(TWSFConfig(df=panel, outcome="y", unitid="unit",
    ...                       time="t", treat="open", target="Cincinnati",
    ...                       L=3, k_y=4, k_z=2, horizon=14,
    ...                       display_graphs=False)).fit()   # doctest: +SKIP
    """

    def __init__(self, config: Union[TWSFConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = TWSFConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(str(exc)) from exc
        if not isinstance(config, TWSFConfig):
            raise MlsynthConfigError(
                f"config must be a TWSFConfig or a dict, got {type(config).__name__}."
            )
        self.config = config

    def fit(self) -> BaseEstimatorResults:
        """Estimate both sides, forecast the horizon, and assemble the result."""
        c = self.config
        inputs = prepare_twsf_inputs(
            df=c.df, outcome=c.outcome, unitid=c.unitid, time=c.time,
            treat=c.treat, target=c.target, horizon=c.horizon, donors=c.donors,
        )
        try:
            fit = fit_twsf(
                y_target_pre=inputs.y_target_pre,
                Y_donors_pre=inputs.Y_donors_pre,
                Y_donors_post=inputs.Y_donors_post,
                L=c.L, k_y=c.k_y, k_z=c.k_z, horizon=c.horizon,
                multistep=c.multistep, alpha_level=c.alpha,
                interval=c.interval,
            )
        except MlsynthEstimationError:
            raise
        except np.linalg.LinAlgError as exc:  # pragma: no cover - numpy's SVD does not fail on a finite design; every finite-design failure is translated upstream by pcr_weights or page_blocks
            raise MlsynthEstimationError(
                f"the TWSF regressions did not solve: {exc}. A near-singular "
                "design usually means k_y or k_z exceeds the signal actually "
                "present; lower them."
            ) from exc

        observed = c.df.loc[c.df[c.unitid] == c.target].sort_values(c.time)
        observed_path = observed[c.outcome].to_numpy(dtype=float)
        last_level = float(observed_path[-1])
        gap = fit.forecast - last_level

        results = BaseEstimatorResults(
            effects=EffectsResults(
                att=float(np.mean(gap)),
                att_std_err=float(np.mean(fit.std_error)),
                additional_effects={
                    "forecast_path": fit.forecast.tolist(),
                    "gap_path": gap.tolist(),
                    "last_observed_level": last_level,
                },
            ),
            time_series=TimeSeriesResults(
                observed_outcome=observed_path,
                counterfactual_outcome=fit.forecast,
                estimated_gap=gap,
                time_periods=np.asarray(inputs.forecast_labels, dtype=object),
                counterfactual_lower=fit.lower,
                counterfactual_upper=fit.upper,
                prediction_interval_level=1.0 - c.alpha,
                prediction_interval_kind=c.interval,
            ),
            weights=WeightsResults(
                donor_weights=dict(zip(inputs.donor_names,
                                       fit.beta.astype(float).tolist())),
                time_weights={f"lag_{i + 1}": float(v)
                              for i, v in enumerate(fit.alpha)},
            ),
            inference=InferenceResults(
                standard_error=float(np.mean(fit.std_error)),
                confidence_level=1.0 - c.alpha,
                method=f"plug-in ({c.interval})",
                details={"std_error_path": fit.std_error.tolist(),
                         "sigma2": fit.sigma2},
            ),
            fit_diagnostics=FitDiagnosticsResults(
                additional_metrics={
                    "sigma2": fit.sigma2,
                    "n_page_blocks": fit.n_blocks,
                    "n_donors": len(inputs.donor_names),
                    "unit_side_periods": int(inputs.y_target_pre.size),
                },
            ),
            method_details=MethodDetailsResults(
                method_name=f"TWSF ({fit.multistep})",
                parameters_used={
                    "L": c.L, "k_y": c.k_y, "k_z": c.k_z,
                    "horizon": c.horizon, "multistep": c.multistep,
                    "alpha": c.alpha, "interval": c.interval,
                    "target": inputs.target_name,
                    "unit_side_end": str(inputs.unit_side_end),
                    "time_side_start": str(inputs.time_side_start),
                    "staggered_adoption": inputs.staggered,
                },
            ),
        )
        if c.display_graphs:
            plot_twsf(results, title=f"TWSF -- {inputs.target_name}",
                      counterfactual_color=c.counterfactual_color,
                      treated_color=c.treated_color)
        return results
