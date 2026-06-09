"""Harmonic Synthetic Control (HSC) estimator.

Implements:

    Liu, Z., & Xu, Y. (2026). "The Harmonic Synthetic Control Method."

HSC addresses the spurious-matching risk of synthetic control on
nonstationary outcomes with unit-specific stochastic trends. Instead of the
binary choice between matching on raw levels (vulnerable to spurious
matching from idiosyncratic trends) and matching on differences (which
discards useful *shared* trend variation), HSC uses a soft, data-driven
allocation:

* donor weights are estimated under a frequency-dependent metric
  ``W_{rho,q}`` that downweights low-frequency (trend-like) residual
  variation, and
* a treated-unit-specific smooth component ``E`` absorbs the low-frequency
  residual and is forecast forward to complete the counterfactual.

A single allocation parameter ``rho in [0, 1]`` -- selected by rolling-origin
cross-validation -- continuously interpolates between SC on ``q``-th
differences (``rho -> 0``) and SC on levels with an intercept/trend
(``rho -> 1``). See ``mlsynth.utils.hsc_helpers`` for the algorithmic pieces.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import (
    EffectsResults,
    FitDiagnosticsResults,
    HSCConfig,
    MethodDetailsResults,
    TimeSeriesResults,
    WeightsResults,
)
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.datautils import balance
from ..utils.hsc_helpers.orchestration import solve_hsc, summarize_effects
from ..utils.hsc_helpers.plotter import plot_hsc
from ..utils.hsc_helpers.setup import prepare_hsc_inputs
from ..utils.hsc_helpers.structures import HSCResults


class HSC:
    """Harmonic Synthetic Control (HSC) estimator.

    Parameters
    ----------
    config : HSCConfig or dict
        Configuration object. See :class:`mlsynth.config_models.HSCConfig`.

    Returns
    -------
    HSCResults
        Donor weights, the fitted smooth component, the counterfactual, and
        the cross-validated allocation ``rho``.

    Notes
    -----
    Following Liu & Xu (2026), this ships the HSC *point* estimator only.
    Uncertainty quantification rests on additional assumptions beyond those
    needed for the point estimate (the authors point to a Cattaneo-Feng-Titiunik
    (2021) style prediction interval) and is deliberately not implemented here.
    """

    def __init__(self, config: Union[HSCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = HSCConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid HSC configuration: {exc}"
                ) from exc

        self.config: HSCConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.treat: str = config.treat

        self.q: int = config.q
        self.rho_grid = list(config.rho_grid)
        self.cv_splits: int = config.cv_splits
        self.ridge = config.ridge  # float (relative) or "sdid"
        self.forecaster: str = config.forecaster
        self.display_graphs: bool = config.display_graphs

    def fit(self) -> HSCResults:
        """Run the HSC pipeline."""

        try:
            balance(self.df, self.unitid, self.time)
        except MlsynthDataError:
            raise
        except Exception as exc:
            raise MlsynthDataError(f"Error balancing panel data: {exc}") from exc

        try:
            inputs = prepare_hsc_inputs(
                df=self.df,
                outcome=self.outcome,
                unitid=self.unitid,
                time=self.time,
                treat=self.treat,
                q=self.q,
                n_splits=self.cv_splits,
            )
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except Exception as exc:
            raise MlsynthDataError(f"Error preparing HSC inputs: {exc}") from exc

        try:
            design = solve_hsc(
                inputs=inputs,
                q=self.q,
                rho_grid=self.rho_grid,
                n_splits=self.cv_splits,
                ridge=self.ridge,
                forecaster=self.forecaster,
            )
            att, cf_full, te = summarize_effects(inputs=inputs, design=design)
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(f"HSC estimation failed: {exc}") from exc

        weights_by_donor = {
            name: float(w)
            for name, w in zip(inputs.donor_names, design.omega)
            if abs(w) > 1e-8
        }

        # Standardized two-family (effect) result contract. The contract gap is
        # the full-length treated-minus-counterfactual series (HSC's own
        # ``treatment_effect`` stays post-only).
        y_full = np.asarray(inputs.y_target, dtype=float)
        cf_arr = np.asarray(cf_full, dtype=float)
        times = np.asarray(inputs.time_labels)
        T0 = inputs.T0
        pre_rmse = float(np.sqrt(np.mean((y_full[:T0] - cf_arr[:T0]) ** 2)))
        results = HSCResults(
            effects=EffectsResults(att=float(att)),
            time_series=TimeSeriesResults(
                observed_outcome=y_full,
                counterfactual_outcome=cf_arr,
                estimated_gap=y_full - cf_arr,
                time_periods=times,
                intervention_time=(times[T0] if T0 < inputs.T else None),
            ),
            weights=WeightsResults(
                donor_weights={str(k): float(v) for k, v in weights_by_donor.items()}),
            fit_diagnostics=FitDiagnosticsResults(rmse_pre=pre_rmse),
            method_details=MethodDetailsResults(method_name="HSC"),
            inputs=inputs,
            design=design,
            counterfactual_full=cf_full,
            treatment_effect=te,
            weights_by_donor=weights_by_donor,
        )

        if self.display_graphs:
            try:
                plot_hsc(results)
            except MlsynthPlottingError:
                raise
            except Exception as exc:
                raise MlsynthPlottingError(f"HSC plotting failed: {exc}") from exc

        return results
