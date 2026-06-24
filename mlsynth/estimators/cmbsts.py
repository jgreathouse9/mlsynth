"""Causal Multivariate Bayesian Structural Time Series (CMBSTS) estimator.

Implements:

    Menchetti, F., & Bojinov, I. (2022). "Estimating the effectiveness of
    permanent price reductions for competing products using multivariate
    Bayesian structural time series models." Annals of Applied Statistics
    16(1): 414-435.

CMBSTS is the multivariate extension of the Bayesian structural time series
counterfactual of Brodersen et al. (2015, ``CausalImpact``). A group of ``d``
outcome series is modelled jointly by a multivariate structural state-space
model -- trend, optional seasonal and cycle components, and a spike-and-slab
regression on control-series paths and exogenous covariates. The pre-period
model is fit by a Gibbs sampler; the counterfactual is forecast from the
posterior predictive distribution and the causal effect is observed minus
predicted, per series, with credible bands. Under partial interference the
non-treated group members (e.g. a competitor brand) carry the spillover effect.

The numerical engine lives in :mod:`mlsynth.utils.cmbsts_helpers` and is
validated cell-by-cell against the authors' ``CausalMBSTS`` R package.
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import CMBSTSConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.cmbsts_helpers.pipeline import run_cmbsts
from ..utils.cmbsts_helpers.structures import CMBSTSResults


class CMBSTS:
    """Causal Multivariate Bayesian Structural Time Series estimator.

    Parameters
    ----------
    config : CMBSTSConfig or dict
        Configuration object. See :class:`mlsynth.config_models.CMBSTSConfig`.
        The treated unit (``treat`` indicator) is the primary series;
        ``group_units`` are the other jointly-modelled series, ``control_units``
        and ``covariates`` form the spike-and-slab regression block.

    Returns
    -------
    CMBSTSResults
        An :class:`~mlsynth.config_models.EffectResult`. The flat accessors
        (``att``, ``att_ci``, ``counterfactual``, ``gap``) resolve over the
        treated series; per-series posterior effects, cumulative effects,
        counterfactual bands, and regressor inclusion probabilities live in the
        typed ``inference_detail`` / ``posterior`` fields. See
        :class:`mlsynth.utils.cmbsts_helpers.structures.CMBSTSResults`.

    Notes
    -----
    - This is a panel state-space estimator; it has no donor weights, so the
      standardized ``weights`` slot is empty with a method note.
    - The credible interval is Bayesian (``method="bayesian_posterior"``); its
      width reflects the posterior predictive uncertainty over the forecast
      horizon, which can be wide for long post-windows.

    References
    ----------
    Menchetti, F., & Bojinov, I. (2022). "Estimating the effectiveness of
    permanent price reductions for competing products using multivariate
    Bayesian structural time series models." Annals of Applied Statistics
    16(1): 414-435.
    Brodersen, K. H., Gallusser, F., Koehler, J., Remy, N., & Scott, S. L.
    (2015). "Inferring causal impact using Bayesian structural time-series
    models." Annals of Applied Statistics 9(1): 247-274.

    Examples
    --------
    >>> from mlsynth import CMBSTS                                     # doctest: +SKIP
    >>> config = {                                                     # doctest: +SKIP
    ...     "df": panel, "outcome": "sales", "unitid": "item",
    ...     "time": "week", "treat": "treated",
    ...     "group_units": ["competitor"], "control_units": ["wine_1", "wine_2"],
    ...     "components": ["trend", "seasonal"], "seas_period": 7,
    ...     "niter": 1000, "seed": 0,
    ... }
    >>> results = CMBSTS(config).fit()                                 # doctest: +SKIP
    """

    def __init__(self, config: Union[CMBSTSConfig, dict]) -> None:
        """Initialize CMBSTS from a Pydantic config or compatible dictionary."""
        if isinstance(config, dict):
            try:
                config = CMBSTSConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(f"Invalid CMBSTS configuration: {exc}") from exc

        self.config: CMBSTSConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.treat: str = config.treat
        self.display_graphs: bool = config.display_graphs

    def fit(self) -> CMBSTSResults:
        """Fit CMBSTS and return a standardized :class:`CMBSTSResults`."""
        try:
            results = run_cmbsts(self.config)
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(f"CMBSTS estimation failed: {exc}") from exc

        pc = self.config.resolved_plot()
        if pc.xlabel is None:
            pc.xlabel = self.time
        if pc.ylabel is None:
            pc.ylabel = self.outcome
        object.__setattr__(results, "plot_config", pc)
        if self.display_graphs:
            try:
                results.plot()
            except MlsynthPlottingError:  # pragma: no cover - defensive error translation
                raise
            except Exception as exc:  # pragma: no cover - defensive error translation
                raise MlsynthPlottingError(f"CMBSTS plotting failed: {exc}") from exc

        return results
