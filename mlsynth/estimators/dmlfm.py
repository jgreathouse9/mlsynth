"""DMLFM -- dynamic multilevel latent factor model.

Pang, X., Liu, L., & Xu, Y. (2022). "A Bayesian Alternative to Synthetic
Control for Comparative Case Studies." *Political Analysis* 30(2):269-288.

A Bayesian counterfactual-imputation estimator. Untreated outcomes follow a
linear model whose covariate coefficients may vary by unit and by time, plus a
latent factor term; Laplacian shrinkage on the loading scales selects the
number of factors, and the treated unit's counterfactual comes from the
posterior predictive distribution, so the credible band is read directly off
the draws.

Cross-validated against ``pblasso`` 1.0.8 on the Abadie, Diamond & Hainmueller
German reunification panel; see ``benchmarks/reference/dmlfm_germany``.
"""

from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np
from pydantic import ValidationError

from ..config_models import BaseEstimatorResults
from ..exceptions import (MlsynthConfigError, MlsynthDataError,
                          MlsynthEstimationError, MlsynthPlottingError)
from ..utils.dmlfm_helpers.config import DMLFMConfig
from ..utils.dmlfm_helpers.pipeline import assemble
from ..utils.dmlfm_helpers.plotter import plot_dmlfm
from ..utils.dmlfm_helpers.sampler import run_gibbs
from ..utils.dmlfm_helpers.setup import prepare_dmlfm_inputs


class DMLFM:
    """Dynamic multilevel latent factor model.

    Parameters
    ----------
    config : DMLFMConfig or dict
        Configuration object. See
        :class:`mlsynth.utils.dmlfm_helpers.config.DMLFMConfig`.

    Returns
    -------
    BaseEstimatorResults
        Posterior counterfactual with a credible band, the ATT posterior mean
        and credible interval, the sorted spectrum of factor-loading scales,
        and the retained draws under ``additional_outputs``.
    """

    def __init__(self, config: Union[DMLFMConfig, Dict[str, Any]]) -> None:
        if isinstance(config, dict):
            try:
                config = DMLFMConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(f"invalid DMLFM configuration: {exc}") from exc
        self.config = config

    def fit(self) -> BaseEstimatorResults:
        cfg = self.config
        try:
            inputs = prepare_dmlfm_inputs(cfg)
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except Exception as exc:
            raise MlsynthDataError(f"DMLFM ingestion failed: {exc}") from exc

        try:
            rng = np.random.default_rng(cfg.seed)
            draws = run_gibbs(inputs, rng)
        except Exception as exc:
            raise MlsynthEstimationError(f"DMLFM sampling failed: {exc}") from exc

        observed = (cfg.df.sort_values([cfg.unitid, cfg.time])
                    .loc[lambda d: d[cfg.unitid].astype(str) == inputs.treated_name,
                         cfg.outcome].to_numpy(float))
        results = assemble(inputs, draws, observed, cfg.alpha)

        if cfg.display_graphs:
            try:
                plot_dmlfm(results, cfg)
            except MlsynthPlottingError:
                raise
            except Exception as exc:
                raise MlsynthPlottingError(f"DMLFM plotting failed: {exc}") from exc
        return results
