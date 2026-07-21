"""Multitask Gaussian Process synthetic control (MTGP).

Implements the Gaussian model of:

    Ben-Michael, E., Arbour, D., Feller, A., Franks, A., & Raphael, S. (2023).
    "Estimating the effects of a California gun control program with multitask
    Gaussian processes." Annals of Applied Statistics 17(2), 985-1016.

MTGP models the control potential outcomes with a multitask Gaussian process
whose kernel is separable over time and units: a global time-GP trend, a
low-rank intrinsic-coregionalization term whose latent factors carry a
squared-exponential smoothness prior over time, and unit intercepts. The
treated unit's post-period cells are masked and imputed; the posterior of that
imputation is the counterfactual, giving a credible band that widens
post-treatment. The posterior is drawn with NUTS (NumPyro, double precision),
so MTGP needs the ``[bayes]`` optional dependency.

See ``mlsynth.utils.mtgp_helpers`` for the algorithmic pieces.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import MTGPConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.mtgp_helpers.model import run_mtgp
from ..utils.mtgp_helpers.plotter import plot_mtgp
from ..utils.mtgp_helpers.setup import prepare_mtgp_inputs
from ..utils.mtgp_helpers.structures import (
    MTGPInference,
    MTGPPosterior,
    MTGPResults,
)
from ..utils.datautils import balance


class MTGP:
    """Multitask Gaussian Process synthetic control (Ben-Michael et al. 2023).

    Parameters
    ----------
    config : MTGPConfig or dict
        Configuration object. See :class:`mlsynth.config_models.MTGPConfig`.

    Returns
    -------
    MTGPResults
        Posterior counterfactual with a credible band that widens post-treatment,
        the ATT posterior mean + credible interval, and NUTS diagnostics.
        Requires ``mlsynth[bayes]``.
    """

    def __init__(self, config: Union[MTGPConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = MTGPConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(f"Invalid MTGP configuration: {exc}") from exc
        self.config: MTGPConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.treat: str = config.treat
        self.display_graphs: bool = config.display_graphs

    def fit(self) -> MTGPResults:
        """Run the MTGP NUTS sampler and assemble results."""
        try:
            balance(self.df, self.unitid, self.time)
        except MlsynthDataError:  # pragma: no cover - defensive passthrough
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthDataError(f"Error balancing panel data: {exc}") from exc

        try:
            inputs = prepare_mtgp_inputs(
                df=self.df, outcome=self.outcome, unitid=self.unitid,
                time=self.time, treat=self.treat, population=self.config.population)
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthDataError(f"Error preparing MTGP inputs: {exc}") from exc

        cfg = self.config
        try:
            draws = run_mtgp(
                inputs.Y, inputs.inv_pop, inputs.T0, treated_col=0,
                n_factors=cfg.n_factors, n_warmup=cfg.n_warmup,
                n_samples=cfg.n_samples, n_chains=cfg.n_chains,
                target_accept=cfg.target_accept, max_tree_depth=cfg.max_tree_depth,
                seed=cfg.seed, progress=cfg.verbose)
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthEstimationError(f"MTGP estimation failed: {exc}") from exc

        cf = np.asarray(draws["counterfactual"], dtype=float)   # (n_draws, T)
        a = cfg.ci_alpha
        cf_mean = cf.mean(0)
        cf_lo = np.percentile(cf, 100 * a / 2, axis=0)
        cf_hi = np.percentile(cf, 100 * (1 - a / 2), axis=0)

        y_obs = np.asarray(inputs.y_target, dtype=float)
        T0 = inputs.T0
        att_samples = (y_obs[None, T0:] - cf[:, T0:]).mean(axis=1)
        att_lo = float(np.percentile(att_samples, 100 * a / 2))
        att_hi = float(np.percentile(att_samples, 100 * (1 - a / 2)))

        inference = MTGPInference(
            counterfactual_mean=cf_mean, counterfactual_lower=cf_lo,
            counterfactual_upper=cf_hi, att_mean=float(att_samples.mean()),
            att_lower=att_lo, att_upper=att_hi, att_samples=att_samples,
            ci_alpha=a)
        posterior = MTGPPosterior(
            counterfactual=cf, sigma=np.asarray(draws["sigma"], dtype=float),
            n_factors=cfg.n_factors, n_draws=cf.shape[0],
            accept_prob=float(draws["accept_prob"]),
            n_divergent=int(draws["n_divergent"]),
            max_rhat=float(draws["max_rhat"]),
            lengthscale_f=float(draws["lengthscale_f"]),
            lengthscale_global=float(draws["lengthscale_global"]))

        results = MTGPResults(inputs=inputs, posterior=posterior,
                              inference_detail=inference)
        if self.display_graphs:
            try:
                plot_mtgp(results)
            except Exception as exc:  # pragma: no cover - defensive translation
                raise MlsynthPlottingError(f"MTGP plotting failed: {exc}") from exc
        return results
