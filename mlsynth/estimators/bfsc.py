"""Bayesian Factor Synthetic Control (BFSC).

Implements:

    Pinkney, S. (2021). "An Improved and Extended Bayesian Synthetic Control."
    arXiv:2103.16244.

BFSC models the no-intervention outcome of every unit by a Bayesian latent
factor model with year and unit effects and a horseshoe+ prior on the loadings
(so the factor count is a soft upper bound). The treated unit's post-period
outcomes are masked and imputed; the posterior of that imputation is the
counterfactual, giving the treatment effect a full credible band. The posterior
is drawn with NUTS (NumPyro), so BFSC needs the ``[bayes]`` optional dependency.

See ``mlsynth.utils.bfsc_helpers`` for the algorithmic pieces.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import BFSCConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.bfsc_helpers.model import run_bfsc
from ..utils.bfsc_helpers.plotter import plot_bfsc
from ..utils.bfsc_helpers.setup import prepare_bfsc_inputs
from ..utils.bfsc_helpers.structures import (
    BFSCInference,
    BFSCPosterior,
    BFSCResults,
)
from ..utils.datautils import balance


class BFSC:
    """Bayesian Factor Synthetic Control (Pinkney 2021).

    Parameters
    ----------
    config : BFSCConfig or dict
        Configuration object. See :class:`mlsynth.config_models.BFSCConfig`.

    Returns
    -------
    BFSCResults
        Posterior counterfactual with a credible band, the ATT posterior mean +
        credible interval, and NUTS diagnostics. Requires ``mlsynth[bayes]``.
    """

    def __init__(self, config: Union[BFSCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = BFSCConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(f"Invalid BFSC configuration: {exc}") from exc
        self.config: BFSCConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.treat: str = config.treat
        self.display_graphs: bool = config.display_graphs

    def fit(self) -> BFSCResults:
        """Run the BFSC NUTS sampler and assemble results."""
        try:
            balance(self.df, self.unitid, self.time)
        except MlsynthDataError:  # pragma: no cover - defensive passthrough
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthDataError(f"Error balancing panel data: {exc}") from exc

        try:
            inputs = prepare_bfsc_inputs(
                df=self.df, outcome=self.outcome, unitid=self.unitid,
                time=self.time, treat=self.treat)
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthDataError(f"Error preparing BFSC inputs: {exc}") from exc

        cfg = self.config
        try:
            draws = run_bfsc(
                inputs.Y, inputs.T0, n_factors=cfg.n_factors,
                n_warmup=cfg.n_warmup, n_samples=cfg.n_samples,
                n_chains=cfg.n_chains, target_accept=cfg.target_accept,
                seed=cfg.seed, progress=cfg.verbose)
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthEstimationError(f"BFSC estimation failed: {exc}") from exc

        cf = np.asarray(draws["counterfactual"], dtype=float)   # (n_draws, T)
        a = cfg.ci_alpha
        cf_mean = cf.mean(0)
        cf_lo = np.percentile(cf, 100 * a / 2, axis=0)
        cf_hi = np.percentile(cf, 100 * (1 - a / 2), axis=0)

        y_obs = np.asarray(inputs.y_target, dtype=float)
        T0, T = inputs.T0, inputs.T
        att_samples = (y_obs[None, T0:] - cf[:, T0:]).mean(axis=1)
        att_lo = float(np.percentile(att_samples, 100 * a / 2))
        att_hi = float(np.percentile(att_samples, 100 * (1 - a / 2)))

        inference = BFSCInference(
            counterfactual_mean=cf_mean, counterfactual_lower=cf_lo,
            counterfactual_upper=cf_hi, att_mean=float(att_samples.mean()),
            att_lower=att_lo, att_upper=att_hi, att_samples=att_samples,
            ci_alpha=a)
        posterior = BFSCPosterior(
            counterfactual=cf, sigma=np.asarray(draws["sigma"], dtype=float),
            n_factors=cfg.n_factors, n_draws=cf.shape[0],
            accept_prob=float(draws["accept_prob"]),
            n_divergent=int(draws["n_divergent"]),
            max_rhat=float(draws["max_rhat"]))

        results = BFSCResults(inputs=inputs, posterior=posterior,
                              inference_detail=inference)
        if self.display_graphs:
            try:
                plot_bfsc(results)
            except Exception as exc:  # pragma: no cover - defensive translation
                raise MlsynthPlottingError(f"BFSC plotting failed: {exc}") from exc
        return results
