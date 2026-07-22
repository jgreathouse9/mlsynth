"""Bayesian synthetic control of Martinez & Vives-i-Bastida (MVBBSC).

Implements:

    Martinez, I. & Vives-i-Bastida, J. (2024). "Bayesian and Frequentist
    Inference for Synthetic Controls." arXiv:2206.01779.

MVBBSC models the treated unit's no-intervention outcome as a simplex-weighted
average of the donor pool with a uniform Dirichlet prior on the weights, a
``HalfNormal`` idiosyncratic scale, and a Gaussian likelihood on the
pre-treatment window. Both the treated and donor series are standardized by
their pre-period moments before fitting and the counterfactual is transformed
back to the outcome scale, so the fit is invariant to the units of the outcome.
The posterior is drawn with NUTS (NumPyro); the counterfactual is the
posterior-predictive draw of the untreated outcome, giving a full credible band
and an ATT credible interval. MVBBSC therefore needs the ``[bayes]`` optional
dependency.

See ``mlsynth.utils.mvbbsc_helpers`` for the algorithmic pieces.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import MVBBSCConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.datautils import balance
from ..utils.mvbbsc_helpers.model import run_mvbbsc
from ..utils.mvbbsc_helpers.plotter import plot_mvbbsc
from ..utils.mvbbsc_helpers.setup import prepare_mvbbsc_inputs
from ..utils.mvbbsc_helpers.structures import (
    MVBBSCInference,
    MVBBSCPosterior,
    MVBBSCResults,
)


class MVBBSC:
    """Bayesian synthetic control of Martinez & Vives-i-Bastida (2024).

    Parameters
    ----------
    config : MVBBSCConfig or dict
        Configuration object. See :class:`mlsynth.config_models.MVBBSCConfig`.

    Returns
    -------
    MVBBSCResults
        Posterior counterfactual with a credible band, the ATT posterior mean +
        credible interval, the posterior-mean simplex donor weights, and NUTS
        diagnostics. Requires ``mlsynth[bayes]``.
    """

    def __init__(self, config: Union[MVBBSCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = MVBBSCConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid MVBBSC configuration: {exc}"
                ) from exc
        self.config: MVBBSCConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.treat: str = config.treat
        self.display_graphs: bool = config.display_graphs

    def fit(self) -> MVBBSCResults:
        """Run the MVBBSC NUTS sampler and assemble results."""
        try:
            balance(self.df, self.unitid, self.time)
        except MlsynthDataError:  # pragma: no cover - defensive passthrough
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthDataError(f"Error balancing panel data: {exc}") from exc

        try:
            inputs = prepare_mvbbsc_inputs(
                df=self.df, outcome=self.outcome, unitid=self.unitid,
                time=self.time, treat=self.treat)
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthDataError(f"Error preparing MVBBSC inputs: {exc}") from exc

        cfg = self.config
        try:
            draws = run_mvbbsc(
                inputs.y_target, inputs.X_all, inputs.T0,
                n_warmup=cfg.n_warmup, n_samples=cfg.n_samples,
                n_chains=cfg.n_chains, target_accept=cfg.target_accept,
                seed=cfg.seed, progress=cfg.verbose)
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthEstimationError(f"MVBBSC estimation failed: {exc}") from exc

        cf = np.asarray(draws["counterfactual"], dtype=float)   # (n_draws, T)
        a = cfg.ci_alpha
        cf_mean = cf.mean(0)
        cf_lo = np.percentile(cf, 100 * a / 2, axis=0)
        cf_hi = np.percentile(cf, 100 * (1 - a / 2), axis=0)

        y_obs = np.asarray(inputs.y_target, dtype=float)
        T0, T = inputs.T0, inputs.T
        if T0 < T:
            att_samples = (y_obs[None, T0:] - cf[:, T0:]).mean(axis=1)
            att_mean = float(att_samples.mean())
            att_lo = float(np.percentile(att_samples, 100 * a / 2))
            att_hi = float(np.percentile(att_samples, 100 * (1 - a / 2)))
        else:  # pragma: no cover - dataprep guarantees a post window here
            att_samples = np.array([])
            att_mean = att_lo = att_hi = float("nan")

        inference = MVBBSCInference(
            counterfactual_mean=cf_mean, counterfactual_lower=cf_lo,
            counterfactual_upper=cf_hi, att_mean=att_mean,
            att_lower=att_lo, att_upper=att_hi, att_samples=att_samples,
            ci_alpha=a)

        W = np.asarray(draws["weights"], dtype=float)           # (n_draws, N)
        weight_means = {
            str(inputs.donor_names[i]): float(W[:, i].mean())
            for i in range(inputs.N)
        }
        posterior = MVBBSCPosterior(
            counterfactual=cf, weights=W,
            sigma=np.asarray(draws["sigma"], dtype=float),
            n_draws=cf.shape[0], accept_prob=float(draws["accept_prob"]),
            n_divergent=int(draws["n_divergent"]),
            max_rhat=float(draws["max_rhat"]))

        results = MVBBSCResults(inputs=inputs, posterior=posterior,
                                inference_detail=inference,
                                weight_means=weight_means)
        if self.display_graphs:
            try:
                plot_mvbbsc(results)
            except Exception as exc:  # pragma: no cover - defensive translation
                raise MlsynthPlottingError(f"MVBBSC plotting failed: {exc}") from exc
        return results
