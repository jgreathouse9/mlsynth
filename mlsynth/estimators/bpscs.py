"""Bayesian Penalized Synthetic Control under Spillovers (BPSCS).

Implements the utility-based shrinkage priors of:

    Fernandez-Morales, E., Oganisian, A., & Lee, Y. (2026). "Bayesian shrinkage
    priors for penalized synthetic control estimators in the presence of
    spillovers." Biometrics 82(2), ujag054.

BPSCS models the treated unit's no-intervention outcome with an autoregressive
linear synthetic control whose donor coefficients carry a shrinkage prior. The
prior scale for each donor is set by a utility blending covariate similarity and
spatial distance to the treated unit, so spatially close (likely spillover-
contaminated) donors are shrunk toward zero rather than excluded. Two priors are
offered: ``dhs`` (distance-horseshoe) and ``ds2`` (distance-spike-and-slab). The
counterfactual is a free-running forward simulation of the treated series; its
posterior gives a credible band that widens through the post-period. Point
summaries use the posterior median (robust to the free-running tail). The
posterior is drawn with NUTS (NumPyro, double precision), so BPSCS needs the
``[bayes]`` optional dependency.

See ``mlsynth.utils.bpscs_helpers`` for the algorithmic pieces.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import BPSCSConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.bpscs_helpers.model import run_bpscs
from ..utils.bpscs_helpers.plotter import plot_bpscs
from ..utils.bpscs_helpers.setup import prepare_bpscs_inputs
from ..utils.bpscs_helpers.structures import (
    BPSCSInference,
    BPSCSPosterior,
    BPSCSResults,
)
from ..utils.datautils import balance


class BPSCS:
    """Bayesian Penalized Synthetic Control under Spillovers (Fernandez-Morales et al. 2026).

    Parameters
    ----------
    config : BPSCSConfig or dict
        Configuration object. See :class:`mlsynth.config_models.BPSCSConfig`.

    Returns
    -------
    BPSCSResults
        Posterior counterfactual with a credible band, the ATT posterior median
        + credible interval, the (signed) donor coefficients, and NUTS
        diagnostics. Requires ``mlsynth[bayes]``.
    """

    def __init__(self, config: Union[BPSCSConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = BPSCSConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(f"Invalid BPSCS configuration: {exc}") from exc
        self.config: BPSCSConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.treat: str = config.treat
        self.display_graphs: bool = config.display_graphs

    def fit(self) -> BPSCSResults:
        """Run the BPSCS NUTS sampler and assemble results."""
        try:
            balance(self.df, self.unitid, self.time)
        except MlsynthDataError:  # pragma: no cover - defensive passthrough
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthDataError(f"Error balancing panel data: {exc}") from exc

        cfg = self.config
        try:
            inputs = prepare_bpscs_inputs(
                df=self.df, outcome=self.outcome, unitid=self.unitid,
                time=self.time, treat=self.treat, covariates=cfg.covariates,
                coords=cfg.coords, kappa_d=cfg.kappa_d,
                inclusion_quantile=cfg.inclusion_quantile, prior=cfg.prior)
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthDataError(f"Error preparing BPSCS inputs: {exc}") from exc

        try:
            draws = run_bpscs(
                inputs.Y_std, inputs.X_cov, inputs.donor_d, inputs.rho, inputs.T0,
                cfg.prior, n_warmup=cfg.n_warmup, n_samples=cfg.n_samples,
                n_chains=cfg.n_chains, target_accept=cfg.target_accept,
                max_tree_depth=cfg.max_tree_depth, seed=cfg.seed, progress=cfg.verbose)
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthEstimationError(f"BPSCS estimation failed: {exc}") from exc

        # counterfactual back on the treated unit's original scale
        cf_std = np.asarray(draws["counterfactual"], dtype=float)      # (n, T)
        cf = cf_std * inputs.stdv0 + inputs.mean_pre_treated
        a = cfg.ci_alpha
        cf_med = np.median(cf, axis=0)
        cf_lo = np.percentile(cf, 100 * a / 2, axis=0)
        cf_hi = np.percentile(cf, 100 * (1 - a / 2), axis=0)

        y_obs = np.asarray(inputs.y_target, dtype=float)
        T0 = inputs.T0
        tau = y_obs[None, :] - cf                                      # (n, T)
        att_samples = tau[:, T0:].mean(axis=1)
        att_med = float(np.median(att_samples))
        att_lo = float(np.percentile(att_samples, 100 * a / 2))
        att_hi = float(np.percentile(att_samples, 100 * (1 - a / 2)))

        inference = BPSCSInference(
            counterfactual_median=cf_med, counterfactual_lower=cf_lo,
            counterfactual_upper=cf_hi, att_median=att_med, att_lower=att_lo,
            att_upper=att_hi, att_samples=att_samples, ci_alpha=a)
        posterior = BPSCSPosterior(
            counterfactual=cf, beta=np.asarray(draws["beta"], dtype=float),
            sigma=np.asarray(draws["sigma"], dtype=float),
            psi=np.asarray(draws["psi"], dtype=float), n_draws=int(draws["n_draws"]),
            accept_prob=float(draws["accept_prob"]),
            n_divergent=int(draws["n_divergent"]),
            max_rhat=float(draws["max_rhat"]), prior=cfg.prior, rho=float(inputs.rho),
            n_included=int(draws["n_included"]))

        results = BPSCSResults(inputs=inputs, posterior=posterior,
                              inference_detail=inference)
        if self.display_graphs:
            try:
                plot_bpscs(results)
            except Exception as exc:  # pragma: no cover - defensive translation
                raise MlsynthPlottingError(f"BPSCS plotting failed: {exc}") from exc
        return results
