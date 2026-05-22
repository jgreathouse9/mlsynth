"""Cluster-based Synthetic Control (CLUSTERSC) estimator.

Bundles two complementary robust SCM families behind a single
orchestrator:

* **PCR-RSC** -- SVD-based donor clustering (Amjad, Shah, Shen 2018)
  plus Principal Component Regression weight estimation (Agarwal,
  Shah, Shen, Song 2021). Frequentist QP or Bayesian posterior
  (Bayani 2022, CUNY dissertation Ch. 1) selected via ``estimator``.
* **RPCA-SC** -- robust low-rank donor denoising (PCP -- Candes, Li,
  Ma, Wright 2011; or HQF -- Wang, Li, So, Liu 2023) followed by
  simplex SCM weights.

``method = "both"`` runs both families in parallel and exposes them
side by side on the :class:`CLUSTERSCResults` container. The
``primary`` field selects which fit drives the convenience aliases
``att`` / ``counterfactual`` / ``gap`` / ``donor_weights``.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import CLUSTERSCConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.clustersc_helpers.pcr import run_pcr
from ..utils.clustersc_helpers.plotter import plot_clustersc
from ..utils.clustersc_helpers.rpca import run_rpca
from ..utils.clustersc_helpers.setup import prepare_clustersc_inputs
from ..utils.clustersc_helpers.structures import (
    CLUSTERSCInference,
    CLUSTERSCResults,
)
from ..utils.datautils import balance


class CLUSTERSC:
    """Cluster-based Synthetic Control estimator.

    Parameters
    ----------
    config : CLUSTERSCConfig or dict
        Configuration object. See
        :class:`mlsynth.config_models.CLUSTERSCConfig`.

    Returns
    -------
    CLUSTERSCResults
        Frozen container with optional PCR-RSC and RPCA-SC fits plus
        Bayesian credible interval inference (PCR estimator =
        "bayesian" only).
    """

    def __init__(self, config: Union[CLUSTERSCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = CLUSTERSCConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid CLUSTERSC configuration: {exc}"
                ) from exc

        self.config: CLUSTERSCConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.method: str = config.method
        self.primary: str = config.primary
        self.pcr_objective: str = config.pcr_objective
        self.clustering: bool = config.clustering
        self.estimator: str = config.estimator
        self.rpca_method: str = config.rpca_method
        self.lambda_penalty = config.lambda_penalty
        self.p = config.p
        self.q = config.q
        self.alpha: float = config.alpha
        self.rank = config.rank
        self.rank_method: str = config.rank_method
        self.cumvar_threshold: float = config.cumvar_threshold
        self.k_clusters = config.k_clusters
        self.k_max: int = config.k_max
        self.n_bayes_samples: int = config.n_bayes_samples
        self.random_state: int = config.random_state
        self.fpca_cumvar: float = config.fpca_cumvar
        self.pcp_lambda = config.pcp_lambda
        self.pcp_mu = config.pcp_mu
        self.pcp_max_iter: int = config.pcp_max_iter
        self.pcp_tol: float = config.pcp_tol
        self.hqf_rank = config.hqf_rank
        self.hqf_cumvar: float = config.hqf_cumvar
        self.hqf_lambda = config.hqf_lambda
        self.hqf_ip: float = config.hqf_ip
        self.hqf_max_iter: int = config.hqf_max_iter
        self.display_graphs: bool = config.display_graphs

    def fit(self) -> CLUSTERSCResults:
        """Run the requested family (or both) and return a :class:`CLUSTERSCResults`."""

        try:
            balance(self.df, self.unitid, self.time)
            inputs = prepare_clustersc_inputs(
                df=self.df,
                outcome=self.outcome,
                treat=self.treat,
                unitid=self.unitid,
                time=self.time,
            )

            pcr_fit = None
            rpca_fit = None
            credible = None

            if self.method in {"pcr", "both"}:
                pcr_fit, credible_band = run_pcr(
                    treated_outcome=inputs.treated_outcome,
                    donor_outcomes=inputs.donor_outcomes,
                    donor_names=inputs.donor_names,
                    T0=inputs.T0,
                    objective=self.pcr_objective,
                    clustering=self.clustering,
                    estimator=self.estimator,
                    rank=self.rank,
                    rank_method=self.rank_method,
                    cumvar_threshold=self.cumvar_threshold,
                    k_clusters=self.k_clusters,
                    k_max=self.k_max,
                    alpha=self.alpha,
                    n_bayes_samples=self.n_bayes_samples,
                    lambda_penalty=self.lambda_penalty,
                    p=self.p,
                    q=self.q,
                    random_state=self.random_state,
                )
                # Convert per-period credible band to an ATT-level interval:
                # tau_t = y_t - cf_t, so a (1-alpha) CI for ATT is
                # (y_post_mean - cf_upper_post_mean, y_post_mean - cf_lower_post_mean).
                credible = None
                if credible_band is not None and inputs.T > inputs.T0:
                    cf_lo, cf_hi = credible_band
                    y_post_mean = float(np.mean(inputs.treated_outcome[inputs.T0:]))
                    credible = (
                        y_post_mean - float(np.mean(cf_hi[inputs.T0:])),
                        y_post_mean - float(np.mean(cf_lo[inputs.T0:])),
                    )

            if self.method in {"rpca", "both"}:
                rpca_fit = run_rpca(
                    treated_outcome=inputs.treated_outcome,
                    donor_outcomes=inputs.donor_outcomes,
                    donor_names=inputs.donor_names,
                    T0=inputs.T0,
                    rpca_method=self.rpca_method,
                    fpca_cumvar=self.fpca_cumvar,
                    k_clusters=self.k_clusters,
                    k_max=self.k_max,
                    pcp_lambda=self.pcp_lambda,
                    pcp_mu=self.pcp_mu,
                    pcp_max_iter=self.pcp_max_iter,
                    pcp_tol=self.pcp_tol,
                    hqf_rank=self.hqf_rank,
                    hqf_cumvar=self.hqf_cumvar,
                    hqf_lambda=self.hqf_lambda,
                    hqf_ip=self.hqf_ip,
                    hqf_max_iter=self.hqf_max_iter,
                    random_state=self.random_state,
                )

            # Select primary fit. If the user picked "pcr" / "rpca" only,
            # primary is unambiguous; if "both", honour config.primary.
            if self.method == "pcr":
                selected = "pcr"
            elif self.method == "rpca":
                selected = "rpca"
            else:
                selected = self.primary

            primary_att = (
                pcr_fit.att if selected == "pcr" and pcr_fit is not None
                else rpca_fit.att if rpca_fit is not None
                else float("nan")
            )

            if credible is not None and self.estimator == "bayesian":
                inference = CLUSTERSCInference(
                    method="bayesian_credible",
                    alpha=float(self.alpha),
                    att=primary_att,
                    credible_interval=credible,
                )
            else:
                inference = CLUSTERSCInference(
                    method="none",
                    alpha=float(self.alpha),
                    att=primary_att,
                )

            results = CLUSTERSCResults(
                inputs=inputs,
                pcr=pcr_fit,
                rpca=rpca_fit,
                inference=inference,
                selected_variant=selected,
                metadata={
                    "method": self.method,
                    "primary": self.primary,
                    "estimator": self.estimator,
                    "rpca_method": self.rpca_method,
                    "pcr_objective": self.pcr_objective,
                    "clustering": self.clustering,
                },
            )

            if self.display_graphs:
                try:
                    plot_clustersc(results)
                except Exception as exc:
                    raise MlsynthPlottingError(
                        f"CLUSTERSC plotting failed: {exc}"
                    ) from exc

            return results

        except (
            MlsynthConfigError,
            MlsynthDataError,
            MlsynthEstimationError,
            MlsynthPlottingError,
        ):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"CLUSTERSC estimation failed: {exc}"
            ) from exc
