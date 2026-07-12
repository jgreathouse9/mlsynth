"""Cluster-based Synthetic Control (CLUSTERSC) estimator.

Bundles two complementary robust SCM families behind a single
orchestrator:

* **PCR-RSC** -- SVD-based donor clustering (Amjad, Shah, Shen 2018)
  plus Principal Component Regression weight estimation (Agarwal,
  Shah, Shen, Song 2021). Frequentist QP or Bayesian Robust Synthetic Control posterior
  (Amjad, Shah, Shen 2018) selected via ``estimator``.
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

from ..config_models import (
    CLUSTERSCConfig,
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
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
        self.standardize_for_rank: bool = config.standardize_for_rank
        self.project_denoised: bool = config.project_denoised
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
        self.cv_lambda: bool = config.cv_lambda
        self.cv_hqf_rank: bool = config.cv_hqf_rank
        self.compute_shen_ci: bool = config.compute_shen_ci
        self.shen_variance: str = config.shen_variance
        self.compute_cft_pi: bool = config.compute_cft_pi
        self.cft_sims: int = config.cft_sims
        self.cft_alpha: float = config.cft_alpha
        self.cft_e_method: str = config.cft_e_method
        self.compute_scpi_pi: bool = config.compute_scpi_pi
        self.scpi_constraint: str = config.scpi_constraint
        self.scpi_sims: int = config.scpi_sims
        self.scpi_e_method: str = config.scpi_e_method
        self.display_graphs: bool = config.display_graphs

    def _standardize_results(
        self, *, inputs, pcr_fit, rpca_fit, primary_fit, selected,
        cluster_inference,
    ) -> CLUSTERSCResults:
        """Assemble the standardized EffectResult from the primary variant."""
        # Scalar inference summary mirrored into the standardized slot.
        ci_lower = ci_upper = std_err = None
        if cluster_inference.scpi is not None:
            lo, hi = cluster_inference.scpi.att_pi
            ci_lower, ci_upper = float(lo), float(hi)
        elif cluster_inference.method == "bayesian_credible":
            lo, hi = cluster_inference.credible_interval
            if np.isfinite(lo) and np.isfinite(hi):
                ci_lower, ci_upper = float(lo), float(hi)
        elif cluster_inference.shen is not None:
            lo, hi = cluster_inference.shen.att_ci_dr
            ci_lower, ci_upper = float(lo), float(hi)
            std_err = float(cluster_inference.shen.att_se_dr)
        elif cluster_inference.cft is not None:
            lo, hi = cluster_inference.cft.att_pi
            ci_lower, ci_upper = float(lo), float(hi)

        std_inference = InferenceResults(
            method=cluster_inference.method,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            standard_error=std_err,
            confidence_level=1.0 - float(self.alpha),
        )

        att = float(primary_fit.att) if primary_fit is not None else None
        effects = EffectsResults(
            att=att,
            additional_effects={
                "pcr_att": float(pcr_fit.att) if pcr_fit is not None else None,
                "rpca_att": float(rpca_fit.att) if rpca_fit is not None else None,
            },
        )
        time_series = TimeSeriesResults(
            observed_outcome=np.asarray(inputs.treated_outcome, dtype=float),
            counterfactual_outcome=(
                np.asarray(primary_fit.counterfactual, dtype=float)
                if primary_fit is not None else None),
            estimated_gap=(
                np.asarray(primary_fit.gap, dtype=float)
                if primary_fit is not None else None),
            time_periods=np.asarray(inputs.time_labels),
            intervention_time=(
                inputs.time_labels[inputs.T0] if inputs.T0 < inputs.T else None),
        )
        donor_weights = (
            {str(k): float(v) for k, v in primary_fit.donor_weights.items()}
            if primary_fit is not None else None)
        weights = WeightsResults(
            donor_weights=donor_weights,
            summary_stats=(
                {"n_selected_donors": int(len(primary_fit.selected_donors))}
                if primary_fit is not None else None),
        )
        fit_diagnostics = FitDiagnosticsResults(
            rmse_pre=float(primary_fit.pre_rmse) if primary_fit is not None else None,
        )
        method_name = (f"PCR-RSC ({self.estimator})" if selected == "pcr"
                       else f"RPCA-SC ({self.rpca_method})")
        method_details = MethodDetailsResults(
            method_name=method_name,
            parameters_used={
                "method": self.method, "clustering": self.clustering,
                "pcr_objective": self.pcr_objective, "rank_method": self.rank_method,
            },
        )

        return CLUSTERSCResults(
            effects=effects,
            time_series=time_series,
            weights=weights,
            inference=std_inference,
            fit_diagnostics=fit_diagnostics,
            method_details=method_details,
            inputs=inputs,
            pcr=pcr_fit,
            rpca=rpca_fit,
            selected_variant=selected,
            cluster_inference=cluster_inference,
            metadata={
                "method": self.method,
                "primary": self.primary,
                "estimator": self.estimator,
                "rpca_method": self.rpca_method,
                "pcr_objective": self.pcr_objective,
                "clustering": self.clustering,
            },
        )

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
                    standardize_for_rank=self.standardize_for_rank,
                    project_denoised=self.project_denoised,
                    k_clusters=self.k_clusters,
                    k_max=self.k_max,
                    alpha=self.alpha,
                    n_bayes_samples=self.n_bayes_samples,
                    lambda_penalty=self.lambda_penalty,
                    p=self.p,
                    q=self.q,
                    compute_shen_ci=self.compute_shen_ci,
                    shen_variance=self.shen_variance,
                    compute_scpi_pi=self.compute_scpi_pi,
                    scpi_constraint=self.scpi_constraint,
                    scpi_sims=self.scpi_sims,
                    scpi_e_method=self.scpi_e_method,
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
                    cv_lambda=self.cv_lambda,
                    cv_hqf_rank=self.cv_hqf_rank,
                    compute_cft_pi=self.compute_cft_pi,
                    cft_alpha=self.cft_alpha,
                    cft_sims=self.cft_sims,
                    cft_e_method=self.cft_e_method,
                    compute_scpi_pi=self.compute_scpi_pi,
                    scpi_constraint=self.scpi_constraint,
                    scpi_sims=self.scpi_sims,
                    scpi_e_method=self.scpi_e_method,
                    scpi_alpha=self.alpha,
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

            # scpi prediction intervals (opt-in) surface as the primary inference
            # when computed for the selected fit -- the user asked for them.
            scpi_obj = None
            if selected == "pcr" and pcr_fit is not None:
                scpi_obj = pcr_fit.metadata.get("scpi_inference")
            elif selected == "rpca" and rpca_fit is not None:
                scpi_obj = rpca_fit.metadata.get("scpi_inference")

            if scpi_obj is not None:
                cluster_inference = CLUSTERSCInference(
                    method=scpi_obj.method,
                    alpha=float(self.alpha),
                    att=primary_att,
                    scpi=scpi_obj,
                )
            elif credible is not None and self.estimator == "bayesian":
                cluster_inference = CLUSTERSCInference(
                    method="bayesian_credible",
                    alpha=float(self.alpha),
                    att=primary_att,
                    credible_interval=credible,
                )
            else:
                # If frequentist OLS PCR ran, surface its Shen et al. CIs.
                shen_obj = (
                    pcr_fit.metadata.get("shen_inference")
                    if pcr_fit is not None
                    else None
                )
                # If RPCA-SC computed CFT prediction intervals, surface those.
                cft_obj = (
                    rpca_fit.metadata.get("cft_inference")
                    if rpca_fit is not None
                    else None
                )
                if shen_obj is not None and selected == "pcr":
                    cluster_inference = CLUSTERSCInference(
                        method=shen_obj.method,
                        alpha=float(self.alpha),
                        att=primary_att,
                        shen=shen_obj,
                    )
                elif cft_obj is not None and selected == "rpca":
                    cluster_inference = CLUSTERSCInference(
                        method=cft_obj.method,
                        alpha=float(self.alpha),
                        att=primary_att,
                        cft=cft_obj,
                    )
                else:
                    cluster_inference = CLUSTERSCInference(
                        method="none",
                        alpha=float(self.alpha),
                        att=primary_att,
                    )

            # ----------------------------------------------------------------
            # Populate the standardized two-family result contract from the
            # primary variant (see agents/agents_results.md). The flat
            # accessors (att / counterfactual / gap / att_ci / donor_weights /
            # pre_rmse) then resolve through BaseEstimatorResults.
            # ----------------------------------------------------------------
            primary_fit = pcr_fit if selected == "pcr" else rpca_fit
            results = self._standardize_results(
                inputs=inputs, pcr_fit=pcr_fit, rpca_fit=rpca_fit,
                primary_fit=primary_fit, selected=selected,
                cluster_inference=cluster_inference,
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
