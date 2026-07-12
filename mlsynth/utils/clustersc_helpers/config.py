"""Configuration for the CLUSTERSC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import Field, model_validator
from ...config_models import BaseEstimatorConfig


class CLUSTERSCConfig(BaseEstimatorConfig):
    """Configuration for the Cluster-based Synthetic Control (CLUSTERSC) estimator.

    CLUSTERSC packages two robust synthetic-control families behind a
    single interface:

    * **PCR-RSC** -- SVD-based donor clustering plus Principal Component
      Regression weight estimation (Amjad, Shah, Shen 2018; Agarwal,
      Shah, Shen, Song 2021). Frequentist QP via :mod:`cvxpy` or
      Bayesian Robust Synthetic Control posterior (Amjad, Shah, Shen 2018).
    * **RPCA-SC** -- robust low-rank donor denoising (PCP -- Candes,
      Li, Ma, Wright 2011; or HQF -- Wang, Li, So, Liu 2023) followed
      by simplex SCM weights.

    Parameters
    ----------
    method : {"pcr", "rpca", "both"}
        Estimation family to run. ``"both"`` runs PCR and RPCA in
        parallel; ``primary`` selects which one exposes ``att`` /
        ``counterfactual`` / ``gap`` on the result object.
    primary : {"pcr", "rpca"}
        Which fit drives the result aliases when ``method = "both"``.
    pcr_objective : {"OLS", "SIMPLEX"}
        Inner SCM weight objective for the PCR family.
    clustering : bool
        Whether to apply SVD-based donor clustering before the PCR fit.
    estimator : {"frequentist", "bayesian"}
        Frequentist QP versus Bayesian posterior for the PCR family.
    rpca_method : {"PCP", "HQF"}
        Robust-PCA decomposition for the RPCA family.
    lambda_penalty, p, q : float or None
        Elastic-net-style regularization knobs forwarded to the
        PCR inner solver.
    alpha : float
        Two-sided level for the Bayesian credible interval (PCR
        estimator = "bayesian" only).
    """

    method: Literal["pcr", "rpca", "both"] = Field(
        default="pcr",
        description="Family to run: pcr, rpca, or both.",
    )
    primary: Literal["pcr", "rpca"] = Field(
        default="pcr",
        description="Which fit drives the result aliases when method='both'.",
    )
    pcr_objective: Literal["OLS", "SIMPLEX"] = Field(
        default="OLS",
        description="Inner SCM weight objective for the PCR family.",
    )
    clustering: bool = Field(
        default=True,
        description="Whether to apply SVD-based donor clustering before the PCR fit.",
    )
    estimator: Literal["frequentist", "bayesian"] = Field(
        default="frequentist",
        description="Frequentist QP or Bayesian posterior for the PCR family.",
    )
    rpca_method: Literal["PCP", "HQF"] = Field(
        default="PCP",
        description="Robust-PCA decomposition for the RPCA family.",
    )
    lambda_penalty: Optional[float] = Field(
        default=None, ge=0.0,
        description="Elastic-net regularization strength (lambda).",
    )
    p: Optional[float] = Field(
        default=None, ge=0.0,
        description="Norm parameter p for the elastic-net penalty.",
    )
    q: Optional[float] = Field(
        default=None, ge=0.0,
        description="Secondary norm parameter q (mixed-norm).",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided credible-interval level (Bayesian PCR only).",
    )
    rank: Optional[int] = Field(
        default=None, ge=1,
        description="Explicit HSVT truncation rank `r` (Rho et al. 2025, Algorithm 2). "
                    "If unset, rank_method controls selection.",
    )
    rank_method: Literal["cumvar", "fixed", "usvt"] = Field(
        default="cumvar",
        description="Rank-selection rule: 'cumvar' (paper default; smallest r "
                    "with cumulative spectral energy >= cumvar_threshold), 'fixed' "
                    "(use explicit `rank`), or 'usvt' (Chatterjee 2015 / Donoho-Gavish).",
    )
    cumvar_threshold: float = Field(
        default=0.95, gt=0.0, le=1.0,
        description="Cumulative-variance target when rank_method='cumvar'. "
                    "Paper Section 6.1 uses 0.95.",
    )
    standardize_for_rank: bool = Field(
        default=True,
        description="If True (default), the cumvar / USVT rank rules operate on the "
                    "column-standardised donor matrix so the spectral comparison "
                    "is not dominated by uncentered level information. The HSVT "
                    "step itself still consumes the raw matrix.",
    )
    project_denoised: bool = Field(
        default=False,
        description="If True, project f_hat through the HSVT-denoised donor matrix "
                    "in both pre and post periods (Rho et al. 2025 Algorithm 4 "
                    "Step 5, paper-strict). Default False uses the raw donor "
                    "outcomes for the counterfactual, matching the practical "
                    "behaviour of Amjad-Shah-Shen (2018) and legacy mlsynth.",
    )
    k_clusters: Optional[int] = Field(
        default=None, ge=1,
        description="Number of clusters for Algorithm 3. If unset, the silhouette "
                    "coefficient picks k in [2, k_max].",
    )
    k_max: int = Field(
        default=8, ge=2,
        description="Upper bound for the silhouette-driven k-search.",
    )
    n_bayes_samples: int = Field(
        default=1000, ge=1,
        description="Posterior sample count for Bayesian PCR credible bands.",
    )
    random_state: int = Field(
        default=0,
        description="Seed forwarded to k-means and the Bayesian sampler.",
    )
    compute_shen_ci: bool = Field(
        default=True,
        description="When True (default) and the frequentist OLS PCR path is used, "
                    "compute Shen-Ding-Sekhon-Yu (2023) per-period and ATT confidence "
                    "intervals under three sources of randomness (HZ / VT / DR).",
    )
    shen_variance: Literal["homoskedastic", "jackknife", "hrk"] = Field(
        default="homoskedastic",
        description="Variance estimator for the Shen et al. (2023) CIs. "
                    "'homoskedastic' (paper eq 19, default), 'jackknife', or 'hrk' "
                    "(Hartley-Rao-Kish; only valid when max(1 - diag(H_perp)) < 1/2 "
                    "for both projections).",
    )
    fpca_cumvar: float = Field(
        default=0.95, gt=0.0, le=1.0,
        description="Cumulative-variance target for the FPCA truncation in "
                    "RPCA-SC Step 1 (Bayani 2021 recommends >= 0.95).",
    )
    pcp_lambda: Optional[float] = Field(
        default=None, ge=0.0,
        description="PCP sparsity penalty (Candes et al. 2011). Defaults to "
                    "1/sqrt(max(N, T)) when unset.",
    )
    pcp_mu: Optional[float] = Field(
        default=None, gt=0.0,
        description="PCP augmented-Lagrangian penalty mu. Defaults to "
                    "N*T / (4 * sum |Y|) (Bayani 2021).",
    )
    pcp_max_iter: int = Field(
        default=1000, ge=1,
        description="Maximum ADMM iterations for the PCP solver.",
    )
    pcp_tol: float = Field(
        default=1e-9, gt=0.0,
        description="Convergence tolerance for the PCP solver (relative to ||Y||_F).",
    )
    hqf_rank: Optional[int] = Field(
        default=None, ge=1,
        description="Explicit factorisation rank for HQF. If unset, picked by "
                    "hqf_cumvar (Bayani 2021 uses 0.999).",
    )
    hqf_cumvar: float = Field(
        default=0.999, gt=0.0, le=1.0,
        description="Cumulative-variance target for HQF rank selection when "
                    "hqf_rank is None.",
    )
    hqf_lambda: Optional[float] = Field(
        default=None, ge=0.0,
        description="Tikhonov factor for HQF (Wang et al. 2023). Defaults to "
                    "1/sqrt(max(m, n)).",
    )
    hqf_ip: float = Field(
        default=1.0, gt=0.0,
        description="HQF noise-scale adaptation factor (Bayani 2021 default 1.0).",
    )
    hqf_max_iter: int = Field(
        default=1000, ge=1,
        description="Maximum iterations for the HQF solver.",
    )
    cv_lambda: bool = Field(
        default=False,
        description="Leave-one-time-period-out CV for PCP's sparsity penalty "
                    "lambda. Sweeps cv_lambda_multipliers x Candes default, picks "
                    "the value with lowest held-out NNLS MSE. On the California "
                    "Prop 99 panel this halves pre-RMSE vs the Candes default.",
    )
    cv_hqf_rank: bool = Field(
        default=False,
        description="Leave-one-time-period-out CV for HQF's factorisation rank. "
                    "Sweeps integer ranks in [1, min(J, T0-1)] and picks the rank "
                    "with lowest held-out NNLS MSE.",
    )
    compute_cft_pi: bool = Field(
        default=False,
        description="Cattaneo-Feng-Titiunik (2021) prediction intervals for the "
                    "RPCA-SC fit. Two-component (in-sample bootstrap + out-of-sample "
                    "Hoeffding bound). Default False because it requires `cft_sims` "
                    "full refits of the pipeline (~0.5s each).",
    )
    cft_sims: int = Field(
        default=200, ge=10,
        description="Number of bootstrap draws for the CFT in-sample component.",
    )
    cft_alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided level for the CFT prediction intervals.",
    )
    cft_e_method: Literal["gaussian"] = Field(
        default="gaussian",
        description="Out-of-sample method for the CFT component. Currently "
                    "only 'gaussian' (Hoeffding sub-Gaussian bound) is supported.",
    )
    compute_scpi_pi: bool = Field(
        default=False,
        description="Route the primary fit's prediction intervals through "
                    "VanillaSC's generalized scpi engine (Cattaneo-Feng-Palomba-"
                    "Titiunik 2025). Applied to the fitted weights and the denoised "
                    "donor matrix under `scpi_constraint`. scpi's Table 3 pairs the "
                    "ridge constraint with Amjad et al. (2018) Robust SC (the PCR "
                    "path). Default False (requires `scpi_sims` QCQP simulations).",
    )
    scpi_constraint: Literal["ols", "simplex", "lasso", "ridge", "L1-L2"] = Field(
        default="ridge",
        description="scpi weight-constraint family for `compute_scpi_pi`. Match it "
                    "to the fit: 'ridge' for RSC/PCR-OLS (Table 3, default), "
                    "'simplex' for the RPCA / SIMPLEX weights, 'ols' / 'lasso' / "
                    "'L1-L2' otherwise.",
    )
    scpi_sims: int = Field(
        default=200, ge=10,
        description="Gaussian draws for the scpi in-sample QCQP simulation.",
    )
    scpi_e_method: Literal["gaussian", "ls", "empirical"] = Field(
        default="gaussian",
        description="Out-of-sample tabulation for the scpi prediction intervals.",
    )
    plot_bands: Literal["pointwise", "simultaneous", "both"] = Field(
        default="pointwise",
        description="Which scpi prediction-interval band(s) to shade on the "
                    "observed-vs-counterfactual plot when `compute_scpi_pi` is "
                    "set: 'pointwise' (default, the per-period band), "
                    "'simultaneous' (the joint-coverage band), or 'both'. No "
                    "band is drawn when scpi intervals were not computed.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalise_legacy_fields(cls, data: Any) -> Any:
        """Accept the legacy field names and uppercase method strings."""
        if not isinstance(data, dict):
            return data
        # objective -> pcr_objective
        if "objective" in data and "pcr_objective" not in data:
            data["pcr_objective"] = data.pop("objective")
        else:
            data.pop("objective", None)
        # cluster -> clustering
        if "cluster" in data and "clustering" not in data:
            data["clustering"] = data.pop("cluster")
        else:
            data.pop("cluster", None)
        # Frequentist -> estimator
        if "Frequentist" in data and "estimator" not in data:
            data["estimator"] = (
                "frequentist" if data.pop("Frequentist") else "bayesian"
            )
        else:
            data.pop("Frequentist", None)
        # ROB / Robust -> rpca_method
        for k in ("ROB", "Robust"):
            if k in data and "rpca_method" not in data:
                data["rpca_method"] = data.pop(k)
            else:
                data.pop(k, None)
        # Uppercase method tags
        if "method" in data and isinstance(data["method"], str):
            mapping = {"PCR": "pcr", "RPCA": "rpca", "BOTH": "both"}
            data["method"] = mapping.get(data["method"], data["method"])
        return data
