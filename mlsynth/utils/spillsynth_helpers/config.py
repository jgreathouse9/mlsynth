"""Configuration for the SPILLSYNTH estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import Field, model_validator
from ...exceptions import MlsynthConfigError
from ...config_models import BaseEstimatorConfig


class SPILLSYNTHConfig(BaseEstimatorConfig):
    """Configuration for the SPILLSYNTH estimator.

    Spillover-aware synthetic-control wrapper. Currently ships the
    Cao & Dowd (2023) estimator under ``method='cd'``: a Ferman-Pinto
    demeaned SCM is fit leave-one-out for every unit, and the
    treatment-and-spillover effect vector is recovered jointly under
    a user-specified spillover structure (Example 3 of the paper:
    each declared affected unit gets its own free spillover
    coefficient). Additional methods (e.g. iSCM, distance-decay
    spillover, P-test inference) will be added behind the same
    ``method`` dispatcher.

    Parameters
    ----------
    method : {"cd"}
        Estimation method. Only ``"cd"`` (Cao-Dowd 2023) is
        implemented for now.
    affected_units : list, optional
        Labels (matching values of ``unitid``) of control units the
        researcher believes are potentially exposed to spillover from
        the treated unit. Each gets its own free spillover coefficient
        in the resulting estimator. If ``None`` (the default), no
        affected units are declared and the estimator reduces to
        vanilla demeaned SCM. The treated unit must NOT appear in this
        list.
    solver : str, optional
        ``cvxpy`` solver name used for the leave-one-out simplex SCM
        fits. Defaults to ``"CLARABEL"``.

    References
    ----------
    Cao, J., & Dowd, C. (2023). *Estimation and Inference for
    Synthetic Control Methods with Spillover Effects.* Working paper.

    Ferman, B., & Pinto, C. (2021). *Synthetic Controls with Imperfect
    Pretreatment Fit.* Quantitative Economics, 12(4), 1197-1221.
    """

    method: Literal["cd", "iscm", "grossi", "sar", "iterative"] = Field(
        default="cd",
        description="Spillover-aware SCM method. 'cd' = Cao & Dowd (2023); "
                    "'iscm' = inclusive SCM (Di Stefano & Mellace 2024); "
                    "'grossi' = Grossi et al. (2025) direct + spillover effects "
                    "under partial interference (penalized SC on far controls); "
                    "'sar' = Sakaguchi & Tagawa (2026) spatial-autoregressive "
                    "Bayesian SCM (relaxes SUTVA via a SAR model on the control "
                    "outcomes; needs spatial_W/spatial_w); "
                    "'iterative' = Melnychuk (2024) iterative 'waterfall' SCM "
                    "(clean each affected donor's post outcomes with its own "
                    "spillover-free synthetic, then refit the treated unit on "
                    "the cleaned pool). "
                    "For 'grossi'/'iterative', affected_units lists the spillover-"
                    "exposed controls.",
    )
    spatial_W: Optional[Any] = Field(
        default=None,
        description="(method='sar') Control-to-control spatial-weight matrix. "
                    "A labelled (N x N) pandas DataFrame indexed by control unit "
                    "label (aligned automatically), or a bare N x N array in "
                    "control-label order. Row-normalised internally.",
    )
    spatial_w: Optional[Any] = Field(
        default=None,
        description="(method='sar') Treated-to-control spatial-weight vector "
                    "(length N): a pandas Series/DataFrame keyed by unit label, "
                    "a dict, or a bare array in control-label order. Normalised "
                    "to sum to one internally.",
    )
    p_factors: int = Field(
        default=1, ge=0,
        description="(method='sar') Number of AR(1) latent factors in the SAR "
                    "panel model. 0 disables the factor block.",
    )
    mcmc_iter: int = Field(
        default=6000, ge=2,
        description="(method='sar') Total MCMC iterations per step.",
    )
    mcmc_burn: int = Field(
        default=2000, ge=1,
        description="(method='sar') Burn-in iterations per step.",
    )
    step_rho: float = Field(
        default=0.02, gt=0.0,
        description="(method='sar') Random-walk Metropolis step for rho.",
    )
    mcmc_seed: int = Field(
        default=0, description="(method='sar') RNG seed for the sampler.",
    )
    propagate_alpha: bool = Field(
        default=True,
        description="(method='sar') If True (default) the credible bands pair "
                    "each rho draw with a synthetic-weight (alpha) draw, "
                    "propagating posterior uncertainty in alpha -- the "
                    "statistically correct interval that attains nominal "
                    "coverage in simulation. If False, alpha is held at its "
                    "posterior mean when sweeping rho (narrower bands; the "
                    "authors' empirical convention). Point estimates are "
                    "unaffected.",
    )
    covariates: Optional[List[str]] = Field(
        default=None,
        description="(method='iscm' only) Covariate columns to match on, "
                    "averaged over the pre-treatment period and fed to the "
                    "bilevel predictor-matching solver. When None, the "
                    "inclusive SCM matches on pre-treatment outcomes only.",
    )
    bilevel_solver: Literal["malo", "mscmt", "penalized"] = Field(
        default="malo",
        description="(method='iscm' with covariates) Bilevel backend for "
                    "predictor matching: 'malo' (Malo et al. 2024 corner "
                    "search), 'mscmt' (Becker-Kloessner 2018 global "
                    "differential-evolution search), or 'penalized' "
                    "(Abadie-L'Hour 2021 pairwise-penalized estimator with "
                    "leave-out lambda selection -- a unique, sparse solution).",
    )
    covariate_windows: Optional[Dict[Any, Any]] = Field(
        default=None,
        description="(method='iscm'/'grossi') Per-covariate averaging window as "
                    "an inclusive (start, end) range of time labels, e.g. "
                    "{'invest': (1964, 1969), 'popdens': (1969, 1969)} "
                    "(Abadie's special-predictor spec). Covariates not listed "
                    "are averaged over the full pre-treatment period.",
    )
    bias_correct: bool = Field(
        default=False,
        description="(method='iscm'/'grossi') Apply the Abadie-L'Hour (2021) "
                    "bias correction to each unit's synthetic-control gap, "
                    "removing the part attributable to residual covariate "
                    "imbalance via a ridge regression of the outcome on the "
                    "covariates. Requires 'covariates'; most useful when the "
                    "covariates genuinely explain the outcome.",
    )
    iscm_intercept: bool = Field(
        default=False,
        description="(method='iscm', outcome-only) Fit a demeaned simplex SCM "
                    "with an unpenalised level shift (the SCM-with-intercept of "
                    "Doudchenko-Imbens 2016, and the backend of Di Stefano & "
                    "Mellace's inclusive-SCM reference). Each series is centred "
                    "by its own pre-period mean before the simplex fit; the "
                    "fitted intercept is added back. Tends to give the affected "
                    "neighbour a larger weight than the plain simplex. Ignored "
                    "in covariate mode.",
    )
    n_boot: int = Field(
        default=0, ge=0,
        description="(method='grossi') Residual-resampling draws for the "
                    "pivotal bias-corrected confidence intervals (Grossi et "
                    "al. eqs. 3.6-3.7). 0 (default) skips inference.",
    )
    ci_level: float = Field(
        default=0.90, gt=0.0, lt=1.0,
        description="(method='grossi') Confidence level for the residual-"
                    "resampling intervals. The paper uses 0.90.",
    )
    seed: int = Field(
        default=0,
        description="(method='grossi') RNG seed for residual resampling.",
    )
    affected_units: Optional[List[Any]] = Field(
        default=None,
        description=(
            "Labels of control units potentially exposed to spillover. "
            "Required for spillover_structure='per_unit' (when p > 0) "
            "and 'homogeneous'. Ignored for 'distance_decay' (which "
            "uses unit_distances). The treated unit must NOT appear here."
        ),
    )
    spillover_structure: Literal["per_unit", "homogeneous", "distance_decay"] = Field(
        default="per_unit",
        description=(
            "A-matrix construction (Cao-Dowd v3 Examples 1/2/3): "
            "'per_unit' (Example 1, default, each affected unit gets its "
            "own free coefficient), 'homogeneous' (Example 2, shared "
            "coefficient b across affected units), 'distance_decay' "
            "(Example 3, alpha_i = b * exp(-d_i) via unit_distances)."
        ),
    )
    unit_distances: Optional[Dict[Any, float]] = Field(
        default=None,
        description=(
            "Required when spillover_structure='distance_decay'. Maps "
            "unit label to a non-negative scalar distance from the "
            "treated unit. Controls absent from the dict are treated as "
            "infinitely far (zero decay weight)."
        ),
    )
    weighting: Literal["identity", "efficient"] = Field(
        default="identity",
        description=(
            "Weighting matrix in the SP estimator. 'identity' = the "
            "standard W = I estimator. 'efficient' = additionally "
            "compute the GMM-weighted variant of Cao-Dowd v3 "
            "Proposition S.1 using W = sample-Omega^{-1} (lower "
            "asymptotic variance); exposed via results.cd.efficient_fit."
        ),
    )
    solver: Optional[str] = Field(
        default=None,
        description="cvxpy solver name for the leave-one-out SCM fits. "
                    "Defaults to CLARABEL.",
    )

    @model_validator(mode="after")
    def _check_affected_units(cls, values: Any) -> Any:
        au = values.affected_units
        structure = values.spillover_structure
        distances = values.unit_distances

        # Structure-specific requirements.
        if structure == "homogeneous" and (au is None or len(au) == 0):
            raise MlsynthConfigError(
                "SPILLSYNTH: spillover_structure='homogeneous' needs at "
                "least one entry in affected_units."
            )
        if structure == "distance_decay":
            if not isinstance(distances, dict) or len(distances) == 0:
                raise MlsynthConfigError(
                    "SPILLSYNTH: spillover_structure='distance_decay' "
                    "requires a non-empty unit_distances={label: d, ...}."
                )

        if au is None:
            return values
        if len(set(au)) != len(au):
            raise MlsynthConfigError(
                "SPILLSYNTH: affected_units contains duplicate labels."
            )
        # Resolve treated units on the spot to surface helpful errors
        # before .fit() runs. Multiple treated units are allowed (Cao-
        # Dowd v3 Section S.1.2), but none of them may also appear in
        # affected_units.
        df = values.df
        treat_col = values.treat
        unit_col = values.unitid
        treated_rows = set(df.loc[df[treat_col] != 0, unit_col].unique())
        overlap = treated_rows.intersection(au)
        if overlap:
            raise MlsynthConfigError(
                f"SPILLSYNTH: treated units {sorted(overlap, key=str)} cannot "
                "also appear in affected_units."
            )
        present = set(df[unit_col].unique())
        missing = [u for u in au if u not in present]
        if missing:
            raise MlsynthConfigError(
                f"SPILLSYNTH: affected_units {missing} not in df[{unit_col!r}]."
            )
        return values

    @model_validator(mode="after")
    def _check_sar(cls, values: Any) -> Any:
        if values.method != "sar":
            return values
        if values.spatial_W is None or values.spatial_w is None:
            raise MlsynthConfigError(
                "SPILLSYNTH: method='sar' requires both spatial_W (control-to-"
                "control weight matrix) and spatial_w (treated-to-control weight "
                "vector)."
            )
        if values.mcmc_burn >= values.mcmc_iter:
            raise MlsynthConfigError(
                "SPILLSYNTH: mcmc_burn must be smaller than mcmc_iter."
            )
        return values
