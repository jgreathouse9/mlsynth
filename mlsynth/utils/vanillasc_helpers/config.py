"""Configuration for the VanillaSC estimator.

Co-located with the VanillaSC helper package. The shared
:class:`~mlsynth.config_models.BaseEstimatorConfig` remains central (it is
common to every estimator); only the per-estimator config lives here, next to
the code it configures. Re-exported from :mod:`mlsynth.config_models` for
backward compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...config_models import BaseEstimatorConfig


class StaggeredSCMSpec(BaseModel):
    """Covariate (multi-feature) matching spec for staggered VanillaSC.

    Mirrors the ``scpi`` package's ``scdataMulti`` specification, applied
    uniformly to every treated unit. The treated units are detected from the
    treatment indicator -- the spec never names them. When a ``staggered_spec``
    is attached to a multi-treated panel, ``VanillaSC`` builds one synthetic
    control per treated unit on these features and reproduces the Cattaneo-Feng-
    Palomba-Titiunik (2025) point estimates and prediction intervals for all
    three causal predictands.
    """

    model_config = ConfigDict(extra="forbid")

    features: List[str] = Field(
        ...,
        description="Matching features (outcome-like series), shared across all "
                    "treated units. Must include the outcome.",
    )
    cov_adj: Optional[List[Any]] = Field(
        default=None,
        description="Covariate-adjustment terms per feature (e.g. "
                    "['constant', 'trend']). None -> no adjustment.",
    )
    constant: bool = Field(
        default=False,
        description="Include a global constant (scpi's 'constant').",
    )
    cointegrated: bool = Field(
        default=False,
        description="Difference the data for cointegration (scpi's "
                    "'cointegrated_data').",
    )

    @field_validator("features")
    @classmethod
    def _features_nonempty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("staggered_spec.features must be non-empty.")
        return v


class VanillaSCConfig(BaseEstimatorConfig):
    """Configuration for the VanillaSC estimator (standard SCM, bilevel engine).

    The ordinary single-treated synthetic control, built on the self-contained
    bilevel machinery. With no covariates it reduces to the well-posed convex
    outcome-matching problem; with covariates it routes through the bilevel
    predictor-weight (``V``) optimisation, with a selectable, reliable backend.

    Parameters
    ----------
    backend : {"auto", "outcome-only", "malo", "mscmt", "penalized"}
        Predictor-weight backend. ``"auto"`` (default) uses ``"outcome-only"``
        (convex simplex fit on pre-treatment outcomes) when no covariates are
        given, and ``"mscmt"`` (global differential-evolution ``V`` search)
        when they are. ``"malo"`` is the Malo et al. (2024) corner search,
        ``"penalized"`` the Abadie-L'Hour (2021) unique/sparse estimator.
    covariates : list of str, optional
        Predictor columns. Each is averaged over its window (see
        ``covariate_windows``) and scaled to unit variance, then matched via
        the bilevel program. ``None`` -> outcome-only matching.
    covariate_windows : dict, optional
        Per-covariate inclusive ``(start, end)`` averaging window of time
        labels (Abadie's special-predictor spec). Covariates not listed are
        averaged over the full pre-treatment period.
    fit_window : tuple, optional
        Inclusive ``(start, end)`` window restricting the outcome fit (the
        dependent SSR, MSCMT's ``times.dep``) to a sub-range of the
        pre-treatment period; predictor matching is unaffected. ``None``
        (default) fits the full pre-period. This is how Abadie-Gardeazabal's
        Basque study fits ``gdpcap`` over 1960-1969 rather than 1955-1969.
    canonical_v : bool or {"min.loss.w", "max.order"}
        Canonicalise the (non-identified) predictor weights for ``mscmt``
        (MSCMT ``determine_v``). The reported ``v_agreement`` is small when
        ``V`` is well identified and large when it is fragile. Default False.
    seed : int
        RNG seed for the ``mscmt`` differential-evolution search.
    mscmt_maxiter, mscmt_popsize : int
        Differential-evolution budget for the ``mscmt`` backend.
    inference : bool or {"placebo", "scpi", "lto"}
        Inference method. ``True``/``"placebo"`` (default) runs Abadie in-space
        placebo inference (refit treating each donor as pseudo-treated; the
        p-value ranks the treated unit's post/pre RMSPE ratio). ``"scpi"`` runs
        Cattaneo-Feng-Titiunik (2021) prediction intervals (in-sample
        simulation + out-of-sample location-scale; exact for the simplex /
        outcome-only synthetic control). ``"lto"`` runs the Lei-Sudijono (2025)
        leave-two-out refined placebo test (O(J^2) reference comparisons; finer
        granularity and non-zero size when ``alpha < 1/N``). ``False`` skips
        inference.
    alpha : float
        Level. For placebo, the confidence statement; for SCPI, used as both
        the in-sample (alpha1) and out-of-sample (alpha2) levels, giving a
        prediction interval with coverage approximately ``1 - 2*alpha``.
    scpi_sims : int
        Number of Gaussian draws for the SCPI in-sample simulation.
    scpi_e_method : {"gaussian", "empirical"}
        Out-of-sample location-scale tabulation for SCPI.
    lto_max_pairs : int, optional
        Cap on the number of donor pairs evaluated by the ``"lto"`` test
        (deterministic subsample via ``seed``). ``None`` (default) uses all
        ``J*(J-1)/2`` pairs; set a cap to keep the O(J^2) cost tractable with
        slow backends.
    """

    backend: Literal["auto", "outcome-only", "malo", "mscmt", "penalized"] = Field(
        default="auto",
        description="Predictor-weight backend (see class docstring).",
    )
    covariates: Optional[List[str]] = Field(
        default=None,
        description="Predictor columns; None -> outcome-only matching.",
    )
    covariate_windows: Optional[Dict[Any, Any]] = Field(
        default=None,
        description="Per-covariate inclusive (start, end) averaging window.",
    )
    fit_window: Optional[Tuple[Any, Any]] = Field(
        default=None,
        description="Inclusive (start, end) time window over which the synthetic "
                    "control is fit to the treated outcome -- the dependent SSR, "
                    "MSCMT's ``times.dep``. Restricts the pre-period outcome "
                    "optimisation to this sub-range (predictors are still matched "
                    "over their own covariate_windows). None (default) uses the "
                    "full pre-treatment period; this is how Abadie-Gardeazabal's "
                    "Basque study fits gdpcap over 1960-1969 rather than 1955-1969.",
    )

    @field_validator("fit_window")
    @classmethod
    def _validate_fit_window(cls, v):
        if v is None:
            return v
        if len(v) != 2:
            raise ValueError(
                "fit_window must be a (start, end) pair of time labels.")
        start, end = v
        if start is not None and end is not None and start > end:
            raise ValueError(
                f"fit_window start ({start}) must not exceed end ({end}).")
        return v
    canonical_v: Union[bool, str] = Field(
        default=False,
        description="Canonicalise mscmt predictor weights ('min.loss.w'/'max.order').",
    )
    seed: int = Field(default=0, description="RNG seed for the mscmt DE search.")
    mscmt_maxiter: int = Field(
        default=300, ge=1, description="mscmt differential-evolution max iterations.",
    )
    mscmt_popsize: int = Field(
        default=15, ge=1, description="mscmt differential-evolution population size.",
    )
    mscmt_prune_shady: bool = Field(
        default=True,
        description="mscmt: drop shady donors (Becker-Kloessner sunny-donor "
                    "reduction) before the outer search. Leaves the optimum "
                    "unchanged; shrinks the inner solve.",
    )
    augment: Optional[Literal["ridge"]] = Field(
        default=None,
        description="Augmentation layer: 'ridge' turns the fit into Augmented "
                    "SCM (Ben-Michael, Feller & Rothstein 2021) -- a ridge "
                    "bias-correction over the simplex base; None -> plain SCM.",
    )
    ridge_lambda: Optional[float] = Field(
        default=None, ge=0.0,
        description="Fixed ridge penalty for augment='ridge'; None -> select by "
                    "leave-one-period-out CV (augsynth's 1-SE rule).",
    )
    residualize: bool = Field(
        default=False,
        description="With augment='ridge' and covariates: stack covariates as "
                    "matching rows (False, augsynth parallel default) or regress "
                    "them out and match on residuals (True, residualize=TRUE).",
    )
    inference: Union[bool, str] = Field(
        default=True,
        description="Inference: True/'placebo' (in-space placebo), 'scpi' "
                    "(Cattaneo-Feng-Titiunik prediction intervals), 'conformal' "
                    "(Chernozhukov-Wuthrich-Zhu test-inversion intervals, the "
                    "augsynth default for ASCM), 'lto' (Lei-Sudijono leave-two-"
                    "out refined placebo), 'ttest' (Chernozhukov-Wuthrich-Zhu "
                    "2025 debiased SC t-test for the ATT), or False.",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Level (placebo confidence / SCPI alpha1 = alpha2).",
    )
    scpi_sims: int = Field(
        default=200, ge=1,
        description="Gaussian draws for the SCPI in-sample simulation.",
    )
    scpi_e_method: Literal["gaussian", "empirical"] = Field(
        default="gaussian",
        description="SCPI out-of-sample location-scale tabulation.",
    )
    staggered_spec: Optional[StaggeredSCMSpec] = Field(
        default=None,
        description="Staggered adoption only. Covariate (multi-feature) matching "
                    "spec (scpi's scdataMulti specification). When set on a "
                    "multi-treated panel, VanillaSC reproduces the Cattaneo-Feng-"
                    "Palomba-Titiunik (2025) point estimates and prediction "
                    "intervals for all three predictands with covariates. None "
                    "(default) -> outcome-only staggered matching.",
    )
    scpi_compat: bool = Field(
        default=False,
        description="Staggered adoption only. When True, the cross-unit "
                    "(TSUA / TAUA) in-sample prediction intervals reproduce the "
                    "scpi package's published numbers bit-for-bit, including its "
                    "1/iota**2 scaling of the time-aggregated in-sample term "
                    "(scdataMulti divides the predictand matrix P by the number "
                    "of treated units iota, and scpi_in_diag divides the "
                    "simulated draws by iota a second time). The statistically "
                    "correct scaling for the average of iota independent per-unit "
                    "errors is a single 1/iota, which is the default (False). "
                    "Use True only to match scpi exactly.",
    )
    lto_max_pairs: Optional[int] = Field(
        default=None, ge=1,
        description="Cap on donor pairs for the 'lto' test (None -> all pairs).",
    )
    oracle_weights: Optional[Dict[Any, float]] = Field(
        default=None,
        description="User-specified donor weights ``{donor_id: weight}``. When "
                    "given, the weight optimization is skipped and these weights "
                    "are used directly (e.g. the 'oracle' known-weights case). "
                    "Donors omitted from the map get weight 0. Supported with "
                    "inference=False or inference='ttest'.",
    )
    ttest_K: Union[int, Literal["auto"]] = Field(
        default=3,
        description="Cross-fitting folds for inference='ttest' (Chernozhukov-"
                    "Wuthrich-Zhu 2025): an int >= 2, or 'auto' to select K from "
                    "the SC-residual persistence and the RAE formula per their "
                    "Sec 3.2. K=3 is the small-T0 benchmark; larger K tightens "
                    "the interval.",
    )

    @field_validator("ttest_K")
    @classmethod
    def _validate_ttest_K(cls, v):
        if v == "auto":
            return v
        if isinstance(v, bool) or not isinstance(v, int) or v < 2:
            raise ValueError("ttest_K must be an integer >= 2 or 'auto'.")
        return v
    penalized_cv: Literal["holdout", "loo", "pensynth"] = Field(
        default="holdout",
        description="Lambda selector for backend='penalized'. 'holdout'/'loo' "
                    "are Abadie-L'Hour time-split criteria; 'pensynth' is van "
                    "Kesteren's cv_pensynth (fit on covariates, validate on the "
                    "held-out pre-period outcome path; needs covariates). "
                    "Ignored when 'penalized_lambda' is set.",
    )
    penalized_lambda: Optional[float] = Field(
        default=None, ge=0.0,
        description="Fixed Abadie-L'Hour penalty for backend='penalized'. When "
                    "set (>= 0), it is used directly and cross-validation "
                    "('penalized_cv') is skipped; larger values approach "
                    "nearest-neighbour matching, an infinitesimal value gives "
                    "the lexicographic (fit-preserving) tie-break, and 0 is the "
                    "unpenalized synthetic control. 'None' (default) selects "
                    "lambda by cross-validation. Ignored for other backends.",
    )
