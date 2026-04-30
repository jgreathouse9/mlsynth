import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, List

# Configuration and Exceptions
from ..config_models import BaseMAREXConfig, LEXSCMConfig
from ..exceptions import MlsynthDataError, MlsynthEstimationError
from ..utils.helperutils import lexplot
# Utilities - Data Handling
from ..utils.datautils import balance

# Utilities - Fast SCM Core Setup
from ..utils.fast_scm_helpers.fast_scm_setup import (
    _prepare_working_df,
    build_candidate_mask,
    build_f_vector,
    build_X_tilde,
    build_Y_matrix,
    build_Z_matrix,
    prepare_experiment_inputs,
    split_periods, IndexSet
)

# Utilities - Search and Evaluation
from ..utils.fast_scm_helpers.fast_scm_bb import branch_and_bound_topK
from ..utils.fast_scm_helpers.fast_scm_control import evaluate_candidates
# Utilities - Power and Ranking
from ..utils.fast_scm_helpers.power_helpers import (select_best_tuple,
    run_mde_analysis,
)

from ..utils.fast_scm_helpers.inference import compute_post_inference, compute_moving_block_conformal_ci

from dataclasses import dataclass, field

from ..utils.fast_scm_helpers.structure import SEDCandidate


@dataclass
class LEXSCMResults:
    summary: pd.DataFrame
    best_candidate: SEDCandidate
    all_candidates: List[SEDCandidate]
    bnb_metadata: Dict[str, Any]

    n_units: int
    n_periods: int
    n_fit_periods: int
    n_pre_periods: int
    n_blank_periods: int
    n_post_periods: int

    # REQUIRED (no defaults)
    timeindex: IndexSet
    outcome: str

    # OPTIONAL (defaults allowed)
    y_pop_mean_t: np.ndarray = field(default_factory=lambda: np.array([]))


class LEXSCM:
    """
    Lexicographic Synthetic Control (LEXSCM) estimator.

    This estimator automatically designs synthetic control experiments by jointly
    optimizing:

        (i) Pre-treatment fit (validity)
        (ii) Statistical power (detectability of effects)

    The user interacts with the estimator through a single entry point:
    `fit()`, while all modeling choices are controlled via the configuration object.

    Parameters
    ----------
    config : dict or LEXSCMConfig
        Configuration object specifying data inputs, identification strategy,
        synthetic control settings, search budget, and inference parameters.

        The following fields are supported:

    Returns
    -------
    LEXSCM
        Initialized estimator ready for `.fit()` execution.

    See Also
    --------
    LEXSCM.fit : Executes the full optimization and estimation pipeline

    Notes
    -----
    - The pipeline integrates: search, estimation, evaluation, power analysis, selection.
    - Designed for fully automated experimental design under constraints.

    References
    ----------
    This implementation is based on the synthetic experimental design framework:

    - https://economics.mit.edu/sites/default/files/2026-02/Synthetic%20Controls%20for%20Experimental%20Design%20Feb%202026.pdf
      Develops the formal framework for selecting treated units using synthetic controls
      to reduce bias and improve experimental design in aggregate settings.

    - https://ivalua.cat/sites/default/files/2023-03/Vives-i-Bastida_2022_anon.pdf
      Provides a practical guide to synthetic experimental design in policy contexts,
      including inference and design trade-offs.

        Required
        --------
        df : pd.DataFrame
            Panel dataset containing unit-level observations over time.
        outcome : str
            Name of the outcome variable.
        unitid : str
            Column identifying observational units.
        time : str
            Time index column.
        candidate_col : str
            Boolean (0/1 or True/False) column indicating units eligible
            for treatment assignment.
        m : int
            Number of units selected per treated tuple.

        Identification / Design
        -----------------------
        post_col : str, optional
            Indicator for post-treatment period (0/1).
        frac_E : float
            Fraction of pre-treatment period used for estimation window E.
        unit_cost_col : str, optional
            Per-unit treatment cost column (must be constant within unit).
        budget : float, optional
            Total budget constraint on selected treated units.
        seed : int
            Random seed for reproducibility.

        Synthetic Control Specification
        -------------------------------
        weight_col : str, optional
            Unit-level weights (e.g., population, revenue).
        covariates : list of str, optional
            Covariates included in synthetic control construction.
        lambda_penalty : float
            Regularization penalty for control mismatch in quadratic program.

        Search / Optimization Budget
        ----------------------------
        top_K : int
            Number of top candidate tuples returned by branch-and-bound.
        top_P : int
            Number of seed units used for BnB initialization.

        Inference / Power Analysis
        --------------------------
        alpha : float
            Significance level for statistical testing.
        n_post_grid : list of int
            Post-treatment horizons used for MDE detectability curves.
        n_sims : int
            Number of Monte Carlo simulations for null distribution.
        post_imputation : {"mean", "max", "double_max"}
            Method for imputing post-treatment outcomes.
        test_statistic : {"mean_abs", "mean", "rms"}
            Test statistic used for treatment effect evaluation.
        delta : float
            Absolute minimum detectable effect threshold.
        relative_delta : float, optional
            Relative MDE threshold (> 1.0).
        target_mde_horizon : str
            Target horizon for MDE evaluation (e.g., early_mde_avg).
        max_shortlist : int
            Maximum number of candidate designs retained after filtering.

        System / Logging
        ----------------
        verbose : bool
            If True, enables progress logging.
    """

    def __init__(self, config):
        if isinstance(config, dict):
            try:
                config = LEXSCMConfig(**config)
            except Exception as e:  # pydantic ValidationError
                raise MlsynthDataError(f"Invalid LEXSCM configuration: {e}") from e

        # =========================================================
        # CORE DATA
        # =========================================================
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time

        # =========================================================
        # IDENTIFICATION
        # =========================================================
        self.candidate_col: str = config.candidate_col
        self.post_col: Optional[str] = config.post_col
        self.m: int = config.m
        self.weight_col: Optional[str] = config.weight_col
        self.unit_cost_col: Optional[str] = config.unit_cost_col
        self.budget: Optional[float] = config.budget
        self.frac_E: float = config.frac_E

        # =========================================================
        # SYNTHETIC CONTROL
        # =========================================================
        self.covariates: Optional[list] = config.covariates
        self.lambda_penalty: float = config.lambda_penalty

        # =========================================================
        # SEARCH / BNB
        # =========================================================
        self.top_K: int = config.top_K
        self.top_P: int = config.top_P

        # =========================================================
        # INFERENCE / POWER
        # =========================================================
        self.alpha: float = config.alpha
        self.n_post_grid: list = config.n_post_grid
        self.n_sims: int = config.n_sims
        self.post_imputation: str = config.post_imputation
        self.test_statistic: str = config.test_statistic
        self.target_mde_horizon: Literal["early_mean", "early_min", "late"] = config.mde_horizon
        self.max_shortlist: int = config.max_shortlist
        
        self.display_graph: bool = config.display_graph

        # =========================================================
        # SYSTEM
        # =========================================================
        self.seed: int = config.seed
        self.verbose: bool = config.verbose

        

    def fit(self, **kwargs) -> "LEXSCM":
        """
        Run the full Synthetic Experiment Design pipeline.

        This method executes the end-to-end workflow:
            - prepares panel data
            - searches over candidate treated unit sets
            - fits synthetic controls
            - evaluates pre-treatment fit
            - estimates statistical power (MDE)
            - selects the optimal design

        Parameters
        ----------
        **kwargs : dict, optional
            Reserved for future extensions (currently unused).

        Returns
        -------
        LEXSCMResults
            Object containing:
            - summary : ranked table of candidate designs
            - best_candidate : selected experimental design
            - all_candidates : full evaluated set
            - bnb_metadata : search diagnostics
            - additional dataset and inference diagnostics
        """
    
        balance(self.df, self.unitid, self.time)

        # Step 1: Prepare working DataFrame using the helper
        working_df, self.post_df = _prepare_working_df(
            self.df,
            self.post_col
        )
        self.pre_df = working_df   # store for convenience if desired

        unit_index = IndexSet.from_labels(
            sorted(working_df[self.unitid].unique())
        )

        time_index = IndexSet.from_labels(
            sorted(working_df[self.time].unique())
        )

        full_time_index = None

        if self.post_col is not None and not self.post_df.empty:
            full_time_index = IndexSet.from_labels(
                sorted(self.df[self.time].unique())
            )

        final_time_index = full_time_index if full_time_index is not None else time_index

        # Step 2: Build candidate mask (aligned with future Y columns)
        candidate_mask = build_candidate_mask(
            working_df=working_df,
            candidate_col=self.candidate_col,
            unit_index=unit_index,
            unitid=self.unitid
        )

        # Step 3: Build Y matrix
        self.Y = build_Y_matrix(
            working_df=working_df,
            outcome=self.outcome,
            time=self.time,
            unitid=self.unitid, unit_index=unit_index
        )

        # Step 4: Build Z matrix (covariates)
        self.Z = build_Z_matrix(
            working_df=working_df,
            covariates=self.covariates,
            time=self.time,
            unitid=self.unitid, unit_index=unit_index
        )

        # Step 5: Build f weighting vector
        self.f = build_f_vector(
            working_df=working_df,
            weight_col=self.weight_col,
            unitid=self.unitid, unit_index=unit_index
        )

        X, f, candidate_idx, T0_pre, N = prepare_experiment_inputs(
            self.Y, self.Z, self.f, candidate_mask, self.m
        )

        # Get logical indices
        E_idx, B_idx, post_idx = split_periods(
            T0=self.pre_df[self.time].nunique(),
            frac_E=self.frac_E,
            post_df=self.post_df,
            time_col=self.time
        )

        # Standardize over estimation period
        X_E, G = build_X_tilde(X, f, E_idx, J=self.Y.shape[1])

        bbresults = branch_and_bound_topK(
            G=G,
            candidate_idx=candidate_idx,
            m=self.m,
            top_K=self.top_K,
            unit_index=unit_index
        )
        


        # ------------------- Stage 2: Evaluate candidates -------------------
        candidate_results = evaluate_candidates(
            candidates=bbresults['top_tuples'],  # now list of Solution objects
            X=X,
            X_E=X_E,
            Y=self.Y,
            f=self.f,
            E_idx=E_idx,  # ← you must pass this now
            B_idx=B_idx,
            lambda_penalty=self.lambda_penalty, index_set=unit_index
        )

        candidate_mdes = run_mde_analysis(
            candidates=candidate_results,
            n_post_grid=self.n_post_grid,
            n_sims=self.n_sims
        )

        
        winner, shortlist = select_best_tuple(
            candidate_mdes,
            mde_horizon="early_min"
        )


        # ------------------- Stage 3: Inference (Post-Intervention) -------------------
        # ==============================================================
        # 5. Post-Intervention Inference 
        # ==============================================================
        # We only compute effects using the selected treated units 
        # (their column indices are stored in .identification.treated_idx)

        # Default: use full pre-period population mean
        y_pop_mean_t = self.Y.mean(axis=1)

        if len(post_idx) > 0 and not self.post_df.empty:
            # Build full timeline matrix (pre + post)
            Y_post = build_Y_matrix(
                working_df=self.post_df,
                outcome=self.outcome,
                time=self.time,
                unitid=self.unitid,
                unit_index=unit_index
            )
            Y_full = np.vstack([self.Y, Y_post])

            # Update population mean over the full timeline
            y_pop_mean_t = Y_full.mean(axis=1)

            # Update predictions and effects for each candidate
            for cand in candidate_results:
                # Get the column indices of the selected treated units for this candidate
                treated_col_idx = np.asarray(cand.identification.treated_idx, dtype=int)

                # Extract weights
                treated_weights = cand.weights.treated      # weights for the m treated units
                control_weights = cand.weights.control      # full control weights (length N)

                # Compute synthetic treated using ONLY the selected treated columns
                synth_treated_full = Y_full[:, treated_col_idx] @ treated_weights

                # Compute synthetic control using all units
                synth_control_full = Y_full @ control_weights

                # Store results
                cand.predictions.synthetic_treated = synth_treated_full
                cand.predictions.synthetic_control = synth_control_full
                cand.predictions.effects = synth_treated_full - synth_control_full

                # Point estimate (Average Treatment Effect) over post periods
                post_gap = cand.predictions.effects[post_idx]
                cand.inference.ate = float(np.mean(post_gap))

                # Optional: store the treated column indices used
                cand.inference.treated_col_idx = treated_col_idx.tolist()

                # Run inference (permutation test + conformal CI)
                cand.inference.p_value = compute_post_inference(
                    candidate=cand,
                    post_idx=post_idx,
                    n_perms=self.n_sims,
                    seed=getattr(self.seed, 'seed', 42)
                ).inference.p_value

                cand = compute_moving_block_conformal_ci(
                    candidate=cand,
                    post_idx=post_idx,
                    alpha=0.05,
                    seed=getattr(self.seed, 'seed', 42)
                )

        # ==============================================================
        # 6. Final Assembly
        # ==============================================================
        # Re-fetch the winner (now with updated post-intervention results)
        best_candidate = next(
            (c for c in candidate_results 
             if c.identification.tuple_id == winner.identification.tuple_id),
            candidate_results[0]   # fallback
        )

        T = len(time_index)

        results = LEXSCMResults(
            summary=shortlist,
            best_candidate=best_candidate,
            all_candidates=candidate_results,
            bnb_metadata=bbresults,
            y_pop_mean_t=y_pop_mean_t,
            n_units=self.Y.shape[1],
            n_periods=T,
            n_pre_periods=len(E_idx) + len(B_idx),
            n_fit_periods=len(E_idx),
            n_blank_periods=len(B_idx),
            n_post_periods=len(post_idx), timeindex=final_time_index, outcome=self.outcome
        )
        
        if self.display_graph:
            lexplot(results)

        return results
