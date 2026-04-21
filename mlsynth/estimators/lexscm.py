import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, List

# Configuration and Exceptions
from ..config_models import BaseMAREXConfig, LEXSCMConfig
from ..exceptions import MlsynthDataError, MlsynthEstimationError

# Utilities - Data Handling
from ..utils.datautils import balance, dataprep

# Utilities - Fast SCM Core Setup
from ..utils.fast_scm_helpers.fast_scm_setup import (
    _prepare_working_df,
    build_candidate_mask,
    build_f_vector,
    build_X_tilde,
    build_Y_matrix,
    build_Z_matrix,
    prepare_experiment_inputs,
    split_periods,
)

# Utilities - Search and Evaluation
from ..utils.fast_scm_helpers.fast_scm_bb import branch_and_bound_topK
from ..utils.fast_scm_helpers.fast_scm_control import evaluate_candidates
# Utilities - Power and Ranking
from ..utils.fast_scm_helpers.power_helpers import (
    mde_summary_table,
    run_mde_analysis, select_best_tuple
)

from ..utils.fast_scm_helpers.post_inference import update_post_inference
from dataclasses import dataclass, field

from ..utils.fast_scm_helpers.structure import SEDCandidate

@dataclass
class LEXSCMResults:
    """
    Final output container for the LEXSCM pipeline.

    This object aggregates all artifacts produced across the full
    Synthetic Experiment Design pipeline:

    1. Combinatorial search (Branch-and-Bound)
    2. Control optimization (QP synthetic control fit)
    3. Evaluation (NMSE + residual diagnostics)
    4. Power analysis (MDE / detectability curves)
    5. Pareto ranking (bias–variance tradeoff)

    Attributes
    ----------
    summary : pd.DataFrame
        Ranked table of all evaluated candidates.
        Includes NMSE, MDE summaries, and composite SED score.
        Used to identify Pareto-optimal designs.

    best_candidate : SEDCandidate
        The single highest-ranked candidate according to the
        SED scoring function (bias–power tradeoff).

    all_candidates : List[SEDCandidate]
        Full list of evaluated candidates produced by:
            branch-and-bound → control QP → evaluation → MDE analysis.

    bnb_metadata : Dict[str, Any]
        Diagnostic statistics from the search procedure:
        number of nodes visited, pruning rate, subset coverage,
        and search efficiency metrics.

    config : Any
        Original configuration object used to run the estimator.
        Stored for reproducibility and auditability.

    Notes
    -----
    This object is intentionally immutable in design (conceptually),
    representing the terminal state of the estimation pipeline.
    """
    summary: pd.DataFrame
    best_candidate: SEDCandidate
    all_candidates: List[SEDCandidate]
    bnb_metadata: Dict[str, Any]
    config: Any

    # required diagnostics (no defaults)
    n_units: int
    n_periods: int
    n_fit_periods: int
    n_pre_periods: int
    n_blank_periods: int
    n_post_periods: int

    # optional diagnostics
    y_pop_mean_t: np.ndarray = field(default_factory=lambda: np.array([]))

class LEXSCM:
"""
    Lexicographic Synthetic Control Method (LEXSCM) for Experimental Design.

    This estimator implements a constrained combinatorial search to identify optimal 
    experimental units, synthesizing methodologies from Vives-i-Bastida (2022) 
    regarding external validity and Abadie & Zhou (2026) regarding optimal design.

    REFERENCES
    ----------
    Vives: https://ivalua.cat/sites/default/files/2023-03/Vives-i-Bastida_2022_anon.pdf
    Abadie and Zhou: https://economics.mit.edu/sites/default/files/2026-02/Synthetic%20Controls%20for%20Experimental%20Design%20Feb%202026.pdf

    MATHEMATICAL FOUNDATIONS
    -----------------------
    1. External Validity (v-weights): Following Vives-i-Bastida (2022), we solve 
       a lexicographic optimization where the first priority is matching the 
       treated tuple to the 'National Mean' or target population.
    2. Optimal Design: Following Abadie & Zhou (2026), we use a Branch-and-Bound 
       framework to minimize the expected mean squared error of the synthetic 
       control estimator by selecting the 'easiest to model' treated units.

    INPUT PARAMETERS (via LEXSCMConfig)
    ----------------------------------
    IDENTIFICATION DESIGN:
        - candidate_col (str): Column indicating units eligible for treatment selection.
        - m (int): Number of units selected for the treated group.
        - post_col (str, optional): Manual indicator for post-treatment period.
        - unit_cost_col (str, optional): Column containing per-unit activation costs.
        - budget (float, optional): Hard total budget cap for the treated group.
        - seed (int): Random seed for MDE Monte Carlo simulations (default: 42).
        - frac_E (float): Fraction of pre-period used for estimation window (default: 0.7).

    SCM SPECIFICATION:
        - weight_col (str, optional): Unit-level importance weights (e.g., population) 
            used to calculate the target population mean for v-weight matching.
        - covariates (List[str], optional): Features to include in the synthetic control.
        - lambda_penalty (float): Regularization for synthetic control weights (default: 0.1).

    SEARCH / COMPUTATIONAL BUDGET:
        - top_K (int): Number of top candidate tuples to evaluate in Stage 2.
        - top_P (int): Number of seed units used to initialize the BnB search.

    POWER / INFERENCE (MDE):
        - alpha (float): Significance level (default: 0.05).
        - n_post_grid (List[int]): Horizons for detectability curve calculation.
        - n_sims (int): Number of Monte Carlo simulations for null distributions.
        - post_imputation (str): Method for MDE signal injection ('mean', 'max', etc.).
        - test_statistic (str): Statistic used for inference ('mean_abs', 'mean', 'rms').

    LEXICOGRAPHIC SELECTION:
        - delta (float): Absolute NMSE tolerance for 'Fit-First' selection.
        - relative_delta (float): Relative NMSE tolerance based on the best-found fit.
        - target_mde_horizon (str): Specific MDE metric used to rank the final shortlist.
        - max_shortlist (int): Maximum number of candidates returned in the summary.

    PIPELINE STAGES
    ---------------
    Stage 1: Combinatorial Search (BnB)
        Efficiently prunes the C(N, m) space using the budget constraint and a 
        convex relaxation of the loss surface.
    
    Stage 2: Synthetic Control Construction
        Solves the Synthetic Control Quadratic Program (QP) for each surviving candidate.

    Stage 3: Power Analysis & Inference
        Estimates Minimum Detectable Effects via permutation tests and calculates 
        Conformal Confidence Intervals for the Treatment Effect.
    """

    # ===================================================================
    #                          Estimator Logic
    # ===================================================================

    def __init__(self, config):
        if isinstance(config, dict):
            try:
                config = LEXSCMConfig(**config)
            except Exception as e:  # pydantic ValidationError
                raise MlsynthDataError(f"Invalid LEXSCM configuration: {e}") from e

        self.config = config # Store the config object
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.candidate_col: str = config.candidate_col
        self.post_col: Optional[str] = config.post_col
        self.weight_col: Optional[str] = config.weight_col
        self.m: Optional[str] = config.m
        self.frac_E: Optional[str] = config.frac_E
        self.top_K: Optional[int] = config.top_K
        self.top_P: Optional[int] = config.top_P
        self.lambda_penalty = config.lambda_penalty
        self.covariates: Optional[List[str]] = config.covariates

        # MDE computation parameters
        self.n_post_grid: List[int] = config.n_post_grid
        self.n_sims: int = config.n_sims
        self.alpha: float = config.alpha
        self.post_imputation: Literal["mean", "max", "double_max"] = config.post_imputation
        self.test_statistic: Literal["mean_abs", "mean", "rms"] = config.test_statistic

        # Lexicographic selection parameters (fit-first, then power)
        self.delta: float = config.delta
        self.relative_delta: Optional[float] = config.relative_delta
        self.target_mde_horizon: str = config.target_mde_horizon
        self.max_shortlist: int = config.max_shortlist


        

    def fit(self, **kwargs) -> "LEXSCM":
        """
        Run the full Synthetic Experiment Design pipeline.

        This method executes the complete estimation workflow:

            1. Data preparation and alignment
            2. Construction of outcome and covariate matrices
            3. Feature standardization
            4. Combinatorial search over candidate treated sets (BnB)
            5. Synthetic control optimization for each candidate
            6. Pre/post fit evaluation (NMSE diagnostics)
            7. Power analysis via MDE simulation
            8. Pareto ranking of experimental designs

        Parameters
        ----------
        kwargs : dict
            Optional overrides for runtime behavior.
            (Currently unused but reserved for future extensions such as:
             - alternative solvers
             - custom ranking weights
             - alternative power models)

        Returns
        -------
        results : LEXSCMResults
            Fully materialized result object containing:
                - ranked candidate table
                - best experimental design
                - full candidate set
                - search diagnostics
                - original configuration

        Notes
        -----
        - This method is computationally intensive due to:
            * combinatorial search complexity
            * repeated QP solves
            * Monte Carlo null simulations

        - The pipeline is deterministic except for:
            * Monte Carlo sampling in MDE estimation

        - Designed for research-grade reproducibility and
          experimental design selection, not real-time inference.

        Pipeline Details
        ----------------
        Step 1: Data preparation
            - Balances panel structure
            - Splits pre/post treatment periods
            - Builds candidate eligibility mask

        Step 2: Matrix construction
            - Y: outcome matrix (unit × time)
            - Z: optional covariates stacked beneath Y
            - f: unit weighting vector

        Step 3: Feature engineering
            - Concatenates Y and Z into X
            - Standardizes over estimation window (X_E)
            - Computes Gram matrix G for BnB

        Step 4: Branch-and-Bound search
            - Enumerates candidate treated tuples of size m
            - Uses convex relaxation for pruning
            - Returns top-K solutions

        Step 5: Evaluation
            - Solves synthetic control QP per candidate
            - Computes synthetic treated/control series
            - Measures in-sample and baseline fit (NMSE)

        Step 6: Power analysis
            - Runs permutation-based MDE estimation
            - Simulates null distributions via Monte Carlo
            - Computes detectability curves across post horizons

        Step 7: Ranking
            - Combines NMSE_B and early MDE
            - Produces Pareto-style SED score
            - Selects optimal experimental design
        """
        # ------------------- Prepare candidate mask -------------------
    
        balance(self.df, self.unitid, self.time)

        # Step 1: Prepare working DataFrame using the helper
        working_df, self.post_df = _prepare_working_df(
            self.df,
            self.post_col
        )
        self.pre_df = working_df   # store for convenience if desired

        # Step 2: Build candidate mask (aligned with future Y columns)
        self.candidate_mask, self.candidate_unit_set, unit_labels = build_candidate_mask(
            working_df=working_df,
            candidate_col=self.candidate_col,
            unitid=self.unitid
        )

        if len(self.candidate_unit_set) < self.config.m:
            raise MlsynthDataError(
                f"Only {len(self.candidate_unit_set)} candidate units in pre-period, "
                f"but m={self.config.m} requested."
            )

        self.unit_costs = None
        if self.config.budget is not None and self.config.unit_cost_col is not None:
            cost_df = working_df[[self.unitid, self.config.unit_cost_col]].drop_duplicates(subset=[self.unitid])
            cost_map = dict(zip(cost_df[self.unitid], cost_df[self.config.unit_cost_col]))

            self.unit_costs = np.array([cost_map.get(uid, 0.0) for uid in unit_labels])

            print(f"INFO: Budget constraint active (${self.config.budget:,.2f}). "
                  f"Using costs from column '{self.config.unit_cost_col}'.")

        # Step 3: Build Y matrix
        self.Y = build_Y_matrix(
            working_df=working_df,
            outcome=self.outcome,
            time=self.time,
            unitid=self.unitid,
            unit_labels=unit_labels
        )

        # Step 4: Build Z matrix (covariates)
        self.Z = build_Z_matrix(
            working_df=working_df,
            covariates=self.config.covariates,
            time=self.time,
            unitid=self.unitid,
            unit_labels=unit_labels
        )

        # Step 5: Build f weighting vector
        self.f = build_f_vector(
            working_df=working_df,
            weight_col=self.config.weight_col,
            unitid=self.unitid,
            unit_labels=unit_labels
        )

        # Final sanity check
        assert self.Y.shape[1] == len(self.candidate_mask),\
            "Y and candidate_mask dimension mismatch!"

        X, f, candidate_idx, T0_pre, N = prepare_experiment_inputs(
            self.Y, self.Z, self.f, self.candidate_mask, self.m
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

        # ------------------- Stage 1: BnB -------------------
        bbresults = branch_and_bound_topK(
            G,
            candidate_idx,
            m=self.m,
            top_K=self.top_K,
            top_P=self.top_P,
            total_units=N,
            # NEW PARAMETERS
            unit_costs=self.unit_costs,
            budget=self.config.budget
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
            lambda_penalty=self.lambda_penalty
        )

        candidate_mdes = run_mde_analysis(
            candidates=candidate_results,
            n_post_grid=self.config.n_post_grid,
            n_sims=self.config.n_sims
        )

        # --- Select best design: Threshold on fit → Optimize power ---
        winner, shortlist = select_best_tuple(
            candidates=candidate_results,
            delta=self.config.delta,
            relative_delta=self.config.relative_delta,
            target_mde_horizon=self.config.target_mde_horizon,
            return_shortlist=True,
            max_shortlist=self.config.max_shortlist
        )

        # ------------------- Stage 3: Inference (Post-Intervention) -------------------
        # We take the full Y matrix (Pre + Post)
        # Note: If post_df was provided, Y already contains those rows
        # ------------------- Stage 3: Inference (Post-Intervention) -------------------
        y_pop_mean_t = self.Y.mean(axis=1)

        if len(post_idx) > 0 and not self.post_df.empty:
            # Build full timeline matrix (pre + post)
            Y_post = build_Y_matrix(
                working_df=self.post_df,
                outcome=self.outcome,
                time=self.time,
                unitid=self.unitid,
                unit_labels=unit_labels
            )
            Y_full = np.vstack([self.Y, Y_post])

            # Update population mean over the full timeline
            y_pop_mean_t = Y_full.mean(axis=1)

            # Update all candidates with post-intervention results
            candidate_results = update_post_inference(
                candidate_results=candidate_results,
                Y_full=Y_full,
                post_idx=post_idx,
                n_sims=self.config.n_sims,
                alpha=0.05,
                seed=getattr(self.config, 'seed', 42)
            )

        # ==============================================================
        # 6. Final Assembly
        # ==============================================================
        # Re-fetch the winner (now with updated post-intervention results)
        best_candidate = next(
            (c for c in candidate_results
             if c.identification.tuple_id == winner.identification.tuple_id),
            candidate_results[0]  # fallback
        )

        unit_name_map = (
            self.df[[self.unitid]]
            .drop_duplicates()
            .set_index(np.arange(self.Y.shape[1]))  # index matches Y columns
            .squeeze()
            .to_dict()
        )

        # Attach weight dictionaries to every candidate
        for cand in candidate_results:
            # --- Treated weights (only the m selected units) ---
            treated_idx = np.asarray(cand.identification.treated_idx, dtype=int)
            treated_weights = cand.weights.treated  # length m

            cand.treated_weights_dict = {
                unit_name_map[i]: float(w)
                for i, w in zip(treated_idx, treated_weights)
            }

            # --- Control weights (all units, but only non-zero for clarity) ---
            control_weights = cand.weights.control  # length N
            cand.control_weights_dict = {
                unit_name_map[i]: float(w)
                for i, w in enumerate(control_weights)
                if abs(w) > 1e-8  # filter out numerical noise
            }

        # Optional: Also attach to winner and shortlist for quick access
        if hasattr(winner, 'identification'):
            winner.treated_weights_dict = winner.treated_weights_dict if hasattr(winner, 'treated_weights_dict') else {}
            winner.control_weights_dict = winner.control_weights_dict if hasattr(winner, 'control_weights_dict') else {}
        
        
        


        results = LEXSCMResults(
            summary=shortlist,
            best_candidate=best_candidate,
            all_candidates=candidate_results,
            bnb_metadata=bbresults,
            config=self.config,
            y_pop_mean_t=y_pop_mean_t,
            n_units=self.Y.shape[1],
            n_periods=len(E_idx) + len(B_idx) + len(post_idx),
            n_pre_periods=len(E_idx) + len(B_idx),
            n_fit_periods=len(E_idx),
            n_blank_periods=len(B_idx),
            n_post_periods=len(post_idx)
        )

        return results

