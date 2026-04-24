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
    split_periods, IndexSet
)

# Utilities - Search and Evaluation
from ..utils.fast_scm_helpers.fast_scm_bb import branch_and_bound_topK
from ..utils.fast_scm_helpers.fast_scm_control import evaluate_candidates
# Utilities - Power and Ranking
from ..utils.fast_scm_helpers.power_helpers import (select_best_tuple,
    run_mde_analysis,
)

from ..utils.fast_scm_helpers.inference import compute_post_inference, compute_conformal_ci

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
    ---

    PIPELINE OVERVIEW
    -----------------

    The estimator operates in three tightly coupled stages:

    Stage 1: Combinatorial Search (Branch-and-Bound)
        - Searches subsets of size m from a candidate pool of units
        - Uses quadratic relaxations of the loss surface for pruning
        - Efficiently explores a combinatorial design space of size C(M, m)
        - Returns a ranked set of top-K candidate experimental designs

    Stage 2: Synthetic Control Construction & Evaluation
        - Solves convex quadratic programs to compute synthetic control weights
        - Constructs synthetic treated and synthetic control time series
        - Computes treatment effects as differences between counterfactual paths
        - Evaluates in-sample and baseline fit using NMSE diagnostics

    Stage 3: Power Analysis (MDE)
        - Estimates Minimum Detectable Effect (MDE) using permutation inference
        - Approximates null distributions via Monte Carlo simulation
        - Computes detectability curves over varying post-treatment horizons
        - Quantifies statistical power as a function of experimental design

    OUTPUT STRUCTURE
    ----------------

    The final output is a Pareto-ranked set of experimental designs that trade off:

        - Pre-treatment fit quality (NMSE_B)
        - Statistical power (MDE curves)
        - Robustness across validation periods

    This enables selection of experimentally optimal unit configurations
    under constraints of limited treated units and observational data.

    DESIGN INTUITION
    ----------------

    The estimator is intended for settings where:

        - Treatment assignment is not predefined
        - Experimental units are large aggregate entities (e.g., regions, markets)
        - Only a small number of units can be assigned treatment
        - Randomization may induce baseline imbalance
        - Power considerations must be integrated into design selection

    NOTES
    -----

    - Deterministic given fixed random seed in MDE simulation
    - Computational cost is dominated by:
        (i) branch-and-bound combinatorial search
        (ii) repeated quadratic program solves
        (iii) Monte Carlo permutation inference
    - Designed for offline experimental design rather than real-time inference
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
        self.m: Optional[str] = config.m
        self.frac_E: Optional[str] = config.frac_E
        self.top_K: Optional[int] = config.top_K
        self.top_P: Optional[int] = config.top_P
        self.lambda_penalty = config.lambda_penalty

        

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

        unit_index = IndexSet.from_labels(
            sorted(working_df[self.unitid].unique())
        )


        time_index = IndexSet.from_labels(sorted(working_df[self.time].unique()))

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
            covariates=self.config.covariates,
            time=self.time,
            unitid=self.unitid, unit_index=unit_index
        )

        # Step 5: Build f weighting vector
        self.f = build_f_vector(
            working_df=working_df,
            weight_col=self.config.weight_col,
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
            lambda_penalty=self.lambda_penalty
        )


        for cand in candidate_results:
            treated_idx = np.asarray(cand.identification.treated_idx, dtype=int)

            cand.treated_units = unit_index.get_labels(treated_idx).tolist()

            cand.treated_unit_weights = dict(zip(
                unit_index.get_labels(treated_idx),
                cand.weights.treated
            ))

            cand.control_unit_weights = dict(zip(
                unit_index.labels,
                cand.weights.control
            ))

        candidate_mdes = run_mde_analysis(
            candidates=candidate_results,
            n_post_grid=self.config.n_post_grid,
            n_sims=self.config.n_sims
        )

        
        winner, shortlist = select_best_tuple(
            candidate_mdes,
            delta=0.015,           # or try 0.01 / 0.02
            relative_delta=1.5,    # I recommend starting with this for marketing
            target_mde_horizon="early_mde_avg",   # or "mde_6w" if you have a fixed test length
            return_shortlist=True
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
                    n_perms=self.config.n_sims,
                    seed=getattr(self.config, 'seed', 42)
                ).inference.p_value

                cand = compute_conformal_ci(
                    candidate=cand,
                    post_idx=post_idx,
                    alpha=0.05,
                    n_perms=self.config.n_sims,
                    seed=getattr(self.config, 'seed', 42)
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
            config=self.config,
            y_pop_mean_t=y_pop_mean_t,
            n_units=self.Y.shape[1],
            n_periods=T,
            n_pre_periods=len(E_idx) + len(B_idx),
            n_fit_periods=len(E_idx),
            n_blank_periods=len(B_idx),
            n_post_periods=len(post_idx)
        )

        return results
