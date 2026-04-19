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
    rank_candidates,
    run_mde_analysis,
)

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
    summary: pd.DataFrame             # The ranked Pareto/MDE table
    best_candidate: SEDCandidate      # The #1 ranked candidate object
    all_candidates: List[SEDCandidate]# Full list of evaluated tuples
    bnb_metadata: Dict[str, Any]      # Search stats (nodes visited, etc.)
    config: Any                       # Store the config used for the run

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

        X, f, candidate_idx, T, N = prepare_experiment_inputs(
            self.Y, self.Z, self.f, self.candidate_mask, self.m
        )

        T0 = int(self.frac_E * T)
        E_idx, B_idx, post_idx = split_periods(T0, T, self.frac_E)

        # Standardize over estimation period
        X_E, G = build_X_tilde(X, f, E_idx, J=self.Y.shape[1])

        # ------------------- Stage 1: BnB -------------------
        bbresults = branch_and_bound_topK(
            G, candidate_idx, m=self.m, top_K=self.top_K, top_P=self.top_P,total_units=N
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
            post_idx=post_idx,
            lambda_penalty=self.lambda_penalty
        )

        # 1. Run the power analysis (Stage 3)
        candidate_mdes = run_mde_analysis(
            candidates=candidate_results,
            n_post_grid=range(2, 13),
            n_sims=100
        )

        # 2. Rank them using the Pareto heuristic (default is 50/50 weight)
        ranked_df = rank_candidates(candidate_mdes, w_bias=0.5)
        
        # Identify the absolute best candidate based on the SED Score
        
        best_tuple_id = ranked_df.iloc[0]['tuple_id']
        best_candidate = next(c for c in candidate_results
                              if c.identification.tuple_id == best_tuple_id)

        # ------------------- Package Results -------------------
        results = LEXSCMResults(
            summary=ranked_df,
            best_candidate=best_candidate,
            all_candidates=candidate_results,
            bnb_metadata=bbresults,
            config=self.config
        )

        return results


