import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

from ..utils.fast_scm_setup import (
    prepare_experiment_inputs, split_periods, build_X_tilde, package_scm_results,
    _prepare_working_df, build_f_vector, build_Y_matrix, build_Z_matrix, build_candidate_mask)
from ..utils.datautils import balance, dataprep
from ..config_models import LEXSCMConfig, BaseMAREXConfig
from ..exceptions import MlsynthDataError, MlsynthEstimationError



from ..utils.fast_scm_bb import branch_and_bound_topK


from ..utils.fast_scm_control import evaluate_candidates





class LEXSCM:
    """
    Synthetic Experiment Design estimator using fast tuple-based selection
    with branch-and-bound (BnB) and vectorized power/MDE analysis.

    This is the recommended class for cases where the treated units are **not known in advance**.
    It automatically searches for the best set of `m` treated units from the candidate pool,
    builds synthetic controls, runs placebo permutation tests on blank periods, and computes
    a Pareto front over NMSE and Minimum Detectable Effect (MDE).
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
        Run the fast synthetic experiment design pipeline.

        Parameters
        ----------
        treat : str, optional
            Column name for a binary treatment indicator (1 = already treated / ineligible for selection).
            If provided, candidate_mask will be built as ~treated.
        candidate_col : str, optional
            Column name containing a boolean or 0/1 mask for units eligible to be treated.
            Takes precedence over `treat` if both are provided.

        Returns
        -------
        self : LEXSCM
            Fitted estimator with results available in self.results and self.best.
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
        assert self.Y.shape[1] == len(self.candidate_mask), \
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
            G, candidate_idx, m=self.m, top_K=self.top_K, top_P=self.top_P
        )

        results = evaluate_candidates(
            bbresults['top_tuples'],
            X,
            X_E,
            self.Y,
            self.f,
            B_idx,
            post_idx,
            self.lambda_penalty
        )
        
        vbvb
