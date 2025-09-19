import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
import warnings
import pydantic # For ValidationError
import cvxpy as cp # For cvxpy.error types

from ..utils.datautils import balance
from ..utils.exputils import _get_per_cluster_param, SCMEXP
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)

from ..config_models import ClusterResults, MAREXResults, MAREXConfig, GlobalResults, StudyConfig


class MAREX:

    def __init__(self, config: MAREXConfig) -> None: # Changed to MAREXConfig

        if isinstance(config, dict):
            config =MAREXConfig(**config)  # convert dict to config object
        # Panel data

        # Panel data
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time

        # Optional design parameters
        self.T0: Optional[int] = config.T0
        self.cluster: str = config.cluster
        self.design: str = config.design

        # Penalization parameters
        self.beta: float = config.beta
        self.lambda1: float = config.lambda1
        self.lambda2: float = config.lambda2
        self.xi: float = config.xi
        self.lambda1_unit: float = config.lambda1_unit
        self.lambda2_unit: float = config.lambda2_unit

        # Additional SCMEXP options
        self.blank_periods: int = getattr(config, "blank_periods", 0)
        self.m_eq: Optional[int] = getattr(config, "m_eq", 1)
        self.m_min: Optional[int] = getattr(config, "m_min", 1)
        self.m_max: Optional[int] = getattr(config, "m_max", 1)
        self.exclusive: bool = getattr(config, "exclusive", True)
        self.solver = getattr(config, "solver", None)
        self.verbose: bool = getattr(config, "verbose", False)

        # Validate cluster column if provided
        if self.cluster and self.cluster not in self.df.columns:
            raise MlsynthDataError(f"Cluster column '{self.cluster}' not found in DataFrame.")

    class DesignResultsProcessor:
        """Process raw SCMEXP output into structured MAREXResults."""

        def __init__(self, scm_result: dict, beta: float, lambda1: float, lambda2: float, xi: float, design: str):
            self.scm_result = scm_result
            self.beta = beta
            self.lambda1 = lambda1
            self.lambda2 = lambda2
            self.xi = xi
            self.design = design

        def get_results(self) -> MAREXResults:
            clusters_out = {}

            w_opt = self.scm_result.get("w_opt")
            v_opt = self.scm_result.get("v_opt")
            z_opt = self.scm_result.get("z_opt")
            Y_fit = self.scm_result.get("Y_fit")
            Y_blank = self.scm_result.get("Y_blank")
            Y_full = self.scm_result.get("Y_full")

            clusters_vector = self.scm_result.get("original_cluster_vector")
            unit_labels = self.scm_result['df'].index.to_list()
            T0 = self.scm_result.get('T0', Y_fit.shape[1] if Y_fit is not None else 0)
            unique_clusters = np.unique(clusters_vector)

            for c in unique_clusters:
                cluster_units_idx = np.where(clusters_vector == c)[0]
                members = [unit_labels[i] for i in cluster_units_idx]
                selection_indicators = z_opt[cluster_units_idx, c] if z_opt is not None else np.zeros(
                    len(cluster_units_idx))
                treated_weights = np.where(
                    selection_indicators == 1,
                    w_opt[cluster_units_idx, c],  # pick only column c
                    0.0
                )
                control_weights = np.where(
                    selection_indicators == 0,
                    v_opt[cluster_units_idx, c],
                    0.0
                )

                unit_weight_map = {
                    "Treated": {
                        members[i]: treated_weights[i]
                        for i in range(len(members))
                        if selection_indicators[i] == 1
                    },
                    "Control": {
                        members[i]: control_weights[i]
                        for i in range(len(members))
                        if selection_indicators[i] == 0 and control_weights[i] > 0
                    }
                }



                pre_means = self.scm_result['Xbar_clusters'][c]
                synthetic_treated, synthetic_control = [
                    self.scm_result[f'y_syn_{grp}_clusters'][c][:T0] for grp in ['treated', 'control']
                ]

                clusters_out[str(c)] = ClusterResults(
                    members=members,
                    cluster_cardinality=len(members),
                    synthetic_treated=synthetic_treated,
                    synthetic_control=synthetic_control,
                    treated_weights=treated_weights,
                    control_weights=control_weights,
                    selection_indicators=selection_indicators,
                    pre_treatment_means=pre_means,
                    rmse=self.scm_result['rmse_cluster'][c],
                    unit_weight_map=unit_weight_map
                )

            # Global pre-treatment results
            w_agg = self.scm_result['w_agg']
            v_agg = self.scm_result['v_agg']
            z_diag = np.array([z_opt[i, c] for i, c in enumerate(clusters_vector)])
            adjusted_treated_weights_agg = np.where(z_diag == 1, w_agg, 0.0)
            adjusted_control_weights_agg = np.where(z_diag == 1, 0.0, v_agg)

            study_config = StudyConfig(
                beta=self.beta,
                lambda1=self.lambda1,
                lambda2=self.lambda2,
                xi=self.xi,
                T0=T0,
                blank_periods=self.scm_result.get("blank_periods", 0),
                design=self.design
            )

            global_pre_results = GlobalResults(
                Y_fit=Y_fit,
                Y_blank=Y_blank,
                Y_full=Y_full,
                treated_weights_agg=adjusted_treated_weights_agg,
                control_weights_agg=adjusted_control_weights_agg
            )

            return MAREXResults(
                clusters=clusters_out,
                study=study_config,
                globres=global_pre_results
            )



    def fit(self) -> MAREXResults:
        # Ensure the panel data is balanced
        balance(self.df, self.unitid, self.time)

        # Validate treatment specification
        if self.m_eq is not None and (self.m_min is not None or self.m_max is not None):
            raise MlsynthConfigError(
                "Cannot specify both 'm_eq' and 'm_min/m_max' at the same time. "
                "Choose either an exact number of treated units per cluster or a range."
            )
        if self.m_eq is None and self.m_min is None and self.m_max is None:
            raise MlsynthConfigError(
                "You must specify either 'm_eq' or at least one of 'm_min'/'m_max' to define treated units."
            )

        # Define unit_labels unconditionally
        unit_labels = self.df[self.unitid].unique()

        # Handle clusters based on whether cluster is specified
        if self.cluster is not None:
            clusters = self.df.drop_duplicates(subset=[self.unitid]).set_index(self.unitid)[self.cluster].reindex(
                unit_labels).to_numpy()
        else:
            clusters = np.zeros(len(unit_labels), dtype=int)

        # Reshape data: units as rows, time as columns
        Y_full = self.df.pivot(index=self.unitid, columns=self.time, values=self.outcome).reindex(unit_labels)

        # Determine total time and T0
        T_total = len(self.df[self.time].unique())
        T0 = self.T0 if self.T0 is not None else T_total - 1  # default T0 = T-1
    
        # Determine blank_periods
        if hasattr(self, "blank_periods") and self.blank_periods is not None:
            blanks = self.blank_periods
        else:
            blanks = T_total - T0  # default blank periods = post-treatment periods
    
        # Validate blank_periods
        if not (0 <= blanks < T0):
            raise ValueError(
                f"blank_periods must be 0 <= blank_periods < T0 (T0={T0}, got blank_periods={blanks})"
            )
        
        # Prepare SCMEXP arguments
        scm_kwargs = dict(
            Y_full=Y_full,
            T0=T0,
            clusters=clusters,
            blank_periods=blanks,
            design=self.design,
            beta=self.beta,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            xi=self.xi,
            lambda1_unit=self.lambda1_unit,
            lambda2_unit=self.lambda2_unit,
            solver=self.solver or cp.ECOS_BB,
            verbose=self.verbose,
            exclusive=self.exclusive,
        )

        if self.m_eq is not None:
            scm_kwargs["m_eq"] = self.m_eq
        else:
            scm_kwargs["m_min"] = self.m_min
            scm_kwargs["m_max"] = self.m_max

        # Run SCMEXP
        raw_results = SCMEXP(**scm_kwargs)

        # Process results with DesignResultsProcessor
        processor = self.DesignResultsProcessor(
            scm_result=raw_results,
            beta=self.beta,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            xi=self.xi,
            design=self.design
        )

        marex_results = processor.get_results()

        return marex_results







