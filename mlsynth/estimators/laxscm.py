import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
import warnings
import pydantic
from mlsynth.utils.inferutils import ag_conformal
from ..utils.datautils import dataprep
from ..exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError, MlsynthPlottingError
from ..utils.resultutils import effects, plot_estimates, _create_RESCM_results
from ..utils.crossval import fit_en_scm, fit_relaxed_scm
from ..config_models import ( # Import Pydantic models
    RESCMConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
    InferenceResults,
    MethodDetailsResults,
)
from dataclasses import dataclass

@dataclass
class EstimatorResults:
    """Dataclass to store SCM estimation results for both L1-INF and L2 relaxation."""
    elastic: Any   # Can be the type returned by _create_estimator_results (Pydantic model)
    relax: Any      # Same as above


class RESCM:
    """
    Fits the Relaxed Balanced Synthetic Control Model (RESCM).

    This method supports multiple relaxation-based and elastic-net-style
    estimators for constructing counterfactual outcomes in panel data.

    The estimator selected depends on the configuration specified in
    `self.models_to_run`:

    Relaxed SCM Estimators (via Opt2.SCopt with `objective_type="relaxed"`):
    ----------------------------------------------------------------------
    - "l2": Standard Euclidean relaxation. Uses `l2_only_penalty` on the weights.
      No specific constraint type is required.

    - "entropy": Entropy-based relaxation. Encourages a high-entropy weight
      distribution. Requires sum-to-one constraints (`constraint_type` either
      "simplex" or "affine").

    - "el": Elementwise (max-norm) relaxation. Limits the largest weight deviation.
      Also requires sum-to-one constraints ("simplex" or "affine").

    Elastic-Net SCM Estimators (via Opt2.SCopt with `objective_type="penalized"`):
    -------------------------------------------------------------------------------
    - L1-L2 Elastic Net: Combines L1 and L2 penalties on the donor weights.
      Inspired by Doudchenko & Imbens (2017). The mixing parameter `alpha`
      controls the L1/L2 balance, `lambda` controls overall penalty strength.
      Optional intercept can be added via `fit_intercept=True`.

    - L1-L∞ Elastic Net: Combines L1 and L∞ penalties. Inspired by Wang et al. (2025).
      The `second_norm="linf"` parameter enables max-norm regularization
      alongside L1, with `alpha` controlling the L1/∞ mixture and `lambda`
      controlling overall shrinkage.

    Parameters
    ----------
    None

    Returns
    -------
    EstimatorResults
        A dataclass containing `l1inf` and `l2` results (depending on which
        models were run). Each result contains:
        - `time_series`: observed and counterfactual outcomes
        - `fit_diagnostics`: RMSE pre/post treatment, balance metrics, etc.

    Notes
    -----
    - The pre-treatment RMSE can be accessed via
      `arco.relax.fit_diagnostics.rmse_pre` for the relaxed SCM.
    - The choice of constraint type in `models_to_run` impacts which relaxation
      objectives are valid. For example, entropy and elementwise relaxation
      require sum-to-one constraints.
    - Under the hood, `Opt2.SCopt` constructs the convex optimization problem
      with `cvxpy`, sets the objective according to the chosen estimator, adds
      constraints, and solves for donor weights.

    References
    ----------
    Liao, C., Shi, Z., & Zheng, Y. (2025). A Relaxation Approach to Synthetic Control.
        arXiv preprint arXiv:2508.01793. https://arxiv.org/abs/2508.01793

    Wang, L., Xing, X., & Ye, Y. (2025). A L-infinity Norm Synthetic Control Approach.
        arXiv preprint arXiv:2510.26053. https://arxiv.org/abs/2510.26053

    Doudchenko, N., & Imbens, G. W. (2017). Balancing, Regression, Difference-In-Differences
        and Synthetic Control Methods: A Synthesis. arXiv preprint arXiv:1610.07748.
        https://arxiv.org/abs/1610.07748
    """

    def __init__(self, config: RESCMConfig) -> None: # Changed to RESCM
        """
        Initializes the Relaxed estimator with a configuration object.

        Parameters
        ----------

        config : RESCM
            A Pydantic model instance containing all configuration parameters
            for the ShC estimator. Since `RESCM` inherits directly from
            `BaseEstimatorConfig` without adding new fields, this includes:
            - df (pd.DataFrame): The input DataFrame.
            - outcome (str): Name of the outcome variable column.
            - treat (str): Name of the treatment indicator column.
            - unitid (str): Name of the unit identifier column.
            - time (str): Name of the time variable column.
            - display_graphs (bool, optional): Whether to display graphs. Defaults to True.
            - save (Union[bool, str, Dict[str, str]], optional): Configuration for saving plots.
              If `False` (default), plots are not saved. If `True`, plots are saved with
              default names. If a `str`, it's used as the base filename. If a `Dict[str, str]`,
              it maps plot keys to full file paths. Defaults to False.
            - counterfactual_color (str, optional): Color for counterfactual line. Defaults to "red".
            - treated_color (str, optional): Color for treated unit line. Defaults to "black".
        """
        if isinstance(config, dict):
            config = RESCMConfig(**config)  # convert dict to config object
        self.config = config # Store the config object
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color # Kept Union for flexibility
        self.treated_color: str = config.treated_color
        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, str] = config.save # Align with BaseEstimatorConfig

        self.models_to_run: Dict[str, Dict[str, Union[bool, float]]] = config.models_to_run

    def fit(self) -> EstimatorResults:
        """Fits the RESCM model using the new Elastic Net SCM and Relaxed SCM."""
        try:
            prepped: Dict[str, Any] = dataprep(
                self.df, self.unitid, self.time, self.outcome, self.treat
            )

            if prepped.get("pre_periods") is None or prepped.get("post_periods") is None:
                raise MlsynthDataError("Pre/post periods missing from dataprep output.")

            elasticres = None
            relaxres = None

            # ---------------- RELAXED SCM (L2) ----------------
            relaxed_cfg = self.models_to_run.get("RELAXED", {"run": False})
            if relaxed_cfg.get("run", False):
                tau_val = relaxed_cfg.get("tau", None)
                n_splits_val = relaxed_cfg.get(
                    "n_splits",
                    max(1, int(np.floor(prepped["pre_periods"] ** 0.5)))
                )
                J = prepped["donor_matrix"].shape[1]
                n_taus_val = relaxed_cfg.get("n_taus", int(np.floor(np.sqrt(np.floor(J)))*2))
                relaxation_method = relaxed_cfg.get("relaxation", "l2")

                relaxresults = fit_relaxed_scm(
                    X_pre=prepped["donor_matrix"][:prepped["pre_periods"]].astype(np.float64),
                    y_pre=prepped["y"][:prepped["pre_periods"]].astype(np.float64).flatten(),
                    X_post=prepped["donor_matrix"][prepped["pre_periods"]:].astype(np.float64),
                    tau=tau_val,
                    n_splits=n_splits_val,
                    n_taus=n_taus_val,
                    y=prepped["y"],
                    donor_names=prepped["donor_names"],
                    relaxation_type=relaxation_method
                )

                relaxres = _create_RESCM_results(
                    combined_raw_estimation_output=relaxresults,
                    prepared_panel_data=prepped
                )

            # ---------------- ELASTIC SCM (L1 / Elastic Net) ----------------
            elastic_cfg = self.models_to_run.get("ELASTIC", {"run": False})
            if elastic_cfg.get("run", False):
                n_splits_val = elastic_cfg.get(
                    "n_splits",
                    max(1, int(np.floor(prepped["pre_periods"] ** 0.5)))
                )
                alpha_val = elastic_cfg.get("alpha", None)
                lam_val = elastic_cfg.get("lambda", None)
                second_norm_val = elastic_cfg.get("enet_type", "L1_L2")

                elasticresults = fit_en_scm(
                    X_pre=prepped["donor_matrix"][:prepped["pre_periods"]],
                    y_pre=prepped["y"][:prepped["pre_periods"]].flatten(),
                    X_post=prepped["donor_matrix"][prepped["pre_periods"]:],
                    alpha=alpha_val,
                    lam=lam_val,
                    fit_intercept=elastic_cfg.get("fit_intercept", False),
                    standardize=elastic_cfg.get("standardize", True),
                    constraint_type=elastic_cfg.get("constraint_type", "unit"),
                    n_splits=n_splits_val,
                    second_norm=second_norm_val,
                    y=prepped["y"],
                    donor_names=prepped["donor_names"]
                )

                elasticres = _create_RESCM_results(
                    combined_raw_estimation_output=elasticresults,
                    prepared_panel_data=prepped
                )


        except Exception as e:
            raise MlsynthEstimationError(f"RESCM fit failed: {e}") from e

        # ---------------- Optional plotting ----------------
        if self.display_graphs:
            counterfactuals_to_plot = []
            cf_names = []

            if relaxres is not None:
                counterfactuals_to_plot.append(relaxres.time_series.counterfactual_outcome)
                cf_names.append(relaxresults['Model']+prepped['treated_unit_name'])

            if elasticres is not None:
                counterfactuals_to_plot.append(elasticres.time_series.counterfactual_outcome)
                cf_names.append(elasticresults['Model']+prepped['treated_unit_name'])

            try:
                plot_estimates(
                    processed_data_dict=prepped,
                    time_axis_label=self.time,
                    unit_identifier_column_name=self.unitid,
                    outcome_variable_label=self.outcome,
                    treatment_name_label=self.treat,
                    treated_unit_name=prepped["treated_unit_name"],
                    observed_outcome_series=prepped["y"],
                    counterfactual_series_list=counterfactuals_to_plot,
                    estimation_method_name="RESCM",
                    counterfactual_names=cf_names,
                    treated_series_color=self.treated_color,
                    counterfactual_series_colors=self.counterfactual_color,
                    save_plot_config=self.save
                )
            except Exception as e:
                warnings.warn(f"Plotting failed: {type(e).__name__}: {e}", UserWarning)

        return EstimatorResults(elastic=elasticres, relax=relaxres)

