import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
import warnings
import pydantic
from mlsynth.utils.inferutils import ag_conformal
from ..utils.datautils import dataprep
from ..exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError, MlsynthPlottingError
from ..utils.resultutils import effects, plot_estimates, _create_RESCM_results
from ..utils.estutils import fit_l1inf_scm, fit_l2_scm
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
    l1inf: Any   # Can be the type returned by _create_estimator_results (Pydantic model)
    l2: Any      # Same as above


class RESCM:
    """
    Implements the Relaxed Balanced SCM

    Attributes
    ----------
    config : RESCMConfig
        The configuration object holding all parameters for the estimator.
    df : pd.DataFrame
        The input DataFrame containing panel data.
        (Inherited from `BaseEstimatorConfig` via `RESCMConfig`)
    outcome : str
        Name of the outcome variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `RESCMConfig`)
    treat : str
        Name of the treatment indicator column in `df`.
        (Inherited from `BaseEstimatorConfig` via `RESCMConfig`)
    unitid : str
        Name of the unit identifier column in `df`.
        (Inherited from `BaseEstimatorConfig` via `RESCM`)
    time : str
        Name of the time variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `RESCM`)
    display_graphs : bool, default True
        Whether to display graphs of results.
        (Inherited from `BaseEstimatorConfig` via `RESCM`)
    save : Union[bool, str, Dict[str, str]], default False
        Configuration for saving plots.
        - If `False` (default), plots are not saved.
        - If `True`, plots are saved with default names in the current directory.
        - If a `str`, it's used as the base filename for saved plots.
        - If a `Dict[str, str]`, it maps specific plot keys (e.g., "estimates_plot")
          to full file paths.
        (Inherited from `BaseEstimatorConfig` via `RESCM`)
    counterfactual_color : str, default "red"
        Color for the counterfactual line in plots.
        (Inherited from `BaseEstimatorConfig` via `RESCM`)
    treated_color : str, default "black"
        Color for the treated unit line in plots.
        (Inherited from `BaseEstimatorConfig` via `RESCM`)

    Methods
    -------
    fit()
        Fits the SHC model and returns standardized estimation results.

    References
    ----------

    Zhu, Rong J. B. "Synthetic Regressing Control Method." arXiv preprint arXiv:2306.02584 (2023).
    https://arxiv.org/abs/2306.02584
    """

    def __init__(self, config: RESCMConfig) -> None: # Changed to RESCM
        """
        Initializes the SHC estimator with a configuration object.

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
        """Fits the RESCM model with optional L2 Relaxed SCM and/or L1-INF SCM."""
        try:
            # Prepare panel data
            prepped: Dict[str, Any] = dataprep(
                self.df, self.unitid, self.time, self.outcome, self.treat
            )

            # Validate periods
            if prepped.get("pre_periods") is None or prepped.get("post_periods") is None:
                raise MlsynthDataError("Pre/post periods missing from dataprep output.")

            def extract_results(res, weight_key="Weights"):
                return {
                    "Effects": res['Results']['Effects'],
                    "Fit": res['Results']['Fit'],
                    "Vectors": res['Results']['Counterfactuals'],
                    weight_key: res.get('donor_dict', res.get('weights')),
                    "Model": res['Model']
                }

            l1res = None
            l2res = None

            # ---------------- Relaxed SCM (L2) ----------------
            relaxed_cfg = self.models_to_run.get("RELAXED", {"run": False})
            if relaxed_cfg.get("run", False):
                tau_val = relaxed_cfg.get("tau", None)
                n_splits_val = relaxed_cfg.get(
                    "n_splits",
                    max(1, int(np.floor(prepped["pre_periods"] ** 0.5)))
                )
                n_taus_val = relaxed_cfg.get("n_taus", 100)

                l2results = fit_l2_scm(
                    X_pre=prepped["donor_matrix"][:prepped["pre_periods"]].astype(np.float64),
                    y_pre=prepped["y"][:prepped["pre_periods"]].astype(np.float64).flatten(),
                    X_post=prepped["donor_matrix"][prepped["pre_periods"]:].astype(np.float64),
                    tau=tau_val,
                    n_splits=n_splits_val,
                    n_taus=n_taus_val,
                    y=prepped["y"],
                    donor_names=prepped["donor_names"]
                )


                l2res = _create_RESCM_results(
                    combined_raw_estimation_output=extract_results(l2results, weight_key="Weights"),
                    prepared_panel_data=prepped
                )

            # ---------------- ELASTIC SCM (L1-INF) ----------------
            elastic_cfg = self.models_to_run.get("ELASTIC", {"run": False})
            if elastic_cfg.get("run", False):
                n_splits_val = elastic_cfg.get(
                    "n_splits",
                    max(1, int(np.floor(2 * (prepped["pre_periods"] ** 0.5))))
                )
                print("Beginning Cross Validation for ELASTIC model...")
                l1infres = fit_l1inf_scm(
                    X_pre=prepped["donor_matrix"][:prepped["pre_periods"]].astype(np.float64),
                    y_pre=prepped["y"][:prepped["pre_periods"]].astype(np.float64).flatten(),
                    X_post=prepped["donor_matrix"][prepped["pre_periods"]:].astype(np.float64),
                    alpha_grid=np.linspace(0.0, 1.0, num=20),
                    intercept=True,
                    n_splits=n_splits_val,
                    n_repeats=1,
                    max_workers=4,
                    y=prepped["y"],
                    donor_names=prepped["donor_names"]
                )

                l1res = _create_RESCM_results(
                    combined_raw_estimation_output=extract_results(l1infres, weight_key="Weights"),
                    prepared_panel_data=prepped
                )

        except (MlsynthDataError, MlsynthConfigError, MlsynthEstimationError):
            raise
        except Exception as e:
            raise MlsynthEstimationError(f"Unexpected error during RESCM fit: {type(e).__name__}: {e}") from e

        # ---------------- Optional plotting ----------------
        if self.display_graphs:
            counterfactuals_to_plot = []
            cf_names = []
            if l2res is not None:
                counterfactuals_to_plot.append(l2res.time_series.counterfactual_outcome)
                cf_names.append("Relaxed SCM")
            if l1res is not None:
                counterfactuals_to_plot.append(l1res.time_series.counterfactual_outcome)
                cf_names.append("L1INF SCM")
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

        return EstimatorResults(l1inf=l1res, l2=l2res)
