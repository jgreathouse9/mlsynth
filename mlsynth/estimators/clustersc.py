import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional

from ..utils.datautils import balance, dataprep
from ..utils.resultutils import effects, plot_estimates # effects is used in fit
from ..utils.estutils import pcr, RPCASYNTH
from ..config_models import ( # Import the Pydantic model
    CLUSTERSCConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
    InferenceResults,
    MethodDetailsResults,
)

from ..exceptions import MlsynthDataError, MlsynthConfigError, MlsynthEstimationError, MlsynthPlottingError
from pydantic import ValidationError


class CLUSTERSC:
    """Implements Cluster-based Synthetic Control methods.

    This estimator provides Average Treatment Effect on the Treated (ATT)
    estimates and donor weights using two main approaches: Principal Component
    Regression (PCR) and Robust PCA Synthetic Control (RPCA SCM).
    The PCR method can be run in Frequentist (Robust Synthetic Control - RSC)
    or Bayesian (Bayesian RSC) mode. RPCA SCM can use Principal Component
    Pursuit (PCP) or Half-Quadratic Regularization (HQF).

    The estimator can run either method ("PCR", "RPCA") individually or both
    ("BOTH") for comparison. It operates on panel data and requires specifying
    outcome, treatment, unit, and time identifiers. Configuration is managed
    via the `CLUSTERSCConfig` Pydantic model.

    Attributes
    ----------
    config : CLUSTERSCConfig
        The configuration object passed during initialization, containing all
        parameters for the estimator.
    df : pd.DataFrame
        The input panel data.
    outcome : str
        Name of the outcome variable column.
    treat : str
        Name of the binary treatment indicator column.
    unitid : str
        Name of the unit identifier column.
    time : str
        Name of the time period column.
    method : str
        Estimation method to use ("PCR", "RPCA", "BOTH").
    objective : str
        Objective function for PCR weight optimization.
    cluster : bool
        Whether to apply clustering before PCR.
    Frequentist : bool
        If True, use Frequentist RSC for PCR; otherwise, Bayesian RSC.
    ROB : str
        Robust PCA decomposition method for RPCA ("PCP" or "HQF").
    display_graphs : bool
        Whether to display plots of results.
    save : Union[bool, Dict[str, str]]
        Plot saving configuration.
    counterfactual_color : Union[str, List[str]]
        Color(s) for counterfactual line(s) in plots.
    treated_color : str
        Color for the treated unit line in plots.

    Methods
    -------
    fit()
        Fits the model according to the specified method and returns results.

    References
    ----------
    Amjad, M., Shah, D., & Shen, D. (2018). "Robust synthetic control."
    *Journal of Machine Learning Research*, 19(22), 1-51.
    Agarwal, A., Shah, D., Shen, D., & Song, D. (2021). "On Robustness of Principal Component Regression."
    *Journal of the American Statistical Association*, 116(536), 1731–45.
    Bayani, M. (2022). "Essays on Machine Learning Methods in Economics." Chapter 1.
    *CUNY Academic Works*.
    Wang, Z., Li, X. P., So, H. C., & Liu, Z. (2023). "Robust PCA via non-convex half-quadratic regularization."
    *Signal Processing*, 204, 108816.
    """

    def __init__(self, config: CLUSTERSCConfig) -> None:
        """Initializes the CLUSTERSC estimator with a configuration object.

        All parameters are passed via the `config` argument, which is an
        instance of `CLUSTERSCConfig`. These parameters are then stored as
        attributes of the estimator instance.

        Parameters
        ----------
        config : CLUSTERSCConfig
            A Pydantic model instance containing all configuration parameters.
            Key fields include:
            - `df` (pd.DataFrame): Input panel data.
            - `outcome` (str): Name of the outcome variable column.
            - `treat` (str): Name of the treatment indicator column.
            - `unitid` (str): Name of the unit identifier column.
            - `time` (str): Name of the time period column.
            - `method` (str): Estimation method: "PCR", "RPCA", or "BOTH".
              Default is "PCR".
            - `objective` (str): For PCR, objective for weight optimization
              (e.g., "OLS", "SIMPLEX"). Default is "OLS".
            - `cluster` (bool): For PCR, whether to apply clustering. Default is True.
            - `Frequentist` (bool): For PCR, True for Frequentist RSC, False for
              Bayesian RSC. Default is True.
            - `ROB` (str): For RPCA, robust PCA method ("PCP" or "HQF").
              Default is "PCP".
            - `display_graphs` (bool): Whether to display plots. Default is True.
            - `save` (Union[bool, str, Dict[str, str]]): Plot saving configuration.
              Default is False.
            - `counterfactual_color` (Union[str, List[str]]): Color(s) for
              counterfactual lines. Default is "red".
            - `treated_color` (str): Color for the treated unit line.
              Default is "black".
            Refer to `CLUSTERSCConfig` and `BaseEstimatorConfig` in
            `mlsynth.config_models` for all available fields and their defaults.
        """
        if isinstance(config, dict):
            config = CLUSTERSCConfig(**config)  # convert dict to config object
        self.config = config # Store the config object
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color # Kept Union for flexibility
        self.treated_color: str = config.treated_color
        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, Dict[str, str]] = config.save # Kept Union for flexibility
        self.objective: str = config.objective
        self.cluster: bool = config.cluster
        self.Frequentist: bool = config.Frequentist
        self.ROB: str = config.ROB
        self.method: str = config.method # Pydantic model ensures it's one of "PCR", "RPCA", "BOTH" (uppercase)

    def _create_single_method_results(
        self,
        raw_results: Dict[str, Any],
        method_name: str,
        prepared_data: Dict[str, Any],
        is_frequentist_pcr: Optional[bool] = None
    ) -> BaseEstimatorResults:
        """Constructs a `BaseEstimatorResults` object for a single method (PCR or RPCA).

        This internal helper takes raw estimation outputs and structures them
        into the standardized Pydantic result models. It handles differences
        in output keys and structures between PCR and RPCA results.

        Parameters
        ----------
        raw_results : Dict[str, Any]
            A dictionary containing the raw outputs from an estimation function
            (e.g., `pcr` or `RPCASYNTH`). Expected keys might include "Effects",
            "Fit", "Vectors", "Weights" (for PCR) or "weights" (for RPCA),
            and optionally "credible_interval".
        method_name : str
            The name of the method for which results are being created,
            typically "PCR" or "RPCA". This influences how some fields in
            `MethodDetailsResults` are populated.
        prepared_data : Dict[str, Any]
            The dictionary returned by `dataprep`, containing processed data like
            `Ywide` (for time periods) and `donor_names`.
        is_frequentist_pcr : Optional[bool], default=None
            Specifies if the PCR method was run in Frequentist mode. This is
            used to correctly label the method in `MethodDetailsResults` (e.g.,
            "RSC" vs "Bayesian RSC") and for populating inference details.

        Returns
        -------
        BaseEstimatorResults
            A Pydantic model instance populated with the structured results
            for the specified `method_name`. This includes effects, fit
            diagnostics, time series data, donor weights, inference details
            (if any), and method-specific details.
        """
        
        # Populate EffectsResults: Extracts ATT, ATT percentage, and stores the full raw "Effects" dictionary.
        effects_results_obj = EffectsResults(
            att=raw_results.get("Effects", {}).get("ATT"),
            att_percent=raw_results.get("Effects", {}).get("ATT_perc"),
            additional_effects=raw_results.get("Effects", {}) # Store the whole dict
        )
        
        # Populate FitDiagnosticsResults: Extracts pre-treatment RMSE and R-squared, stores full raw "Fit" dictionary.
        # Note: effects.calculate utility returns "T0 RMSE" and "T0 R2" keys.
        fit_diagnostics_results_obj = FitDiagnosticsResults(
            pre_treatment_rmse=raw_results.get("Fit", {}).get("T0 RMSE"), 
            pre_treatment_r_squared=raw_results.get("Fit", {}).get("T0 R2"), 
            additional_metrics=raw_results.get("Fit", {}) # Store the whole dict
        )

        # Extract time periods from the wide-format outcome DataFrame prepared by dataprep.
        outcome_wide_df = prepared_data.get("Ywide")
        time_periods_array: Optional[np.ndarray] = None
        if isinstance(outcome_wide_df, pd.DataFrame):
            time_periods_array = outcome_wide_df.index.to_numpy() # Use DataFrame index as time periods.
            
        # Populate TimeSeriesResults: Extracts observed outcome, counterfactual, gap, and time periods.
        time_series_results_obj = TimeSeriesResults(
            observed_outcome=raw_results.get("Vectors", {}).get("Observed Unit"),
            counterfactual_outcome=raw_results.get("Vectors", {}).get("Counterfactual"),
            estimated_gap=raw_results.get("Vectors", {}).get("Gap"),
            time_periods=time_periods_array
        )

        # Initialize variables for donor weights and any additional weight-related outputs.
        donor_weights_map = None
        additional_w_outputs_dict = None
        
        # Handle donor weights differently based on whether the method is PCR or RPCA.
        # PCR raw results for "Weights" is expected to be a list: [donor_weights_dict, non_zero_weights_dict].
        if method_name == "PCR":
            raw_pcr_weights_package = raw_results.get("Weights")
            if isinstance(raw_pcr_weights_package, list) and len(raw_pcr_weights_package) > 0:
                raw_pcr_donor_weights = raw_pcr_weights_package[0] # First element is the main donor weights.
                if isinstance(raw_pcr_donor_weights, dict):
                    donor_weights_map = {str(k): v for k, v in raw_pcr_donor_weights.items()} # Ensure keys are strings.
                
                # Second element (if present) contains non-zero weights for PCR.
                if len(raw_pcr_weights_package) > 1 and isinstance(raw_pcr_weights_package[1], dict):
                    raw_pcr_non_zero_weights = raw_pcr_weights_package[1]
                    additional_w_outputs_dict = {
                        "non_zero_weights": {str(k): v for k, v in raw_pcr_non_zero_weights.items()}
                    }
        # RPCA raw results for "weights" is expected to be a dictionary directly.
        elif method_name == "RPCA":
            raw_rpca_weights_package = raw_results.get("weights")
            if isinstance(raw_rpca_weights_package, dict):
                donor_weights_map = {str(k): v for k, v in raw_rpca_weights_package.items()} # Ensure keys are strings.
        
        # Populate WeightsResults with the processed donor weights.
        weights_results_obj = WeightsResults(
            donor_weights=donor_weights_map, 
            additional_outputs=additional_w_outputs_dict
        )

        # Populate InferenceResults if Bayesian credible intervals are present (typically for Bayesian PCR).
        inference_results_obj = None
        if "credible_interval" in raw_results:
            inference_results_obj = InferenceResults(
                details={"credible_interval": raw_results["credible_interval"]},
                method="Bayesian Credible Interval" if method_name == "PCR" and not is_frequentist_pcr else None
            )
        
        # Determine the final, more descriptive method name for reporting in results.
        final_method_name_in_results = method_name
        if method_name == "PCR":
            final_method_name_in_results = "Bayesian RSC" if not is_frequentist_pcr else "RSC" # Distinguish Frequentist/Bayesian PCR.
        elif method_name == "RPCA":
            final_method_name_in_results = "RPCA Synth" # Standard name for RPCA SCM.

        # Populate MethodDetailsResults with the method name and key parameters used.
        method_details_results_obj = MethodDetailsResults(
            name=final_method_name_in_results,
            parameters_used={ # Store relevant parameters based on the method.
                "objective": self.objective if method_name == "PCR" else None,
                "cluster": self.cluster if method_name == "PCR" else None,
                "Frequentist": is_frequentist_pcr if method_name == "PCR" else None,
                "ROB": self.ROB if method_name == "RPCA" else None,
            }
        )
        
        # Assemble and return the final BaseEstimatorResults object.
        return BaseEstimatorResults(
            effects=effects_results_obj,
            fit_diagnostics=fit_diagnostics_results_obj,
            time_series=time_series_results_obj,
            weights=weights_results_obj,
            inference=inference_results_obj,
            method_details=method_details_results_obj
        )

    def fit(self) -> BaseEstimatorResults:
        """Fits the CLUSTERSC model using the specified method (PCR, RPCA, or BOTH).

        The fitting process involves:
        1. Balancing the input panel data using `mlsynth.utils.datautils.balance`.
        2. Preprocessing data into a matrix format using `mlsynth.utils.datautils.dataprep`.
        3. Running the selected estimation method(s):
           - If `method` is "PCR" or "BOTH": Principal Component Regression is
             performed using `mlsynth.utils.estutils.pcr`. Effects are calculated.
           - If `method` is "RPCA" or "BOTH": Robust PCA Synthetic Control is
             performed using `mlsynth.utils.estutils.RPCASYNTH`.
        4. Consolidating results from each method into a `BaseEstimatorResults`
           object, where individual method results are stored in the
           `sub_method_results` dictionary.
        5. Optionally displaying plots of the observed vs. counterfactual outcomes.

        Returns
        -------
        BaseEstimatorResults
            A Pydantic model instance containing the standardized estimation results.
            This top-level object primarily serves as a container for results from
            the sub-methods (PCR and/or RPCA) if `method` is "PCR", "RPCA", or "BOTH".
            The detailed results for each executed method are stored in the
            `sub_method_results` attribute, which is a dictionary.
            - `sub_method_results` (Dict[str, BaseEstimatorResults]):
              A dictionary where keys are method names (e.g., "PCR", "RPCA") and
              values are `BaseEstimatorResults` Pydantic models containing the
              specific outputs for that method.

            Each nested `BaseEstimatorResults` object (for PCR or RPCA) includes:
            - `effects` (`EffectsResults`): Contains ATT, ATT percentage, and any
              other effect-related metrics.
            - `fit_diagnostics` (`FitDiagnosticsResults`): Includes pre-treatment
              RMSE, R-squared, and other fit statistics.
            - `time_series` (`TimeSeriesResults`): Holds observed outcomes,
              counterfactual outcomes, the estimated gap, and time periods.
            - `weights` (`WeightsResults`): Contains donor weights. For PCR,
              `additional_outputs` may include non-zero weights.
            - `inference` (`Optional[InferenceResults]`): For Bayesian PCR, this
              may contain credible interval details.
            - `method_details` (`MethodDetailsResults`): Specifies the exact
              method name (e.g., "RSC", "Bayesian RSC", "RPCA Synth") and key
              parameters used for that sub-method.

        Examples
        --------
        >>> from mlsynth import CLUSTERSC
        >>> from mlsynth.config_models import CLUSTERSCConfig
        >>> import pandas as pd
        >>> import numpy as np
        >>> # Create sample data
        >>> data = pd.DataFrame({
        ...     'unit': np.repeat(np.arange(1, 11), 20),
        ...     'time': np.tile(np.arange(1, 21), 10),
        ...     'outcome': np.random.rand(200) + np.repeat(np.arange(0,10),20)*0.1,
        ...     'treated': ((np.repeat(np.arange(1, 11), 20) == 1) & (np.tile(np.arange(1, 21), 10) >= 15)).astype(int)
        ... })
        >>> # Configure and run PCR method
        >>> config_pcr = CLUSTERSCConfig(
        ...     df=data, outcome='outcome', treat='treated', unitid='unit', time='time',
        ...     method="PCR", objective="OLS", cluster=True, Frequentist=True,
        ...     display_graphs=False
        ... )
        >>> estimator_pcr = CLUSTERSC(config=config_pcr)
        >>> results_pcr_container = estimator_pcr.fit()
        >>> pcr_results = results_pcr_container.sub_method_results["PCR"]
        >>> print(f"PCR ATT: {pcr_results.effects.att}")

        >>> # Configure and run RPCA method
        >>> config_rpca = CLUSTERSCConfig(
        ...     df=data, outcome='outcome', treat='treated', unitid='unit', time='time',
        ...     method="RPCA", ROB="PCP", display_graphs=False
        ... )
        >>> estimator_rpca = CLUSTERSC(config=config_rpca)
        >>> results_rpca_container = estimator_rpca.fit()
        >>> rpca_results = results_rpca_container.sub_method_results["RPCA"]
        >>> print(f"RPCA ATT: {rpca_results.effects.att}")

        >>> # Configure and run BOTH methods
        >>> config_both = CLUSTERSCConfig(
        ...     df=data, outcome='outcome', treat='treated', unitid='unit', time='time',
        ...     method="BOTH", objective="SIMPLEX", ROB="HQF", display_graphs=False
        ... )
        >>> estimator_both = CLUSTERSC(config=config_both)
        >>> results_both_container = estimator_both.fit()
        >>> pcr_both_results = results_both_container.sub_method_results["PCR"]
        >>> rpca_both_results = results_both_container.sub_method_results["RPCA"]
        >>> print(f"BOTH - PCR ATT: {pcr_both_results.effects.att}")
        >>> print(f"BOTH - RPCA ATT: {rpca_both_results.effects.att}")
        """
        # Balance the panel data to ensure consistent time periods across units.
        # This step can raise MlsynthDataError if data issues are found (e.g., inconsistent time periods).
        balance(self.df, self.unitid, self.time) 

        # Prepare data into matrix format (X0, X1, Y0, Y1, etc.) required by estimation utilities.
        # This step can also raise MlsynthDataError for issues like no treated/control units, insufficient pre-periods, etc.
        prepared_data: Dict[str, Any] = dataprep( 
            self.df, self.unitid, self.time, self.outcome, self.treat
        )

        # Initialize containers for raw results from PCR and RPCA methods.
        raw_results_pcr: Optional[Dict[str, Any]] = None
        raw_results_rpca: Optional[Dict[str, Any]] = None
        # Initialize the top-level results object. Sub-method results will be added to its `sub_method_results` dict.
        final_results = BaseEstimatorResults(sub_method_results={}) 

        try:
            # --- Run Estimation based on the configured method ---
            # If method is "PCR" or "BOTH", execute the Principal Component Regression.
            if self.method.upper() == "PCR" or self.method.upper() == "BOTH":
                # Call the pcr utility function with prepared data and configuration.
                # This can raise MlsynthDataError, MlsynthConfigError, or MlsynthEstimationError.
                pcr_output: Dict[str, Any] = pcr( 
                    donor_outcomes_matrix=prepared_data["donor_matrix"],
                    treated_unit_outcome_vector=prepared_data["y"],
                    scm_objective_model_type=self.objective,
                    all_donor_names=list(prepared_data["donor_names"]) if prepared_data.get("donor_names") is not None else [],
                    num_pre_treatment_periods=prepared_data["pre_periods"],
                    enable_clustering=self.cluster,
                    use_frequentist_scm=self.Frequentist,
                )
                # Calculate effects, fit diagnostics, and time series components from PCR output.
                calculated_effects, calculated_fit_diagnostics, calculated_time_series_components = effects.calculate(
                    prepared_data["y"], # Observed outcome for the treated unit.
                    pcr_output["cf_mean"], # Counterfactual outcome from PCR.
                    prepared_data["pre_periods"],
                    prepared_data["post_periods"],
                )
                # Structure the raw PCR results into a dictionary.
                raw_results_pcr = {
                    "Effects": calculated_effects, 
                    "Fit": calculated_fit_diagnostics, 
                    "Vectors": calculated_time_series_components,
                    # Store both raw weights and rounded non-zero weights for PCR.
                    "Weights": [pcr_output["weights"], {k: round(v, 3) for k, v in pcr_output["weights"].items() if v != 0}],
                }
                # If Bayesian PCR was run (not Frequentist) and credible intervals are available, add them.
                if not self.Frequentist and "credible_interval" in pcr_output:
                    raw_results_pcr["credible_interval"] = pcr_output["credible_interval"]

            # If method is "RPCA" or "BOTH", execute Robust PCA Synthetic Control.
            if self.method.upper() == "RPCA" or self.method.upper() == "BOTH":
                # Call the RPCASYNTH utility function.
                # This can raise MlsynthDataError, MlsynthConfigError, or MlsynthEstimationError.
                raw_results_rpca = RPCASYNTH(self.df, self.config.model_dump(), prepared_data)

            # --- Construct Pydantic Results from raw outputs ---
            # If PCR was run, create its standardized results object.
            if raw_results_pcr:
                final_results.sub_method_results["PCR"] = self._create_single_method_results(
                    raw_results_pcr, "PCR", prepared_data, self.Frequentist
                )
            
            # If RPCA was run, create its standardized results object.
            if raw_results_rpca:
                final_results.sub_method_results["RPCA"] = self._create_single_method_results(
                    raw_results_rpca, "RPCA", prepared_data
                )
        
        # --- Error Handling for the estimation block ---
        except (MlsynthDataError, MlsynthConfigError, MlsynthEstimationError) as e:
            # Re-raise known custom errors from utility functions or direct raises within this method.
            raise e
        except ValidationError as e: # Catch Pydantic validation errors during results model creation.
            raise MlsynthEstimationError(f"Error creating results structure: {str(e)}") from e
        except KeyError as e: # Catch missing keys if .get() was not used for a critical piece of data.
            raise MlsynthEstimationError(f"Missing expected key in estimation results: {str(e)}") from e
        except Exception as e:
            # Wrap any other unexpected errors from the estimation process.
            raise MlsynthEstimationError(f"An unexpected error occurred during CLUSTERSC estimation: {str(e)}") from e
        
        # --- Plotting (if enabled) ---
        if self.display_graphs: # Only proceed if display_graphs is True.
            # Initialize containers for plotting data.
            counterfactual_outcomes_for_plot: Dict[str, np.ndarray] = {} # Stores CF outcomes by method name.
            legend_names_for_counterfactuals: List[str] = [] # Stores legend names for each CF line.
            counterfactual_vectors_for_plot: List[np.ndarray] = [] # List of CF vectors to plot.

            # Collect PCR counterfactual outcome and legend name if PCR results exist.
            pcr_results_object = final_results.sub_method_results.get("PCR")
            if pcr_results_object and pcr_results_object.time_series and pcr_results_object.time_series.counterfactual_outcome is not None:
                counterfactual_outcomes_for_plot["PCR"] = pcr_results_object.time_series.counterfactual_outcome
                # Determine legend name for PCR based on Frequentist/Bayesian mode.
                pcr_legend_name = "Bayesian RSC" if not self.Frequentist and self.method != "RPCA" else "RSC"
                if self.method == "PCR" or self.method == "BOTH": # Add to legend only if PCR was a requested method.
                     legend_names_for_counterfactuals.append(pcr_legend_name)
            
            # Collect RPCA counterfactual outcome and legend name if RPCA results exist.
            rpca_results_object = final_results.sub_method_results.get("RPCA")
            if rpca_results_object and rpca_results_object.time_series and rpca_results_object.time_series.counterfactual_outcome is not None:
                counterfactual_outcomes_for_plot["RPCA"] = rpca_results_object.time_series.counterfactual_outcome
                if self.method == "RPCA" or self.method == "BOTH": # Add to legend only if RPCA was a requested method.
                     legend_names_for_counterfactuals.append("RPCA Synth")

            # Assemble the list of counterfactual vectors to be plotted based on the chosen method.
            if self.method == "BOTH": # If both methods were run, plot both CFs.
                if "PCR" in counterfactual_outcomes_for_plot:
                     counterfactual_vectors_for_plot.append(counterfactual_outcomes_for_plot["PCR"])
                if "RPCA" in counterfactual_outcomes_for_plot:
                     counterfactual_vectors_for_plot.append(counterfactual_outcomes_for_plot["RPCA"])
            elif self.method == "PCR" and "PCR" in counterfactual_outcomes_for_plot: # If only PCR, plot PCR CF.
                counterfactual_vectors_for_plot.append(counterfactual_outcomes_for_plot["PCR"])
            elif self.method == "RPCA" and "RPCA" in counterfactual_outcomes_for_plot: # If only RPCA, plot RPCA CF.
                counterfactual_vectors_for_plot.append(counterfactual_outcomes_for_plot["RPCA"])

            # Proceed with plotting only if there are valid counterfactual vectors.
            if counterfactual_vectors_for_plot: 
                # Determine colors for counterfactual lines. Handles single string or list of strings for colors.
                counterfactual_line_colors: List[str] = (
                    [self.counterfactual_color] 
                    if isinstance(self.counterfactual_color, str)
                    else self.counterfactual_color 
                )
                # Adjust color list if only one CF is plotted but multiple colors were provided, or vice-versa.
                if len(counterfactual_vectors_for_plot) == 1 and isinstance(counterfactual_line_colors, list) and counterfactual_line_colors:
                    counterfactual_line_colors = [counterfactual_line_colors[0]] # Use first color if one CF.
                elif len(counterfactual_vectors_for_plot) > 1 and isinstance(self.counterfactual_color, str):
                    # Repeat single color if multiple CFs but only one color string given.
                    counterfactual_line_colors = [self.counterfactual_color] * len(counterfactual_vectors_for_plot)
                
                try:
                    # Call the generic plot_estimates utility.
                    # This can raise MlsynthPlottingError or MlsynthDataError.
                    plot_estimates( 
                        processed_data_dict=prepared_data, 
                        time_axis_label=self.time,
                        unit_identifier_column_name=self.unitid,
                        outcome_variable_label=self.outcome,
                        treatment_name_label=self.treat,
                        treated_unit_name=prepared_data["treated_unit_name"],
                        observed_outcome_series=prepared_data["y"], # Observed outcome vector.
                        counterfactual_series_list=[cf.flatten() for cf in counterfactual_vectors_for_plot], # List of counterfactual vectors.
                        estimation_method_name="CLUSTERSC",
                        counterfactual_names=legend_names_for_counterfactuals, # Names for legend.
                        treated_series_color=self.treated_color,
                        counterfactual_series_colors=counterfactual_line_colors,
                        save_plot_config=self.save, 
                    )
                except (MlsynthPlottingError, MlsynthDataError) as e: # Catch known plotting-related errors.
                    # Issue a warning if plotting fails but allow the rest of the results to be returned.
                    print(f"Warning: Plotting failed for CLUSTERSC - {str(e)}") 
                except Exception as e: # Catch any other unexpected error during plotting.
                    print(f"Warning: An unexpected error occurred during CLUSTERSC plotting - {str(e)}")

        # Return the structured results.
        return final_results
