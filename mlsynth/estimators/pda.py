import numpy as np
import pandas as pd # Added for type hinting
from typing import Dict, Any, List, Union, Optional # Added for type hinting
import warnings
import pydantic # For ValidationError

from ..utils.datautils import balance, dataprep
from ..utils.resultutils import effects, plot_estimates # effects is used internally but not part of final pdaest
from ..utils.estutils import (
    pda as pda_estimator_func, # Renamed to avoid conflict with class name
)
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..config_models import ( # Import the Pydantic model
    PDAConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
    InferenceResults,
    MethodDetailsResults,
)


class PDA:
    """
    Implements the Panel Data Approach (PDA) for treatment effect estimation.

    This class provides a unified interface to several PDA methods, including:
    - **LASSO (L1 Penalty):** Uses L1 regularization for variable selection and
      estimation, as described in Li & Bell (2017).
    - **L2-Relaxation ('l2'):** Employs L2 regularization, potentially with a
      user-specified treatment effect (tau_l2), based on Shi & Wang (2024).
    - **Forward Selection ('fs'):** Uses a forward stepwise procedure for selecting
      control units, as detailed in Shi & Huang (2023).

    The estimator takes panel data and a configuration object specifying the
    chosen PDA method and other common parameters. The `fit` method then
    applies the selected algorithm to estimate the treatment effect.

    Attributes
    ----------
    config : PDAConfig
        The configuration object holding all parameters for the estimator.
    df : pd.DataFrame
        The input DataFrame containing panel data.
        (Inherited from `BaseEstimatorConfig` via `PDAConfig`)
    outcome : str
        Name of the outcome variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `PDAConfig`)
    treat : str
        Name of the treatment indicator column in `df`.
        (Inherited from `BaseEstimatorConfig` via `PDAConfig`)
    unitid : str
        Name of the unit identifier column in `df`.
        (Inherited from `BaseEstimatorConfig` via `PDAConfig`)
    time : str
        Name of the time variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `PDAConfig`)
    display_graphs : bool, default True
        Whether to display graphs of results.
        (Inherited from `BaseEstimatorConfig` via `PDAConfig`)
    save : Union[bool, str, Dict[str, str]], default False
        Configuration for saving plots.
        - If `False` (default), plots are not saved.
        - If `True`, plots are saved with default names in the current directory.
        - If a `str`, it's used as the base filename for saved plots.
        - If a `Dict[str, str]`, it maps specific plot keys (e.g., "estimates_plot")
          to full file paths.
        (Inherited from `BaseEstimatorConfig` via `PDAConfig`)
    counterfactual_color : Union[str, List[str]], default "red"
        Color for the counterfactual line(s) in plots. Can be a single color string
        or a list of color strings if multiple counterfactuals are plotted (though
        PDA typically plots one).
        (Inherited from `BaseEstimatorConfig` via `PDAConfig`)
    treated_color : str, default "black"
        Color for the treated unit line in plots.
        (Inherited from `BaseEstimatorConfig` via `PDAConfig`)
    method : str, default "fs"
        The PDA method to use: 'LASSO', 'l2', or 'fs'.
        (From `PDAConfig`)
    tau : Optional[float], default None
        User-specified treatment effect value, primarily used as `tau_l2`
        for the 'l2' method.
        (From `PDAConfig`)

    References
    ----------
    Shi, Z. & Huang, J. (2023). "Forward-selected panel data approach for program evaluation."
    *Journal of Econometrics*, Volume 234, Issue 2, Pages 512-535.
    DOI: https://doi.org/10.1016/j.jeconom.2021.04.009

    Li, Kathleen T., and David R. Bell. "Estimation of Average Treatment Effects with Panel Data:
    Asymptotic Theory and Implementation." *Journal of Econometrics* 197, no. 1 (March, 2017): 65-75.
    DOI: https://doi.org/10.1016/j.jeconom.2016.01.011

    Shi, Z. & Wang, Y. (2024). "L2-relaxation for Economic Prediction."
    DOI: https://doi.org/10.13140/RG.2.2.11670.97609.
    """

    def __init__(self, config: PDAConfig) -> None: # Changed to PDAConfig
        """
        Initializes the PDA estimator with a configuration object.

        Parameters
        ----------
        config : PDAConfig
            A Pydantic model instance containing all configuration parameters
            for the PDA estimator. This includes:
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
            - counterfactual_color (Union[str, List[str]], optional): Color for counterfactual
              line(s). Defaults to "red".
            - treated_color (str, optional): Color for treated unit line. Defaults to "black".
            - method (str, optional): Type of PDA: 'LASSO', 'l2', or 'fs'. Defaults to "fs".
            - tau (Optional[float], optional): User-specified treatment effect for 'l2' method. Defaults to None.
        """
        if isinstance(config, dict):
            config = PDAConfig(**config)  # convert dict to config object
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
        self.method: str = config.method
        self.tau: Optional[float] = config.tau

        # Pydantic handles type validation for tau and pattern validation for method.
        # The original manual validation can be removed.
        # if self.tau is not None and not isinstance(self.tau, (int, float)):
        #     raise ValueError("tau must be a numeric value.")
        # if self.method.lower() not in ["lasso", "l2", "fs"]: # .lower() was used, Pydantic pattern is case-sensitive
        #     raise ValueError("Method must be 'LASSO', 'l2', or 'fs'.")


    def fit(self) -> BaseEstimatorResults:
        """
        Fits the Panel Data Approach (PDA) model using the specified method.

        The method performs data balancing and preparation, then invokes the
        core PDA estimation function (`pda_estimator_func`) from `estutils`.
        The results from this function are then mapped to the standardized
        `BaseEstimatorResults` Pydantic model. Supported PDA methods include
        'LASSO', 'l2' (L2-Relaxation), and 'fs' (Forward Selection).

        Returns
        -------
        BaseEstimatorResults
            An object containing the standardized estimation results. Key fields include:
            - effects (EffectsResults): Contains treatment effect estimates.
                - att (Optional[float]): Average Treatment Effect on the Treated.
                - att_se (Optional[float]): Standard error for the ATT.
                - att_p_value (Optional[float]): P-value for the ATT.
                - other_effects (Optional[Dict[str, Any]]): Dictionary of other
                  named effects.
            - fit_diagnostics (FitDiagnosticsResults): Contains goodness-of-fit metrics.
                - rmse (Optional[float]): Root Mean Squared Error.
                - r_squared (Optional[float]): R-squared value.
                - mspe (Optional[float]): Mean Squared Prediction Error (overall).
                - pre_mspe (Optional[float]): MSPE for the pre-treatment period.
                - post_mspe (Optional[float]): MSPE for the post-treatment period.
                - other_diagnostics (Optional[Dict[str, Any]]): Dictionary of other
                  diagnostic metrics.
            - time_series (TimeSeriesResults): Contains time-series data.
                - observed_outcome (Optional[List[float]]): Observed outcome for the
                  treated unit.
                - synthetic_outcome (Optional[List[float]]): Estimated counterfactual
                  outcome for the treated unit.
                - treatment_effect_timeseries (Optional[List[float]]): Estimated gap
                  (treatment effect) over time.
                - time_periods (Optional[List[Any]]): Time periods corresponding to
                  the series data.
                - other_series (Optional[Dict[str, List[Any]]]): Dictionary of other
                  relevant time series.
            - weights (WeightsResults): Contains weights assigned by the model.
                - donor_weights (Optional[Dict[str, float]]): Weights assigned to
                  donor (control) units.
                - predictor_weights (Optional[Dict[str, float]]): Weights assigned to
                  predictors (typically None for PDA).
                - other_weights (Optional[Dict[str, Any]]): Dictionary of other
                  types of weights.
            - inference (InferenceResults): Contains statistical inference results.
                - p_value (Optional[float]): General p-value if applicable.
                - confidence_interval_lower (Optional[float]): Lower bound of CI.
                - confidence_interval_upper (Optional[float]): Upper bound of CI.
                - placebo_distribution (Optional[List[float]]): Distribution of
                  placebo effects.
                - other_inference_metrics (Optional[Dict[str, Any]]): Dictionary of
                  other inference-related metrics.
                (Note: PDA's direct output primarily populates ATT p-value and SE in
                `EffectsResults`; other inference fields are typically None unless
                derived from post-estimation procedures not part of this core `fit`.)
            - method_details (MethodDetailsResults): Details about the estimation method.
                - name (Optional[str]): Name of the estimation method used (e.g., "PDA-FS").
                - parameters (Optional[Dict[str, Any]]): Key parameters used in
                  the estimation.
                - other_details (Optional[Dict[str, Any]]): Dictionary of other
                  method-specific details.
            - raw_results (Optional[Dict[str, Any]]): The raw dictionary output from
              the underlying estimation function, for detailed inspection or
              backward compatibility.

        Examples
        --------
        # doctest: +SKIP
        >>> import pandas as pd
        >>> from mlsynth.estimators.pda import PDA
        >>> from mlsynth.config_models import PDAConfig
        >>> # Load or create panel data
        >>> data = pd.DataFrame({
        ...     'country': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        ...     'year': [2000, 2001, 2002, 2000, 2001, 2002, 2000, 2001, 2002],
        ...     'gdp_growth': [2.0, 2.5, 1.0, 3.0, 3.2, 2.8, 1.5, 1.7, 1.2],
        ...     'policy_change': [0, 0, 1, 0, 0, 0, 0, 0, 0] # Country A had policy change in 2002
        ... })
        >>> # Using Forward Selection (fs) method
        >>> pda_fs_config = PDAConfig(
        ...     df=data, outcome="gdp_growth", treat="policy_change",
        ...     unitid="country", time="year", method="fs", display_graphs=False
        ... )
        >>> pda_fs_estimator = PDA(config=pda_fs_config)
        >>> results_fs = pda_fs_estimator.fit()
        >>> print(f"FS ATT: {results_fs.effects.att}")
        >>> # Using LASSO method
        >>> pda_lasso_config = PDAConfig(
        ...     df=data, outcome="gdp_growth", treat="policy_change",
        ...     unitid="country", time="year", method="LASSO", display_graphs=False
        ... )
        >>> pda_lasso_estimator = PDA(config=pda_lasso_config)
        >>> results_lasso = pda_lasso_estimator.fit()
        >>> print(f"LASSO ATT: {results_lasso.effects.att}")
        >>> # Using L2-Relaxation method with a specified tau
        >>> pda_l2_config = PDAConfig(
        ...     df=data, outcome="gdp_growth", treat="policy_change",
        ...     unitid="country", time="year", method="l2", tau=-0.5, display_graphs=False
        ... )
        >>> pda_l2_estimator = PDA(config=pda_l2_config)
        >>> results_l2 = pda_l2_estimator.fit()
        >>> print(f"L2 ATT (tau=-0.5): {results_l2.effects.att}")
        """
        try:
            # Step 1: Balance the panel data to ensure consistent time periods across units.
            balance(self.df, self.unitid, self.time)

            # Step 2: Prepare data into matrices required by the PDA estimation function.
            # `dataprep` returns a dictionary containing 'y' (treated outcomes), 'donor_matrix' (control outcomes),
            # 'pre_periods', 'post_periods', 'donor_names', 'treated_unit_name', 'time_index', etc.
            prepared_data: Dict[str, Any] = dataprep(
                self.df, self.unitid, self.time, self.outcome, self.treat
            )

            if np.isnan(prepared_data["donor_matrix"]).any():
                raise MlsynthEstimationError("Donor matrix contains NaN values after preprocessing.")
            if np.isnan(prepared_data["y"]).any():
                raise MlsynthEstimationError("Treated outcome vector contains NaN values after preprocessing.")

            # Step 3: Call the core PDA estimation function from `estutils`.
            # This function handles the logic for the chosen PDA method ('LASSO', 'l2', or 'fs').
            raw_pda_estimation_output: Dict[str, Any] = pda_estimator_func(
                prepared_data, # The dictionary of prepared data matrices and info
                len(prepared_data["donor_names"]), # Number of donor units
                pda_method_type=self.method, # Specified PDA method ('LASSO', 'l2', 'fs')
                l2_regularization_param_override=self.tau # Tau parameter, primarily for 'l2' method
            )

            # Step 4: Map the raw dictionary output from `pda_estimator_func` to standardized Pydantic models.
            
            # --- Effects ---
            effects_data = raw_pda_estimation_output.get("Effects", {})
            effects_results = EffectsResults(
                att=effects_data.get("Average Treatment Effect"),
                att_se=effects_data.get("Standard Error"),
                att_p_value=effects_data.get("P-value"),
                # Store any other effect-related metrics
                other_effects={k: v for k, v in effects_data.items() if k not in ["Average Treatment Effect", "Standard Error", "P-value"]}
            )

            # --- Fit Diagnostics ---
            fit_data = raw_pda_estimation_output.get("Fit", {})
            fit_diagnostics_results = FitDiagnosticsResults(
                rmse=fit_data.get("RMSE"),
                r_squared=fit_data.get("R-squared"), # Key might be 'R-squared' or 'R2'
                mspe=fit_data.get("MSPE"), # Overall MSPE if available
                pre_mspe=fit_data.get("Pre-MSPE"), # Pre-treatment MSPE
                post_mspe=fit_data.get("Post-MSPE"), # Post-treatment MSPE
                # Store any other diagnostic metrics
                other_diagnostics={k: v for k, v in fit_data.items() if k not in ["RMSE", "R-squared", "MSPE", "Pre-MSPE", "Post-MSPE"]}
            )
            
            # --- Time Series Data ---
            vectors_data = raw_pda_estimation_output.get("Vectors", {})
            
            obs_unit_data = vectors_data.get("Observed Unit") # Treated unit's observed outcome series
            cf_data = vectors_data.get("Counterfactual")      # Estimated counterfactual series
            gap_data = vectors_data.get("Gap")                # Estimated treatment effect (gap) series
            time_idx_data = prepared_data.get("time_index")   # Time periods from dataprep

            time_series_results = TimeSeriesResults(
                # Squeeze NumPy arrays to ensure they are 1D if they come out as 2D with one column
                observed_outcome=obs_unit_data.squeeze() if isinstance(obs_unit_data, np.ndarray) else obs_unit_data,
                synthetic_outcome=cf_data.squeeze() if isinstance(cf_data, np.ndarray) else cf_data,
                treatment_effect_timeseries=(
                    # Handle cases where gap_data might be 2D (e.g., from some internal calculations)
                    gap_data[:, 0] if isinstance(gap_data, np.ndarray) and gap_data.ndim == 2 else
                    gap_data if isinstance(gap_data, np.ndarray) else 
                    None 
                ),
                # Ensure time_periods is a NumPy array
                time_periods=time_idx_data if isinstance(time_idx_data, np.ndarray) else np.array(time_idx_data) if time_idx_data is not None else None,
                # Store any other relevant time series from vectors_data
                other_series={
                    k: (v.squeeze() if isinstance(v, np.ndarray) and v.ndim > 1 else v if isinstance(v, np.ndarray) else np.array(v) if v is not None else None) 
                    for k,v in vectors_data.items() 
                    if k not in ["Observed Unit", "Counterfactual", "Gap"]
                }
            )

            # --- Weights ---
            weights_val = raw_pda_estimation_output.get("Weights")
            donor_weights_dict: Optional[Dict[str, float]] = None
            if isinstance(weights_val, np.ndarray): # If weights are an array, map to donor names
                donor_names = prepared_data.get("donor_names", [])
                donor_weights_dict = {name: weights_val[i] for i, name in enumerate(donor_names) if i < len(weights_val)}
            elif isinstance(weights_val, dict): # If weights are already a dict
                donor_weights_dict = weights_val
                
            weights_results = WeightsResults(
                donor_weights=donor_weights_dict
                # predictor_weights are typically not a direct output of PDA methods focused on unit weights.
            )

            # --- Method Details ---
            method_details_results = MethodDetailsResults(
                name=raw_pda_estimation_output.get("method", self.method.upper()), # Use method from raw output or config
                parameters={"method_choice": self.method, "tau_l2_if_applicable": self.tau} # Store key config parameters
            )
            
            # --- Inference ---
            # PDA's core `pda_estimator_func` might provide SE/p-value for ATT in 'Effects'.
            # Other inference fields (CI, placebo) are typically None unless post-estimation steps are added.
            inference_results = InferenceResults() # Defaults to None for most fields

            # Assemble the final standardized results object
            final_results = BaseEstimatorResults(
                effects=effects_results,
                fit_diagnostics=fit_diagnostics_results,
                time_series=time_series_results,
                weights=weights_results,
                inference=inference_results,
                method_details=method_details_results,
                raw_results=raw_pda_estimation_output # Include the raw output for detailed inspection
            )

        except (MlsynthDataError, MlsynthConfigError, MlsynthEstimationError):
            # Re-raise specific custom errors from utilities or config validation.
            raise # Re-raise specific custom errors
        except pydantic.ValidationError as ve:
            # Catch Pydantic validation errors during results model creation.
            raise MlsynthEstimationError(f"Error creating results model for PDA: {ve}") from ve
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            # Catch common Python errors that might occur during data manipulation or dictionary access.
            raise MlsynthEstimationError(f"PDA estimation failed due to an unexpected error: {e}") from e
        except Exception as e:
            # Catch-all for any other unexpected errors.
            raise MlsynthEstimationError(f"An unexpected error occurred during PDA fitting: {e}") from e

        # Step 5: Optionally display or save plots of the results.
        if self.display_graphs:
            try:
                # Determine the method name for the plot title/legend.
                plot_estimation_method_name: str = final_results.method_details.name if final_results.method_details and final_results.method_details.name else self.method.upper()
                # Get the counterfactual series for plotting.
                plot_counterfactual_series: Optional[List[float]] = final_results.time_series.synthetic_outcome if final_results.time_series else None
                
                if plot_counterfactual_series is not None:
                    # Construct a name for the counterfactual series in the plot legend.
                    counterfactual_plot_name: str = f'{plot_estimation_method_name} {prepared_data["treated_unit_name"]}' # type: ignore
                    plot_estimates( 
                        processed_data_dict=prepared_data, 
                        time_axis_label=self.time,
                        unit_identifier_column_name=self.unitid,
                        outcome_variable_label=self.outcome,
                        treatment_name_label=self.treat,
                        treated_unit_name=prepared_data["treated_unit_name"],
                        observed_outcome_series=prepared_data["y"], # Observed outcome vector.
                        counterfactual_series_list=[cf_data.flatten()], # List of counterfactual vectors.
                        estimation_method_name="PDA",
                        counterfactual_names=[plot_estimation_method_name], # Names for legend.
                        treated_series_color=self.treated_color,
                        counterfactual_series_colors=self.counterfactual_color,
                        save_plot_config=self.save, 
                    )
                else:
                    # Warn if the counterfactual series is not available for plotting.
                    warnings.warn("PDA: Counterfactual series not available for plotting.", UserWarning)
            except (MlsynthPlottingError, MlsynthDataError) as plot_err:
                # Warn if plotting fails due to known plotting or data issues.
                warnings.warn(f"PDA: Plotting failed with error: {plot_err}", UserWarning)
            except Exception as plot_err: 
                # Warn for any other unexpected plotting errors.
                warnings.warn(f"PDA: An unexpected error occurred during plotting: {plot_err}", UserWarning)

        # Step 6: Return the final standardized results object.
        return final_results
