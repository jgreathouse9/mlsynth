import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib import rc_context
import matplotlib.colors as mcolors
import random
from typing import Optional, Dict, List, Tuple, Any, Union
from mlsynth.exceptions import MlsynthDataError, MlsynthPlottingError # MlsynthConfigError, MlsynthEstimationError might be needed later

from ..config_models import ( # Import the Pydantic models
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
    InferenceResults,
    MethodDetailsResults,
)


def plot_estimates(
    processed_data_dict: Dict[str, Any],
    time_axis_label: str,
    unit_identifier_column_name: str, # Not directly used in plot, but context
    outcome_variable_label: str,
    treatment_name_label: str,
    treated_unit_name: str,
    observed_outcome_series: np.ndarray,
    counterfactual_series_list: List[np.ndarray],
    estimation_method_name: str,
    treated_series_color: str,
    counterfactual_series_colors: List[str],
    counterfactual_names: Optional[List[str]] = None,
    save_plot_config: Union[bool, Dict[str, str]] = False,
    uncertainty_intervals_array: Optional[np.ndarray] = None,
) -> None:
    """Plot observed outcomes against one or more counterfactual estimates.

    This function generates a time-series plot showing the observed trajectory
    for a treated unit alongside one or more estimated counterfactual
    trajectories. It includes options for customizing appearance, saving the
    plot, and displaying uncertainty intervals.

    Parameters
    ----------
    processed_data_dict : Dict[str, Any]
        A dictionary containing processed data, typically the output of a
        data preparation step (e.g., `mlsynth.utils.datautils.dataprep`).
        It should contain the following keys:

        - **"Ywide"** (:py:class:`pandas.DataFrame`):
            A DataFrame where the index represents time periods (e.g.,
            `DatetimeIndex`) used for x-axis labels.
        - **"pre_periods"** (:py:class:`int`):
            The number of pre-treatment periods. This is used to draw a
            vertical line indicating the start of the treatment.

    time_axis_label : str
        Label for the x-axis (e.g., "Year", "Time Period").
    unit_identifier_column_name : str
        Identifier for the unit (e.g., "CountryID"). Not directly used in the
        plot itself but often part of the context when calling this function.
    outcome_variable_label : str
        Label for the y-axis, representing the outcome variable being plotted
        (e.g., "GDP per Capita").
    treatment_name_label : str
        Name of the treatment or intervention (e.g., "Policy Change"). This is
        used in the label for the vertical line indicating treatment start.
    treated_unit_name : str
        Name or identifier of the treated unit (e.g., "California"). Used in
        the legend for the observed outcome line.
    observed_outcome_series : np.ndarray
        Observed outcome vector for the treated unit. Shape (T,), where T is
        the total number of time periods (pre and post).
    counterfactual_series_list : List[np.ndarray]
        A list of counterfactual prediction vectors. Each NumPy array in the
        list should have shape (T,) and represent an estimated counterfactual
        series for the treated unit.
    estimation_method_name : str
        Name of the estimation method (e.g., "FSCM", "TSSC"). Used for
        constructing default plot filenames if `save_plot_config` is True.
    treated_series_color : str
        Color for the line representing the observed (treated unit) outcomes.
        Can be a named color (e.g., "black", "red") or a hex string.
    counterfactual_series_colors : List[str]
        A list of colors for each counterfactual series in `counterfactual_series_list`.
        The length of this list should ideally match `len(counterfactual_series_list)`.
        If `counterfactual_series_list` is longer, additional random colors will be generated.
    counterfactual_names : Optional[List[str]], default None
        A list of custom names for each counterfactual series, used in the
        legend. If None, generic names like "Artificial 1", "Artificial 2"
        are used. Length should match `len(counterfactual_series_list)`.
    save_plot_config : Union[bool, Dict[str, str]], default False
        Controls saving the plot:

        - If ``False`` (default): The plot is displayed using `plt.show()`
          but not saved.
        - If ``True``: The plot is saved with a default filename (e.g.,
          `{estimation_method_name}_{treated_unit_name}.png`) and PNG format
          in the current working directory.
        - If a ``dict``: Allows specifying 'filename', 'extension', and
          'directory'. Example:
          `{'filename': 'my_plot', 'extension': 'pdf', 'directory': 'output_plots/'}`.
          If 'display' is also in the dict and set to ``False``, `plt.show()`
          is skipped.
    uncertainty_intervals_array : Optional[np.ndarray], default None
        An optional 2D NumPy array of shape (T, 2) representing uncertainty
        intervals (e.g., prediction intervals from conformal inference).
        The first column (`uncertainty_intervals_array[:, 0]`) is the lower bound, and the
        second column (`uncertainty_intervals_array[:, 1]`) is the upper bound. If provided,
        these are plotted as a shaded region, typically around the first
        counterfactual in `counterfactual_series_list`.

    Returns
    -------

    None
        This function displays and/or saves a matplotlib plot and does not
        return any value.

    Examples
    --------

    >>> import pandas as pd
    >>> from unittest.mock import patch
    >>> # Sample data for plotting
    >>> time_index_ex = pd.to_datetime(['2000-01-01', '2001-01-01', '2002-01-01', '2003-01-01'])
    >>> processed_data_dict_ex = {
    ...     "Ywide": pd.DataFrame(index=time_index_ex),
    ...     "pre_periods": 2  # Treatment starts after 2001
    ... }
    >>> observed_outcome_series_ex = np.array([10, 11, 12, 9])
    >>> cf1_ex = np.array([10, 11, 11.5, 11])
    >>> cf2_ex = np.array([10, 11, 11.8, 11.2])
    >>> uncertainty_intervals_array_ex = np.array([[np.nan, np.nan], [np.nan, np.nan],
    ...                                           [11.0, 12.0], [10.5, 11.5]]) # For post-periods
    >>>
    >>> # Mock plt.show to run non-interactively
    >>> with patch("matplotlib.pyplot.show") as mock_show: # doctest: +SKIP
    ...     plot_estimates( # doctest: +SKIP
    ...         processed_data_dict=processed_data_dict_ex, # doctest: +SKIP
    ...         time_axis_label="Year",
    ...         unit_identifier_column_name="UnitX",
    ...         outcome_variable_label="Sales",
    ...         treatment_name_label="Ad Campaign",
    ...         treated_unit_name="Store A",
    ...         observed_outcome_series=observed_outcome_series_ex,
    ...         counterfactual_series_list=[cf1_ex, cf2_ex],
    ...         estimation_method_name="SCM_v1",
    ...         treated_series_color="black",
    ...         counterfactual_series_colors=["blue", "green"],
    ...         counterfactual_names=["SCM Estimate", "Robust SCM"], # doctest: +SKIP
    ...         save_plot_config=False, # To prevent file creation during test # doctest: +SKIP
    ...         uncertainty_intervals_array=uncertainty_intervals_array_ex # doctest: +SKIP
    ...     ) # doctest: +SKIP
    ...     mock_show.assert_called_once() # Verifies plt.show() was called # doctest: +SKIP
    """
    # --- Input Validation and Data Extraction ---
    try:
        # Extract time index and number of pre-treatment periods from the processed data dictionary.
        # 'Ywide' is expected to be a DataFrame with a time-based index.
        time_index = processed_data_dict["Ywide"].index
        pre_periods = processed_data_dict["pre_periods"]
    except KeyError as e:
        # Raise an error if essential keys are missing from processed_data_dict.
        raise MlsynthDataError(
            f"processed_data_dict is missing required keys ('Ywide' or 'pre_periods'). Original error: {e}"
        ) from e

    # Validate the types and dimensions of input series.
    if not isinstance(observed_outcome_series, np.ndarray) or observed_outcome_series.ndim != 1:
        raise MlsynthDataError("observed_outcome_series must be a 1D NumPy array.")
    if not isinstance(counterfactual_series_list, list) or not all(isinstance(cf, np.ndarray) and cf.ndim == 1 for cf in counterfactual_series_list):
        raise MlsynthDataError("counterfactual_series_list must be a list of 1D NumPy arrays.")
    
    # Validate uncertainty intervals array if provided.
    if uncertainty_intervals_array is not None:
        if not isinstance(uncertainty_intervals_array, np.ndarray) or uncertainty_intervals_array.ndim != 2 or uncertainty_intervals_array.shape[1] != 2:
            raise MlsynthDataError("uncertainty_intervals_array must be a 2D NumPy array with 2 columns (lower, upper).")
        if uncertainty_intervals_array.shape[0] != len(observed_outcome_series):
            # Ensure uncertainty intervals align with the observed series length.
            raise MlsynthDataError("uncertainty_intervals_array must have the same number of rows as the observed_outcome_series.")


    # --- Plotting Setup ---
    # Define a consistent theme for the plot aesthetics.
    plot_theme_settings = {
        "figure.facecolor": "white",
        "figure.figsize": (11, 5),
        "figure.dpi": 100,
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "lines.linewidth": 1.2,
        "patch.facecolor": "#0072B2",  # Blue shade for patches
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.size": 14,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.grid": True,
        "axes.facecolor": "white",
        "axes.linewidth": 0.1,
        "axes.titlesize": "large",
        "axes.titleweight": "bold",
        "axes.labelsize": "medium",
        "axes.labelweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.titlepad": 25,
        "axes.labelpad": 20,
        "grid.alpha": 0.1,
        "grid.linewidth": 0.5,
        "grid.color": "#000000",
        "legend.framealpha": 0.5,
        "legend.fancybox": True,
        "legend.borderpad": 0.5,
        "legend.loc": "best",
        "legend.fontsize": "small",
    }

    # Apply the defined theme using a context manager.
    with rc_context(rc=plot_theme_settings):

        # Create a simple numeric time axis (0, 1, 2, ...) for plotting.
        # The actual time labels (e.g., dates) will be set using time_index.
        time_points_numeric = np.arange(len(observed_outcome_series))
        
        # Validate pre_periods against the length of the time_index.
        if pre_periods >= len(time_index):
            raise MlsynthDataError(
                f"pre_periods ({pre_periods}) is out of bounds for the time_index "
                f"(length {len(time_index)})."
            )
        # Get the actual date/time label for the treatment start.
        treatment_start_date_formatted = time_index[pre_periods]


        # --- Plotting Core Elements ---
        # Add a vertical dashed line to indicate the start of the treatment period.
        plt.axvline(
            x=pre_periods, # Position on the numeric time axis.
            color="grey",
            linestyle="--",
            linewidth=1.8,
            label=f"{treatment_name_label}, {treatment_start_date_formatted}", # Legend entry for the line.
        )

        # Ensure there are enough colors for all counterfactual series.
        # If not enough colors are provided, generate random distinct colors.
        current_counterfactual_series_colors = list(counterfactual_series_colors) # Make a mutable copy.
        unique_colors_needed = len(counterfactual_series_list) - len(current_counterfactual_series_colors)

        if unique_colors_needed > 0:
            all_named_colors = list(mcolors.CSS4_COLORS.keys()) # Get all available named CSS4 colors.
            # Create a set of colors already in use to avoid duplicates.
            forbidden_colors = set(current_counterfactual_series_colors + [treated_series_color])
            # Filter out forbidden colors.
            candidate_colors = [
                c for c in all_named_colors if c not in forbidden_colors
            ]
            random.shuffle(candidate_colors) # Shuffle to get random choices.
            # Append the needed number of unique colors.
            current_counterfactual_series_colors += candidate_colors[:unique_colors_needed]

        # Plot each counterfactual series.
        for counterfactual_index, current_counterfactual_series in enumerate(counterfactual_series_list):
            # Determine the label for the legend.
            label = (
                counterfactual_names[counterfactual_index]
                if counterfactual_names and counterfactual_index < len(counterfactual_names) # Check bounds for names
                else f"Artificial {counterfactual_index + 1}" # Default name if not provided.
            )
            # Cycle through provided colors if there are more series than colors.
            color = current_counterfactual_series_colors[counterfactual_index % len(current_counterfactual_series_colors)]
            plt.plot(time_points_numeric, current_counterfactual_series, label=label, linestyle="-", color=color, linewidth=1.5)

        # Plot the observed outcome series for the treated unit.
        plt.plot(time_points_numeric, observed_outcome_series, label=f"{treated_unit_name}", color=treated_series_color, linewidth=1.5)

        # Plot uncertainty intervals (e.g., prediction intervals) if provided.
        if uncertainty_intervals_array is not None:
            lower_uncertainty_bound = uncertainty_intervals_array[:, 0]
            upper_uncertainty_bound = uncertainty_intervals_array[:, 1]
            # Fill the area between lower and upper bounds.
            plt.fill_between(
                time_points_numeric, lower_uncertainty_bound, upper_uncertainty_bound, color="grey", alpha=0.4, label="Prediction Interval"
            )

        # --- Plot Finalization ---
        # Set axis labels.
        plt.xlabel(time_axis_label)
        # Note: y-axis label is set by outcome_variable_label in title construction.

        # Extract min and max time index values for the title.
        min_time_index = time_index.min()
        max_time_index = time_index.max()

        # Format time index values as strings, handling datetime objects appropriately.
        min_time_index_str = (
            min_time_index.strftime("%Y-%m-%d")
            if hasattr(min_time_index, "strftime") # Check if it's a datetime-like object.
            else str(min_time_index)
        )
        max_time_index_str = (
            max_time_index.strftime("%Y-%m-%d")
            if hasattr(max_time_index, "strftime")
            else str(max_time_index)
        )

        # Set the plot title, incorporating outcome variable and time range.
        plt.title(
            f"Causal Impact on {outcome_variable_label}, {min_time_index_str} to {max_time_index_str}", loc="left"
        )

        # Display legend and grid.
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5) # Light grid for readability.

        # --- Save or Display Plot ---
        if save_plot_config:
            # Determine filename, extension, and directory based on save_plot_config.
            if isinstance(save_plot_config, dict):
                filename = save_plot_config.get("filename", f"{estimation_method_name}_{treated_unit_name}")
                extension = save_plot_config.get("extension", "png")
                directory = save_plot_config.get("directory", os.getcwd()) # Default to current dir.
            else: # save_plot_config is True (use defaults)
                filename = f"{estimation_method_name}_{treated_unit_name}"
                extension = "png"
                directory = os.getcwd()

            os.makedirs(directory, exist_ok=True) # Ensure the save directory exists.
            filepath = os.path.join(directory, f"{filename}.{extension}")
            
            try:
                plt.savefig(filepath)
                print(f"Plot saved to: {filepath}")
            except OSError as e: # Catch potential OS errors during save.
                raise MlsynthPlottingError(f"Failed to save plot to {filepath}. Original error: {e}") from e

        # Display the plot if not saving or if display is explicitly enabled in config.
        if not save_plot_config or (isinstance(save_plot_config, dict) and save_plot_config.get("display", True)):
            plt.show()
        
        plt.close() # Close the plot figure to free up memory, especially in loops or batch processing.


class effects:
    @staticmethod
    def calculate(
        observed_outcome_series: np.ndarray,
        counterfactual_outcome_series: np.ndarray,
        num_pre_treatment_periods: int,
        num_actual_post_periods: int,
        significance_level: float = 0.1
    ) -> tuple: # Simplified for now
        """Minimal docstring to test parsing.

        This is a short description.

        Parameters
        ----------
        observed_outcome_series : np.ndarray
            Placeholder description.
        counterfactual_outcome_series : np.ndarray
            Placeholder description.
        num_pre_treatment_periods : int
            Placeholder description.
        num_actual_post_periods : int
            Placeholder description.
        significance_level : float, optional
            Placeholder description.

        Returns
        -------
        tuple
            Placeholder description.

        Examples
        --------
        >>> # Placeholder example
        >>> pass
        """
        # --- Pre-treatment Fit Statistics ---
        # Calculate residuals in the pre-treatment period.
        pre_treatment_residuals = observed_outcome_series[:num_pre_treatment_periods] - counterfactual_outcome_series[:num_pre_treatment_periods]
        
        # Calculate the denominator for R-squared: variance of observed outcomes in pre-treatment.
        mean_sq_error_pre_treatment_denom = np.mean((observed_outcome_series[:num_pre_treatment_periods] - np.mean(observed_outcome_series[:num_pre_treatment_periods]))**2)
        
        # Calculate R-squared for the pre-treatment period.
        # R-squared = 1 - (MSE_residuals / Var_observed_outcomes)
        # Handle cases with no pre-periods or zero variance in observed outcomes.
        r_squared_pre_treatment = 1 - (np.mean(pre_treatment_residuals**2)) / mean_sq_error_pre_treatment_denom if num_pre_treatment_periods > 0 and mean_sq_error_pre_treatment_denom != 0 else np.nan

        # --- Post-treatment Effect Calculations ---
        if num_actual_post_periods > 0:
            # Define the slice for the post-treatment period.
            post_period_slice = slice(num_pre_treatment_periods, num_pre_treatment_periods + num_actual_post_periods)
            
            # Average Treatment Effect on the Treated (ATT)
            average_treatment_effect_treated = np.mean(observed_outcome_series[post_period_slice] - counterfactual_outcome_series[post_period_slice])
            
            # Mean of the counterfactual in the post-treatment period (for Percent ATT).
            mean_counterfactual_post_treatment = np.mean(counterfactual_outcome_series[post_period_slice])
            # Percent ATT: ATT as a percentage of the mean counterfactual.
            average_treatment_effect_treated_percent = (100 * average_treatment_effect_treated / mean_counterfactual_post_treatment) if mean_counterfactual_post_treatment != 0 else np.nan
            
            # Components for Standardized ATT (SATT)
            # Scaled mean squared pre-treatment residuals.
            scaled_mean_sq_pre_treatment_residuals = (num_actual_post_periods / num_pre_treatment_periods) * np.mean(pre_treatment_residuals**2) if num_pre_treatment_periods > 0 else np.nan
            mean_sq_pre_treatment_residuals = np.mean(pre_treatment_residuals**2) if num_pre_treatment_periods > 0 else np.nan
            # Denominator for SATT, based on pre-treatment residual variance.
            standard_error_denominator_for_satt = np.sqrt(scaled_mean_sq_pre_treatment_residuals + mean_sq_pre_treatment_residuals) if not (np.isnan(scaled_mean_sq_pre_treatment_residuals) or np.isnan(mean_sq_pre_treatment_residuals)) else np.nan
            
            # Standardized ATT.
            standardized_average_treatment_effect = (np.sqrt(num_actual_post_periods) * average_treatment_effect_treated / standard_error_denominator_for_satt) if standard_error_denominator_for_satt != 0 and not np.isnan(standard_error_denominator_for_satt) else np.nan
            
            # Total Treatment Effect (TTE): sum of differences in the post-period.
            total_treatment_effect = np.sum(observed_outcome_series[post_period_slice] - counterfactual_outcome_series[post_period_slice])
            
            # Time series of ATT for each post-treatment period.
            att_time_series = observed_outcome_series[post_period_slice] - counterfactual_outcome_series[post_period_slice]
            # Time series of Percent ATT.
            percent_att_time_series = np.full_like(att_time_series, np.nan) # Initialize with NaNs.
            non_zero_cf_post_mask = counterfactual_outcome_series[post_period_slice] != 0 # Avoid division by zero.
            percent_att_time_series[non_zero_cf_post_mask] = 100 *\
                att_time_series[non_zero_cf_post_mask] / counterfactual_outcome_series[post_period_slice][non_zero_cf_post_mask]
            # Time series of SATT (currently a placeholder, as per-period SATT might need different scaling).
            satt_time_series = np.full_like(att_time_series, np.nan) 
            
            # RMSE of the gap in the post-treatment period (T1 RMSE).
            post_treatment_rmse = round(np.std(observed_outcome_series[post_period_slice] - counterfactual_outcome_series[post_period_slice]), 3)

        else: # Handle case with no post-treatment periods.
            average_treatment_effect_treated = np.nan
            average_treatment_effect_treated_percent = np.nan
            standardized_average_treatment_effect = np.nan
            total_treatment_effect = np.nan
            att_time_series = np.array([])
            percent_att_time_series = np.array([])
            satt_time_series = np.array([])
            post_treatment_rmse = np.nan

        # --- Compile Results into Dictionaries ---
        # Dictionary for goodness-of-fit statistics.
        fit_statistics_dict = {
            "T0 RMSE": round(np.sqrt(np.mean(pre_treatment_residuals**2)), 3) if num_pre_treatment_periods > 0 else np.nan, # Pre-treatment RMSE
            "T1 RMSE": post_treatment_rmse, # Post-treatment RMSE (std dev of post-treatment gap)
            "R-Squared": round(r_squared_pre_treatment, 3), # Pre-treatment R-squared
            "Pre-Periods": num_pre_treatment_periods,
            "Post-Periods": num_actual_post_periods,
        }

        # Dictionary for treatment effect metrics.
        treatment_effects_dict = {
            "ATT": round(average_treatment_effect_treated, 3) if not np.isnan(average_treatment_effect_treated) else np.nan,
            "Percent ATT": round(average_treatment_effect_treated_percent, 3) if not np.isnan(average_treatment_effect_treated_percent) else np.nan,
            "SATT": round(standardized_average_treatment_effect, 3) if not np.isnan(standardized_average_treatment_effect) else np.nan,
            "TTE": round(total_treatment_effect, 3) if not np.isnan(total_treatment_effect) else np.nan,
            "ATT_Time": np.round(att_time_series, 3),
            "PercentATT_Time": np.round(percent_att_time_series, 3),
            "SATT_Time": np.round(satt_time_series, 3), # Placeholder SATT time series
        }

        # Calculate the gap series (observed - counterfactual) for all periods.
        gap_series = observed_outcome_series - counterfactual_outcome_series

        # Create a relative time column: 0 for treatment start, negative for pre, positive for post.
        relative_time_column_for_gap = np.arange(gap_series.shape[0]) - num_pre_treatment_periods + 1
        # Combine gap series and relative time into a 2-column matrix.
        gap_matrix = np.column_stack((gap_series, relative_time_column_for_gap))

        # Dictionary for key time series vectors.
        cf_series_for_dict = np.full_like(observed_outcome_series, np.nan, dtype=float) # Default to NaNs, ensure float
        if counterfactual_outcome_series is not None and\
           isinstance(counterfactual_outcome_series, np.ndarray) and\
           counterfactual_outcome_series.size == observed_outcome_series.size:
            try:
                # Ensure counterfactual_outcome_series is 1D before reshape
                reshaped_cf = counterfactual_outcome_series.flatten().reshape(-1, 1)
                cf_series_for_dict = np.round(reshaped_cf, 3)
            except Exception: # Broad catch if reshape/round fails
                # If error, cf_series_for_dict remains as NaNs
                pass
        
        time_series_vectors_dict = {
            "Observed Unit": np.round(observed_outcome_series.reshape(-1, 1), 3), # Reshape to column vector
            "Counterfactual": cf_series_for_dict, # Use the robustly prepared series
            "Gap": np.round(gap_matrix, 3), # Gap series with relative time
        }

        return treatment_effects_dict, fit_statistics_dict, time_series_vectors_dict


def SDID_plot(
    sdid_results_dict: Dict[str, Any],
    title: str = "Event Study: Synthetic Difference-in-Differences",
    y_axis_label: str = "Treatment Effect",
    x_axis_label: str = "Event Time (Relative to Treatment)",
) -> None:
    """Plot event study estimates from Synthetic Difference-in-Differences (SDID).

    This function visualizes treatment effects over event time, typically
    showing point estimates and confidence intervals derived from an SDID
    estimation procedure.

    Parameters
    ----------
    sdid_results_dict : Dict[str, Any]
        The output dictionary from an SDID estimator (or a similarly structured
        dictionary). It is expected to contain a key "pooled_estimates".
        `sdid_results_dict["pooled_estimates"]` should be a dictionary where:

        - Keys are event times (e.g., -2, -1, 0, 1, 2), convertible to integers.
          Event time 0 usually represents the treatment period.
        - Values are dictionaries, each containing at least:
            - "tau" (float): The point estimate of the treatment effect for that event time.
            - "ci" (Tuple[float, float] or List[float]): A tuple or list of two
              floats representing the lower and upper bounds of the confidence
              interval for the treatment effect.

    title : str, optional
        Title for the plot.
        Default is "Event Study: Synthetic Difference-in-Differences".
    y_axis_label : str, optional
        Label for the y-axis. Default is "Treatment Effect".
    x_axis_label : str, optional
        Label for the x-axis. Default is "Event Time (Relative to Treatment)".

    Returns
    -------
    None
        This function displays a matplotlib plot and does not return any value.

    Examples
    --------
    >>> from unittest.mock import patch
    >>> # Sample SDID results structure
    >>> sdid_results_ex = {
    ...     "pooled_estimates": {
    ...         "-2": {"tau": 0.1, "ci": [-0.5, 0.7]},
    ...         "-1": {"tau": -0.05, "ci": [-0.6, 0.5]},
    ...         "0": {"tau": 2.5, "ci": [1.8, 3.2]}, # Treatment period
    ...         "1": {"tau": 2.8, "ci": [2.0, 3.6]},
    ...         "2": {"tau": 2.6, "ci": [1.9, 3.3]}
    ...     }
    ... }
    >>> # Mock plt.show to run non-interactively
    >>> with patch("matplotlib.pyplot.show") as mock_show: # doctest: +SKIP
    ...     SDID_plot(sdid_results_ex, title="My SDID Event Study") # doctest: +SKIP
    ...     mock_show.assert_called_once() # Verifies plt.show() was called # doctest: +SKIP
    """
    # --- Input Validation and Data Extraction ---
    try:
        # Attempt to access the 'pooled_estimates' key from the input dictionary.
        pooled_estimates = sdid_results_dict["pooled_estimates"]
    except KeyError as e:
        # If 'pooled_estimates' is missing, raise a data error.
        raise MlsynthDataError("sdid_results_dict is missing the required 'pooled_estimates' key.") from e
    except TypeError as e: 
        # If sdid_results_dict is not a dictionary (e.g., None or other type), raise a data error.
        raise MlsynthDataError("sdid_results_dict must be a dictionary.") from e


    # --- Data Preparation for Plotting ---
    # Initialize lists to store extracted data points for the plot.
    event_times_list, point_estimates_effects, confidence_interval_lower_bounds, confidence_interval_upper_bounds = [], [], [], []

    if not isinstance(pooled_estimates, dict):
        # Further validation that 'pooled_estimates' itself is a dictionary.
        raise MlsynthDataError("'pooled_estimates' must be a dictionary.")

    # Iterate through each event time and its corresponding estimate data.
    for event_time_key, current_estimate_data in pooled_estimates.items():
        try:
            # Convert event time key (e.g., "-2", "0", "1") to an integer.
            event_times_list.append(int(event_time_key))
        except ValueError as e:
            raise MlsynthDataError(f"Event time key '{event_time_key}' cannot be converted to an integer.") from e
        
        # Validate that the data for the current event time is a dictionary.
        if not isinstance(current_estimate_data, dict):
            raise MlsynthDataError(f"Value for event time key '{event_time_key}' must be a dictionary.")

        try:
            # Extract the point estimate ('tau') and confidence interval ('ci').
            point_estimates_effects.append(current_estimate_data["tau"])
            ci = current_estimate_data["ci"]
            # Validate the format of the confidence interval.
            if not (isinstance(ci, (list, tuple)) and len(ci) == 2):
                raise MlsynthDataError(f"Confidence interval 'ci' for event time '{event_time_key}' must be a list or tuple of two numbers.")
            confidence_interval_lower_bounds.append(ci[0]) # Lower bound
            confidence_interval_upper_bounds.append(ci[1]) # Upper bound
        except KeyError as e:
            # Handle missing 'tau' or 'ci' keys.
            raise MlsynthDataError(f"Estimate data for event time '{event_time_key}' is missing required keys ('tau' or 'ci').") from e
        except (TypeError, IndexError) as e: 
            # Handle malformed 'ci' (e.g., not a list/tuple, wrong length).
             raise MlsynthDataError(f"Confidence interval 'ci' for event time '{event_time_key}' is malformed. Expected list/tuple of two numbers. Error: {e}") from e

    # Handle the case where no estimates were extracted (e.g., empty pooled_estimates).
    if not event_times_list:
        # Currently, this will proceed and might result in an empty plot or matplotlib errors.
        # Consider adding a warning or raising an error for empty plots if desired.
        pass


    # --- Sorting Data by Event Time ---
    # Sort all extracted lists based on event_times_list to ensure correct plotting order.
    sorted_indices = np.argsort(event_times_list)
    event_times_sorted = [event_times_list[i] for i in sorted_indices]
    point_estimates_effects_sorted = [point_estimates_effects[i] for i in sorted_indices]
    lower_ci_sorted = [confidence_interval_lower_bounds[i] for i in sorted_indices]
    upper_ci_sorted = [confidence_interval_upper_bounds[i] for i in sorted_indices]

    # --- Plotting ---
    plt.figure(figsize=(10, 6)) # Set figure size.
    
    # Plot the point estimates of treatment effects.
    plt.plot(event_times_sorted, point_estimates_effects_sorted, "o-", label="Estimated Effect", color="blue") # 'o-' for markers and line.
    
    # Fill the area between lower and upper confidence interval bounds.
    plt.fill_between(
        event_times_sorted, lower_ci_sorted, upper_ci_sorted, color="blue", alpha=0.2, label="95% CI" # Assuming 95% CI, label can be more generic.
    )

    # Add reference lines: vertical at treatment start (event time 0) and horizontal at zero effect.
    plt.axvline(x=0, color="black", linestyle="--", label="Treatment Start")
    plt.axhline(y=0, color="gray", linestyle="-")

    # Set plot labels, title, legend, and grid.
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7) # Add a light grid.
    plt.tight_layout() # Adjust plot to prevent labels from overlapping.
    plt.show() # Display the plot.




def DID_org(result_dict: Dict, preppeddict: Dict, method_name: str = "FDID") -> BaseEstimatorResults:
    """
    Convert a FDID/DID result dictionary to a BaseEstimatorResults Pydantic model.

    Parameters
    ----------
    result_dict : dict
        The DID/FDID output dictionary with nested 'Effects', 'Fit', 'Vectors', 'Inference'.
    preppeddict : dict
        Dictionary with prep info, e.g., 'time_labels'.
    method_name : str, optional
        Name of the method (default "FDID").

    Returns
    -------
    BaseEstimatorResults
        Standardized Pydantic model with all results mapped.
    """
    # Extract nested results
    effects_data = result_dict.get("Effects", {})
    fit_data = result_dict.get("Fit", {})
    vectors_data = result_dict.get("Vectors", {})
    inference_data = result_dict.get("Inference", {})

    # --- Effects ---
    effects_model = EffectsResults(
        att=effects_data.get("ATT"),
        att_percent=effects_data.get("Percent ATT"),
        additional_effects={k: v for k, v in effects_data.items() if k not in ["ATT", "Percent ATT"]}
    )

    # --- Fit diagnostics ---
    fit_model = FitDiagnosticsResults(
        r_squared_pre=fit_data.get("R-Squared"),
        rmse_pre=fit_data.get("T0 RMSE")
    )

    # --- Time series ---
    time_series_model = TimeSeriesResults(
        observed_outcome=vectors_data.get("Observed"),
        counterfactual_outcome=vectors_data.get("Counterfactual"),
        estimated_gap=vectors_data.get("Gap")[:, 0] if vectors_data.get("Gap") is not None else None,
        time_periods=np.asarray(preppeddict.get("time_labels"))
    )

    # --- Weights ---
    selected_names = result_dict.get("selected_names", [])
    donor_weights = {name: 1.0 / len(selected_names) for name in selected_names} if selected_names else None
    weights_model = WeightsResults(
        donor_weights=donor_weights,
        summary_stats={
            "num_selected": len(selected_names),
            "total_weight": sum(donor_weights.values()) if donor_weights else None
        } if donor_weights else None
    )

    # --- Inference ---
    inference_model = InferenceResults(
        p_value=inference_data.get("P-Value"),
        ci_lower=inference_data.get("95% CI", (None, None))[0],
        ci_upper=inference_data.get("95% CI", (None, None))[1],
        standard_error=inference_data.get("SE"),
        details={k: v for k, v in inference_data.items() if k not in ["P-Value", "95% CI", "SE"]}
    )

    # --- Method details ---
    method_model = MethodDetailsResults(
        method_name=method_name,
        is_recommended=None,
        parameters_used={"selected_names": list(selected_names)}
    )

    # --- Assemble BaseEstimatorResults ---
    base_results = BaseEstimatorResults(
        effects=effects_model,
        fit_diagnostics=fit_model,
        time_series=time_series_model,
        weights=weights_model,
        inference=inference_model,
        method_details=method_model,
        raw_results=result_dict
    )

    # Include intermediary if present
    if "intermediary" in result_dict:
        base_results.raw_results["intermediary"] = result_dict["intermediary"]

    return base_results






def _build_fscm_results(weight_dicts: Dict[str, any],
        prepped_data: Dict[str, any],
        resultfscm: Dict[str, any],
        donor_names: list[str]
) -> BaseEstimatorResults:
    """
    Construct a standardized master BaseEstimatorResults object
    from weight vectors/dicts and prepped data for all SCM methods.
    """
    Y = prepped_data['donor_matrix']
    y = prepped_data['y']
    T_pre = prepped_data['pre_periods']
    T_post = prepped_data['post_periods']

    sub_results: Dict[str, BaseEstimatorResults] = {}

    for method_name, w in weight_dicts.items():
        # Convert numpy array to dict if needed
        if isinstance(w, np.ndarray):
            w = {name: float(val) for name, val in zip(donor_names, w)}

        # Compute counterfactual
        weights_array = np.array(list(w.values()))
        y_hat = Y @ weights_array

        # Compute effects
        eff_dict, fit_dict, vectors_info = effects.calculate(y, y_hat, T_pre, T_post)

        # Wrap effects into standardized EffectsResults
        effects_res = EffectsResults(
            att=eff_dict.get('ATT'),
            att_percent=eff_dict.get('Percent ATT'),
            att_std_err=eff_dict.get('ATT_SE'),
            additional_effects={k: v for k, v in eff_dict.items() if k not in ['ATT', 'Percent ATT', 'ATT_SE']}
        )

        # Wrap fit dictionary into FitDiagnosticsResults
        fit_res = FitDiagnosticsResults(
            rmse_pre=fit_dict.get('T0 RMSE'),
            rmse_post=fit_dict.get('T1 RMSE'),
            r_squared_pre=fit_dict.get('R-Squared'),
            additional_metrics={k: v for k, v in fit_dict.items() if k not in ['T0 RMSE', 'T1 RMSE', 'R-Squared']}
        )

        # Wrap time series
        ts_res = TimeSeriesResults(
            observed_outcome=vectors_info.get('Observed Unit'),
            counterfactual_outcome=vectors_info.get('Counterfactual'),
            estimated_gap=vectors_info.get('Gap'),
            time_periods=vectors_info.get('Time Periods')
        )

        # Additional outputs for FSCM / augmented SCM
        additional_outputs = {}
        if method_name == "Forward SCM":
            additional_outputs['candidate_dict'] = resultfscm['Forward SCM']['candidate_dict']
        if method_name == "Forward Augmented SCM":
            additional_outputs['validation_results'] = resultfscm['Forward Aumented SCM'].get('validation_results')
            additional_outputs['lambda'] = resultfscm['Forward Aumented SCM'].get('lambda')

        # Wrap weights
        weights_res = WeightsResults(donor_weights=w)

        # Construct BaseEstimatorResults for this method
        sub_results[method_name] = BaseEstimatorResults(
            effects=effects_res,
            fit_diagnostics=fit_res,
            time_series=ts_res,
            weights=weights_res,
            method_details={"name": method_name},
            additional_outputs=additional_outputs,
            raw_results=resultfscm.get(method_name)
        )

    # Construct master results object
    master_results = BaseEstimatorResults(
        sub_method_results=sub_results,
        additional_outputs=prepped_data,
        raw_results=resultfscm
    )

    return master_results


