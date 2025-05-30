import numpy as np
import cvxpy as cp
from collections import defaultdict
from copy import deepcopy
from scipy import stats
from typing import Tuple, List, Optional, Dict, Any, DefaultDict
import warnings

from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError, MlsynthEstimationError


def estimate_cohort_sdid_effects(
    cohort_adoption_period: int,
    cohort_data_dict: Dict[str, Any],
    pooled_event_time_effects_accumulator: DefaultDict[float, List[Tuple[int, float]]]
) -> Dict[str, Any]:
    """Estimate Synthetic Difference-in-Differences (SDID) effects for a specific cohort.

    This function calculates SDID treatment effects, synthetic control outcomes,
    and related metrics for a single cohort of treated units. It involves
    estimating unit (donor) weights and time weights, computing a bias
    correction term, and then deriving the treatment effects relative to the
    cohort's specific treatment adoption period `cohort_adoption_period`.

    The results from each cohort (event-time effects and number of treated units)
    are accumulated into the `pooled_event_time_effects_accumulator` dictionary,
    which is modified in place.

    Parameters
    ----------
    cohort_adoption_period : int
        Adoption period (treatment start time) for the current cohort. This is
        typically a specific time period index (e.g., year).
    cohort_data_dict : Dict[str, Any]
        A dictionary containing data specific to the current cohort. Expected keys:
        - "y" : np.ndarray
            Outcome matrix for treated units in this cohort.
            Shape (total_time_periods, N_treated_cohort), where total_time_periods is the total number
            of time periods in the panel, and N_treated_cohort is the number
            of treated units in this specific cohort.
        - "donor_matrix" : np.ndarray
            Matrix of outcomes for all donor units available to this cohort.
            Shape (total_time_periods, N_donors).
        - "total_periods" : int
            Total number of time periods (total_time_periods) in the panel.
        - "pre_periods" : int
            Number of pre-treatment periods (num_pre_treatment_periods_cohort) relative to this cohort's
            adoption period `cohort_adoption_period`.
        - "post_periods" : int
            Number of post-treatment periods (num_post_treatment_periods_cohort) relative to `cohort_adoption_period`.
        - "treated_indices" : List[int]
            List of original indices identifying the treated units in this cohort.
            Used to determine `N_treated_cohort`.
    pooled_event_time_effects_accumulator : DefaultDict[float, List[Tuple[int, float]]]
        A dictionary (typically `collections.defaultdict(list)`) that accumulates
        event-time effects across all cohorts.
        - Keys are event times `ell` (float, relative to treatment start, e.g., -2, -1, 0, 1, 2).
        - Values are lists of tuples, where each tuple is `(N_treated_cohort, effect_value)`.
        This dictionary is updated in place by this function, adding the
        contributions from the current cohort.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing detailed results for the processed cohort:
        - "effects" : np.ndarray
            Array of (event_time, treatment_effect) pairs for all total_time_periods periods.
            Shape (total_time_periods, 2).
        - "pre_effects" : np.ndarray
            Array of (event_time, treatment_effect) pairs for pre-intervention periods.
            Shape (N_pre_effects, 2) or empty if no pre-effects.
        - "post_effects" : np.ndarray
            Array of (event_time, treatment_effect) pairs for post-intervention
            periods (including event time 0). Shape (N_post_effects, 2) or empty.
        - "actual" : np.ndarray
            Mean actual outcome trajectory for the treated units in this cohort.
            Shape (total_time_periods,).
        - "counterfactual" : np.ndarray
            Raw synthetic control outcome trajectory (cohort_donor_outcomes_matrix @ optimal_unit_weights_vector).
            Shape (total_time_periods,). Can contain NaNs if weights are not estimated.
        - "fitted_counterfactual" : np.ndarray
            Bias-corrected synthetic control outcome trajectory.
            Shape (total_time_periods,). Can contain NaNs.
        - "att" : float
            Average Treatment Effect on the Treated (ATT) for this cohort,
            averaged over its post-intervention periods. NaN if no post-periods
            or if effects cannot be calculated.
        - "treatment_effects_series" : np.ndarray
            Time series of treatment effects (actual - fitted_counterfactual)
            for all total_time_periods periods. Shape (total_time_periods,).
        - "ell" : np.ndarray
            Array of event times relative to this cohort's treatment start `cohort_adoption_period`.
            Shape (total_time_periods,). For example, `ell = 0` corresponds to period `cohort_adoption_period`.

    Examples
    --------
    >>> # Conceptual example due to complex data setup
    >>> # Assume 'adoption_period_ex' is the treatment start year for this cohort
    >>> adoption_period_ex = 2005
    >>> # Assume 'cohort_data_example' is a dict with keys like "y", "donor_matrix", etc.
    >>> # and 'pooled_effects_accumulator_ex' is a defaultdict(list)
    >>> total_periods_ex, n_treated_ex, n_donors_ex, n_pre_periods_ex = 10, 2, 5, 5 # Example dimensions
    >>> cohort_data_example_ex = {
    ...     "y": np.random.rand(total_periods_ex, n_treated_ex),
    ...     "donor_matrix": np.random.rand(total_periods_ex, n_donors_ex),
    ...     "total_periods": total_periods_ex,
    ...     "pre_periods": n_pre_periods_ex, # Number of pre-treatment periods for this cohort
    ...     "post_periods": total_periods_ex - n_pre_periods_ex,
    ...     "treated_indices": list(range(n_treated_ex))
    ... }
    >>> pooled_effects_accumulator_ex = defaultdict(list)
    >>> # Mock dependent functions for a runnable example
    >>> with warnings.catch_warnings(): # Suppress potential warnings from mock data
    ...     warnings.simplefilter("ignore")
    ...     # Mocking internal weight and regularization functions
    ...     # These would normally perform complex optimizations
    ...     mock_zeta_ex = 0.1
    ...     mock_unit_w_ex = np.full(n_donors_ex, 1.0/n_donors_ex)
    ...     mock_time_w_ex = np.full(n_pre_periods_ex, 1.0/n_pre_periods_ex)
    ...     from unittest.mock import patch
    ...     with patch('mlsynth.utils.sdidutils.compute_regularization', return_value=mock_zeta_ex), \
    ...          patch('mlsynth.utils.sdidutils.unit_weights', return_value=(0.0, mock_unit_w_ex)), \
    ...          patch('mlsynth.utils.sdidutils.fit_time_weights', return_value=(0.0, mock_time_w_ex)):
    ...         results_cohort_ex = estimate_cohort_sdid_effects(
    ...             adoption_period_ex, cohort_data_example_ex, pooled_effects_accumulator_ex
    ...         )
    >>> print(f"Cohort ATT: {results_cohort_ex['att']:.3f}") # Example output
    Cohort ATT: ...
    >>> # pooled_effects_accumulator_ex would be updated in place
    >>> # print(len(pooled_effects_accumulator_ex[-1])) # Example check
    """
    # Input Validation
    # Ensure cohort_adoption_period is an integer, as it represents a specific time index.
    if not isinstance(cohort_adoption_period, int):
        raise MlsynthConfigError("cohort_adoption_period must be an integer.")
    # Ensure cohort_data_dict is a dictionary and contains all necessary keys.
    if not isinstance(cohort_data_dict, dict):
        raise MlsynthDataError("cohort_data_dict must be a dictionary.")
    
    required_keys = ["y", "donor_matrix", "total_periods", "pre_periods", "post_periods", "treated_indices"]
    for key in required_keys:
        if key not in cohort_data_dict:
            raise MlsynthDataError(f"Missing required key '{key}' in cohort_data_dict.")

    # Extract data for the current cohort from the input dictionary.
    # `y` is the outcome matrix for treated units in this cohort.
    # `donor_matrix` contains outcomes for all potential donor units.
    # `total_periods` is the total number of time periods in the panel.
    # `pre_periods` is the number of pre-treatment periods for this cohort.
    # `post_periods` is the number of post-treatment periods for this cohort.
    # `treated_indices` identifies the treated units in this cohort.
    cohort_treated_outcomes_matrix: np.ndarray = cohort_data_dict["y"]
    cohort_donor_outcomes_matrix: np.ndarray = cohort_data_dict["donor_matrix"]
    total_time_periods: int = cohort_data_dict["total_periods"]
    num_pre_treatment_periods_cohort: int = cohort_data_dict["pre_periods"]
    num_post_treatment_periods_cohort: int = cohort_data_dict["post_periods"]
    treated_indices: List[int] = cohort_data_dict["treated_indices"]

    # Validate types and dimensions of extracted data.
    if not isinstance(cohort_treated_outcomes_matrix, np.ndarray):
        raise MlsynthDataError("'y' in cohort_data_dict must be a NumPy array.")
    if cohort_treated_outcomes_matrix.ndim != 2:
        raise MlsynthDataError("'y' must be a 2D array (total_periods, N_treated_cohort).")
    if not isinstance(cohort_donor_outcomes_matrix, np.ndarray):
        raise MlsynthDataError("'donor_matrix' in cohort_data_dict must be a NumPy array.")
    if cohort_donor_outcomes_matrix.ndim != 2:
        raise MlsynthDataError("'donor_matrix' must be a 2D array (total_periods, N_donors).")
    
    if not isinstance(total_time_periods, int) or total_time_periods <= 0:
        raise MlsynthDataError("'total_periods' must be a positive integer.")
    if not isinstance(num_pre_treatment_periods_cohort, int) or num_pre_treatment_periods_cohort < 0:
        raise MlsynthDataError("'pre_periods' must be a non-negative integer.")
    if not isinstance(num_post_treatment_periods_cohort, int) or num_post_treatment_periods_cohort < 0:
        raise MlsynthDataError("'post_periods' must be a non-negative integer.")

    # Validate consistency of dimensions.
    if cohort_treated_outcomes_matrix.shape[0] != total_time_periods:
        raise MlsynthDataError(f"Shape mismatch: 'y' has {cohort_treated_outcomes_matrix.shape[0]} periods, expected {total_time_periods}.")
    if cohort_donor_outcomes_matrix.shape[0] != total_time_periods:
        raise MlsynthDataError(
            f"Shape mismatch: 'donor_matrix' has {cohort_donor_outcomes_matrix.shape[0]} periods, expected {total_time_periods}."
        )
    if len(treated_indices) != cohort_treated_outcomes_matrix.shape[1]:
        raise MlsynthDataError(
            f"Shape mismatch: 'y' has {cohort_treated_outcomes_matrix.shape[1]} treated units, "
            f"but 'treated_indices' has {len(treated_indices)} elements."
        )
    # Warn if pre and post periods sum to more than total periods, which might indicate an issue or specific setup.
    if num_pre_treatment_periods_cohort + num_post_treatment_periods_cohort > total_time_periods:
         warnings.warn(
            f"Sum of pre_periods ({num_pre_treatment_periods_cohort}) and post_periods ({num_post_treatment_periods_cohort}) "
            f"exceeds total_periods ({total_time_periods}). This might be valid if cohort_adoption_period implies overlap "
            "or specific event windowing, but double-check data setup.", UserWarning
        )

    # Calculate the mean outcome trajectory for the treated units in this cohort.
    cohort_mean_treated_outcome_series: np.ndarray = cohort_treated_outcomes_matrix.mean(axis=1)
    
    # Prepare data for weight estimation: donor outcomes and mean treated outcome in the pre-treatment period.
    donor_outcomes_pre_treatment_cohort: np.ndarray = cohort_donor_outcomes_matrix[:num_pre_treatment_periods_cohort, :]
    mean_treated_outcome_pre_treatment_cohort: np.ndarray = cohort_mean_treated_outcome_series[:num_pre_treatment_periods_cohort]

    # Initialize weight-related variables.
    optimal_unit_weight_intercept: Optional[float] = None
    optimal_unit_weights_vector: Optional[np.ndarray] = None
    optimal_time_weight_intercept: Optional[float] = None
    optimal_time_weights_vector: Optional[np.ndarray] = None

    # Estimate unit (donor) weights and time weights.
    # Weights are only estimated if there are pre-treatment periods.
    if num_pre_treatment_periods_cohort == 0:
        # Not enough pre-periods to estimate weights, regularization parameter is not applicable.
        regularization_parameter_zeta = np.nan
    else:
        # Compute regularization parameter zeta based on donor outcomes and post-treatment period length.
        regularization_parameter_zeta = compute_regularization(donor_outcomes_pre_treatment_cohort, num_post_treatment_periods_cohort)
        # Estimate unit weights (omega) and intercept.
        optimal_unit_weight_intercept, optimal_unit_weights_vector = unit_weights(
            donor_outcomes_pre_treatment_cohort, mean_treated_outcome_pre_treatment_cohort, regularization_parameter_zeta
        )

        # Estimate time weights (lambda) and intercept if there are post-treatment periods and valid donor data.
        if num_post_treatment_periods_cohort == 0:
            pass # optimal_time_weights_vector remains None if no post-periods.
        else:
            # Calculate mean donor outcomes in the post-treatment period.
            if cohort_donor_outcomes_matrix[num_pre_treatment_periods_cohort:, :].size == 0:
                 mean_donor_outcomes_post_treatment_cohort = np.full(cohort_donor_outcomes_matrix.shape[1], np.nan) # Handle empty post-period donor matrix.
            else:
                 mean_donor_outcomes_post_treatment_cohort = cohort_donor_outcomes_matrix[num_pre_treatment_periods_cohort:, :].mean(axis=0)
            
            # Proceed with time weight estimation if mean post-treatment donor outcomes are not all NaN.
            if np.all(np.isnan(mean_donor_outcomes_post_treatment_cohort)):
                pass # optimal_time_weights_vector remains None if all post-period donor means are NaN.
            else:
                optimal_time_weight_intercept, optimal_time_weights_vector = fit_time_weights(
                    donor_outcomes_pre_treatment_cohort, mean_donor_outcomes_post_treatment_cohort
                )

    # Assign final weights, handling cases where estimation might have failed (returned None).
    final_optimal_unit_weights: np.ndarray
    if optimal_unit_weights_vector is None:
        # If unit weights couldn't be estimated, fill with NaNs.
        final_optimal_unit_weights = np.full(cohort_donor_outcomes_matrix.shape[1] if cohort_donor_outcomes_matrix.shape[1] > 0 else 0, np.nan)
    else:
        final_optimal_unit_weights = optimal_unit_weights_vector
    
    final_optimal_time_weights: np.ndarray
    if optimal_time_weights_vector is None:
        # If time weights couldn't be estimated, fill with NaNs.
        final_optimal_time_weights = np.full(num_pre_treatment_periods_cohort if num_pre_treatment_periods_cohort > 0 else 0, np.nan)
    else:
        final_optimal_time_weights = optimal_time_weights_vector

    # Calculate bias correction term and raw synthetic control series.
    bias_correction: float
    raw_synthetic_control_series: np.ndarray
    
    # If weights are NaN or no pre-periods, bias correction and synthetic control are NaN.
    if np.any(np.isnan(final_optimal_unit_weights)) or np.any(np.isnan(final_optimal_time_weights)) or num_pre_treatment_periods_cohort == 0:
        bias_correction = np.nan
        raw_synthetic_control_series = np.full(total_time_periods, np.nan)
    else:
        # Construct the raw synthetic control by applying unit weights to donor outcomes.
        raw_synthetic_control_series = cohort_donor_outcomes_matrix @ final_optimal_unit_weights
        
        # Calculate the bias correction term: lambda' * Y_treated_pre - lambda' * Y_donors_pre * omega
        bias_correction_term1 = final_optimal_time_weights @ mean_treated_outcome_pre_treatment_cohort
        bias_correction_term2_intermediate = donor_outcomes_pre_treatment_cohort @ final_optimal_unit_weights
        bias_correction_term2 = final_optimal_time_weights @ bias_correction_term2_intermediate
        
        if np.isnan(bias_correction_term1) or np.isnan(bias_correction_term2):
            bias_correction = np.nan # If any part of bias correction is NaN, the whole term is NaN.
        else:
            bias_correction = bias_correction_term1 - bias_correction_term2
        
    # Calculate event-time effects (tau).
    # Event times are relative to the cohort's specific treatment adoption period.
    # `cohort_adoption_period - 1` aligns event time 0 with the first treatment period.
    event_times_relative_to_cohort_treatment: np.ndarray = np.arange(total_time_periods) - (cohort_adoption_period - 1)
    
    # Treatment effect (tau) = Actual_Treated_Outcome - (Raw_Synthetic_Control + Bias_Correction).
    # NaNs in raw_synthetic_control_series or bias_correction will propagate to tau.
    cohort_treatment_effects_series_tau: np.ndarray = cohort_mean_treated_outcome_series - (raw_synthetic_control_series + bias_correction)
    
    # Combine event times and treatment effects into a single array.
    cohort_effects_all_times: np.ndarray = np.column_stack((event_times_relative_to_cohort_treatment, cohort_treatment_effects_series_tau))

    # Separate pre-treatment and post-treatment effects.
    pre_mask: np.ndarray = event_times_relative_to_cohort_treatment < 0
    post_mask: np.ndarray = event_times_relative_to_cohort_treatment >= 0 # Event time 0 is considered post-treatment.
    
    pre_effects_array: np.ndarray = (
        np.column_stack((event_times_relative_to_cohort_treatment[pre_mask], cohort_treatment_effects_series_tau[pre_mask]))
        if np.any(pre_mask)
        else np.array([]) # Return empty array if no pre-treatment effects.
    )
    post_effects_array: np.ndarray = (
        np.column_stack((event_times_relative_to_cohort_treatment[post_mask], cohort_treatment_effects_series_tau[post_mask]))
        if np.any(post_mask)
        else np.array([]) # Return empty array if no post-treatment effects.
    )

    # Accumulate event-time effects for pooling across cohorts.
    # The accumulator is modified in place.
    num_treated_in_cohort: int = len(cohort_data_dict["treated_indices"])
    if post_effects_array.size > 0:
        for event_time, treatment_effect_value in zip(post_effects_array[:,0], post_effects_array[:,1]):
            if not np.isnan(treatment_effect_value): # Only add non-NaN effects to the accumulator.
                 pooled_event_time_effects_accumulator[event_time].append((num_treated_in_cohort, treatment_effect_value))
    if pre_effects_array.size > 0:
        for event_time, treatment_effect_value in zip(pre_effects_array[:,0], pre_effects_array[:,1]):
            if not np.isnan(treatment_effect_value): # Only add non-NaN effects to the accumulator.
                 pooled_event_time_effects_accumulator[event_time].append((num_treated_in_cohort, treatment_effect_value))
        
    # Calculate Average Treatment Effect on the Treated (ATT) for this cohort.
    # This is the mean of post-treatment effects.
    average_treatment_effect_cohort: float = np.nanmean(cohort_treatment_effects_series_tau[post_mask]) if np.any(post_mask) else np.nan

    # Return a dictionary of results for this cohort.
    return {
        "effects": cohort_effects_all_times,
        "pre_effects": pre_effects_array,
        "post_effects": post_effects_array,
        "actual": cohort_mean_treated_outcome_series,
        "counterfactual": raw_synthetic_control_series, 
        "fitted_counterfactual": raw_synthetic_control_series + bias_correction,
        "att": average_treatment_effect_cohort,
        "treatment_effects_series": cohort_treatment_effects_series_tau,
        "ell": event_times_relative_to_cohort_treatment,
    }


def estimate_event_study_sdid(
    prepped_event_study_data: Dict[str, Any], placebo_iterations: int = 1000, seed: int = 1400
) -> Dict[str, Any]:
    """
    Estimate event-study SDID effects with placebo inference for variance, SE, and 95% CI.

    Parameters
    ----------
    prepped_event_study_data : Dict[str, Any]
        Preprocessed data from a function like `dataprep_event_study_sdid`.
        Expected to contain a 'cohorts' key, which is a dictionary mapping
        cohort adoption periods (int) to cohort-specific data dictionaries.
    placebo_iterations : int, optional
        Number of placebo resamples (B) for variance estimation, by default 1000.
    seed : int, optional
        Random seed for reproducibility of placebo sampling, by default 1400.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing various estimates:
        - "tau_a_ell" : Dict[int, Dict[str, Any]]
            Per-cohort detailed results. Keys are cohort adoption periods.
            Values are dictionaries from `estimate_cohort_sdid_effects`.
        - "tau_ell" : Dict[float, float]
            Pooled event-time effects (weighted average across cohorts).
            Keys are event times `ell`, values are the pooled effect estimates.
        - "att" : float
            Overall Average Treatment Effect on Treated, aggregated across all
            cohorts and post-treatment periods.
        - "att_se" : float
            Standard error for the overall ATT, estimated via placebo inference.
        - "att_ci" : List[float]
            95% Confidence interval [lower, upper] for the overall ATT.
        - "cohort_estimates" : Dict[int, Dict[str, Any]]
            Per-cohort summary statistics. Keys are cohort adoption periods.
            Values are dicts with "att", "att_se", "att_ci", and "event_estimates"
            (a dict of event_time -> {tau, se, ci}).
        - "pooled_estimates" : Dict[float, Dict[str, Any]]
            Pooled event-time estimates with SE and CI. Keys are event times `ell`.
            Values are dicts with "tau", "se", "ci".
        - "placebo_att_values" : List[float]
            List of ATT values obtained from each placebo iteration. Useful for
            diagnostics or alternative inference methods.

    Examples
    --------
    >>> # Conceptual example due to the complexity of `prepped_event_study_data` data
    >>> # `prepped_data_example` would be the output of a data preparation function
    >>> # specific to event study SDID, containing multiple cohorts.
    >>> prepped_data_example = {
    ...     "cohorts": {
    ...         2005: { # Data for cohort treated in 2005
    ...             "y": np.random.rand(10, 2), "donor_matrix": np.random.rand(10, 5),
    ...             "total_periods": 10, "pre_periods": 5, "post_periods": 5,
    ...             "treated_indices": [0, 1]
    ...         },
    ...         2006: { # Data for cohort treated in 2006
    ...             "y": np.random.rand(10, 1), "donor_matrix": np.random.rand(10, 5),
    ...             "total_periods": 10, "pre_periods": 6, "post_periods": 4,
    ...             "treated_indices": [2]
    ...         }
    ...     }
    ... }
    >>> # Mock dependent functions for a runnable example
    >>> from unittest.mock import patch
    >>> mock_zeta = 0.1
    >>> mock_unit_w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    >>> mock_time_w_c1 = np.full(5, 0.2)
    >>> mock_time_w_c2 = np.full(6, 1/6)
    >>> # This example is highly simplified and primarily tests structure
    >>> with warnings.catch_warnings(): # Suppress potential warnings
    ...     warnings.simplefilter("ignore")
    ...     # Mocking internal weight and regularization functions
    ...     # Need to handle calls for each cohort within estimate_cohort_sdid_effects
    ...     # and also for each placebo iteration within estimate_placebo_variance
    ...     # This level of mocking is complex for a simple docstring example.
    ...     # We'll assume the function runs and check for key existence.
    ...     with patch('mlsynth.utils.sdidutils.compute_regularization', return_value=mock_zeta), \
    ...          patch('mlsynth.utils.sdidutils.unit_weights', return_value=(0.0, mock_unit_w)), \
    ...          patch('mlsynth.utils.sdidutils.fit_time_weights', side_effect=[(0.0, mock_time_w_c1), (0.0, mock_time_w_c2)] * (1 + 10)): # 1 real + B mock iterations
    ...         results_event_study = estimate_event_study_sdid(prepped_data_example, placebo_iterations=10, seed=42)
    >>> print("Overall ATT:", results_event_study["att"]) # Example output
    Overall ATT: ...
    >>> print("Pooled estimate for event time 0:", results_event_study["pooled_estimates"].get(0.0, {}).get("tau"))
    Pooled estimate for event time 0: ...
    >>> assert "placebo_att_values" in results_event_study
    """
    # Input Validation
    # Ensure prepped_event_study_data is a dictionary and contains the 'cohorts' key.
    if not isinstance(prepped_event_study_data, dict) or "cohorts" not in prepped_event_study_data:
        raise MlsynthDataError("prepped_event_study_data must be a dict with a 'cohorts' key.")
    # Ensure 'cohorts' itself is a dictionary.
    if not isinstance(prepped_event_study_data["cohorts"], dict):
        raise MlsynthDataError("'cohorts' in prepped_event_study_data must be a dictionary.")
    # Validate placebo_iterations and seed.
    if not isinstance(placebo_iterations, int) or placebo_iterations < 0:
        raise MlsynthConfigError("placebo_iterations must be a non-negative integer.")
    if not isinstance(seed, int): # seed can be any int for np.random.seed
        raise MlsynthConfigError("seed must be an integer.")

    # Extract the dictionary of all cohorts' data.
    all_cohorts_data_dict: Dict[int, Dict[str, Any]] = prepped_event_study_data["cohorts"]
    # Initialize dictionaries to store results.
    results_by_cohort: Dict[int, Dict[str, Any]] = {} # Stores detailed results for each cohort.
    # Accumulator for event-time effects, to be pooled across cohorts.
    current_pooled_event_time_effects_accumulator: DefaultDict[float, List[Tuple[int, float]]] = defaultdict(list)

    # Iterate through each cohort, estimate its SDID effects, and accumulate pooled effects.
    for cohort_period, current_cohort_data in all_cohorts_data_dict.items():
        if not isinstance(cohort_period, int): # Validate cohort key (adoption period).
            raise MlsynthDataError(f"Cohort key {cohort_period} must be an integer (adoption period).")
        # `estimate_cohort_sdid_effects` calculates effects for the current cohort
        # and updates `current_pooled_event_time_effects_accumulator` in place.
        results_by_cohort[cohort_period] = estimate_cohort_sdid_effects(
            cohort_period, current_cohort_data, current_pooled_event_time_effects_accumulator
        )

    # Calculate final pooled event-time effects (tau_ell).
    # This is a weighted average of cohort-specific event-time effects,
    # weighted by the number of treated units in each cohort contributing to that event time.
    final_pooled_event_time_effects: Dict[float, float] = {
        event_time: sum(num_treated_units_in_effect_tuple * effect_value_in_tuple for num_treated_units_in_effect_tuple, effect_value_in_tuple in effects_for_current_event_time) /
             sum(num_treated_units_in_effect_tuple for num_treated_units_in_effect_tuple, _ in effects_for_current_event_time)
        for event_time, effects_for_current_event_time in current_pooled_event_time_effects_accumulator.items()
        if sum(num_treated_units_in_effect_tuple for num_treated_units_in_effect_tuple, _ in effects_for_current_event_time) > 0 # Avoid division by zero if no units contribute.
    }

    # Calculate the overall Average Treatment Effect on the Treated (ATT).
    # This is an aggregation of cohort-specific ATTs, weighted by the number of
    # treated units and post-treatment periods in each cohort.
    total_weighted_att_sum: float = 0.0
    total_post_exposure_units_times_periods: int = 0
    for cohort_period, current_cohort_data in all_cohorts_data_dict.items():
        current_cohort_att = results_by_cohort[cohort_period]["att"] # Get ATT for the current cohort.
        if not np.isnan(current_cohort_att): # Only include non-NaN ATTs.
            num_treated_in_cohort = len(current_cohort_data["treated_indices"])
            num_post_periods_cohort = current_cohort_data["post_periods"]
            # Weight by (number of treated units * number of post-treatment periods).
            total_weighted_att_sum += current_cohort_att * num_treated_in_cohort * num_post_periods_cohort
            total_post_exposure_units_times_periods += num_treated_in_cohort * num_post_periods_cohort
    
    # Calculate overall ATT; NaN if no valid contributions.
    overall_average_treatment_effect: float = total_weighted_att_sum / total_post_exposure_units_times_periods if total_post_exposure_units_times_periods > 0 else np.nan

    # Estimate variances for ATTs and event-time effects using placebo inference.
    placebo_variances_results: Dict[str, Any] = estimate_placebo_variance(
        prepped_event_study_data, placebo_iterations, seed
    )

    # Combine actual estimates with placebo-based standard errors and confidence intervals for each cohort.
    final_cohort_estimates_with_ci: Dict[int, Dict[str, Any]] = {}
    for cohort_period, current_cohort_result_data in results_by_cohort.items():
        current_cohort_att_value = current_cohort_result_data["att"]
        # Get placebo-estimated variance for this cohort's ATT.
        current_cohort_att_variance = placebo_variances_results["cohort_variances"].get(cohort_period, np.nan)
        current_cohort_att_standard_error = np.sqrt(current_cohort_att_variance) if current_cohort_att_variance >=0 and not np.isnan(current_cohort_att_variance) else np.nan
        
        # Calculate SE and CI for each event-time effect within this cohort.
        event_time_estimates_for_current_cohort: Dict[int, Dict[str, Any]] = {}
        for effect_arr_name_loop_var in ["pre_effects", "post_effects"]: # Iterate over pre and post effects arrays.
            current_effects_array = current_cohort_result_data[effect_arr_name_loop_var]
            if current_effects_array.size > 0:
                for current_event_time_value, current_tau_value in zip(current_effects_array[:, 0], current_effects_array[:, 1]):
                    current_event_time_int = int(current_event_time_value)
                    # Get placebo-estimated variance for this specific event-time effect.
                    current_event_time_variance = placebo_variances_results["event_variances"].get(cohort_period, {}).get(current_event_time_int, np.nan)
                    current_event_time_standard_error = np.sqrt(current_event_time_variance) if current_event_time_variance >=0 and not np.isnan(current_event_time_variance) else np.nan
                    event_time_estimates_for_current_cohort[current_event_time_int] = {
                        "tau": current_tau_value,
                        "se": current_event_time_standard_error,
                        # Calculate 95% CI using normal approximation.
                        "ci": [current_tau_value - stats.norm.ppf(0.975) * current_event_time_standard_error, current_tau_value + stats.norm.ppf(0.975) * current_event_time_standard_error] if not np.isnan(current_event_time_standard_error) else [np.nan, np.nan]
                    }
        final_cohort_estimates_with_ci[cohort_period] = {
            "att": current_cohort_att_value,
            "att_se": current_cohort_att_standard_error,
            "att_ci": [current_cohort_att_value - stats.norm.ppf(0.975) * current_cohort_att_standard_error, current_cohort_att_value + stats.norm.ppf(0.975) * current_cohort_att_standard_error] if not np.isnan(current_cohort_att_standard_error) else [np.nan, np.nan],
            "event_estimates": event_time_estimates_for_current_cohort
        }
        
    # Combine pooled event-time estimates with placebo-based SE and CI.
    final_pooled_estimates_with_ci: Dict[float, Dict[str, Any]] = {}
    for current_event_time_value, current_tau_value in final_pooled_event_time_effects.items():
        # Get placebo-estimated variance for this pooled event-time effect.
        current_pooled_event_variance = placebo_variances_results["pooled_event_variances"].get(current_event_time_value, np.nan)
        current_pooled_event_standard_error = np.sqrt(current_pooled_event_variance) if current_pooled_event_variance >=0 and not np.isnan(current_pooled_event_variance) else np.nan
        final_pooled_estimates_with_ci[current_event_time_value] = {
            "tau": current_tau_value,
            "se": current_pooled_event_standard_error,
            "ci": [current_tau_value - stats.norm.ppf(0.975) * current_pooled_event_standard_error, current_tau_value + stats.norm.ppf(0.975) * current_pooled_event_standard_error] if not np.isnan(current_pooled_event_standard_error) else [np.nan, np.nan]
        }
    
    # Get placebo-estimated variance and SE for the overall ATT.
    overall_att_variance = placebo_variances_results["att_variance"]
    overall_att_standard_error = np.sqrt(overall_att_variance) if overall_att_variance >=0 and not np.isnan(overall_att_variance) else np.nan

    # Retrieve the list of raw placebo ATT values for diagnostics.
    placebo_att_values_raw_list = placebo_variances_results.get("placebo_att_values")

    # Return a comprehensive dictionary of results.
    return {
        "tau_a_ell": results_by_cohort,
        "tau_ell": final_pooled_event_time_effects,
        "att": overall_average_treatment_effect,
        "att_se": overall_att_standard_error,
        "att_ci": [overall_average_treatment_effect - stats.norm.ppf(0.975) * overall_att_standard_error, overall_average_treatment_effect + stats.norm.ppf(0.975) * overall_att_standard_error] if not np.isnan(overall_att_standard_error) else [np.nan, np.nan],
        "cohort_estimates": final_cohort_estimates_with_ci,
        "pooled_estimates": final_pooled_estimates_with_ci,
        "placebo_att_values": placebo_att_values_raw_list
    }


def estimate_placebo_variance(
    prepped_event_study_data: Dict[str, Any], num_placebo_iterations: int, seed: int
) -> Dict[str, Any]:
    """
    Estimate variance of ATT and event-time effects using placebo inference.

    Parameters
    ----------
    prepped_event_study_data : Dict[str, Any]
        Preprocessed data from `dataprep_event_study_sdid` or similar.
    num_placebo_iterations : int
        Number of placebo iterations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing variance estimates and placebo ATT values:
        - "att_variance" (float): Variance of the overall ATT.
        - "cohort_variances" (Dict[int, float]): Variances of cohort-specific ATTs.
          Keys are cohort adoption periods.
        - "event_variances" (Dict[int, Dict[int, float]]): Variances of
          cohort-specific event-time effects. Outer keys are cohort adoption
          periods, inner keys are event times `ell`.
        - "pooled_event_variances" (Dict[float, float]): Variances of pooled
          event-time effects. Keys are event times `ell`.
        - "placebo_att_values" (List[float]): List of ATT values from each
          placebo iteration, useful for diagnostics.

    Notes
    -----
    This function performs placebo tests by iteratively reassigning control units
    as pseudo-treated units and re-estimating effects. The variance of these
    placebo effects is then used as an estimate of the variance of the actual
    treatment effects.
    A warning is issued if the number of unique control units is less than the
    total number of treated units across all cohorts, as this may compromise
    the reliability of placebo inference.

    Examples
    --------
    >>> # Conceptual example due to the complexity of `prepped_event_study_data` data
    >>> # `prepped_data_example` would be the output of a data preparation function
    >>> # specific to event study SDID, containing multiple cohorts.
    >>> prepped_data_example = {
    ...     "cohorts": {
    ...         2005: { # Data for cohort treated in 2005
    ...             "y": np.random.rand(10, 2), "donor_matrix": np.random.rand(10, 5),
    ...             "total_periods": 10, "pre_periods": 5, "post_periods": 5,
    ...             "treated_indices": [0, 1] # Original treated indices
    ...         },
    ...         2006: { # Data for cohort treated in 2006
    ...             "y": np.random.rand(10, 1), "donor_matrix": np.random.rand(10, 5),
    ...             "total_periods": 10, "pre_periods": 6, "post_periods": 4,
    ...             "treated_indices": [2] # Original treated index
    ...         }
    ...     }
    ... }
    >>> # Mock dependent functions for a runnable example
    >>> from unittest.mock import patch
    >>> mock_zeta = 0.1
    >>> mock_unit_w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    >>> mock_time_w_c1 = np.full(5, 0.2)
    >>> mock_time_w_c2 = np.full(6, 1/6)
    >>> # This example is highly simplified.
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     # Mocking internal weight and regularization functions.
    ...     # The side_effect needs to cover calls for each cohort and each placebo iteration.
    ...     # For num_placebo_iterations=3, and 2 cohorts, estimate_cohort_sdid_effects is called 2*3=6 times.
    ...     # Each call to estimate_cohort_sdid_effects calls fit_time_weights once.
    ...     # So, fit_time_weights needs 6 return values.
    ...     fit_time_weights_returns = []
    ...     for _ in range(3): # num_placebo_iterations
    ...         fit_time_weights_returns.append((0.0, mock_time_w_c1)) # For cohort 2005 placebo
    ...         fit_time_weights_returns.append((0.0, mock_time_w_c2)) # For cohort 2006 placebo
    ...
    ...     with patch('mlsynth.utils.sdidutils.compute_regularization', return_value=mock_zeta), \
    ...          patch('mlsynth.utils.sdidutils.unit_weights', return_value=(0.0, mock_unit_w)), \
    ...          patch('mlsynth.utils.sdidutils.fit_time_weights', side_effect=fit_time_weights_returns):
    ...         variance_results = estimate_placebo_variance(prepped_data_example, num_placebo_iterations=3, seed=42)
    >>> print(f"ATT Variance: {variance_results['att_variance']}") # Example output
    ATT Variance: ...
    >>> assert "placebo_att_values" in variance_results
    >>> assert len(variance_results["placebo_att_values"]) <= 3 # Can be less if NaNs occur
    """
    # Input Validation
    if not isinstance(prepped_event_study_data, dict) or "cohorts" not in prepped_event_study_data:
        raise MlsynthDataError("prepped_event_study_data must be a dict with a 'cohorts' key.")
    if not isinstance(prepped_event_study_data["cohorts"], dict):
        raise MlsynthDataError("'cohorts' in prepped_event_study_data must be a dictionary.")
    if not isinstance(num_placebo_iterations, int) or num_placebo_iterations < 0:
        raise MlsynthConfigError("num_placebo_iterations must be a non-negative integer.")
    if not isinstance(seed, int):
        raise MlsynthConfigError("seed must be an integer.")

    # Set random seed for reproducibility of placebo sampling.
    np.random.seed(seed)
    # Extract original cohort data.
    original_cohorts_data_dict: Dict[int, Dict[str, Any]] = prepped_event_study_data["cohorts"]
    
    # Initialize lists and dictionaries to store results from placebo iterations.
    placebo_overall_att_list: List[float] = [] # Stores overall ATT from each placebo iteration.
    # Stores cohort-specific ATTs from each placebo iteration.
    placebo_cohort_att_dict: DefaultDict[int, List[float]] = defaultdict(list)
    # Stores cohort-specific event-time effects from each placebo iteration.
    placebo_cohort_event_effects_dict: DefaultDict[int, DefaultDict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    # Stores pooled event-time effects from each placebo iteration.
    placebo_pooled_event_effects_dict: DefaultDict[float, List[float]] = defaultdict(list)

    # Collect all unique donor indices from all cohorts to form the pool of control units for placebo assignment.
    all_donor_indices_from_all_cohorts: List[int] = []
    for _, current_original_cohort_data in original_cohorts_data_dict.items():
        # Assuming donor_matrix columns correspond to donor indices 0 to N_donors-1 for that cohort's donor pool.
        all_donor_indices_from_all_cohorts.extend(list(range(current_original_cohort_data["donor_matrix"].shape[1])))
    
    # Get unique control unit indices available for placebo assignment.
    unique_control_unit_indices: List[int] = sorted(list(set(all_donor_indices_from_all_cohorts)))
    
    # Calculate the total number of treated units across all original cohorts.
    total_treated_units_sum: int = sum(
        len(current_original_cohort_data["treated_indices"]) for _, current_original_cohort_data in original_cohorts_data_dict.items()
    )

    # Warn if the number of unique control units is less than the total number of treated units.
    # This situation can compromise the reliability of placebo inference, as control units
    # might be repeatedly chosen as pseudo-treated units, or there might not be enough
    # distinct units to form credible placebo groups.
    if len(unique_control_unit_indices) < total_treated_units_sum :
        warnings.warn(
            "Placebo inference might be unreliable: number of unique control units "
            "is less than total number of treated units across all cohorts. "
            "Consider if units can be controls for multiple cohorts or if data is limited."
            )

    # Perform placebo iterations.
    for iteration_idx in range(num_placebo_iterations):
        # Create a deep copy of the original cohort data for this placebo iteration to avoid modifying original data.
        current_placebo_iteration_cohorts_data: Dict[int, Dict[str, Any]] = deepcopy(original_cohorts_data_dict)
        # Maintain a list of available control indices for assignment within this iteration.
        current_iteration_available_control_indices: List[int] = list(unique_control_unit_indices)
        
        # For each cohort in this placebo iteration, assign pseudo-treated units from the control pool.
        for current_placebo_cohort_period, current_placebo_cohort_data in current_placebo_iteration_cohorts_data.items():
            num_treated_needed = len(current_placebo_cohort_data["treated_indices"]) # Number of pseudo-treated units needed for this cohort.
            
            current_placebo_treated_indices: np.ndarray
            # If more pseudo-treated units are needed than available unique controls, sample with replacement.
            if num_treated_needed > len(current_iteration_available_control_indices):
                current_placebo_treated_indices = np.random.choice(
                    unique_control_unit_indices, size=num_treated_needed, replace=True
                )
            else:
                # Otherwise, sample without replacement from the currently available controls.
                current_placebo_treated_indices = np.random.choice(
                    current_iteration_available_control_indices, size=num_treated_needed, replace=False
                )
                # Remove chosen controls from the available list for this iteration to ensure they are not re-picked by other cohorts (if sampling w/o replacement).
                for chosen_control_index in current_placebo_treated_indices:
                    if chosen_control_index in current_iteration_available_control_indices: 
                         current_iteration_available_control_indices.remove(chosen_control_index)

            # Update the placebo cohort data with the new pseudo-treated unit indices and their outcomes.
            current_placebo_cohort_data["treated_indices"] = list(current_placebo_treated_indices)
            # The outcomes 'y' for these pseudo-treated units are taken from the original donor_matrix.
            current_placebo_cohort_data["y"] = current_placebo_cohort_data["donor_matrix"][:, current_placebo_treated_indices]

        # Estimate effects for this placebo iteration.
        current_placebo_iteration_pooled_effects_accumulator: DefaultDict[float, List[Tuple[int, float]]] = defaultdict(list)
        current_placebo_iteration_cohort_atts: Dict[int, float] = {} # Stores ATTs for each cohort in this placebo iteration.

        # Estimate SDID effects for each placebo cohort.
        for current_placebo_cohort_period, current_placebo_cohort_data in current_placebo_iteration_cohorts_data.items():
            current_placebo_cohort_result = estimate_cohort_sdid_effects(
                current_placebo_cohort_period, current_placebo_cohort_data, current_placebo_iteration_pooled_effects_accumulator
            )
            current_placebo_iteration_cohort_atts[current_placebo_cohort_period] = current_placebo_cohort_result["att"]

            # Collect event-time effects for this placebo cohort.
            for effect_array_name_key in ["pre_effects", "post_effects"]:
                current_placebo_effect_array_from_result = current_placebo_cohort_result[effect_array_name_key]
                if current_placebo_effect_array_from_result.size > 0:
                    for event_time_from_effect_array, tau_value_from_effect_array in zip(current_placebo_effect_array_from_result[:,0], current_placebo_effect_array_from_result[:,1]):
                        placebo_cohort_event_effects_dict[current_placebo_cohort_period][int(event_time_from_effect_array)].append(tau_value_from_effect_array)
        
        # Calculate the overall ATT for this placebo iteration.
        total_post_exposure_placebo: int = 0
        weighted_att_sum_placebo: float = 0.0
        # Use original cohort sizes and post-periods for weighting the placebo ATTs.
        for cohort_period_key_for_att_aggregation, original_cohort_data_for_att_aggregation in original_cohorts_data_dict.items():
            placebo_att_for_cohort_period_key = current_placebo_iteration_cohort_atts.get(cohort_period_key_for_att_aggregation, np.nan)
            if not np.isnan(placebo_att_for_cohort_period_key):
                num_treated_for_cohort_period_key = len(original_cohort_data_for_att_aggregation["treated_indices"])
                num_post_periods_for_cohort_period_key = original_cohort_data_for_att_aggregation["post_periods"]
                weighted_att_sum_placebo += placebo_att_for_cohort_period_key * num_treated_for_cohort_period_key * num_post_periods_for_cohort_period_key
                total_post_exposure_placebo += num_treated_for_cohort_period_key * num_post_periods_for_cohort_period_key
        
        current_placebo_iteration_overall_att = weighted_att_sum_placebo / total_post_exposure_placebo if total_post_exposure_placebo > 0 else np.nan
        if not np.isnan(current_placebo_iteration_overall_att): # Only store non-NaN placebo ATTs.
            placebo_overall_att_list.append(current_placebo_iteration_overall_att)

        # Store cohort-specific ATTs from this placebo iteration.
        for cohort_period_key_for_att_aggregation, placebo_att_value_for_cohort_p_in_dict_update in current_placebo_iteration_cohort_atts.items():
            if not np.isnan(placebo_att_value_for_cohort_p_in_dict_update):
                placebo_cohort_att_dict[cohort_period_key_for_att_aggregation].append(placebo_att_value_for_cohort_p_in_dict_update)

        # Store pooled event-time effects from this placebo iteration.
        for event_time_from_effect_array, effects_list_for_event_time_in_dict_update in current_placebo_iteration_pooled_effects_accumulator.items():
            sum_num_treated_for_event_time_in_dict_update = sum(n_val for n_val, _ in effects_list_for_event_time_in_dict_update)
            if sum_num_treated_for_event_time_in_dict_update > 0: # Avoid division by zero.
                current_placebo_pooled_effect_for_event_time_in_dict_update = sum(n_val * tau_val for n_val, tau_val in effects_list_for_event_time_in_dict_update) / sum_num_treated_for_event_time_in_dict_update
                placebo_pooled_event_effects_dict[event_time_from_effect_array].append(current_placebo_pooled_effect_for_event_time_in_dict_update)

    # Calculate variances from the collected placebo effects.
    # Variance of the overall ATT. ddof=1 for sample variance.
    final_overall_att_variance = np.var(placebo_overall_att_list, ddof=1) if len(placebo_overall_att_list) > 1 else np.nan
    # Variances of cohort-specific ATTs.
    final_cohort_att_variances: Dict[int, float] = {
        cohort_period_key: np.var(effects_list_for_variance_calc, ddof=1) if len(effects_list_for_variance_calc) > 1 else np.nan
        for cohort_period_key, effects_list_for_variance_calc in placebo_cohort_att_dict.items()
    }
    # Variances of cohort-specific event-time effects.
    final_cohort_event_time_variances: Dict[int, Dict[int, float]] = {
        cohort_period_key: {
            event_time_value: np.var(effects_list_for_variance_calc, ddof=1) if len(effects_list_for_variance_calc) > 1 else np.nan
            for event_time_value, effects_list_for_variance_calc in event_time_data_for_cohort_variance_calc.items()
        }
        for cohort_period_key, event_time_data_for_cohort_variance_calc in placebo_cohort_event_effects_dict.items()
    }
    # Variances of pooled event-time effects.
    final_pooled_event_time_variances: Dict[float, float] = {
        event_time_value: np.var(effects_list_for_variance_calc, ddof=1) if len(effects_list_for_variance_calc) > 1 else np.nan
        for event_time_value, effects_list_for_variance_calc in placebo_pooled_event_effects_dict.items()
    }

    # Return the calculated variances and the list of placebo ATTs.
    return {
        "att_variance": final_overall_att_variance,
        "cohort_variances": final_cohort_att_variances,
        "event_variances": final_cohort_event_time_variances,
        "pooled_event_variances": final_pooled_event_time_variances,
        "placebo_att_values": placebo_overall_att_list 
    }


def fit_time_weights(
    donor_outcomes_pre_treatment: np.ndarray, mean_donor_outcomes_post_treatment: np.ndarray
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """
    Fit time weights for SDID.

    Parameters
    ----------
    donor_outcomes_pre_treatment : np.ndarray
        Donor outcomes in pre-treatment period, shape (T0, N_donors).
    mean_donor_outcomes_post_treatment : np.ndarray
        Mean outcome of each donor unit in post-treatment period, shape (N_donors,).

    Returns
    -------
    Tuple[Optional[float], Optional[np.ndarray]]
        - intercept : Optional[float]
            The estimated intercept term (beta_0 in some notations).
            Returns `None` if the optimization fails or does not converge.
        - time_weights : Optional[np.ndarray]
            The estimated time weights (lambda_t in some notations).
            Shape (num_pre_treatment_periods,). These weights sum to 1 and are non-negative.
            Returns `None` if the optimization fails or does not converge.

    Notes
    -----
    This function solves an optimization problem to find time weights and an
    intercept that best reconstruct the average post-treatment donor outcomes
    using a weighted average of pre-treatment donor outcomes.
    The objective is to minimize the sum of squared differences between
    `mean_donor_outcomes_post_treatment` and
    `intercept + time_weights @ donor_outcomes_pre_treatment`, subject to `sum(time_weights) = 1`
    and `time_weights >= 0`.

    Examples
    --------
    >>> T0_ex, N_donors_ex = 5, 3
    >>> Y0_pre_donors_ex = np.random.rand(T0_ex, N_donors_ex)
    >>> Y0_post_donors_mean_ex = np.random.rand(N_donors_ex)
    >>> intercept_val, time_w_val = fit_time_weights(Y0_pre_donors_ex, Y0_post_donors_mean_ex)
    >>> if time_w_val is not None:
    ...     print(f"Time weights shape: {time_w_val.shape}")
    ...     print(f"Sum of time weights: {np.sum(time_w_val):.2f}")
    Time weights shape: (5,)
    Sum of time weights: 1.00
    """
    # Input Validation
    # Ensure donor_outcomes_pre_treatment is a 2D NumPy array.
    if not isinstance(donor_outcomes_pre_treatment, np.ndarray):
        raise MlsynthDataError("donor_outcomes_pre_treatment must be a NumPy array.")
    if donor_outcomes_pre_treatment.ndim != 2:
        raise MlsynthDataError("donor_outcomes_pre_treatment must be a 2D array (T0, N_donors).")
    # Ensure mean_donor_outcomes_post_treatment is a 1D NumPy array.
    if not isinstance(mean_donor_outcomes_post_treatment, np.ndarray):
        raise MlsynthDataError("mean_donor_outcomes_post_treatment must be a NumPy array.")
    if mean_donor_outcomes_post_treatment.ndim != 1:
        raise MlsynthDataError("mean_donor_outcomes_post_treatment must be a 1D array (N_donors,).")
    
    # Get dimensions: number of pre-treatment periods and number of donors.
    num_pre_treatment_periods, num_donors_pre = donor_outcomes_pre_treatment.shape
    num_donors_post = mean_donor_outcomes_post_treatment.shape[0]

    # Validate dimensions.
    if num_pre_treatment_periods == 0: # Must have pre-treatment periods to fit weights.
        raise MlsynthDataError("donor_outcomes_pre_treatment cannot have zero pre-treatment periods (num_pre_treatment_periods must be > 0).")
    if num_donors_pre == 0: # Must have donors.
        # This case might be implicitly handled by CVXPY if num_donors_post is also 0,
        # but explicit check is better. If num_donors_post > 0, it's a mismatch.
        raise MlsynthDataError("donor_outcomes_pre_treatment cannot have zero donors if mean_donor_outcomes_post_treatment has donors.")

    # Number of donors must be consistent between pre-treatment and post-treatment data.
    if num_donors_pre != num_donors_post:
        raise MlsynthDataError(
            f"Shape mismatch: donor_outcomes_pre_treatment has {num_donors_pre} donors, "
            f"but mean_donor_outcomes_post_treatment has {num_donors_post} donors."
        )

    # Define CVXPY optimization variables.
    intercept_variable = cp.Variable() # Intercept term (beta_0).
    time_weights_variable = cp.Variable(num_pre_treatment_periods, nonneg=True) # Time weights (lambda_t), constrained to be non-negative.
    
    # Define the prediction model: intercept + time_weights @ donor_outcomes_pre_treatment.
    # This reconstructs the mean post-treatment donor outcomes using weighted pre-treatment donor outcomes.
    prediction = intercept_variable + (time_weights_variable @ donor_outcomes_pre_treatment)
    # Define constraints: time weights must sum to 1.
    constraints = [cp.sum(time_weights_variable) == 1]
    # Define the objective: minimize sum of squared differences between actual and predicted mean post-treatment donor outcomes.
    objective = cp.Minimize(cp.sum_squares(prediction - mean_donor_outcomes_post_treatment))
    # Create and solve the CVXPY problem.
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.CLARABEL) # Using CLARABEL solver.
    except cp.error.SolverError as e: # Catch solver errors.
        raise MlsynthEstimationError(f"CVXPY solver failed in fit_time_weights: {e}") from e

    # Check problem status and return results if optimal or optimal_inaccurate.
    if problem.status in ["optimal", "optimal_inaccurate"]:
        return intercept_variable.value, time_weights_variable.value
    # If solver finishes but status is not optimal/optimal_inaccurate,
    # it implies a failure to find a good solution. Returning None, None is current behavior.
    # This indicates the optimization did not converge to a satisfactory solution.
    return None, None


def compute_regularization(
    donor_outcomes_pre_treatment: np.ndarray, num_post_treatment_periods: int
) -> float:
    """
    Compute regularization parameter zeta for unit weights.

    Parameters
    ----------
    donor_outcomes_pre_treatment : np.ndarray
        Donor outcomes in pre-treatment period, shape (T0, N_donors).
    num_post_treatment_periods : int
        Number of post-treatment periods (used as a proxy for N_tr_post in original papers).

    Returns
    -------
    float
        The calculated regularization parameter zeta. If `donor_outcomes_pre_treatment` has
        fewer than 2 time periods, a fallback value (currently 1.0, though this
        might indicate insufficient data for robust estimation) is used for
        `std_dev_of_first_differenced_donor_outcomes`, which then influences zeta.

    Notes
    -----
    The regularization parameter `zeta` is calculated as:
    `zeta = (num_post_treatment_periods ** 0.25) * std_dev_of_first_differenced_donor_outcomes`
    where `std_dev_of_first_differenced_donor_outcomes` is the standard deviation of the first-differenced
    outcomes of donor units in the pre-treatment period. This aims to adapt
    the regularization strength based on the variability of donor outcomes and
    the length of the post-treatment period.

    Examples
    --------
    >>> T0_ex, N_donors_ex = 10, 5
    >>> Y0_pre_donors_ex = np.random.rand(T0_ex, N_donors_ex) * 100
    >>> T_post_ex = 5
    >>> zeta = compute_regularization(Y0_pre_donors_ex, T_post_ex)
    >>> print(f"Zeta: {zeta:.2f}")
    Zeta: ...

    >>> # Example with insufficient pre-treatment periods for diff
    >>> Y0_short_pre_donors_ex = np.random.rand(1, N_donors_ex)
    >>> zeta_short = compute_regularization(Y0_short_pre_donors_ex, T_post_ex)
    >>> # Based on fallback std_dev_of_first_differenced_donor_outcomes = 1.0
    >>> # Expected: (5**0.25) * 1.0 = 1.495...
    >>> print(f"Zeta for short pre-period: {zeta_short:.2f}")
    Zeta for short pre-period: 1.50
    """
    # Input Validation
    if not isinstance(donor_outcomes_pre_treatment, np.ndarray):
        raise MlsynthDataError("donor_outcomes_pre_treatment must be a NumPy array.")
    if donor_outcomes_pre_treatment.ndim != 2:
        # Allow 0 donors for flexibility, std calculation will handle it or fallback.
        # However, if shape[0] (periods) is 0, diff will fail.
        raise MlsynthDataError("donor_outcomes_pre_treatment must be a 2D array (T0, N_donors).")
    if not isinstance(num_post_treatment_periods, int) or num_post_treatment_periods < 0:
        raise MlsynthConfigError("num_post_treatment_periods must be a non-negative integer.")

    # Calculate the standard deviation of the first-differenced donor outcomes in the pre-treatment period.
    # This term captures the volatility of donor outcomes.
    if donor_outcomes_pre_treatment.shape[0] < 2 : # Need at least 2 pre-treatment periods to calculate differences.
        # Fallback value if insufficient pre-treatment periods for differencing.
        # This implies high uncertainty or reliance on the num_post_treatment_periods term.
        std_dev_of_first_differenced_donor_outcomes = 1.0 
    elif donor_outcomes_pre_treatment.shape[1] == 0: # No donors.
        std_dev_of_first_differenced_donor_outcomes = 1.0 # Fallback if no donors to calculate differences from.
    else:
        # Calculate first differences of donor outcomes along the time axis (axis=0).
        diffs = np.diff(donor_outcomes_pre_treatment, axis=0)
        if diffs.size == 0: # Can happen if T0=1 (already caught by shape[0]<2) or N_donors=0 (caught by shape[1]==0).
             std_dev_of_first_differenced_donor_outcomes = 1.0 # Fallback if differences result in an empty array.
        else:
             # Calculate standard deviation of these differences. ddof=1 for sample standard deviation.
             std_dev_of_first_differenced_donor_outcomes = np.std(diffs.flatten(), ddof=1)
             if np.isnan(std_dev_of_first_differenced_donor_outcomes): # If all diffs were NaN, leading to NaN std.
                 std_dev_of_first_differenced_donor_outcomes = 1.0 # Fallback if std dev is NaN.

    # Calculate zeta: (num_post_treatment_periods ^ 0.25) * std_dev_of_first_differenced_donor_outcomes.
    # The term (num_post_treatment_periods ^ 0.25) scales regularization by the length of the post-treatment period.
    # A longer post-treatment period might suggest a need for stronger regularization.
    regularization_parameter_zeta: float = (num_post_treatment_periods**0.25) * std_dev_of_first_differenced_donor_outcomes
    return regularization_parameter_zeta


def unit_weights(
    donor_outcomes_pre_treatment: np.ndarray,
    mean_treated_outcome_pre_treatment: np.ndarray,
    regularization_parameter_zeta: float
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """
    Fit unit (donor) weights for SDID.

    Parameters
    ----------
    donor_outcomes_pre_treatment : np.ndarray
        Donor outcomes in pre-treatment period, shape (T0, N_donors).
    mean_treated_outcome_pre_treatment : np.ndarray
        Mean outcome of treated units in pre-treatment period, shape (T0,).
    regularization_parameter_zeta : float
        Regularization parameter.

    Returns
    -------
    Tuple[Optional[float], Optional[np.ndarray]]
        - intercept : Optional[float]
            The estimated intercept term (beta_0 in some notations).
            Returns `None` if the optimization fails or does not converge.
        - unit_weights : Optional[np.ndarray]
            The estimated donor weights (omega_j in some notations).
            Shape (N_donors,). These weights sum to 1 and are non-negative.
            Returns `None` if the optimization fails or does not converge.

    Notes
    -----
    This function solves an optimization problem to find donor weights and an
    intercept that best reconstruct the pre-treatment trajectory of the
    (mean) treated unit using a weighted average of donor unit outcomes.
    The objective is to minimize the sum of squared differences between
    `mean_treated_outcome_pre_treatment` and `intercept + donor_outcomes_pre_treatment @ unit_weights`,
    plus an L2 penalty on the `unit_weights` scaled by `regularization_parameter_zeta`.
    Constraints are `sum(unit_weights) = 1` and `unit_weights >= 0`.

    Examples
    --------
    >>> T0_ex, N_donors_ex = 10, 5
    >>> Y0_pre_donors_ex = np.random.rand(T0_ex, N_donors_ex)
    >>> y_pre_mean_treated_ex = np.random.rand(T0_ex)
    >>> zeta_ex = 0.1
    >>> intercept_val, unit_w_val = unit_weights(
    ...     Y0_pre_donors_ex, y_pre_mean_treated_ex, zeta_ex
    ... )
    >>> if unit_w_val is not None:
    ...     print(f"Unit weights shape: {unit_w_val.shape}")
    ...     print(f"Sum of unit weights: {np.sum(unit_w_val):.2f}")
    Unit weights shape: (5,)
    Sum of unit weights: 1.00
    """
    # Input Validation
    if not isinstance(donor_outcomes_pre_treatment, np.ndarray):
        raise MlsynthDataError("donor_outcomes_pre_treatment must be a NumPy array.")
    if donor_outcomes_pre_treatment.ndim != 2: # Must be 2D: (Time, Donors)
        raise MlsynthDataError("donor_outcomes_pre_treatment must be a 2D array (T0, N_donors).")
    if not isinstance(mean_treated_outcome_pre_treatment, np.ndarray):
        raise MlsynthDataError("mean_treated_outcome_pre_treatment must be a NumPy array.")
    if mean_treated_outcome_pre_treatment.ndim != 1: # Must be 1D: (Time,)
        raise MlsynthDataError("mean_treated_outcome_pre_treatment must be a 1D array (T0,).")
    if not isinstance(regularization_parameter_zeta, (float, int)) or regularization_parameter_zeta < 0:
        raise MlsynthConfigError("regularization_parameter_zeta must be a non-negative float or int.")

    # Get dimensions: number of pre-treatment periods and number of donors.
    num_pre_treatment_periods, num_donors = donor_outcomes_pre_treatment.shape
    
    # Validate dimensions.
    if num_pre_treatment_periods == 0: # Must have pre-treatment periods.
        raise MlsynthDataError("donor_outcomes_pre_treatment cannot have zero pre-treatment periods (num_pre_treatment_periods must be > 0).")
    if num_donors == 0: # Must have donors.
        raise MlsynthDataError("donor_outcomes_pre_treatment cannot have zero donors (num_donors must be > 0).")
    # Number of pre-treatment periods must match between donor outcomes and mean treated outcome.
    if mean_treated_outcome_pre_treatment.shape[0] != num_pre_treatment_periods:
        raise MlsynthDataError(
            f"Shape mismatch: donor_outcomes_pre_treatment has {num_pre_treatment_periods} pre-periods, "
            f"but mean_treated_outcome_pre_treatment has {mean_treated_outcome_pre_treatment.shape[0]}."
        )

    # Define CVXPY optimization variables.
    intercept_variable = cp.Variable() # Intercept term (beta_0).
    unit_weights_variable = cp.Variable(num_donors, nonneg=True) # Unit (donor) weights (omega_j), constrained to be non-negative.
    
    # Define the prediction model: intercept + donor_outcomes_pre_treatment @ unit_weights.
    # This reconstructs the mean pre-treatment treated outcome using weighted donor outcomes.
    prediction = intercept_variable + donor_outcomes_pre_treatment @ unit_weights_variable
    
    # Define the L2 regularization penalty on unit weights.
    # Penalty = T0 * zeta^2 * sum_squares(omega).
    # Ensure penalty term is well-defined even if num_pre_treatment_periods is 0 (though caught above by validation).
    penalty_coefficient = num_pre_treatment_periods * (float(regularization_parameter_zeta)**2)
    penalty = penalty_coefficient * cp.sum_squares(unit_weights_variable)
    
    # Define the objective: minimize sum of squared differences between actual and predicted mean treated outcome, plus the L2 penalty.
    objective = cp.Minimize(cp.sum_squares(prediction - mean_treated_outcome_pre_treatment) + penalty)
    # Define constraints: unit weights must sum to 1.
    constraints = [cp.sum(unit_weights_variable) == 1]
    # Create and solve the CVXPY problem.
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.CLARABEL) # Using CLARABEL solver.
    except cp.error.SolverError as e: # Catch solver errors.
        raise MlsynthEstimationError(f"CVXPY solver failed in unit_weights: {e}") from e

    # Check problem status and return results if optimal or optimal_inaccurate.
    if problem.status in ["optimal", "optimal_inaccurate"]:
        return intercept_variable.value, unit_weights_variable.value
    # If solver finishes but status is not optimal/optimal_inaccurate,
    # it implies a failure to find a good solution. Returning None, None indicates this.
    return None, None
