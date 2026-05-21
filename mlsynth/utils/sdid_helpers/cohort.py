"""Per-cohort SDID estimator.

Implements the cohort-specific SDID effects from Arkhangelsky et al. (2021)
as re-expressed in Equations 2 and 3 of Ciccia (2024). For each cohort
with adoption period ``a`` this routine:

* fits unit weights omega and time weights lambda on the cohort's
  pre-treatment window (the heavy lifting lives in :mod:`weights`),
* computes the bias-corrected synthetic-control trajectory,
* extracts the cohort-specific event-time effects
  ``tau_{a, ell} = Y_{0, a-1+ell} - Y_{0, a-1+ell}^{SC} - bias_correction``
  (Equation 3 of Ciccia 2024),
* averages those into the cohort ATT ``tau_a^sdid`` (Equation 4),
* and pushes each event-time effect into the pooled accumulator that
  feeds the event-study aggregation in :mod:`event_study`.

Function body and signatures are verbatim from the previous
``sdidutils.estimate_cohort_sdid_effects`` so the Prop 99 numbers do not
shift across the refactor.
"""

import warnings
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np

from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)

from .weights import compute_regularization, fit_time_weights, unit_weights
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
