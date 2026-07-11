"""Event-study SDID aggregation.

Implements the pooled and aggregate event-study estimators from Ciccia
(2024, arXiv:2407.09565). Given per-cohort effects from :mod:`cohort`,
this module aggregates them into:

* the pooled event-time effects ``tau_ell^sdid`` (Equation 6, paper),
  with weights proportional to the per-cohort treated-unit counts at
  each event-time horizon;
* the overall ATT (Equation 7) as a treated-unit-weighted average of
  the pooled event-study effects;
* placebo-based standard errors and confidence intervals for both.

Function body of ``estimate_event_study_sdid`` is verbatim from the
previous ``sdidutils`` location.
"""

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Tuple

import numpy as np
from scipy import stats

from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)

from .cohort import estimate_cohort_sdid_effects
from .inference import (
    estimate_bootstrap_variance,
    estimate_jackknife_variance,
    estimate_placebo_variance,
)


def _empty_variance_result() -> Dict[str, Any]:
    """A variance result that carries no inference (``vce='noinference'``)."""

    return {
        "att_variance": np.nan,
        "cohort_variances": {},
        "event_variances": {},
        "pooled_event_variances": {},
        "placebo_att_values": [],
    }


def _dispatch_variance(
    prepped_event_study_data: Dict[str, Any],
    vce: str,
    resample_iterations: int,
    seed: int,
) -> Dict[str, Any]:
    """Select the ATT variance estimator by ``vce``.

    'placebo' (default) and 'bootstrap' use ``resample_iterations`` resamples;
    'jackknife' is deterministic; 'noinference' skips variance estimation.
    """

    if vce == "noinference" or (resample_iterations == 0 and vce in ("placebo", "bootstrap")):
        return _empty_variance_result()
    if vce == "placebo":
        return estimate_placebo_variance(prepped_event_study_data, resample_iterations, seed)
    if vce == "jackknife":
        return estimate_jackknife_variance(prepped_event_study_data)
    if vce == "bootstrap":
        return estimate_bootstrap_variance(prepped_event_study_data, resample_iterations, seed)
    raise MlsynthConfigError(
        f"Unknown vce '{vce}'. Expected one of 'placebo', 'jackknife', "
        "'bootstrap', 'noinference'."
    )


def estimate_event_study_sdid(
    prepped_event_study_data: Dict[str, Any], placebo_iterations: int = 1000, seed: int = 1400,
    vce: str = "placebo",
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
    if vce not in ("placebo", "jackknife", "bootstrap", "noinference"):
        raise MlsynthConfigError(
            f"Unknown vce '{vce}'. Expected one of 'placebo', 'jackknife', "
            "'bootstrap', 'noinference'."
        )

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

    # Estimate variances for ATTs and event-time effects using the selected
    # variance estimator (placebo / jackknife / bootstrap / noinference).
    placebo_variances_results: Dict[str, Any] = _dispatch_variance(
        prepped_event_study_data, vce, placebo_iterations, seed
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
        "placebo_att_values": placebo_att_values_raw_list,
        "vce": vce,
    }

