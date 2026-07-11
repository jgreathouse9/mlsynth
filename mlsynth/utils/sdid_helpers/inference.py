"""Variance estimators for SDID.

Arkhangelsky et al. (2021) propose three procedures for the variance of the
SDID ATT, extended to the staggered / event-study setting by Clarke et al.
(2023). All three are provided here and selected through the ``vce`` config
option:

* ``placebo`` (Algorithm 4) -- control units are repeatedly reassigned as
  pseudo-treated units, the full SDID pipeline is rerun, and the variance of
  the resulting effects estimates the variance of the actual estimator. This is
  the only procedure defined for a single treated unit and is the default.
* ``jackknife`` (Algorithm 3) -- the fitted unit/time weights are held fixed
  and each unit is left out in turn; the variance follows the standard
  fixed-weights jackknife. Undefined (NaN) when a cohort has a single treated
  unit, matching the ``synthdid`` R package.
* ``bootstrap`` (Algorithm 2) -- units are resampled with replacement and the
  full SDID estimate (weights re-fit) is recomputed on each resample; the
  variance is the variance of the resampled estimates. Undefined (NaN) for a
  single treated unit, matching ``synthdid``.

The jackknife and bootstrap are implemented for the block (single adoption
period) design, matching ``synthdid``'s ``vcov.R``; staggered adoption uses the
placebo procedure.
"""

import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Any, DefaultDict, Dict, List, Tuple

import numpy as np

from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)

from .cohort import estimate_cohort_sdid_effects
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
            # Drop the pseudo-treated columns from the donor pool: a control unit
            # reassigned as pseudo-treated must be re-fit on the *remaining*
            # donors, exactly as the real estimate excludes the treated unit
            # from its own donor pool (Arkhangelsky et al. 2021, Algorithm 4;
            # synthdid ``vcov.R`` ``placebo_se``). Leaving it in lets the
            # synthetic control reconstruct the pseudo-treated unit from itself,
            # collapsing the placebo effect toward zero and deflating the
            # variance. ``np.unique`` guards the with-replacement fallback so a
            # duplicated draw does not misalign the deletion.
            current_placebo_cohort_data["donor_matrix"] = np.delete(
                current_placebo_cohort_data["donor_matrix"],
                np.unique(current_placebo_treated_indices),
                axis=1,
            )

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


def _sum_normalize(weights: np.ndarray) -> np.ndarray:
    """Renormalize a weight vector to sum to one.

    Mirrors ``synthdid``'s ``sum_normalize``: a zero vector maps to uniform
    weights (used only as a benign fixed-weights fallback).
    """

    total = float(np.sum(weights))
    if total != 0.0:
        return weights / total
    return np.full(weights.shape[0], 1.0 / weights.shape[0]) if weights.size else weights


def _fixed_weight_cohort_att(
    treated_matrix: np.ndarray,
    donor_matrix: np.ndarray,
    unit_weights_vector: np.ndarray,
    time_weights_vector: np.ndarray,
    num_pre_treatment_periods: int,
) -> float:
    """SDID cohort ATT for *fixed* unit and time weights (no QP re-solve).

    Reproduces the ATT that :func:`estimate_cohort_sdid_effects` computes, but
    given the weights rather than re-fitting them -- the closed-form weighted
    DID underlying Algorithm 3 (jackknife). With the mean treated series
    ``ybar``, the synthetic series ``s = donor @ omega``, time weights
    ``lambda`` over the pre-period and the post-period average taken over event
    times ``>= 0`` (indices ``>= pre``), the estimate is

    ``att = [mean_post(ybar) - lambda . ybar_pre] - [mean_post(s) - lambda . s_pre]``,

    which is algebraically identical to
    ``mean_post( ybar - (s + bias_correction) )`` in the cohort estimator.
    """

    ybar = np.asarray(treated_matrix, dtype=float).mean(axis=1)
    synthetic = np.asarray(donor_matrix, dtype=float) @ np.asarray(unit_weights_vector, dtype=float)
    lam = np.asarray(time_weights_vector, dtype=float)
    pre = int(num_pre_treatment_periods)

    treated_bias = lam @ ybar[:pre] - lam @ synthetic[:pre]
    tau_series = ybar - (synthetic + treated_bias)
    return float(np.mean(tau_series[pre:]))


def _single_cohort(prepped_event_study_data: Dict[str, Any], method_name: str):
    """Validate and unpack the single-cohort (block-design) payload.

    The jackknife and bootstrap are implemented for the block design (one
    adoption period), matching ``synthdid``'s ``vcov.R``. Staggered adoption
    (multiple cohorts) raises so the caller can fall back to the placebo
    procedure rather than return a silently wrong number.
    """

    if not isinstance(prepped_event_study_data, dict) or "cohorts" not in prepped_event_study_data:
        raise MlsynthDataError("prepped_event_study_data must be a dict with a 'cohorts' key.")
    cohorts = prepped_event_study_data["cohorts"]
    if not isinstance(cohorts, dict):
        raise MlsynthDataError("'cohorts' in prepped_event_study_data must be a dictionary.")
    if len(cohorts) != 1:
        raise MlsynthEstimationError(
            f"The {method_name} variance is implemented for the block (single "
            "adoption period) design only; the panel has "
            f"{len(cohorts)} adoption cohorts. Use vce='placebo' for staggered "
            "adoption."
        )
    (period, cohort_data), = cohorts.items()
    return int(period), cohort_data


def _nan_variance_result(period: int) -> Dict[str, Any]:
    """A variance result whose ATT variance is undefined (NaN)."""

    return {
        "att_variance": np.nan,
        "cohort_variances": {period: np.nan},
        "event_variances": {},
        "pooled_event_variances": {},
        "placebo_att_values": [],
    }


def estimate_jackknife_variance(
    prepped_event_study_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Fixed-weights jackknife variance of the SDID ATT (Algorithm 3).

    Holds the fitted unit weights ``omega`` and time weights ``lambda`` fixed,
    leaves out each unit (treated or control) in turn, and recomputes the ATT
    from the closed-form weighted DID. When a control is dropped, ``omega`` is
    renormalized over the retained controls (``synthdid``'s ``sum_normalize``);
    when a treated unit is dropped, ``omega`` is unchanged. The variance is the
    standard jackknife form ``((N - 1) / N) * sum_i (att_(-i) - mean)^2``,
    matching ``synthdid``'s ``jackknife()``.

    Returns NaN when the cohort has a single treated unit (leaving out the sole
    treated unit is undefined) or when the fitted weights are not available.
    """

    period, cohort_data = _single_cohort(prepped_event_study_data, "jackknife")
    treated_matrix = np.asarray(cohort_data["y"], dtype=float)
    donor_matrix = np.asarray(cohort_data["donor_matrix"], dtype=float)
    num_pre = int(cohort_data["pre_periods"])
    num_treated = treated_matrix.shape[1]
    num_donors = donor_matrix.shape[1]
    num_units = num_treated + num_donors

    # A single treated unit leaves the jackknife undefined (Algorithm 3;
    # synthdid returns NA). Need at least two units overall to form a variance.
    if num_treated < 2 or num_units < 2:
        return _nan_variance_result(period)

    fitted = estimate_cohort_sdid_effects(period, cohort_data, defaultdict(list))
    omega = np.asarray(fitted["unit_weights"], dtype=float)
    lam = np.asarray(fitted["time_weights"], dtype=float)
    if omega.size == 0 or lam.size == 0 or np.any(np.isnan(omega)) or np.any(np.isnan(lam)):
        return _nan_variance_result(period)  # pragma: no cover - weights only NaN on a degenerate cohort the guards above already exclude

    leave_one_out: List[float] = []
    # Leave out each control unit: drop its donor column and renormalize omega.
    for donor_index in range(num_donors):
        reduced_donors = np.delete(donor_matrix, donor_index, axis=1)
        reduced_omega = _sum_normalize(np.delete(omega, donor_index))
        leave_one_out.append(
            _fixed_weight_cohort_att(treated_matrix, reduced_donors, reduced_omega, lam, num_pre)
        )
    # Leave out each treated unit: drop its column; omega is unchanged.
    for treated_index in range(num_treated):
        reduced_treated = np.delete(treated_matrix, treated_index, axis=1)
        leave_one_out.append(
            _fixed_weight_cohort_att(reduced_treated, donor_matrix, omega, lam, num_pre)
        )

    loo = np.asarray(leave_one_out, dtype=float)
    if np.any(np.isnan(loo)):
        return _nan_variance_result(period)  # pragma: no cover - fixed-weight ATT is finite whenever the fitted weights are
    # synthdid centers on the mean of the leave-one-out estimates, not att_hat.
    variance = ((num_units - 1) / num_units) * float(np.sum((loo - loo.mean()) ** 2))
    return {
        "att_variance": variance,
        "cohort_variances": {period: variance},
        "event_variances": {},
        "pooled_event_variances": {},
        "placebo_att_values": [],
    }


def estimate_bootstrap_variance(
    prepped_event_study_data: Dict[str, Any],
    num_bootstrap_iterations: int,
    seed: int,
) -> Dict[str, Any]:
    """Clustered (block) bootstrap variance of the SDID ATT (Algorithm 2).

    Resamples the ``N`` units with replacement, discards a resample that is all
    treated or all control, and recomputes the full SDID estimate -- weights
    re-fit -- on each resample. The variance is the population variance of the
    resampled ATTs (matching ``synthdid``'s ``sqrt((B-1)/B) * sd(...)``).

    Returns NaN when the cohort has a single treated unit, matching
    ``synthdid``, whose bootstrap is undefined unless more than one unit is
    treated.
    """

    if not isinstance(num_bootstrap_iterations, int) or num_bootstrap_iterations < 0:
        raise MlsynthConfigError("num_bootstrap_iterations must be a non-negative integer.")
    if not isinstance(seed, int):
        raise MlsynthConfigError("seed must be an integer.")

    period, cohort_data = _single_cohort(prepped_event_study_data, "bootstrap")
    treated_matrix = np.asarray(cohort_data["y"], dtype=float)
    donor_matrix = np.asarray(cohort_data["donor_matrix"], dtype=float)
    num_treated = treated_matrix.shape[1]
    num_donors = donor_matrix.shape[1]
    num_units = num_treated + num_donors

    # synthdid returns NA unless more than one unit is treated.
    if num_treated < 2 or num_bootstrap_iterations < 2:
        return _nan_variance_result(period)

    # Pool all units as columns: controls first, treated last, with a label.
    pooled_columns = np.concatenate([donor_matrix, treated_matrix], axis=1)
    is_treated = np.concatenate([np.zeros(num_donors, dtype=bool), np.ones(num_treated, dtype=bool)])

    np.random.seed(seed)
    bootstrap_atts: List[float] = []
    attempts = 0
    max_attempts = max(num_bootstrap_iterations * 100, 1000)
    while len(bootstrap_atts) < num_bootstrap_iterations and attempts < max_attempts:
        attempts += 1
        sampled = np.random.choice(num_units, size=num_units, replace=True)
        treated_mask = is_treated[sampled]
        num_treated_boot = int(treated_mask.sum())
        num_donors_boot = num_units - num_treated_boot
        # Discard degenerate resamples (no treated or no control units).
        if num_treated_boot == 0 or num_donors_boot == 0:
            continue
        boot_cohort = dict(cohort_data)
        boot_cohort["donor_matrix"] = pooled_columns[:, sampled[~treated_mask]]
        boot_cohort["y"] = pooled_columns[:, sampled[treated_mask]]
        boot_cohort["treated_indices"] = list(range(num_treated_boot))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimate_cohort_sdid_effects(period, boot_cohort, defaultdict(list))
        att = result["att"]
        if not np.isnan(att):
            bootstrap_atts.append(float(att))

    if len(bootstrap_atts) < 2:  # pragma: no cover - only when nearly every resample is degenerate, impossible with >1 treated and >1 control
        return _nan_variance_result(period)
    # Population variance, matching synthdid's sqrt((B-1)/B) * sd(.).
    variance = float(np.var(bootstrap_atts, ddof=0))
    return {
        "att_variance": variance,
        "cohort_variances": {period: variance},
        "event_variances": {},
        "pooled_event_variances": {},
        "placebo_att_values": [],
    }
