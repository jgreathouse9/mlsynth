import numpy as np
import pandas as pd
# Add `Literal` to the existing typing import:
from typing import Optional, Dict, List, Tuple, Any, Literal
from mlsynth.exceptions import MlsynthDataError

# Constants for dictionary keys used in logictreat and dataprep
KEY_NUM_TREATED_UNITS = "Num Treated Units"
KEY_POST_PERIODS = "Post Periods"
KEY_TREATED_INDEX = "Treated Index"
KEY_PRE_PERIODS = "Pre Periods"
KEY_TOTAL_PERIODS = "Total Periods"
KEY_FIRST_TREAT_PERIODS = "First Treat Periods"
KEY_PRE_PERIODS_BY_UNIT = "Pre Periods by Unit"
KEY_POST_PERIODS_BY_UNIT = "Post Periods by Unit"

KEY_TREATED_UNIT_NAME = "treated_unit_name"
KEY_YWIDE = "Ywide"
KEY_Y = "y"
KEY_DONOR_NAMES = "donor_names"
KEY_DONOR_MATRIX = "donor_matrix"
KEY_COHORTS = "cohorts"
KEY_COHORT_TREATED_UNITS = "treated_units"

# Constants for proxy_dataprep input map keys
KEY_DONOR_PROXIES = "donorproxies"
KEY_SURROGATE_VARS = "surrogatevars"


def logictreat(treatment_matrix: np.ndarray) -> Dict[str, Any]:
    """Analyze a treatment matrix to determine treatment timings and unit counts.

    Identifies the pre-treatment and post-treatment periods for single or multiple
    treated units. For multiple treated units, it determines the first treatment
    period for each.

    Parameters
    ----------
    treatment_matrix : np.ndarray
        A 2D NumPy array where rows represent time periods and columns
        represent units. A value of 1 indicates treatment, 0 otherwise.
        Shape (n_periods, n_units).

    Returns
    -------
    Dict[str, Any]
        A dictionary containing treatment analysis results. The keys and
        their meanings depend on whether a single or multiple treated units
        are detected:

        If a single treated unit is found:
            "Num Treated Units" : int
                Always 1.
            "Post Periods" : int
                Number of post-treatment periods for the treated unit.
            "Treated Index" : np.ndarray
                1D array containing the column index of the treated unit.
                Shape (1,).
            "Pre Periods" : int
                Number of pre-treatment periods for the treated unit.
            "Total Periods" : int
                Total number of time periods for the treated unit.

        If multiple treated units are found:
            "Num Treated Units" : int
                Number of unique treated units detected.
            "Treated Index" : np.ndarray
                1D array containing the column indices of all treated units.
                Shape (num_treated_units,).
            "First Treat Periods" : np.ndarray
                1D array, where each element is the first treatment period
                (0-indexed) for the corresponding treated unit in "Treated Index".
                Shape (num_treated_units,).
            "Pre Periods by Unit" : np.ndarray
                1D array, number of pre-treatment periods for each treated unit.
                Shape (num_treated_units,).
            "Post Periods by Unit" : np.ndarray
                1D array, number of post-treatment periods for each treated unit.
                Shape (num_treated_units,).
            "Total Periods" : int
                Total number of time periods in the input `treatment_matrix`.

    Raises
    ------
    MlsynthDataError
        If `treatment_matrix` is not a NumPy array.
        If no treated units are found (zero treated observations).
        If a treated unit has no post-treatment period.
        If treatment is not sustained for a treated unit (i.e., a 0 appears
        after a 1 in its treatment vector).
    """
    if not isinstance(treatment_matrix, np.ndarray):
        raise MlsynthDataError("treatment_matrix must be a NumPy array")

    # Validate that treatment_matrix contains only binary values (0, 1), ignoring NaNs.
    # NaNs are excluded from this validation as they might represent missing data,
    # not necessarily invalid treatment states.
    unique_values = np.unique(treatment_matrix)
    valid_treatment_values = unique_values[~np.isnan(unique_values)] # Exclude NaNs from validation
    if not np.all(np.isin(valid_treatment_values, [0, 1])):
        raise MlsynthDataError("Treatment indicator must be a binary variable (0 or 1).")

    num_treated_observations = np.count_nonzero(treatment_matrix == 1) # Count only 1s for "treated"
    if not num_treated_observations > 0:
        # Ensure there's at least one instance of treatment (value of 1).
        raise MlsynthDataError("No treated units found (zero treated observations with value 1)")

    num_units = treatment_matrix.shape[1]
    num_periods = treatment_matrix.shape[0]

    # Identify treated units: a unit is considered treated if it has at least one '1' in its treatment vector.
    treated_unit_mask = np.any(treatment_matrix == 1, axis=0) # Check along time axis for each unit
    treated_indices = np.where(treated_unit_mask)[0] # Get column indices of treated units
    num_treated_units = len(treated_indices)

    # === SINGLE TREATED UNIT CASE ===
    # This block handles the scenario where exactly one unit is identified as treated.
    if num_treated_units == 1:
        treated_unit_treatment_vector = treatment_matrix[:, treated_indices[0]]
        # Find all periods where this single treated unit has a treatment indicator of 1.
        treatment_period_indices = np.where(treated_unit_treatment_vector == 1)[0]
        if not len(treatment_period_indices) > 0:
            # This should ideally be caught by the earlier `num_treated_observations` check,
            # but it's a safeguard for the single unit logic.
            raise MlsynthDataError("Treated unit has no post-treatment period")
        # The first treatment period is the first time index where treatment is 1.
        first_treatment_period_index = treatment_period_indices[0]
        # Ensure treatment is sustained: once a unit is treated, it should remain treated.
        # Check if all values from the first treatment period onwards are 1.
        if not np.all(treated_unit_treatment_vector[first_treatment_period_index:] == 1):
            raise MlsynthDataError("Treatment is not sustained for the treated unit.")

        # Post-treatment periods are counted from the first treatment period to the end.
        num_post_treatment_periods = int(np.sum(treated_unit_treatment_vector[first_treatment_period_index:]))
        # Pre-treatment periods are all periods before the first treatment period.
        num_pre_treatment_periods = int(first_treatment_period_index)
        total_periods_for_unit = int(num_pre_treatment_periods + num_post_treatment_periods)

        return {
            KEY_NUM_TREATED_UNITS: 1,
            KEY_POST_PERIODS: num_post_treatment_periods,
            KEY_TREATED_INDEX: treated_indices,
            KEY_PRE_PERIODS: num_pre_treatment_periods,
            KEY_TOTAL_PERIODS: total_periods_for_unit,
        }

    # === MULTIPLE TREATED UNITS CASE ===
    # This block handles scenarios with more than one treated unit.
    # It calculates treatment timings individually for each treated unit.
    else:
        # Initialize an array to store the first treatment period for each unit.
        first_treatment_period_by_unit = np.full(num_units, fill_value=np.nan)
        for unit_idx in treated_indices: # Iterate through identified treated units
            unit_treatment_vector = treatment_matrix[:, unit_idx]
            unit_treatment_period_indices = np.where(unit_treatment_vector == 1)[0]
            if not len(unit_treatment_period_indices) > 0:
                # Safeguard: ensure each marked treated unit actually has treatment periods.
                raise MlsynthDataError(f"Unit {unit_idx} has no post-treatment period")
            # Store the first period (0-indexed) where treatment is 1 for this unit.
            first_treatment_period_by_unit[unit_idx] = unit_treatment_period_indices[0]
            # Verify sustained treatment for this unit.
            if not np.all(unit_treatment_vector[unit_treatment_period_indices[0] :] == 1):
                raise MlsynthDataError(f"Treatment is not sustained for unit {unit_idx}")

        # Extract first treatment periods specifically for the treated units.
        first_treatment_period_for_treated_units = first_treatment_period_by_unit[treated_indices].astype(int)
        # For multiple units, pre-treatment periods are up to their individual first treatment time.
        num_pre_treatment_periods_by_unit = first_treatment_period_for_treated_units
        # Post-treatment periods are from their first treatment time to the end of the panel.
        num_post_treatment_periods_by_unit = num_periods - num_pre_treatment_periods_by_unit

        return {
            KEY_NUM_TREATED_UNITS: num_treated_units,
            KEY_TREATED_INDEX: treated_indices,
            KEY_FIRST_TREAT_PERIODS: first_treatment_period_for_treated_units,
            KEY_PRE_PERIODS_BY_UNIT: num_pre_treatment_periods_by_unit,
            KEY_POST_PERIODS_BY_UNIT: num_post_treatment_periods_by_unit,
            KEY_TOTAL_PERIODS: num_periods,
        }



# ─── New private helper (place between logictreat() and dataprep()) ────────
def _build_covariate_matrix(
    df: pd.DataFrame,
    unit_id_column_name: str,
    time_period_column_name: str,
    covariates: List[str],
    pre_periods: int,
    unit_order: List[Any],
    *,
    aggregation: str = "pre_mean",
    normalize: bool = True,
) -> Tuple[np.ndarray, Tuple[str, ...], np.ndarray, np.ndarray]:
    """Per-unit covariate matrix aligned to ``unit_order``.

    Aggregates each covariate column into a single value per unit. The
    default ``"pre_mean"`` aggregation takes the mean over the first
    ``pre_periods`` rows; time-invariant covariates collapse to their
    constant value.

    With ``normalize=True`` each column is centered by its cross-unit mean
    and scaled by its cross-unit std (computed across the units in
    ``unit_order``) so all covariates are unit-free, which is the standard
    pre-step before applying SCM-style predictor weights or computing
    standardized mean differences for balance diagnostics.

    Returns
    -------
    cov_matrix : np.ndarray, shape ``(len(unit_order), len(covariates))``
        Per-unit values for each covariate, in the order given by
        ``unit_order`` and ``covariates``.
    cov_names : tuple of str
        The covariate names, in the same order as the columns of
        ``cov_matrix``.
    cov_means : np.ndarray, shape ``(len(covariates),)``
        Cross-unit means used for centering (zeros when ``normalize=False``).
    cov_scales : np.ndarray, shape ``(len(covariates),)``
        Cross-unit stds used for scaling (ones when ``normalize=False``).
    """
    rows = []
    for cov in covariates:
        if cov not in df.columns:
            raise MlsynthDataError(f"covariate {cov!r} not present in df.columns")
        pivot = df.pivot(index=time_period_column_name,
                         columns=unit_id_column_name, values=cov)
        if aggregation == "pre_mean":
            if pre_periods <= 0:
                raise MlsynthDataError(
                    "covariate_aggregation='pre_mean' requires at least one "
                    "pre-treatment period"
                )
            agg = pivot.iloc[:pre_periods, :].mean(axis=0)
        else:
            raise MlsynthDataError(
                f"Unknown covariate_aggregation: {aggregation!r}"
            )
        rows.append(agg.reindex(unit_order).to_numpy(dtype=float))

    cov_matrix = np.array(rows).T                                # (N, M)

    if normalize:
        means = cov_matrix.mean(axis=0)
        scales = cov_matrix.std(axis=0, ddof=1)
        scales = np.where(scales > 1e-12, scales, 1.0)
        cov_matrix = (cov_matrix - means) / scales
    else:
        means = np.zeros(cov_matrix.shape[1])
        scales = np.ones(cov_matrix.shape[1])

    return cov_matrix, tuple(covariates), means, scales






# ─── Updated dataprep() ────────────────────────────────────────────────────
def dataprep(
    df: pd.DataFrame,
    unit_id_column_name: str,
    time_period_column_name: str,
    outcome_column_name: str,
    treatment_indicator_column_name: str,
    allow_no_donors: bool = False,
    *,
    covariates: Optional[List[str]] = None,
    covariate_aggregation: Literal["pre_mean"] = "pre_mean",
    normalize_covariates: bool = True,
    marex: bool = False,
) -> Dict[str, Any]:
    """Prepare data for synthetic control methods.

    Pivots a long DataFrame into a wide format required by many synthetic
    control estimators. It identifies treated and donor units, and separates
    data into pre-treatment and post-treatment periods. Handles cases with
    a single treated unit or multiple treated units (by cohorting based on
    treatment start time).

    Parameters
    ----------
    df : pd.DataFrame
        The input panel data in long format. Must contain columns for unit
        identifiers, time periods, outcome variable, and treatment indicator.
    unit_id_column_name : str
        Name of the column in `df` that identifies unique units.
    time_period_column_name : str
        Name of the column in `df` that identifies time periods.
    outcome_column_name : str
        Name of the column in `df` for the outcome variable.
    treatment_indicator_column_name : str
        Name of the column in `df` for the treatment indicator (0 or 1).
    allow_no_donors : bool, default False
        If True, do not raise when zero donor units remain after dropping
        the treated unit.
    covariates : list of str, optional
        Column names to fold into a per-unit covariate matrix. When
        supplied, the returned dict gains
        ``covariate_matrix`` / ``covariate_names`` / ``covariate_means`` /
        ``covariate_scales`` plus
        ``donor_covariate_matrix`` / ``treated_covariate_matrix`` (single
        treated case). Time-varying covariates are aggregated per the
        ``covariate_aggregation`` mode; time-invariant covariates collapse
        to their constant value. ``None`` (default) leaves the return dict
        identical to the pre-extension behaviour.
    covariate_aggregation : {"pre_mean"}, default "pre_mean"
        How to collapse each covariate column to a single value per unit.
        Currently only ``"pre_mean"`` (the pre-treatment mean) is supported.
    normalize_covariates : bool, default True
        If True, z-score each covariate column across units (the standard
        pre-step for SCM-style predictor weights and SMD diagnostics).
    marex : bool, default False
        If True, additionally compute the Abadie-Zhou-style stacked
        predictor matrix ``X = [Y_pre; covariates^T]`` plus its
        population-weighted aggregate, surfacing the inputs that the
        MAREX-family experimental-design estimators consume.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the prepared data. Always populates the
        backward-compatible keys (`treated_unit_name`, `Ywide`, `y`,
        `donor_names`, `donor_matrix`, `total_periods`, `pre_periods`,
        `post_periods`, `time_labels` in the single-treated case;
        `Ywide`, `cohorts`, `time_labels` in the cohort case). When the
        opt-in kwargs above are non-default, additionally populates:

        ``covariate_matrix`` : (N, M)
        ``covariate_names`` : tuple of str
        ``covariate_means`` : (M,)
        ``covariate_scales`` : (M,)
        ``donor_covariate_matrix`` : (n_donors, M)        — single-treated only
        ``treated_covariate_matrix`` : (1, M)             — single-treated only
        ``predictor_matrix`` : (T_pre + M, N)             — when ``marex=True``
        ``predictor_block_sizes`` : (T_pre, M)            — when ``marex=True``
        ``f_weights`` : (N,)                              — when ``marex=True``
        ``X_population_mean`` : (T_pre + M,)              — when ``marex=True``

    Raises
    ------
    MlsynthDataError
        - If no donor units are found after pivoting and selecting for a single
          treated unit case.
        - If there are zero pre-treatment periods for a single treated unit case.
        - If a covariate column name is not present in ``df.columns``.
        - If ``covariate_aggregation`` is unknown.
    """
    # Pivot the treatment indicator column to a wide format (time x units).
    treatment_matrix_wide = df.pivot(
        index=time_period_column_name,
        columns=unit_id_column_name,
        values=treatment_indicator_column_name,
    )
    treatment_array_wide = treatment_matrix_wide.to_numpy()
    treatment_analysis_results = logictreat(treatment_array_wide)

    # ─── Single treated unit ───────────────────────────────────────────
    if len(treatment_analysis_results[KEY_TREATED_INDEX]) == 1:
        num_post_treatment_periods = treatment_analysis_results[KEY_POST_PERIODS]
        num_pre_treatment_periods = treatment_analysis_results[KEY_PRE_PERIODS]
        total_periods_for_unit = treatment_analysis_results[KEY_TOTAL_PERIODS]
        treated_unit_column_index = treatment_analysis_results[KEY_TREATED_INDEX]

        outcome_matrix_wide = df.pivot(
            index=time_period_column_name,
            columns=unit_id_column_name,
            values=outcome_column_name,
        )
        treated_unit_name = outcome_matrix_wide.columns[treated_unit_column_index[0]]
        treated_unit_outcome_vector = outcome_matrix_wide[treated_unit_name].to_numpy()
        donor_outcome_df_wide = outcome_matrix_wide.drop(
            outcome_matrix_wide.columns[treated_unit_column_index[0]], axis=1
        )
        donor_names = donor_outcome_df_wide.columns
        donor_outcome_array_wide = donor_outcome_df_wide.to_numpy()

        if donor_outcome_df_wide.shape[1] == 0:
            if not allow_no_donors:
                raise MlsynthDataError("No donor units found after pivoting and selecting.")

        if num_pre_treatment_periods == 0:
            raise MlsynthDataError("Not enough pre-treatment periods (0 pre-periods found).")

        out: Dict[str, Any] = {
            "treated_unit_name": treated_unit_name,
            "Ywide": outcome_matrix_wide,
            "y": treated_unit_outcome_vector,
            "donor_names": donor_names,
            "donor_matrix": donor_outcome_array_wide,
            "total_periods": total_periods_for_unit,
            "pre_periods": num_pre_treatment_periods,
            "post_periods": num_post_treatment_periods,
            "time_labels": outcome_matrix_wide.index,
        }

        if covariates:
            # Order is donors first, then the single treated unit, so that
            # ``donor_covariate_matrix`` and ``treated_covariate_matrix`` can
            # be sliced off the unified ``covariate_matrix`` cleanly.
            unit_order = list(donor_names) + [treated_unit_name]
            cov_matrix, cov_names, cov_means, cov_scales = _build_covariate_matrix(
                df, unit_id_column_name, time_period_column_name,
                covariates, num_pre_treatment_periods, unit_order,
                aggregation=covariate_aggregation,
                normalize=normalize_covariates,
            )
            n_donors = len(donor_names)
            out["covariate_matrix"] = cov_matrix                      # (N, M)
            out["covariate_names"] = cov_names                        # tuple of str
            out["covariate_means"] = cov_means                        # (M,)
            out["covariate_scales"] = cov_scales                      # (M,)
            out["donor_covariate_matrix"] = cov_matrix[:n_donors]     # (n_donors, M)
            out["treated_covariate_matrix"] = cov_matrix[n_donors:]   # (1, M)

            if marex:
                # Abadie-Zhou stacked predictor matrix (T_pre + M, N).
                Y_pre_all = (
                    outcome_matrix_wide.iloc[:num_pre_treatment_periods, :]
                    .reindex(columns=unit_order)
                    .to_numpy(dtype=float)
                )
                X_stacked = np.vstack([Y_pre_all, cov_matrix.T])
                out["predictor_matrix"] = X_stacked
                out["predictor_block_sizes"] = (num_pre_treatment_periods,
                                                len(cov_names))
                # Equal-weighted population aggregate X̄ (the design objective's
                # target). Users with a non-uniform ``f`` can recompute from
                # ``predictor_matrix`` directly.
                f = np.ones(len(unit_order)) / len(unit_order)
                out["f_weights"] = f
                out["X_population_mean"] = X_stacked @ f

        elif marex:
            # MAREX mode without covariates: still expose the f / X̄ pair
            # built from outcomes only.
            Y_pre_all = (
                outcome_matrix_wide.iloc[:num_pre_treatment_periods, :]
                .to_numpy(dtype=float)
            )
            f = np.ones(Y_pre_all.shape[1]) / Y_pre_all.shape[1]
            out["predictor_matrix"] = Y_pre_all
            out["predictor_block_sizes"] = (num_pre_treatment_periods, 0)
            out["f_weights"] = f
            out["X_population_mean"] = Y_pre_all @ f

        return out

    # ─── Multiple treated units (cohort mode) ──────────────────────────
    else:
        outcome_matrix_wide = df.pivot(
            index=time_period_column_name,
            columns=unit_id_column_name,
            values=outcome_column_name,
        )
        treatment_matrix_wide_multi = df.pivot(
            index=time_period_column_name,
            columns=unit_id_column_name,
            values=treatment_indicator_column_name,
        )

        first_treatment_time_by_unit = (treatment_matrix_wide_multi == 1).idxmax().to_dict()
        cohort_treatment_start_times_map: Dict[Any, List[Any]] = {}
        for unit_id, start_time in first_treatment_time_by_unit.items():
            if treatment_matrix_wide_multi.loc[start_time, unit_id] == 1:
                cohort_treatment_start_times_map.setdefault(start_time, []).append(unit_id)

        all_treated_units = set()
        for units in cohort_treatment_start_times_map.values():
            all_treated_units.update(units)

        cohort_details_map: Dict[Any, Dict[str, Any]] = {}
        for start_time, cohort_units in cohort_treatment_start_times_map.items():
            cohort_treated_matrix = outcome_matrix_wide[cohort_units].to_numpy()
            num_post = outcome_matrix_wide.shape[0] - outcome_matrix_wide.index.get_loc(start_time)
            num_pre = outcome_matrix_wide.index.get_loc(start_time)
            donor_units = [u for u in outcome_matrix_wide.columns
                           if u not in all_treated_units]
            cohort_donor_matrix = outcome_matrix_wide[donor_units].to_numpy()
            cohort_details_map[start_time] = {
                "treated_units": cohort_units,
                "y": cohort_treated_matrix,
                "donor_names": donor_units,
                "donor_matrix": cohort_donor_matrix,
                "total_periods": outcome_matrix_wide.shape[0],
                "pre_periods": num_pre,
                "post_periods": num_post,
            }

        out_multi: Dict[str, Any] = {
            "Ywide": outcome_matrix_wide,
            "cohorts": cohort_details_map,
            "time_labels": outcome_matrix_wide.index,
        }

        if covariates:
            # Use the earliest pre-period length across cohorts as the
            # aggregation window so the covariate aggregation is stable
            # regardless of which cohort consumes the result.
            min_pre = (min(c["pre_periods"] for c in cohort_details_map.values())
                       if cohort_details_map else 0)
            unit_order = list(outcome_matrix_wide.columns)
            cov_matrix, cov_names, cov_means, cov_scales = _build_covariate_matrix(
                df, unit_id_column_name, time_period_column_name,
                covariates, max(min_pre, 1), unit_order,
                aggregation=covariate_aggregation,
                normalize=normalize_covariates,
            )
            out_multi["covariate_matrix"] = cov_matrix
            out_multi["covariate_names"] = cov_names
            out_multi["covariate_means"] = cov_means
            out_multi["covariate_scales"] = cov_scales

            if marex:
                Y_pre_all = (
                    outcome_matrix_wide.iloc[:min_pre, :]
                    .reindex(columns=unit_order)
                    .to_numpy(dtype=float)
                )
                X_stacked = np.vstack([Y_pre_all, cov_matrix.T])
                f = np.ones(len(unit_order)) / len(unit_order)
                out_multi["predictor_matrix"] = X_stacked
                out_multi["predictor_block_sizes"] = (min_pre, len(cov_names))
                out_multi["f_weights"] = f
                out_multi["X_population_mean"] = X_stacked @ f

        elif marex:
            min_pre = (min(c["pre_periods"] for c in cohort_details_map.values())
                       if cohort_details_map else 0)
            Y_pre_all = outcome_matrix_wide.iloc[:min_pre, :].to_numpy(dtype=float)
            f = (np.ones(Y_pre_all.shape[1]) / Y_pre_all.shape[1]
                 if Y_pre_all.size else None)
            out_multi["predictor_matrix"] = Y_pre_all
            out_multi["predictor_block_sizes"] = (min_pre, 0)
            out_multi["f_weights"] = f
            out_multi["X_population_mean"] = Y_pre_all @ f if f is not None else None

        return out_multi


def balance(df: pd.DataFrame, unit_id_column_name: str, time_period_column_name: str) -> None:
    """Check if the panel is strongly balanced.

    A strongly balanced panel means every unit has an observation for every
    time period, and there are no duplicate unit-time observations.

    Parameters
    ----------
    df : pd.DataFrame
        The input panel data. Must contain columns specified by `unit_id_column_name`
        and `time_period_column_name`.
    unit_id_column_name : str
        The name of the column in `df` that identifies the units. (Formerly `unit_col`)
    time_period_column_name : str
        The name of the column in `df` that identifies the time periods. (Formerly `time_col`)

    Returns
    -------
    None
        This function does not return a value but raises an error if the
        panel is not strongly balanced or contains duplicates.

    Raises
    ------
    MlsynthDataError
        If duplicate unit-time observations are found.
        If the panel is not strongly balanced (i.e., not all units have
        observations for all time periods).
    """
    # Check for unique observations: each unit-time pair should be unique.
    if df.duplicated([unit_id_column_name, time_period_column_name]).any():
        raise MlsynthDataError(
            "Duplicate observations found. Ensure each combination of unit and time is unique."
        )

    # The following commented-out block represents a previous, more complex approach to checking balance.
    # It involved creating a wide matrix of observation counts.
    # # Group by unit and count the number of observations for each unit
    # unit_time_observation_counts = (
    #     df.groupby([unit_id_column_name, time_period_column_name], observed=False).size().unstack(fill_value=0)
    # )
    # # Check if all units have the same number of observations for all time periods present in the data
    # # A simple check for strong balance is that all cells in the unstacked count matrix should be 1
    # # (assuming no missing time periods for any unit that has at least one observation).
    # # The original logic `(unit_time_observation_counts.nunique(axis=1) == 1).all()` checks if each unit has the same number of time periods.
    # # For strong balance, we also need each unit to have *all* time periods.
    # # A more direct check:
    # if not (unit_time_observation_counts > 0).all().all() or not (unit_time_observation_counts.sum(axis=1) == unit_time_observation_counts.shape[1]).all():
    #      # This checks if all cells are >0 (meaning every unit has every time period)
    #      # and if the sum of observations per unit equals the total number of unique time periods.
    #      # However, the original nunique check is simpler if the goal is just that all units have the *same count* of periods.
    #      # For true strong balance, every unit must appear in every time period.
    #      # A robust check:
    #      num_unique_time_periods_in_data = df[time_period_column_name].nunique()
    #      observations_per_unit_check = df.groupby(unit_id_column_name)[time_period_column_name].nunique()
    #      if not (observations_per_unit_check == num_unique_time_periods_in_data).all():
    #         raise MlsynthDataError( # Changed to MlsynthDataError
    #             "The panel is not strongly balanced. Not all units have observations "
    #             "for all time periods."
    #         )

    # Simplified and more direct check for strong balance:
    # 1. Determine the total number of unique time periods present in the entire dataset.
    total_unique_time_periods = df[time_period_column_name].nunique()
    
    # 2. For each unit, count the number of unique time periods for which it has observations.
    #    `groupby(unit_id_column_name)[time_period_column_name].count()` also works if there are no duplicates,
    #    but `nunique()` is more robust if we only care about the presence of each time period per unit.
    observations_per_unit = df.groupby(unit_id_column_name)[time_period_column_name].nunique() # Changed from .count() to .nunique() for robustness
    
    # 3. Check if all units have observations for *all* unique time periods found in the dataset.
    #    If `(observations_per_unit == total_unique_time_periods).all()` is true,
    #    it means every unit has an observation for every distinct time period that exists in the data.
    if not (observations_per_unit == total_unique_time_periods).all():
        raise MlsynthDataError(
            "The panel is not strongly balanced. Not all units have observations "
            "for all unique time periods in the dataset."
            )
    # 4. Additionally, ensure all units have the same number of observations.
    #    This is somewhat redundant if the above check passes and implies a rectangular structure,
    #    but it's a good explicit check. If all units have `total_unique_time_periods` observations,
    #    then `observations_per_unit.nunique()` will be 1.
    if observations_per_unit.nunique() != 1:
        raise MlsynthDataError(
            "The panel is not strongly balanced. Units have different numbers of observations."
            )


def clean_surrogates2(
    surrogate_matrix: np.ndarray,
    donor_covariates_matrix: np.ndarray,
    treated_unit_covariates_matrix: np.ndarray,
    num_pre_treatment_periods: int,
    common_covariates_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Clean surrogate variables by orthogonalizing them against covariates.

    This function adjusts each surrogate variable in `surrogate_matrix` by removing the
    linear influence of a set of covariates. The covariates for projection
    are formed by combining `donor_covariates_matrix` (donor pool covariates) and
    `treated_unit_covariates_matrix` (treated unit covariates), potentially augmented by
    `common_covariates_matrix`. The cleaning process involves estimating a linear
    relationship between each surrogate and the combined covariates using data
    from the pre-treatment period (`num_pre_treatment_periods`), and then
    subtracting this predicted component from the surrogate across all time periods.

    Parameters
    ----------
    surrogate_matrix : np.ndarray
        Matrix of surrogate variables to be cleaned.
        Shape (n_periods, n_surrogates).
    donor_covariates_matrix : np.ndarray
        Matrix of covariates for the donor pool.
        Shape (n_periods, n_donor_covariates).
    treated_unit_covariates_matrix : np.ndarray
        Matrix of covariates for the treated unit.
        Shape (n_periods, n_treated_covariates).
        Typically, `n_donor_covariates` equals `n_treated_covariates`.
    num_pre_treatment_periods : int
        Number of pre-treatment periods. The linear projection coefficients
        (`projection_coefficients`) are estimated using data up to this period.
    common_covariates_matrix : Optional[np.ndarray], default None
        Additional common covariates to include in the projection for both
        donor and treated sides. Shape (n_periods, n_common_covariates).

    Returns
    -------
    np.ndarray
        The matrix of cleaned surrogate variables, where the influence of
        the specified covariates has been removed.
        Shape (n_periods, n_surrogates).
    """
    # Initialize a list to store the cleaned surrogate vectors.
    cleaned_surrogate_vectors_list = []
    # Iterate through each surrogate variable (column in surrogate_matrix).
    for i in range(surrogate_matrix.shape[1]):
        current_surrogate_vector = np.copy(surrogate_matrix[:, i]) # Work with a copy.
        # Augment donor and treated covariates with common covariates if provided.
        if common_covariates_matrix is not None:
            augmented_donor_covariates = np.column_stack((donor_covariates_matrix, common_covariates_matrix))
            augmented_treated_covariates = np.column_stack((treated_unit_covariates_matrix, common_covariates_matrix))
        else:
            augmented_donor_covariates = donor_covariates_matrix
            augmented_treated_covariates = treated_unit_covariates_matrix

        target_surrogate_vector_for_cleaning = current_surrogate_vector
        # Estimate projection_coefficients (beta hats) using OLS on pre-treatment data.
        # The model is: surrogate_pre = treated_covariates_pre * projection_coefficients + error
        # projection_coefficients = (treated_covariates_pre' * treated_covariates_pre)^-1 * (treated_covariates_pre' * surrogate_pre)
        # Here, it's formulated slightly differently but achieves the same via `solve`.
        # The term `augmented_donor_covariates` is used in forming the cross-product matrix,
        # which implies a specific structure or assumption about the relationship being modeled.
        # This part might warrant a more detailed mathematical explanation in external documentation
        # if the formulation is non-standard OLS for residualization.
        
        # Form (X_donor_pre' * X_treated_pre)
        pre_treatment_donor_cross_treated_covariates = augmented_donor_covariates[:num_pre_treatment_periods].T @ augmented_treated_covariates[:num_pre_treatment_periods]
        # Form (X_donor_pre' * Y_surrogate_pre)
        pre_treatment_donor_cross_target_surrogate = augmented_donor_covariates[:num_pre_treatment_periods].T @ target_surrogate_vector_for_cleaning[:num_pre_treatment_periods]

        # Solve for projection_coefficients: (X_donor_pre'X_treated_pre) * beta = (X_donor_pre'Y_surrogate_pre)
        projection_coefficients = np.linalg.solve(pre_treatment_donor_cross_treated_covariates, pre_treatment_donor_cross_target_surrogate)
        
        # Clean the surrogate by subtracting the component predicted by treated_unit_covariates.
        # Cleaned_Surrogate = Original_Surrogate - (Treated_Covariates_all_periods * projection_coefficients)
        cleaned_surrogate_vector = target_surrogate_vector_for_cleaning - augmented_treated_covariates.dot(projection_coefficients)
        cleaned_surrogate_vectors_list.append(cleaned_surrogate_vector)

    # Combine the cleaned vectors back into a matrix.
    cleaned_surrogate_matrix = np.column_stack(cleaned_surrogate_vectors_list)
    return cleaned_surrogate_matrix


def proxy_dataprep(
    df: pd.DataFrame,
    surrogate_units: List[Any],
    proxy_variable_column_names_map: Dict[str, List[str]],
    unit_id_column_name: str = "Artist",
    time_period_column_name: str = "Date",
    num_total_periods: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct surrogate and surrogate-proxy matrices from long-form data.

    This function takes a long-format DataFrame and pivots it to create
    two wide-format matrices for a specified set of surrogate units:
    1.  A surrogate matrix (often denoted as `X` in models like ProximalSC),
        derived from variables specified by `proxy_variable_column_names_map['donorproxies']`.
    2.  A surrogate-proxy matrix (often denoted `Z1`), derived from variables
        specified by `proxy_variable_column_names_map['surrogatevars']`.

    The function assumes that `proxy_variable_column_names_map['donorproxies']` and
    `proxy_variable_column_names_map['surrogatevars']` each contain the name of a single column
    in `df` to be used for constructing these matrices.

    Parameters
    ----------
    df : pd.DataFrame
        The input panel data in long format. Must contain columns for unit
        identifiers, time periods, and the variables specified in `proxy_variable_column_names_map`.
    surrogate_units : List[Any]
        A list of unit identifiers (matching values in `unit_id_column_name`) that will
        form the columns of the output matrices.
    proxy_variable_column_names_map : Dict[str, List[str]]
        A dictionary specifying the variables to use for constructing the
        matrices. It must contain two keys:

        - 'donorproxies': A list containing the name of the column in `df`
          to use for the surrogate matrix. (Formerly `proxy_vars`)
        - 'surrogatevars': A list containing the name of the column in `df`
          to use for the surrogate-proxy matrix.
    unit_id_column_name : str, default "Artist"
        Name of the column in `df` that identifies unique units. (Formerly `id_col`)
    time_period_column_name : str, default "Date"
        Name of the column in `df` that identifies time periods. This column
        will form the index of the pivoted DataFrames before conversion to NumPy arrays. (Formerly `time_col`)
    num_total_periods : Optional[int], default None
        Total number of time periods. This parameter is not explicitly used in
        the current implementation's logic but is included for potential API
        consistency or future extensions. (Formerly `T`)

    Returns
    -------
    surrogate_matrix : np.ndarray
        The constructed surrogate matrix (X).
        Shape (n_time_periods, n_surrogate_units).
    surrogate_proxy_matrix : np.ndarray
        The constructed surrogate-proxy matrix (Z1).
        Shape (n_time_periods, n_surrogate_units).

    Raises
    ------
    KeyError
        - If `proxy_variable_column_names_map` does not contain 'donorproxies' or 'surrogatevars',
          or if the lists associated with these keys are empty.
    """
    # Construct the surrogate matrix (X).
    # 1. Filter the DataFrame for rows corresponding to the specified surrogate_units.
    # 2. Pivot the table: time periods as index, surrogate units as columns,
    #    and values from the column specified by `proxy_variable_column_names_map[KEY_DONOR_PROXIES][0]`.
    #    Assumes `KEY_DONOR_PROXIES` list contains exactly one column name.
    surrogate_data_wide_df = df[df[unit_id_column_name].isin(surrogate_units)].pivot(
        index=time_period_column_name, columns=unit_id_column_name, values=proxy_variable_column_names_map[KEY_DONOR_PROXIES][0]
    )
    surrogate_matrix = surrogate_data_wide_df.to_numpy() # Convert to NumPy array.

    # Construct the surrogate-proxy matrix (Z1).
    # Similar logic as above, but uses the variable specified by `proxy_variable_column_names_map[KEY_SURROGATE_VARS][0]`.
    surrogate_proxy_data_wide_df = df[df[unit_id_column_name].isin(surrogate_units)].pivot(
        index=time_period_column_name, columns=unit_id_column_name, values=proxy_variable_column_names_map[KEY_SURROGATE_VARS][0]
    )
    surrogate_proxy_matrix = surrogate_proxy_data_wide_df.to_numpy() # Convert to NumPy array.

    return surrogate_matrix, surrogate_proxy_matrix


def build_donor_segments(ell_hat, m, T0, n):
    N = T0 - m - n + 1
    L_full = np.column_stack([ell_hat[i:i + m] for i in range(N)])
    ell_eval = ell_hat[-m:]
    L_post = np.column_stack([ell_hat[i + m:i + m + n] for i in range(N)])
    return L_full, L_post, ell_eval


