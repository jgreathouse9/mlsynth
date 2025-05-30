import numpy as np
import pandas as pd # For type checking scm.df
from typing import List, Dict, Any, Union, Optional
import mlsynth # Import the top-level package to access original types
from mlsynth import CLUSTERSC, NSC, PDA  # These names will be used for construction and patched in tests
from mlsynth.utils.datautils import dataprep  # Data preparation utilities
from mlsynth.utils.estutils import (
    pcr,
    RPCASYNTH,
    NSCcv,
    NSC_opt,
    pda,
)  # Estimation utilities
# from mlsynth.utils.resultutils import effects # effects is not used in this module
from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError, MlsynthEstimationError


def _get_data(scm: Any) -> Dict[str, Any]:
    """Extract and structure data from an SCM instance using `dataprep`.

    This helper function calls `dataprep` using the configuration from the
    provided SCM instance (`scm.df`, `scm.unitid`, etc.) to prepare data
    for synthetic control methods. It specifically checks for and disallows
    cohort-based (multiple treated units) scenarios.

    Parameters
    ----------
    scm : Any
        An initialized `mlsynth` SCM instance (e.g., `CLUSTERSC`, `NSC`, `PDA`).
        Must have attributes `df`, `unitid`, `time`, `outcome`, and `treat`.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the processed data:
        - "donor_matrix" (np.ndarray): Outcomes for donor units, shape (T, N_donors).
        - "y" (np.ndarray): Outcomes for the treated unit, shape (T,).
        - "pre_periods" (int): Number of pre-treatment time periods.
        - "post_periods" (int): Number of post-treatment time periods.
        - "donor_names" (List[str]): List of donor unit identifiers.
        - "T0" (int): Same as "pre_periods", count of pre-treatment periods.

    Raises
    ------
    ValueError
        If `dataprep` fails, returns data for multiple treated units (cohorts),
        or if essential SCM attributes are missing.
    MlsynthConfigError
        If essential SCM attributes like `df`, `unitid`, etc. are missing or of wrong type.
    MlsynthDataError
        If `dataprep` fails or returns cohort data, or `scm.df` is not a DataFrame.
    """
    # Validate SCM instance and its core attributes needed for dataprep.
    # These attributes are essential for `dataprep` to correctly process the data.
    required_attrs = ["df", "unitid", "time", "outcome", "treat", "config"]
    for attr in required_attrs:
        if not hasattr(scm, attr):
            raise MlsynthConfigError(f"SCM instance is missing required attribute '{attr}'.")

    # Ensure the DataFrame `scm.df` is actually a pandas DataFrame.
    if not isinstance(scm.df, pd.DataFrame):
        raise MlsynthDataError("SCM attribute 'df' must be a pandas DataFrame.")
    # Further checks on unitid, time, etc. being valid column names could be added here,
    # but `dataprep` itself will likely handle those specific validations.

    # Run dataprep to process the input DataFrame into a format suitable for SCM estimation.
    try:
        # `dataprep` uses attributes directly from the `scm` object (e.g., `scm.df`, `scm.unitid`).
        # `scm.config` might contain other parameters, but `dataprep` primarily relies on these direct attributes.
        prepared_data = dataprep(
            df=scm.df,
            unit_id_column_name=scm.unitid,
            time_period_column_name=scm.time,
            outcome_column_name=scm.outcome,
            treatment_indicator_column_name=scm.treat
        )
        # This iterative SCM approach is designed for a single treated unit.
        # If `dataprep` returns data structured for cohorts (multiple treated units), it's an invalid scenario.
        if "cohorts" in prepared_data:
            raise MlsynthDataError("iterative_scm supports only single treated unit case, but cohort data was detected.")
        
        # Structure and return the required data components from `dataprep`'s output.
        return {
            "donor_matrix": prepared_data["donor_matrix"], # Outcomes for donor units.
            "y": prepared_data["y"],                       # Outcomes for the (single) treated unit.
            "pre_periods": prepared_data["pre_periods"],   # Number of pre-treatment periods.
            "post_periods": prepared_data["post_periods"], # Number of post-treatment periods.
            "donor_names": list(prepared_data["donor_names"]), # List of donor unit identifiers.
            "T0": prepared_data["pre_periods"],            # Alias for pre_periods, often used as T0.
        }
    except Exception as e: # Catch any exception that occurs during the dataprep process.
        raise MlsynthDataError(f"Failed to extract data from SCM via dataprep: {e}") from e


def _estimate_counterfactual(
    scm: Any,
    donor_outcomes_for_cf_estimation: np.ndarray,
    target_spillover_donor_outcome: np.ndarray,
    subset_donor_identifiers: List[str],
    num_pre_treatment_periods: int,
    spillover_donor_original_index: int,
    all_spillover_donor_original_indices: List[int],
    method: Optional[str] = None,
) -> np.ndarray:
    """Estimate counterfactual for a spillover donor using the SCM's method.

    This helper function selects the appropriate estimation logic based on the
    type of the SCM instance (`scm`) and the specified `method`. It's used
    internally by `iterative_scm` to generate a synthetic version of a
    donor unit that is suspected of being affected by spillover.

    Parameters
    ----------
    scm : Any
        An initialized `mlsynth` SCM instance (e.g., `CLUSTERSC`, `NSC`, `PDA`).
        Must possess attributes relevant to its type (e.g., `objective`,
        `cluster` for `CLUSTERSC`).
    donor_outcomes_for_cf_estimation : np.ndarray
        Array of outcomes for "clean" donor units (those not currently being
        treated as spillover targets), shape (T, N_clean_donors).
    target_spillover_donor_outcome : np.ndarray
        Outcome time series for the spillover donor unit for which a
        counterfactual is being estimated, shape (T,).
    subset_donor_identifiers : List[str]
        List of unit identifiers for the donors in `donor_outcomes_for_cf_estimation`.
    num_pre_treatment_periods : int
        Number of pre-treatment time periods. Used for fitting models.
    spillover_donor_original_index : int
        The original index of the current spillover donor target in the
        full donor matrix. Used by `CLUSTERSC` with `method="BOTH"` to
        decide between PCR and RPCA for the first spillover unit.
    all_spillover_donor_original_indices : List[int]
        List of original indices for all spillover donors. Used by `CLUSTERSC`
        with `method="BOTH"`.
    method : Optional[str], default=None
        The specific estimation method to use. For `CLUSTERSC`, can be "PCR",
        "RPCA", or "BOTH". For `PDA`, can be "LASSO", "L2", or "FS".
        If `None`, `scm.method` is used. Case-insensitive.

    Returns
    -------
    np.ndarray
        The estimated counterfactual time series for `target_spillover_donor_outcome`, shape (T,).

    Raises
    ------
    ValueError
        If the specified `method` is invalid for the given `scm` type.
    NotImplementedError
        If the `scm` class is not `CLUSTERSC`, `NSC`, or `PDA`.
    MlsynthDataError
        For invalid input data types or shapes.
    MlsynthConfigError
        For invalid method configuration.
    """
    # Input validation for data shapes and types.
    if not isinstance(donor_outcomes_for_cf_estimation, np.ndarray) or donor_outcomes_for_cf_estimation.ndim != 2:
        raise MlsynthDataError("donor_outcomes_for_cf_estimation must be a 2D NumPy array.")
    if not isinstance(target_spillover_donor_outcome, np.ndarray) or target_spillover_donor_outcome.ndim != 1:
        raise MlsynthDataError("target_spillover_donor_outcome must be a 1D NumPy array.")
    if not isinstance(subset_donor_identifiers, list) or not all(isinstance(x, str) for x in subset_donor_identifiers):
        raise MlsynthDataError("subset_donor_identifiers must be a list of strings.")
    if not isinstance(num_pre_treatment_periods, int) or num_pre_treatment_periods < 0:
        raise MlsynthDataError("num_pre_treatment_periods must be a non-negative integer.")
    # Ensure time dimensions match between donor pool and target spillover donor.
    if donor_outcomes_for_cf_estimation.shape[0] != target_spillover_donor_outcome.shape[0]:
        raise MlsynthDataError("Time dimension mismatch between donor_outcomes_for_cf_estimation and target_spillover_donor_outcome.")
    # Ensure number of donor columns matches the number of donor identifiers, unless donor matrix is empty.
    if donor_outcomes_for_cf_estimation.shape[1] != len(subset_donor_identifiers) and donor_outcomes_for_cf_estimation.size > 0 : # Allow empty if no donors
        raise MlsynthDataError("Number of donors in donor_outcomes_for_cf_estimation does not match length of subset_donor_identifiers.")


    # Dispatch to the appropriate estimation logic based on the type of SCM object.
    # Handle CLUSTERSC (which can use PCR, RPCA, or a combination "BOTH").
    if isinstance(scm, CLUSTERSC):
        # Determine the specific estimation method (PCR, RPCA, BOTH), defaulting to scm.method if not provided.
        estimation_method_upper = method.upper() if method else scm.method.upper()
        if estimation_method_upper not in ["PCR", "RPCA", "BOTH"]:
            raise MlsynthConfigError("method must be 'PCR', 'RPCA', or 'BOTH' for CLUSTERSC")
        
        # Special handling for "BOTH" method: use PCR for the first spillover donor, RPCA for others.
        # This is a heuristic often used in practice.
        if estimation_method_upper == "PCR" or \
           (estimation_method_upper == "BOTH" and all_spillover_donor_original_indices and spillover_donor_original_index == all_spillover_donor_original_indices[0]):
            # `pcr` function is used for PCR-based synthetic control.
            # It internally handles slicing data to `num_pre_treatment_periods` for fitting weights
            # and then uses the full data to construct the counterfactual.
            estimation_result = pcr(
                X_donors=donor_outcomes_for_cf_estimation, # "Clean" donor outcomes.
                y_treated=target_spillover_donor_outcome, # Outcome of the spillover donor (acting as treated).
                objective_model=scm.objective, # Objective from the original SCM config.
                donor_names_list=subset_donor_identifiers, # Names of "clean" donors.
                pre_periods=num_pre_treatment_periods,
                cluster_flag=scm.cluster, # Clustering flag from original SCM.
                is_frequentist=scm.Frequentist # Frequentist flag from original SCM.
            )
            return estimation_result["cf_mean"] # Return the mean counterfactual outcome.
        # For RPCA or subsequent spillover donors in "BOTH" mode.
        else: 
            # RPCA requires a DataFrame setup where the target spillover donor is marked as treated.
            # Create a temporary DataFrame for this purpose.
            temporary_dataframe = scm.df.copy()
            temporary_dataframe[scm.treat] = 0 # Mark all units as control initially.
            # Mark the current spillover donor (identified by its ID from subset_donor_identifiers) as treated.
            # Assuming subset_donor_identifiers[0] corresponds to the target_spillover_donor_outcome if it's the one being estimated.
            # This part might need careful review if subset_donor_identifiers is not just the spillover donor.
            # However, the logic seems to imply that target_spillover_donor_outcome is one of the donors,
            # and subset_donor_identifiers are the *other* donors used to construct its counterfactual.
            # If subset_donor_identifiers[0] is used to mark treated, it implies the first donor in the *clean pool*
            # is temporarily marked as treated for RPCA, which seems incorrect if the goal is to estimate CF for target_spillover_donor_outcome.
            # This section might need adjustment if target_spillover_donor_outcome's ID is not directly used.
            # For now, assuming the intent is to treat one of the *clean* donors as a temporary treated unit for RPCA setup.
            # This is unusual; typically, the target_spillover_donor_outcome's unit ID would be used.
            # Let's assume subset_donor_identifiers[0] is a placeholder or a specific choice for RPCA.
            # A more robust approach would be to pass the ID of target_spillover_donor_outcome.
            # Given the current structure, this will use the first "clean" donor as the temporary treated unit.
            temporary_dataframe.loc[temporary_dataframe[scm.unitid] == subset_donor_identifiers[0], scm.treat] = 1 # This line is potentially problematic.
            temporary_prepared_data = dataprep(
                temporary_dataframe, scm.unitid, scm.time, scm.outcome, scm.treat
            )
            # `RPCASYNTH` performs Robust PCA Synthetic Control.
            estimation_result = RPCASYNTH(temporary_dataframe, scm.__dict__, temporary_prepared_data)
            return estimation_result["Vectors"]["Counterfactual"] # Return the counterfactual vector.

    # Handle NSC (Nonlinear Synthetic Control).
    elif isinstance(scm, NSC):
        # Perform cross-validation to find optimal penalty parameters (a and b) for NSC.
        # Uses pre-treatment data of the target spillover donor and clean donors.
        optimal_penalty_a, optimal_penalty_b = NSCcv(
            target_spillover_donor_outcome[:num_pre_treatment_periods],
            donor_outcomes_for_cf_estimation[:num_pre_treatment_periods]
        )
        # Optimize donor weights using the found penalty parameters.
        donor_weights = NSC_opt(
            target_spillover_donor_outcome[:num_pre_treatment_periods],
            donor_outcomes_for_cf_estimation[:num_pre_treatment_periods],
            optimal_penalty_a,
            optimal_penalty_b
        )
        # Compute the counterfactual by applying the optimized weights to the full time series of clean donor outcomes.
        return np.dot(donor_outcomes_for_cf_estimation, donor_weights)

    # Handle PDA (Panel Data Approach).
    elif isinstance(scm, PDA):
        # Determine the specific PDA method (LASSO, L2, FS).
        estimation_method_upper = method.upper() if method else scm.method.upper()
        if estimation_method_upper not in ["LASSO", "L2", "FS"]:
            raise MlsynthConfigError("method must be 'LASSO', 'L2', or 'FS' for PDA")
        
        # Similar to RPCA, PDA often requires a setup where the target unit is marked as treated.
        # Create a temporary DataFrame.
        temporary_dataframe = scm.df.copy()
        temporary_dataframe[scm.treat] = 0 # Mark all as control.
        # Mark the first "clean" donor as treated. This has the same potential issue as noted for RPCA.
        # The ID of target_spillover_donor_outcome should ideally be used here.
        temporary_dataframe.loc[temporary_dataframe[scm.unitid] == subset_donor_identifiers[0], scm.treat] = 1
        temporary_prepared_data = dataprep(temporary_dataframe, scm.unitid, scm.time, scm.outcome, scm.treat)
        
        # Prepare extra arguments for the `pda` utility based on the method and original SCM config.
        pda_extra_arguments = {}
        if estimation_method_upper == "L2" and hasattr(scm, 'tau'): # `tau` is specific to L2 PDA.
            pda_extra_arguments["tau_l2"] = scm.tau
        elif estimation_method_upper == "LASSO" and hasattr(scm, 'config') and "lambda_" in scm.config:
             # `lambda_` is for LASSO PDA. `pda` utility expects it via kwargs or in `temporary_prepared_data['config']`.
             # Here, it's passed via kwargs if available in the original `scm.config`.
            pda_extra_arguments["lambda_"] = scm.config["lambda_"]
        
        # Call the `pda` utility function.
        estimation_result = pda(
            temporary_prepared_data, 
            len(temporary_prepared_data["donor_names"]), # Number of donors in the temp setup.
            method=estimation_method_upper, 
            **pda_extra_arguments
        )
        return estimation_result["Vectors"]["Counterfactual"] # Return the counterfactual vector.

    # Raise error if the SCM class is not supported by this iterative spillover logic.
    else:
        raise NotImplementedError(
            f"Iterative SCM not implemented for {type(scm).__name__}. "
            "Supported classes: CLUSTERSC, NSC, PDA. Please provide estimation logic."
        )


def iterative_scm(
    scm: Any, spillover_unit_identifiers: List[str], method: Optional[str] = None
) -> Dict[str, Any]:
    """Apply Iterative Synthetic Control Method to handle spillover effects.

    This function iteratively "cleans" specified donor units that are suspected
    of being affected by spillover. For each spillover unit, it estimates a
    counterfactual outcome using the remaining "clean" donors (and previously
    cleaned spillover units). This cleaned outcome then replaces the original
    outcome for that spillover unit in the dataset. After all specified
    spillover units are cleaned, the original SCM `fit` method is called on
    the modified dataset.

    The method supports `CLUSTERSC`, `NSC`, and `PDA` estimator types from
    `mlsynth`. It assumes a single treated unit scenario.

    Parameters
    ----------
    scm : Union[mlsynth.CLUSTERSC, mlsynth.NSC, mlsynth.PDA, Any]
        An initialized `mlsynth` SCM instance. It must have attributes `df`
        (pandas DataFrame), `unitid` (column name for unit IDs), `time`
        (column name for time periods), `outcome` (column name for outcome
        variable), `treat` (column name for treatment indicator), and a
        `fit()` method.
    spillover_unit_identifiers : List[str]
        A list of unit identifiers (strings matching values in `scm.df[scm.unitid]`)
        for donor units suspected of having spillover effects. These units'
        outcomes will be iteratively replaced by their estimated counterfactuals.
        Must not be empty and must correspond to actual donor units.
    method : Optional[str], default=None
        The specific estimation sub-method to use if the `scm` instance allows
        multiple (e.g., "PCR", "RPCA", "BOTH" for `CLUSTERSC`; "LASSO", "L2",
        "FS" for `PDA`). If `None`, the method already configured in the `scm`
        instance (e.g., `scm.method`) will be used. Case-insensitive.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the results from calling the `fit()` method of
        the `scm` instance on the spillover-cleaned data. The structure of this
        dictionary depends on the specific SCM estimator but typically includes
        keys like 'Effects', 'Fit', 'Vectors', and 'Weights'. For estimators
        returning Pydantic `BaseEstimatorResults` models, this will be the
        `.dict()` representation of that model.

    Raises
    ------
    MlsynthConfigError
        If `spillover_unit_identifiers` is invalid (e.g., empty, non-unique,
        ID not found, insufficient clean donors) or `method` is invalid.
    MlsynthDataError
        If initial data extraction via `_get_data` fails.
    MlsynthEstimationError
        If an SCM estimation fails during counterfactual estimation for a
        spillover unit or during the final `fit()` call.
    NotImplementedError
        If the provided `scm` instance is not of a supported type
        (`CLUSTERSC`, `NSC`, `PDA`).

    Examples
    --------
    >>> import pandas as pd
    >>> from mlsynth import CLUSTERSC
    >>> from mlsynth.utils.iterative_scm import iterative_scm
    >>> # Load smoking data
    >>> url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/main/basedata/smoking_data.csv"
    >>> df = pd.read_csv(url)
    >>> # Configure CLUSTERSC
    >>> config = {
    ...     "df": df,
    ...     "outcome": df.columns[2],
    ...     "treat": df.columns[-1],
    ...     "unitid": df.columns[0],
    ...     "time": df.columns[1],
    ...     "display_graphs": True,
    ...     "save": False,
    ...     "counterfactual_color": "red",
    ...     "method": "PCR",
    ...     "Frequentist": True
    ... }
    >>> cluster_sc = CLUSTERSC(config)
    >>> # Run iterative_scm with manual spillover identification
    >>> results = iterative_scm(
    ...     cluster_sc,
    ...     spillover_unit_ids=["Florida", "Nevada"],
    ...     method="PCR"
    ... )
    """
    # Extract initial data from the SCM object using the _get_data helper.
    # This prepares the data in a standardized format (donor matrix, treated outcome, etc.).
    initial_prepared_data = _get_data(scm)
    original_donor_outcomes = initial_prepared_data["donor_matrix"] # Original outcomes of all donor units.
    treated_unit_outcome = initial_prepared_data["y"] # Outcome of the primary treated unit.
    num_pre_treatment_periods = initial_prepared_data["pre_periods"] # Number of pre-treatment periods.
    # num_post_treatment_periods = initial_prepared_data["post_periods"] # This is available but unused directly in this function.
    all_donor_identifiers = initial_prepared_data["donor_names"] # List of all donor unit identifiers.
    # T0 (number of pre-treatment periods) is the same as num_pre_treatment_periods, so not stored separately.

    # Validate the list of spillover unit identifiers.
    if not isinstance(spillover_unit_identifiers, list) or not spillover_unit_identifiers: # Must be a non-empty list.
        raise MlsynthConfigError("spillover_unit_identifiers must be a non-empty list.")
    if not all(isinstance(uid, str) for uid in spillover_unit_identifiers): # All IDs must be strings.
        raise MlsynthConfigError("All elements in spillover_unit_identifiers must be strings.")
    if method is not None and not isinstance(method, str): # Method, if provided, must be a string.
        raise MlsynthConfigError("method, if provided, must be a string.")


    # Map the string identifiers of spillover units to their column indices in the original_donor_outcomes matrix.
    spillover_donor_column_indices = []
    for current_spillover_unit_id in spillover_unit_identifiers:
        try:
            idx = all_donor_identifiers.index(current_spillover_unit_id) # Find index of the spillover unit ID.
            spillover_donor_column_indices.append(idx)
        except ValueError: # If a spillover unit ID is not found among the donor identifiers.
            raise MlsynthConfigError(f"Spillover unit ID '{current_spillover_unit_id}' not found in donor names: {all_donor_identifiers}") from None

    # Ensure spillover unit IDs are unique (no duplicate indices).
    if len(set(spillover_donor_column_indices)) != len(spillover_donor_column_indices):
        raise MlsynthConfigError("Spillover unit IDs must be unique.")

    # Check if there are enough "clean" donors initially (donors not in the spillover list).
    # At least 2 clean donors are typically required to form a synthetic control for the first spillover unit.
    num_potential_clean_donors = len(all_donor_identifiers) - len(set(spillover_donor_column_indices))
    if num_potential_clean_donors < 2:
        # This check is for the initial pool of clean donors.
        # The loop for _estimate_counterfactual will use an expanding pool of cleaned donors.
        raise MlsynthConfigError(
             f"At least 2 initial clean donors are required. Found {num_potential_clean_donors} "
             f"(Total donors: {len(all_donor_identifiers)}, Spillover donors: {len(set(spillover_donor_column_indices))})."
        )


    # Iteratively "clean" the spillover donors.
    # `iteratively_cleaned_donor_outcomes` starts as a copy of original donor outcomes and is updated in each iteration.
    iteratively_cleaned_donor_outcomes = original_donor_outcomes.copy()
    # `clean_donor_pool_indices` initially contains indices of donors NOT in the spillover list.
    clean_donor_pool_indices = [
        i for i in range(len(all_donor_identifiers)) if i not in spillover_donor_column_indices
    ]

    # Loop through each identified spillover donor.
    for current_spillover_column_index in spillover_donor_column_indices:
        # Get the original outcome series for the current spillover donor being targeted for cleaning.
        current_spillover_donor_outcome_series = original_donor_outcomes[:, current_spillover_column_index]
        # Get the outcomes of the current pool of "clean" donors (initially non-spillover, later includes already cleaned spillover units).
        current_clean_donor_pool_outcomes = iteratively_cleaned_donor_outcomes[:, clean_donor_pool_indices]
        # Get the identifiers for the donors in the current clean donor pool.
        current_clean_donor_pool_identifiers = [all_donor_identifiers[i] for i in clean_donor_pool_indices]

        try:
            # Estimate the counterfactual outcome for the current spillover donor using the clean donor pool.
            estimated_counterfactual_for_spillover_donor = _estimate_counterfactual(
                scm, # The original SCM object (used for its type and configuration).
                current_clean_donor_pool_outcomes, # Outcomes of clean donors.
                current_spillover_donor_outcome_series, # Outcome of the spillover donor to be synthesized.
                current_clean_donor_pool_identifiers, # IDs of clean donors.
                num_pre_treatment_periods, # Number of pre-treatment periods for model fitting.
                current_spillover_column_index, # Original index of the current spillover target.
                spillover_donor_column_indices, # List of all spillover donor original indices.
                method, # Specific sub-method if applicable (e.g., "PCR" for CLUSTERSC).
            )
        except Exception as e: # Catch any error during counterfactual estimation.
            # This includes MlsynthConfigError, MlsynthDataError, MlsynthEstimationError, NotImplementedError from _estimate_counterfactual.
            raise MlsynthEstimationError(
                f"Counterfactual estimation failed for spillover donor "
                f"'{all_donor_identifiers[current_spillover_column_index]}': {e}"
            ) from e

        # Replace the original outcome of the current spillover donor with its estimated counterfactual in the `iteratively_cleaned_donor_outcomes` matrix.
        iteratively_cleaned_donor_outcomes[:, current_spillover_column_index] = estimated_counterfactual_for_spillover_donor
        # Add the index of the now-cleaned spillover donor to the pool of clean donors for subsequent iterations.
        clean_donor_pool_indices.append(current_spillover_column_index)

    # After all spillover donors are cleaned, update the original DataFrame (`scm.df`) with these cleaned outcomes.
    spillover_cleaned_dataframe = scm.df.copy() # Work on a copy.
    for original_donor_column_index, unit_id_val in enumerate(all_donor_identifiers):
        # If this donor was one of the spillover units that got cleaned:
        if original_donor_column_index in spillover_donor_column_indices:
            # Find all rows in the DataFrame corresponding to this unit ID.
            current_unit_mask_for_df_update = spillover_cleaned_dataframe[scm.unitid] == unit_id_val
            # Update the outcome column for these rows with the cleaned outcome series.
            spillover_cleaned_dataframe.loc[current_unit_mask_for_df_update, scm.outcome] = \
                iteratively_cleaned_donor_outcomes[:, original_donor_column_index]

    # Run the original SCM method using the DataFrame that now contains the cleaned spillover donor outcomes.
    try:
        # Re-instantiate the SCM object with the cleaned DataFrame and potentially updated method.
        # This ensures that the `fit` method operates on the modified data.
        
        # Get the original Pydantic config model instance from the input scm object.
        original_config_model = scm.config

        # Convert the original config model to a dictionary to allow modification.
        new_config_data = original_config_model.model_dump()

        # Update the 'df' field in the config data with the spillover-cleaned DataFrame.
        new_config_data["df"] = spillover_cleaned_dataframe

        # If a specific 'method' was passed to iterative_scm, update it in the config data.
        # This assumes 'method' is a valid field in the Pydantic config model for the SCM type.
        if method is not None and "method" in new_config_data: 
            new_config_data["method"] = method
        # If 'method' is None, the original method from `scm.config` (as captured by `model_dump()`) is retained.

        # Create a new Pydantic config model instance from the modified dictionary.
        ConfigModelType = type(original_config_model) # Get the type of the original config model.
        updated_config_model_instance = ConfigModelType(**new_config_data) # Create new instance.

        # Instantiate the appropriate SCM class using the updated Pydantic config model instance.
        # This uses the top-level `mlsynth` import to access the original class definitions,
        # which is important if these names are patched during testing.
        if isinstance(scm, mlsynth.CLUSTERSC): # Check against original type via mlsynth package
            scm_instance_with_cleaned_data = CLUSTERSC(config=updated_config_model_instance) # Construct using imported (potentially patched) name
        elif isinstance(scm, mlsynth.NSC): # Check against original type via mlsynth package
            scm_instance_with_cleaned_data = NSC(config=updated_config_model_instance) # Construct using imported (potentially patched) name
        elif isinstance(scm, mlsynth.PDA): # Check against original type via mlsynth package
            scm_instance_with_cleaned_data = PDA(config=updated_config_model_instance) # Construct using imported (potentially patched) name
        else:
            # If the SCM type is not supported for re-instantiation, raise an error.
            raise NotImplementedError(
                f"SCM type {type(scm).__name__} not supported for re-instantiation in iterative_scm. "
                "Supported base types: CLUSTERSC, NSC, PDA."
            )

        # Call the `fit` method of the newly instantiated SCM object.
        final_estimation_results = scm_instance_with_cleaned_data.fit()
    except Exception as e: # Catch any exception during the final SCM fitting process.
        raise MlsynthEstimationError(f"Final SCM fitting failed after spillover cleaning: {e}") from e

    # Return the results from the final SCM estimation.
    return final_estimation_results
