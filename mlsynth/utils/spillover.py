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

    (Docstring unchanged except logic now forbids method='BOTH' here.)
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
    if donor_outcomes_for_cf_estimation.shape[0] != target_spillover_donor_outcome.shape[0]:
        raise MlsynthDataError("Time dimension mismatch between donor_outcomes_for_cf_estimation and target_spillover_donor_outcome.")
    if donor_outcomes_for_cf_estimation.shape[1] != len(subset_donor_identifiers) and donor_outcomes_for_cf_estimation.size > 0:
        raise MlsynthDataError("Number of donors in donor_outcomes_for_cf_estimation does not match length of subset_donor_identifiers.")

    if isinstance(scm, CLUSTERSC):
        estimation_method_upper = method.upper() if method else scm.method.upper()
        if estimation_method_upper not in ["PCR", "RPCA"]:
            raise MlsynthConfigError(
                "`_estimate_counterfactual` only supports 'PCR' or 'RPCA'. "
                "To use both, pass `method='BOTH'` to `iterative_scm`, not to `_estimate_counterfactual`."
            )

        if estimation_method_upper == "PCR":
            estimation_result = pcr(
                donor_outcomes_matrix=donor_outcomes_for_cf_estimation,
                treated_unit_outcome_vector=target_spillover_donor_outcome,
                scm_objective_model_type=scm.objective,
                all_donor_names=subset_donor_identifiers,
                num_pre_treatment_periods=num_pre_treatment_periods,
                enable_clustering=scm.cluster,
                use_frequentist_scm=scm.Frequentist
            )
            return estimation_result["cf_mean"]

        else:  # RPCA path
            temporary_dataframe = scm.df.copy()

            tempprepped = dataprep(
                temporary_dataframe, scm.unitid, scm.time, scm.outcome, scm.treat
            )

            donor_names = tempprepped["donor_names"]
            spillunit = donor_names[spillover_donor_original_index]
            post_T = tempprepped["post_periods"]

            unit_mask = temporary_dataframe[scm.unitid] == spillunit
            post_rows = temporary_dataframe[unit_mask].tail(post_T).index
            temporary_dataframe.loc[post_rows, scm.treat] = True

            temporary_prepared_data = dataprep(
                temporary_dataframe, scm.unitid, scm.time, scm.outcome, scm.treat
            )

            estimation_result = RPCASYNTH(temporary_dataframe, scm.__dict__, tempprepped)
            return estimation_result["Vectors"]["Counterfactual"]

    else:
        raise NotImplementedError(
            f"Iterative SCM not implemented for {type(scm).__name__}. "
            "Supported classes: CLUSTERSC."
        )



def iterative_scm(
    scm: Any, spillover_unit_identifiers: List[str], method: Optional[str] = None
) -> Dict[str, Any]:

    """
    Estimate treatment effects using Iterative Synthetic Control while accounting for spillover contamination.

    This method modifies the standard SCM framework to address the problem of contamination in the donor pool
    caused by potential spillover effects. It iteratively replaces the observed outcomes of specified
    "spillover" donor units with synthetic counterfactuals constructed using only "clean" donors,
    and then re-fits the SCM model on the cleaned data.

    Parameters
    ----------
    scm : Any
        An initialized SCM instance. The instance must already be configured
        with its outcome, treatment, unit, and time variables, as well as the data `df`.

    spillover_unit_identifiers : List[str]
        A list of donor unit identifiers (strings) that are believed to be affected by spillover from
        the treatment. These units will be treated as pseudo-treated units, and counterfactuals for
        their outcomes will be estimated and substituted into the donor pool.

    method : Optional[str], default=None
        The method to use for estimating the counterfactuals for spillover donors. Must be one of:
        - "PCR": Use Principal Component Regression.
        - "RPCA": Use Robust PCA.
        - "BOTH": Run the entire iterative SCM procedure separately with both PCR and RPCA and return both results.
        If `None`, the method specified in the SCM config will be used.

    Returns
    -------
    Dict[str, Any]
        If `method="BOTH"`: a dictionary with keys `"PCR"` and `"RPCA"`, each containing the SCM result
        dictionary from the final fit after spillover cleaning using that method.

        If `method` is "PCR" or "RPCA", or inferred from the SCM config: a dictionary containing the
        standard SCM fit result after spillover adjustment.

    Raises
    ------
    MlsynthConfigError
        - If required attributes are missing or improperly specified in the SCM object.
        - If invalid spillover units are provided or there are too few remaining clean donors.
        - If an invalid `method` is passed to `_estimate_counterfactual`.

    MlsynthEstimationError
        - If counterfactual estimation for any spillover donor fails.
        - If the final SCM fit fails after cleaning the donor pool.

    NotImplementedError
        - If the SCM type is not supported for use with this function. Currently supports `CLUSTERSC` only.

    Notes
    -----
    - This function assumes the underlying SCM object represents a single treated unit setting, not cohorts.
    - If `method='BOTH'`, the SCM model is fit twice: once with PCR-based cleaning, once with RPCA-based cleaning.
    - Each spillover donor is processed iteratively: the donor pool is updated after estimating each donor's
      counterfactual, so later donors benefit from prior cleaned estimates.

    Example
    -------
    >>> from mlsynth import CLUSTERSC
    >>> from mlsynth.utils.spillover import iterative_scm
    >>> scm = CLUSTERSC(config)
    >>> results = iterative_scm(scm, spillover_unit_identifiers=["Austria", "France"], method="BOTH")
    >>> pcr_result = results["PCR"]
    >>> rpca_result = results["RPCA"]
    """
    if method is not None and method.upper() == "BOTH":
        results = {}
        for submethod in ["PCR", "RPCA"]:
            result = iterative_scm(scm, spillover_unit_identifiers, method=submethod)
            results[submethod] = result
        return results

    # Extract initial data from the SCM object using the _get_data helper.
    initial_prepared_data = _get_data(scm)
    original_donor_outcomes = initial_prepared_data["donor_matrix"]
    treated_unit_outcome = initial_prepared_data["y"]
    num_pre_treatment_periods = initial_prepared_data["pre_periods"]
    all_donor_identifiers = initial_prepared_data["donor_names"]

    if not isinstance(spillover_unit_identifiers, list) or not spillover_unit_identifiers:
        raise MlsynthConfigError("spillover_unit_identifiers must be a non-empty list.")
    if not all(isinstance(uid, str) for uid in spillover_unit_identifiers):
        raise MlsynthConfigError("All elements in spillover_unit_identifiers must be strings.")
    if method is not None and not isinstance(method, str):
        raise MlsynthConfigError("method, if provided, must be a string.")

    spillover_donor_column_indices = []
    for current_spillover_unit_id in spillover_unit_identifiers:
        try:
            idx = all_donor_identifiers.index(current_spillover_unit_id)
            spillover_donor_column_indices.append(idx)
        except ValueError:
            raise MlsynthConfigError(f"Spillover unit ID '{current_spillover_unit_id}' not found in donor names: {all_donor_identifiers}") from None

    if len(set(spillover_donor_column_indices)) != len(spillover_donor_column_indices):
        raise MlsynthConfigError("Spillover unit IDs must be unique.")

    num_potential_clean_donors = len(all_donor_identifiers) - len(set(spillover_donor_column_indices))
    if num_potential_clean_donors < 2:
        raise MlsynthConfigError(
             f"At least 2 initial clean donors are required. Found {num_potential_clean_donors} "
             f"(Total donors: {len(all_donor_identifiers)}, Spillover donors: {len(set(spillover_donor_column_indices))})."
        )

    iteratively_cleaned_donor_outcomes = original_donor_outcomes.copy()
    clean_donor_pool_indices = [
        i for i in range(len(all_donor_identifiers)) if i not in spillover_donor_column_indices
    ]

    for current_spillover_column_index in spillover_donor_column_indices:
        current_spillover_donor_outcome_series = original_donor_outcomes[:, current_spillover_column_index]
        current_clean_donor_pool_outcomes = iteratively_cleaned_donor_outcomes[:, clean_donor_pool_indices]
        current_clean_donor_pool_identifiers = [all_donor_identifiers[i] for i in clean_donor_pool_indices]

        try:
            estimated_counterfactual_for_spillover_donor = _estimate_counterfactual(
                scm,
                current_clean_donor_pool_outcomes,
                current_spillover_donor_outcome_series,
                current_clean_donor_pool_identifiers,
                num_pre_treatment_periods,
                current_spillover_column_index,
                spillover_donor_column_indices,
                method,
            )
        except Exception as e:
            raise MlsynthEstimationError(
                f"Counterfactual estimation failed for spillover donor "
                f"'{all_donor_identifiers[current_spillover_column_index]}': {e}"
            ) from e

        iteratively_cleaned_donor_outcomes[:, current_spillover_column_index] = estimated_counterfactual_for_spillover_donor.flatten()
        clean_donor_pool_indices.append(current_spillover_column_index)

    spillover_cleaned_dataframe = scm.df.copy()
    for original_donor_column_index, unit_id_val in enumerate(all_donor_identifiers):
        if original_donor_column_index in spillover_donor_column_indices:
            current_unit_mask_for_df_update = spillover_cleaned_dataframe[scm.unitid] == unit_id_val
            spillover_cleaned_dataframe.loc[current_unit_mask_for_df_update, scm.outcome] = \
                iteratively_cleaned_donor_outcomes[:, original_donor_column_index]

    try:
        original_config_model = scm.config
        new_config_data = original_config_model.model_dump()
        new_config_data["df"] = spillover_cleaned_dataframe
        if method is not None and "method" in new_config_data:
            new_config_data["method"] = method

        ConfigModelType = type(original_config_model)
        updated_config_model_instance = ConfigModelType(**new_config_data)

        if isinstance(scm, mlsynth.CLUSTERSC):
            scm_instance_with_cleaned_data = CLUSTERSC(config=updated_config_model_instance)
        else:
            raise NotImplementedError(
                f"SCM type {type(scm).__name__} not supported for re-instantiation in iterative_scm. "
                "Supported base types: CLUSTERSC."
            )

        final_estimation_results = scm_instance_with_cleaned_data.fit()
    except Exception as e:
        raise MlsynthEstimationError(f"Final SCM fitting failed after spillover cleaning: {e}") from e


    return final_estimation_results
