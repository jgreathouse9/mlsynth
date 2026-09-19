"""Data preparation for the Forward Difference-in-Differences estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Any, List, Optional

from ...exceptions import MlsynthDataError, MlsynthEstimationError
from ..datautils import balance, dataprep
from .structures import FDIDInputs, FDIDStaggeredInputs


def prepare_panel(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
) -> dict:
    """Balance the panel and run it through ``dataprep``.

    Returns the raw :func:`mlsynth.utils.datautils.dataprep` dictionary. It
    carries a ``"cohorts"`` key when the panel has several treated units, which
    is what routes the fit to the staggered pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Long panel with outcome, treatment, unit, and time columns.
    outcome, treat, unitid, time : str
        Column names identifying the outcome, treatment indicator, unit,
        and time period.

    Returns
    -------
    dict
        The prepared panel.

    Raises
    ------
    MlsynthDataError
        If panel balancing or data preparation fails.
    """
    try:
        keys = balance(df, unitid, time)
    except Exception as e:  # noqa: BLE001 - re-wrap as repository error
        raise MlsynthDataError(f"Error balancing panel data: {str(e)}") from e

    try:
        return dataprep(df, unitid, time, outcome, treat, keys=keys)
    except MlsynthDataError:
        raise
    except Exception as e:  # noqa: BLE001 # pragma: no cover - dataprep raises
        # MlsynthDataError for every malformed panel it recognises; this arm
        # only catches a failure mode it does not, which no fixture can stage.
        raise MlsynthDataError(f"Error preparing data matrices: {str(e)}") from e


def prepare_staggered_inputs(
    prepped: dict, verbose: bool = True
) -> FDIDStaggeredInputs:
    """Package ``dataprep``'s cohort output for the staggered pipeline.

    Every cohort in ``dataprep``'s cohort mode is handed the same donor block --
    the units never treated anywhere in the panel -- so the pool is read once
    and shared. That is the eligibility rule the staggered estimator needs:
    a unit that adopts at any point is excluded from every treated unit's
    criterion as well as from its counterfactual.

    Parameters
    ----------
    prepped : dict
        A :func:`prepare_panel` result carrying ``"cohorts"``.
    verbose : bool, default True
        Whether per-unit selection paths are recorded.

    Returns
    -------
    FDIDStaggeredInputs

    Raises
    ------
    MlsynthEstimationError
        If no cohort carries a treated unit, or the cohorts disagree about the
        donor pool.
    """
    cohorts = prepped.get("cohorts") or {}
    if not cohorts:
        raise MlsynthEstimationError(
            "No treated cohorts were found in the prepared panel."
        )

    wide = prepped["Ywide"]
    time_labels = np.asarray(prepped["time_labels"])
    index = list(wide.index)

    donor_names: Optional[List[Any]] = None
    treated_names: List[Any] = []
    adoption_index: List[int] = []
    columns: List[np.ndarray] = []

    for start_time, cohort in sorted(cohorts.items(), key=lambda kv: index.index(kv[0])):
        names = list(cohort["donor_names"])
        if donor_names is None:
            donor_names = names
        elif names != donor_names:  # pragma: no cover - dataprep builds the
            # donor block from one global set of never-treated units, so every
            # cohort is handed the same list; this guards a change to that.
            raise MlsynthEstimationError(
                "Cohorts disagree about the never-treated donor pool; the "
                "staggered estimator needs one pool that is clean throughout."
            )
        g = int(cohort["pre_periods"])
        for j, unit in enumerate(cohort["treated_units"]):
            treated_names.append(unit)
            adoption_index.append(g)
            columns.append(np.asarray(cohort["y"], dtype=float)[:, j])

    donor_matrix = np.asarray(
        wide[donor_names].to_numpy(), dtype=float
    ) if donor_names else np.empty((len(index), 0))

    return FDIDStaggeredInputs(
        treated_matrix=np.column_stack(columns),
        treated_names=treated_names,
        adoption_index=np.asarray(adoption_index, dtype=int),
        donor_matrix=donor_matrix,
        donor_names=list(donor_names or []),
        time_labels=time_labels,
        T=len(index),
        verbose=verbose,
    )


def prepare_fdid_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    verbose: bool = True,
) -> FDIDInputs:
    """Balance the panel, pivot it, and package it into :class:`FDIDInputs`.

    Parameters
    ----------
    df : pd.DataFrame
        Long panel with outcome, treatment, unit, and time columns.
    outcome, treat, unitid, time : str
        Column names identifying the outcome, treatment indicator, unit,
        and time period.
    verbose : bool, default True
        Whether the forward-selection path should be recorded step by step.

    Returns
    -------
    FDIDInputs
        Preprocessed panel ready for forward selection.

    Raises
    ------
    MlsynthDataError
        If panel balancing or data preparation fails (e.g. no donor units).
    MlsynthEstimationError
        If fewer than two pre-treatment periods are available.
    """
    prepped = prepare_panel(df, outcome, treat, unitid, time)

    pre_periods = prepped.get("pre_periods")
    if pre_periods is None or pre_periods < 2:
        raise MlsynthEstimationError("Insufficient pre-periods for estimation.")

    return FDIDInputs(
        y=np.asarray(prepped["y"], dtype=float),
        donor_matrix=np.asarray(prepped["donor_matrix"], dtype=float),
        pre_periods=int(pre_periods),
        post_periods=int(prepped["post_periods"]),
        T=int(prepped["total_periods"]),
        donor_names=list(prepped["donor_names"]),
        time_labels=np.asarray(prepped["time_labels"]),
        treated_unit_name=prepped["treated_unit_name"],
        verbose=verbose,
        prepped=prepped,
    )
