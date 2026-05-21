"""Data preparation for SDID.

Calls :func:`mlsynth.utils.datautils.dataprep` and packages its return
shape (single-treated or cohorts) into a uniform ``cohorts_dict`` that
the math helpers consume. This replaces the inline ``if "cohorts" not in
prep`` restructuring block that used to live in ``SDID.fit()``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance, dataprep
from .structures import SDIDInputs


def prepare_sdid_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
) -> SDIDInputs:
    """Prepare panel data for the SDID pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form balanced panel.
    outcome, treat, unitid, time : str
        Column names identifying the outcome, treatment indicator, units,
        and time periods.

    Returns
    -------
    SDIDInputs
        Pre-processed cohorts payload and metadata.
    """

    balance(df, unitid, time)
    prep: Dict[str, Any] = dataprep(df, unitid, time, outcome, treat)

    if "cohorts" in prep:
        # ``dataprep`` keys cohorts by the actual time *label* (e.g. 2010),
        # but the cohort estimator's ``ell = arange(T) - (a - 1)`` math expects
        # ``a`` to be the cohort's time *index* (1-based). Build a label -> index
        # map from the panel's time axis so the event-time labels come out
        # centered on each cohort's first treated period.
        time_labels_arr = np.asarray(prep["time_labels"])
        label_to_index = {
            label: position + 1
            for position, label in enumerate(time_labels_arr)
        }
        cohorts_dict = {
            int(label_to_index[k]): _coerce_cohort_payload(v)
            for k, v in prep["cohorts"].items()
        }
        # Earliest cohort drives the pre/post counts surfaced on inputs.
        earliest = min(cohorts_dict.keys())
        n_pre = int(cohorts_dict[earliest]["pre_periods"])
        n_post = int(cohorts_dict[earliest]["post_periods"])
        # Treated unit name: in the cohort path, dataprep does not return a
        # single 'treated_unit_name'; surface the first treated label of the
        # earliest cohort instead (used for plotting only).
        first_treated = cohorts_dict[earliest]["treated_indices"]
        treated_unit_name = first_treated[0] if first_treated else None
        # Donor pool labels come from any cohort (they're shared across the
        # cohorts return shape).
        first_label = sorted(prep["cohorts"].keys())[0]
        donor_names = list(prep["cohorts"][first_label]["donor_names"])
    else:
        pre = prep.get("pre_periods")
        post = prep.get("post_periods")
        total = prep.get("total_periods")
        if pre is None or post is None:
            raise MlsynthDataError(
                "dataprep output missing pre_periods/post_periods for the "
                "single-treated-unit case."
            )
        if total is None:
            warnings.warn(
                "'total_periods' missing from dataprep single-unit output; "
                "computing as pre_periods + post_periods.",
                UserWarning,
            )
            total = pre + post

        # ``cohort_key`` is the 1-based index of the *first treated period* so
        # that ``ell = arange(T) - (cohort_key - 1)`` puts ell = 0 on the first
        # treated period (e.g. 1989 for Prop 99). Setting it to ``pre`` instead
        # would shift the post-treatment mask one period early and include the
        # last pre-treatment period in the post-treatment ATT — see
        # Arkhangelsky et al. (2021), Table 1 for the canonical -15.6 value.
        cohort_key = int(pre) + 1
        cohorts_dict = {
            cohort_key: {
                "y": prep["y"].reshape(-1, 1),
                "donor_matrix": prep["donor_matrix"],
                "treated_indices": [prep["treated_unit_name"]],
                "pre_periods": int(pre),
                "post_periods": int(post),
                "total_periods": int(total),
            }
        }
        n_pre = int(pre)
        n_post = int(post)
        treated_unit_name = prep["treated_unit_name"]
        donor_names = list(prep["donor_names"])

    Ywide = prep["Ywide"]
    time_labels = np.asarray(prep["time_labels"])

    return SDIDInputs(
        cohorts_dict=cohorts_dict,
        treated_unit_name=treated_unit_name,
        donor_names=donor_names,
        time_labels=time_labels,
        n_pre=n_pre,
        n_post=n_post,
        Ywide=Ywide,
        outcome=outcome,
    )


def _coerce_cohort_payload(raw_cohort: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt ``dataprep`` cohort entries to the schema the math expects.

    ``dataprep`` returns the cohort treated-outcome matrix as ``y`` but
    keys the treated-unit list as ``treated_units``. The math helpers
    expect ``treated_indices``. Both naming conventions co-exist in the
    library; this shim bridges them without touching either.
    """

    payload = dict(raw_cohort)
    if "treated_indices" not in payload:
        payload["treated_indices"] = list(raw_cohort.get("treated_units", []))
    # The disaggregate-cohort y is already (T, n_treated_in_cohort).
    return payload
