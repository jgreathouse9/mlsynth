"""Plotting wrapper for the Forward Difference-in-Differences estimator."""

from __future__ import annotations

import warnings
from typing import List, Union

from ...exceptions import MlsynthDataError, MlsynthPlottingError
from ..resultutils import plot_estimates
from .structures import FDIDResults


def plot_fdid(
    results: FDIDResults,
    *,
    time: str,
    unitid: str,
    outcome: str,
    treat: str,
    treated_color: str,
    counterfactual_color: Union[str, List[str]],
    save: Union[bool, dict],
) -> None:
    """Plot observed vs FDID and DID counterfactuals.

    Plotting failures are downgraded to warnings so a rendering problem
    never masks a successful estimation.
    """
    inputs = results.inputs
    treated_name = inputs.treated_unit_name
    try:
        plot_estimates(
            processed_data_dict=inputs.prepped,
            time_axis_label=time,
            unit_identifier_column_name=unitid,
            outcome_variable_label=outcome,
            treatment_name_label=treat,
            treated_unit_name=treated_name,
            observed_outcome_series=inputs.y,
            counterfactual_series_list=[
                results.fdid.counterfactual,
                results.did.counterfactual,
            ],
            estimation_method_name="FDID",
            counterfactual_names=[
                f"FDID {treated_name}",
                f"DID {treated_name}",
            ],
            treated_series_color=treated_color,
            save_plot_config=save,
            counterfactual_series_colors=counterfactual_color,
        )
    except (MlsynthPlottingError, MlsynthDataError) as e:
        warnings.warn(f"Plotting failed: {str(e)}", UserWarning)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"Unexpected plotting error: {str(e)}", UserWarning)
