"""Assemble typed FDID results from the raw estimation dictionaries."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .structures import DID, FDID, FDIDInputs, FDIDMethodFit, FDIDResults


def _build_method_fit(
    name: str,
    raw: Dict[str, Any],
    treated: np.ndarray,
    selected_indices: List[int],
    selected_names: List[Any],
    pre_periods: int,
) -> FDIDMethodFit:
    """Map one ``did_from_mean`` result dictionary to an :class:`FDIDMethodFit`."""
    effects = raw["Effects"]
    fit = raw["Fit"]
    inference = raw["Inference"]
    vectors = raw["Vectors"]

    counterfactual = np.asarray(vectors["Counterfactual"], dtype=float)
    gap = np.asarray(treated, dtype=float) - counterfactual

    donor_weights = (
        {name_: 1.0 / len(selected_names) for name_ in selected_names}
        if selected_names
        else {}
    )

    return FDIDMethodFit(
        name=name,
        counterfactual=counterfactual,
        gap=gap,
        att=effects.get("ATT"),
        att_se=inference.get("SE"),
        att_percent=effects.get("Percent ATT"),
        satt=effects.get("SATT"),
        pre_rmse=fit.get("T0 RMSE"),
        r_squared=fit.get("R-Squared"),
        intercept=inference.get("Intercept"),
        p_value=inference.get("P-Value"),
        ci=inference.get("95% CI", (np.nan, np.nan)),
        selected_indices=list(selected_indices),
        selected_names=list(selected_names),
        donor_weights=donor_weights,
        r2_path=raw.get("R2_at_each_step"),
        intermediary=raw.get("intermediary"),
    )


def assemble_fdid_results(
    selector_output: Dict[str, Dict[str, Any]],
    inputs: FDIDInputs,
) -> FDIDResults:
    """Build the typed :class:`FDIDResults` container.

    Parameters
    ----------
    selector_output : dict
        ``{"DID": ..., "FDID": ...}`` as returned by
        :func:`mlsynth.utils.fdid_helpers.estimation.forward_did_select`.
    inputs : FDIDInputs
        Preprocessed panel.

    Returns
    -------
    FDIDResults
        Container exposing the FDID (primary) and DID fits.
    """
    treated = inputs.y
    donor_names = list(inputs.donor_names)

    fdid_raw = selector_output[FDID]
    fdid_indices = fdid_raw.get("selected_controls", [])
    fdid_names = fdid_raw.get("selected_names", [donor_names[i] for i in fdid_indices])
    fdid_fit = _build_method_fit(
        FDID, fdid_raw, treated, fdid_indices, fdid_names, inputs.pre_periods
    )

    did_raw = selector_output[DID]
    all_indices = list(range(len(donor_names)))
    did_fit = _build_method_fit(
        DID, did_raw, treated, all_indices, donor_names, inputs.pre_periods
    )

    return FDIDResults(inputs=inputs, fdid=fdid_fit, did=did_fit)
