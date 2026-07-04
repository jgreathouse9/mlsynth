"""Glue the SCMO pieces together: resolve schemes, run fits, attach inference."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .estimation import fit_scheme, model_average
from .inference import conformal_inference
from .structures import AVERAGED, CONCATENATED, MA, SEPARATE, SCMOInputs, SCMOMethodFit

_METHOD_TO_SCHEMES = {
    "TLP": [CONCATENATED],
    "SBMF": [AVERAGED],
    "BOTH": [CONCATENATED, AVERAGED, MA],
}


def resolve_schemes(schemes: Optional[List[str]], method: str) -> List[str]:
    """Pick the weighting schemes to run (explicit ``schemes`` win over ``method``)."""
    if schemes:
        return list(schemes)
    return list(_METHOD_TO_SCHEMES.get(method, [CONCATENATED]))


def derive_treatment(
    df: pd.DataFrame, unitid: str, time: str, treat: str
) -> Tuple[Any, Any, List[Any]]:
    """Read treated unit, intervention time, and pre-period years off the treat column."""
    treated_rows = df[df[treat] == 1]
    if treated_rows.empty:
        raise MlsynthDataError(f"No treated rows (treat == 1) found in '{treat}'.")
    treated_units = pd.unique(treated_rows[unitid])
    if len(treated_units) != 1:
        raise MlsynthDataError(
            f"SCMO expects exactly one treated unit; found {len(treated_units)}."
        )
    treated_unit = treated_units[0]
    intervention_time = treated_rows.loc[treated_rows[unitid] == treated_unit, time].min()
    pre_years = sorted(t for t in pd.unique(df[time]) if t < intervention_time)
    return treated_unit, intervention_time, pre_years


def build_spec(
    explicit_spec: Optional[Dict[str, Any]],
    outcome: str,
    addout,
    pre_years: List[Any],
) -> Dict[str, Any]:
    """Use an explicit spec, or stack the outcome + auxiliary outcomes over the pre-period."""
    if explicit_spec is not None:
        return explicit_spec
    aux = [addout] if isinstance(addout, str) and addout else (list(addout) if addout else [])
    outcomes = [outcome] + [a for a in aux if a]
    return {"year": pre_years, "vars": {o: o for o in outcomes}}


def _resolve_metric_weights(
    inputs: SCMOInputs, weights: str, pcr_metric_weights, pcr_cv_grid,
    pcr_cv_horizon: int, pcr_cv_min_train, pcr_rank, pcr_cumvar,
) -> Tuple[Optional[List[float]], Dict[str, Any]]:
    """Resolve the per-metric PCR weights (equal / explicit / rolling-origin CV)."""
    if weights != "pcr" or pcr_metric_weights is None:
        return None, {}
    from .pcr_cv import metric_ids, metric_order, rolling_origin_pcr_cv

    order = metric_order(metric_ids(inputs.predictor_labels))
    if pcr_metric_weights == "cv":
        resolved, order, mse_table = rolling_origin_pcr_cv(
            inputs, grid=pcr_cv_grid, horizon=pcr_cv_horizon,
            min_train=pcr_cv_min_train, pcr_rank=pcr_rank, pcr_cumvar=pcr_cumvar)
        info = {"pcr_metric_weights": {order[i]: resolved[i] for i in range(len(order))},
                "pcr_cv_mse": {float(k): float(v) for k, v in mse_table.items()}}
        return resolved, info
    resolved = [float(w) for w in pcr_metric_weights]
    if len(resolved) != len(order):
        raise MlsynthDataError(
            f"pcr_metric_weights has length {len(resolved)}, expected one per "
            f"metric ({len(order)}: {order}).")
    return resolved, {"pcr_metric_weights": {order[i]: resolved[i] for i in range(len(order))}}


def run_scmo(
    inputs: SCMOInputs,
    schemes: List[str],
    demean: bool,
    conformal_alpha: float,
    conformal_q: float = 1.0,
    augment: Optional[str] = None,
    ridge_lambda: Optional[float] = None,
    weights: str = "simplex",
    pcr_rank: Optional[int] = None,
    pcr_cumvar: float = 0.95,
    pcr_metric_weights=None,
    pcr_cv_grid: Optional[List[float]] = None,
    pcr_cv_horizon: int = 1,
    pcr_cv_min_train: Optional[int] = None,
) -> Dict[str, SCMOMethodFit]:
    """Fit each requested scheme and attach CWZ conformal inference to every fit."""
    metric_weights, mw_info = _resolve_metric_weights(
        inputs, weights, pcr_metric_weights, pcr_cv_grid, pcr_cv_horizon,
        pcr_cv_min_train, pcr_rank, pcr_cumvar)

    base = [s for s in schemes if s != MA]
    fits: Dict[str, SCMOMethodFit] = {
        s: fit_scheme(inputs, s, demean, augment, ridge_lambda,
                      weights, pcr_rank, pcr_cumvar, metric_weights) for s in base}
    if mw_info and CONCATENATED in fits:
        f = fits[CONCATENATED]
        fits[CONCATENATED] = replace(f, metadata={**f.metadata, **mw_info})

    if MA in schemes:
        components = [fits[s] for s in (CONCATENATED, AVERAGED) if s in fits]
        if len(components) < 2:                       # MA needs at least two models
            components = list(fits.values())
        if components:
            fits[MA] = model_average(inputs, components)

    for name in list(fits):
        fit = fits[name]
        _, p_value, ci = conformal_inference(
            inputs.y_treated, fit.counterfactual, inputs.T0,
            alpha=conformal_alpha, q=conformal_q,
        )
        fits[name] = replace(fit, p_value=p_value, ci=ci)
    return fits
