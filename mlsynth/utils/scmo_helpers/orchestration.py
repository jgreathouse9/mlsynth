"""Glue the SCMO pieces together: resolve schemes, run fits, attach inference."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .estimation import fit_scheme, model_average
from .inference import conformal_intervals, placebo_test
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


def run_scmo(
    inputs: SCMOInputs,
    schemes: List[str],
    demean: bool,
    inference: str,
    conformal_alpha: float,
) -> Dict[str, SCMOMethodFit]:
    """Fit each requested scheme and attach the requested inference."""
    base = [s for s in schemes if s != MA]
    fits: Dict[str, SCMOMethodFit] = {s: fit_scheme(inputs, s, demean) for s in base}

    if MA in schemes:
        components = [fits[s] for s in (CONCATENATED, AVERAGED) if s in fits]
        if len(components) < 2:                       # MA needs at least two models
            components = list(fits.values())
        if components:
            fits[MA] = model_average(inputs, components)

    for name in list(fits):
        fit = fits[name]
        if inference == "permutation":
            scheme_for_placebo = CONCATENATED if name == MA else name
            p, _ = placebo_test(inputs, scheme_for_placebo, demean)
            fits[name] = replace(fit, p_value=p)
        elif inference == "conformal":
            ci = conformal_intervals(inputs, fit.counterfactual, conformal_alpha)
            fits[name] = replace(fit, metadata={**fit.metadata, "conformal": ci})
    return fits
