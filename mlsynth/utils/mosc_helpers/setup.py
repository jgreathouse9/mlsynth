"""Long-DataFrame -> NumPy boundary for MOSC (wraps ``dataprep``).

Everything a fit needs to refuse is refused here, before any sampling happens.
The library declines Postel's first half on purpose: a panel that MOSC cannot
identify from must raise, because the alternative is a counterfactual that looks
like an estimate and is not one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from .factor import NON_NEGATIVE_ONLY
from .structures import MOSCInputs

def check_modelling_scale(panel: np.ndarray, factor_model: str, outcome_scale: str) -> None:
    """Refuse a likelihood whose support excludes the panel it would be fit to.

    The constraint belongs to the scale the factor model sees, not to the outcome
    as supplied: first differences of a count series are routinely negative, so a
    panel that is admissible in levels need not be admissible differenced.
    """
    if factor_model not in NON_NEGATIVE_ONLY:
        return
    modelled = np.diff(panel, axis=0) if outcome_scale == "difference" else panel
    if modelled.min() < 0:
        scale = "first differences of the outcome" if outcome_scale == "difference" else "the outcome"
        raise MlsynthDataError(
            f"factor_model={factor_model!r} places a Poisson likelihood on "
            f"{scale}, which has no support for negative values, and the panel "
            f"holds values as low as {modelled.min():.4g}. Use "
            f"factor_model='ppca' for a real-valued outcome."
        )


#: The outcome regression fits one coefficient per latent factor plus a
#: treatment dummy and an intercept, across units. Below this many spare units
#: the fit interpolates and the counterfactual carries no information.
_SPARE_UNITS_REQUIRED = 3


def prepare_mosc_inputs(
    df: pd.DataFrame,
    outcome: str,
    unitid: str,
    time: str,
    treat: str,
    n_factors: int,
    factor_model: str,
    outcome_scale: str = "level",
) -> MOSCInputs:
    """Pivot a long panel into MOSC's ``(T, N)`` matrix, treated unit first."""
    prepared = dataprep(df, unitid, time, outcome, treat)
    if "cohorts" in prepared:
        raise MlsynthDataError(
            "MOSC takes a single treated unit; the panel carries multiple "
            "treatment cohorts. Fit each cohort separately."
        )

    y_target = np.asarray(prepared["y"], dtype=float)
    donors = np.asarray(prepared["donor_matrix"], dtype=float)
    total_periods = int(prepared["total_periods"])
    pre_periods = int(prepared["pre_periods"])

    if pre_periods < 2:
        raise MlsynthDataError(
            f"MOSC needs at least 2 pre-treatment periods to learn the "
            f"confounding structure; got {pre_periods}."
        )
    if total_periods - pre_periods < 1:  # pragma: no cover - a treated period forces one
        raise MlsynthDataError("MOSC needs at least 1 post-treatment period.")

    panel = np.column_stack([y_target, donors])
    n_units = panel.shape[1]

    if n_units < n_factors + _SPARE_UNITS_REQUIRED:
        raise MlsynthDataError(
            f"MOSC adjusts for {n_factors} latent factors across {n_units} units, "
            f"which leaves the outcome regression too few units to fit. Supply at "
            f"least {n_factors + _SPARE_UNITS_REQUIRED} units, or lower n_factors."
        )
    if n_factors >= pre_periods:
        raise MlsynthDataError(
            f"MOSC cannot identify {n_factors} factors from {pre_periods} "
            f"pre-treatment periods; n_factors must be smaller."
        )
    if not np.isfinite(panel).all():
        raise MlsynthDataError(
            "MOSC requires a balanced panel with no missing outcomes."
        )
    check_modelling_scale(panel, factor_model, outcome_scale)

    return MOSCInputs(
        panel=panel,
        y_target=y_target,
        pre_periods=pre_periods,
        total_periods=total_periods,
        n_units=n_units,
        treated_unit_name=str(prepared.get("treated_unit_name", "treated")),
        donor_names=list(prepared.get("donor_names", range(donors.shape[1]))),
        time_labels=np.asarray(prepared["time_labels"]),
    )
