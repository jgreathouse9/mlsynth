"""Per-candidate design fit -> standardized result models.

For a candidate test-market set, fit the synthetic control on the **full
pre-period** (the SC you would actually deploy, distinct from the lookback power
fits) and package the result into the library's standardized Pydantic models,
mirroring how LEXSCM's ``SEDCandidate`` groups one candidate's outputs:

- :class:`~mlsynth.config_models.WeightsResults` -- the donor weights, with the
  level ``intercept`` carried as a **sibling field** next to ``weights``;
- :class:`~mlsynth.config_models.TimeSeriesResults` -- observed / counterfactual
  / gap over the pre-period;
- :class:`~mlsynth.config_models.FitDiagnosticsResults` -- ``rmse_pre`` plus the
  scaled L2 imbalance.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from mlsynth.config_models import (
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
)

from .shaping import aggregate_treated, donor_matrix
from .fit import fit_augsynth_once

_WEIGHT_EPS = 1e-8


@dataclass
class CandidateDesign:
    """One candidate test-market set's design, à la LEXSCM ``SEDCandidate``.

    ``weights`` and ``intercept`` are siblings (the intercept is the level shift
    that accompanies the donor weights). The ``mde`` / ``power`` / ``rank`` slots
    are filled later, when the design fit is merged with the aggregated ranking.
    """

    candidate: frozenset
    weights: WeightsResults
    intercept: float
    time_series: TimeSeriesResults
    fit_diagnostics: FitDiagnosticsResults
    augment: Optional[str] = None
    lambda_: Optional[float] = None
    mde: Optional[float] = None
    power: Optional[float] = None
    rank: Optional[float] = None


def design_fit(
    Ywide: pd.DataFrame,
    candidate: frozenset,
    *,
    how: str = "sum",
    augment: Optional[str] = "ridge",
    fixed_effects: bool = False,
) -> CandidateDesign:
    """Fit the deployable synthetic control for one candidate on the pre-period.

    Parameters
    ----------
    Ywide : pd.DataFrame
        The (pre-period) wide panel from ``geoex_dataprep``.
    candidate : frozenset
        The candidate test-market set.
    how : {"sum", "mean"}, default "sum"
        Treated-aggregation (match the value used for scoring).
    augment : {"ridge", None}, default "ridge"
        Point-fit estimator.
    fixed_effects : bool, default False
        Unit fixed effects (augsynth ``fixed_effects=TRUE``). When set, the SCM
        is fit on the mean of the treated units (matching the realized report).

    Returns
    -------
    CandidateDesign
        Standardized per-candidate result (weights + intercept + time series +
        fit diagnostics).
    """
    treated = aggregate_treated(Ywide, candidate, how=("mean" if fixed_effects else how))
    donors = donor_matrix(Ywide, candidate)
    y = treated.to_numpy()
    Y0 = donors.to_numpy()

    fit = fit_augsynth_once(
        y, Y0, augment=augment, donor_names=[str(c) for c in donors.columns],
        fixed_effects=fixed_effects,
    )

    donor_weights = {
        str(name): float(w)
        for name, w in zip(donors.columns, fit.weights)
        if abs(w) > _WEIGHT_EPS
    }
    weights = WeightsResults(
        donor_weights=donor_weights,
        summary_stats={
            "n_treated": len(candidate),
            "n_donors": int(donors.shape[1]),
            "augment": augment,
        },
    )

    counterfactual = fit.predict(Y0)
    time_series = TimeSeriesResults(
        observed_outcome=y,
        counterfactual_outcome=counterfactual,
        estimated_gap=y - counterfactual,
        time_periods=np.asarray(Ywide.index),
        intervention_time=None,
    )

    fit_diagnostics = FitDiagnosticsResults(
        rmse_pre=fit.pre_rmspe,
        additional_metrics={"scaled_l2": fit.scaled_l2},
    )

    return CandidateDesign(
        candidate=candidate,
        weights=weights,
        intercept=float(fit.intercept),
        time_series=time_series,
        fit_diagnostics=fit_diagnostics,
        augment=augment,
        lambda_=fit.lambda_,
    )
