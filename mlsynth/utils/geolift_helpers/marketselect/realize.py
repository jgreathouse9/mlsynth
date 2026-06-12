"""Realize a design once post-treatment outcomes exist.

Apply a chosen test region's pre-period synthetic control to the *full* panel,
and build the standardized realized effect report (the ``DesignResult.report``):
observed / counterfactual / gap over pre+post, the conformal per-period
prediction intervals, and the joint-null p-value.
"""

from typing import Optional

import numpy as np
import pandas as pd

from mlsynth.config_models import (
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
    TimeSeriesResults,
    WeightsResults,
)
from mlsynth.utils.bilevel.ridge_inference import conformal_intervals

from .helpers.shaping import aggregate_treated, donor_matrix
from .helpers.fit import fit_augsynth_once

_WEIGHT_EPS = 1e-8


def realize_design(
    Ywide_full: pd.DataFrame,
    candidate: frozenset,
    pre_periods: int,
    *,
    how: str = "sum",
    augment: Optional[str] = "ridge",
    alpha: float = 0.1,
    q: float = 1.0,
    ns: int = 1000,
    seed: int = 0,
    conformal_type: str = "iid",
) -> BaseEstimatorResults:
    """Realize one candidate design on the full (pre+post) panel.

    Parameters
    ----------
    Ywide_full : pd.DataFrame
        The full wide panel (pre + post), e.g. ``geoex_dataprep(df)["Ywide"]``.
    candidate : frozenset
        The chosen test-market set (e.g. the design winner).
    pre_periods : int
        Number of pre-treatment periods (the split point).
    how, augment, alpha, q, ns, seed
        Aggregation / estimator / conformal-inference settings.

    Returns
    -------
    BaseEstimatorResults
        The realized effect report: ATT, observed/counterfactual/gap time series
        (with the intervention boundary), conformal inference (joint p-value plus
        per-period effect intervals in ``inference.details``), weights, fit
        diagnostics.
    """
    treated = aggregate_treated(Ywide_full, candidate, how=how).to_numpy()
    donors = donor_matrix(Ywide_full, candidate)
    Y0 = donors.to_numpy()
    n_periods = treated.shape[0]

    fit = fit_augsynth_once(
        treated[:pre_periods], Y0[:pre_periods],
        augment=augment, donor_names=[str(c) for c in donors.columns],
    )
    counterfactual = fit.predict(Y0)
    gap = treated - counterfactual

    ci = conformal_intervals(
        treated, Y0, pre_periods,
        lambda_=fit.lambda_, alpha=alpha, q=q, ns=ns, seed=seed,
        conformal_type=conformal_type,
    )

    time_labels = np.asarray(Ywide_full.index)
    intervention = time_labels[pre_periods] if pre_periods < n_periods else None
    att = float(np.mean(gap[pre_periods:])) if pre_periods < n_periods else None

    donor_weights = {
        str(name): float(w)
        for name, w in zip(donors.columns, fit.weights)
        if abs(w) > _WEIGHT_EPS
    }

    return BaseEstimatorResults(
        effects=EffectsResults(att=att),
        time_series=TimeSeriesResults(
            observed_outcome=treated,
            counterfactual_outcome=counterfactual,
            estimated_gap=gap,
            time_periods=time_labels,
            intervention_time=intervention,
        ),
        inference=InferenceResults(
            p_value=float(ci.joint_p_value),
            method="conformal",
            confidence_level=1.0 - alpha,
            details={
                "periods": list(ci.periods),
                "att": np.asarray(ci.att, dtype=float),
                "lower": np.asarray(ci.lower, dtype=float),
                "upper": np.asarray(ci.upper, dtype=float),
                "p_value": np.asarray(ci.p_value, dtype=float),
            },
        ),
        weights=WeightsResults(
            donor_weights=donor_weights,
            summary_stats={"intercept": float(fit.intercept), "augment": augment},
        ),
        fit_diagnostics=FitDiagnosticsResults(
            rmse_pre=float(fit.pre_rmspe),
            additional_metrics={"scaled_l2": float(fit.scaled_l2)},
        ),
    )
