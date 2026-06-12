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
    fixed_effects: bool = False,
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
    donors = donor_matrix(Ywide_full, candidate)
    Y0 = donors.to_numpy()
    n_units = len(candidate)

    # augsynth fits the *mean* of the treated units (``colMeans``); under fixed
    # effects this is what reproduces the effect estimate and conformal p-value,
    # and the p-value is invariant to the reporting scale. So fit (and run
    # inference) on the per-unit mean series, then rescale the reported paths by
    # the unit count when ``how="sum"`` (the summed incremental). Without fixed
    # effects, preserve the historical behaviour (aggregate as ``how``).
    fit_how = "mean" if fixed_effects else how
    treated_fit = aggregate_treated(Ywide_full, candidate, how=fit_how).to_numpy()
    n_periods = treated_fit.shape[0]

    fit = fit_augsynth_once(
        treated_fit[:pre_periods], Y0[:pre_periods],
        augment=augment, donor_names=[str(c) for c in donors.columns],
        fixed_effects=fixed_effects,
    )
    cf_fit = fit.predict(Y0)
    gap_fit = treated_fit - cf_fit

    ci = conformal_intervals(
        treated_fit, Y0, pre_periods,
        lambda_=fit.lambda_, alpha=alpha, q=q, ns=ns, seed=seed,
        conformal_type=conformal_type, fixed_effects=fixed_effects,
    )

    # Reporting scale: ``how="sum"`` reports the summed incremental across the
    # treated units; ``how="mean"`` reports the per-unit effect.
    scale = float(n_units) if (fixed_effects and how == "sum") else 1.0
    treated = treated_fit * scale
    counterfactual = cf_fit * scale
    gap = gap_fit * scale

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
                "att": np.asarray(ci.att, dtype=float) * scale,
                "lower": np.asarray(ci.lower, dtype=float) * scale,
                "upper": np.asarray(ci.upper, dtype=float) * scale,
                "p_value": np.asarray(ci.p_value, dtype=float),
            },
        ),
        weights=WeightsResults(
            donor_weights=donor_weights,
            summary_stats={
                "intercept": float(fit.intercept),
                "augment": augment,
                "fixed_effects": fixed_effects,
            },
        ),
        fit_diagnostics=FitDiagnosticsResults(
            rmse_pre=float(fit.pre_rmspe),
            additional_metrics={"scaled_l2": float(fit.scaled_l2)},
        ),
    )
