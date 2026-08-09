"""Realize an GEOX design once post-treatment outcomes exist.

:meth:`mlsynth.GEOX.fit` chooses a test region before the experiment runs, so
its result is a design with an empty ``report`` slot. This module fills that
slot: apply the chosen region's pre-period fit across the full panel and package
the realized effect into the standardized result models.

The readout runs on the same engine the design scored itself with, whichever
that was, because GEOX's claim is that the design is chosen by the estimator
that will analyse the result. A minimum detectable effect computed one way and a
readout computed another would leave the reported power describing a test nobody
ran.

For the same reason the readout inherits the design's donor pool. Where a
spillover constraint barred a treated market's conflict-neighbours from its
synthetic control, ``exclude`` carries that restriction through, so the realized
effect is measured against the comparison the MDE was promised for.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ...config_models import (
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
    TimeSeriesResults,
    WeightsResults,
)
from ...exceptions import MlsynthConfigError
from .engines import resolve_engine
from .shaping import aggregate_treated, donor_matrix


def realize_design(
    Ywide_full: pd.DataFrame,
    candidate: Iterable,
    pre_periods: int,
    *,
    how: str = "mean",
    exclude: Optional[Iterable] = None,
    alpha: float = 0.1,
    n_draws: int = 200,
    seed: int = 0,
    cpic: Optional[float] = None,
    engine: str = "sdid",
    engine_kwargs: Optional[dict] = None,
) -> BaseEstimatorResults:
    """Realize one candidate design on the full (pre + post) panel.

    Parameters
    ----------
    Ywide_full : pandas.DataFrame
        The full wide panel, pre and post, as ``geoex_dataprep(...)["Ywide"]``.
    candidate : iterable
        The treated markets, normally the design's winning region.
    pre_periods : int
        Periods before the intervention; the split point.
    how : {"mean", "sum"}, default "mean"
        Reporting scale. The fit runs on the per-market mean either way, since
        that is the series the design was scored on; ``"sum"`` multiplies the
        reported paths by the region size to give the summed incremental.
    exclude : iterable, optional
        Markets barred from the donor pool beyond the treated region itself
        (the spillover exclusion). ``None`` leaves the candidate complement.
    alpha : float, default 0.1
        Significance level carried into the reported confidence level.
    n_draws : int, default 200
        Placebo draws behind the standard error.
    seed : int, default 0
        Seed for the placebo draws.
    cpic : float, optional
        Cost per incremental conversion. When set, the realized spend
        ``cpic x summed incremental outcome`` is reported.

    Returns
    -------
    BaseEstimatorResults
        ATT over the post window, the observed / counterfactual / gap paths over
        the whole panel, placebo inference, both SDID weight vectors, and the
        pre-period fit diagnostics.

    Raises
    ------
    MlsynthConfigError
        If ``how`` is not ``"mean"``/``"sum"``, or ``pre_periods`` leaves no
        pre-period or no post-period.
    MlsynthDataError
        If a candidate market is absent, or no donors remain.
    """
    if how not in ("mean", "sum"):
        raise MlsynthConfigError(f"how must be 'sum' or 'mean'; got {how!r}.")
    n_periods = Ywide_full.shape[0]
    if not 0 < pre_periods < n_periods:
        if pre_periods >= n_periods:
            raise MlsynthConfigError(
                f"pre_periods={pre_periods} leaves no post-treatment period in "
                f"a panel of {n_periods}. The design cannot be realized until "
                "outcomes are observed.")
        raise MlsynthConfigError(
            f"pre_periods must be in (0, {n_periods}); got {pre_periods}.")

    members = sorted(map(str, candidate))
    # The fit runs on the mean regardless of the reporting scale: that is the
    # series every backtest was scored on, so realizing on the sum would answer
    # with a different fit than the one the MDE describes.
    treated = aggregate_treated(Ywide_full, candidate, how="mean").to_numpy()
    donors_df = donor_matrix(Ywide_full, candidate, exclude=exclude)
    donors = donors_df.to_numpy()

    eng = resolve_engine(engine)
    start, end = pre_periods, n_periods - 1
    ekw = dict(engine_kwargs or {})
    fit = eng.fit_once(treated, donors, pre_periods, start, end, len(members),
                       **ekw)
    tau = eng.att(fit, treated, start, end)
    p_value, inf_details = eng.point_inference(
        fit, treated, donors, pre_periods, start, end,
        n_draws=n_draws, n_tr=len(members), seed=seed, **ekw)

    # Reporting scale. The statistic is a ratio, so it is scale-free and the
    # p-value is unmoved; only the reported paths change units.
    k = float(len(members)) if how == "sum" else 1.0
    observed = treated * k
    counterfactual = fit.counterfactual * k
    gap = observed - counterfactual

    # Cost tracks the summed incremental outcome across the treated markets,
    # whatever scale the paths are reported in.
    tau_path = treated - fit.counterfactual
    incremental = float(np.sum(tau_path[pre_periods:])) * len(members)
    cost = float(cpic) * incremental if cpic is not None else None

    return BaseEstimatorResults(
        effects=EffectsResults(att=float(np.mean(gap[pre_periods:]))),
        time_series=TimeSeriesResults(
            observed_outcome=observed,
            counterfactual_outcome=counterfactual,
            estimated_gap=gap,
            time_periods=np.asarray(Ywide_full.index),
            intervention_time=Ywide_full.index[pre_periods],
        ),
        inference=InferenceResults(
            p_value=float(p_value),
            method=str(inf_details.get("method", engine)),
            confidence_level=1.0 - alpha,
            details={**inf_details, "att_mean_scale": float(tau)},
        ),
        weights=WeightsResults(
            donor_weights={str(name): float(w)
                           for name, w in zip(donors_df.columns,
                                              fit.donor_weights)},
            time_weights=({period: float(w)
                           for period, w in zip(Ywide_full.index[:pre_periods],
                                                fit.time_weights)}
                          if fit.time_weights is not None else None),
            summary_stats={
                "how": how,
                "treated_markets": members,
                "engine": engine,
                "cpic": cpic,
                "cost": cost,
                **{k: float(v) for k, v in fit.extras.items()},
            },
        ),
        fit_diagnostics=FitDiagnosticsResults(
            rmse_pre=float(fit.pre_rmspe),
            additional_metrics={"scaled_l2": float(fit.scaled_l2)},
        ),
    )
