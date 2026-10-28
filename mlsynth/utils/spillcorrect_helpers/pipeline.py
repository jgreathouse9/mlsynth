"""SPILLCORRECT's pipeline: correct, recompute the design's inference, report.

The design is kept: its treated and control composites are applied to the
corrected panel, and its inference is recomputed there with the design's own
settings (MAREX's ``compute_inference``). With the default window the
correction starts at the end of the design's fitting window, so the blank
periods the band and the test are calibrated on carry the correction's own
prediction error, as the post periods do.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ...config_models import (EffectsResults, FitDiagnosticsResults,
                              InferenceResults, MethodDetailsResults,
                              TimeSeriesResults, WeightsResults)
from ...exceptions import MlsynthConfigError, MlsynthEstimationError
from ..marex_helpers.inference import compute_inference
from ..marex_helpers.structures import MAREXInferenceSettings
from .correct import inclusive_correction, iterative_correction
from .setup import prepare_correction
from .structures import CorrectionInputs, SpillCorrectEstimate, SpillCorrectResults


def run_spillcorrect(config) -> SpillCorrectResults:
    """Correct the design's contaminated controls and report the three arms."""
    inp = prepare_correction(config)
    settings = _settings(config)
    start = inp.fit_end if config.window == "blank_and_post" else inp.T0

    uncorrected, unc_c = _estimate(inp.Y, inp, settings)
    fix = None
    if inp.corrected:
        if config.method == "inclusive":
            syn_t = inp.wt @ inp.Y
            fix = inclusive_correction(inp.Y, inp.corrected, inp.clean, start,
                                       treated_series=syn_t, gap=syn_t - inp.wc @ inp.Y,
                                       v=inp.wc[inp.corrected], scale=config.scale)
        else:
            fix = iterative_correction(inp.Y, inp.corrected, inp.clean, start,
                                       scale=config.scale)
        corrected, cor_c = _estimate(fix.cleaned, inp, settings)
    else:
        corrected, cor_c = uncorrected, unc_c

    restricted, res_c, res_w = None, None, None
    if config.design_config is not None:
        restricted, res_c, res_w = _restricted(config, inp)

    if config.method == "restricted":
        headline, head_c, head_w = restricted, res_c, res_w
    else:
        headline, head_c, head_w = corrected, cor_c, _weights(inp.labels, inp.wc)

    names = [inp.labels[i] for i in inp.corrected]
    observed = inp.wt @ inp.Y
    t = (np.asarray(inp.time_labels) if inp.time_labels is not None
         else np.arange(observed.shape[0]))
    gap = observed - head_c
    return SpillCorrectResults(
        effects=EffectsResults(att=headline.att),
        time_series=TimeSeriesResults(
            observed_outcome=observed, counterfactual_outcome=head_c,
            estimated_gap=gap, time_periods=t, intervention_time=t[inp.T0]),
        fit_diagnostics=FitDiagnosticsResults(
            rmse_pre=float(np.sqrt(np.mean(gap[:inp.fit_end] ** 2)))),
        inference=(InferenceResults(
            p_value=headline.p_value, confidence_level=1.0 - settings.alpha,
            method=f"design permutation ({settings.global_test}), recomputed")
            if headline.p_value is not None else None),
        weights=WeightsResults(
            donor_weights=head_w,
            summary_stats={"constraint": "the design's control composite"}),
        method_details=MethodDetailsResults(
            method_name=f"SPILLCORRECT/{config.method}",
            parameters_used={"scale": config.scale, "window": config.window,
                             "correction_start": int(start),
                             "n_corrected": len(names)}),
        method=config.method, scale=config.scale, window=config.window,
        correction_start=int(start),
        corrected=headline, uncorrected=uncorrected, restricted=restricted,
        restricted_weights=res_w,
        corrected_units=names,
        skipped_units=[inp.labels[i] for i in inp.skipped],
        donor_units=[inp.labels[i] for i in inp.clean],
        n_corrected=len(names),
        spillover=({u: np.asarray(fix.spillover[i]) for i, u in enumerate(names)}
                   if fix is not None else {}),
        cleaning_rmspe=({u: float(fix.cleaning_rmspe[i]) for i, u in enumerate(names)}
                        if fix is not None else {}),
        control_weights=_weights(inp.labels, inp.wc),
        composite_weight=({u: float(fix.composite_weight[i]) for i, u in enumerate(names)}
                          if fix is not None and fix.composite_weight is not None else None),
        omega_det=fix.omega_det if fix is not None else None,
    )


def _settings(config) -> Optional[MAREXInferenceSettings]:
    s = config.design.inference_settings
    if s is None:
        return None
    if not isinstance(s, MAREXInferenceSettings):
        raise MlsynthConfigError(
            "SPILLCORRECT recomputes MAREX's inference; this design's inference "
            f"settings are {type(s).__name__}.")
    if config.global_test is not None:
        s = s.model_copy(update={"global_test": config.global_test})
    return s


def _estimate(Y: np.ndarray, inp: CorrectionInputs,
              settings: Optional[MAREXInferenceSettings]) -> Tuple[SpillCorrectEstimate, np.ndarray]:
    """The design's estimate and inference on panel ``Y``, and its full control series."""
    syn_t, syn_c = inp.wt @ Y, inp.wc @ Y
    effect = syn_t - syn_c
    post = slice(inp.T0, inp.T0 + inp.post_periods)
    inf = None
    if settings is not None:
        inf = compute_inference(syn_t, syn_c, inp.T0, inp.fit_end, inp.blank_periods,
                                control_outcomes=Y[inp.pool], **settings.inference_kwargs())
    return SpillCorrectEstimate(
        att=float(effect[post].mean()), effect=effect[post], counterfactual=syn_c[post],
        placebo=effect[inp.fit_end:inp.T0],
        ci=None if inf is None else np.asarray(inf.ci)[post],
        p_value=None if inf is None else float(inf.global_p_value),
        inference=inf), syn_c


def _weights(labels, w) -> dict:
    return {u: float(x) for u, x in zip(labels, w) if x > 0}


def _restricted(config, inp: CorrectionInputs):
    """The design re-solved with the affected controls dropped from the control
    pool, its treated set held (MAREX's ``to_be_treated``, and the affected
    units marked as bordering the treated with ``exclude_bordering_donors``)."""
    from ...estimators.scexp import MAREX

    design = config.design
    by_str = {str(u): u for u in config.df[config.unitid].unique()}
    units = [by_str[u] for u in inp.labels]
    treated = [by_str[str(u)] for u in design.composite.treated]
    affected = [by_str[str(u)] for u in config.affected_units]
    border = pd.DataFrame(0.0, index=units, columns=units)
    border.loc[affected, treated] = 1.0
    border.loc[treated, affected] = 1.0
    cfg = {**config.design_config, "df": config.df, "outcome": config.outcome,
           "unitid": config.unitid, "time": config.time, "to_be_treated": treated,
           "adjacency": border, "exclude_bordering_donors": True}
    if config.global_test is not None:
        cfg["global_test"] = config.global_test
    r = MAREX(cfg).fit()
    if set(map(str, r.composite.treated)) != set(map(str, design.composite.treated)):
        raise MlsynthEstimationError(                   # pragma: no cover - to_be_treated holds them
            "SPILLCORRECT: the restricted re-solve treated other units than the design.")
    g = r.globres
    T = inp.T0 + inp.post_periods
    syn_t = np.asarray(g.synthetic_treated, dtype=float)[:T]
    syn_c = np.asarray(g.synthetic_control, dtype=float)[:T]
    effect = syn_t - syn_c
    post = slice(inp.T0, T)
    inf = g.inference
    est = SpillCorrectEstimate(
        att=float(effect[post].mean()), effect=effect[post], counterfactual=syn_c[post],
        placebo=effect[inp.fit_end:inp.T0],
        ci=None if inf is None else np.asarray(inf.ci)[post],
        p_value=None if inf is None else float(inf.global_p_value), inference=inf)
    return est, syn_c, {str(u): float(x) for u, x in r.composite.control.items()}
