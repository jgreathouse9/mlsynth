"""Top-level SDID procedure (Ciccia 2024-style event-study aggregation).

Sequence:

1. :func:`prepare_sdid_inputs` packs the panel into a uniform cohorts dict.
2. :func:`estimate_event_study_sdid` fits all cohorts, aggregates the
   pooled event-study estimator, and runs the placebo procedure.
3. :func:`assemble_results` wraps the raw dictionary into typed frozen
   dataclasses (``SDIDResults`` etc.).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ...config_models import (
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
    MethodDetailsResults,
    TimeSeriesResults,
    WeightsResults,
)
from .event_study import estimate_event_study_sdid
from .setup import prepare_sdid_inputs
from .structures import (
    SDIDCohort,
    SDIDEventEffect,
    SDIDEventStudy,
    SDIDInference,
    SDIDInputs,
    SDIDResults,
)


def _att_p_value(att: float, se: float, placebo: np.ndarray) -> float:
    """Two-sided p-value for the ATT.

    Uses the permutation form ``(k + 1) / (B + 1)`` when a placebo distribution
    is available (the placebo vce), and otherwise the asymptotic-normal
    ``2 * (1 - Phi(|att| / se))`` from the estimated variance (jackknife /
    bootstrap), matching the confidence intervals those methods construct.
    """

    if att is None or np.isnan(att):
        return float("nan")
    if placebo is not None and len(placebo) > 0:
        placebo_arr = np.asarray(placebo, dtype=float)
        return float((np.sum(np.abs(placebo_arr) >= abs(att)) + 1) / (len(placebo_arr) + 1))
    if se is not None and np.isfinite(se) and se > 0:
        from scipy.stats import norm
        return float(2.0 * (1.0 - norm.cdf(abs(att) / se)))
    return float("nan")


def _assemble_cohort(period: int, raw_cohort_summary: Dict[str, Any],
                     cohort_input: Dict[str, Any],
                     per_cohort_full: Dict[str, Any],
                     intercept_adjust: bool = False) -> SDIDCohort:
    """Convert one cohort's raw dict into a typed ``SDIDCohort``.

    ``intercept_adjust`` selects which counterfactual the cohort carries: the
    raw weighted-donor series (default) or the intercept-shifted series that is
    level-matched to the treated unit over the pre-period.
    """

    event_effects = {
        int(ell): SDIDEventEffect(
            ell=int(ell),
            tau=float(payload["tau"]),
            se=float(payload["se"]),
            ci=(float(payload["ci"][0]), float(payload["ci"][1])),
        )
        for ell, payload in raw_cohort_summary.get("event_estimates", {}).items()
    }

    att_ci = raw_cohort_summary.get("att_ci", [float("nan"), float("nan")])
    cf_key = "fitted_counterfactual" if intercept_adjust else "counterfactual"
    unit_w = per_cohort_full.get("unit_weights")
    time_w = per_cohort_full.get("time_weights")
    return SDIDCohort(
        adoption_period=int(period),
        n_treated=len(cohort_input.get("treated_indices", [])),
        n_post=int(cohort_input.get("post_periods", 0)),
        att=float(raw_cohort_summary.get("att", float("nan"))),
        att_se=float(raw_cohort_summary.get("att_se", float("nan"))),
        att_ci=(float(att_ci[0]), float(att_ci[1])),
        event_effects=event_effects,
        actual=np.asarray(per_cohort_full.get("actual"), dtype=float),
        counterfactual=np.asarray(
            per_cohort_full.get(cf_key, per_cohort_full.get("counterfactual")),
            dtype=float,
        ),
        unit_weights=None if unit_w is None else np.asarray(unit_w, dtype=float),
        time_weights=None if time_w is None else np.asarray(time_w, dtype=float),
    )


def _donor_weight_map(inputs: SDIDInputs, cohorts: Dict[int, SDIDCohort]):
    """Map donor names to SDID unit weights for the single-cohort (block) case.

    With one treated cohort the donor columns line up with ``inputs.donor_names``,
    so the unit weights ``omega`` become a readable ``{donor: weight}`` mapping.
    Staggered designs vary the donor set by cohort, so the per-cohort weights are
    left on each ``SDIDCohort.unit_weights`` instead.
    """
    if len(cohorts) != 1:
        return None
    cohort = next(iter(cohorts.values()))
    if cohort.unit_weights is None:
        return None
    names = list(map(str, np.asarray(inputs.donor_names)))
    weights = np.asarray(cohort.unit_weights, dtype=float)
    if len(names) != weights.shape[0]:
        return None
    return {name: float(w) for name, w in zip(names, weights)}


def assemble_results(inputs: SDIDInputs, raw: Dict[str, Any],
                     intercept_adjust: bool = False) -> SDIDResults:
    """Wrap the raw dict from ``estimate_event_study_sdid`` into typed objects."""

    # Pooled event-study estimator (Equation 6).
    pooled = raw.get("pooled_estimates", {})
    if pooled:
        ells = sorted(pooled.keys())
        tau_arr = np.asarray([pooled[e]["tau"] for e in ells], dtype=float)
        se_arr = np.asarray([pooled[e]["se"] for e in ells], dtype=float)
        ci_arr = np.asarray([pooled[e]["ci"] for e in ells], dtype=float)
        event_study = SDIDEventStudy(
            event_times=np.asarray(ells),
            tau=tau_arr,
            se=se_arr,
            ci=ci_arr,
        )
    else:
        event_study = SDIDEventStudy(
            event_times=np.asarray([]),
            tau=np.asarray([]),
            se=np.asarray([]),
            ci=np.asarray([]).reshape(0, 2),
        )

    # Per-cohort objects (Equations 2 and 3).
    cohort_summaries = raw.get("cohort_estimates", {})
    per_cohort_full = raw.get("tau_a_ell", {})
    cohorts = {
        int(period): _assemble_cohort(
            period=int(period),
            raw_cohort_summary=cohort_summaries.get(period, {}),
            cohort_input=inputs.cohorts_dict[int(period)],
            per_cohort_full=per_cohort_full.get(period, {}),
            intercept_adjust=intercept_adjust,
        )
        for period in inputs.cohorts_dict
    }

    # Overall ATT and placebo inference (Equation 7).
    placebo = np.asarray(raw.get("placebo_att_values") or [], dtype=float)
    att = float(raw.get("att", float("nan")))
    se = float(raw.get("att_se", float("nan")))
    ci_pair = raw.get("att_ci", [float("nan"), float("nan")])
    vce_method = raw.get("vce", "placebo")
    inference = SDIDInference(
        att=att,
        se=se,
        ci=(float(ci_pair[0]), float(ci_pair[1])),
        p_value=_att_p_value(att, se, placebo),
        placebo_att=placebo,
        method=vce_method,
        n_placebo=int(len(placebo)),
    )

    # Treated-unit-weighted aggregate trajectory across cohorts -> the flat
    # standardized counterfactual / gap (a single cohort reduces to its path).
    labels = np.asarray(inputs.time_labels)
    if cohorts:
        w = np.array([max(c.n_treated, 1) for c in cohorts.values()], dtype=float)
        actual = np.average(
            np.vstack([np.asarray(c.actual, dtype=float) for c in cohorts.values()]),
            axis=0, weights=w)
        cf = np.average(
            np.vstack([np.asarray(c.counterfactual, dtype=float) for c in cohorts.values()]),
            axis=0, weights=w)
    else:
        actual = cf = np.full(labels.shape[0], np.nan)
    gap = actual - cf
    n_pre = int(inputs.n_pre)
    pre_rmse = (float(np.sqrt(np.mean(gap[:n_pre] ** 2)))
                if n_pre > 0 and np.isfinite(gap[:n_pre]).all() else None)
    std_inference = InferenceResults(
        standard_error=None if np.isnan(se) else float(se),
        ci_lower=None if np.isnan(ci_pair[0]) else float(ci_pair[0]),
        ci_upper=None if np.isnan(ci_pair[1]) else float(ci_pair[1]),
        p_value=None if np.isnan(inference.p_value) else float(inference.p_value),
        method="placebo",
        details=inference,
    )

    return SDIDResults(
        inputs=inputs,
        inference_detail=inference,
        event_study=event_study,
        cohorts=cohorts,
        raw=raw,
        effects=EffectsResults(
            att=None if np.isnan(att) else float(att),
            att_std_err=None if np.isnan(se) else float(se)),
        time_series=TimeSeriesResults(
            observed_outcome=actual,
            counterfactual_outcome=cf,
            estimated_gap=gap,
            time_periods=labels,
            intervention_time=(labels[n_pre] if 0 <= n_pre < labels.shape[0] else None)),
        weights=WeightsResults(
            donor_weights=_donor_weight_map(inputs, cohorts),
            summary_stats={"constraint": "SDID unit + time weights (per cohort)"}),
        fit_diagnostics=FitDiagnosticsResults(rmse_pre=pre_rmse),
        inference=std_inference,
        method_details=MethodDetailsResults(method_name="SDID", is_recommended=True),
    )


def run_sdid(
    df,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    B: int = 500,
    seed: int = 1400,
    vce: str = "placebo",
    intercept_adjust: bool = False,
) -> SDIDResults:
    """End-to-end SDID pipeline producing a typed ``SDIDResults`` object."""

    inputs = prepare_sdid_inputs(
        df=df, outcome=outcome, treat=treat, unitid=unitid, time=time
    )
    raw = estimate_event_study_sdid(
        prepped_event_study_data={"cohorts": inputs.cohorts_dict},
        placebo_iterations=int(B),
        seed=int(seed),
        vce=vce,
    )
    return assemble_results(inputs=inputs, raw=raw,
                            intercept_adjust=intercept_adjust)
