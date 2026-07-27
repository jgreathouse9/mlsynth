"""Assemble a :class:`COMPSCResults` from the fitted compositional weights.

Turns the log-ratio fit into per-category trajectories and effects, runs the
Aitchison placebo test, computes the baseline-sensitivity diagnostic, and
populates the standardized sub-models for the target category so the flat
accessors resolve.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ...config_models import InferenceResults, WeightsResults
from ..results_helpers import build_effect_submodels
from .pipeline import fit_compsc, geometry_sensitivity, placebo_test
from .structures import (
    CompscCategoryFit,
    CompscInputs,
    CompscPlacebo,
    COMPSCResults,
)


def assemble_compsc_results(
    inputs: CompscInputs, geometry: str, inference: str, placebo_screen: float,
) -> COMPSCResults:
    """Fit the compositional weights and build the public result container."""
    P, T0 = inputs.P, inputs.T0
    K = P.shape[2]
    base = inputs.baseline_index

    fit = fit_compsc(P, T0, geometry=geometry, base=base)
    w = fit["weights"]
    obs = fit["observed"]
    cf = fit["counterfactual"]
    gaps = fit["share_effects"]

    donor_weights: Dict[str, float] = {
        str(inputs.donor_labels[j]): float(w[j])
        for j in range(len(inputs.donor_labels))
    }

    categories = tuple(
        CompscCategoryFit(
            name=inputs.outcomes[k],
            att=float(gaps[T0:, k].mean()),
            observed=obs[:, k].copy(),
            counterfactual=cf[:, k].copy(),
            gap=gaps[:, k].copy(),
        )
        for k in range(K)
    )
    att_vector = np.array([c.att for c in categories], dtype=float)

    placebo = None
    inf = None
    if inference == "placebo":
        pt = placebo_test(P, T0, geometry=geometry, base=base,
                          screen=placebo_screen)
        labels = [inputs.treated_label] + list(inputs.donor_labels)
        placebo = CompscPlacebo(
            p_value=pt["p_value"],
            treated_ratio=pt["treated_ratio"],
            ratios={lab: float(r) for lab, r in zip(labels, pt["ratios"])},
            pre_rmspe={lab: float(r) for lab, r in zip(labels, pt["pre_rmspe"])},
            retained=tuple(lab for lab, keep in zip(labels, pt["retained"]) if keep),
            n_retained=pt["n_retained"],
            screen=placebo_screen,
        )
        inf = InferenceResults(
            method="compsc_placebo_aitchison",
            p_value=pt["p_value"],
            details={
                "test_statistic": "post/pre Aitchison RMSPE ratio",
                "treated_ratio": pt["treated_ratio"],
                "n_retained": pt["n_retained"],
                "pre_fit_screen": placebo_screen,
            },
        )

    sensitivity = geometry_sensitivity(P, T0, w)

    ti = inputs.target_index
    tgt = categories[ti]
    weights = WeightsResults(
        donor_weights=dict(donor_weights),
        summary_stats={
            "n_donors": len(inputs.donor_labels),
            "geometry": geometry,
            "n_stacked_observations": (K - 1) * T0,
        },
    )
    submodels = build_effect_submodels(
        observed_outcome=tgt.observed, counterfactual_outcome=tgt.counterfactual,
        n_pre_periods=T0, n_post_periods=P.shape[1] - T0,
        time_periods=inputs.time_labels, weights=weights, inference=inf,
        method_name=f"COMPSC-{geometry.upper()}",
        effects_overrides={"att": tgt.att},
        additional_metrics={
            "aitchison_rmspe_pre": fit["rmspe_pre"],
            "aitchison_rmspe_post": fit["rmspe_post"],
            "share_rmspe_pre": fit["share_rmspe_pre"],
        },
    )

    return COMPSCResults(
        categories=categories,
        observed_composition=obs,
        counterfactual_composition=cf,
        share_effects=gaps,
        log_ratio_effects=fit["log_ratio_effects"],
        aitchison_distance=fit["aitchison_distance"],
        att_vector=att_vector,
        sum_constraint=float(att_vector.sum()),
        pre_treatment_aitchison_rmspe=fit["rmspe_pre"],
        pre_treatment_share_rmspe=fit["share_rmspe_pre"],
        placebo=placebo,
        geometry=geometry,
        baseline=(inputs.outcomes[base] if geometry == "alr" else None),
        target=inputs.outcomes[ti],
        baseline_sensitivity=(None if sensitivity is None
                              else sensitivity["max_l1_weight_shift"]),
        **submodels,
    )
