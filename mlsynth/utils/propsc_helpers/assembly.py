"""Assemble a :class:`PROPSCResults` from the fitted common weights.

Turns the estimator's weight vectors into per-proportion trajectories and
effects, runs the fixed-weights jackknife, and populates the standardized
sub-models for the target proportion so the flat accessors resolve.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from ...config_models import InferenceResults, WeightsResults
from ..results_helpers import build_effect_submodels
from .pipeline import estimate_common_weights, jackknife_se
from .structures import PropscInputs, PropscProportionFit, PROPSCResults

_Z95 = 1.959963984540054


def _proportion_paths(Y, N0, T0, omega, lam, k):
    """Treated-average, synthetic, and gap paths for outcome ``k``."""
    observed = Y[N0:, :, k].mean(axis=0)                 # (T,)
    synth = omega @ Y[:N0, :, k]                         # (T,)
    gap = observed - synth
    offset = float(lam @ gap[:T0]) if len(lam) else 0.0  # SDID intercept shift
    cf = synth + offset
    return observed, cf, observed - cf


def assemble_propsc_results(
    inputs: PropscInputs, method: str, inference: str,
) -> PROPSCResults:
    """Fit the common weights and build the public result container."""
    Y, N0, T0 = inputs.Y, inputs.N0, inputs.T0
    K = Y.shape[2]

    fit = estimate_common_weights(Y, N0, T0, method=method)
    att = fit["estimate"]
    omega = np.asarray(fit["omega"], dtype=float)
    lam = np.asarray(fit["lambda"], dtype=float)

    if inference == "jackknife":
        se = jackknife_se(Y, N0, omega, lam)
    else:
        se = np.full(K, np.nan)

    donor_weights: Dict[str, float] = {
        str(inputs.donor_labels[j]): float(omega[j]) for j in range(N0)
    }

    proportions = []
    for k in range(K):
        observed, cf, gap = _proportion_paths(Y, N0, T0, omega, lam, k)
        se_k = float(se[k])
        if math.isfinite(se_k) and se_k > 0:
            ci = (att[k] - _Z95 * se_k, att[k] + _Z95 * se_k)
            p = 2.0 * (1.0 - _normal_cdf(abs(att[k] / se_k)))
        else:
            ci = (float("nan"), float("nan"))
            p = float("nan")
        proportions.append(PropscProportionFit(
            name=inputs.outcomes[k], att=float(att[k]), se=se_k, ci=ci,
            p_value=p, observed=observed, counterfactual=cf, gap=gap,
            donor_weights=dict(donor_weights),
        ))

    ti = inputs.target_index
    tgt = proportions[ti]
    time_weights = (
        {inputs.time_labels[t]: float(lam[t]) for t in range(T0)}
        if len(lam) else None
    )
    weights = WeightsResults(
        donor_weights=dict(donor_weights),
        time_weights=time_weights,
        summary_stats={"n_donors": N0, "method": method},
    )
    inf = None
    if math.isfinite(tgt.se):
        inf = InferenceResults(
            method=f"propsc_{'jackknife'}", standard_error=tgt.se,
            ci_lower=tgt.ci[0], ci_upper=tgt.ci[1], p_value=tgt.p_value,
            confidence_level=0.95,
        )

    n_post = Y.shape[1] - T0
    submodels = build_effect_submodels(
        observed_outcome=tgt.observed, counterfactual_outcome=tgt.counterfactual,
        n_pre_periods=T0, n_post_periods=n_post,
        time_periods=inputs.time_labels, weights=weights, inference=inf,
        method_name=f"PROPSC-{method.upper()}",
        effects_overrides={"att": tgt.att, "att_std_err": tgt.se},
    )

    return PROPSCResults(
        proportions=tuple(proportions),
        att_vector=att.astype(float),
        se_vector=se.astype(float),
        sum_constraint=float(att.sum()),
        method=method,
        target=inputs.outcomes[ti],
        time_weights=lam.astype(float),
        **submodels,
    )


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
