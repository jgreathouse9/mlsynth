"""OSC orchestrator: regularized nuisance -> orthogonalized ATT -> Series-HAC
fixed-smoothing inference. Mirrors the reference ``OrthoganilzedSCE`` end to end.
"""
from __future__ import annotations

import numpy as np

from .regularized import estimate_delta, estimate_eta
from .orthogonal import orthogonalized_att
from .serieshac import cpe_optimal_h, series_hac_variance, ttest_pvalue, ttest_ci

from ...config_models import (
    BaseEstimatorResults,
    InferenceResults,
    MethodDetailsResults,
)
from ..results_helpers import build_effect_submodels, make_weights_results
from .setup import build_orthsc_inputs


def orthogonalized_sce(pre_y0, pre_yj, Z, post_y0, post_yj, *,
                       alpha: float = 0.05, beta0: float = 0.0,
                       include_constant: bool = True):
    """Run the full Orthogonalized Synthetic Control estimate + inference.

    Returns ``dict`` with ``beta``, ``pvalue``, ``ci`` (lo, hi), ``df`` (smoothing
    K), ``control_weights`` (delta), ``instrument_weights`` (eta).
    """
    T0 = np.atleast_2d(np.asarray(pre_yj, float)).shape[1]
    T1 = np.atleast_2d(np.asarray(post_yj, float)).shape[1]

    delta = estimate_delta(pre_y0, pre_yj, Z, scaled=True,
                           include_constant=include_constant, T1=T1)["delta"]
    eta = estimate_eta(pre_y0, pre_yj, post_y0, post_yj, Z, scaled=True,
                       include_constant=include_constant)["eta"]
    o = orthogonalized_att(pre_y0, pre_yj, Z, post_y0, post_yj, delta, eta,
                           include_constant=include_constant)
    beta, preg, postg = o["beta"], o["preg"], o["postg"]

    h = cpe_optimal_h(preg, p=1, sig=0.05)
    V = series_hac_variance(preg, postg, eta, h)
    n = min(T0, T1)
    return {
        "beta": beta,
        "pvalue": ttest_pvalue(beta, V, h, n, beta0=beta0),
        "ci": ttest_ci(beta, V, h, alpha),
        "df": h,
        "control_weights": delta,
        "instrument_weights": eta,
        "variance": V,
        "se": float(np.sqrt(V / n)),
    }


def run_orthsc(config) -> BaseEstimatorResults:
    """Fit ORTHSC from a config and assemble standardized :class:`BaseEstimatorResults`."""
    ins = build_orthsc_inputs(config)
    out = orthogonalized_sce(
        ins["pre_y0"], ins["pre_yj"], ins["Z"], ins["post_y0"], ins["post_yj"],
        alpha=config.alpha, beta0=config.beta0,
        include_constant=config.include_constant)

    delta = out["control_weights"]
    beta = out["beta"]
    # Synthetic-control counterfactual (the plotted path); the reported ATT is
    # the *orthogonalized* beta, not the naive post-gap mean.
    counterfactual = ins["YJ"].T @ delta                         # (T,)
    y = ins["y"]
    pre = ins["pre"]

    weights = make_weights_results(
        {c: float(w) for c, w in zip(ins["controls"], delta)},
        constraint="simplex (non-negative, sum to 1)",
        extra={
            "instruments": ins["instruments"],
            "instrument_weights": {
                name: float(w) for name, w in zip(
                    ins["instruments"] + (["constant"] if config.include_constant else [])
                    + ["post_moment"], out["instrument_weights"])
            },
        },
    )
    inference = InferenceResults(
        p_value=float(out["pvalue"]),
        ci_lower=float(out["ci"][0]), ci_upper=float(out["ci"][1]),
        standard_error=float(out["se"]),
        confidence_level=1.0 - config.alpha,
        method="orthogonalized SC fixed-smoothing t-test (Fry 2026)",
        details={"smoothing_K": int(out["df"]), "variance": float(out["variance"])},
    )
    submodels = build_effect_submodels(
        observed_outcome=y, counterfactual_outcome=counterfactual,
        n_pre_periods=pre, n_post_periods=int(len(y) - pre),
        time_periods=ins["time_labels"], weights=weights, inference=inference,
        effects_overrides={"att": beta, "att_std_err": float(out["se"])},
        intervention_time=(ins["time_labels"][pre] if pre < len(ins["time_labels"]) else None),
    )
    return BaseEstimatorResults(
        **submodels,
        method_details=MethodDetailsResults(
            method_name="ORTHSC",
            parameters_used={
                "smoothing_K": int(out["df"]),
                "n_controls": len(ins["controls"]),
                "n_instruments": len(ins["instruments"]),
                "include_constant": config.include_constant,
            },
        ),
        additional_outputs={
            "treated_name": ins["treated_name"],
            "controls": ins["controls"],
            "instruments": ins["instruments"],
            "pre_periods": pre,
            "orthogonalized_att": beta,
        },
    )
