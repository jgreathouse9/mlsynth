"""Run the requested proximal estimators and assemble per-method fits.

Dispatches over ``inputs.methods`` -- any of ``PI``, ``PIS``, ``PIPost``,
``SPSC`` -- and packages each into a :class:`ProximalMethodFit`
(counterfactual, gap, ATT, GMM/HAC standard error, pre/post RMSE, donor
weights). Only the requested methods run; the config layer guarantees the
inputs each method needs are present.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .dr import estimate_dr
from .pi import estimate_pi
from .pipw import estimate_pipw
from .pis import estimate_pi_surrogate
from .pipost import estimate_pi_surrogate_post
from .spsc import conformal_intervals, estimate_spsc
from .structures import (
    PI,
    PIPOST,
    PIPW,
    PIS,
    SPSC,
    DR,
    PROXIMALInputs,
    ProximalMethodFit,
)


def _rmse(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values ** 2)))


def _donor_weights(inputs: PROXIMALInputs, alpha: np.ndarray) -> Dict:
    return {name: float(w) for name, w in zip(inputs.donor_names, alpha)}


def _build_fit(
    name: str,
    inputs: PROXIMALInputs,
    counterfactual: np.ndarray,
    time_varying_effect: np.ndarray,
    att_se: float,
    alpha: np.ndarray,
) -> ProximalMethodFit:
    gap = inputs.y - counterfactual
    return ProximalMethodFit(
        name=name,
        counterfactual=counterfactual,
        gap=gap,
        time_varying_effect=time_varying_effect,
        att=float(np.mean(gap[inputs.T0:])),
        att_se=None if att_se is None or not np.isfinite(att_se) else float(att_se),
        pre_rmse=_rmse(gap[: inputs.T0]),
        post_rmse=_rmse(gap[inputs.T0:]),
        alpha_weights=alpha,
        donor_weights=_donor_weights(inputs, alpha),
    )


def _run_pi(inputs: PROXIMALInputs) -> ProximalMethodFit:
    cf, alpha, se = estimate_pi(
        inputs.y, inputs.donor_outcomes, inputs.donor_proxies,
        inputs.T0, inputs.n_post, inputs.T, inputs.bandwidth,
    )
    return _build_fit(PI, inputs, cf, inputs.y - cf, se, alpha)


def _run_pis(inputs: PROXIMALInputs) -> ProximalMethodFit:
    _, effect, alpha, se = estimate_pi_surrogate(
        inputs.y, inputs.donor_outcomes, inputs.donor_proxies,
        inputs.surrogate_proxies, inputs.surrogate_outcomes,
        inputs.T0, inputs.n_post, inputs.T, inputs.bandwidth,
    )
    return _build_fit(PIS, inputs, inputs.y - effect, effect, se, alpha)


def _run_pipost(inputs: PROXIMALInputs) -> ProximalMethodFit:
    _, effect, alpha, se = estimate_pi_surrogate_post(
        inputs.y, inputs.donor_outcomes, inputs.donor_proxies,
        inputs.surrogate_proxies, inputs.surrogate_outcomes,
        inputs.T0, inputs.n_post, inputs.bandwidth,
    )
    return _build_fit(PIPOST, inputs, inputs.y - effect, effect, se, alpha)


def _run_spsc(inputs: PROXIMALInputs) -> ProximalMethodFit:
    cf, gamma, att, se, trend, lam, effect_path, path_se = estimate_spsc(
        inputs.y, inputs.donor_outcomes, inputs.T0,
        detrend=inputs.spsc_detrend,
        spline_df=inputs.spsc_spline_df,
        ridge_lambda=inputs.spsc_lambda,
        basis_degree=inputs.spsc_basis_degree,
        att_degree=inputs.spsc_att_degree,
        detrend_basis=inputs.spsc_detrend_basis,
        detrend_degree=inputs.spsc_detrend_degree,
    )
    gap = inputs.y - cf
    variant = "SPSC-DT" if inputs.spsc_detrend else "SPSC-NoDT"
    if inputs.spsc_basis_degree > 1:
        variant += f"-NP{inputs.spsc_basis_degree}"   # nonparametric sieve degree
    if inputs.spsc_att_degree > 0:
        variant += f"-ATT{inputs.spsc_att_degree}"    # time-varying ATT path
    metadata = {
        "variant": variant,
        "detrend": inputs.spsc_detrend,
        "basis_degree": inputs.spsc_basis_degree,
        "att_degree": inputs.spsc_att_degree,
        "detrend_basis": inputs.spsc_detrend_basis,
        "ridge_lambda": lam,
        "trend": trend,
        "effect_path": effect_path,
        "effect_path_se": path_se,
    }
    if inputs.spsc_conformal:
        metadata["conformal"] = conformal_intervals(
            inputs.y, inputs.donor_outcomes, inputs.T0,
            gamma=gamma, ridge_lambda=lam,
            detrend=inputs.spsc_detrend, spline_df=inputs.spsc_spline_df,
            att_se=se, periods=inputs.spsc_conformal_periods,
            period_se=path_se,
            basis_degree=inputs.spsc_basis_degree,
            att_degree=inputs.spsc_att_degree,
            detrend_basis=inputs.spsc_detrend_basis,
            detrend_degree=inputs.spsc_detrend_degree,
        )
    return ProximalMethodFit(
        name=SPSC,
        counterfactual=cf,
        gap=gap,
        time_varying_effect=gap,
        att=float(att),
        att_se=None if se is None or not np.isfinite(se) else float(se),
        pre_rmse=_rmse(gap[: inputs.T0]),
        post_rmse=_rmse(gap[inputs.T0:]),
        alpha_weights=gamma,
        donor_weights=_donor_weights(inputs, gamma),
        metadata=metadata,
    )


def _run_dr(inputs: PROXIMALInputs) -> ProximalMethodFit:
    cf, alpha, beta, att, se = estimate_dr(
        inputs.y, inputs.donor_outcomes, inputs.donor_proxies, inputs.T0, inputs.bandwidth,
    )
    gap = inputs.y - cf
    return ProximalMethodFit(
        name=DR,
        counterfactual=cf,
        gap=gap,
        time_varying_effect=gap,
        att=float(att),
        att_se=None if se is None or not np.isfinite(se) else float(se),
        pre_rmse=_rmse(gap[: inputs.T0]),
        post_rmse=_rmse(gap[inputs.T0:]),
        alpha_weights=alpha,
        donor_weights={name: float(w) for name, w in zip(inputs.donor_names, alpha[1:])},
        metadata={"intercept": float(alpha[0]), "treatment_bridge_beta": beta,
                  "outcome_bridge_att": float(np.mean(gap[inputs.T0:]))},
    )


def _run_pipw(inputs: PROXIMALInputs) -> ProximalMethodFit:
    beta, att, se = estimate_pipw(
        inputs.y, inputs.donor_outcomes, inputs.donor_proxies, inputs.T0, inputs.bandwidth,
    )
    nan = np.full(inputs.T, np.nan)
    return ProximalMethodFit(
        name=PIPW,
        counterfactual=nan,           # weighting estimator: no imputed trajectory
        gap=nan,
        time_varying_effect=nan,
        att=float(att),
        att_se=None if se is None or not np.isfinite(se) else float(se),
        pre_rmse=float("nan"),
        post_rmse=float("nan"),
        alpha_weights=beta,
        donor_weights={},
        metadata={"treatment_bridge_beta": beta},
    )


_RUNNERS = {
    PI: _run_pi, PIS: _run_pis, PIPOST: _run_pipost, SPSC: _run_spsc,
    DR: _run_dr, PIPW: _run_pipw,
}


def run_proximal(inputs: PROXIMALInputs) -> Dict[str, ProximalMethodFit]:
    """Run each estimator named in ``inputs.methods`` and return the fits.

    Parameters
    ----------
    inputs : PROXIMALInputs
        Prepared panel from :func:`prepare_proximal_inputs`.

    Returns
    -------
    dict
        ``{method_name: ProximalMethodFit}`` for the requested methods, in
        request order.
    """

    fits: Dict[str, ProximalMethodFit] = {}
    for method in inputs.methods:
        runner = _RUNNERS.get(method)
        if runner is not None:
            fits[method] = runner(inputs)
    return fits
