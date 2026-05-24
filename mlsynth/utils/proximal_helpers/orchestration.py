"""Run the proximal methods and assemble per-method fits.

Drives the up-to-three proximal estimators on a prepared
:class:`PROXIMALInputs` and packages each into a
:class:`ProximalMethodFit` (counterfactual, gap, ATT, GMM/HAC standard
error, pre/post RMSE, donor weights). PI always runs; PIS and PIPost run
only when surrogate data is available.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .estimation import (
    estimate_pi,
    estimate_pi_surrogate,
    estimate_pi_surrogate_post,
)
from .structures import (
    PI,
    PIPOST,
    PIS,
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


def run_proximal(inputs: PROXIMALInputs) -> Dict[str, ProximalMethodFit]:
    """Run every applicable proximal method on ``inputs``.

    Parameters
    ----------
    inputs : PROXIMALInputs
        Prepared panel from :func:`prepare_proximal_inputs`.

    Returns
    -------
    dict
        ``{method_name: ProximalMethodFit}`` with ``"PI"`` always present
        and ``"PIS"``/``"PIPost"`` present when surrogates are configured.
    """

    fits: Dict[str, ProximalMethodFit] = {}

    # --- PI ---
    pi_counterfactual, pi_alpha, pi_se = estimate_pi(
        inputs.y,
        inputs.donor_outcomes,
        inputs.donor_proxies,
        inputs.T0,
        inputs.n_post,
        inputs.T,
        inputs.bandwidth,
    )
    fits[PI] = _build_fit(
        PI, inputs, pi_counterfactual, inputs.y - pi_counterfactual, pi_se, pi_alpha
    )

    if not inputs.has_surrogates:
        return fits

    # --- PIS ---
    _, pis_effect, pis_alpha, pis_se = estimate_pi_surrogate(
        inputs.y,
        inputs.donor_outcomes,
        inputs.donor_proxies,
        inputs.surrogate_proxies,
        inputs.surrogate_outcomes,
        inputs.T0,
        inputs.n_post,
        inputs.T,
        inputs.bandwidth,
    )
    fits[PIS] = _build_fit(
        PIS, inputs, inputs.y - pis_effect, pis_effect, pis_se, pis_alpha
    )

    # --- PIPost ---
    _, pipost_effect, pipost_alpha, pipost_se = estimate_pi_surrogate_post(
        inputs.y,
        inputs.donor_outcomes,
        inputs.donor_proxies,
        inputs.surrogate_proxies,
        inputs.surrogate_outcomes,
        inputs.T0,
        inputs.n_post,
        inputs.bandwidth,
    )
    fits[PIPOST] = _build_fit(
        PIPOST, inputs, inputs.y - pipost_effect, pipost_effect, pipost_se, pipost_alpha
    )

    return fits
