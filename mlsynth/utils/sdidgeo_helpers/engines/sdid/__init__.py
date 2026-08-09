"""Synthetic DiD as an SDIDGEO scoring engine.

A thin adapter over :mod:`mlsynth.utils.sdidgeo_helpers.engine`: unit weights
over donors and time weights over pre-periods (Arkhangelsky, Athey, Hirshberg,
Imbens and Wager 2021), tested against the placebo standard error of their
Algorithm 4. Jackknife and bootstrap are undefined for the single treated series
a candidate region collapses to, which is why the placebo procedure is the one
wired here.

The adapter adds no arithmetic. Every value it returns comes from the functions
the pipeline called directly before the seam existed, so introducing the seam
leaves the design's answer bit-identical.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from .. import Engine, EngineFit
from ...engine import (
    normal_p_value,
    placebo_sigma,
    sdid_att,
    sdid_fit_once,
)


def fit_once(y, Y0, n_pre: int, start: int, end: int, n_tr: int) -> EngineFit:
    """Fit SDID on the pre-period and predict across the panel."""
    fit = sdid_fit_once(y, Y0, n_pre, start, end, n_tr=n_tr)
    return EngineFit(
        counterfactual=fit.counterfactual,
        donor_weights=fit.omega,
        time_weights=fit.lam,
        pre_rmspe=fit.pre_rmspe,
        scaled_l2=fit.scaled_l2,
        extras={"zeta": fit.zeta, "bias_correction": fit.bias_correction},
    )


def att(fit: EngineFit, y, start: int, end: int) -> float:
    """Mean gap over the treatment window."""
    y = np.asarray(y, dtype=float).ravel()
    return float(np.mean(y[start:end + 1] - fit.counterfactual[start:end + 1]))


def sweep_p_values(
    fit: EngineFit, y, Y0, n_pre: int, start: int, end: int,
    effect_sizes: Iterable[float], *, n_draws: int = 200, n_tr: int = 1,
    seed: int = 0, analytic: bool = True,
) -> Dict[str, Any]:
    """Test every effect size on one backtest.

    The placebo standard error does not depend on the injected effect, so it is
    drawn once and reused across the grid. With ``analytic`` the ATT is shifted
    by ``effect_size x mean(treated_post)`` instead of refitting: the fit uses
    pre-period data alone, so injecting into the post window moves the ATT by
    exactly that amount and leaves the counterfactual untouched.
    """
    y = np.asarray(y, dtype=float).ravel()
    Y0 = np.asarray(Y0, dtype=float)
    tau0 = att(fit, y, start, end)
    baseline = float(np.mean(y[start:end + 1]))
    sigma = placebo_sigma(y, Y0, n_pre, start, end, n_draws=n_draws,
                          n_tr=n_tr, seed=seed)

    taus, ps = [], []
    for es in effect_sizes:
        if analytic:
            tau = tau0 + float(es) * baseline
        else:
            from ...simulate import inject_effect      # local: avoids a cycle
            tau = att(fit, inject_effect(y, start, end, es), start, end)
        taus.append(tau)
        ps.append(normal_p_value(tau, sigma))
    return {"tau": taus, "p_value": ps, "sigma": sigma}


def point_inference(
    fit: EngineFit, y, Y0, n_pre: int, start: int, end: int, *,
    n_draws: int = 200, n_tr: int = 1, seed: int = 0,
) -> Tuple[float, Dict[str, Any]]:
    """One window's test, for the realized readout."""
    y = np.asarray(y, dtype=float).ravel()
    Y0 = np.asarray(Y0, dtype=float)
    tau = att(fit, y, start, end)
    sigma = placebo_sigma(y, Y0, n_pre, start, end, n_draws=n_draws,
                          n_tr=n_tr, seed=seed)
    return normal_p_value(tau, sigma), {
        "method": "placebo",
        "sigma": sigma,
        "att": tau,
        "n_draws": int(n_draws),
    }


ENGINE = Engine(name="sdid", fit_once=fit_once, att=att,
                sweep_p_values=sweep_p_values, point_inference=point_inference)

__all__ = ["ENGINE", "fit_once", "att", "sweep_p_values", "point_inference"]
