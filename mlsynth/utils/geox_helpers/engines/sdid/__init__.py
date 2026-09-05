"""Synthetic DiD as an GEOX scoring engine.

A thin adapter over :mod:`mlsynth.utils.geox_helpers.engine`: unit weights
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

from .. import Engine, EngineFit, placebo_detection_boundary, placebo_interval
from .....exceptions import MlsynthConfigError
from ...engine import (
    normal_p_value,
    placebo_sigma,
    sdid_att,
    sdid_fit_once,
)


def _lambda_dispersion(y, fit, n_pre: int) -> float:
    """Pre-period gap dispersion under the fit's own time weights.

    SDID matches a lambda-weighted average of pre-periods, so the unweighted
    RMSE covers periods the fit set aside. The gap is centred on its
    lambda-weighted mean, which the level term absorbs, so this measures what
    the fit leaves once the level is out.
    """
    lam = np.asarray(fit.lam, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    gap = y[:n_pre] - fit.counterfactual[:n_pre]
    if lam.shape[0] != n_pre or not np.isfinite(gap).all():
        return float("nan")
    centred = gap - float(lam @ gap)
    return float(np.sqrt(lam @ (centred ** 2)))


def fit_once(y, Y0, n_pre: int, start: int, end: int, n_tr: int,
             **_ignored: Any) -> EngineFit:
    """Fit SDID on the pre-period and predict across the panel.

    Settings belonging to another engine are ignored, so the pipeline can pass
    one bundle whichever engine is running.
    """
    fit = sdid_fit_once(y, Y0, n_pre, start, end, n_tr=n_tr)
    return EngineFit(
        counterfactual=fit.counterfactual,
        donor_weights=fit.omega,
        time_weights=fit.lam,
        pre_rmspe=fit.pre_rmspe,
        scaled_l2=fit.scaled_l2,
        extras={"zeta": fit.zeta, "bias_correction": fit.bias_correction,
                "pre_rmspe_lambda": _lambda_dispersion(y, fit, n_pre)},
    )


def att(fit: EngineFit, y, start: int, end: int) -> float:
    """Mean gap over the treatment window."""
    y = np.asarray(y, dtype=float).ravel()
    return float(np.mean(y[start:end + 1] - fit.counterfactual[start:end + 1]))


def sweep_p_values(
    fit: EngineFit, y, Y0, n_pre: int, start: int, end: int,
    effect_sizes: Iterable[float], *, n_draws: int = 200, n_tr: int = 1,
    seed: int = 0, analytic: bool = True, inference: str = "placebo",
    alpha: float = 0.1, **_ignored: Any,
) -> Dict[str, Any]:
    """Test every effect size on one backtest.

    The placebo standard error does not depend on the injected effect, so it is
    drawn once and reused across the grid. With ``analytic`` the ATT is shifted
    by ``effect_size x mean(treated_post)`` instead of refitting: the fit uses
    pre-period data alone, so injecting into the post window moves the ATT by
    exactly that amount and leaves the counterfactual untouched.

    Keyword arguments belonging to another engine are ignored, so the pipeline
    can pass one settings bundle whichever engine is running.
    """
    if inference != "placebo":
        raise MlsynthConfigError(
            f"the sdid engine supports inference='placebo' only; got "
            f"{inference!r}. Conformal inference assumes the pre-period "
            "residuals are exchangeable, and SDID's time weights exist to say "
            "they are not.")
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
    up, down = placebo_detection_boundary(tau0, baseline, sigma, alpha)
    return {"tau": taus, "p_value": ps, "sigma": sigma, "tau0": tau0,
            "boundary_up": up, "boundary_down": down}


def point_inference(
    fit: EngineFit, y, Y0, n_pre: int, start: int, end: int, *,
    n_draws: int = 200, n_tr: int = 1, seed: int = 0, alpha: float = 0.1,
    inference: str = "placebo", **_ignored: Any,
) -> Tuple[float, Dict[str, Any]]:
    """One window's test and interval, for the realized readout."""
    y = np.asarray(y, dtype=float).ravel()
    Y0 = np.asarray(Y0, dtype=float)
    tau = att(fit, y, start, end)
    sigma = placebo_sigma(y, Y0, n_pre, start, end, n_draws=n_draws,
                          n_tr=n_tr, seed=seed)
    lo, hi = placebo_interval(tau, sigma, alpha)
    return normal_p_value(tau, sigma), {
        "method": "placebo",
        "sigma": sigma,
        "att": tau,
        "n_draws": int(n_draws),
        "ci_lower": lo,
        "ci_upper": hi,
    }


def detection_boundary(
    fit: EngineFit, y, Y0, n_pre: int, start: int, end: int, *,
    alpha: float = 0.1, n_draws: int = 200, n_tr: int = 1, seed: int = 0,
    **_ignored: Any,
):
    """Effects at which this backtest starts detecting, in each direction."""
    y = np.asarray(y, dtype=float).ravel()
    Y0 = np.asarray(Y0, dtype=float)
    sigma = placebo_sigma(y, Y0, n_pre, start, end, n_draws=n_draws,
                          n_tr=n_tr, seed=seed)
    return placebo_detection_boundary(
        att(fit, y, start, end), float(np.mean(y[start:end + 1])), sigma, alpha)


ENGINE = Engine(name="sdid", fit_once=fit_once, att=att,
                sweep_p_values=sweep_p_values, point_inference=point_inference,
                detection_boundary=detection_boundary)

__all__ = ["ENGINE", "fit_once", "att", "sweep_p_values",
           "point_inference", "detection_boundary"]
