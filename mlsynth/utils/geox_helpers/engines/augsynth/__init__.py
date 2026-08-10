"""Augmented SCM as an GEOX scoring engine.

Augsynth (Ben-Michael, Feller & Rothstein 2021) fitted through
:class:`~mlsynth.utils.bilevel.engine.BilevelSCM`, which is the estimator
GeoLift scores its market selections with. Pairing it with the design harness
lets one panel be scored two ways, so the objective can be varied without
varying the inference and the other way round.

Two details decide whether this reproduces GeoLift's numbers.

Fixed effects. With ``fixed_effects=True`` (augsynth's ``fixedeff``, GeoLift's
default) every unit is demeaned by its own pre-period mean before fitting and
the level is restored by an intercept, so the fit matches shapes and the donor
pool cannot absorb a treated-unit level shift. Combined with a mean-of-units
treated aggregate, that is what reproduces augsynth's effect estimate and its
conformal p-value.

Scaled imbalance. ``scaled_l2`` here is augsynth's ratio
``||X0 w - X1|| / ||X0 w_unif - X1||`` on the centred matching matrices, which
is a different quantity from the SDID engine's ``pre_rmspe`` over the treated
spread. Each engine reports its own estimator's imbalance measure, so the number
is comparable across candidates within a design and not across engines.

Inference. Conformal is the default because GeoLift uses it, and it is available
here and not on the SDID path: the permutation argument needs the pre-period
residuals to be exchangeable across the matching window, which is exactly what
SDID's time weights deny. Placebo reassignment needs only a donor-built
counterfactual, so it is offered on this engine too -- that combination isolates
the objective from the inference.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from .....exceptions import MlsynthConfigError
from .....utils.bilevel.engine import BilevelSCM
from .....utils.bilevel.ridge_inference import (
    conformal_att_interval,
    conformal_pvalue,
)
from .. import Engine, EngineFit, placebo_detection_boundary, placebo_interval
from ...engine import normal_p_value, placebo_sigma


def _window(y, Y0, end: int):
    """The panel truncated at the end of the tested window.

    ``conformal_pvalue`` tests ``resids[pre:]`` -- everything from the split to
    the end of what it is handed. A backtest window ends before the panel does,
    so passing the whole panel tested ``end - pre + 1`` periods plus every
    period after the window: for backtest ``s`` that is ``D + s - 1`` instead of
    ``D``, and only ``s = 1`` coincided. Those trailing periods are no part of
    the pseudo-experiment and carry none of its injected effect.

    Truncating is a no-op for a realized readout, whose window already ends at
    the last period.
    """
    return y[:end + 1], Y0[:end + 1]


def _scaled_l2(A: np.ndarray, B: np.ndarray, w: np.ndarray) -> float:
    """Augsynth scaled L2 imbalance ``||Bw - A|| / ||B w_unif - A||``.

    Unitless: 0 is a perfect pre-fit, 1 is no better than averaging every donor,
    above 1 is worse than the average. ``nan`` when the uniform-weight
    imbalance is zero, which leaves the ratio undefined.
    """
    J = B.shape[1]
    denom = float(np.linalg.norm(B @ np.full(J, 1.0 / J) - A))
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(B @ w - A) / denom)


def fit_once(y, Y0, n_pre: int, start: int, end: int, n_tr: int, *,
             augment: Optional[str] = "ridge", fixed_effects: bool = True,
             **_ignored: Any) -> EngineFit:
    """Fit augmented SCM on the pre-period and predict across the panel."""
    y = np.asarray(y, dtype=float).ravel()
    Y0 = np.asarray(Y0, dtype=float)
    if augment not in ("ridge", None):
        raise MlsynthConfigError(
            f"augment must be 'ridge' or None; got {augment!r}.")

    y_pre, Y0_pre = y[:n_pre], Y0[:n_pre]
    # The simplex path is the fixed-effect SCM already, so it is always
    # demeaned; the flag only switches the ridge path.
    demean = fixed_effects or augment is None
    if demean:
        A, B = y_pre - y_pre.mean(), Y0_pre - Y0_pre.mean(axis=0)
    else:
        A, B = y_pre, Y0_pre

    res = BilevelSCM(augment=augment).fit(A, B)
    w = np.asarray(res.W, dtype=float).ravel()
    intercept = float(np.mean(y_pre - Y0_pre @ w)) if demean else 0.0

    return EngineFit(
        counterfactual=intercept + Y0 @ w,
        donor_weights=w,
        time_weights=None,          # augsynth weights donors alone
        pre_rmspe=float(res.pre_rmspe),
        scaled_l2=_scaled_l2(A, B, w),
        extras={
            "pre_rmspe_lambda": float("nan"),   # augsynth weights donors only
            "intercept": intercept,
            "augment": augment,
            "fixed_effects": bool(fixed_effects),
            "lambda_": (float(res.lambda_) if res.lambda_ is not None else None),
        },
    )


def att(fit: EngineFit, y, start: int, end: int) -> float:
    """Mean gap over the treatment window."""
    y = np.asarray(y, dtype=float).ravel()
    return float(np.mean(y[start:end + 1] - fit.counterfactual[start:end + 1]))


def sweep_p_values(
    fit: EngineFit, y, Y0, n_pre: int, start: int, end: int,
    effect_sizes: Iterable[float], *, inference: str = "conformal",
    ns: int = 1000, n_draws: int = 200, n_tr: int = 1, seed: int = 0,
    conformal_type: str = "block", analytic: bool = True, alpha: float = 0.1,
    finite_sample: bool = False, **_ignored: Any,
) -> Dict[str, Any]:
    """Test every effect size on one backtest.

    The two procedures differ in what leaves the loop. A placebo standard error
    does not depend on the injected effect, so it is drawn once and reused. A
    conformal p-value tests the injected series against permutations of the
    pre-period residuals, so it moves with the effect and is recomputed for each
    one -- which is where this path's cost lives.
    """
    from ...simulate import inject_effect

    y = np.asarray(y, dtype=float).ravel()
    Y0 = np.asarray(Y0, dtype=float)
    tau0 = att(fit, y, start, end)
    baseline = float(np.mean(y[start:end + 1]))
    lam = fit.extras.get("lambda_")
    fixed_effects = bool(fit.extras.get("fixed_effects", True))

    sigma = None
    if inference == "placebo":
        sigma = placebo_sigma(y, Y0, n_pre, start, end, n_draws=n_draws,
                              n_tr=n_tr, seed=seed)
    elif inference != "conformal":
        raise MlsynthConfigError(
            f"inference must be 'conformal' or 'placebo'; got {inference!r}.")

    taus, ps = [], []
    for es in effect_sizes:
        injected = inject_effect(y, start, end, es)
        tau = (tau0 + float(es) * baseline if analytic
               else att(fit, injected, start, end))
        taus.append(tau)
        if inference == "placebo":
            ps.append(normal_p_value(tau, sigma))
        else:
            y_w, Y0_w = _window(injected, Y0, end)
            ps.append(float(conformal_pvalue(
                y_w, Y0_w, n_pre, lambda_=lam, ns=ns, seed=seed,
                conformal_type=conformal_type, fixed_effects=fixed_effects,
                finite_sample=finite_sample)))
    up, down = (placebo_detection_boundary(tau0, baseline, sigma, alpha)
                if inference == "placebo" else (float("nan"), float("nan")))
    return {"tau": taus, "p_value": ps, "sigma": sigma,
            "boundary_up": up, "boundary_down": down}


def point_inference(
    fit: EngineFit, y, Y0, n_pre: int, start: int, end: int, *,
    inference: str = "conformal", ns: int = 1000, n_draws: int = 200,
    n_tr: int = 1, seed: int = 0, conformal_type: str = "block",
    alpha: float = 0.1, finite_sample: bool = False, **_ignored: Any,
) -> Tuple[float, Dict[str, Any]]:
    """One window's test, for the realized readout."""
    y = np.asarray(y, dtype=float).ravel()
    Y0 = np.asarray(Y0, dtype=float)
    tau = att(fit, y, start, end)
    if inference == "placebo":
        sigma = placebo_sigma(y, Y0, n_pre, start, end, n_draws=n_draws,
                              n_tr=n_tr, seed=seed)
        lo, hi = placebo_interval(tau, sigma, alpha)
        return normal_p_value(tau, sigma), {
            "method": "placebo", "sigma": sigma, "att": tau,
            "n_draws": int(n_draws), "ci_lower": lo, "ci_upper": hi}
    if inference != "conformal":
        raise MlsynthConfigError(
            f"inference must be 'conformal' or 'placebo'; got {inference!r}.")
    y_w, Y0_w = _window(y, Y0, end)
    fe = bool(fit.extras.get("fixed_effects", True))
    lam = fit.extras.get("lambda_")
    p = float(conformal_pvalue(
        y_w, Y0_w, n_pre, lambda_=lam, ns=ns, seed=seed,
        conformal_type=conformal_type, fixed_effects=fe,
        finite_sample=finite_sample))
    # The interval is the set of constant effects this same test does not
    # reject, located by bracketing and bisection. A fixed grid does not work:
    # the set is narrow relative to any span wide enough to be safe, so a grid
    # steps over it and reports it empty.
    #
    # Both non-numeric answers are passed through as they come. An infinity is
    # an unbounded side: nothing in that direction is excluded, which is where
    # block conformal lands whenever `alpha` is below its 1/T floor. A NaN pair
    # is an empty set, which is what a multiplicative lift gives once the noise
    # is small enough to resolve it against a constant null. Collapsing either
    # to `None` would make the two indistinguishable.
    lo, hi = conformal_att_interval(
        y_w, Y0_w, n_pre, alpha=alpha, lambda_=lam, ns=ns, seed=seed,
        conformal_type=conformal_type, fixed_effects=fe,
        finite_sample=finite_sample)
    return p, {"method": "conformal", "att": tau, "ns": int(ns),
               "conformal_type": conformal_type,
               "ci_lower": (None if np.isnan(lo) else lo),
               "ci_upper": (None if np.isnan(hi) else hi)}


def detection_boundary(
    fit: EngineFit, y, Y0, n_pre: int, start: int, end: int, *,
    alpha: float = 0.1, inference: str = "conformal", n_draws: int = 200,
    n_tr: int = 1, seed: int = 0, **_ignored: Any,
):
    """Effects at which this backtest starts detecting, in each direction.

    Only defined under placebo inference. A conformal p-value re-permutes
    against the injected series, so it is not analytic in the effect size and
    the crossing would have to be searched for; absent is reported instead of a
    number that is not what it claims.
    """
    if inference != "placebo":
        return float("nan"), float("nan")
    y = np.asarray(y, dtype=float).ravel()
    Y0 = np.asarray(Y0, dtype=float)
    sigma = placebo_sigma(y, Y0, n_pre, start, end, n_draws=n_draws,
                          n_tr=n_tr, seed=seed)
    return placebo_detection_boundary(
        att(fit, y, start, end), float(np.mean(y[start:end + 1])), sigma, alpha)


ENGINE = Engine(name="augsynth", fit_once=fit_once, att=att,
                sweep_p_values=sweep_p_values, point_inference=point_inference,
                detection_boundary=detection_boundary)

__all__ = ["ENGINE", "fit_once", "att", "sweep_p_values",
           "point_inference", "detection_boundary"]
