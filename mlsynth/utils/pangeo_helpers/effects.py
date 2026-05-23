"""Realized ATT estimation for a PANGEO design with post-period data.

PANGEO is a *design* method: with only pre-treatment history it returns the
supergeo pairs and the treatment/control assignment. If the experiment has
since run -- i.e. the panel carries a ``post_col`` marking post-treatment
periods -- the **same design** (built on the pre-period alone) is scored
against the realized post outcomes here, with inference following the
**Augmented Difference-in-Differences** estimator of Li & Van den Bulte
(2022, *Marketing Science* 42(4):746-767).

The estimator and its inference
-------------------------------

For a treated supergeo aggregate :math:`y^{T}_t` and a control supergeo
aggregate :math:`y^{C}_t`, the counterfactual is the regression projection

.. math::

   y^{T}_t = \\delta_1 + \\delta_2\\, y^{C}_t + \\gamma\\, t + e_t ,
   \\qquad t = 1,\\dots,T_1 ,

fit by least squares on the **pre-period** (the *augmented* DiD: the scale
:math:`\\delta_2` is free rather than forced to 1, and a linear time trend
:math:`\\gamma t` is included). Writing :math:`x_t = (1, y^{C}_t, t)'` and
:math:`\\hat\\delta` for the OLS estimate, the per-period treatment effect is
:math:`\\hat u_t = y^{T}_t - x_t'\\hat\\delta` and the ATT is
:math:`\\hat\\Delta = T_2^{-1}\\sum_{t=T_1+1}^{T}\\hat u_t`.

Li & Van den Bulte show (Propositions 3.1-3.3; Web Appendix C) that
:math:`\\sqrt{T_2}(\\hat\\Delta-\\Delta)\\to N(0,\\Sigma_1+\\Sigma_2)`, which
gives the **prediction-variance** standard error (their C.13)

.. math::

   \\widehat{\\operatorname{Var}}(\\hat\\Delta)
   = \\hat\\omega^2\\Big[\\,\\bar x_{\\text{post}}'
       \\big(\\textstyle\\sum_{t=1}^{T_1} x_t x_t'\\big)^{-1}
       \\bar x_{\\text{post}} \\;+\\; \\tfrac{1}{T_2}\\,\\Big] ,

where :math:`\\bar x_{\\text{post}}` is the post-period mean of :math:`x_t`
and :math:`\\hat\\omega^2` is the residual variance, estimated over the long
pre-period (a Newey-West/Bartlett long-run variance with lag
:math:`\\lfloor T_1^{1/4}\\rfloor` to allow serial correlation; lag 0 is the
i.i.d. case :math:`\\hat\\sigma^2_e=\\hat e'\\hat e/(T_1-k)`). The two terms
are the coefficient-estimation variance (Σ₁) and the post-period averaging
variance (Σ₂). The CI is :math:`\\hat\\Delta\\pm z_{1-\\alpha/2}\\,\\text{SE}`.

Why this estimator suits the supergeo gap
-----------------------------------------

The theory explicitly admits **trend and unit-root (integrated) common
factors** (Li & Van den Bulte Assumptions C2/C3, Prop 3.3). The
augmentation :math:`\\delta_2` makes treated-on-control a *cointegrating*
regression, scaling out a shared integrated factor; the trend term absorbs
deterministic drift; and the prediction-variance term automatically inflates
when the post-period control drifts outside its pre-period range, pricing
the extrapolation uncertainty. The validity condition is that the
**residual** :math:`e_t` be (weakly dependent) stationary -- which the
augmentation + trend deliver. The arm and **program** ATTs apply this
single-treated-unit estimator to the treated-size-weighted supergeo
aggregate at each level; the program number is the headline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from .parallelism import _wavg


@dataclass(frozen=True)
class AttEstimate:
    """An augmented-DiD ATT (Li & Van den Bulte 2022) at one level.

    Attributes
    ----------
    level : str
        ``"program"`` or an arm label.
    att : float
        Augmented-DiD ATT on the (population-weighted) outcome scale.
    att_pct : float
        ATT as a percentage of the post-period counterfactual level.
    baseline : float
        Mean post-period counterfactual outcome used for ``att_pct``
        (the predicted treated series absent treatment).
    se : float
        Prediction-variance standard error (Li & Van den Bulte C.13).
    ci_lower, ci_upper : float
        Confidence interval for the absolute ATT.
    ci_lower_pct, ci_upper_pct : float
        The same interval as a percentage of baseline.
    p_value : float
        Two-sided normal p-value for the null of no effect.
    n_post : int
        Number of post-treatment periods averaged.
    scale : float
        Fitted augmentation coefficient :math:`\\hat\\delta_2` (1.0 if the
        augmentation is disabled, i.e. plain DiD).
    observed : np.ndarray
        Observed treated supergeo aggregate over pre + post periods.
    counterfactual : np.ndarray
        Augmented-DiD counterfactual prediction of the treated aggregate over
        the same periods; the gap in the post window is the per-period effect.
    """

    level: str
    att: float
    att_pct: float
    baseline: float
    se: float
    ci_lower: float
    ci_upper: float
    ci_lower_pct: float
    ci_upper_pct: float
    p_value: float
    n_post: int
    scale: float
    observed: np.ndarray = field(default_factory=lambda: np.empty(0))
    counterfactual: np.ndarray = field(default_factory=lambda: np.empty(0))


@dataclass(frozen=True)
class PangeoEffects:
    """Realized ATT for a PANGEO design scored against post-period data.

    Attributes
    ----------
    program : AttEstimate
        Headline program-level ATT (pooled across all arms).
    arms : dict
        ``{arm_label: AttEstimate}``.
    pair_att : dict
        ``{arm_label: [per-pair ATT, ...]}`` (point estimates).
    n_post : int
        Number of post periods.
    weighted : bool
        Whether a population weight was used in the aggregation.
    alpha : float
        Significance level for the intervals.
    """

    program: AttEstimate
    arms: Dict[Any, AttEstimate]
    pair_att: Dict[Any, List[float]]
    n_post: int
    weighted: bool
    alpha: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Tidy table of the program and per-arm ATT estimates."""
        rows = [_att_row(self.program)]
        rows += [_att_row(c) for c in self.arms.values()]
        return pd.DataFrame(rows).set_index("level")


def _att_row(e: AttEstimate) -> Dict[str, Any]:
    return {
        "level": e.level, "att": e.att, "att_pct": e.att_pct, "se": e.se,
        "ci_lower": e.ci_lower, "ci_upper": e.ci_upper,
        "ci_lower_pct": e.ci_lower_pct, "ci_upper_pct": e.ci_upper_pct,
        "p_value": e.p_value, "scale": e.scale, "n_post": e.n_post,
    }


def _hac_lag(n: int) -> int:
    """Newey-West truncation lag, Li & Van den Bulte's ``O(T^{1/4})`` rule."""
    return max(0, int(np.floor(n ** 0.25)))


def _lr_variance(resid: np.ndarray, k: int) -> float:
    """Newey-West/Bartlett long-run variance of the (mean-zero) residuals.

    Lag 0 reduces to the degrees-of-freedom-corrected residual variance
    ``e'e/(T-k)`` -- the i.i.d. case in Li & Van den Bulte C.13.
    """
    n = resid.size
    g0 = float(resid @ resid) / max(n - k, 1)
    lag = _hac_lag(n)
    if lag == 0 or n <= k + 1:
        return g0
    omega = g0
    for j in range(1, lag + 1):
        w = 1.0 - j / (lag + 1)
        gj = float(resid[j:] @ resid[:-j]) / max(n - k, 1)
        omega += 2.0 * w * gj
    return max(omega, 1e-18)


def adid_counterfactual(
    YT: np.ndarray, YC: np.ndarray, n_pre: int,
    augment: bool = True, trend: bool = True,
) -> np.ndarray:
    """Treated-series counterfactual from the (augmented) DiD fit.

    Fit ``yT = d1 [+ d2*yC] [+ g*t]`` on the first ``n_pre`` periods and
    return the predicted *treated* trajectory over all periods -- the line
    PANGEO plots against the observed treated aggregate. For plain DiD the
    counterfactual is ``yC + (d1 [+ g*t])``; for augmented DiD it is the
    projection ``d1 + d2*yC [+ g*t]`` directly.
    """
    n = YT.size
    t = np.arange(n, dtype=float)
    cols = [np.ones(n)] + ([YC] if augment else []) + ([t] if trend else [])
    X = np.column_stack(cols)
    y = YT if augment else YT - YC
    XE = X[:n_pre]
    try:
        beta = np.linalg.solve(XE.T @ XE, XE.T @ y[:n_pre])
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XE, y[:n_pre], rcond=None)[0]
    yhat = X @ beta
    return yhat if augment else YC + yhat


def _adid(
    YT: np.ndarray, YC: np.ndarray, n_pre: int, n_post: int,
    augment: bool, trend: bool, alpha: float,
) -> Dict[str, float]:
    """Li & Van den Bulte augmented-DiD ATT + prediction-variance CI.

    Regress the treated aggregate on a constant, the (optionally augmented)
    control aggregate and a (optional) linear trend over the pre-period;
    project the counterfactual onto the post-period; return the ATT, SE, CI
    and p-value.
    """
    t = np.arange(n_pre + n_post, dtype=float)
    if augment:
        cols = [np.ones_like(t), YC]
        y = YT.astype(float)
    else:                                   # plain DiD: regress the gap
        cols = [np.ones_like(t)]
        y = (YT - YC).astype(float)
    if trend:
        cols.append(t)
    X = np.column_stack(cols)
    k = X.shape[1]

    Xpre, ypre = X[:n_pre], y[:n_pre]
    Xpost, ypost = X[n_pre:], y[n_pre:]
    beta, *_ = np.linalg.lstsq(Xpre, ypre, rcond=None)
    e = ypre - Xpre @ beta                  # pre-period residuals
    u = ypost - Xpost @ beta                # per-period post effects
    att = float(u.mean())

    omega2 = _lr_variance(e, k)             # pre-period (long-run) residual var
    S_pre = Xpre.T @ Xpre
    xbar = Xpost.mean(axis=0)
    pred_term = float(xbar @ np.linalg.solve(S_pre, xbar))   # Σ₁ contribution
    var = omega2 * (pred_term + 1.0 / n_post)                # + Σ₂ contribution
    se = float(np.sqrt(max(var, 0.0)))

    z = norm.ppf(1.0 - alpha / 2.0)
    if se > 0:
        p_value = float(2.0 * (1.0 - norm.cdf(abs(att) / se)))
    else:
        p_value = 0.0 if att != 0 else 1.0
    scale = float(beta[1]) if augment else 1.0
    # Percent ATT is relative to the post-period counterfactual prediction
    # (cf. mlsynth.utils.resultutils.effects.calculate): the counterfactual
    # treated series is y^T_t - u_t, so its post mean is mean(y^T_post) - att.
    baseline = float(YT[n_pre:].mean() - att)
    cf = X @ beta if augment else (YC + X @ beta)   # treated counterfactual
    return {"att": att, "se": se, "ci_lower": att - z * se,
            "ci_upper": att + z * se, "p_value": p_value, "scale": scale,
            "baseline": baseline, "counterfactual": cf}


def _pair_records(arm_design, pos, Y_post, weights):
    """Per-pair full-pre treated/control aggregates, post values and weight."""
    recs = []
    for p in arm_design.pairs:
        t_idx = [pos[u] for u in p.treatment]
        c_idx = [pos[u] for u in p.control]
        recs.append({
            "YT_pre": np.asarray(p.treatment_mean, dtype=float),
            "YC_pre": np.asarray(p.control_mean, dtype=float),
            "YT_post": _wavg(Y_post, t_idx, weights),
            "YC_post": _wavg(Y_post, c_idx, weights),
            "weight": (float(weights[t_idx].sum()) if weights is not None
                       else len(t_idx)),
        })
    return recs


def _stack(recs):
    """Stack a level's pairs into ``(P, .)`` matrices and normalised weights."""
    w = np.array([r["weight"] for r in recs], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(len(recs), 1.0 / len(recs))
    YT = np.hstack([np.vstack([r["YT_pre"] for r in recs]),
                    np.vstack([r["YT_post"] for r in recs])])
    YC = np.hstack([np.vstack([r["YC_pre"] for r in recs]),
                    np.vstack([r["YC_post"] for r in recs])])
    return w, YT, YC


def _design_matrix(YC: np.ndarray, n_total: int, augment: bool, trend: bool):
    """Stack the ADID regressors ``[1, (y^C), (t)]`` for one or many series.

    ``YC`` is ``(n_total,)`` for a single series or ``(P, n_total)`` for a
    batch; returns ``X`` of shape ``(..., n_total, k)`` and the response
    transform (``y = y^T`` when augmenting, else the gap ``y^T - y^C``).
    """
    t = np.arange(n_total, dtype=float)
    if YC.ndim == 1:
        cols = [np.ones(n_total)] + ([YC] if augment else []) \
            + ([t] if trend else [])
        return np.column_stack(cols)
    P = YC.shape[0]
    ones = np.ones((P, n_total))
    tt = np.broadcast_to(t, (P, n_total))
    cols = [ones] + ([YC] if augment else []) + ([tt] if trend else [])
    return np.stack(cols, axis=2)


def _adid_att_batch(YT, YC, n_pre, augment, trend) -> np.ndarray:
    """Vectorised ATT point estimate for a batch of ``(P, T)`` pairs.

    A single batched OLS (``einsum`` Gram matrices + one ``solve``) -- no
    Python loop over pairs. Matches :func:`_adid` up to the equivalence of
    ``lstsq`` and ``solve`` for full-rank designs.
    """
    X = _design_matrix(YC, YT.shape[1], augment, trend)     # (P, T, k)
    y = YT if augment else YT - YC                          # (P, T)
    Xpre, ypre = X[:, :n_pre, :], y[:, :n_pre]
    Xpost, ypost = X[:, n_pre:, :], y[:, n_pre:]
    XtX = np.einsum("ptk,ptj->pkj", Xpre, Xpre)
    Xty = np.einsum("ptk,pt->pk", Xpre, ypre)
    beta = np.linalg.solve(XtX, Xty[..., None])[..., 0]     # (P, k)
    u = ypost - np.einsum("ptk,pk->pt", Xpost, beta)
    return u.mean(axis=1)


def _aggregate(level, recs, alpha, augment, trend) -> AttEstimate:
    w, MT, MC = _stack(recs)
    n_pre = recs[0]["YT_pre"].size
    n_post = recs[0]["YT_post"].size
    YT = w @ MT                              # treated-size-weighted aggregates
    YC = w @ MC

    r = _adid(YT, YC, n_pre, n_post, augment, trend, alpha)
    baseline = r["baseline"]                 # mean post-period counterfactual
    pct = (lambda x: x / baseline * 100.0) if abs(baseline) > 1e-12 \
        else (lambda x: float("nan"))
    return AttEstimate(
        level=level, att=r["att"], att_pct=pct(r["att"]), baseline=baseline,
        se=r["se"], ci_lower=r["ci_lower"], ci_upper=r["ci_upper"],
        ci_lower_pct=pct(r["ci_lower"]), ci_upper_pct=pct(r["ci_upper"]),
        p_value=r["p_value"], n_post=int(n_post), scale=r["scale"],
        observed=YT, counterfactual=r["counterfactual"],
    )


def compute_pangeo_effects(
    results, inputs, Y_post: np.ndarray, *, alpha: float = 0.05,
    augment: bool = True, trend: bool = True,
) -> PangeoEffects:
    """Augmented-DiD ATT (Li & Van den Bulte 2022) for a design scored on
    post outcomes, at the program and arm levels.

    Parameters
    ----------
    results : PangeoResults
        The frozen design (pairs + assignment) built on the pre-period.
    inputs : PangeoInputs
        Pre-period inputs (supplies unit order and population weights).
    Y_post : np.ndarray
        Post-period outcomes, shape ``(N, T_post)``, rows aligned with
        ``inputs.unit_names``.
    alpha : float
        Significance level for the CIs / p-values.
    augment : bool
        Free augmentation coefficient ``delta_2`` on the control aggregate
        (the *augmented* DiD). ``False`` forces ``delta_2 = 1`` (plain DiD).
    trend : bool
        Include a linear time-trend regressor.
    """
    pos = {u: i for i, u in enumerate(inputs.unit_names)}
    weights = inputs.weights
    n_post = int(Y_post.shape[1])

    by_arm = {arm: _pair_records(d, pos, Y_post, weights)
              for arm, d in results.arm_designs.items()}
    all_recs = [r for recs in by_arm.values() for r in recs]

    pair_att: Dict[Any, List[float]] = {}
    for arm, recs in by_arm.items():
        _, MT, MC = _stack(recs)
        pair_att[arm] = _adid_att_batch(
            MT, MC, recs[0]["YT_pre"].size, augment, trend).tolist()

    program = _aggregate("program", all_recs, alpha, augment, trend)
    arms = {arm: _aggregate(str(arm), recs, alpha, augment, trend)
            for arm, recs in by_arm.items()}

    return PangeoEffects(
        program=program, arms=arms, pair_att=pair_att, n_post=n_post,
        weighted=weights is not None, alpha=alpha,
        metadata={
            "estimator": "Augmented DiD (Li & Van den Bulte 2022); "
                         "treated-size-weighted supergeo aggregate per level",
            "inference": "prediction-variance SE (C.13) with Newey-West "
                         "long-run residual variance",
            "augment": augment, "trend": trend,
        },
    )
