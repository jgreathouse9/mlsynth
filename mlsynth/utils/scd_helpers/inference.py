"""Point estimator and repeated-cross-section inference for SCD.

The point estimator differences the survey-weighted group means and fits a
simplex synthetic control on the pre-period. Inference follows Canen & Song
(2025): individual influence functions give a :math:`\\sqrt{n}`, fixed-``T``
pointwise variance ``sigma2_t``, and the weight confidence set widens the band
by the range of counterfactuals over weights the data cannot rule out. The two
combine by a Bonferroni split (``kappa`` to the weight set, ``alpha - kappa``
to the pointwise term).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from scipy.stats import norm

from .structures import InferenceOperators, SCDInputs
from .weights import confidence_set, lambda_vector, solve_scd_weights


@dataclass(frozen=True)
class PointFit:
    """Point-estimator quantities shared by the report and the inference path."""

    hat_w: np.ndarray
    theta: np.ndarray
    gfull: np.ndarray
    Gfull: np.ndarray
    hG: np.ndarray
    hg: np.ndarray
    lam: np.ndarray


def fit_point(inputs: SCDInputs, differencing: str) -> PointFit:
    """Weighted group means -> differencing -> simplex fit -> effect path."""
    T0 = inputs.T0
    M = inputs.group_means.T                       # (Ttot, K+1), columns = groups
    pre = M[:T0]
    lam = lambda_vector(differencing, T0)
    base = lam @ pre                               # per-group reference level
    gfull = M[:, 0] - base[0]
    Gfull = M[:, 1:] - base[1:][None, :]
    hg, hG = gfull[:T0], Gfull[:T0]
    hat_w = solve_scd_weights(hG, hg)
    theta = gfull - Gfull @ hat_w
    return PointFit(hat_w=hat_w, theta=theta, gfull=gfull, Gfull=Gfull, hG=hG, hg=hg, lam=lam)


def _rc_variance(inputs: SCDInputs, hat_w: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Repeated-cross-section pointwise variance ``sigma2_t`` (all periods).

    Uses the length-``(K+1)`` treated/donor lookup ``c(1, -w)[G]`` -- the
    correct form (the upstream RC code indexes ``w`` at 0 for treated rows,
    which R silently drops, misaligning the donor weights).
    """
    K, n, Ttot, T0 = inputs.K, inputs.n, inputs.Ttot, inputs.T0
    gm, njt = inputs.group_means, inputs.n_jt
    G, t, Y, wt = inputs.G, inputs.t, inputs.Y, inputs.weight
    delta = np.zeros(Ttot)
    delta[:T0] = lam
    wv = np.concatenate([[1.0], -hat_w])           # index by G (0=treated)
    psi = wt * (n / njt[G, t - 1]) * (Y - gm[G, t - 1])
    base2 = (psi * wv[G]) ** 2
    sv = np.zeros(Ttot)
    np.add.at(sv, t - 1, base2)
    total_const = float(np.sum(base2 * delta[t - 1] ** 2))
    return (1.0 / n) * (total_const + sv * (1.0 - 2.0 * delta))


def _weight_variance(inputs: SCDInputs, fit: PointFit) -> tuple:
    """Return ``(precomp, sqrtP, hatV_trace)`` for the confidence-set metric."""
    K, n, T0 = inputs.K, inputs.n, inputs.T0
    gm, njt = inputs.group_means, inputs.n_jt
    G, t, Y, wt = inputs.G, inputs.t, inputs.Y, inputs.weight
    hG, lam = fit.hG, fit.lam
    # B2: orthonormal basis of the K-1 non-null directions of the centering matrix.
    Mm = np.eye(K) - np.ones((K, K)) / K
    evals, U = np.linalg.eigh(Mm)
    B2 = U[:, np.abs(evals) >= 1e-10]
    bar_mu = hG.mean(axis=0)
    wv = np.concatenate([[1.0], -fit.hat_w])
    Vsum = np.zeros((K - 1, K - 1))
    for pt in range(1, T0 + 1):
        idx = t == pt
        Gi = G[idx]
        psi = wt[idx] * (n / njt[Gi, pt - 1]) * (Y[idx] - gm[Gi, pt - 1])
        pw = psi * wv[Gi]
        v_it = float(np.sum(pw ** 2) / n)
        mu = hG[pt - 1]
        Mt = (np.outer(mu, mu) / T0
              - lam[pt - 1] * (np.outer(bar_mu, mu) + np.outer(mu, bar_mu))
              + T0 * lam[pt - 1] ** 2 * np.outer(bar_mu, bar_mu))
        Vsum += v_it * (B2.T @ Mt @ B2)
    hatV = Vsum / T0
    precomp = B2 @ np.linalg.solve(hatV, B2.T)
    ev, Us = np.linalg.eigh(precomp)
    sqrtP = (Us * np.sqrt(np.clip(ev, 0, None))) @ Us.T
    return precomp, sqrtP, float(np.trace(hatV))


def build_inference_operators(inputs: SCDInputs, differencing: str) -> InferenceOperators:
    """Fit the point estimator and assemble the full inference machinery."""
    fit = fit_point(inputs, differencing)
    T0 = inputs.T0
    hat_H = (fit.hG.T @ fit.hG) / T0
    hat_h = (fit.hG.T @ fit.hg) / T0
    precomp, sqrtP, hatV_trace = _weight_variance(inputs, fit)
    sigma2 = _rc_variance(inputs, fit.hat_w, fit.lam)
    se = np.sqrt(np.clip(sigma2, 0, None) / inputs.n)
    return InferenceOperators(
        hat_w=fit.hat_w, theta=fit.theta, gfull=fit.gfull, Gfull=fit.Gfull,
        hat_H=hat_H, hat_h=hat_h, precomp=precomp, sqrtP=sqrtP,
        sigma2=sigma2, se=se, hatV_trace=hatV_trace,
        K=inputs.K, n=inputs.n, T0=T0, Ttot=inputs.Ttot,
    )


def per_period_bands(
    ops: InferenceOperators,
    inputs: SCDInputs,
    alpha: float,
    kappa: float,
    tol: float,
    n_grid: int,
    radius: float,
    random_state: int,
) -> Dict[str, Any]:
    """Assemble per-period confidence bands and the aggregate ATT interval.

    The weight confidence set contributes the min/max counterfactual across
    weights the data cannot reject; the pointwise term adds ``z * se_t`` with
    ``z`` at level ``alpha - kappa`` (the Bonferroni split).
    """
    z = float(norm.ppf(1.0 - (alpha - kappa) / 2.0))
    cs = confidence_set(ops, kappa=kappa, tol=tol, n_grid=n_grid,
                        radius=radius, random_state=random_state)
    if cs.shape[0] == 0:                                     # pragma: no cover
        cs = ops.hat_w[None, :]
    synth = cs @ ops.Gfull.T                                 # (Ncs, Ttot)
    diffs = ops.gfull[None, :] - synth                       # effect under each accepted w
    lower = diffs.min(axis=0) - z * ops.se
    upper = diffs.max(axis=0) + z * ops.se

    T0, T1 = inputs.T0, inputs.T1
    post = slice(T0, ops.Ttot)
    mean_g = ops.gfull[post].mean()
    mean_G = ops.Gfull[post].mean(axis=0)
    mean_eff = mean_g - cs @ mean_G                          # ATT under each accepted w
    se_bar = float(np.sqrt(np.sum(ops.sigma2[post] / ops.n)) / T1)
    att = float(ops.theta[post].mean())
    att_ci = (float(mean_eff.min() - z * se_bar), float(mean_eff.max() + z * se_bar))

    return {
        "z": z,
        "lower": lower,
        "upper": upper,
        "se": ops.se,
        "sigma2": ops.sigma2,
        "hatV_trace": ops.hatV_trace,
        "conf_set_size": int(cs.shape[0]),
        "n_grid": n_grid,
        "att": att,
        "att_std_err": se_bar,
        "att_ci": att_ci,
    }
