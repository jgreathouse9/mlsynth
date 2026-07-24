"""Inference for RRSC (He, Li, Shi & Miao 2026).

* :func:`run_cbb` -- circular block bootstrap (reference ``CBB.ci``). The
  per-unit bootstrap variance drives a normal (Wald) confidence interval and
  p-value, as the authors' analyses report.
* :func:`run_conformal` -- the moving-block (cyclic-shift) permutation test of
  Chernozhukov, Wuthrich & Zhu (2021) on the per-period counterfactual
  residuals; valid with a short post-period.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import norm

from .pipeline import counterfactual_panel, point_effects


def cbb_sampling(block_length: int, M: int, rng) -> np.ndarray:
    """Circular block bootstrap resampling indices over ``0..M-1`` (reference
    ``CBB.sampling``): draw random block starts and wrap around."""
    idx = []
    while len(idx) < M:
        k = int(rng.integers(0, M))
        for j in range(block_length):
            idx.append((k + j) % M)
            if len(idx) >= M:
                break
    return np.asarray(idx[:M], dtype=int)


def run_cbb(inputs, r, *, block_length, reps, separation, update_dif, lts_alpha,
            delta, alpha, rng) -> Dict[str, np.ndarray]:
    """Per-unit CBB estimates: point effect, bootstrap variance, normal CI and
    p-value. Returns arrays aligned to ``inputs.unit_names``."""
    Y, X, T0, T, regime = inputs.Y, inputs.X, inputs.T0, inputs.T, inputs.regime
    N = inputs.N
    est, _aux = point_effects(Y, X, r, regime, update_dif=update_dif,
                              lts_alpha=lts_alpha, delta=delta)
    pre_idx = np.where(X == 0)[0]
    post_idx = np.where(X == 1)[0]
    boot = np.empty((reps, N))
    for b in range(reps):
        if separation:
            samp = np.concatenate([pre_idx[cbb_sampling(block_length, T0, rng)],
                                   post_idx[cbb_sampling(block_length, T - T0, rng)]])
        else:
            samp = cbb_sampling(block_length, T, rng)
        Yb, Xb = Y[:, samp], X[samp]
        try:
            eff_b, _ = point_effects(Yb, Xb, r, regime, update_dif=update_dif,
                                     lts_alpha=lts_alpha, delta=delta)
        except Exception:                                  # pragma: no cover
            eff_b = np.full(N, np.nan)
        boot[b] = eff_b
    var = np.nanmean((boot - np.nanmean(boot, axis=0)) ** 2, axis=0)   # ddof=0, ref CBB
    sd = np.sqrt(np.maximum(var, 0.0))
    z = norm.ppf(1 - alpha / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        pval = 2 * norm.cdf(-np.abs(est) / np.where(sd > 0, sd, np.nan))
    pval = np.where(np.isfinite(pval), pval, 1.0)
    return dict(est=est, var=var, ci_lower=est - z * sd, ci_upper=est + z * sd,
                pvalue=pval)


def run_conformal(inputs, r, *, update_dif, lts_alpha, delta) -> Dict[str, np.ndarray]:
    """Per-unit moving-block permutation p-values (Chernozhukov et al. 2021) for
    the sharp null of no effect, on the counterfactual-proxy residuals."""
    Y, T0, T, regime = inputs.Y, inputs.T0, inputs.T, inputs.regime
    N = inputs.N
    est, aux = point_effects(Y, inputs.X, r, regime, update_dif=update_dif,
                             lts_alpha=lts_alpha, delta=delta)
    CF = counterfactual_panel(Y, T0, aux["Lambda"], aux["sigma"], aux["mu"], delta=delta)
    U = Y - CF                                            # per-period residuals
    n_post = T - T0

    def stat(u):
        return np.sum(np.abs(u[T0:])) / np.sqrt(n_post)

    pval = np.empty(N)
    for j in range(N):
        s_obs = stat(U[j])
        shifts = np.array([stat(np.roll(U[j], s)) for s in range(T)])
        pval[j] = 1.0 - np.mean(shifts < s_obs)
    return dict(est=est, var=np.full(N, np.nan),
                ci_lower=np.full(N, np.nan), ci_upper=np.full(N, np.nan),
                pvalue=pval)
