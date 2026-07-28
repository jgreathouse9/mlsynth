"""The ESC ensemble engine (Lian & Pu 2026), pure NumPy + scikit-learn.

Implements Algorithm 1 / eqs (3)-(8): ``n_learners`` LASSO base learners, each
on a block-subsampled pre-period index set crossed with a random donor subspace;
the ensemble mean (or median / trimmed mean) is the point counterfactual and the
pointwise prediction interval is the empirical quantile spread of the ensemble.
"""

from __future__ import annotations

import warnings
from math import ceil, floor
from typing import List

import numpy as np
from scipy.stats import trim_mean
from sklearn.linear_model import Lasso, LassoCV

from .config import ESCConfig
from .structures import ESCFit, ESCInputs


def _blocks(T0: int, b: int) -> List[np.ndarray]:
    """Partition ``0..T0-1`` into contiguous blocks of length ``b`` (the last
    block absorbs the remainder), the ESC time-block units (Section 3.2)."""
    nblk = max(1, ceil(T0 / b))
    return [np.arange(k * b, min((k + 1) * b, T0)) for k in range(nblk)]


def _penalty(X: np.ndarray, y: np.ndarray, rule: str, n_alphas: int, cv: int) -> float:
    """LASSO penalty from a cross-validated path: the 1-SE rule (conservative)
    or the CV minimum. ``cv`` is clamped to a feasible fold count."""
    n = len(y)
    k = int(max(2, min(cv, n)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lcv = LassoCV(alphas=n_alphas, cv=k, max_iter=5000, random_state=0).fit(X, y)
    if rule == "min":
        return float(lcv.alpha_)
    mse = lcv.mse_path_.mean(axis=1)
    se = lcv.mse_path_.std(axis=1) / np.sqrt(lcv.mse_path_.shape[1])
    jmin = int(np.argmin(mse))
    thr = mse[jmin] + se[jmin]
    cand = np.where(mse <= thr)[0]              # alphas_ descending -> first = largest
    return float(lcv.alphas_[cand[0]]) if cand.size else float(lcv.alpha_)


def fit_esc(inputs: ESCInputs, config: ESCConfig) -> ESCFit:
    """Fit the ESC ensemble and return the aggregated fit + prediction interval."""
    y, X = np.asarray(inputs.y, float), np.asarray(inputs.donor_matrix, float)
    T0, T, J = inputs.T0, inputs.T, inputs.J
    ypre, Xpre = y[:T0], X[:T0]
    rng = np.random.default_rng(config.seed)

    blocks = _blocks(T0, config.block_length)
    nblk = len(blocks)
    ksub = min(nblk, max(2, round(config.block_keep * nblk))) if nblk >= 2 else 1
    ssize = max(1, floor(config.subspace_ratio * J))

    # Global penalty: used directly under lambda_scope='global', and as the
    # fallback when a per-learner subsample is too small to cross-validate.
    lam_global = _penalty(Xpre, ypre, config.lambda_rule, config.n_alphas,
                          min(config.cv_folds, T0))

    m = int(config.n_learners)
    ens = np.full((m, T), np.nan)
    coefs = np.zeros((m, J))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in range(m):
            Ksel = rng.choice(nblk, size=ksub, replace=False)
            I = np.concatenate([blocks[k] for k in sorted(Ksel)])
            S = rng.choice(J, size=ssize, replace=False)
            Xtr, ytr = Xpre[np.ix_(I, S)], ypre[I]
            if config.lambda_scope == "local" and len(I) >= 4:
                try:
                    lam = _penalty(Xtr, ytr, config.lambda_rule, config.n_alphas,
                                   min(config.cv_folds, len(I)))
                except Exception:                       # pragma: no cover - CV fallback
                    lam = lam_global
            else:
                lam = lam_global
            fit = Lasso(alpha=lam, max_iter=5000).fit(Xtr, ytr)
            ens[r] = fit.intercept_ + X[:, S] @ fit.coef_
            coefs[r, S] = fit.coef_

    if config.aggregate == "mean":
        cf = np.nanmean(ens, axis=0)
    elif config.aggregate == "median":
        cf = np.nanmedian(ens, axis=0)
    else:
        cf = trim_mean(ens, config.trim, axis=0)

    a = config.alpha
    lo = np.nanquantile(ens, a / 2.0, axis=0)
    hi = np.nanquantile(ens, 1.0 - a / 2.0, axis=0)
    gap = y - cf
    has_post = T > T0
    att = float(np.mean(gap[T0:])) if has_post else float("nan")

    # Effective ensemble weights: for mean aggregation the point counterfactual
    # is exactly X @ (mean coef) + mean intercept, so the averaged coefficient is
    # the honest "net donor contribution" (an informative summary otherwise).
    Wfull = coefs.mean(axis=0)
    donor_weights = {inputs.donor_names[j]: float(Wfull[j])
                     for j in range(J) if abs(Wfull[j]) > 1e-10}
    pre_rmse = float(np.sqrt(np.mean(gap[:T0] ** 2)))

    att_ci = (None, None)
    att_se = None
    if config.inference and has_post:
        memb = np.nanmean(y[None, T0:] - ens[:, T0:], axis=1)   # per-learner ATT
        att_ci = (float(np.nanquantile(memb, a / 2.0)),
                  float(np.nanquantile(memb, 1.0 - a / 2.0)))
        att_se = float(np.nanstd(memb))

    params = {
        "n_learners": m, "block_length": config.block_length,
        "subspace_ratio": config.subspace_ratio, "block_keep": config.block_keep,
        "subspace_size": int(ssize), "n_blocks": int(nblk), "blocks_kept": int(ksub),
        "alpha": a, "lambda_rule": config.lambda_rule,
        "lambda_scope": config.lambda_scope, "aggregate": config.aggregate,
        "lambda_global": float(lam_global), "T0": int(T0), "J": int(J),
    }
    return ESCFit(
        counterfactual=cf, gap=gap, att=att, lower=lo, upper=hi, ensemble=ens,
        donor_weights=donor_weights, pre_rmse=pre_rmse, att_ci=att_ci,
        att_se=att_se, level=1.0 - a, params=params,
    )
