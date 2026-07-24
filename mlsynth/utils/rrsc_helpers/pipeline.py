"""RRSC estimation pipeline (He, Li, Shi & Miao 2026).

Both regimes share the two-stage skeleton: estimate untreated factor loadings
from the pre-period, then recover the average direct/interference effects as the
residual of a robust regression against those loadings.

* large-N (Section 4): EM factor analysis + Huber-IRLS factor regression on the
  post-period mean.
* fixed-N (Section 3): ``factanal`` loadings (rescaled by the covariance square
  root) + high-breakdown LTS, then a Donoho-Johnstone selection of valid
  controls and an OLS refit on them.

:func:`point_effects` recomputes the per-unit effects from an arbitrary
``(Y, X)`` so the circular block bootstrap can call it on resampled panels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy.linalg import sqrtm

from ...exceptions import MlsynthEstimationError
from .factor import fa_em, factanal_mle, select_nfactors
from .robust import lts_fit, robust_factor_estimation

_EPS = np.finfo(float).eps


@dataclass
class RRSCFit:
    """Point-estimate artifacts of an RRSC fit."""
    regime: str
    r: int
    effects: np.ndarray            # per-unit average effect (treated first)
    Lambda: np.ndarray             # N x r loadings
    sigma: np.ndarray              # N residual sd
    mu: np.ndarray                 # N pre-period means
    counterfactual_panel: np.ndarray   # N x T proxy mu_i + lambda_i' f_t
    communality: np.ndarray        # N fraction of pre-variance explained
    selection: Optional[np.ndarray]    # fixed-N valid-control mask (else None)
    extras: Dict[str, Any]


def _loadings(Y, T0, r, regime):
    """Regime-specific loadings, residual sd, pre-means and communality."""
    Ypre = Y[:, :T0]
    mu = Ypre.mean(axis=1)
    if regime == "large_n":
        tYc = Ypre.T - Ypre.T.mean(axis=0)              # T0 x N, column-centered
        Lambda, Sig = fa_em(tYc, r)
        sigma = np.sqrt(np.maximum(Sig, _EPS))
        var_pre = np.maximum(np.mean(tYc ** 2, axis=0), _EPS)
        comm = np.clip(1.0 - Sig / var_pre, 0.0, 1.0)
    else:                                               # fixed_n
        load, Psi = factanal_mle(Ypre.T, r)
        cov_pre = np.cov(Ypre)
        Lambda = sqrtm(cov_pre).real @ load
        resid_var = np.diag(cov_pre - Lambda @ Lambda.T)
        sigma = np.sqrt(np.maximum(resid_var, _EPS))
        comm = np.clip(1.0 - Psi, 0.0, 1.0)
    return Lambda, sigma, mu, comm


def point_effects(Y, X, r, regime, update_dif=True, lts_alpha=0.5, delta=1.345):
    """Per-unit average direct/interference effects for a panel ``(Y, X)``.

    Recomputed from scratch (loadings included) so the circular block bootstrap
    can call it on resampled panels. Returns ``(effects, aux)`` where ``aux``
    carries regime artifacts used by the point fit.
    """
    T0 = int((X == 0).sum())
    T = Y.shape[1]
    N = Y.shape[0]
    Ypre, Ypost = Y[:, X == 0], Y[:, X == 1]
    Lambda, sigma, mu, comm = _loadings(Y, T0, r, regime)
    if regime == "large_n":
        Yreg = Ypost.mean(axis=1)
        f = robust_factor_estimation(Yreg, Lambda, sigma, delta=delta)
        effects = Yreg - Lambda @ f
        aux = dict(Lambda=Lambda, sigma=sigma, mu=mu, comm=comm, selection=None)
        return effects, aux
    # fixed-N
    Yreg = (Ypost.mean(axis=1) - Ypre.mean(axis=1)) if update_dif else Ypost.mean(axis=1)
    coef, _raw, _w = lts_fit(Lambda, Yreg, alpha=lts_alpha)
    est1 = Yreg - Lambda @ coef
    ev = np.sort(np.linalg.eigvalsh((np.cov(Y) + np.cov(Y).T) / 2))
    k = int(np.floor(N / 2) + r)
    sigma2 = float(np.sum(ev[:k]))
    thresh = np.sqrt(2 * np.log(N * T) * sigma2 / T)
    sel = np.abs(est1) <= thresh
    if not sel.any():                                   # pragma: no cover
        idx = np.argsort(est1)[:k]
        sel = np.zeros(N, dtype=bool)
        sel[idx] = True
    Ls, ys = Lambda[sel], Yreg[sel]
    alpha_hat = np.linalg.solve(Ls.T @ Ls + 1e-8 * np.eye(r), Ls.T @ ys)
    effects = Yreg - Lambda @ alpha_hat
    aux = dict(Lambda=Lambda, sigma=sigma, mu=mu, comm=comm, selection=sel)
    return effects, aux


def implied_donor_weights(Lambda, unit_names):
    """Implied (non-unique) donor weights for the treated unit's counterfactual.

    RRSC is a factor-model estimator: the treated counterfactual is
    ``mu_0 + lambda_0' f_t``, not a weighted average of donors, so it has no
    donor weights in the synthetic-control sense. Following the MC-NNM
    convention, report the *implied* weights obtained by projecting the treated
    unit's loading vector onto the control units' loadings by least squares
    (a min-norm, hence non-unique, solution). The estimator's actual object is
    the loadings and the robust factor regression.
    """
    from ..results_helpers import make_weights_results
    Lambda = np.asarray(Lambda, dtype=float)
    lam0, Lc = Lambda[0], Lambda[1:]                 # treated / control loadings
    w, *_ = np.linalg.lstsq(Lc.T, lam0, rcond=None)  # min_w || Lc' w - lambda_0 ||
    names = [str(u) for u in np.asarray(unit_names)[1:]]
    donor_weights = {n: float(wi) for n, wi in zip(names, w)}
    return make_weights_results(
        donor_weights,
        constraint="implied, non-unique (factor-loading projection)",
        extra={
            "weights_are": "implied_non_unique",
            "note": ("RRSC is a factor-model estimator; these donor weights are "
                     "derived by projecting the treated unit's factor loadings "
                     "onto the control units' loadings and are not unique. The "
                     "estimator's actual object is the loadings and the robust "
                     "factor regression."),
        },
    )


def _gls_f(Yt, Lambda, sigma):
    W = 1.0 / sigma ** 2
    A = Lambda.T @ (W[:, None] * Lambda)
    return np.linalg.solve((A + A.T) / 2 + _EPS * np.eye(Lambda.shape[1]),
                           Lambda.T @ (W * Yt))


def counterfactual_panel(Y, T0, Lambda, sigma, mu, delta=1.345):
    """Per-period counterfactual proxy ``mu_i + lambda_i' f_t`` (eq. 7): GLS
    factor scores pre-intervention, Huber-robust regression post-intervention."""
    N, T = Y.shape
    Yc = Y - mu[:, None]
    CF = np.empty((N, T))
    for t in range(T):
        if t < T0:
            f = _gls_f(Yc[:, t], Lambda, sigma)
        else:
            f = robust_factor_estimation(Yc[:, t], Lambda, sigma, delta=delta)
        CF[:, t] = mu + Lambda @ f
    return CF


def run_rrsc(inputs, config) -> RRSCFit:
    """Point estimate: select the number of factors, fit the loadings, recover
    the per-unit effects, and build the counterfactual panel."""
    Y, X, T0, T = inputs.Y, inputs.X, inputs.T0, inputs.T
    regime = inputs.regime
    r = config.n_factors
    if r is None:
        r = select_nfactors(Y[:, :T0], k_max=config.max_factors)
    if r >= inputs.N:
        raise MlsynthEstimationError(
            f"RRSC: number of factors r={r} must be < number of units N={inputs.N}."
        )
    try:
        effects, aux = point_effects(
            Y, X, r, regime, update_dif=config.update_dif,
            lts_alpha=config.lts_alpha, delta=config.huber_delta,
        )
        CF = counterfactual_panel(Y, T0, aux["Lambda"], aux["sigma"], aux["mu"],
                                  delta=config.huber_delta)
    except (np.linalg.LinAlgError, ValueError) as exc:    # pragma: no cover
        raise MlsynthEstimationError(f"RRSC point estimation failed ({exc}).") from exc
    return RRSCFit(
        regime=regime, r=int(r), effects=effects, Lambda=aux["Lambda"],
        sigma=aux["sigma"], mu=aux["mu"], counterfactual_panel=CF,
        communality=aux["comm"], selection=aux["selection"],
        extras={"update_dif": config.update_dif, "lts_alpha": config.lts_alpha},
    )
