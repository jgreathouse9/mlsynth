"""Regression-based predictor weights ``V`` -- Stata ``synth``'s default rule.

``synth`` runs the Abadie-Diamond-Hainmueller nested optimisation only when the
caller passes ``nested``. Its default is cheaper and different (``synth2.sthlp``
lines 157-162):

    by default ``synth2`` uses a data-driven regression based method to obtain
    the variable weights contained in the V-matrix

``allsynth`` forwards ``nested`` as an ordinary optional flag, so published
results that did not request it -- which is most of them -- were produced by
this rule. Reproducing them needs it; the bilevel backends solve a different
program.

The algorithm
-------------
``synth2.ado`` calls two Mata subroutines in sequence::

    mata: normalize("Xtr", "Xco")                        // predictors first
    mata: regsval("Xtr_norm", "Xco_norm", "Ztr", "Zco")  // then outcome on them

Both ship compiled, inside ``l/lsynth2_mata_subr.mlib``; the package contains no
source for either. What the library's symbol table fixes::

    normalize:  storemat  xtrout xcoout  quadvariance  diag  xtrmat xcomat
    regsval:    sval  xtrout xcoout ztrout zcoout  beta  trace  diag  vmat

and the only Mata builtins referenced anywhere in it are ``diag``,
``quadvariance`` and ``trace``. So:

* ``normalize`` divides each predictor by its standard deviation over the
  pooled treated-plus-control block. No ``mean`` is referenced anywhere in the
  library, so it is scale-only, not centring.
* ``regsval`` receives both predictor blocks and both outcome blocks, so the
  treated unit is included in the regression.
* ``V = diag(bb') / trace(bb')`` -- squared coefficients, trace-normalised.

Two choices the symbols do *not* determine, because the regression solve
compiles to an opcode rather than a name lookup: whether ``beta`` comes from one
pooled regression over all ``(unit, period)`` observations or one per
pre-period, and whether an intercept is fitted. Both are exposed as arguments
and neither is guessed silently -- on the Wiltshire (2023) panel the two
readings tie (total absolute error 1.854 vs 1.768 over eight reported cells),
so the repository has no basis for picking one. See issue #334.

Ordering is load-bearing. Normalising the predictors *before* the regression,
rather than scaling afterwards at the weighting step, moved Wiltshire's Panel A
from -1.43 to -1.78 against a target of -1.77, and repaired a sign flip on his
Panel D.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .simplex import simplex_lstsq
from .structure import BilevelProblem, BilevelSolution

__all__ = ["regression_v", "solve_regression"]

_EPS = 1e-12


def _pooled_scale(X: np.ndarray) -> np.ndarray:
    """``normalize``: per-predictor standard deviation, no centring.

    Mata's ``quadvariance`` is a quad-precision variance; the transformation
    applied is a division by its square root. A predictor with no variation is
    left alone rather than divided by zero -- it carries no information for the
    regression either way, and its coefficient will be zero.
    """
    sd = X.std(axis=1, ddof=1) if X.shape[1] > 1 else np.zeros(X.shape[0])
    return np.where(sd < _EPS, 1.0, sd)


def regression_v(
    X1: np.ndarray,
    X0: np.ndarray,
    y1_pre: np.ndarray,
    Y0_pre: np.ndarray,
    *,
    normalize: bool = True,
    include_treated: bool = True,
    pooled: bool = False,
    fit_intercept: bool = True,
) -> np.ndarray:
    """Predictor weights ``v_h`` on the simplex.

    Parameters
    ----------
    X1, X0 : np.ndarray
        Treated predictor vector ``(K,)`` and donor predictor matrix ``(K, J)``.
    y1_pre, Y0_pre : np.ndarray
        Treated ``(T0,)`` and donor ``(T0, J)`` pre-treatment outcomes.
    normalize : bool
        Scale predictors by the pooled standard deviation before regressing.
        ``True`` reproduces ``normalize`` running ahead of ``regsval``. Setting
        it ``False`` regresses on raw predictors, which is a *different*
        estimator, not a formatting choice -- see the module docstring.
    include_treated : bool
        Pool the treated unit into the regression, as ``regsval``'s signature
        implies. ``False`` regresses on donors alone.
    pooled : bool
        One regression stacking every ``(unit, period)`` observation, versus one
        per pre-period with the squared coefficients summed. Not determined by
        the reference; see the module docstring.
    fit_intercept : bool
        Include a constant. With ``normalize=True`` the regressors are scaled
        but not centred, so this is not a no-op.

    Returns
    -------
    np.ndarray
        ``(K,)`` non-negative weights summing to one. Falls back to uniform
        weights when every coefficient vanishes.
    """
    X1 = np.asarray(X1, dtype=float).ravel()
    X0 = np.asarray(X0, dtype=float)
    y1_pre = np.asarray(y1_pre, dtype=float).ravel()
    Y0_pre = np.asarray(Y0_pre, dtype=float)
    K = X1.shape[0]
    if K == 0:
        return np.zeros(0)

    # Pool the blocks -- treated first, matching regsval(Xtr, Xco, Ztr, Zco).
    if include_treated:
        X_all = np.column_stack([X1, X0])            # (K, 1 + J)
        Z_all = np.column_stack([y1_pre, Y0_pre])    # (T0, 1 + J)
    else:
        X_all, Z_all = X0, Y0_pre

    if normalize:
        X_all = X_all / _pooled_scale(X_all)[:, None]

    D = X_all.T                                      # (N, K)
    if fit_intercept:
        D = np.column_stack([np.ones(D.shape[0]), D])

    if pooled:
        big_D = np.tile(D, (Z_all.shape[0], 1))
        beta, *_ = np.linalg.lstsq(big_D, Z_all.reshape(-1), rcond=None)
        b = beta[1:] if fit_intercept else beta
        sq = b ** 2
    else:
        beta, *_ = np.linalg.lstsq(D, Z_all.T, rcond=None)     # (K[+1], T0)
        b = beta[1:] if fit_intercept else beta
        sq = np.sum(b ** 2, axis=1)                  # diag(b b')

    total = sq.sum()                                 # trace(b b')
    if not np.isfinite(total) or total <= _EPS:
        return np.full(K, 1.0 / K)
    return sq / total


def solve_regression(
    prob: BilevelProblem,
    *,
    normalize: bool = True,
    include_treated: bool = True,
    pooled: bool = False,
    fit_intercept: bool = True,
    max_iter: int = 20000,
    tol: float = 1e-11,
) -> BilevelSolution:
    """Backend entry point: regression ``V``, then the lower-level simplex fit.

    Unlike ``malo``/``mscmt`` there is no outer search -- ``V`` is a closed-form
    function of the data, and only the lower level is solved. ``lower_bound`` is
    reported for comparability with the searching backends: it is the
    unconstrained outcome fit, so ``upper_loss - lower_bound`` still reads as
    the price paid for matching on predictors rather than on the outcome.
    """
    V = regression_v(
        prob.X1, prob.X0, prob.y1_pre, prob.Y0_pre,
        normalize=normalize, include_treated=include_treated,
        pooled=pooled, fit_intercept=fit_intercept,
    )
    root = np.sqrt(V)
    W = np.asarray(
        simplex_lstsq(prob.X0 * root[:, None], prob.X1 * root,
                      max_iter=max_iter, tol=tol), dtype=float).ravel()

    resid = prob.y1_pre - prob.Y0_pre @ W
    upper = float(resid @ resid)
    pres = prob.X1 - prob.X0 @ W
    lower = float(np.sum(V * pres ** 2))
    W_star = np.asarray(
        simplex_lstsq(prob.Y0_pre, prob.y1_pre, max_iter=max_iter, tol=tol),
        dtype=float).ravel()
    r_star = prob.y1_pre - prob.Y0_pre @ W_star
    return BilevelSolution(
        V=V, W=W, upper_loss=upper, lower_loss=lower,
        lower_bound=float(r_star @ r_star), stage="regression",
        iterations=0,
        metadata={"rule": "regression", "normalize": normalize,
                  "include_treated": include_treated, "pooled": pooled,
                  "fit_intercept": fit_intercept,
                  "n_predictors_weighted": int((V > 1e-6).sum())},
    )
