"""Ridge augmentation for the bilevel SCM engine (Augmented SCM).

The ridge-augmented synthetic control of Ben-Michael, Feller & Rothstein (2021)
(``progfunc="Ridge"`` in the ``augsynth`` R package) is *not* a separate base
estimator -- it is a **bias-correction layer on top of any base donor weights**.
Given simplex SCM weights ``W`` and the pre-treatment outcomes, it adds a ridge
correction that closes the residual pre-treatment imbalance the simplex SCM
cannot, at the cost of leaving the simplex (the augmented weights may go
negative and need not sum to one). That is exactly why it lives in the bilevel
package: any backend's ``W`` can be augmented, so the capability rides along
"wherever the bilevel solver goes".

The structure follows ``sdfordham/pysyncon``'s ``AugSynth``, with the penalty
selection corrected to reproduce **augsynth R** exactly on its canonical Kansas
example (ATT -0.0401). The two corrections, both about the *scale* on which the
penalty is chosen, mirror augsynth's ``fit_ridgeaug_formatted``:

* outcomes are **centered** (each period minus its control-unit mean) *before*
  the lambda grid and cross-validation are built, so the chosen lambda is on the
  same scale as the centered ridge solve (pysyncon builds them on the raw
  outcomes, so it picks a lambda far too large and barely augments);
* the CV standard error divides by **sqrt(n_folds)**, not sqrt(n_lambdas).

A useful invariance keeps this clean: for simplex weights (``sum w = 1``), the
base SCM fit is identical on raw vs. centered outcomes -- the per-period shift
cancels -- so the engine's existing backends need no change; only this layer
centers, and only for its own correction.

The ridge formula (augsynth ``fit_ridgeaug_inner`` / pysyncon ``solve_ridge``):
with centered treated pre-vector ``A``, centered donor matrix ``B`` and base
weights ``W``,

    M = A - B @ W                              # residual imbalance
    N = (B @ B.T + lambda * I)^{-1}            # ridge-regularised Gram inverse
    W_ridge = M @ N @ B
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

_EPS = 1e-12


def simplex_qp(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Exact simplex SCM weights: ``min ||A - B @ w||^2`` s.t. ``w >= 0``,
    ``sum w = 1``, solved as a QP with cvxpy/CLARABEL.

    Augmented SCM augments an *accurately solved* simplex SCM (augsynth uses
    quadprog). The bilevel package's iterative FISTA primitive
    (``simplex_lstsq``) under-converges on ill-conditioned, long pre-periods, so
    the ridge base is solved exactly here.

    Parameters
    ----------
    B : numpy.ndarray, shape (m, J)
        Donor matching matrix (``m`` matching rows, ``J`` donors).
    A : numpy.ndarray, shape (m,)
        Treated unit's matching vector.

    Returns
    -------
    numpy.ndarray, shape (J,)
    """
    import cvxpy as cp

    B = np.asarray(B, dtype=float)
    A = np.asarray(A, dtype=float).ravel()
    w = cp.Variable(B.shape[1], nonneg=True)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(A - B @ w)), [cp.sum(w) == 1])
    problem.solve(solver=cp.CLARABEL)
    if w.value is None:  # pragma: no cover - degenerate fallback
        problem.solve(solver=cp.SCS)
    weights = np.clip(np.asarray(w.value, dtype=float).ravel(), 0.0, None)
    s = weights.sum()
    return weights / s if s > _EPS else weights


def solve_ridge(
    A: np.ndarray, B: np.ndarray, W: np.ndarray, lambda_: float
) -> np.ndarray:
    """Ridge augmentation to the base weights (augsynth/pysyncon ``solve_ridge``).

    Parameters
    ----------
    A : numpy.ndarray, shape (m,)
        Treated unit's (centered) matching vector.
    B : numpy.ndarray, shape (m, J)
        Donor (centered) matching matrix.
    W : numpy.ndarray, shape (J,)
        Base (simplex) SCM weights.
    lambda_ : float
        Ridge penalty.

    Returns
    -------
    numpy.ndarray, shape (J,)
        The additive ridge correction ``W_ridge``; the augmented weights are
        ``W + W_ridge``.
    """
    A = np.asarray(A, dtype=float).ravel()
    B = np.asarray(B, dtype=float)
    W = np.asarray(W, dtype=float).ravel()
    M = A - B @ W
    N = np.linalg.inv(B @ B.T + lambda_ * np.identity(B.shape[0]))
    return M @ N @ B


def generate_lambdas(
    X: np.ndarray, lambda_min_ratio: float = 1e-8, n_lambda: int = 20
) -> np.ndarray:
    """A singular-value-scaled geometric grid of ridge penalties.

    ``lambda_max`` is the squared largest singular value of the (centered) donor
    matrix ``X`` (augsynth ``get_lambda_max``); the grid descends geometrically
    to ``lambda_max * lambda_min_ratio``. Matches augsynth's ``create_lambda_list``
    (exponents ``0..n_lambda`` -> ``n_lambda + 1`` points), so the 1-SE grid
    choice reproduces augsynth.
    """
    X = np.asarray(X, dtype=float)
    sing = np.linalg.svd(X, compute_uv=False)
    lambda_max = float(sing[0]) ** 2.0
    scaler = lambda_min_ratio ** (1.0 / n_lambda)
    return lambda_max * (scaler ** np.arange(n_lambda + 1))


class _HoldoutSplitter:
    """Iterate over leave-``holdout_len``-out splits of the matching *rows*
    (pre-treatment periods). Mirrors augsynth's ``get_lambda_errors`` loop."""

    def __init__(self, B: np.ndarray, A: np.ndarray, holdout_len: int = 1) -> None:
        if B.shape[0] != A.shape[0]:
            raise ValueError("B and A must have the same number of rows.")
        if holdout_len < 1:
            raise ValueError("holdout_len must be at least 1.")
        if holdout_len >= B.shape[0]:
            raise ValueError("holdout_len must be less than the number of rows.")
        self.B = B
        self.A = A
        self.holdout_len = holdout_len

    def __iter__(self):
        m = self.B.shape[0]
        for idx in range(m - self.holdout_len + 1):
            hold = slice(idx, idx + self.holdout_len)
            keep = np.ones(m, dtype=bool)
            keep[hold] = False
            yield self.B[keep], self.B[hold], self.A[keep], self.A[hold]


def cross_validate(
    base_weights_fn,
    X0: np.ndarray,
    X1: np.ndarray,
    lambdas: np.ndarray,
    holdout_len: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-one-period-out CV of the ridge penalty (augsynth ``get_lambda_errors``).

    For each holdout split, refit the base weights on the retained periods (via
    ``base_weights_fn(B_train, A_train)``), apply the ridge augmentation for each
    candidate lambda, and score the squared error on the held-out period(s).

    Parameters
    ----------
    base_weights_fn : callable
        ``(B, A) -> W`` base-weight solver (e.g. the simplex SCM fit), used to
        refit on each training fold.
    X0 : numpy.ndarray, shape (m, J)
        Centered donor matrix.
    X1 : numpy.ndarray, shape (m,)
        Centered treated vector.
    lambdas : numpy.ndarray
        Candidate ridge penalties.
    holdout_len : int, optional
        Block length held out each fold, by default ``1``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(lambdas, errors_mean, errors_se)`` -- the per-lambda mean CV error and
        its standard error across folds (``sd / sqrt(n_folds)``).
    """
    X0 = np.asarray(X0, dtype=float)
    X1 = np.asarray(X1, dtype=float).ravel()
    lambdas = np.asarray(lambdas, dtype=float)
    res = []
    for B_t, B_v, A_t, A_v in _HoldoutSplitter(X0, X1, holdout_len=holdout_len):
        W = base_weights_fn(B_t, A_t)
        fold = []
        for lam in lambdas:
            W_aug = W + solve_ridge(A=A_t, B=B_t, W=W, lambda_=lam)
            fold.append(float(np.sum((A_v - B_v @ W_aug) ** 2)))
        res.append(fold)
    arr = np.asarray(res, dtype=float)
    n_folds = arr.shape[0]
    errors_mean = arr.mean(axis=0)
    # augsynth ``get_lambda_errors``: sd(x)/sqrt(length(x)) over the folds.
    errors_se = arr.std(axis=0) / np.sqrt(n_folds)
    return lambdas, errors_mean, errors_se


def best_lambda(
    lambdas: np.ndarray,
    errors_mean: np.ndarray,
    errors_se: np.ndarray,
    min_1se: bool = True,
) -> float:
    """Select the ridge penalty from a CV curve (augsynth ``choose_lambda``).

    With ``min_1se`` (default), the *largest* lambda whose mean error is within
    one standard error of the minimum (the parsimonious 1-SE rule); otherwise the
    lambda at the outright minimum.
    """
    errors_mean = np.asarray(errors_mean, dtype=float)
    errors_se = np.asarray(errors_se, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    if min_1se:
        threshold = errors_mean.min() + errors_se[int(errors_mean.argmin())]
        return float(lambdas[errors_mean <= threshold].max())
    return float(lambdas[int(errors_mean.argmin())])


@dataclass
class RidgeAugmentResult:
    """Outcome of :func:`ridge_augment_weights`.

    Attributes
    ----------
    W : np.ndarray
        Augmented donor weights ``W_base + W_ridge`` (off the simplex in
        general).
    W_base : np.ndarray
        The base simplex SCM weights, before augmentation.
    W_ridge : np.ndarray
        The additive ridge correction.
    lambda_ : float
        Ridge penalty used.
    cv : dict or None
        Cross-validation curve (``lambdas``, ``errors_mean``, ``errors_se``)
        when lambda was selected by CV; ``None`` when a fixed lambda was given.
    """

    W: np.ndarray
    W_base: np.ndarray
    W_ridge: np.ndarray
    lambda_: float
    cv: Optional[Dict[str, Any]] = field(default=None)


def ridge_augment_weights(
    y_pre: np.ndarray,
    Y0_pre: np.ndarray,
    *,
    base_weights_fn=None,
    lambda_: Optional[float] = None,
    n_lambda: int = 20,
    lambda_min_ratio: float = 1e-8,
    holdout_length: int = 1,
    min_1se: bool = True,
) -> RidgeAugmentResult:
    """Ridge-augment a simplex SCM fit on the pre-treatment outcomes.

    Centers the outcomes (each period minus its donor mean), fits the base
    simplex SCM, selects the ridge penalty by leave-one-period-out CV when
    ``lambda_`` is ``None`` (augsynth's 1-SE rule by default), and returns the
    augmented weights. The base SCM (and each CV fold) is solved by
    ``base_weights_fn``; the default :func:`simplex_qp` is an exact QP, since
    Augmented SCM augments an accurately-solved simplex (augsynth uses quadprog).

    Parameters
    ----------
    y_pre : np.ndarray, shape (T0,)
        Treated pre-treatment outcomes.
    Y0_pre : np.ndarray, shape (T0, J)
        Donor pre-treatment outcomes.
    base_weights_fn : callable, optional
        ``(B, A) -> W`` base-weight solver; defaults to :func:`simplex_qp`.
    lambda_ : float, optional
        Fixed ridge penalty; if ``None`` (default) it is chosen by CV.
    n_lambda, lambda_min_ratio, holdout_length, min_1se
        CV / grid hyper-parameters (augsynth defaults).
    """
    y_pre = np.asarray(y_pre, dtype=float).ravel()
    Y0_pre = np.asarray(Y0_pre, dtype=float)
    if base_weights_fn is None:
        base_weights_fn = simplex_qp

    # Center each period by its donor (control) mean -- augsynth's
    # ``X_cent <- apply(X, 2, x - mean(x[trt==0]))``. The base simplex fit is
    # invariant to this shift (sum w = 1), but the ridge correction is not.
    mu = Y0_pre.mean(axis=1)
    B = Y0_pre - mu[:, None]          # centered donor matrix (T0, J)
    A = y_pre - mu                    # centered treated vector (T0,)

    W_base = np.asarray(base_weights_fn(B, A), dtype=float).ravel()

    cv: Optional[Dict[str, Any]] = None
    if lambda_ is None:
        lambdas = generate_lambdas(B, lambda_min_ratio=lambda_min_ratio,
                                   n_lambda=n_lambda)
        lambdas, mean, se = cross_validate(
            base_weights_fn, B, A, lambdas, holdout_len=holdout_length
        )
        lambda_ = best_lambda(lambdas, mean, se, min_1se=min_1se)
        cv = {"lambdas": lambdas, "errors_mean": mean, "errors_se": se}

    W_ridge = solve_ridge(A, B, W_base, float(lambda_))
    return RidgeAugmentResult(
        W=W_base + W_ridge, W_base=W_base, W_ridge=W_ridge,
        lambda_=float(lambda_), cv=cv,
    )
