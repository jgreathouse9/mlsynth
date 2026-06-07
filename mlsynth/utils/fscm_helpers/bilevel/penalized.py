"""Penalized synthetic control backend (Abadie & L'Hour 2021).

A third backend for :func:`solve_bilevel`, complementary to the Malo corner
search and the MSCMT global search. Where those optimise the predictor
weights ``V`` of the bilevel program, the penalized estimator fixes
``Gamma = I`` (predictors enter equally, after standardisation) and instead
adds a *pairwise matching* penalty that resolves the non-uniqueness of the
donor weights ``W`` directly. For treated predictors ``X1``, donor predictors
``X0`` and penalty ``lambda >= 0`` it solves (paper eq. 5)

    min_W  || X1 - X0 W ||^2  +  lambda * sum_j W_j || X1 - X_j ||^2
    s.t.   W >= 0,  sum_j W_j = 1.

The penalty trades component-wise fit (the synthetic control objective) for
aggregate closeness (the nearest-neighbour matching objective). By Theorem 1
of the paper, for any ``lambda > 0`` the solution is **unique and sparse**
(at most ``p + 1`` nonzero weights) -- which is exactly why this backend does
not suffer the malo/mscmt disagreement that arises when ``X1`` lies inside the
donor convex hull (where infinitely many ``W`` fit equally well).

The penalty ``lambda`` is chosen by leave-one-out cross-validation over the
donor pool (paper Section 4.1), the default selector the authors advocate:
for each donor ``j``, build its penalized synthetic control from the other
donors and score how well it predicts ``j``'s (pre-period) outcome; pick the
``lambda`` with the smallest total error.

This module also provides the authors' **bias correction** (Section 2.4,
eq. 7), an Abadie-Imbens (2011) adjustment that removes the part of the gap
attributable to residual predictor imbalance ``X1 - X0 W`` via a regression
of the outcome on the predictors. It is a standalone utility usable with the
weights from *any* backend.

References
----------
Abadie, A., & L'Hour, J. (2021). A Penalized Synthetic Control Estimator for
Disaggregated Data. Journal of the American Statistical Association,
116(536), 1817-1834. https://doi.org/10.1080/01621459.2021.1971535
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np

from .simplex import mspe, project_simplex, simplex_lstsq
from .structure import BilevelProblem, BilevelSolution

_EPS = 1e-12

# CV needs at least this many pre-periods so the train/validation split is
# non-degenerate (a single training period cannot identify donor weights).
_MIN_CV_TPRE = 4


def _spectral_radius(Q: np.ndarray, iters: int = 40) -> float:
    """Largest eigenvalue of a symmetric PSD matrix by power iteration."""
    n = Q.shape[0]
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    nx = np.linalg.norm(x)
    if nx < _EPS:  # pragma: no cover - a random Gaussian init vector is never exactly zero
        return 1.0
    x /= nx
    lam = 1.0
    for _ in range(iters):
        y = Q @ x
        ny = np.linalg.norm(y)
        if ny < _EPS:
            return _EPS
        x = y / ny
        lam = float(x @ (Q @ x))
    return lam


def _simplex_qp(Q: np.ndarray, c: np.ndarray, *, max_iter: int = 2000,
                tol: float = 1e-9, warn: bool = False) -> np.ndarray:
    """Minimize ``w' Q w + c' w`` over the probability simplex via FISTA.

    ``Q`` must be symmetric positive semidefinite (here ``Q = X0' X0``). If
    ``warn`` and the iteration limit is hit before the step norm meets ``tol``,
    a :class:`RuntimeWarning` is emitted.
    """
    n = Q.shape[0]
    if n == 1:
        return np.array([1.0])
    L = 2.0 * _spectral_radius(Q) + _EPS
    step = 1.0 / L
    w = np.full(n, 1.0 / n)
    z = w.copy()
    t = 1.0
    for _ in range(max_iter):
        grad = 2.0 * (Q @ z) + c
        w_new = project_simplex(z - step * grad)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = w_new + ((t - 1.0) / t_new) * (w_new - w)
        if np.linalg.norm(w_new - w) < tol:
            return w_new
        w, t = w_new, t_new
    if warn:
        warnings.warn(
            f"penalized simplex QP did not converge within max_iter={max_iter} "
            f"(tol={tol}); returned weights may be sub-optimal.",
            RuntimeWarning,
            stacklevel=2,
        )
    return w


def penalized_weights(X1: np.ndarray, X0: np.ndarray, lam: float, *,
                      max_iter: int = 2000, tol: float = 1e-9,
                      warn: bool = False) -> np.ndarray:
    """Penalized synthetic-control weights (eq. 5) for one target.

    Parameters
    ----------
    X1 : np.ndarray
        Target predictor vector, shape ``(K,)``.
    X0 : np.ndarray
        Donor predictor matrix, shape ``(K, J)``.
    lam : float
        Penalty ``lambda >= 0``. ``lambda > 0`` guarantees a unique, sparse
        solution; ``lambda -> 0`` is the (possibly non-unique) pure synthetic
        control; large ``lambda`` approaches nearest-neighbour matching.
    warn : bool
        Forwarded to the inner QP: warn on non-convergence.

    Raises
    ------
    ValueError
        If ``lam < 0`` (a negative penalty makes the objective non-convex).
    """
    if lam < 0:
        raise ValueError(f"penalty lambda must be non-negative, got {lam}.")
    X1 = np.asarray(X1, dtype=float)
    X0 = np.asarray(X0, dtype=float)
    Q = X0.T @ X0
    d2 = np.sum((X1[:, None] - X0) ** 2, axis=0)        # pairwise discrepancies
    c = -2.0 * (X0.T @ X1) + float(lam) * d2
    return _simplex_qp(Q, c, max_iter=max_iter, tol=tol, warn=warn)


def _build_block(prob: BilevelProblem, periods: slice, use_outcomes: bool,
                 use_predictors: bool):
    """Matching block (treated vector, donor matrix) for ``prob``.

    Rows are the pre-period outcomes over ``periods`` (lagged-outcome
    predictors, kept in *levels* so the synthetic tracks the outcome path) and,
    optionally, the covariate predictors ``X`` (standardised to unit variance,
    Abadie-L'Hour's ``Gamma = I`` convention). Outcomes are not re-standardised
    -- z-scoring each period would discard the level/trend the synthetic must
    reproduce -- so the pairwise penalty is dominated by outcome distance, with
    the covariates breaking ties.
    """
    m1_rows, M0_rows = [], []
    if use_outcomes:
        m1_rows.append(prob.y1_pre[periods])
        M0_rows.append(prob.Y0_pre[periods])            # (len, J)
    if use_predictors and prob.n_predictors > 0:
        X0 = prob.X0
        ctr = X0.mean(axis=1)
        scl = X0.std(axis=1) + 1e-8
        m1_rows.append((prob.X1 - ctr) / scl)
        M0_rows.append((X0 - ctr[:, None]) / scl[:, None])
    return np.concatenate(m1_rows), np.vstack(M0_rows)


def _loo_cv_lambda(prob: BilevelProblem, lam_grid: np.ndarray, *,
                   use_outcomes: bool, use_predictors: bool,
                   train_frac: float, max_iter: int, tol: float):
    """ALH Section 4.1 leave-one-out CV over the donor pool, time-split.

    Each donor is treated as a pseudo-treated unit and matched to the others
    on a *training* slice of the pre-period (plus covariates); the penalty is
    scored by how well the resulting weights predict the donor's outcome on a
    held-out *validation* slice. Note: on strongly trending panels this
    criterion tends to over-penalise toward nearest-neighbour, because convex
    combinations cannot extrapolate the trend on the held-out block -- prefer
    the treated holdout (Section 4.2) there. Returns ``(best_lambda, curve)``.
    """
    T0 = prob.Tpre
    n_train = min(max(2, int(round(T0 * train_frac))), T0 - 1)
    tr, va = slice(0, n_train), slice(n_train, T0)
    Y0 = prob.Y0_pre
    J = prob.n_donors
    _, M0_tr = _build_block(prob, tr, use_outcomes, use_predictors)
    cv_curve = np.full(len(lam_grid), np.inf)
    for li, lam in enumerate(lam_grid):
        total = 0.0
        for j in range(J):
            others = [k for k in range(J) if k != j]
            w = penalized_weights(M0_tr[:, j], M0_tr[:, others], lam,
                                  max_iter=max_iter, tol=tol)
            total += float(np.sum((Y0[va][:, j] - Y0[va][:, others] @ w) ** 2))
        cv_curve[li] = total
    return float(lam_grid[int(np.argmin(cv_curve))]), cv_curve


def _pensynth_lam_grid(prob: BilevelProblem, nlambda: int = 100) -> np.ndarray:
    """Van Kesteren's exponential lambda grid for ``cv_pensynth``.

    Runs from ``1e-11`` to an empirical ``lmax`` at which the penalty reaches
    the nearest-neighbour limit: ``lmax = sum_j |X1.X0_j| / ||X1 - X0_j||^2``
    over the (standardised) covariates. Requires predictors.
    """
    m1, M0 = _build_block(prob, slice(0, prob.Tpre),
                          use_outcomes=False, use_predictors=True)
    d2 = np.sum((m1[:, None] - M0) ** 2, axis=0)
    finite = d2 > _EPS
    lmax = (float(np.sum(np.abs((m1 @ M0)[finite] / d2[finite])))
            if finite.any() else 1.0)
    lmax = max(lmax, 1e-9)
    return np.exp(np.linspace(np.log(1e-11), np.log(lmax), nlambda))


def _pensynth_cv_lambda(prob: BilevelProblem, lam_grid: np.ndarray, *,
                        use_outcomes: bool, use_predictors: bool,
                        train_frac: float, max_iter: int, tol: float):
    """Van Kesteren ``cv_pensynth``: fit on covariates, validate on outcomes.

    Unlike the time-split ``holdout``/``loo`` selectors, there is no train/
    validation split of the pre-period. For each ``lambda`` the donor weights
    are formed purely from covariate balance (penalised), then scored by the
    mean squared error on the *full* held-out pre-intervention outcome path.
    This cross-fit (covariates -> outcomes) avoids the trend-extrapolation bias
    that makes the time-split criteria over-penalise toward nearest-neighbour.
    Requires covariate predictors; ``use_outcomes``/``train_frac`` are accepted
    for a uniform selector signature but ignored. Returns ``(best, curve)``.
    """
    if prob.n_predictors == 0:
        raise ValueError(
            "cv='pensynth' needs covariate predictors (X) to match on; none "
            "supplied. Use cv='holdout'/'loo' for outcome-path matching."
        )
    m1, M0 = _build_block(prob, slice(0, prob.Tpre),
                          use_outcomes=False, use_predictors=True)
    y1, Y0 = prob.y1_pre, prob.Y0_pre
    cv_curve = np.full(len(lam_grid), np.inf)
    for li, lam in enumerate(lam_grid):
        w = penalized_weights(m1, M0, lam, max_iter=max_iter, tol=tol)
        cv_curve[li] = float(np.mean((y1 - Y0 @ w) ** 2))
    return float(lam_grid[int(np.argmin(cv_curve))]), cv_curve


def _holdout_cv_lambda(prob: BilevelProblem, lam_grid: np.ndarray, *,
                       use_outcomes: bool, use_predictors: bool,
                       train_frac: float, max_iter: int, tol: float):
    """ALH Section 4.2 pre-intervention holdout on the treated unit.

    Match the treated unit on a *training* slice of the pre-period (plus
    covariates) and pick the ``lambda`` that best predicts its outcome on the
    held-out tail of the pre-period. This targets the treated unit's fit
    directly and is robust on trending panels. Returns ``(best_lambda, curve)``.
    """
    T0 = prob.Tpre
    n_train = min(max(2, int(round(T0 * train_frac))), T0 - 1)
    tr, va = slice(0, n_train), slice(n_train, T0)
    m1_tr, M0_tr = _build_block(prob, tr, use_outcomes, use_predictors)
    y1_va, Y0_va = prob.y1_pre[va], prob.Y0_pre[va]
    cv_curve = np.full(len(lam_grid), np.inf)
    for li, lam in enumerate(lam_grid):
        w = penalized_weights(m1_tr, M0_tr, lam, max_iter=max_iter, tol=tol)
        cv_curve[li] = float(np.sum((y1_va - Y0_va @ w) ** 2))
    return float(lam_grid[int(np.argmin(cv_curve))]), cv_curve


def solve_penalized(
    prob: BilevelProblem,
    *,
    lam="cv",
    cv: str = "holdout",
    lam_grid: Optional[Sequence[float]] = None,
    use_outcomes: bool = True,
    use_predictors: bool = True,
    train_frac: float = 0.7,
    max_iter: int = 2000,
    tol: float = 1e-9,
) -> BilevelSolution:
    """Penalized SCM backend (Abadie & L'Hour 2021).

    Unlike the ``malo``/``mscmt`` backends (which optimise predictor weights
    ``V`` and match covariates in the lower level), the penalized estimator has
    no upper level: it matches on the *pretreatment outcome path* (and the
    covariates, if any), in the spirit of Abadie-L'Hour's predictor vector,
    and resolves the donor-weight non-uniqueness with the pairwise penalty.

    Parameters
    ----------
    prob : BilevelProblem
        Outcome and predictor matrices (covariates assumed standardised).
    lam : float or "cv"
        Penalty. ``"cv"`` (default) selects ``lambda`` by cross-validation
        (the selector chosen by ``cv``); a float fixes it.
    cv : {"holdout", "loo", "pensynth"}
        Cross-validation selector used when ``lam="cv"``. ``"holdout"``
        (default) is the treated-unit pre-intervention holdout (Section 4.2),
        robust on trending panels; ``"loo"`` is the donor leave-one-out
        (Section 4.1), which can over-penalise toward nearest-neighbour when
        the outcome trends. ``"pensynth"`` is van Kesteren's ``cv_pensynth``
        criterion: fit the weights on the covariates and pick the ``lambda``
        minimising MSE on the held-out pre-intervention outcome path (no
        time-split); requires covariate predictors.
    lam_grid : sequence of float, optional
        Candidate ``lambda`` values. Defaults to ``geomspace(1e-4, 1e2, 25)``,
        or van Kesteren's exponential ``1e-11 .. lmax`` grid when
        ``cv="pensynth"``.
    use_outcomes, use_predictors : bool
        Whether to match on the pre-period outcome path and/or the covariate
        predictors. At least one must be true; both default to true.
    train_frac : float
        Fraction of pre-periods used for matching in the CV criterion; the
        rest are the validation periods.
    max_iter, tol : int, float
        FISTA stopping controls for the inner simplex QP.

    Returns
    -------
    BilevelSolution
        ``stage="penalized"``; ``V`` is uniform (``Gamma = I``); the chosen
        ``lambda``, sparsity, and CV curve live in ``metadata``.
    """
    if not (use_outcomes or (use_predictors and prob.n_predictors > 0)):
        raise ValueError("penalized backend needs outcomes and/or predictors to match on.")
    pensynth_mode = isinstance(lam, str) and cv == "pensynth"
    if pensynth_mode and prob.n_predictors == 0:
        raise ValueError(
            "cv='pensynth' needs covariate predictors (X) to match on; none "
            "supplied. Use cv='holdout'/'loo' for outcome-path matching."
        )
    if lam_grid is None:
        lam_grid = (_pensynth_lam_grid(prob) if pensynth_mode
                    else np.geomspace(1e-4, 1e2, 25))
    lam_grid = np.asarray(lam_grid, dtype=float)

    if isinstance(lam, str):
        selector = {"holdout": _holdout_cv_lambda, "loo": _loo_cv_lambda,
                    "pensynth": _pensynth_cv_lambda}.get(cv)
        if selector is None:
            raise ValueError(
                f"Unknown cv selector {cv!r}; expected 'holdout', 'loo' or 'pensynth'."
            )
        # The time-split selectors need enough pre-periods to split; the
        # pensynth cross-fit does not (it scores the whole pre-period path).
        if cv in ("holdout", "loo") and prob.Tpre < _MIN_CV_TPRE:
            raise ValueError(
                f"cv='{cv}' needs at least {_MIN_CV_TPRE} pre-periods to form a "
                f"non-degenerate split, got Tpre={prob.Tpre}; pass a fixed numeric "
                f"lam instead."
            )
        lam_val, cv_curve = selector(
            prob, lam_grid, use_outcomes=use_outcomes, use_predictors=use_predictors,
            train_frac=train_frac, max_iter=max_iter, tol=tol)
        selected_by = cv
        # A boundary selection means the optimum likely lies outside the grid.
        if lam_grid.size > 1 and (
            lam_val <= lam_grid.min() + _EPS or lam_val >= lam_grid.max() - _EPS
        ):
            warnings.warn(
                f"CV selected lambda={lam_val:g} at the edge of lam_grid "
                f"[{lam_grid.min():g}, {lam_grid.max():g}]; the optimum may lie "
                f"outside the grid -- consider widening it.",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        lam_val, cv_curve, selected_by = float(lam), None, "fixed"

    # Final weights: match on the full pre-period block. In pensynth mode the
    # model is defined by covariate matching (the outcome path was the holdout),
    # so the final fit uses predictors only -- consistent with the CV criterion.
    fit_outcomes = False if pensynth_mode else use_outcomes
    fit_predictors = True if pensynth_mode else use_predictors
    m1, M0 = _build_block(prob, slice(0, prob.Tpre), fit_outcomes, fit_predictors)
    W = penalized_weights(m1, M0, lam_val, max_iter=max_iter, tol=tol, warn=True)
    V = np.full(max(prob.n_predictors, 1), 1.0 / max(prob.n_predictors, 1))  # Gamma = I

    W_unc = simplex_lstsq(prob.Y0_pre, prob.y1_pre, warn=True)
    lower_bound = mspe(prob.y1_pre, prob.Y0_pre, W_unc)
    upper = mspe(prob.y1_pre, prob.Y0_pre, W)
    lower_loss = float(np.sum((prob.X1 - prob.X0 @ W) ** 2)) if prob.n_predictors else 0.0

    return BilevelSolution(
        V=V, W=W, upper_loss=upper, lower_loss=lower_loss,
        lower_bound=lower_bound, stage="penalized", iterations=0,
        metadata={
            "backend": "penalized",
            "lambda": lam_val,
            "lambda_selected_by": selected_by,
            "n_nonzero": int(np.sum(W > 1e-6)),
            "matched_on": "+".join(
                ([("outcomes")] if fit_outcomes else [])
                + (["predictors"] if fit_predictors and prob.n_predictors else [])
            ),
            "cv_curve": None if cv_curve is None else [float(x) for x in cv_curve],
            "lam_grid": [float(x) for x in lam_grid],
        },
    )


def bias_corrected_gaps(
    W: np.ndarray, X1: np.ndarray, X0: np.ndarray,
    y1: np.ndarray, Y0: np.ndarray, *, ridge: float = 1e-2,
) -> np.ndarray:
    """Abadie-L'Hour / Abadie-Imbens bias correction of the SC gap (eq. 7).

    Removes the part of the gap attributable to residual predictor imbalance
    ``X1 - X0 W`` via a per-period (ridge) regression of the donor outcome on
    the donor predictors:

        tau_bc(t) = (y1_t - Y0_t W) - (X1 - X0 W)' beta_t,

    where ``beta_t`` is the regression slope of ``Y0_t`` on ``X0`` across
    donors (the intercept cancels because ``sum_j W_j = 1``). Works with the
    weights from any backend; ``X1, X0`` must be in the same (standardised)
    space used to compute ``W``.

    A ridge penalty regularises the slope. This is important: when the
    predictors weakly explain the outcome, the unregularised slope is huge and
    the "correction" injects noise instead of removing bias (the paper cites
    ridge-based ``mu_0`` for exactly this reason). With standardised
    predictors a small ridge keeps the correction bounded; ``ridge=0`` recovers
    ordinary least squares.

    Parameters
    ----------
    W : np.ndarray
        Donor weights, shape ``(J,)``.
    X1 : np.ndarray
        Target predictors, shape ``(K,)``.
    X0 : np.ndarray
        Donor predictors, shape ``(K, J)``.
    y1 : np.ndarray
        Target outcomes over the periods of interest, shape ``(T,)``.
    Y0 : np.ndarray
        Donor outcomes over the same periods, shape ``(T, J)``.
    ridge : float
        Ridge penalty on the regression slope (default ``1e-2``).

    Returns
    -------
    np.ndarray
        Bias-corrected gaps, shape ``(T,)``.
    """
    W = np.asarray(W, dtype=float)
    X1 = np.asarray(X1, dtype=float)
    X0 = np.asarray(X0, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    Y0 = np.asarray(Y0, dtype=float)

    J, K = X0.shape[1], X0.shape[0]
    # Centre predictors and outcomes across donors so the intercept drops out;
    # then ridge-regress the centred outcome on the centred predictors.
    Xc = X0 - X0.mean(axis=1, keepdims=True)            # (K, J)
    Yc = Y0 - Y0.mean(axis=1, keepdims=True)            # (T, J)
    G = Xc @ Xc.T + ridge * np.eye(K)                   # (K, K)
    beta = np.linalg.solve(G, Xc @ Yc.T)               # (K, T)
    imbalance = X1 - X0 @ W                             # (K,)
    correction = imbalance @ beta                       # (T,)
    raw_gap = y1 - Y0 @ W                               # (T,)
    return raw_gap - correction
