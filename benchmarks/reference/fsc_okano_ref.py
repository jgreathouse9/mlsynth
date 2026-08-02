"""Reference port of the Okano-Kurisu (2026) functional synthetic control.

Okano, R. & Kurisu, D. (2026). *"Functional Synthetic Control Methods for Metric
Space-Valued Outcomes."* arXiv:2601.07539.
Authors' R code: https://github.com/RyoOkano21/FSC

This is a *pre-build* reference implementation: mlsynth does not yet ship an FSC
estimator, and :mod:`benchmarks.cases.fsc_okano` pins this port against the
paper's three empirical applications so a build starts from validated ground.
When the estimator lands the case is re-pointed at ``mlsynth.FSC`` and this
module becomes its oracle.

Function-by-function correspondence with ``main_functions.R``:

===========================  ==================================================
``fsc_weights``              ``FSCM`` (paper eq. 1)
``augmented_fsc_weights``    ``FSCM_aug`` / ``FSCM_aug_covmat`` (eqs. 8-9)
``cross_val``                ``cross_val`` / ``cross_val_covmat`` (Remark 5)
``select_lambda``            ``optimise(obj, interval=c(a, b))``
``rearrangement``            ``Rearrangement::rearrangement`` via ``modif``
``cubic_bsplines``           ``cubicBsplines::Bsplines``
===========================  ==================================================

Two things about the method are worth stating up front, because they are what
make it cheap to build here. First, the whole estimator is the scalar augmented
SCM of Ben-Michael, Feller & Rothstein (2021) applied to a *stacked* design: the
metric-space outcome is carried into a Hilbert space by an isometry, evaluated on
a grid, and the ``(period, grid point)`` pairs are stacked into one long matching
vector. Consequently the base simplex solve reuses mlsynth's own exact QP
(:func:`mlsynth.utils.bilevel.ridge_augment.simplex_qp`) unchanged, and eq. (9)
is the ridge formula already implemented in that module with the pre-period axis
replaced by the stacked axis. Second, the basis expansion is not exotic: the
Fortran behind ``cubicBsplines::Bsplines`` returns exactly the partition-of-unity
cubic B-spline design matrix, so :func:`cubic_bsplines` is a thin wrapper over
:func:`scipy.interpolate.BSpline.design_matrix` with the knot vector the R
package builds internally (verified to 3e-15 over the grids used here).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar

from mlsynth.utils.bilevel.ridge_augment import simplex_qp

# R's optimise() default: .Machine$double.eps^0.25
R_OPTIMISE_TOL = float(np.finfo(float).eps ** 0.25)


# ---------------------------------------------------------------------------
# basis
# ---------------------------------------------------------------------------
def cubic_bsplines(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """``cubicBsplines::Bsplines(x, knots)`` -- an ``(len(x), len(knots) + 2)`` basis.

    The R function calls the Fortran ``cubicBsplines_general``, which evaluates
    de Boor's divided-difference formula on the knot sequence extended by three
    equispaced knots on each side. That is the standard cubic B-spline design
    matrix on the extended knots, which scipy computes directly.
    """
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    left, right = knots[1] - knots[0], knots[-1] - knots[-2]
    t = np.concatenate([
        knots[0] - left * np.arange(3, 0, -1),
        knots,
        knots[-1] + right * np.arange(1, 4),
    ])
    return BSpline.design_matrix(x, t, 3, extrapolate=True).toarray()


# ---------------------------------------------------------------------------
# projection back onto Psi(M)
# ---------------------------------------------------------------------------
def rearrangement(grid: np.ndarray, values: np.ndarray, n: int = 1000) -> np.ndarray:
    """``Rearrangement::rearrangement`` for one covariate, ``stochastic=FALSE``.

    Not a plain sort: the R routine rescales the grid to the unit interval,
    linearly interpolates onto ``n + 1`` equispaced points, and returns the
    type-7 empirical quantiles of those values at the rescaled grid. On the
    mortality application a plain sort lands at 0.0640 where this lands at the
    published 0.0634, so the detail is load-bearing.
    """
    grid = np.asarray(grid, dtype=float)
    nx = (grid - grid.min()) / (grid.max() - grid.min())
    interpolated = np.interp(np.arange(n + 1) / n, nx, np.asarray(values, float))
    return np.quantile(interpolated, nx, method="linear")


def project_quantile(grid: np.ndarray, values: np.ndarray,
                     low: float, high: float) -> np.ndarray:
    """``helper_functions.R::modif`` -- truncate to ``[low, high]``, rearrange."""
    return rearrangement(grid, np.clip(values, low, high))


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------
def fsc_weights(Y: np.ndarray, n_pre: int, *, round_to: int | None = 4) -> np.ndarray:
    """FSC weights, paper eq. (1). ``Y`` is ``(N, T, M)`` with unit 0 treated.

    ``round_to=4`` reproduces ``FSCM``'s ``round(weight_scm, 4)``, which the
    reference scripts carry into every downstream number.
    """
    treated = Y[0, :n_pre].ravel()                          # (n_pre * M,)
    donors = Y[1:, :n_pre].reshape(Y.shape[0] - 1, -1).T    # (n_pre * M, N - 1)
    w = np.asarray(simplex_qp(donors, treated), dtype=float).ravel()
    return w if round_to is None else np.round(w, round_to)


def _center_pre(Y: np.ndarray, n_pre: int) -> np.ndarray:
    """Subtract the donor mean at each pre-period (paper section 3.2)."""
    Yc = Y.copy()
    Yc[:, :n_pre] -= Yc[1:, :n_pre].mean(axis=0, keepdims=True)
    return Yc


def augmented_fsc_weights(Y: np.ndarray, n_pre: int, lam: float, *,
                          basis: np.ndarray | None = None,
                          grid_width: float = 1.0) -> np.ndarray:
    """Ridge augmented FSC weights, paper eq. (9).

    ``basis=None`` is ``FSCM_aug_covmat``: the Euclidean case, where the
    coordinates are already the coefficients in the standard basis.

    The scale convention follows ``main_functions.R:96`` -- the curves and the
    basis both drop their first grid row, and the sum is divided rather than
    multiplied by the grid width. A uniform rescaling of the basis is absorbed
    by a matching rescaling of ``lam``, so this affects the selected penalty but
    not the weights it produces.
    """
    Yc = _center_pre(Y, n_pre)
    gamma = fsc_weights(Yc, n_pre)
    if basis is None:
        R = Yc
    else:
        R = Yc[:, :, 1:] @ basis[1:, :] / grid_width         # (N, T, K)
    r0 = R[1:, :n_pre].reshape(R.shape[0] - 1, -1)           # (N - 1, K * n_pre)
    r1 = R[0, :n_pre].ravel()
    gram = r0.T @ r0 + lam * np.eye(r0.shape[1])
    return gamma + r0 @ np.linalg.solve(gram, r1 - r0.T @ gamma)


def synthetic(Y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted donor average, ``(T, M)``."""
    return np.einsum("j,jtm->tm", np.asarray(weights, float), Y[1:])


# ---------------------------------------------------------------------------
# penalty selection
# ---------------------------------------------------------------------------
def cross_val(lam: float, Y: np.ndarray, n_pre: int, **kw) -> float:
    """Leave-one-pre-period-out CV error (paper Remark 5)."""
    total = 0.0
    for k in range(n_pre):
        keep = [t for t in range(Y.shape[1]) if t != k]
        w = augmented_fsc_weights(Y[:, keep], n_pre - 1, lam, **kw)
        total += float(np.sum((Y[0, k] - w @ Y[1:, k]) ** 2))
    return total


def select_lambda(Y: np.ndarray, n_pre: int, low: float, high: float,
                  **kw) -> float:
    """``optimise(obj_func, interval=c(low, high))`` -- bounded Brent."""
    result = minimize_scalar(lambda L: cross_val(L, Y, n_pre, **kw),
                             bounds=(low, high), method="bounded",
                             options={"xatol": R_OPTIMISE_TOL})
    return float(result.x)


# ---------------------------------------------------------------------------
# reported quantities
# ---------------------------------------------------------------------------
def pre_treatment_fit(Y: np.ndarray, n_pre: int, synth: np.ndarray, *,
                      drop_first_grid_point: bool) -> float:
    """``sqrt(sum_t ||Y_1t - synth_t||^2)`` over the pre-periods.

    ``drop_first_grid_point`` reproduces the reference scripts' ``sum(diffs[-1])``.
    They apply it to the FSC leg of all three applications and to the augmented
    leg of two of them; see :mod:`benchmarks.cases.fsc_okano` for which number
    each published figure was computed under.
    """
    sl = slice(1, None) if drop_first_grid_point else slice(None)
    gap = Y[0, :n_pre, sl] - synth[:n_pre, sl]
    return float(np.sqrt(np.sum(gap ** 2)))
