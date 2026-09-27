"""Hsiao & Zhou (2019 JAE) DGP6/DGP7 and the seven counterfactual methods.

The readable oracle for the replication spike. Every construction cites the
paper.

DGP6 (Eq 32): y_it = g_1i f_1t + g_2i f_2t + u_it, f and g iid N(0,1).
DGP7:         the same with f_1t and f_2t random walks.
Errors (Eq 33): u_it = (1+b^2) v_it + b v_{i+1,t} + b v_{i-1,t}, b = 1,
                v_it ~ N(0, s_i^2), s_i^2 ~ 0.5 (chi2(1) + 1).

Neither DGP has covariates and neither has a treatment effect, so the treated
unit's observed series IS its own counterfactual and the criteria measure
prediction error directly.

Methods, Section 6 (E1)-(E7):
  E1 PCA   Bai (2009) factors from the controls, then Xu (2017) Step 2-3.
  E2 CCE   no beta to estimate here, so y_1t on an intercept and ALL controls.
  E3 CPDA  LASSO-selected subset of (y~_t - X~_t beta); with no X this is E4.
  E4 PDA   LASSO-selected subset of y~_t, refit by OLS (Li & Bell 2017).
  E5 PDAX  E4 with the treated unit's X added to the pool; with no X this is E4.
  E6 MA    mean-corrected simple average of E1-E5 (Eqs 23-24).
  E7 MB    mean- and scale-corrected simple average of E1-E5 (Eqs 25-26).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LassoCV


# ----------------------------------------------------------------------
# The data-generating processes
# ----------------------------------------------------------------------

def simulate(dgp: str, n_co: int, T0: int, T2: int, rng) -> tuple:
    """One draw. Returns (y_treated, Y_controls) with the treated unit first."""
    T = T0 + T2
    N = n_co + 1

    f = rng.standard_normal((T, 2))
    if dgp == "dgp7":                       # factors follow random walks
        f = np.cumsum(f, axis=0)
    loadings = rng.standard_normal((N, 2))

    # Equation 33: weak cross-sectional dependence, b = 1. The neighbours are
    # i-1 and i+1 with no wrap: unit 1 has only unit 2 as a neighbour, so it
    # shares one v with the donor pool. Rolling the array instead would hand
    # the treated unit a second shared component from unit N, which is what
    # the semiparametric methods predict from -- it makes every one of them
    # look better than the design allows.
    b = 1.0
    sigma2 = 0.5 * (rng.chisquare(1, N) + 1.0)
    v = rng.standard_normal((T, N)) * np.sqrt(sigma2)
    pad = np.zeros((T, 1))
    ahead = np.hstack([v[:, 1:], pad])                  # v_{i+1}, 0 at i = N
    behind = np.hstack([pad, v[:, :-1]])                # v_{i-1}, 0 at i = 1
    u = (1 + b**2) * v + b * ahead + b * behind

    Y = f @ loadings.T + u
    return Y[:, 0], Y[:, 1:]


# ----------------------------------------------------------------------
# E1: the parametric approach
# ----------------------------------------------------------------------

def pca_counterfactual(y1, Yco, T0, r):
    """Bai (2009) factors from the controls, Xu (2017) Steps 1-3.

    On a pure factor model Bai's least-squares estimator is the principal
    components of the control panel, so F is its leading r left singular
    vectors. Step 2 fits the treated unit's loadings on the pre-period only,
    Step 3 predicts with them (Equation 5).
    """
    U, s, _ = np.linalg.svd(Yco, full_matrices=False)
    F = U[:, :r] * s[:r]                              # (T, r)
    g1, *_ = np.linalg.lstsq(F[:T0], y1[:T0], rcond=None)
    return F @ g1


def select_r_by_cv(Yco, T0, r_max=5):
    """Xu (2017)'s leave-one-period-out cross-validation over r.

    Hold out each pre-period in turn, refit the treated loadings on the rest,
    and score the held-out prediction. The paper estimates r this way when it
    is treated as unknown; it also reports the case where r is known, and the
    tables do not say which they used, so the spike measures both.
    """
    Ypre = Yco[:T0]
    U, s, _ = np.linalg.svd(Yco, full_matrices=False)
    best, best_mse = 1, np.inf
    for r in range(1, min(r_max, Yco.shape[1], T0 - 2) + 1):
        F = (U[:, :r] * s[:r])[:T0]
        err = []
        for t in range(T0):
            keep = np.arange(T0) != t
            # All control loadings in one solve: the held-out period's
            # residuals across units are the score for this r.
            G, *_ = np.linalg.lstsq(F[keep], Ypre[keep], rcond=None)
            err.append(np.mean((Ypre[t] - F[t] @ G) ** 2))
        mse = float(np.mean(err))
        if mse < best_mse:
            best, best_mse = r, mse
    return best


# ----------------------------------------------------------------------
# E2: the semiparametric approach with no selection
# ----------------------------------------------------------------------

def cce_counterfactual(y1, Yco, T0):
    """y_1t on an intercept and every control, Equation 11 with w unrestricted.

    With no covariates there is no beta step, so v~_t = y~_t and Equation 15
    reduces to an unrestricted pre-period regression on the whole donor pool.
    When n_co + 1 > T0 that design is rank deficient; lstsq returns the
    minimum-norm solution, where R's lm() would drop aliased columns. The two
    differ, and the paper does not say which it used.
    """
    X = np.column_stack([np.ones(len(y1)), Yco])
    coef, *_ = np.linalg.lstsq(X[:T0], y1[:T0], rcond=None)
    return X @ coef


# ----------------------------------------------------------------------
# E3/E4/E5: the nonparametric approach with LASSO selection
# ----------------------------------------------------------------------

def pda_counterfactual(y1, Yco, T0, rng, refit=True):
    """LASSO-select donors on the pre-period, then refit by OLS.

    Li & Bell (2017) select with the LASSO and Hsiao et al. (2012) predict
    from the selected subset, Equation 22. glmnet's cross-validated penalty is
    what the paper used; LassoCV is the same object here, with the intercept
    the equation carries.
    """
    Xpre, ypre = Yco[:T0], y1[:T0]
    n_splits = max(3, min(10, T0 // 3))
    fit = LassoCV(cv=n_splits, max_iter=20000,
                  random_state=int(rng.integers(0, 2**31 - 1))).fit(Xpre, ypre)
    keep = np.flatnonzero(np.abs(fit.coef_) > 0)
    pda_counterfactual.last_kept = int(keep.size)      # read by experiment.py
    pda_counterfactual.last_alpha = float(fit.alpha_)
    if keep.size == 0 or not refit:
        return fit.predict(Yco)
    X = np.column_stack([np.ones(T0), Xpre[:, keep]])
    coef, *_ = np.linalg.lstsq(X, ypre, rcond=None)
    return np.column_stack([np.ones(len(y1)), Yco[:, keep]]) @ coef


# ----------------------------------------------------------------------
# E6/E7: model averaging
# ----------------------------------------------------------------------

def average_counterfactuals(paths, y1, T0, scale=False):
    """Equations 23-26. ``paths`` are the M counterfactuals over all T."""
    bar = np.mean(np.asarray(paths), axis=0)
    if not scale:
        a = float(np.mean(y1[:T0] - bar[:T0]))          # Equation 24
        return a + bar
    X = np.column_stack([np.ones(T0), bar[:T0]])        # Equation 26
    coef, *_ = np.linalg.lstsq(X, y1[:T0], rcond=None)
    return coef[0] + coef[1] * bar


# ----------------------------------------------------------------------
# The three criteria
# ----------------------------------------------------------------------

def criteria(errors, actual, predicted):
    """MAB, the root of the mean squared error, the mean squared error, MAP.

    The paper's rows are labelled MAB / MSE / MAP. Both roots are returned so
    the spike can say which of the two its "MSE" row is.
    """
    e = np.asarray(errors, dtype=float)
    return {
        "MAB": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "MSE": float(np.mean(e ** 2)),
        "MAP": float(np.mean(np.abs(predicted) / np.abs(actual))),
    }
