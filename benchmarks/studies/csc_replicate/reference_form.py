"""The reference's stacked construction, transliterated, against eq. (24).

CVXR cannot be installed here (CRAN is unreachable through the sandbox
proxy), so ``run_rwscm`` cannot be executed for a cell-for-cell check. The
next best thing is to rebuild the reference's *own* linear algebra in cvxpy
-- the long stacked residual vector, the replicated donor block, the
``vec(t(X_pre) %*% alphas %*% H)`` interaction term and the ``O1_n1Tn1``
intercept design -- and confirm it reaches the same optimum as the compact
matrix program in ``core.py``.

The reference builds (``Simulation/2_Code_CSC_IDiD_FDiD.R``):

    Y_pre_long   = c(t(Y_pre))                      # (n1 * T0,) unit-major
    X_pre_full   = rbind of t(X_pre) n1 times       # (n1 * T0, n0)
    objective    = sum_squares(Y_pre_long - O1 %*% mu - X_pre_full %*% wei
                               - vec(t(X_pre) %*% alphas %*% H))
    constraints  = wei %*% t(l_N) + alphas %*% H >= 0
                   t(l_K) %*% (wei %*% t(l_N) + alphas %*% H) == t(l_N)
                   mu[1] == 0

with ``H`` the (K x n1) matrix of treated covariates. Agreement of the two
objective values is a check on the transcription of eq. (24); it is not a
check on the authors' solver.
"""

from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np

from core import csc_weights
from dgp import simulate


def reference_form_objective(Y1_pre, Y0_pre, X1) -> tuple[float, np.ndarray]:
    """Solve the reference's stacked program; return (fit SSR, weights)."""
    n1, T0 = Y1_pre.shape
    n0 = Y0_pre.shape[0]
    K = X1.shape[1]

    y_long = Y1_pre.reshape(-1)  # c(t(Y_pre)): unit-major, time fast
    X_pre = Y0_pre.T  # t(X_pre) in the reference: (T0, n0)
    X_pre_full = np.tile(X_pre, (n1, 1))  # (n1*T0, n0)
    H = X1.T  # (K, n1)

    # O1_n1Tn1: unit-major intercept dummies with unit 1's column all zero.
    O1 = np.zeros((n1 * T0, n1))
    for i in range(1, n1):
        O1[i * T0 : (i + 1) * T0, i] = 1.0

    wei = cp.Variable(n0)
    alphas = cp.Variable((n0, K))
    mu = cp.Variable(n1)

    # vec(t(X_pre) %*% alphas %*% H) stacks columns, i.e. unit-major.
    interaction = cp.vec(X_pre @ alphas @ H, order="F")

    resid = y_long - O1 @ mu - X_pre_full @ wei - interaction
    W = np.ones((n1, 1)) @ cp.reshape(wei, (1, n0), order="C") + X1 @ alphas.T
    constraints = [W >= 0, W @ np.ones(n0) == np.ones(n1), mu[0] == 0]

    prob = cp.Problem(cp.Minimize(cp.sum_squares(resid)), constraints)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.solve(solver=cp.CLARABEL)
    return float(prob.value), np.asarray(W.value, dtype=float)


if __name__ == "__main__":
    gaps = []
    for seed in range(10):
        p = simulate(rng=np.random.default_rng(seed))
        tr, do = p.treated_idx, p.donor_idx
        T0 = p.Y.shape[1] - 1
        Y1_pre, Y0_pre, X1 = p.Y[tr, :T0], p.Y[do, :T0], p.X_dummies[tr]

        ref_obj, _ = reference_form_objective(Y1_pre, Y0_pre, X1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compact = csc_weights(Y1_pre, Y0_pre, X1, intercept="pin_first")
        gaps.append(abs(ref_obj - compact.objective) / max(1.0, ref_obj))
        print(
            f"seed {seed:2d}  reference-form SSR {ref_obj:10.4f}   "
            f"eq.(24) SSR {compact.objective:10.4f}   "
            f"rel. gap {gaps[-1]:.2e}"
        )
    print(f"\nmax relative gap over 10 panels: {max(gaps):.2e}")
