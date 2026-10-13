"""Synthetic-control weights for SSC (intercept + simplex, batch over units).

Each unit's untreated outcome is modelled as ``a_i + Y_t' b_i`` where ``b_i``
lies on the simplex (non-negative, sums to one) with ``b_ii = 0`` -- i.e. a
demeaned synthetic control of unit ``i`` on *all other* units (Cao, Lu & Wu
2026, eq. 2.1). Fitting every unit in turn yields the intercept vector ``a`` and
the weight matrix ``B`` used throughout the estimator and its inference.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def sc_weights_one(y: np.ndarray, X: np.ndarray) -> Tuple[float, np.ndarray]:
    """Demeaned simplex synthetic control of one unit on the others.

    Solves ``min_b || (y - mean y) - (X - mean X) b ||^2`` subject to
    ``b >= 0`` and ``sum(b) = 1``, then recovers the intercept
    ``a = mean(y) - mean(X) b``.

    Parameters
    ----------
    y : np.ndarray, shape (T0,)
        Treated unit's pre-treatment series.
    X : np.ndarray, shape (T0, N-1)
        Donor units' pre-treatment series (columns).

    Returns
    -------
    a : float
        Intercept.
    b : np.ndarray, shape (N-1,)
        Simplex weights on the donors.
    """
    # Kept on cvxpy deliberately. This program is not identified on the
    # authors' panel: 32 donors against 15 pre-periods at rank 7, so the
    # treated unit is exactly reproducible by a whole face of the simplex and
    # the pre-period fit does not pin down a weight vector. The active set
    # reaches an exact fit (objective 3.1e-33 against CLARABEL's 1.1e-09) and
    # lands on a vertex with support 5 where CLARABEL lands in the interior
    # with support 32. In sample the two agree to 2e-05; out of sample they
    # diverge, and the paper's cartel-outcome estimates move by up to 0.03 --
    # six times the Path-A tolerance.
    #
    # A least-norm tie-break is the obvious alternative and was measured. Max
    # absolute deviation from the committed reference, where
    # `benchmarks/cases/ssc_guanajuato.py` pins att_max_abs_diff at
    # 0.001 +/- 0.0015:
    #
    #     rule                        all cells   co_num   rate outcomes
    #     cvxpy CLARABEL (shipped)    1.015e-03  1.015e-03    1.867e-04
    #     plain active set            2.979e-02  1.659e-02    2.077e-04
    #     least norm                  8.184e-03  8.184e-03    2.076e-04
    #
    # Two traps sit around those numbers. The rate outcomes barely move, so a
    # check reading only those reports agreement; and `war`, which is a cartel
    # outcome, moves only 8.1e-05 to 8.3e-05, so quoting one cartel series in
    # place of the worst one reports agreement too. co_num is the cell that
    # decides it.
    #
    # Least norm is relabelling-invariant here, which it was not when this
    # comment was first written: donor relabelling moves a weight by 0.0 over 33
    # blocks and 8 permutations each, against 0.0526 median and 0.50 max under
    # the plain active set. The earlier reading of 0.03 was taken while the ridge
    # in `solve_simplex_qp_least_norm` sat under the pivot loop's release
    # threshold, so the rule did not bind; that is fixed, and the invariance it
    # promises now holds on this block.
    #
    # What remains is that 8.184e-03 exceeds the pin's band, and the band is
    # narrower than the identified set. On the co_num panel the optimum is a
    # single point on 10 of the 33 programs, and the identified interval of the
    # counterfactual that feeds the ATT has a median width of 0.0488, exceeding
    # 0.0025 on 20 of 33. All three rules above land inside that set, so the pin
    # separates them by which point they choose and not by whether the estimate
    # is right. Matching the published numbers means reproducing the reference
    # solver's choice among a continuum, so the solver stays as it is until the
    # identification question is settled.
    import cvxpy as cp

    yd = y - y.mean()
    Xd = X - X.mean(axis=0, keepdims=True)
    n = X.shape[1]
    b = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(yd - Xd @ b)),
                      [b >= 0, cp.sum(b) == 1])
    prob.solve(solver=cp.CLARABEL)
    bv = np.clip(np.asarray(b.value, dtype=float), 0.0, None)
    s = bv.sum()
    bv = bv / s if s > 0 else np.full(n, 1.0 / n)
    a = float(y.mean() - X.mean(axis=0) @ bv)
    return a, bv


def synthetic_control_batch(Y_pre: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit :func:`sc_weights_one` for every unit (each treated, others donors).

    Parameters
    ----------
    Y_pre : np.ndarray, shape (N, T0)
        Pre-treatment outcomes (rows are units, columns are periods).

    Returns
    -------
    a_hat : np.ndarray, shape (N,)
        Per-unit intercepts.
    B_hat : np.ndarray, shape (N, N)
        Weight matrix; row ``i`` holds unit ``i``'s donor weights with a zero
        on the diagonal.
    """
    N, _ = Y_pre.shape
    a_hat = np.zeros(N)
    B_hat = np.zeros((N, N))
    for i in range(N):
        others = [j for j in range(N) if j != i]
        a_i, b_i = sc_weights_one(Y_pre[i], Y_pre[others].T)
        a_hat[i] = a_i
        B_hat[i, others] = b_i
    return a_hat, B_hat
