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
    # six times the Path-A tolerance. Matching the published numbers means
    # reproducing the reference solver's choice among a continuum, so the
    # solver stays as it is until the identification question is settled.
    #
    # A least-norm tie-break is the obvious alternative and was measured. It
    # does better than the plain active set and still does not replicate:
    # reporting the minimiser of least Euclidean norm moves the worst cell of
    # the cartel-count outcome (co_num) from 1.01e-03 to 9.07e-03 against the
    # committed reference, where `benchmarks/cases/ssc_guanajuato.py` pins
    # att_max_abs_diff at 0.001 +/- 0.0015. Two traps sit around that number.
    # The rate outcomes barely move (1.87e-04 to 2.08e-04), so a check reading
    # only those reports agreement; and `war`, which is a cartel outcome, moves
    # only 8.1e-05 to 8.3e-05, so quoting one cartel series in place of the
    # worst one reports agreement too. co_num is the cell that decides it.
    #
    # The rule also fails to deliver what it promises here. This block is 33
    # units by 15 clean periods at demeaned rank 7, with two sets of exactly
    # coincident demeaned paths (four units and five, the latter constant
    # before treatment), so the face is approached only to the conditioning of
    # that block. Relabelling the donors still moves a weight by 0.03 under
    # least norm, against 0.48 under the plain active set.
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
