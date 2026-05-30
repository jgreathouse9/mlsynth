"""MUSC quadratic-program solver.

The estimator implemented here is the Modified Unbiased Synthetic
Control of Bottmer, Imbens, Spiess & Warnick (2024), JBES 42(2),
762–773. We use the paper's matrix-form parametrisation
(``M ∈ R^{N × (N+1)}``: a free intercept column plus an ``N × N``
weight block) and toggle the column-sums-to-zero constraint to switch
between the ``MUSC`` variant and the standard ``SC`` comparator.

Formally, let ``Y_pre ∈ R^{T_pre × N}`` be the time-major pre-period
outcome matrix. We solve

.. math::

   \\min_{M} \\sum_{i = 1}^{N} \\sum_{s \\in \\mathcal{T}_1}
       \\left( M_{i, 0} + \\sum_{j = 1}^{N} M_{i, j} Y_{j, s} \\right)^2

subject to

* :math:`M_{i, i+1} = 1` for all :math:`i` (the treated-self loading);
* :math:`M_{i, j+1} \\in [-1, 0]` for all :math:`i \\neq j`
  (non-positive control weights, bounded below by -1);
* :math:`\\sum_{j = 1}^{N} M_{i, j+1} = 0` for every :math:`i` (the SC
  adding-up restriction, i.e. donor weights sum to 1 in the canonical
  sign);
* :math:`\\sum_{i = 1}^{N} M_{i, j+1} = 0` for every :math:`j`,
  **only when ``column_balance=True``** (the MUSC unbiasedness
  restriction; see Lemma 1).

The intercept column is free for both variants — this is the MSC
modification of Doudchenko & Imbens (2016), which the paper treats as
implicit (see Table 2). The reduction to the standard SC weights when
``column_balance=False`` matches the ``sc_estimator.m`` baseline in
the paper's MATLAB replication archive.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cvxpy as cp
import numpy as np

from ...exceptions import MlsynthEstimationError


def solve_musc_qp(
    Y_pre: np.ndarray,
    *,
    column_balance: bool,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, str]:
    """Solve the MUSC (or standard-SC) quadratic programme.

    Parameters
    ----------
    Y_pre : np.ndarray
        Pre-treatment outcome matrix of shape ``(T_pre, N)`` --
        time-major: one row per pre-period, one column per unit.
    column_balance : bool
        When ``True``, impose ``sum_i M[i, j+1] = 0`` for every
        ``j ≥ 1`` -- the MUSC unbiasedness restriction. When ``False``
        the QP reduces to the standard SC estimator (one fit per
        treated row, the rest of the panel as donors).
    solver : str, optional
        cvxpy solver name; defaults to ``"CLARABEL"`` which ships with
        cvxpy ≥ 1.4. Override only if benchmarking.
    verbose : bool
        Forwarded to cvxpy.

    Returns
    -------
    (M, status) : (np.ndarray, str)
        ``M`` is the ``(N, N+1)`` weight matrix and ``status`` is the
        cvxpy solver status string. The first column of ``M`` is the
        per-row intercept; columns ``1..N`` are the within-row
        weights, with ``M[i, i+1] = 1`` and off-diagonals in
        ``[-1, 0]``.

    Raises
    ------
    MlsynthEstimationError
        If the cvxpy solver returns a non-optimal status.
    """
    Y_pre = np.asarray(Y_pre, dtype=float)
    if Y_pre.ndim != 2:
        raise MlsynthEstimationError(
            f"Y_pre must be 2-D (T_pre x N); got shape {Y_pre.shape}."
        )
    T_pre, N = Y_pre.shape
    if T_pre < 2 or N < 3:
        raise MlsynthEstimationError(
            f"MUSC requires T_pre ≥ 2 and N ≥ 3; got T_pre={T_pre}, N={N}."
        )

    alpha = cp.Variable(N)                      # intercept column M[:, 0]
    W = cp.Variable((N, N))                     # weight block M[:, 1:]

    # Off-diagonal mask: True off the diagonal, False on it.
    off = np.ones((N, N), dtype=bool)
    np.fill_diagonal(off, False)

    constraints = [
        cp.diag(W) == 1.0,                       # treated-self loading
        cp.sum(W, axis=1) == 0.0,                # SC adding-up (row sums)
        W[off] <= 0.0,
        W[off] >= -1.0,
    ]
    if column_balance:
        # MUSC unbiasedness restriction (Bottmer et al. 2024 eq. 2.6).
        constraints.append(cp.sum(W, axis=0) == 0.0)

    # Residual matrix R ∈ R^{N x T_pre}:
    #     R[i, s] = α_i + Σ_j W[i, j] * Y_pre[s, j].
    # In compact form: alpha[:, None] + W @ Y_pre.T.
    residuals = alpha[:, None] + W @ Y_pre.T

    prob = cp.Problem(cp.Minimize(cp.sum_squares(residuals)), constraints)
    try:
        prob.solve(solver=solver or "CLARABEL", verbose=verbose)
    except Exception as exc:                                    # noqa: BLE001
        raise MlsynthEstimationError(
            f"MUSC QP failed to solve (column_balance={column_balance}): {exc}"
        ) from exc

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise MlsynthEstimationError(
            f"MUSC QP returned non-optimal status: {prob.status} "
            f"(column_balance={column_balance})."
        )

    M = np.hstack(
        [
            np.asarray(alpha.value, dtype=float).reshape(N, 1),
            np.asarray(W.value, dtype=float),
        ]
    )
    return M, str(prob.status)


def predict_counterfactual(
    M: np.ndarray, Y_full: np.ndarray, treated_idx: int
) -> np.ndarray:
    """Synthetic counterfactual for the treated unit, length ``T``.

    Bottmer et al. parametrise the residual so that the treated-self
    loading is +1; the synthetic prediction is therefore
    ``Y_{treat,t} − (α + Σ_j M[treat, j+1] Y_{j, t})``, i.e. the
    treated outcome minus the row-residual. Equivalently:

        synth_t = − M[treat, 0] − Σ_{j ≠ treat} M[treat, j+1] Y_{j, t}.

    We return the equivalent ``Y_{treat, t} − residual`` form because
    it is numerically the cleanest expression.
    """
    Y_full = np.asarray(Y_full, dtype=float)
    if Y_full.ndim != 2 or Y_full.shape[1] != M.shape[1] - 1:
        raise MlsynthEstimationError(
            f"Y_full shape {Y_full.shape} incompatible with M shape {M.shape}."
        )
    alpha_i = float(M[treated_idx, 0])
    w_i = M[treated_idx, 1:]                                 # length N
    residual = alpha_i + Y_full @ w_i                        # length T
    return Y_full[:, treated_idx] - residual


def att_for_unit(
    M: np.ndarray, Y_full: np.ndarray, treated_idx: int, T0: int
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Convenience: gap, counterfactual, ATT, and pre-RMSE for one unit.

    Returns
    -------
    counterfactual : np.ndarray, length ``T``
    gap : np.ndarray, length ``T`` (treated minus counterfactual)
    att : float (mean of ``gap[T0:]``)
    pre_rmse : float (RMSE of ``gap[:T0]``)
    """
    Y_full = np.asarray(Y_full, dtype=float)
    counterfactual = predict_counterfactual(M, Y_full, treated_idx)
    gap = Y_full[:, treated_idx] - counterfactual
    att = float(gap[T0:].mean()) if gap.shape[0] > T0 else float("nan")
    pre_rmse = (
        float(np.sqrt((gap[:T0] ** 2).mean()))
        if T0 > 0
        else float("nan")
    )
    return counterfactual, gap, att, pre_rmse
