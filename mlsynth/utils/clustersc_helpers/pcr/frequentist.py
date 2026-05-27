"""Frequentist OLS weight solver for PCR-SC.

Implements the *learn* step of Algorithm 2 of Rho et al. (2025):
given the HSVT-denoised pre-period donor matrix
:math:`\\widetilde{M}^- \\in \\mathbb{R}^{T_0 \\times J}` and the
target pre-period vector :math:`x_0^- \\in \\mathbb{R}^{T_0}`,
return

.. math::

   \\widehat{f} = \\arg\\min_{f \\in \\mathbb{R}^J}
                 \\| \\widetilde{M}^- f - x_0^- \\|_2^2.

Optionally extends to ``p``-norm (ridge / lasso) regularisation via
``lambda_penalty``, ``p``, ``q`` (Appendix E of the paper compares OLS,
Ridge, Lasso): ``min_f ||x0 - M f||_2^2 + lambda ||f||_p^q``. For the
unregularised path (``lambda_penalty in {None, 0}``) the closed-form
pseudo-inverse solver is used -- the paper's exact Algorithm 2 Step 3.

The regularised path is solved directly with cvxpy, so this module has no
dependency on the legacy ``estutils`` optimizer.
"""

from __future__ import annotations

from typing import List, Optional

import cvxpy as cp
import numpy as np

from ....exceptions import MlsynthEstimationError

# Defaults mirror the historical estutils.Opt behaviour (ridge, p = q = 2).
_DEFAULT_P_NORM = 2
_DEFAULT_Q_EXPONENT = 2


def solve_ols(
    denoised_donor_pre: np.ndarray,
    target_pre: np.ndarray,
    donor_names: Optional[List[str]] = None,
    lambda_penalty: Optional[float] = None,
    p: Optional[float] = None,
    q: Optional[float] = None,
) -> np.ndarray:
    """Return the OLS weight vector :math:`\\widehat{f}`.

    Parameters
    ----------
    denoised_donor_pre : np.ndarray
        HSVT-denoised pre-period donor matrix, shape ``(T0, J)``.
    target_pre : np.ndarray
        Treated unit's pre-period outcomes, shape ``(T0,)``.
    donor_names : list of str, optional
        Unused; retained for call-signature compatibility.
    lambda_penalty, p, q : float or None
        Optional ``p``-norm regularisation. ``None`` (or ``0``) means the
        paper's plain OLS via pseudo-inverse. Defaults are ridge (``p = q = 2``).
    """
    if denoised_donor_pre.ndim != 2:
        raise MlsynthEstimationError("denoised_donor_pre must be 2D (T0, J).")
    if denoised_donor_pre.shape[0] != target_pre.shape[0]:
        raise MlsynthEstimationError(
            f"Pre-period length mismatch: donors have {denoised_donor_pre.shape[0]} "
            f"rows but target has {target_pre.shape[0]}."
        )

    if lambda_penalty in (None, 0, 0.0):
        # Paper Algorithm 2: f_hat = pinv(M_pre) @ x_pre.
        return np.linalg.pinv(denoised_donor_pre) @ target_pre

    lam = float(lambda_penalty)
    p_norm = _DEFAULT_P_NORM if p is None else p
    q_exp = _DEFAULT_Q_EXPONENT if q is None else q

    w = cp.Variable(denoised_donor_pre.shape[1])
    residual = target_pre - denoised_donor_pre @ w
    penalty = lam * cp.power(cp.norm(w, p_norm), q_exp)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(residual) + penalty))
    problem.solve(solver=cp.OSQP if p_norm == 1 else cp.CLARABEL)

    if w.value is None:
        raise MlsynthEstimationError(
            f"Regularised OLS solve did not converge (status: {problem.status})."
        )
    return np.asarray(w.value, dtype=float)
