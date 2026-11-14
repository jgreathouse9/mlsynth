"""Frequentist OLS weight solver for PCR-SC.

Implements the *learn* step of Algorithm 2 of Rho et al. (2025):
given the HSVT-denoised pre-period donor matrix
:math:`\\widetilde{M}^- \\in \\mathbb{R}^{T_0 \\times J}` and the
target pre-period vector :math:`x_0^- \\in \\mathbb{R}^{T_0}`,
return

.. math::

   \\widehat{f} = \\arg\\min_{f \\in \\mathbb{R}^J}
                 \\| \\widetilde{M}^- f - x_0^- \\|_2^2,

the minimum-norm minimiser.

Optionally extends to ``p``-norm (ridge / lasso) regularisation via
``lambda_penalty``, ``p``, ``q`` (Appendix E of the paper compares OLS,
Ridge, Lasso): ``min_f ||x0 - M f||_2^2 + lambda ||f||_p^q``.

On the unregularised path (``lambda_penalty in {None, 0}``) the minimiser is
the truncated SVD solution :math:`\\sum_{i \\le r} (u_i^\\top x_0^- / s_i)\\,v_i`
(Golub & Van Loan 2013, Sec. 5.5.4), built from the top-``r`` singular factors
of the design, as the authors' implementation builds it (``tslib``'s
``pInverseMatrixFromSVD``). The rank ``r`` is the HSVT rank the caller passes,
or without one the design's numerical rank at ``max(T0, J) * eps * s_1``. The
rank is not decided by ``numpy.linalg.pinv``'s default tolerance of
``1e-15 * s_1``: an HSVT reconstruction has rank ``r`` in exact arithmetic
only, its trailing singular values are rounding (up to about 18 eps * s_1
measured), and a pseudoinverse that keeps one of them moves by at least
``1 / ||E||`` (Wedin 1973, BIT 13, Lemma 7.5 and Theorem 7.2; Bjorck 1996,
Sec. 1.4.2, Theorem 1.4.1). On a 1600 x 99 panel at ``r = 25`` that took the
weights to 1.9e12. ``mlsynth/tests/test_pcr_ols_rank.py`` holds the
construction and carries the full references.

The regularised path is solved directly with cvxpy, so this module has no
dependency on the legacy ``estutils`` optimizer.
"""

from __future__ import annotations

from typing import List, Optional

import cvxpy as cp
import numpy as np

from ....exceptions import MlsynthEstimationError
from ...pcr.core import pcr_weights

# Defaults mirror the historical estutils.Opt behaviour (ridge, p = q = 2).
_DEFAULT_P_NORM = 2
_DEFAULT_Q_EXPONENT = 2


def _check_rank(rank) -> int:
    """``rank`` as an ``int``; a bool, a non-integer or a value below 1 is refused."""
    if (isinstance(rank, (bool, np.bool_))
            or not isinstance(rank, (int, np.integer)) or rank < 1):
        raise MlsynthEstimationError(
            f"rank must be a positive integer; got {rank!r}.")
    return int(rank)


def solve_ols(
    denoised_donor_pre: np.ndarray,
    target_pre: np.ndarray,
    donor_names: Optional[List[str]] = None,
    lambda_penalty: Optional[float] = None,
    p: Optional[float] = None,
    q: Optional[float] = None,
    rank: Optional[int] = None,
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
        paper's plain OLS, the minimum-norm solution. Defaults are ridge
        (``p = q = 2``).
    rank : int, optional
        The HSVT rank ``r`` the design was truncated at. The unregularised
        weights are then the truncated SVD solution from the design's top-``r``
        singular factors, so the rank is the one the method fixed. Without it,
        the design's numerical rank at ``max(T0, J) * eps * s_1`` is used, the
        convention of :func:`numpy.linalg.lstsq`. In both cases a retained
        direction at or below that tolerance is dropped
        (:func:`mlsynth.utils.pcr.core.pcr_weights`). Validated whenever given;
        the regularised path does not use it.

    Raises
    ------
    MlsynthEstimationError
        If the design is not 2-D, its rows do not match the target, it has no
        pre-period rows or no donors, an input is not finite, ``rank`` is not a
        positive integer, or no direction of the design clears the tolerance
        (numerical rank 0, as for an all-zero design).
    """
    denoised_donor_pre = np.asarray(denoised_donor_pre, dtype=float)
    target_pre = np.asarray(target_pre, dtype=float)
    if denoised_donor_pre.ndim != 2:
        raise MlsynthEstimationError("denoised_donor_pre must be 2D (T0, J).")
    if denoised_donor_pre.shape[0] != target_pre.shape[0]:
        raise MlsynthEstimationError(
            f"Pre-period length mismatch: donors have {denoised_donor_pre.shape[0]} "
            f"rows but target has {target_pre.shape[0]}."
        )
    if 0 in denoised_donor_pre.shape:
        raise MlsynthEstimationError(
            f"The design has shape {denoised_donor_pre.shape}: no pre-period "
            "rows or no donors to weight."
        )
    if not (np.isfinite(denoised_donor_pre).all()
            and np.isfinite(target_pre).all()):
        raise MlsynthEstimationError(
            "denoised_donor_pre and target_pre must be finite.")
    if rank is not None:
        rank = _check_rank(rank)

    if lambda_penalty in (None, 0, 0.0):
        # Algorithm 2 Step 3, at the rank HSVT fixed (or, without one, at the
        # design's numerical rank). pinv(denoised_donor_pre) would decide the
        # rank again at 1e-15 * s_1, under the reconstruction's rounding.
        r = min(denoised_donor_pre.shape) if rank is None else rank
        return pcr_weights(denoised_donor_pre, target_pre, r)

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
