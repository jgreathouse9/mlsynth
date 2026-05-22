"""Frequentist OLS weight solver for PCR-SC.

Implements the *learn* step of Algorithm 2 of Rho et al. (2025):
given the HSVT-denoised pre-period donor matrix
:math:`\\widetilde{M}^- \\in \\mathbb{R}^{T_0 \\times J}` and the
target pre-period vector :math:`x_0^- \\in \\mathbb{R}^{T_0}`,
return

.. math::

   \\widehat{f} = \\arg\\min_{f \\in \\mathbb{R}^J}
                 \\| \\widetilde{M}^- f - x_0^- \\|_2^2.

Optionally extends to elastic-net-style regularisation via
``lambda_penalty``, ``p``, ``q`` (Appendix E of the paper compares OLS,
Ridge, Lasso). For the unregularised path (``lambda_penalty in {None, 0}``)
the closed-form pseudo-inverse solver is used -- which is the paper's
exact Algorithm 2 Step 3.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ....exceptions import MlsynthEstimationError
from ...estutils import Opt


def solve_ols(
    denoised_donor_pre: np.ndarray,
    target_pre: np.ndarray,
    donor_names: List[str],
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
    donor_names : list of str
        Donor labels, required by :class:`Opt.SCopt` when regularisation
        is on.
    lambda_penalty, p, q : float or None
        Optional elastic-net knobs. ``None`` (or ``0``) means the paper's
        plain OLS via pseudo-inverse.
    """
    if denoised_donor_pre.ndim != 2:
        raise MlsynthEstimationError(
            "denoised_donor_pre must be 2D (T0, J)."
        )
    if denoised_donor_pre.shape[0] != target_pre.shape[0]:
        raise MlsynthEstimationError(
            f"Pre-period length mismatch: donors have {denoised_donor_pre.shape[0]} "
            f"rows but target has {target_pre.shape[0]}."
        )

    if lambda_penalty in (None, 0, 0.0):
        # Paper Algorithm 2: f_hat = pinv(M_pre) @ x_pre.
        return np.linalg.pinv(denoised_donor_pre) @ target_pre

    problem = Opt.SCopt(
        num_control_units=denoised_donor_pre.shape[1],
        target_outcomes_pre_treatment=target_pre,
        num_pre_treatment_periods=target_pre.shape[0],
        donor_outcomes_pre_treatment=denoised_donor_pre,
        scm_model_type="OLS",
        donor_names=list(donor_names),
        lambda_penalty=lambda_penalty,
        p=p,
        q=q,
    )
    primal = problem.solution.primal_vars
    return np.asarray(primal[next(iter(primal))], dtype=float)
