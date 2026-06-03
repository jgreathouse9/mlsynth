"""CV-blended ensemble of the SIV and projected estimators.

Section 5.1 of Gulek and Vives-i-Bastida (2024) constructs::

    \\hat\\theta^E(\\alpha) = \\alpha \\hat\\theta^{SIV}
                            + (1 - \\alpha) \\hat\\theta^P,

with ``\\alpha`` picked on a held-out validation block by minimising
the MSE of the convex combination of debiased outcomes::

    \\alpha^* = \\arg\\min_{\\alpha \\in [0, 1]}
        \\frac{1}{J (T_0 - T_v)}
        \\|\\alpha \\tilde Y^{P, T_v} + (1 - \\alpha) \\tilde Y^{T_v}\\|^2_2

evaluated over the validation block ``(T_v, T_0]``. With one
scalar parameter on a compact interval the minimum is closed-form::

    Define a_t = \\tilde Y^P_{i,t} - \\tilde Y^{SIV}_{i,t},
           b_t = \\tilde Y^{SIV}_{i,t};
    then ||\\alpha a + b||^2 is minimised at
           \\alpha^* = -(a \\cdot b) / (a \\cdot a),
    clipped to ``[0, 1]``.
"""

from __future__ import annotations

import numpy as np


def select_alpha(
    Y_tilde_siv_val: np.ndarray,
    Y_tilde_proj_val: np.ndarray,
) -> float:
    """Closed-form solution of the validation-block convex blend.

    Parameters
    ----------
    Y_tilde_siv_val, Y_tilde_proj_val : np.ndarray
        ``(J, T_val)`` debiased-outcome residuals over the validation
        block for the SIV and projected pipelines.

    Returns
    -------
    float
        Optimal ``alpha`` clipped to ``[0, 1]``.
    """

    siv = Y_tilde_siv_val.reshape(-1)
    proj = Y_tilde_proj_val.reshape(-1)
    if siv.size == 0:
        return 1.0

    # We minimise ||alpha proj + (1 - alpha) siv||^2 in alpha
    diff = proj - siv
    denom = float(diff @ diff)
    if denom <= 1e-12:
        return 1.0
    alpha = -float(diff @ siv) / denom
    return float(np.clip(alpha, 0.0, 1.0))
