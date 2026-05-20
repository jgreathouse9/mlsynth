"""Pre-treatment estimation/holdout split helpers for SPCD inference.

Implements the train-on-E / calibrate-on-B discipline used by LEXSCM and
adapted here for SPCD. The first ``frac_E`` of the pretreatment window
is used to fit the design; the remaining periods are held out and used
to produce out-of-sample residuals for conformal inference and power
analysis.

Key guarantee
-------------
With ``enable_inference=True`` the design is fit on ``Y_E`` only,
regardless of whether ``Y_post`` is supplied. This means a user who has
not yet treated (planning mode) and a user who has treated
(retrospective mode) both receive the *same* recommended design as long
as their pretreatment data are identical.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError


def split_pre_window(
    Y_pre: np.ndarray, frac_E: float = 0.7, min_blank_size: int = 5
) -> Tuple[np.ndarray, np.ndarray, int, int, bool]:
    """Split the pretreatment matrix into estimation (E) and holdout (B).

    Parameters
    ----------
    Y_pre : np.ndarray
        Pretreatment outcome matrix of shape ``(T_pre, N)``.
    frac_E : float
        Fraction of pretreatment periods to use for estimation. The
        remaining ``1 - frac_E`` periods form the holdout window. Values
        outside ``[0.1, 0.95]`` are rejected.
    min_blank_size : int
        Minimum number of holdout periods required for inference to be
        meaningful. If ``T_B < min_blank_size`` the function still
        returns a valid split (so that the design can be fit on
        ``Y_E``) but flags ``can_infer=False`` so the caller can skip
        the inference / MDE machinery with a warning.

    Returns
    -------
    Y_E : np.ndarray
        Estimation window of shape ``(n_E, N)``. ``n_E = floor(T_pre * frac_E)``.
    Y_B : np.ndarray
        Holdout window of shape ``(n_B, N)``. ``n_B = T_pre - n_E``.
    n_E : int
        Number of periods in the estimation window.
    n_B : int
        Number of periods in the holdout window.
    can_infer : bool
        ``True`` if ``n_B >= min_blank_size``, ``False`` otherwise.

    Raises
    ------
    MlsynthConfigError
        If ``frac_E`` is out of range.
    MlsynthDataError
        If the resulting estimation window has fewer than 2 periods, in
        which case SPCD cannot be fit at all.
    """

    if not 0.1 <= frac_E <= 0.95:
        raise MlsynthConfigError(
            f"holdout_frac_E={frac_E} is outside the supported range [0.1, 0.95]."
        )

    T_pre = Y_pre.shape[0]
    n_E = int(T_pre * frac_E)
    n_B = T_pre - n_E

    if n_E < 2:
        raise MlsynthDataError(
            f"Estimation window has only {n_E} periods after applying "
            f"frac_E={frac_E} to T_pre={T_pre}. SPCD requires at least 2."
        )

    Y_E = Y_pre[:n_E, :]
    Y_B = Y_pre[n_E:, :]

    can_infer = n_B >= min_blank_size
    return Y_E, Y_B, n_E, n_B, can_infer


def compute_holdout_residuals(
    Y_B: np.ndarray, contrast_weights: np.ndarray
) -> np.ndarray:
    """Compute out-of-sample residuals on the holdout window.

    The contrast weights were learned on ``Y_E`` only, so

        r_B = Y_B @ contrast_weights

    is the synthetic gap on data the optimization never saw. Under
    the linear-factor model with no treatment, ``r_B`` is a zero-mean
    noise series whose empirical distribution is used as the
    calibration set for conformal inference and the noise pool for
    Monte Carlo MDE.

    Parameters
    ----------
    Y_B : np.ndarray
        Holdout window of shape ``(n_B, N)``.
    contrast_weights : np.ndarray
        Length-N signed weights from the SPCD design fit on ``Y_E``.

    Returns
    -------
    residuals_B : np.ndarray
        Length-``n_B`` out-of-sample synthetic gap.
    """

    return Y_B @ contrast_weights
