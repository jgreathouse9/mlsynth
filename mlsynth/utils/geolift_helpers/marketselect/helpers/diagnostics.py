"""Pre-fit quality diagnostics for GeoLift market-selection scoring.

The fit's own RMSE comes free off the engine result (``BilevelSCMResult.pre_rmspe``
= ``sqrt(mean((y_pre - Y0_pre @ W)**2))``). The engine does *not* expose the
scaled L2 imbalance, so this module supplies it as one pure leaf.
"""

import numpy as np

from mlsynth.exceptions import MlsynthConfigError


def scaled_l2_imbalance(treated_pre, donor_pre, weights) -> float:
    """augsynth scaled L2 imbalance: ``||X0 w - X1|| / ||X0 w_unif - X1||``.

    The numerator is the fitted pre-period imbalance; the denominator is the
    imbalance under uniform donor weights (the plain donor average). The ratio
    is unitless: ``0`` = perfect pre-fit, ``1`` = no better than averaging all
    donors, ``>1`` = worse than the average.

    Computed on the same (raw) pre-period arrays the engine uses for
    ``pre_rmspe`` and the counterfactual, so RMSE and scaled-L2 share one
    consistent scale. (augsynth's own scaled-L2 is on the *centered* matching
    matrices; pass those instead for a centered/faithful variant.)

    Parameters
    ----------
    treated_pre : array-like, shape (T0,)
        Treated unit's pre-period outcomes (``X1``).
    donor_pre : array-like, shape (T0, J)
        Donor pre-period outcomes (``X0``).
    weights : array-like, shape (J,)
        Fitted donor weights ``W``.

    Returns
    -------
    float
        The scaled L2 imbalance, or ``nan`` if the uniform-weights imbalance is
        zero (the donor average already reproduces the treated pre-period
        exactly, so the ratio is undefined).

    Raises
    ------
    MlsynthConfigError
        If the array shapes are inconsistent.
    """
    X1 = np.asarray(treated_pre, dtype=float).ravel()
    X0 = np.asarray(donor_pre, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()

    if X0.ndim != 2:
        raise MlsynthConfigError(f"donor_pre must be 2-D (T0, J); got shape {X0.shape}.")
    T0, J = X0.shape
    if X1.shape[0] != T0:
        raise MlsynthConfigError(
            f"treated_pre length {X1.shape[0]} != donor_pre rows {T0}."
        )
    if w.shape[0] != J:
        raise MlsynthConfigError(f"weights length {w.shape[0]} != donor count {J}.")

    l2 = float(np.sqrt(np.sum((X0 @ w - X1) ** 2)))
    unif_l2 = float(np.sqrt(np.sum((X0 @ np.full(J, 1.0 / J) - X1) ** 2)))
    if unif_l2 == 0.0:
        return float("nan")
    return l2 / unif_l2
