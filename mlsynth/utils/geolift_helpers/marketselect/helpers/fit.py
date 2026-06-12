"""A single augsynth fit for GeoLift market-selection scoring.

The most important leaf of the scoring layer: fit the synthetic control once on
a pseudo-experiment's pre-period and return the weights, the level intercept,
and the two fit-quality metrics (RMSE and scaled L2 imbalance).

Two estimators, selectable per fit:

- ``augment="ridge"`` (default) -- Augmented SCM (augsynth). The bilevel engine
  centers per period by the donor mean internally, so the level is absorbed
  into the weights; the wrapper intercept is ``0`` and the prediction is
  ``Y0 @ W``.
- ``augment=None`` -- plain simplex SCM **with an intercept**. The outcomes are
  centered by their own pre-period means, the simplex weights are fit on the
  centered data, and the prediction is ``intercept + Y0 @ W`` (the demeaned /
  fixed-effect SCM). Fitting on centered data means ``pre_rmspe`` and
  ``scaled_l2`` come out already intercept-adjusted.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.bilevel.engine import BilevelSCM

from .diagnostics import scaled_l2_imbalance


def fit_intercept(y_pre, Y0_pre, weights) -> float:
    """Level-matching intercept: the mean pre-period residual ``y - Y0 @ w``.

    Equals ``mean(y_pre) - mean(Y0_pre, axis=0) @ w`` -- the scalar shift that
    aligns the treated and synthetic pre-period means.
    """
    y = np.asarray(y_pre, dtype=float).ravel()
    Y0 = np.asarray(Y0_pre, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    return float(np.mean(y - Y0 @ w))


def counterfactual(Y0, weights, intercept: float = 0.0) -> np.ndarray:
    """Synthetic outcome path ``intercept + Y0 @ weights`` over all rows of ``Y0``."""
    Y0 = np.asarray(Y0, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    return float(intercept) + Y0 @ w


@dataclass
class AugsynthFit:
    """Result of :func:`fit_augsynth_once`.

    Attributes
    ----------
    weights : np.ndarray
        Donor weights ``W``, shape ``(J,)``.
    intercept : float
        Level intercept (``0.0`` for ``augment="ridge"``).
    donor_names : list of str
        Donor labels, in ``weights`` order.
    pre_rmspe : float
        Pre-period RMSE of the fit (intercept-adjusted).
    scaled_l2 : float
        Scaled L2 imbalance of the fit (intercept-adjusted).
    augment : str or None
        The augmentation used (``"ridge"`` or ``None``).
    lambda_ : float or None
        The CV-selected ridge penalty (``augment="ridge"``); ``None`` for the
        simplex fit. Reuse it for the conformal refits so they fix the penalty
        instead of re-cross-validating (augsynth's behaviour).
    """

    weights: np.ndarray
    intercept: float
    donor_names: List[str]
    pre_rmspe: float
    scaled_l2: float
    augment: Optional[str]
    lambda_: Optional[float] = None

    def predict(self, Y0) -> np.ndarray:
        """Counterfactual path ``intercept + Y0 @ weights`` over all rows of ``Y0``."""
        return counterfactual(Y0, self.weights, self.intercept)


def fit_augsynth_once(
    y_pre,
    Y0_pre,
    *,
    augment: Optional[str] = "ridge",
    donor_names: Optional[List[str]] = None,
) -> AugsynthFit:
    """Fit the synthetic control once on a pseudo-experiment's pre-period.

    Parameters
    ----------
    y_pre : array-like, shape (T0,)
        Treated (aggregated candidate) pre-period outcomes.
    Y0_pre : array-like, shape (T0, J)
        Donor pre-period outcomes.
    augment : {"ridge", None}, default "ridge"
        ``"ridge"`` for Augmented SCM; ``None`` for plain simplex SCM with an
        intercept (demeaned SCM).
    donor_names : list of str, optional
        Donor labels (default ``donor_0 ...``).

    Returns
    -------
    AugsynthFit

    Raises
    ------
    MlsynthConfigError
        If ``augment`` is unknown, or the array shapes are inconsistent.
    """
    y = np.asarray(y_pre, dtype=float).ravel()
    Y0 = np.asarray(Y0_pre, dtype=float)
    if Y0.ndim != 2 or Y0.shape[0] != y.shape[0]:
        raise MlsynthConfigError(
            f"Y0_pre must be (T0, J) with T0={y.shape[0]}; got shape {Y0.shape}."
        )
    if augment not in ("ridge", None):
        raise MlsynthConfigError(f"augment must be 'ridge' or None; got {augment!r}.")

    J = Y0.shape[1]
    if donor_names is None:
        donor_names = [f"donor_{j}" for j in range(J)]

    # Pick the matching matrices: raw for ridge (it centers internally), or
    # mean-centered for the simplex-with-intercept path. Fitting on the centered
    # matrices makes pre_rmspe / scaled_l2 already intercept-adjusted.
    if augment == "ridge":
        A, B = y, Y0
    else:
        A, B = y - y.mean(), Y0 - Y0.mean(axis=0)

    res = BilevelSCM(augment=augment).fit(A, B, donor_names=donor_names)
    W = np.asarray(res.W, dtype=float)
    intercept = 0.0 if augment == "ridge" else fit_intercept(y, Y0, W)

    return AugsynthFit(
        weights=W,
        intercept=float(intercept),
        donor_names=list(donor_names),
        pre_rmspe=float(res.pre_rmspe),
        scaled_l2=float(scaled_l2_imbalance(A, B, W)),
        augment=augment,
        lambda_=(float(res.lambda_) if res.lambda_ is not None else None),
    )
