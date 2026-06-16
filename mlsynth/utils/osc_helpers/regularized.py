"""Regularized nuisance estimation for OSC: the control weights (delta) and the
orthogonalization weights (eta).

Both solve a penalized program that picks, among the values making the sample
moment conditions near zero, the one minimizing a penalty -- this drives the
(partially identified) nuisance to a unique element of the identified set. delta
is simplex-constrained (a synthetic control); eta is normalized so the
orthogonalized moments identify the ATT.
"""
from __future__ import annotations

import numpy as np


def estimate_delta(pre_y0, pre_yj, Z, scaled: bool = True):
    """Regularized IV control weights.

    Parameters
    ----------
    pre_y0 : ndarray (T0,)         treated unit, pre-period
    pre_yj : ndarray (J, T0)       control units x pre-period
    Z      : ndarray (Q, T0)       instrument units x pre-period
    scaled : bool                  rescale each series by its variance first

    Returns
    -------
    dict with ``delta`` (J,) on the simplex and ``lambda_`` (the tuning value).
    """
    raise NotImplementedError


def estimate_eta(pre_y0, pre_yj, post_y0, post_yj, Z, scaled: bool = True):
    """Regularized, normalized orthogonalization weights ``eta``.

    Returns ``dict`` with ``eta`` (1d) and ``lambda_``.
    """
    raise NotImplementedError
