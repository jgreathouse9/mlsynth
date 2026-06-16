"""The orthogonalized ATT and its pre/post moment residuals.

Given the regularized nuisance estimates (delta, eta), the ATT is read off the
orthogonalized moment conditions; because they are Neyman-orthogonal to the
control weights, beta is insensitive to which delta in the identified set was
chosen. The pre/post residual paths feed the Series-HAC variance.
"""
from __future__ import annotations

import numpy as np


def orthogonalized_att(pre_y0, pre_yj, Z, post_y0, post_yj, delta, eta,
                       include_constant: bool = True):
    """Compute the orthogonalized ATT and the moment residual paths.

    Returns
    -------
    dict with ``beta`` (float), ``preg`` (Q-1, T0), ``postg`` (T1,).
    """
    raise NotImplementedError
