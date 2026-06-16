"""The orthogonalized ATT and its pre/post moment residuals.

Given the regularized nuisance estimates (delta, eta), the ATT is read off the
orthogonalized moment conditions on the *unscaled* outcomes; because the moments
are Neyman-orthogonal to the control weights, beta is insensitive to which delta
in the identified set was chosen. The pre/post residual paths feed the
Series-HAC variance.
"""
from __future__ import annotations

import numpy as np

from mlsynth.exceptions import MlsynthEstimationError


def orthogonalized_att(pre_y0, pre_yj, Z, post_y0, post_yj, delta, eta,
                       include_constant: bool = True):
    """Compute the orthogonalized ATT and the moment residual paths.

    Returns ``dict`` with ``beta`` (float), ``preg`` (Q, T0), ``postg`` (T1,).
    """
    y0 = np.asarray(pre_y0, float).ravel()
    YJ = np.atleast_2d(np.asarray(pre_yj, float))
    Z = np.atleast_2d(np.asarray(Z, float))
    py0 = np.asarray(post_y0, float).ravel()
    PYJ = np.atleast_2d(np.asarray(post_yj, float))
    delta = np.asarray(delta, float).ravel()
    eta = np.asarray(eta, float).ravel()
    J, T0 = YJ.shape
    T1 = PYJ.shape[1]
    if include_constant:
        Z = np.vstack([Z, np.ones(T0)])
    if eta[-1] == 0.0:
        raise MlsynthEstimationError("eta's post-moment weight is zero; cannot normalize.")

    # Stacked moments: instrument pre-moments then the post-period ATT moment.
    g0 = np.concatenate([Z @ y0 / T0, [py0.mean()]])              # (Q+1,)
    gdelta = np.vstack([Z @ YJ.T / T0, PYJ.mean(axis=1)[None, :]])  # (Q+1, J)
    beta = float(eta @ (g0 - gdelta @ delta) / eta[-1])

    postg = py0 - PYJ.T @ delta - beta                           # (T1,)
    preg = Z * (y0 - YJ.T @ delta)[None, :]                      # (Q, T0)
    return {"beta": beta, "preg": preg, "postg": postg}
