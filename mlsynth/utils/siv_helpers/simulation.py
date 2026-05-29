r"""Gulek & Vives (2024) Section 6 simulation helper for SIV.

Implements the Syrian-calibrated Monte Carlo used in Table 1 of the
paper to compare 2SLS-TWFE with the Synthetic IV (SIV) estimator. The
DGP couples treatment ``R`` to outcome ``Y`` through both a common
factor structure :math:`\mu_i' f_t` *and* an idiosyncratic correlation
:math:`\mathrm{corr}(\varepsilon_{it}, \eta_{it}) = \rho`, so 2SLS-TWFE
is biased even with valid post-period assignment of the instrument.

The structural model is

.. math::

   Y_{it} &= \theta R_{it} + \mu_i' f_t + \varepsilon_{it}, \\
   R_{it} &= (\gamma Z_{it} + \eta_{it}) \cdot \mathbf{1}\{t \ge T_0\}, \\
   Z_{it} &= (Z_i' g_t) \cdot \mathbf{1}\{t \ge T_0\},

with AR(1) factor :math:`f_t = \kappa f_{t-1} + u_{f,t}` and AR(1)
instrument-share :math:`g_t = \kappa g_{t-1} + u_{g,t}`. The shocks
share three correlated bivariate normals,

.. math::

   (u_{f,t}, u_{g,t}) &\sim \mathcal{N}\!\left(0,
       \begin{pmatrix} \sigma_f^2 & \rho_g\sigma_f\sigma_g \\
                       \rho_g\sigma_f\sigma_g & \sigma_g^2 \end{pmatrix}\right), \\
   (Z_i, \mu_i) &\sim \mathcal{N}\!\left(0,
       \begin{pmatrix} \sigma_z^2 & \rho_z\sigma_z\sigma_\mu \\
                       \rho_z\sigma_z\sigma_\mu & \sigma_\mu^2 \end{pmatrix}\right), \\
   (\varepsilon_{it}, \eta_{it}) &\sim \mathcal{N}\!\left(0,
       \begin{pmatrix} \sigma_\varepsilon^2 & \rho\sigma_\varepsilon\sigma_\lambda \\
                       \rho\sigma_\varepsilon\sigma_\lambda & \sigma_\lambda^2 \end{pmatrix}\right).

Defaults match the Syrian calibration of Section 6: :math:`J = 26`
donors, :math:`T = 16` periods, :math:`T_0 = 10`, true coefficient
:math:`\theta = -0.16`, :math:`\sigma_\varepsilon^2 = \sigma_\eta^2 =
0.035`, :math:`\sigma_\mu = 0.5`, :math:`\sigma_z = 0.2`, :math:`\kappa
= 0.5`, :math:`\gamma = 1`. The three correlation knobs :math:`\rho =
\rho_z = \rho_g` are passed jointly as ``r``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SIVSample:
    """One draw from the Section 6 DGP.

    Attributes
    ----------
    df : pd.DataFrame
        Long panel with columns ``unit`` / ``time`` / ``y`` / ``r`` /
        ``z`` ready for :class:`mlsynth.SIV`.
    Y, R, Z : np.ndarray
        Outcome, treatment, and instrument arrays, shape ``(J, T)``.
    J, T, T0 : int
        Donor count, periods, and the post-period start index.
    """

    df: pd.DataFrame
    Y: np.ndarray
    R: np.ndarray
    Z: np.ndarray
    J: int
    T: int
    T0: int


def simulate_siv_sample(
    J: int = 26,
    T: int = 16,
    T0: int = 10,
    theta: float = -0.16,
    kappa: float = 0.5,
    sigma_eps: float = float(np.sqrt(0.035)),
    sigma_lam: float = float(np.sqrt(0.035)),
    sigma_mu: float = 0.5,
    sigma_z: float = 0.2,
    sigma_f: float = 0.2,
    sigma_g: float = 1.0,
    gamma: float = 1.0,
    r: float = 0.5,
    rng: np.random.Generator | None = None,
) -> SIVSample:
    r"""Draw one sample from the Gulek & Vives Section 6 DGP.

    Parameters
    ----------
    J, T, T0 : int
        Number of donor units, total periods, and post-period start.
    theta : float, default ``-0.16``
        True structural coefficient on the (endogenous) treatment.
    kappa : float, default ``0.5``
        AR(1) persistence shared by the factor and the instrument share.
    sigma_eps, sigma_lam : float
        Standard deviations of :math:`\varepsilon_{it}` and
        :math:`\eta_{it}`; defaults reproduce the Syrian
        :math:`\sigma^2 = 0.035`.
    sigma_mu, sigma_z, sigma_f, sigma_g : float
        Cross-section and time-series scale parameters.
    gamma : float, default ``1.0``
        First-stage slope of :math:`R_{it}` on :math:`Z_{it}`.
    r : float, default ``0.5``
        Single correlation knob applied jointly to
        :math:`(\rho, \rho_z, \rho_g)`. Section 6 sweeps
        :math:`r \in \{0.5, 0.7, 0.9\}`.
    rng : np.random.Generator, optional
        NumPy RNG. Defaults to ``np.random.default_rng()``.

    Returns
    -------
    SIVSample
    """
    rng = rng or np.random.default_rng()
    cov_fg = np.array([[sigma_f ** 2, r * sigma_f * sigma_g],
                        [r * sigma_f * sigma_g, sigma_g ** 2]])
    uf_ug = rng.multivariate_normal([0.0, 0.0], cov_fg, size=T)
    f = np.zeros(T); g = np.zeros(T)
    for t in range(1, T):
        f[t] = kappa * f[t - 1] + uf_ug[t, 0]
        g[t] = kappa * g[t - 1] + uf_ug[t, 1]

    cov_zm = np.array([[sigma_z ** 2, r * sigma_z * sigma_mu],
                        [r * sigma_z * sigma_mu, sigma_mu ** 2]])
    zm = rng.multivariate_normal([0.0, 0.0], cov_zm, size=J)
    Z_unit = zm[:, 0]
    mu = zm[:, 1]

    cov_eh = np.array([[sigma_eps ** 2, r * sigma_eps * sigma_lam],
                        [r * sigma_eps * sigma_lam, sigma_lam ** 2]])
    eh = rng.multivariate_normal([0.0, 0.0], cov_eh, size=(J, T))
    eps = eh[:, :, 0]; eta = eh[:, :, 1]

    post = (np.arange(T) >= T0).astype(float)
    Z = np.outer(Z_unit, g) * post[None, :]
    R = (gamma * Z + eta) * post[None, :]
    Y = theta * R + np.outer(mu, f) + eps

    rows = [{"unit": f"u{i:02d}", "time": t,
             "y": float(Y[i, t]), "r": float(R[i, t]), "z": float(Z[i, t])}
            for i in range(J) for t in range(T)]
    return SIVSample(df=pd.DataFrame(rows), Y=Y, R=R, Z=Z, J=J, T=T, T0=T0)
