r"""Rho et al. (2026) Section 5 simulation helper for TASC.

Implements the state-space DGP from the TASC paper's ablation studies
(Section 5.2). Panels are drawn directly from TASC's own generative
model

.. math::

   x_t &= A x_{t-1} + q_{t-1}, \quad q_{t-1} \sim \mathcal{N}(0, Q), \\
   y_t &= H x_t + r_t,        \quad r_t \sim \mathcal{N}(0, R),

with :math:`x_0 \sim \mathcal{N}(0, Q)`, a stable random :math:`A`
(spectral radius :math:`\rho < 1`), and Gaussian :math:`H \in
\mathbb{R}^{N \times d}`. The covariances are scaled diagonals
:math:`Q = q_{\mathrm{scale}} I_d` and :math:`R = r_{\mathrm{scale}}
I_N`; the paper varies :math:`(q_{\mathrm{scale}}, r_{\mathrm{scale}})`
across four regimes (small/large state-perturbation × small/large
observation noise) -- the cells in Figures 3-4. The TASC paper reports
that with :math:`(q, r) = (0.01, 0.1)` the average :math:`|r_t|`
across simulations is about 0.084 and with :math:`(q, r) = (0.1, 1.0)`
it is about 0.836; these defaults reproduce both numbers.

The first row of the resulting :math:`N \times T` matrix is treated as
the target unit (post-period observations censored); the remaining
:math:`n = N - 1` rows are donors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TASCSample:
    """One draw from the Section 5 state-space DGP.

    Attributes
    ----------
    df : pd.DataFrame
        Long panel with columns ``unit`` / ``time`` / ``y`` / ``treat``
        ready for :class:`mlsynth.TASC`.
    Y : np.ndarray
        Full observation matrix, shape ``(N, T)``. Row 0 is the target.
    A, H : np.ndarray
        True transition matrix (``d x d``) and emission matrix
        (``N x d``) used to generate the data.
    Q, R : np.ndarray
        True state-perturbation and observation-noise covariances.
    N, T, T0, d : int
        Number of units (1 treated + donors), total periods, the
        post-period start index, and the latent state dimension.
    """

    df: pd.DataFrame
    Y: np.ndarray
    A: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    N: int
    T: int
    T0: int
    d: int


def _stable_transition(d: int, rho: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Draw a stable d-by-d matrix with spectral radius rho."""
    M = rng.standard_normal((d, d))
    eigvals = np.linalg.eigvals(M)
    return M * (rho / max(np.abs(eigvals).max(), 1e-8))


def simulate_tasc_sample(
    N: int = 38,
    T: int = 100,
    T0: int = 50,
    d: int = 5,
    q_scale: float = 0.01,
    r_scale: float = 0.1,
    rho: float = 0.95,
    rng: np.random.Generator | None = None,
) -> TASCSample:
    r"""Draw one sample from the TASC Section 5 DGP.

    Parameters
    ----------
    N : int, default ``38``
        Total units (1 treated + ``N - 1`` donors). Defaults to the
        donor-pool size of the paper's Prop 99 example.
    T, T0 : int
        Total periods and the post-period start index. The paper's
        defaults are ``T = 100``, ``T_0 = 50``.
    d : int, default ``5``
        True latent-state dimension. The paper's ablation sweeps
        ``d_true`` over :math:`\{3, 5, 10, 20\}`.
    q_scale, r_scale : float
        Diagonal scale of the state-perturbation and observation-noise
        covariances. Paper's regimes:

        ============= ============= =================
        Regime        q_scale       r_scale
        ============= ============= =================
        small Q, R    0.01          0.1
        small Q, big  0.01          1.0
        big Q, small  0.1           0.1
        big Q, R      0.1           1.0
        ============= ============= =================

    rho : float, default ``0.95``
        Target spectral radius of :math:`A` (stationary but persistent).
    rng : np.random.Generator, optional
        NumPy RNG. Defaults to ``np.random.default_rng()``.

    Returns
    -------
    TASCSample
    """
    rng = rng or np.random.default_rng()
    A = _stable_transition(d, rho, rng)
    H = rng.standard_normal((N, d))
    Q = q_scale * np.eye(d)
    R = r_scale * np.eye(N)

    x = rng.multivariate_normal(np.zeros(d), Q)
    Y = np.zeros((N, T))
    for t in range(T):
        if t > 0:
            x = A @ x + rng.multivariate_normal(np.zeros(d), Q)
        Y[:, t] = H @ x + rng.multivariate_normal(np.zeros(N), R)

    rows = [{"unit": "treated", "time": t, "y": float(Y[0, t]),
             "treat": int(t >= T0)} for t in range(T)]
    for j in range(1, N):
        rows.extend({"unit": f"donor{j:03d}", "time": t,
                     "y": float(Y[j, t]), "treat": 0} for t in range(T))

    return TASCSample(df=pd.DataFrame(rows), Y=Y, A=A, H=H, Q=Q, R=R,
                       N=N, T=T, T0=T0, d=d)
