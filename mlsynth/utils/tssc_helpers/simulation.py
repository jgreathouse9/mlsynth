r"""Li & Shankar (2023) Figure 2 simulation helper for TSSC.

Implements the Monte Carlo design used in
``TSSC_Figure2_MSE_Ratio.m`` from the paper's replication package:
three latent common factors driving every unit with **homogeneous**
loadings :math:`b = [1, 1, 1]'`, plus an additive intercept and iid
:math:`\mathcal{N}(0, 1)` idiosyncratic noise. The treated unit's index
is unit 0; donors are units :math:`1, \ldots, N - 1`.

.. math::

   y_{kt} = \alpha + f_t' b_k + \varepsilon_{kt}, \qquad
   k = 1, \ldots, N, \quad t = 1, \ldots, T,

with :math:`b_k = (1, 1, 1)'` for every unit (the "SC restrictions
hold" regime, where plain SC dominates MSCc in MSE), :math:`\alpha = 1`,
and the three factors:

.. math::

   f_{1, t+1} &= 0.2(t+1) - 0.8\sqrt{t+1} + 0.8 f_{1, t} + u_{1, t},  \\
   f_{2, t+1} &= -0.6 f_{2, t} + u_{2, t+1} + 0.8 u_{2, t}, \\
   f_{3, t+2} &= u_{3, t+2} + 0.9 u_{3, t+1} + 0.4 u_{3, t},

with :math:`u_{kt} \sim \mathcal{N}(0, 1)` and initial values zero.
:math:`f_1` is a nonlinear AR(1) trend; :math:`f_2` is ARMA(1, 1);
:math:`f_3` is MA(2). True ATT is zero — the MATLAB code adds a
treatment-effect path :math:`\Delta_t` but its size :math:`C_{TE} = 0`
in the published Figure 2 setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TSSCSample:
    """One draw from the Figure 2 DGP.

    Attributes
    ----------
    df : pd.DataFrame
        Long panel with columns ``unit`` / ``time`` / ``y`` / ``treat``
        ready for :class:`mlsynth.TSSC`.
    y_treated : np.ndarray
        Treated outcome over the full timeline, shape ``(T,)``.
    donors : np.ndarray
        Donor outcomes, shape ``(T, N_co)``.
    factors : np.ndarray
        Common factor matrix, shape ``(T, 3)``.
    T1, T2, N_co : int
        Pre-treatment periods, post-treatment periods, and donor count.
    """

    df: pd.DataFrame
    y_treated: np.ndarray
    donors: np.ndarray
    factors: np.ndarray
    T1: int
    T2: int
    N_co: int


def _factors(T: int, rng: np.random.Generator) -> np.ndarray:
    """Three latent factors (nonlinear AR(1) + ARMA(1,1) + MA(2))."""
    u1 = rng.standard_normal(T)
    u2 = rng.standard_normal(T)
    u3 = rng.standard_normal(T)
    f1 = np.zeros(T); f2 = np.zeros(T); f3 = np.zeros(T)
    for k in range(T - 1):
        f1[k + 1] = 0.2 * (k + 1) - 0.8 * np.sqrt(k + 1) + 0.8 * f1[k] + u1[k]
    for k in range(T - 2):
        f2[k + 1] = -0.6 * f2[k] + u2[k + 1] + 0.8 * u2[k]
    for k in range(T - 2):
        f3[k + 2] = u3[k + 2] + 0.9 * u3[k + 1] + 0.4 * u3[k]
    return np.column_stack([f1, f2, f3])


def simulate_tssc_sample(
    T1: int = 76,
    T2: int = 34,
    N_co: int = 10,
    alpha: float = 1.0,
    rng: np.random.Generator | None = None,
) -> TSSCSample:
    r"""Draw one sample from the Li & Shankar Figure 2 DGP.

    Defaults match the MATLAB Mock_data_code dimensions (``T = 110``,
    ``T_1 = 76``, ``N_{co} = 10``); pass smaller ``T1``/``T2`` for the
    Figure 2 sweep.

    Parameters
    ----------
    T1, T2 : int
        Pre- and post-treatment period counts.
    N_co : int
        Number of donor units (10 in the paper's left-panel exercise,
        30 in the right-panel exercise).
    alpha : float, default 1.0
        Constant added to every unit's outcome.
    rng : np.random.Generator, optional
        NumPy RNG. Defaults to ``np.random.default_rng()``.

    Returns
    -------
    TSSCSample
    """
    rng = rng or np.random.default_rng()
    T = T1 + T2
    N = N_co + 1
    f = _factors(T, rng)                                # (T, 3)
    b = np.ones((3, N))                                  # homogeneous loadings
    eps = rng.standard_normal((T, N))
    y_all = alpha + f @ b + eps                          # (T, N), ATT = 0
    y_tr   = y_all[:, 0]
    donors = y_all[:, 1:]                                # (T, N_co)

    rows = [{"unit": "treated", "time": t, "y": float(y_tr[t]),
             "treat": int(t >= T1)} for t in range(T)]
    for j in range(N_co):
        rows.extend({"unit": f"donor{j:02d}", "time": t,
                     "y": float(donors[t, j]), "treat": 0} for t in range(T))

    return TSSCSample(df=pd.DataFrame(rows), y_treated=y_tr, donors=donors,
                       factors=f, T1=T1, T2=T2, N_co=N_co)
