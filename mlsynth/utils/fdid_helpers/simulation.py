r"""Web Appendix E Monte Carlo DGPs for the Forward DiD method.

Implements the four data-generating processes from Li, Shi & Huang (2023)
Web Appendix E. Each draw produces one treated unit and ``N`` controls over
``T1 + T2`` periods, generated from three common factors:

.. math::

   f_{1t} &= 0.8 f_{1,t-1} + v_{1t}, \\
   f_{2t} &= -0.6 f_{2,t-1} + v_{2t} + 0.8 v_{2,t-1}, \\
   f_{3t} &= v_{3t} + 0.9 v_{3,t-1} + 0.4 v_{3,t-2},

with :math:`v_{kt} \sim \mathcal{N}(0, 1)` and outcomes

.. math::

   y_{tr,t} &= a_0 + c_0 \mathbf{1}' f_t + \varepsilon_{tr,t}, \\
   y_{it}   &= 1   + c_1 \mathbf{1}' f_t + \varepsilon_{it}  \quad i \le N/2, \\
   y_{it}   &= 1   + c_2 \mathbf{1}' f_t + \varepsilon_{it}  \quad i > N/2,

where :math:`\varepsilon_{it} \sim \mathcal{N}(0, 1)`. The four DGPs vary
:math:`(a_0, c_0, c_1, c_2)`::

    DGP  (a_0, c_0, c_1, c_2)
    1    (1, 1, 1, 1) — all controls match (DiD is applicable)
    2    (1, 1, 1, 2) — half the controls have mismatched loadings
    3    (2, 1, 1, 1) — treated has a different intercept
    4    (2, 1, 1, 2) — intercept and half-mismatched loadings

True ATT is zero in every DGP (matching the paper's PMSE convention; the
PMSE is invariant to a constant treatment effect).

Note
----
The appendix prints ``f_2t = -0.6 f_{1,t-1} + ...`` for the lag term, but
the Monte Carlo numbers in Li's Table 5 match the alternative reading
``-0.6 f_{2,t-1}`` (ARMA(1,1) on :math:`f_2` itself). The latter is used
here — it reproduces the paper's DID PMSE values closely (within ~3%) while
the literal reading reproduces only the FDID column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


_DGP_PARAMS: dict[int, Tuple[float, float, float, float]] = {
    1: (1.0, 1.0, 1.0, 1.0),
    2: (1.0, 1.0, 1.0, 2.0),
    3: (2.0, 1.0, 1.0, 1.0),
    4: (2.0, 1.0, 1.0, 2.0),
}


@dataclass(frozen=True)
class FDIDSample:
    """One draw from a Web Appendix E DGP.

    Attributes
    ----------
    df : pd.DataFrame
        Long panel with columns ``unit`` / ``time`` / ``y`` / ``treat`` ready
        to feed to :class:`mlsynth.FDID`.
    Y_treated : np.ndarray
        Treated-unit outcome path, shape ``(T,)``.
    Y_controls : np.ndarray
        Control outcomes, shape ``(N, T)``. Rows 0..N//2-1 carry loading
        ``c_1``; rows N//2..N-1 carry loading ``c_2``.
    T1, T2 : int
        Pre- / post-treatment period counts.
    dgp : int
        Which of the four DGPs was drawn.
    """

    df: pd.DataFrame
    Y_treated: np.ndarray
    Y_controls: np.ndarray
    T1: int
    T2: int
    dgp: int


def _factors(T: int, rng: np.random.Generator, burn: int = 50) -> np.ndarray:
    """Three common factor paths, shape ``(T, 3)``.

    A burn-in of 50 periods washes out the deterministic-zero initial
    condition; the returned array is the stationary tail of length ``T``.
    """
    L = T + burn
    v = rng.standard_normal((L, 3))
    f = np.zeros((L, 3))
    for t in range(1, L):
        f[t, 0] = 0.8 * f[t - 1, 0] + v[t, 0]
        f[t, 1] = -0.6 * f[t - 1, 1] + v[t, 1] + 0.8 * v[t - 1, 1]
        lag2 = v[t - 2, 2] if t >= 2 else 0.0
        f[t, 2] = v[t, 2] + 0.9 * v[t - 1, 2] + 0.4 * lag2
    return f[burn:]


def simulate_fdid_sample(
    dgp: int,
    N: int = 60,
    T1: int = 24,
    T2: int = 12,
    rng: np.random.Generator | None = None,
) -> FDIDSample:
    """Draw one sample from FDID Web Appendix E DGP ``dgp`` (1-4).

    Parameters
    ----------
    dgp : int
        Which DGP to draw (1, 2, 3, or 4).
    N : int, default 60
        Number of control units (the paper uses ``N = 60``).
    T1, T2 : int
        Pre- and post-treatment period counts.
    rng : np.random.Generator, optional
        NumPy RNG. Defaults to ``np.random.default_rng()``.

    Returns
    -------
    FDIDSample
    """
    if dgp not in _DGP_PARAMS:
        raise ValueError(f"dgp must be in {{1, 2, 3, 4}}; got {dgp}.")
    rng = rng or np.random.default_rng()
    a0, c0, c1, c2 = _DGP_PARAMS[dgp]
    T = T1 + T2
    f_sum = _factors(T, rng).sum(axis=1)               # 1' f_t, shape (T,)

    # Treated outcome; the paper convention sets the true ATT to zero so that
    # PMSE = E[\widehat{ATT}^2] is the squared bias plus variance.
    eps_tr = rng.standard_normal(T)
    y_tr = a0 + c0 * f_sum + eps_tr

    # Controls in two loading groups
    half = N // 2
    eps = rng.standard_normal((N, T))
    y_ctrl = np.empty((N, T))
    y_ctrl[:half, :] = 1.0 + c1 * f_sum[None, :] + eps[:half, :]
    y_ctrl[half:, :] = 1.0 + c2 * f_sum[None, :] + eps[half:, :]

    rows = [{"unit": "treated", "time": t, "y": float(y_tr[t]),
             "treat": int(t >= T1)} for t in range(T)]
    for i in range(N):
        rows.extend({"unit": f"c{i:03d}", "time": t,
                     "y": float(y_ctrl[i, t]), "treat": 0}
                    for t in range(T))
    df = pd.DataFrame(rows)
    return FDIDSample(df=df, Y_treated=y_tr, Y_controls=y_ctrl,
                      T1=T1, T2=T2, dgp=dgp)
