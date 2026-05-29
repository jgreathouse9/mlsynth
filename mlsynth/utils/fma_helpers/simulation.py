r"""Li & Sonnier (2023) DGP1 / DGP2 simulation helpers for the FMA estimator.

Implements the data-generating processes from the paper's Web Appendix E.

**DGP1** (Section 4.1, stationary factors, Hsiao-Ching-Wan 2012):

.. math::

   f_{1t} &= 0.8\, f_{1,t-1} + v_{1t}, \\
   f_{2t} &= -0.68\, f_{2,t-1} + v_{2t} + 0.8\, v_{2,t-1}, \\
   f_{3t} &= v_{3t} + 0.9\, v_{3,t-1} + 0.4\, v_{3,t-2},
   \qquad v_{kt} \sim \mathcal{N}(0, 1).

**DGP2** (Web Appendix E.1, non-stationary factors):

.. math::

   F_{1t} &= (0.2 + \xi_t)\, t + \varepsilon_{5t}
       \qquad \text{(linear trend, random slope)} \\
   F_{2t} &= F_{2,t-1} + \varepsilon_{4t}
       \qquad \text{(drift-less unit root)} \\
   F_{3t} &= \sqrt{t} + \varepsilon_{6t}
            + 0.9\, \varepsilon_{6,t-1} + 0.4\, \varepsilon_{6,t-2}
       \qquad \text{(sqrt-}t\text{ trend, MA(2) errors)}

with :math:`\xi_t \sim \mathrm{Uniform}[0,1]` and
:math:`\varepsilon_{kt} \sim \mathcal{N}(0, 1)`. (The paper's TeX prints
``F_{2t} = F_{4,t-1} + ε_{4t}``, which is a typo; the surrounding text
describes :math:`F_{2t}` as a drift-less unit root, so ``F_{4,t-1}`` reads
as ``F_{2,t-1}``.)

Either DGP feeds the same factor model

.. math::

   y_{it}^0 = \alpha + F_t' \lambda_i + e_{it}, \qquad i = 1, \ldots, N,
   \quad t = 1, \ldots, T,

with loadings :math:`\lambda_{il} \stackrel{\text{iid}}{\sim} \mathcal{N}(1, 1)`
and idiosyncratic shocks

.. math::

   e_{1t} \sim \mathcal{N}(0, \sigma_{\text{tr}}^2), \qquad
   e_{it} \sim \mathcal{N}(0, \sigma_{\text{co}}^2) \quad (i \ge 2).

The paper studies three variance regimes via the keyword ``variance_case``:

* ``"equal"``           — :math:`\sigma_{\text{tr}}^2 = \sigma_{\text{co}}^2 = 1`
  (Figures 2, W.5; both the new CI and Xu's bootstrap nominal);
* ``"treated_smaller"`` — :math:`\sigma_{\text{tr}} = 0.5 \sigma_{\text{co}}`
  (Figures 3, W.6; Xu's bootstrap overcovers);
* ``"treated_larger"``  — :math:`\sigma_{\text{tr}} = 2 \sigma_{\text{co}}`
  (Figures 4, W.7; Xu's bootstrap undercovers).

True ATT = 0 in every draw; the paper's centred statistic
:math:`\sqrt{T_2}(\hat\Delta_1 - \Delta_1)` is invariant to a constant
treatment effect (see the equation following Equation 4.2), so coverage is
unaffected by setting :math:`\Delta_{1t} = 0`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import pandas as pd


VarianceCase = Literal["equal", "treated_smaller", "treated_larger"]
DGP = Literal["dgp1", "dgp2"]

_VARIANCE_CASE_PARAMS: dict[str, Tuple[float, float]] = {
    "equal":           (1.0, 1.0),    # Figures 2, W.5
    "treated_smaller": (0.5, 1.0),    # Figures 3, W.6
    "treated_larger":  (2.0, 1.0),    # Figures 4, W.7
}


@dataclass(frozen=True)
class FMASample:
    """One draw from a Li & Sonnier DGP.

    Attributes
    ----------
    df : pd.DataFrame
        Long panel with columns ``unit`` / ``time`` / ``y`` / ``D`` ready
        for :class:`mlsynth.FMA`.
    Y_treated : np.ndarray
        Treated-unit outcome path, shape ``(T,)``.
    Y_controls : np.ndarray
        Control outcomes, shape ``(N_co, T)``.
    factors : np.ndarray
        Common factor matrix, shape ``(T, 3)``.
    T1, T2 : int
        Pre- / post-treatment period counts.
    sigma_tr, sigma_co : float
        Idiosyncratic error standard deviations actually drawn.
    dgp : str
        Which DGP was drawn (``"dgp1"`` or ``"dgp2"``).
    """

    df: pd.DataFrame
    Y_treated: np.ndarray
    Y_controls: np.ndarray
    factors: np.ndarray
    T1: int
    T2: int
    sigma_tr: float
    sigma_co: float
    dgp: str = "dgp1"


def _factors_dgp1(T: int, rng: np.random.Generator,
                   burn: int = 50) -> np.ndarray:
    """Three stationary factors (DGP1), shape ``(T, 3)``.

    A 50-period burn-in washes out the deterministic zero initial condition
    so the returned slice is the stationary tail.
    """
    L = T + burn
    v = rng.standard_normal((L, 3))
    f = np.zeros((L, 3))
    for t in range(1, L):
        f[t, 0] = 0.8 * f[t - 1, 0] + v[t, 0]
        f[t, 1] = -0.68 * f[t - 1, 1] + v[t, 1] + 0.8 * v[t - 1, 1]
        lag2 = v[t - 2, 2] if t >= 2 else 0.0
        f[t, 2] = v[t, 2] + 0.9 * v[t - 1, 2] + 0.4 * lag2
    return f[burn:]


def _factors_dgp2(T: int, rng: np.random.Generator) -> np.ndarray:
    """Three non-stationary factors (DGP2, Web Appendix E.1)."""
    xi   = rng.uniform(0.0, 1.0, T)
    eps5 = rng.standard_normal(T)            # F1 idiosyncratic shock
    eps4 = rng.standard_normal(T)            # F2 unit-root innovation
    eps6 = rng.standard_normal(T)            # F3 MA(2) innovation
    t_idx = np.arange(1, T + 1, dtype=float)

    f1 = (0.2 + xi) * t_idx + eps5
    f2 = np.cumsum(eps4)                     # drift-less unit root from F2_0 = 0
    f3 = np.empty(T)
    for t in range(T):
        lag1 = eps6[t - 1] if t >= 1 else 0.0
        lag2 = eps6[t - 2] if t >= 2 else 0.0
        f3[t] = np.sqrt(t + 1) + eps6[t] + 0.9 * lag1 + 0.4 * lag2
    return np.stack([f1, f2, f3], axis=1)


_FACTOR_DRAWS = {
    "dgp1": _factors_dgp1,
    "dgp2": _factors_dgp2,
}


def simulate_fma_sample(
    dgp: DGP = "dgp1",
    N_co: int = 30,
    T1: int = 30,
    T2: int = 20,
    variance_case: VarianceCase = "equal",
    alpha: float = 0.0,
    rng: np.random.Generator | None = None,
) -> FMASample:
    r"""Draw one sample from a Li & Sonnier (2023) DGP.

    Parameters
    ----------
    dgp : {"dgp1", "dgp2"}, default ``"dgp1"``
        Stationary (DGP1, Section 4.1) or non-stationary (DGP2,
        Web Appendix E.1) factor process.
    N_co : int, default 30
        Number of control units (the paper's coverage exercise uses 30).
    T1, T2 : int, default 30, 20
        Pre- and post-treatment period counts.
    variance_case : {"equal", "treated_smaller", "treated_larger"}
        Which of the paper's three variance regimes to simulate.
    alpha : float, default 0.0
        Intercept shared by every unit (the paper's :math:`\alpha`; coverage
        is invariant to its value).
    rng : np.random.Generator, optional
        NumPy RNG. Defaults to ``np.random.default_rng()``.

    Returns
    -------
    FMASample
    """
    if variance_case not in _VARIANCE_CASE_PARAMS:
        raise ValueError(
            f"variance_case must be one of "
            f"{sorted(_VARIANCE_CASE_PARAMS)}; got {variance_case!r}."
        )
    if dgp not in _FACTOR_DRAWS:
        raise ValueError(
            f"dgp must be one of {sorted(_FACTOR_DRAWS)}; got {dgp!r}."
        )
    rng = rng or np.random.default_rng()
    sigma_tr, sigma_co = _VARIANCE_CASE_PARAMS[variance_case]

    T = T1 + T2
    F = _FACTOR_DRAWS[dgp](T, rng)                        # (T, 3)
    N = N_co + 1
    lam = rng.normal(loc=1.0, scale=1.0, size=(N, 3))     # λ_il ~ N(1, 1)

    e = np.empty((N, T))
    e[0]  = rng.normal(0.0, sigma_tr, T)                  # treated
    e[1:] = rng.normal(0.0, sigma_co, (N_co, T))          # controls
    Y = alpha + F @ lam.T + e.T                           # (T, N), ATT = 0

    rows = [{"unit": "treated", "time": t, "y": float(Y[t, 0]),
             "D": int(t >= T1)} for t in range(T)]
    for i in range(1, N):
        rows.extend({"unit": f"c{i:03d}", "time": t, "y": float(Y[t, i]),
                     "D": 0} for t in range(T))
    df = pd.DataFrame(rows)

    return FMASample(df=df, Y_treated=Y[:, 0], Y_controls=Y[:, 1:].T,
                      factors=F, T1=T1, T2=T2,
                      sigma_tr=sigma_tr, sigma_co=sigma_co, dgp=dgp)
