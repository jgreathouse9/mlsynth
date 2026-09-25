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
from typing import Optional, Tuple

import numpy as np

from ...exceptions import MlsynthConfigError
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


# ---------------------------------------------------------------------------
# Li (2020) JASA: the simulation designs behind Tables 1-4
# ---------------------------------------------------------------------------

#: The eight loading configurations. DGP1-DGP2 and DGP5-DGP8 run on the
#: stationary first factor; DGP3 and DGP4 replace it by a unit root and
#: otherwise repeat DGP1 and DGP2.
LI2020_DGPS: Tuple[str, ...] = (
    "dgp1", "dgp2", "dgp3", "dgp4", "dgp5", "dgp6", "dgp7", "dgp8",
)

#: DGP -> (treated loading, first-half control loading, second-half control
#: loading). The first four are written by the paper for N = 11 as "units 2
#: to 7 carry ones and 8 to 11 carry zeros", which is the same halves rule
#: with a zero second half only at that N, so they are held separately.
_LI2020_HALVES = {
    "dgp5": (1.0, 2.0, -0.5),
    "dgp6": (1.0, -2.0, 0.5),
    "dgp7": (1.0, 0.5, 0.2),
    "dgp8": (0.2, 1.0, 0.5),
}
_LI2020_SPARSE = {"dgp1": 1.0, "dgp2": 2.0, "dgp3": 1.0, "dgp4": 2.0}
_LI2020_UNIT_ROOT = frozenset({"dgp3", "dgp4"})


def li2020_loadings(dgp: str, N: int) -> np.ndarray:
    r"""The ``N x 3`` loading matrix :math:`B` for one of the paper's designs.

    Row 0 is the treated unit. Section 5.1 gives the first four:

    .. math::

       \text{DGP1}: b_1 = \mathbf{1}_3;\ b_j = \mathbf{1}_3\ (j = 2..7);\
                    b_j = \mathbf{0}_3\ (j = 8..11),

    with DGP2 raising the treated unit's loadings to 2 so the treated and
    control units are drawn from heterogeneous distributions, and DGP3 and
    DGP4 repeating those two on a unit-root first factor. Six of the ten
    controls load on the factors and four do not, which is what makes donor
    selection meaningful.

    Section 5.4 gives four more for the large-``N`` study, each splitting
    the controls into equal halves:

    ====== ============ ================= ==================
    DGP    treated      controls 2..(N+1)/2  (N+3)/2..N
    ====== ============ ================= ==================
    DGP5   1            2                 -0.5
    DGP6   1            -2                0.5
    DGP7   1            0.5               0.2
    DGP8   0.2          1                 0.5
    ====== ============ ================= ==================

    DGP5 makes the sum-to-one restriction correct and DGP6 makes it badly
    wrong; DGP7 gives the treated unit a much stronger factor correlation
    than any control; DGP8 reverses that.

    Parameters
    ----------
    dgp : str
        One of :data:`LI2020_DGPS`.
    N : int
        Total units, treated included. The paper uses 11 for Tables 1-2 and
        ``N`` in {11, 21, 31, 51, 81} for Tables 3-4. ``N`` must be odd for
        the halves configurations to split evenly.

    Returns
    -------
    np.ndarray
        Shape ``(N, 3)``.
    """
    if dgp not in LI2020_DGPS:
        raise MlsynthConfigError(
            f"dgp must be one of {list(LI2020_DGPS)}; got {dgp!r}."
        )
    N = int(N)
    if dgp in _LI2020_SPARSE:
        if N < 7:
            raise MlsynthConfigError(
                f"N must be at least 7 for {dgp!r}, which loads units 2 to 7 "
                f"on the factors; got N = {N}."
            )
        B = np.zeros((N, 3), dtype=float)
        B[0] = _LI2020_SPARSE[dgp]
        B[1:7] = 1.0
        return B

    if N < 3 or N % 2 == 0:
        raise MlsynthConfigError(
            f"N must be odd and at least 3 for {dgp!r}, so the controls split "
            f"into two equal halves; got N = {N}."
        )
    treated, first, second = _LI2020_HALVES[dgp]
    B = np.empty((N, 3), dtype=float)
    B[0] = treated
    half = (N + 1) // 2
    B[1:half] = first
    B[half:] = second
    return B


def _li2020_factors(
    T: int, rng: np.random.Generator, *, unit_root: bool, burn: int = 100,
) -> np.ndarray:
    r"""The three common factors of Equation 26.

    .. math::

       f_{1t} &= 0.8 f_{1,t-1} + \epsilon_{1t}, \\
       f_{2t} &= -0.6 f_{2,t-1} + \epsilon_{2t} + 0.8 \epsilon_{2,t-1}, \\
       f_{3t} &= \epsilon_{3t} + 0.9 \epsilon_{3,t-1} + 0.4 \epsilon_{3,t-2},

    with :math:`\epsilon_{jt}` iid standard normal. ``unit_root`` replaces
    the first line by :math:`f_{1t} = f_{1,t-1} + \epsilon_{1t}`, which is
    what separates DGP3 and DGP4 from DGP1 and DGP2.

    The article prints the second line as
    :math:`f_{2t} = -0.6 f_{1,t-1} + \dots`, carrying the first factor's
    lag into the second factor's own recursion. That is a typo: the process
    is Hsiao, Ching and Wan's, where each factor follows its own ARMA, and
    :func:`_factors` above renders the same family with ``f2``'s own lag.
    Reading it literally would also leave :math:`f_2` a function of
    :math:`f_1`, which contradicts the three-factor structure the
    identification argument rests on.

    A burn-in of 100 periods discards the zero initial condition of the two
    stationary factors. It is drawn in the unit-root case too, so that a
    stationary and a unit-root run from the same seed share their innovations
    and therefore their :math:`f_2` and :math:`f_3` exactly -- which makes a
    DGP1-against-DGP3 comparison paired, and leaves the first factor as the
    only thing that moved. A random walk has no stationary distribution to
    burn into, so its retained window is shifted to start at zero, as the
    paper's :math:`f_{1,0} = 0` has it; the level of a unit root is not
    identified separately from the intercept :math:`a` in any case.
    """
    L = T + burn
    eps = rng.standard_normal((L + 2, 3))
    f = np.zeros((L, 3))
    for t in range(1, L):
        f[t, 0] = (f[t - 1, 0] if unit_root else 0.8 * f[t - 1, 0]) + eps[t, 0]
        f[t, 1] = -0.6 * f[t - 1, 1] + eps[t, 1] + 0.8 * eps[t - 1, 1]
        lag2 = eps[t - 2, 2] if t >= 2 else 0.0
        f[t, 2] = eps[t, 2] + 0.9 * eps[t - 1, 2] + 0.4 * lag2
    out = f[-T:].copy()
    if unit_root:
        out[:, 0] -= out[0, 0]
    return out


def _li2020_effect(
    T2: int, alpha0: float, rng: np.random.Generator,
) -> np.ndarray:
    r"""Equation 27's treatment effect path.

    .. math::

       \Delta_{1t} = \alpha_0
           \Big[\frac{e^{z_t}}{1 + e^{z_t}} + 1\Big], \qquad
       z_t = 0.5 z_{t-1} + \eta_t, \quad \eta_t \sim N(0, 0.5^2).

    :math:`z_t` is a mean-zero stationary AR(1), so it is symmetric about
    zero and :math:`E[e^{z}/(1 + e^{z})] = 1/2`. The population effect is
    therefore :math:`1.5\alpha_0`, and :math:`\alpha_0 = 0` gives no effect
    at all -- the case the coverage tables use, since the centred statistic
    does not depend on the effect's size.
    """
    z = np.zeros(T2)
    eta = rng.normal(0.0, 0.5, T2)
    for t in range(1, T2):
        z[t] = 0.5 * z[t - 1] + eta[t]
    return float(alpha0) * (np.exp(z) / (1.0 + np.exp(z)) + 1.0)


@dataclass(frozen=True)
class Li2020Sample:
    """One draw from a Li (2020) design.

    Attributes
    ----------
    df : pd.DataFrame
        Long panel with ``unit`` / ``time`` / ``y`` / ``D``, ready for
        :class:`~mlsynth.TSSC` or :class:`~mlsynth.PDA`.
    Y_treated : np.ndarray
        The treated unit's observed series, shape ``(T,)``: the untreated
        path plus ``effect``.
    Y_treated_untreated : np.ndarray
        The treated unit's counterfactual, shape ``(T,)``.
    Y_controls : np.ndarray
        Control outcomes, shape ``(T, N - 1)``.
    factors : np.ndarray
        The three common factors, shape ``(T, 3)``.
    loadings : np.ndarray
        ``B``, shape ``(N, 3)``, treated unit first.
    errors : np.ndarray
        Idiosyncratic errors, shape ``(T, N)``.
    effect : np.ndarray
        ``Delta_1t`` over all ``T`` periods, zero before ``T1``.
    true_att : float
        ``1.5 * alpha0``, the population average post-period effect that
        an MSE against the truth is computed from.
    dgp, N, T1, T2, alpha0, error : the design, echoed.
    """

    df: "pd.DataFrame"
    Y_treated: np.ndarray
    Y_treated_untreated: np.ndarray
    Y_controls: np.ndarray
    factors: np.ndarray
    loadings: np.ndarray
    errors: np.ndarray
    effect: np.ndarray
    true_att: float
    dgp: str
    N: int
    T1: int
    T2: int
    alpha0: float
    error: str


def simulate_li2020_sample(
    dgp: str = "dgp1",
    N: int = 11,
    T1: int = 90,
    T2: int = 20,
    alpha0: float = 0.0,
    error: str = "uniform",
    rng: Optional[np.random.Generator] = None,
) -> Li2020Sample:
    r"""One draw from Li (2020), Equation 26.

    .. math::

       y^0_t = a + B f_t + u_t, \qquad t = 1, \dots, T,

    with :math:`a = \mathbf{1}_N`, :math:`B` from :func:`li2020_loadings`,
    :math:`f_t` from Equation 26's three ARMA processes, and
    :math:`u_{it}` iid with unit variance. The treated unit is row 0 and
    receives Equation 27's effect over the post-period.

    The defaults are the paper's own: ``T1 = 90``, ``T2 = 20``, ``N = 11``,
    errors uniform on :math:`[-\sqrt{3}, \sqrt{3}]`, and ``alpha0 = 0``.
    Section 5.2 reports that the normal and uniform errors give virtually
    identical results and that ``alpha0 = 0`` and ``alpha0 = 1`` do too,
    because the centred statistic does not depend on the effect's size.

    Parameters
    ----------
    dgp : str
        One of :data:`LI2020_DGPS`.
    N : int
        Total units including the treated one.
    T1, T2 : int
        Pre- and post-treatment lengths.
    alpha0 : float
        Scale of the treatment effect; 0 gives none.
    error : {"uniform", "normal"}
        Distribution of :math:`u_{it}`, both with unit variance.
    rng : numpy.random.Generator, optional

    Returns
    -------
    Li2020Sample
    """
    import pandas as pd

    if error not in ("uniform", "normal"):
        raise MlsynthConfigError(
            f"error must be 'uniform' or 'normal'; got {error!r}."
        )
    B = li2020_loadings(dgp, N)
    rng = rng if rng is not None else np.random.default_rng()
    T1, T2 = int(T1), int(T2)
    T = T1 + T2

    f = _li2020_factors(T, rng, unit_root=dgp in _LI2020_UNIT_ROOT)
    if error == "uniform":
        u = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), (T, int(N)))
    else:
        u = rng.standard_normal((T, int(N)))

    Y0 = 1.0 + f @ B.T + u                      # (T, N), a = 1
    effect = np.zeros(T)
    effect[T1:] = _li2020_effect(T2, alpha0, rng)

    y_untreated = Y0[:, 0].copy()
    y_treated = y_untreated + effect
    controls = Y0[:, 1:]

    units = ["treated"] + [f"control{j}" for j in range(1, int(N))]
    df = pd.DataFrame({
        "unit": np.repeat(units, T),
        "time": np.tile(np.arange(T), int(N)),
        "y": np.concatenate([y_treated, controls.T.ravel()]),
        "D": np.concatenate([
            (np.arange(T) >= T1).astype(int),
            np.zeros((int(N) - 1) * T, dtype=int),
        ]),
    })

    return Li2020Sample(
        df=df, Y_treated=y_treated, Y_treated_untreated=y_untreated,
        Y_controls=controls, factors=f, loadings=B, errors=u, effect=effect,
        true_att=1.5 * float(alpha0), dgp=dgp, N=int(N), T1=T1, T2=T2,
        alpha0=float(alpha0), error=error,
    )
