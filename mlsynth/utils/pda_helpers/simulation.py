r"""Shi & Huang (2023) Table-1 Monte Carlo DGP for the forward-selected PDA.

Reusable simulator behind the ``pda_table1`` Path-B benchmark (forward selection
vs LASSO). Re-implements the four-factor data-generating process of Shi & Huang
(2023), *"Forward-selected panel data approach for program evaluation,"* Journal
of Econometrics 234(2), 512-535, Table 1.

The DGP
-------
One treated unit and ``N`` controls are observed over ``T = T1 + T2`` periods
(the paper sets ``T1 = T2``). Each outcome is a noisy combination of four common
factors plus idiosyncratic noise ``N(0, 0.5)``,

.. math::

   y_{it} = \lambda_i' f_t + \varepsilon_{it}.

The paper studies two factor structures. Under the **i.i.d.** structure
(``dynamic_factors=False``, the benchmark default) all four factors are
independent ``N(0, 1)``. Under the **dynamic** structure the factors carry
distinct serial dependence: ``f1`` i.i.d. ``N(0,1)``; ``f2`` AR(1) 0.9; ``f3``
MA(2) (0.8, 0.4); ``f4`` ARMA(1,1) (0.5, 0.5). The benchmark uses the i.i.d.
structure because that is the regime in which Shi & Huang's Table 1 reports
forward selection *and* their (modified-BIC) Lasso as **both correctly sized** --
so the size comparison isolates the estimator, not factor dynamics.

The factor loadings :math:`\lambda_i \in \mathbb{R}^4` separate the donor pool:
the treated unit and the **first four** controls draw their loadings from
``U(1, 2)`` (the *relevant* donors), while the remaining ``N - 4`` controls draw
from ``U(-0.1, 0.1)`` (near-zero loadings -- the *irrelevant* donors). Forward
selection should recover roughly the handful of relevant donors; LASSO
over-selects into the irrelevant pool.

The post-period treatment effect follows one of the paper's seven shock
processes :math:`\Delta_t` (``shock=`` one of ``"D1"``..``"D7"``), added to the
treated unit over the ``T2`` post periods:

* ``D1`` -- :math:`\Delta_t = 0` (null; the clean size cell);
* ``D2`` -- :math:`\Delta_t \sim N(0, 1)` (null in mean);
* ``D3`` -- :math:`\Delta_t = 0.3\,\Delta_{t-1} + w_t,\; w_t \sim N(0,1)` (null);
* ``D4`` -- :math:`\Delta_t \sim N(0.5, 1)` (power);
* ``D5`` -- :math:`\Delta_t \sim N(1, 1)` (power);
* ``D6`` -- :math:`\Delta_t = 0.35 + 0.3\,\Delta_{t-1} + w_t` (power);
* ``D7`` -- :math:`\Delta_t = 0.7 + 0.3\,\Delta_{t-1} + w_t` (power).

The null is true under ``D1``-``D3`` and false under ``D4``-``D7``.

Determinism
-----------
Loadings, factor innovations, and idiosyncratic noise are drawn from the *same*
generator, so a fixed ``seed`` reproduces a draw bit-for-bit. Loadings are
redrawn with the shocks, so each replication is an independent draw from the DGP
-- the correct design for a size/power study (size and power are properties of
the sampling distribution over draws).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

__all__ = ["PDASimSample", "simulate_pda_panel"]

_N_RELEVANT = 4                  # relevant controls (plus the treated unit)
_N_FACTORS = 4
_NOISE_SD = float(np.sqrt(0.5))  # idiosyncratic N(0, 0.5)
_AR1 = 0.9                        # f2: AR(1)
_MA2 = (0.8, 0.4)                # f3: MA(2)
_ARMA = (0.5, 0.5)               # f4: ARMA(1,1) (ar, ma)
_NULL_SHOCKS = frozenset({"D1", "D2", "D3"})


@dataclass(frozen=True)
class PDASimSample:
    """One draw from the Shi & Huang (2023) factor DGP.

    Attributes
    ----------
    df : pd.DataFrame
        Long panel (columns ``unit`` / ``time`` / ``y`` / ``treat``) ready for
        :class:`mlsynth.PDA`. Treated unit ``"treated"``; relevant donors are
        ``c000``-``c003``.
    Y_treated, Y_controls : np.ndarray
        Treated path ``(T,)`` (incl. effect) and control matrix ``(N, T)``.
    relevant_donors : list of str
        Labels of the 4 relevant controls.
    T1, T2 : int
        Pre- / post-treatment period counts.
    shock : str
        The ``D1``..``D7`` shock label (or ``"custom"`` for a numeric effect).
    is_null : bool
        Whether the data-generating ATE is zero (size cell).
    """

    df: pd.DataFrame
    Y_treated: np.ndarray
    Y_controls: np.ndarray
    relevant_donors: List[str]
    T1: int
    T2: int
    shock: str
    is_null: bool


def _factors(T: int, rng: np.random.Generator, dynamic: bool,
             burn: int = 100) -> np.ndarray:
    """Four common factor paths, shape ``(T, 4)``.

    ``dynamic=False`` returns four i.i.d. ``N(0,1)`` factors; ``dynamic=True``
    returns the paper's (i.i.d., AR(1), MA(2), ARMA(1,1)) structure. A burn-in
    washes out the zero initial condition for the serially-dependent factors.
    """
    if not dynamic:
        return rng.standard_normal((T, _N_FACTORS))

    L = T + burn
    v = rng.standard_normal((L, _N_FACTORS))   # innovations
    f = np.zeros((L, _N_FACTORS))
    ar_arma, ma_arma = _ARMA
    m1, m2 = _MA2
    for t in range(L):
        f[t, 0] = v[t, 0]                                          # f1 i.i.d.
        f[t, 1] = (_AR1 * f[t - 1, 1] if t >= 1 else 0.0) + v[t, 1]  # f2 AR(1)
        lag1 = v[t - 1, 2] if t >= 1 else 0.0
        lag2 = v[t - 2, 2] if t >= 2 else 0.0
        f[t, 2] = v[t, 2] + m1 * lag1 + m2 * lag2                 # f3 MA(2)
        prev = f[t - 1, 3] if t >= 1 else 0.0
        vlag = v[t - 1, 3] if t >= 1 else 0.0
        f[t, 3] = ar_arma * prev + v[t, 3] + ma_arma * vlag       # f4 ARMA(1,1)
    return f[burn:]


def _shock(name: str, T2: int, rng: np.random.Generator) -> np.ndarray:
    """The post-period treatment shock :math:`\\Delta_t`, length ``T2``."""
    w = rng.standard_normal(T2)
    if name == "D1":
        return np.zeros(T2)
    if name == "D2":
        return w
    if name == "D4":
        return 0.5 + w
    if name == "D5":
        return 1.0 + w
    if name in ("D3", "D6", "D7"):
        const = {"D3": 0.0, "D6": 0.35, "D7": 0.7}[name]
        d = np.empty(T2)
        prev = 0.0
        for t in range(T2):
            prev = const + 0.3 * prev + w[t]
            d[t] = prev
        return d
    raise ValueError(f"unknown shock {name!r}; expected D1..D7.")


def simulate_pda_panel(
    N: int = 100,
    T1: int = 100,
    T2: Optional[int] = None,
    shock: str = "D1",
    dynamic_factors: bool = False,
    effect: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> PDASimSample:
    """Draw one sample from the Shi & Huang (2023) Table-1 factor DGP.

    Parameters
    ----------
    N : int, default 100
        Number of control units (the paper uses ``N = 100``).
    T1 : int, default 100
        Pre-treatment period count.
    T2 : int, optional
        Post-treatment period count. Defaults to ``T1`` (paper sets ``T1=T2``).
    shock : str, default "D1"
        Treatment-shock process, one of ``"D1"``..``"D7"`` (see module docstring).
        Ignored when ``effect`` is given.
    dynamic_factors : bool, default False
        ``False`` -> i.i.d. factors (benchmark default); ``True`` -> the paper's
        AR/MA/ARMA factor structure.
    effect : float, optional
        Constant post-period shift; overrides ``shock`` when set (a null cell at
        ``0.0``, a power cell when positive).
    rng : np.random.Generator, optional
        RNG; takes precedence over ``seed``.
    seed : int, optional
        Convenience seed used to build a default generator when ``rng`` is None.
    """
    if N <= _N_RELEVANT:
        raise ValueError(f"N must exceed {_N_RELEVANT} relevant controls; got {N}.")
    if T1 <= 0:
        raise ValueError(f"T1 must be positive; got {T1}.")
    if rng is None:
        rng = np.random.default_rng(seed)
    T2 = T1 if T2 is None else T2
    if T2 <= 0:
        raise ValueError(f"T2 must be positive; got {T2}.")
    T = T1 + T2

    f = _factors(T, rng, dynamic=dynamic_factors)            # (T, 4)

    lam_treated = rng.uniform(1.0, 2.0, size=_N_FACTORS)
    lam_ctrl = np.empty((N, _N_FACTORS))
    lam_ctrl[:_N_RELEVANT] = rng.uniform(1.0, 2.0, size=(_N_RELEVANT, _N_FACTORS))
    lam_ctrl[_N_RELEVANT:] = rng.uniform(
        -0.1, 0.1, size=(N - _N_RELEVANT, _N_FACTORS))

    y_tr = f @ lam_treated + _NOISE_SD * rng.standard_normal(T)
    if effect is not None:
        label, is_null = "custom", (effect == 0.0)
        y_tr[T1:] += float(effect)
    else:
        label, is_null = shock, shock in _NULL_SHOCKS
        y_tr[T1:] += _shock(shock, T2, rng)

    y_ctrl = lam_ctrl @ f.T + _NOISE_SD * rng.standard_normal((N, T))

    rows = [{"unit": "treated", "time": t, "y": float(y_tr[t]),
             "treat": int(t >= T1)} for t in range(T)]
    for i in range(N):
        rows.extend({"unit": f"c{i:03d}", "time": t,
                     "y": float(y_ctrl[i, t]), "treat": 0} for t in range(T))
    df = pd.DataFrame(rows)
    relevant = [f"c{i:03d}" for i in range(_N_RELEVANT)]
    return PDASimSample(df=df, Y_treated=y_tr, Y_controls=y_ctrl,
                        relevant_donors=relevant, T1=T1, T2=T2,
                        shock=label, is_null=is_null)


# --------------------------------------------------------------------------- #
# Li & Bell (2017) DGP3 -- the LASSO-PDA out-of-sample-prediction simulation
# (Table 2: a dense three-factor model with N > T1).
# --------------------------------------------------------------------------- #
_LIBELL_BURN = 100


def _libell_factors(T: int, rng: np.random.Generator) -> np.ndarray:
    """Li & Bell (2017) Eq. 5.1 three factors, shape ``(T, 3)``.

    ``f1`` AR(1) 0.8; ``f2`` ARMA(1,1) (-0.68, 0.8); ``f3`` MA(2) (0.9, 0.4);
    innovations i.i.d. ``N(0, 1)``. A burn-in removes the zero initial state.
    """
    L = T + _LIBELL_BURN
    v = rng.standard_normal((L, 3))
    f = np.zeros((L, 3))
    for t in range(L):
        f[t, 0] = (0.8 * f[t - 1, 0] if t >= 1 else 0.0) + v[t, 0]
        f[t, 1] = ((-0.68 * f[t - 1, 1] if t >= 1 else 0.0)
                   + v[t, 1] + (0.8 * v[t - 1, 1] if t >= 1 else 0.0))
        f[t, 2] = (v[t, 2] + (0.9 * v[t - 1, 2] if t >= 1 else 0.0)
                   + (0.4 * v[t - 2, 2] if t >= 2 else 0.0))
    return f[_LIBELL_BURN:]


def simulate_libell_panel(
    N: int = 31, T1: int = 25, T2: int = 10, sigma2: float = 1.0,
    rng: Optional[np.random.Generator] = None, seed: Optional[int] = None,
) -> pd.DataFrame:
    """Draw one untreated panel from Li & Bell (2017) DGP3 (Eq. 5.2).

    ``y_it^0 = a_i + b_i' f_t + u_it`` with ``a_i = 1``, loadings
    ``b_ji ~ N(1, 1)`` (a *dense* factor model), idiosyncratic
    ``u_it ~ N(0, sigma2)``, and the Eq. 5.1 factors. Unit ``0`` is the
    "treated" unit (no effect is injected -- the case measures out-of-sample
    prediction of its untreated path), treated over the ``T2`` post-periods.
    Returns a long ``unit``/``time``/``y``/``treat`` frame for :class:`mlsynth.PDA`.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    T = T1 + T2
    f = _libell_factors(T, rng)                       # (T, 3)
    a = np.ones(N)
    B = rng.normal(1.0, 1.0, size=(N, 3))             # b_ji ~ N(1, 1)
    u = rng.normal(0.0, np.sqrt(sigma2), size=(N, T))
    Y0 = a[:, None] + B @ f.T + u                      # (N, T)

    rows = [{"unit": "treated" if i == 0 else f"c{i:03d}", "time": t,
             "y": float(Y0[i, t]), "treat": int(i == 0 and t >= T1)}
            for i in range(N) for t in range(T)]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Shi & Wang (2024) Section 5.1-5.2 -- the L2-relaxation PDA size/power
# simulation (Table 2: a dense strong-factor model with N = 100).
# --------------------------------------------------------------------------- #
# Innovation SDs that normalise each factor to unit unconditional variance.
_SW_FACTOR_SD = (1.0, 0.19 ** 0.5, (5.0 / 9.0) ** 0.5, (3.0 / 7.0) ** 0.5)
_SW_NOISE_SD = 0.5 ** 0.5          # idiosyncratic N(0, 0.5)
_SW_NULL_SHOCKS = frozenset({"D1", "D2", "D3"})


def _shiwang_factors(T: int, rng: np.random.Generator, burn: int = 200) -> np.ndarray:
    """Shi & Wang (2024) Sec. 5.1 four factors, shape ``(T, 4)``.

    ``f1`` i.i.d.; ``f2`` AR(1) 0.9; ``f3`` MA(2) (0.8, 0.4); ``f4`` ARMA(1,1)
    (0.5, 0.5); innovation variances set so each factor has unit unconditional
    variance. A burn-in removes the zero initial state.
    """
    L = T + burn
    v = rng.standard_normal((L, 4)) * np.asarray(_SW_FACTOR_SD)
    f = np.zeros((L, 4))
    for t in range(L):
        f[t, 0] = v[t, 0]
        f[t, 1] = (0.9 * f[t - 1, 1] if t >= 1 else 0.0) + v[t, 1]
        f[t, 2] = (v[t, 2] + (0.8 * v[t - 1, 2] if t >= 1 else 0.0)
                   + (0.4 * v[t - 2, 2] if t >= 2 else 0.0))
        f[t, 3] = ((0.5 * f[t - 1, 3] if t >= 1 else 0.0) + v[t, 3]
                   + (0.5 * v[t - 1, 3] if t >= 1 else 0.0))
    return f[burn:]


def _shiwang_loadings(n: int, rng: np.random.Generator) -> np.ndarray:
    """Strong loadings ~ Uniform([-0.5,-0.3] U [0.3,0.5]), shape ``(n, 4)``."""
    return rng.choice([-1.0, 1.0], size=(n, 4)) * rng.uniform(0.3, 0.5, size=(n, 4))


def simulate_shiwang_panel(
    N: int = 100, T1: int = 50, T2: Optional[int] = None, shock: str = "D1",
    rng: Optional[np.random.Generator] = None, seed: Optional[int] = None,
):
    """Draw one sample from Shi & Wang (2024) Table-2 (single-treated, strong factors).

    ``y_it = lambda_i' f_t + u_it`` with strong loadings, ``u_it ~ N(0, 0.5)``,
    and unit-variance factors; the treated unit (index 0) gets the post-period
    shock ``Delta_t`` (``D1`` = 0 for size; ``D4`` adds a constant 0.3 for power).
    Returns ``(y_treated, Y_controls, T1)`` -- arrays, since the L2 size study
    drives the estimator at the array level. The null holds for ``D1``-``D3``.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    T2 = T1 if T2 is None else T2
    T = T1 + T2
    f = _shiwang_factors(T, rng)                       # (T, 4)
    y = f @ _shiwang_loadings(1, rng)[0] + _SW_NOISE_SD * rng.standard_normal(T)
    Yc = _shiwang_loadings(N, rng) @ f.T + _SW_NOISE_SD * rng.standard_normal((N, T))
    if shock in ("D4", "D5", "D6", "D7", "D8", "D9"):
        const = {"D4": 0.3, "D5": 0.3, "D6": 0.3, "D7": 0.5, "D8": 0.5, "D9": 0.5}[shock]
        y[T1:] += const
    elif shock not in ("D1", "D2", "D3"):
        raise ValueError(f"unknown shock {shock!r}; expected D1..D9.")
    return y, Yc, T1
