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
