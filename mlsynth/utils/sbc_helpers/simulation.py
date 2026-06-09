"""Nonstationary DGPs for the SBC simulation study (Shi, Xi & Xie 2025, Sec. 4).

Re-implements the paper's three Table-1 data-generating processes so the
simulation study can be benchmarked without the authors' code. Each ``draw``
returns an *untreated* ``(NU, T)`` panel (row 0 = the unit that will be treated):

* Model 1 -- independent random walks with drift (the "spurious regression"
  regime: no shared structure);
* Model 2 -- idiosyncratic unit-root trends driven by two common *stationary*
  AR(phi) factors in the increments (shared short-run structure, no cointegration);
* Model 3 -- partial cointegration: half the units (incl. the treated) share two
  random-walk factors in levels, the rest follow Model 2.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

NU_DEFAULT = 12


def _ar(phi: float, T: int, rng: np.random.Generator) -> np.ndarray:
    f = np.zeros(T)
    for t in range(1, T):
        f[t] = phi * f[t - 1] + rng.normal()
    return f


def simulate_shi_xi_xie(
    model: int, T0: int, *, H: int = 2, NU: int = NU_DEFAULT,
    phi: float = 0.5, rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Draw one untreated ``(NU, T)`` panel from Shi-Xi-Xie Model ``{1,2,3}``.

    ``T = T0 + H``; row 0 is the (eventually) treated unit.
    """
    rng = rng or np.random.default_rng()
    T = T0 + H
    if model == 1:                                   # independent random walks
        return np.cumsum(rng.normal(0, 0.5, NU)[:, None]
                         + rng.normal(0, 1, (NU, T)), axis=1)
    if model == 2:                                   # unit-root + common AR(1) factors
        f1, f2 = _ar(phi, T, rng), _ar(phi, T, rng)
        lam = rng.normal(0, 1, (NU, 2))
        incr = lam[:, [0]] * f1 + lam[:, [1]] * f2 + rng.normal(0, 1, (NU, T))
        return np.cumsum(incr, axis=1)
    if model == 3:                                   # partial cointegration
        half = NU // 2
        frw1, frw2 = np.cumsum(rng.normal(0, 1, T)), np.cumsum(rng.normal(0, 1, T))
        far1, far2 = _ar(phi, T, rng), _ar(phi, T, rng)
        e = rng.normal(0, 1, (NU, T))
        Y = np.empty((NU, T))
        lrw = rng.normal(0, T0 ** (-1 / 3), (half, 2))
        lar = rng.normal(0, 1, (half, 2))
        Y[:half] = (lrw[:, [0]] * frw1 + lrw[:, [1]] * frw2
                    + lar[:, [0]] * far1 + lar[:, [1]] * far2 + e[:half])
        l2 = rng.normal(0, 1, (NU - half, 2))
        Y[half:] = np.cumsum(l2[:, [0]] * far1 + l2[:, [1]] * far2 + e[half:], axis=1)
        return Y
    raise ValueError(f"model must be 1, 2 or 3; got {model}")
