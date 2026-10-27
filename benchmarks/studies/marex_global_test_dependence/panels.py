"""Factor-model panels with factors that move over time in six ways.

The panel is Abadie and Zhao's Assumption 1 at the dimensions of their Section 5
(``J = 15`` units, ``R = 7`` observed and ``F = 11`` unobserved covariates on
Uniform(0, 1), errors N(0, 1)), the ``marex_cumulative_conformal`` study's
``designs.panel``:

    y_jt = delta_t + theta_t' Z_j + lambda_t' mu_j + eps_jt.

The period effects ``delta_t``, ``theta_t`` and ``lambda_t`` follow one regime,
each with the mean and spread of Uniform(0, hi):

- ``iid``: drawn independently each period, Theorem 2's exchangeable case;
- ``ar5``, ``ar8``, ``ar95``: a stationary AR(1) at 0.5, 0.8 and 0.95;
- ``rw``: a random walk scaled to the uniform's spread over the panel;
- ``season``: an annual cycle (period 52, a random phase per coordinate) plus an
  AR(1) at 0.5, in equal parts.

``design`` solves MAREX on a panel (``m_eq = 3``, exact backend, the default
blank window, inference on) and keeps what the global test needs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

J, R_OB, F_UNOB = 15, 7, 11
REGIMES = ("iid", "ar5", "ar8", "ar95", "rw", "season")
_PHI = {"ar5": 0.5, "ar8": 0.8, "ar95": 0.95}
_BURN = 400


def _ar(T, k, phi, rng):
    z = np.empty((T + _BURN, k))
    z[0] = rng.standard_normal(k)
    for t in range(1, T + _BURN):
        z[t] = phi * z[t - 1] + np.sqrt(1 - phi ** 2) * rng.standard_normal(k)
    return z[_BURN:]                                   # unit variance


def factor(T: int, k: int, hi: float, regime: str, rng: np.random.Generator) -> np.ndarray:
    sd = hi / np.sqrt(12.0)
    if regime == "iid":
        return rng.uniform(0.0, hi, (T, k))
    if regime in _PHI:
        return hi / 2.0 + sd * _ar(T, k, _PHI[regime], rng)
    if regime == "rw":
        return hi / 2.0 + sd * np.cumsum(rng.standard_normal((T, k)), axis=0) / np.sqrt(T)
    if regime == "season":
        t = np.arange(T)[:, None]
        cycle = np.sqrt(2.0) * np.sin(2 * np.pi * t / 52 + rng.uniform(0, 2 * np.pi, k))
        return hi / 2.0 + sd * (cycle + _ar(T, k, 0.5, rng)) / np.sqrt(2.0)
    raise ValueError(f"regime must be one of {REGIMES}; got {regime!r}")


def panel(T: int, regime: str, rng: np.random.Generator) -> np.ndarray:
    """Outcomes, ``(T, J)``."""
    Z = rng.uniform(0, 1, (R_OB, J))
    mu = rng.uniform(0, 1, (F_UNOB, J))
    delta = factor(T, 1, 20.0, regime, rng)
    theta = factor(T, R_OB, 10.0, regime, rng)
    lam = factor(T, F_UNOB, 10.0, regime, rng)
    return delta + theta @ Z + lam @ mu + rng.standard_normal((T, J))


@dataclass
class Design:
    w: np.ndarray            # MAREX's global treated weights, by unit
    v: np.ndarray            # its control weights
    T0: int
    blank: int
    marex_p: float           # MAREX's own global p-value on the panel it was fitted to

    @property
    def TcE(self) -> int:
        return self.T0 - self.blank


def design(Y: np.ndarray, T0: int, H: int) -> Design:
    from mlsynth import MAREX
    T = Y.shape[0]
    df = pd.DataFrame({"unit": np.repeat(np.arange(J), T), "time": np.tile(np.arange(T), J),
                       "y": Y.T.reshape(-1)})
    res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time", "T0": T0,
                 "T_post": H, "m_eq": 3, "backend": "exact", "inference": True,
                 "display_graph": False}).fit()
    g = res.globres
    return Design(w=np.asarray(g.treated_weights_agg, float), v=np.asarray(g.control_weights_agg, float),
                  T0=T0, blank=int(res.study.blank_periods), marex_p=float(g.inference.global_p_value))


def contrast(Y: np.ndarray, d: Design, H: int):
    """The design's contrast on the blank and the post periods."""
    u = Y @ (d.w - d.v)
    return u[d.TcE:d.T0], u[d.T0:d.T0 + H]


def shipped_pvalue(Y: np.ndarray, d: Design, H: int) -> float:
    """MAREX's global p-value for panel ``Y`` under design ``d``, through the
    library's ``compute_inference`` with the arguments MAREX passes it."""
    import inspect

    from mlsynth.utils.marex_helpers.inference import compute_inference
    from mlsynth.utils.marex_helpers.orchestration import solve_marex
    defaults = {k: inspect.signature(solve_marex).parameters[k].default
                for k in ("alpha", "max_combinations", "random_state")}
    inf = compute_inference(Y @ d.w, Y @ d.v, d.T0, d.TcE, d.blank, T_post=H,
                            control_outcomes=Y[:, d.w <= 0].T, **defaults)
    return float(inf.global_p_value)
