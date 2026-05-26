"""Data-generating process and Monte Carlo harness for the SHC method.

Faithful re-implementation of the simulation design in Chen, Yang & Yang,
*Synthetic Historical Control for Policy Evaluation* (2024), Section 3.1,
used here to validate :class:`mlsynth.SHC`.

The DGP (their Eqs. 35-37) is a single time series

.. math::  y_t = \\ell_t + \\delta_t d_t + \\varepsilon_t,

with no intervention effect (:math:`\\delta_t = 0`) so that the exercise
measures how well SHC recovers the latent time-varying confounder
:math:`\\ell_t` in the post-intervention period. The latent component is a
globally :math:`C^1` curve built from alternating pieces: on each
macro-segment of width :math:`4m` the first half is a cosine "local trend"
:math:`f_i` and the second half is a cubic Hermite connector :math:`g_i`
chosen to match :math:`f_i` and :math:`f_{i+1}` in level and slope at the
knots (the "spline restriction", Eq. 36).

There are :math:`h` historical cosine shapes; the treated block's shape is
their convex combination :math:`f_{h+1} = \\sum_{i=1}^h w_{f,i} f_i`, which
encodes Assumption 2(b) (the treated pre-segment is reproducible from its
historical counterparts). This construction reproduces the paper's exact
dimensions: with :math:`h = 4` it gives :math:`T_o = m(4h+1)` (425 for
:math:`m = 25`, 850 for :math:`m = 50`) and :math:`N = T_o - n - (m-1)`
historical blocks (376 and 776, respectively).

* ``Regular-l``:   :math:`(\\alpha_i, P_i) = (0, P)` for every shape, so the
  local trends recur identically.
* ``Irregular-l``: :math:`(\\alpha_i, P_i) = (0, P) + (U_\\alpha, U_P)` with
  :math:`U_\\alpha \\sim U(-1, 1)`, :math:`U_P \\sim U(0, 50)`, so the shapes
  differ in amplitude and periodicity.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd


def _cosine_shape(u: np.ndarray, alpha: float, P: float) -> np.ndarray:
    """Local cosine trend f(u) = alpha + cos(u / P)."""
    return alpha + np.cos(u / P)


def _cosine_deriv(u: np.ndarray, P: float) -> np.ndarray:
    """Derivative of the local cosine trend."""
    return -np.sin(u / P) / P


def _hermite_cubic(s: np.ndarray, width: float, v0: float, d0: float,
                   v1: float, d1: float) -> np.ndarray:
    """Cubic on ``[0, width]`` matching value/slope (v0, d0) at 0 and (v1, d1) at width.

    Returns ``a + b s + c s^2 + d s^3`` with ``a = v0``, ``b = d0`` and
    ``c, d`` solved from the two endpoint conditions at ``s = width``.
    """
    a, b = v0, d0
    w, w2, w3 = width, width ** 2, width ** 3
    # [w2 w3; 2w 3w2] [c; d] = [v1 - a - b w; d1 - b]
    A = np.array([[w2, w3], [2.0 * w, 3.0 * w2]])
    rhs = np.array([v1 - a - b * w, d1 - b])
    c, d = np.linalg.solve(A, rhs)
    return a + b * s + c * s ** 2 + d * s ** 3


def simulate_shc_latent(
    *,
    m: int = 25,
    h: int = 4,
    n: int = 25,
    P: float = 10.0,
    w_f: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    regular: bool = True,
    seed: int = 0,
) -> Tuple[np.ndarray, int, int]:
    """Construct the latent component ``ell_t`` and return ``(ell, T_o, N)``.

    The series spans ``t = 1, ..., T_o + n`` with ``T_o = m * (4h + 1)`` and
    ``N = T_o - n - (m - 1)`` historical blocks.
    """
    if len(w_f) != h:
        raise ValueError(f"w_f must have length h={h}; got {len(w_f)}.")
    rng = np.random.default_rng(seed)
    w_f = np.asarray(w_f, dtype=float)

    if regular:
        alpha = np.zeros(h)
        Pis = np.full(h, float(P))
    else:
        alpha = rng.uniform(-1.0, 1.0, h)
        Pis = float(P) + rng.uniform(0.0, 50.0, h)

    # Shape h+1 (the treated block) is the convex combination of the h
    # historical cosine shapes, evaluated on a common local coordinate.
    def shape_value(i: int, u: np.ndarray) -> np.ndarray:
        if i < h:  # historical shapes 0..h-1
            return _cosine_shape(u, alpha[i], Pis[i])
        return sum(w_f[j] * _cosine_shape(u, alpha[j], Pis[j]) for j in range(h))

    def shape_deriv(i: int, u: np.ndarray) -> np.ndarray:
        if i < h:
            return _cosine_deriv(u, Pis[i])
        return sum(w_f[j] * _cosine_deriv(u, Pis[j]) for j in range(h))

    cycle = 4 * m
    half = 2 * m
    n_macro = h + 1
    T_o = m * (4 * h + 1)
    T = T_o + n

    ell = np.empty(T, dtype=float)
    t = np.arange(1, T + 1)
    for i in range(n_macro):
        start = 1 + cycle * i
        # cosine half: [start, start + 2m)
        cos_mask = (t >= start) & (t < start + half)
        u_cos = t[cos_mask] - start
        ell[cos_mask] = shape_value(i, u_cos)
        # cubic half: [start + 2m, start + 4m); connects shape i to shape i+1
        cub_mask = (t >= start + half) & (t < start + cycle)
        if cub_mask.any() and i + 1 <= h:
            s = (t[cub_mask] - (start + half)).astype(float)
            v0 = float(shape_value(i, np.array([float(half)]))[0])
            d0 = float(shape_deriv(i, np.array([float(half)]))[0])
            v1 = float(shape_value(i + 1, np.array([0.0]))[0])
            d1 = float(shape_deriv(i + 1, np.array([0.0]))[0])
            ell[cub_mask] = _hermite_cubic(s, float(half), v0, d0, v1, d1)

    N = T_o - n - (m - 1)
    return ell, T_o, N


def simulate_shc_panel(
    *,
    m: int = 25,
    h: int = 4,
    n: int = 25,
    P: float = 10.0,
    sigma: float = 0.1,
    w_f: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    regular: bool = True,
    seed: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate one SHC simulation panel as a long DataFrame.

    Returns
    -------
    df : pandas.DataFrame
        Long panel for a single treated unit with columns ``unit``,
        ``time`` (1..T_o+n), ``y`` (= ell_t + noise), and ``treated``
        (0 for ``t <= T_o``, 1 afterwards).
    info : dict
        ``latent`` (the true ell_t, shape (T_o+n,)), ``latent_post``
        (ell over the post window), ``T_o``, ``N``, ``m``, ``n``,
        ``time`` (the integer time index).
    """
    ell, T_o, N = simulate_shc_latent(
        m=m, h=h, n=n, P=P, w_f=w_f, regular=regular, seed=seed,
    )
    rng = np.random.default_rng(seed + 10_000)
    T = T_o + n
    y = ell + rng.normal(0.0, sigma, size=T)
    time = np.arange(1, T + 1)
    treated = (time > T_o).astype(int)

    df = pd.DataFrame({
        "unit": np.ones(T, dtype=int),
        "time": time,
        "y": y,
        "treated": treated,
    })
    info = {
        "latent": ell,
        "latent_post": ell[T_o:T_o + n],
        "latent_pre_block": ell[T_o - m:T_o],
        "T_o": T_o,
        "N": N,
        "m": m,
        "n": n,
        "time": time,
    }
    return df, info
