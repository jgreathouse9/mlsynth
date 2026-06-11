r"""Sakaguchi & Tagawa (2026) spatial-autoregressive spillover DGP.

Reusable simulator behind the ``spillsynth_sar_mc`` Path-B benchmark. It is the
data-generating process of the SAR spillover paper's simulation study (and of the
package's own SAR tests): the untreated control outcomes follow a spatial
autoregression driven by a row-normalised control-to-control weight matrix
:math:`\mathbf{W}` and a treated-to-control weight vector :math:`\mathbf{w}`,

.. math::

   \mathbf{Y}^c_t = \rho\,(\mathbf{w}\,Y_{0t} + \mathbf{W}\,\mathbf{Y}^c_t)
       + \mathbf{u}_t,

with the treated unit's untreated outcome :math:`Y_{0t} = \boldsymbol{\alpha}^\top
\mathbf{Y}^c_t`. A post-period treatment effect :math:`\tau_t \sim N(1, 1)` is
added to the treated unit and propagates to the controls through the SAR term --
so a naive synthetic control is biased by the spillover while the SAR estimator
recovers both the treatment effect and the spatial coefficient :math:`\rho`. At
:math:`\rho = 0` the panel collapses to a standard (no-spillover) synthetic
control, the regime in which the SAR estimator must nest ordinary SCM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .sampler import row_normalize

__all__ = ["SARSimSample", "rook_adjacency", "simulate_sar_panel"]


@dataclass(frozen=True)
class SARSimSample:
    """One draw from the SAR spillover DGP.

    Attributes
    ----------
    df : pandas.DataFrame
        Long panel (columns ``unit`` / ``time`` / ``y`` / ``d``) ready for
        :class:`mlsynth.SPILLSYNTH`. The treated unit is ``"T"``; controls are
        ``c0``..``c{N-1}``.
    spatial_W : pandas.DataFrame
        Row-normalised control-to-control spatial-weight matrix ``(N, N)``.
    spatial_w : pandas.Series
        Row-normalised treated-to-control spatial-weight vector ``(N,)``.
    true_att : float
        Realised average post-treatment effect on the treated unit.
    rho : float
        The spatial-autoregressive coefficient used to generate the panel.
    n_controls : int
        Number of control units ``N``.
    """

    df: pd.DataFrame
    spatial_W: pd.DataFrame
    spatial_w: pd.Series
    true_att: float
    rho: float
    n_controls: int


def rook_adjacency(n_rows: int, n_cols: int) -> np.ndarray:
    """Binary rook (4-neighbour) adjacency matrix for an ``n_rows x n_cols`` grid."""
    N = n_rows * n_cols
    W = np.zeros((N, N))
    idx = lambda r, c: r * n_cols + c
    for r in range(n_rows):
        for c in range(n_cols):
            i = idx(r, c)
            if r > 0:
                W[i, idx(r - 1, c)] = 1
            if r < n_rows - 1:
                W[i, idx(r + 1, c)] = 1
            if c > 0:
                W[i, idx(r, c - 1)] = 1
            if c < n_cols - 1:
                W[i, idx(r, c + 1)] = 1
    return W


def simulate_sar_panel(
    rho: float = 0.6,
    n_rows: int = 4,
    n_cols: int = 4,
    T: int = 30,
    T0: int = 20,
    sigma2: float = 0.1,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> SARSimSample:
    """Draw one spatial-spillover panel from the SAR DGP.

    Parameters
    ----------
    rho : float, default 0.6
        Spatial-autoregressive coefficient (``0`` gives a no-spillover panel).
    n_rows, n_cols : int, default 4
        Grid dimensions; the control pool is ``n_rows * n_cols`` units on a rook
        lattice.
    T, T0 : int, default 30, 20
        Total and pre-treatment period counts.
    sigma2 : float, default 0.1
        Innovation variance.
    rng : numpy.random.Generator, optional
        Generator; takes precedence over ``seed``.
    seed : int, optional
        Convenience seed used to build a generator when ``rng`` is None.

    Returns
    -------
    SARSimSample
        The long panel, the spatial-weight matrix / vector, and the realised ATT.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if not 1 <= T0 < T:
        raise ValueError("T0 must satisfy 1 <= T0 < T.")
    if n_rows < 1 or n_cols < 1:
        raise ValueError("grid dimensions must be positive.")
    N = n_rows * n_cols
    Wn = row_normalize(rook_adjacency(n_rows, n_cols))
    w = np.zeros(N)
    w[: min(4, N)] = 1.0
    wn = w / w.sum()
    alpha = np.zeros(N)
    alpha[0] = 0.5
    if N > 1:
        alpha[1] = -0.2
    alpha[2:4] = 0.4
    alpha[4:min(10, N)] = 0.1 / 6
    IN = np.eye(N)
    Ainv = np.linalg.inv(IN - rho * Wn - rho * np.outer(wn, alpha))
    Apost = np.linalg.inv(IN - rho * Wn)
    err = rng.normal(0, np.sqrt(sigma2), (T, N))
    Yc0 = (Ainv @ err.T).T
    Y00 = Yc0 @ alpha
    tau = rng.normal(1.0, 1.0, T - T0)
    Y0 = Y00.copy()
    Y0[T0:] += tau
    Yc = Yc0.copy()
    for t in range(T0, T):
        Yc[t] = Apost @ (rho * wn * Y0[t] + err[t])

    labels = ["T"] + [f"c{i}" for i in range(N)]
    Ypanel = np.vstack([Y0[None, :], Yc.T])
    rows = []
    for ui, lab in enumerate(labels):
        for t in range(T):
            rows.append({"unit": lab, "time": t, "y": Ypanel[ui, t],
                         "d": int(lab == "T" and t >= T0)})
    df = pd.DataFrame(rows)
    Wdf = pd.DataFrame(Wn, index=labels[1:], columns=labels[1:])
    wser = pd.Series(wn, index=labels[1:])
    return SARSimSample(df=df, spatial_W=Wdf, spatial_w=wser,
                        true_att=float(tau.mean()), rho=float(rho), n_controls=N)
