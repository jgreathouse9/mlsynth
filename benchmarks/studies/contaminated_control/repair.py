"""The contamination, the four arms, and the two decision thresholds.

A MAREX design is locked at ``T0``: treated weights ``w``, control weights
``v``. During the experiment an exogenous event shifts one control market
``k*`` by ``pi`` over the post-period. The arms differ in what they do about
it once ``k*`` is known.

The estimation error of the uncorrected estimator is exactly ``-v[k*] * pi``,
which is algebra and holds in every DGP. It scales with the weight, not with
the size of the shock, so a contaminated market carrying zero weight costs
nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlsynth.utils.solvers.simplex import simplex_lstsq


@dataclass(frozen=True)
class Reconstruction:
    """``k*`` rebuilt from the clean control markets."""

    kstar: int
    vk: float
    clean: np.ndarray
    weights: np.ndarray
    thr_in: float
    thr_oos: float
    n_blank_blocks: int


def block_rms(x: np.ndarray, blk: int) -> float:
    """RMS of the means of consecutive blocks of length ``blk``."""
    n = len(x) // blk
    if n < 1:
        raise ValueError(f"series of length {len(x)} holds no block of {blk}")
    return float(np.sqrt(np.mean(x[:n * blk].reshape(n, blk).mean(axis=1) ** 2)))


def reconstruct(YN: np.ndarray, v: np.ndarray, T0: int, *,
                max_blank_blocks: int = 2) -> Reconstruction:
    """Rebuild the most heavily weighted control market from the clean ones.

    Two thresholds come back with it, both estimates of the RMS error this
    reconstruction will make over the post-window:

    ``thr_in``
        block RMS of the pre-period residual of the full-pre fit. This is an
        in-sample residual, so it understates the out-of-sample error.
    ``thr_oos``
        the same statistic measured on a blank window that the fit never saw
        (Abadie and Zhao reserve blank periods for their inference; this
        reuses the device).

    The reconstruction itself always uses the full pre-period fit, which is the
    best available rebuild of ``k*``. The blank window only sets the threshold.
    """
    T, J = YN.shape
    Tp = T - T0
    kstar = int(np.argmax(v))
    clean = np.array([j for j in range(J) if v[j] > 0 and j != kstar])
    if clean.size < 2:
        raise ValueError(f"only {clean.size} clean control markets carry weight")
    if T0 // Tp < 2:
        raise ValueError(f"pre-period of {T0} holds under two blocks of {Tp}")

    weights = simplex_lstsq(YN[:T0, clean], YN[:T0, kstar])
    thr_in = block_rms(YN[:T0, kstar] - YN[:T0, clean] @ weights, Tp)

    nbk = max_blank_blocks
    while nbk >= 1 and T0 - nbk * Tp < Tp:
        nbk -= 1
    if nbk < 1:
        raise ValueError(f"pre-period of {T0} leaves no room for a blank window")
    fit_end = T0 - nbk * Tp
    w_oos = simplex_lstsq(YN[:fit_end, clean], YN[:fit_end, kstar])
    thr_oos = block_rms(YN[fit_end:T0, kstar] - YN[fit_end:T0, clean] @ w_oos, Tp)

    return Reconstruction(kstar=kstar, vk=float(v[kstar]), clean=clean,
                          weights=weights, thr_in=thr_in, thr_oos=thr_oos,
                          n_blank_blocks=nbk)


def arms(Yt: np.ndarray, w: np.ndarray, v: np.ndarray, T0: int,
         rec: Reconstruction, pi: float) -> dict:
    """Every arm's estimate at contamination ``pi``, plus the oracle.

    ``iterative``
        Melnychuk (2024): overwrite ``k*``'s post-period with its clean
        synthetic and leave ``v`` alone, so the design's pre-commitment holds.
    ``iscm``
        Di Stefano and Mellace (2024) with one affected unit. With the treated
        markets kept out of ``k*``'s donor pool the cross-weight is zero, the
        system is triangular, and this coincides with ``iterative``.
    ``renorm``
        zero ``v[k*]`` and rescale the rest, which changes ``v``.
    """
    post = slice(T0, Yt.shape[0])
    Yc = Yt.copy()
    Yc[post, rec.kstar] += pi

    rebuilt = Yc[post][:, rec.clean] @ rec.weights
    oracle = float(np.mean(Yt[post] @ w - Yt[post] @ v))
    naive = float(np.mean(Yc[post] @ w - Yc[post] @ v))

    Yi = Yc.copy()
    Yi[post, rec.kstar] = rebuilt
    iterative = float(np.mean(Yi[post] @ w - Yi[post] @ v))

    gamma = float(np.mean(Yc[post, rec.kstar] - rebuilt))
    omega = np.array([[1.0, -rec.vk], [0.0, 1.0]])
    iscm = float(np.linalg.solve(omega, np.array([naive, gamma]))[0])

    vn = v.copy()
    vn[rec.kstar] = 0.0
    vn = vn / vn.sum()

    return dict(oracle=oracle, naive=naive, iterative=iterative, iscm=iscm,
                renorm=float(np.mean(Yc[post] @ w - Yc[post] @ vn)), gamma=gamma)
