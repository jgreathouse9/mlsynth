"""Feng's Section 5 Monte Carlo: the three DGPs, LPCA, and the GPCA baseline.

Ported from ``Simulation/Feng-2024_LPCA_simul_suppfuns.R`` (``dgp`` / ``compute``
/ ``comstat`` / ``sim``) and its driver ``Feng-2024_LPCA_simul.R``. The three
models are rows 1-3 of ``Feng-2024_LPCA_simul_models.csv``, all at
``n = p = 1000``, half the columns for matching and half for the local SVD.

Table 1 of the paper reports, per model and per ``K``, the maximum absolute
error over every cell of the held-out block, and the prediction error at three
units whose latent ``alpha`` sits at the 0.1, 0.5 and 0.9 sample quantiles --
those units have their last-period entry zeroed, which is the simulation's
stand-in for a treated cell.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lpca_oracle import pseudo_max_distance, select_rank  # noqa: E402

# Feng-2024_LPCA_simul_models.csv, rows 1-3.
MODELS = (
    {"name": "Model 1", "fn": 1, "sigma": 0.5, "binary": False},
    {"name": "Model 2", "fn": 2, "sigma": 0.5, "binary": False},
    {"name": "Model 3", "fn": 3, "sigma": 1.0, "binary": True},
)
N = P = 1000
MAX_COMPONENTS = 3      # nlam
MATCH_FRACTION = 0.5    # prop


def k_sequence(n: int = N) -> list[int]:
    """The three neighbourhood sizes, reproducing the R script's rounding.

    ``K <- n^(2*1/(2*1+1))`` evaluates to 99.99999999999999 in double precision,
    so ``floor(K * c(0.5, 1, 1.5))`` gives 49, 99, 149 -- the column headings in
    Table 1 -- and not the 50, 100, 150 an exact cube root would give.
    """
    K = n ** (2 * 1 / (2 * 1 + 1))
    return [int(np.floor(K * c)) for c in (0.5, 1.0, 1.5)]


def surface(kind: int, alpha: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """The latent surface ``eta_l(alpha_i)`` for each model (R: ``funlist``).

    The R code fixes ``sig = 0.1`` in models 1 and 2, so model 1's exponent is
    ``-(u - v)^2 / 0.01``; the paper's Section 5 prints ``-10(u - v)^2``. The
    code is what Table 1 was produced from.
    """
    u, v = alpha[:, None], eta[None, :]
    if kind == 1:
        return (1 / np.sqrt(2 * np.pi) / 0.1) * np.exp(-((u - v) ** 2) / 0.1 ** 2)
    if kind == 2:
        return np.exp(-np.abs(u - v) / 0.1)
    return 1 - (1 + np.exp(15 * (0.8 * np.abs(u - v)) ** 0.8 - 0.1)) ** -1


def draw(model, rng, n: int = N, p: int = P):
    """One simulated dataset (R: ``dgp``), plus the three evaluation units."""
    alpha = rng.uniform(0, 1, n)
    eta = rng.uniform(0, 1, p)
    mean = surface(model["fn"], alpha, eta)
    if model["binary"]:
        x = (rng.uniform(0, 1, (n, p)) <= mean).astype(float)
    else:
        x = mean + rng.normal(0, model["sigma"], (n, p))
    # R: nth(alpha, round(q*n), index.return=T) -- the units at the alpha quantiles.
    order = np.argsort(alpha, kind="stable")
    evaluated = order[[int(round(q * n)) - 1 for q in (0.1, 0.5, 0.9)]]
    return x, mean, evaluated


def eigenvalue_ratio_rank(X: np.ndarray, rmax: int = 9) -> int:
    """Ahn-Horenstein (2013) eigenvalue-ratio factor count (R: ``HDRFA::PCA_FN``)."""
    s = np.linalg.svd(X, compute_uv=False)
    lam = s ** 2 / (X.shape[0] * X.shape[1])
    lam = lam[: rmax + 1]
    return int(np.argmax(lam[:-1] / lam[1:]) + 1)


def gpca_fit(x: np.ndarray, double_demean: bool) -> np.ndarray:
    """Global PCA baseline: rank by eigenvalue ratio, then a truncated SVD.

    Table 1's GPCA column is the doubly demeaned variant, matching the paper's
    statement that the factor count comes from doubly demeaned data.
    """
    col_mean = x.mean(axis=0)
    centred = x - col_mean
    row_mean = centred.mean(axis=1)[:, None] if double_demean else 0.0
    centred = centred - row_mean
    r = eigenvalue_ratio_rank(centred)
    U, s, Vt = np.linalg.svd(centred, full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vt[:r] + row_mean + col_mean


def one_replication(model, rng, n: int = N, p: int = P):
    """Return ``{K or 'gpca': (max_abs_error, err_q10, err_q50, err_q90)}``.

    R reference: ``sim``. The three evaluation units get their last-period entry
    zeroed before anything is fitted, and the per-unit errors are read off that
    column.
    """
    p1 = int(round(p * MATCH_FRACTION))
    x, mean, evaluated = draw(model, rng, n, p)
    x[evaluated, p - 1] = 0.0                      # R: x[eval.ind, p] <- 0
    truth = mean[:, p1:]
    block = x[:, p1:]

    def stats(est):
        dev = np.abs(est - truth)
        return (float(dev.max()), *(float(v) for v in dev[evaluated, -1]))

    out = {}
    distance = pseudo_max_distance(x[:, :p1])      # shared across the K grid
    for K in k_sequence(n):
        fitted = np.empty_like(truth)
        for i in range(n):
            # R: tmp.d <= nth(tmp.d, K) -- a threshold, so exact ties widen the
            # neighbourhood. Model 3 is binary, where the distance can tie.
            kth = np.partition(distance[i], K - 1)[K - 1]
            rows = np.flatnonzero(distance[i] <= kth)
            neighbourhood = block[rows]
            # Only one row of the rank-r reconstruction is ever read, and it is
            # ``(U[pos, :r] U[:, :r]') B`` -- the right singular vectors cancel.
            # So the left factor of the K x K Gram suffices, which is 7x faster
            # than the K x p SVD and agrees with it to 5e-14.
            w, V = np.linalg.eigh(neighbourhood @ neighbourhood.T)
            top = np.argsort(w)[::-1][:MAX_COMPONENTS]
            s = np.sqrt(np.maximum(w[top], 0.0))
            U = V[:, top]
            r = select_rank(s, K)
            pos = int(np.flatnonzero(rows == i)[0])
            fitted[i] = (U[:, :r] @ U[pos, :r]) @ neighbourhood
        out[K] = stats(fitted)
    out["gpca"] = stats(gpca_fit(x, double_demean=True)[:, p1:])
    return out
