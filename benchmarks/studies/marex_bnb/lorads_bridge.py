"""Solve a lift with LoRADS and certify its multipliers exactly.

LoRADS (Han et al., "A low-rank ADMM splitting approach for semidefinite
programming", INFORMS J. Computing 2024; github.com/INFORMSJoC/2024.0701) is a
Burer-Monteiro method: an augmented Lagrangian phase on ``T = U U'`` of low
rank, then an ADMM phase. The relaxations here have nearly rank-one solutions
(the optimal design's own lift), which suits it.

The lift is written in SDPA form on its face, ``T`` of size ``K``:

- ``T_00 = 1``;
- one row per off-diagonal entry ``Y_ij = p_i' T p_j`` (``p_i`` the ``i``-th
  row of the integer map ``P``): ``Y_ij = 0`` where the lift fixes it at zero,
  ``Y_ij - x_ij = 0`` with a nonnegative LP column ``x_ij`` elsewhere. Diagonal
  entries are nonnegative already, since ``T`` is PSD.

LoRADS reads the objective block negated (SDPA's convention), so the file
carries ``-P'CP``. Its multipliers ``lambda`` satisfy ``S = C - sum lambda_i
A_i`` and are written by a three-line patch to its ``main.c`` (environment
variable ``LORADS_DUAL_OUT``), divided by its internal objective scale. They
map to the certificate's multipliers as ``y0 = lambda_0`` and ``H_ij = H_ji =
lambda_ij / 2``; ``certified_dnn.certify`` clips ``H`` to its sign, forms
``S`` exactly and bounds its smallest eigenvalue. Nothing LoRADS computes is
trusted beyond being a point at which to evaluate the certificate.

The binary is found at ``$LORADS_BIN``.
"""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
import time
from fractions import Fraction

import numpy as np

from benchmarks.studies.marex_bnb.certified_dnn import best_shift, certify, objective
from benchmarks.studies.marex_bnb.certify import down


def _rows(lift):
    N = len(lift.rows)
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def write_sdpa(lift, Cf, path):
    P = lift.P.astype(float)
    K = P.shape[1]
    Ct = P.T @ Cf @ P
    pairs = _rows(lift)
    free = [k for k, (i, j) in enumerate(pairs) if not lift.zero[i, j]]
    col = {k: c + 1 for c, k in enumerate(free)}
    out = [str(1 + len(pairs)), "2", f"{K} -{len(free)}",
           " ".join(["1"] + ["0"] * len(pairs))]
    iu = np.triu_indices(K)
    for a, b in zip(*iu):
        if Ct[a, b] != 0:
            out.append(f"0 1 {a + 1} {b + 1} {-Ct[a, b]:.17g}")
    out.append("1 1 1 1 1")
    for k, (i, j) in enumerate(pairs):
        A = (np.outer(P[i], P[j]) + np.outer(P[j], P[i])) / 2
        for a, b in zip(*np.nonzero(np.triu(A))):
            out.append(f"{k + 2} 1 {a + 1} {b + 1} {A[a, b]:.17g}")
        if k in col:
            out.append(f"{k + 2} 2 {col[k]} {col[k]} -1")
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    return pairs


def lorads_bound(panel, lift, args=("--timesLogRank", "1.0"), timeout=3600):
    """``(certified lower bound, LoRADS status, seconds)`` for one lift."""
    n = panel.Bc.shape[1]
    diag = max(float(panel.G.frac(j, j)) for j in range(n))
    shift = math.frexp(diag)[1]
    C = objective(panel.G, lift, shift)
    Cf = C.float()
    with tempfile.TemporaryDirectory() as tmp:
        sdpa = os.path.join(tmp, "lift.dat-s")
        dual = os.path.join(tmp, "dual.txt")
        pairs = write_sdpa(lift, Cf, sdpa)
        t0 = time.perf_counter()
        subprocess.run([os.environ["LORADS_BIN"], sdpa, *args],
                       env={**os.environ, "LORADS_DUAL_OUT": dual},
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=timeout, check=False)
        secs = time.perf_counter() - t0
        if not os.path.exists(dual):
            return 0.0, "no output", secs
        with open(dual) as fh:
            status = fh.readline().split()[0]
            lam = np.array([float(x) for x in fh.read().split()])
    N = len(lift.rows)
    H = np.zeros((N, N))
    for k, (i, j) in enumerate(pairs):
        H[i, j] = H[j, i] = lam[k + 1] / 2
    y0 = lam[0]
    Hc = np.where(lift.zero, H, np.clip(H, 0.0, None))
    S = lift.P.T.astype(float) @ (Cf - Hc) @ lift.P.astype(float)
    S[0, 0] -= y0
    _, d = best_shift(S, y0, lift.tau)
    lb = certify(C, lift, y0 - d, H, None)
    return down(max(lb, Fraction(0)) * (1 << shift)), status, secs
