"""Becker and Klossner's sunny/shady donor screen for simplex programs.

Write ``x_j = B[:, j] - A`` for the centred donor columns and
``H = conv(x_1, ..., x_J)``. The screen asks one question per donor:

    alpha*(j) = min { alpha >= 0 : alpha x_j in H }

Donor ``j`` is sunny when ``alpha*(j) = 1`` and shady when ``alpha*(j) < 1``, i.e.
when some shorter multiple of its own column already lies in the hull. Since
``x_j`` is one of the generators of ``H``, ``alpha = 1`` is always feasible, so
``alpha* <= 1`` and the test is one-sided by construction.

Two facts give the screen its use, and both are pinned in
``mlsynth/tests/test_sunny_donors.py``:

Proposition 1. No donor is sunny exactly when ``0 in H``, which is exactly when
``min ||A - B w||`` over the simplex is zero. So a screen that returns all-shady
has found an exact fit, not a set of columns to drop.

Proposition 2. If no exact fit exists, every optimum puts zero weight on every
shady donor. At an optimum ``u* = B w* - A`` the simplex stationarity conditions
give ``<x_k, u*> = min_i <x_i, u*> = ||u*||^2`` for each ``k`` in the support, and
the feasible move ``w* - t e_j + t lambda`` built from ``alpha x_j = sum_i lambda_i
x_i`` sends ``u*`` to ``u* - t (1 - alpha) x_j``. Together these force
``alpha ||u*||^2 >= ||u*||^2``, so ``alpha < 1`` implies ``u* = 0``. The conclusion
holds at every optimum, so the columns may be dropped even when the argmin is a
face.

The hypothesis on Proposition 2 is not slack that can be ignored. A donor with
``x_j = 0`` reproduces the treated path on its own and is shady with
``alpha*(j) = 0``; dropping it would discard an exact fit. Proposition 1 is what
rules this out: when at least one donor is sunny there is no exact fit, and only
then is pruning licensed. ``sunny_support`` encodes that branch.

``alpha*`` costs one linear program per donor. ``certified_sunny`` decides a subset
of them from a single Gram matrix instead. For any direction ``c``,
``alpha x_j = sum_i lambda_i x_i`` gives ``alpha c'x_j = sum_i lambda_i c'x_i >=
min_i c'x_i``, so whenever ``c'x_j > 0``

    alpha*(j) >= (min_i c'x_i) / (c'x_j),

and a bound reaching 1 certifies donor ``j`` sunny. Sweeping ``c`` over the donor
columns themselves reads every such bound off ``G = Xt' Xt``: row ``i`` supplies
``min_k G[i, k] / G[i, j]`` for each ``j`` with ``G[i, j] > 0``. The direction that
certifies a donor is usually not its own column, so the ``c = x_j`` term alone
leaves most of the power unused -- on 40x8 Gaussian designs the own-column test
certifies nothing at all while the full sweep certifies 40 percent of donors.

Where the screen pays, the certificate helps least. Over 200 Gaussian designs, 40
per regime, it certifies 40 percent of donors at 40x8, 29 percent at 12x12, 10
percent at 8x40 and 2 percent at 5x74 -- and the first two regimes have every donor
sunny, so there is nothing there to prune. At 5x74, where only 10 percent of donors
are sunny and pruning pays, skipping the certified donors cuts 2 percent off the
runtime. The certificate is sound and near free, not a replacement for the linear
program.

Reference: Becker and Klossner (2017), MSCMT; ``isSunny`` in ``R/Helpers.r`` and the
donor loop in ``R/multiOpt.r``.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import linprog

__all__ = ["sunny_alphas", "certified_sunny", "sunny_donors", "sunny_support"]

#: A donor counts as sunny when ``alpha*`` reaches 1 to within this slack. The
#: linear program returns ``alpha*`` to roughly solver precision, so the default
#: sits several orders above that and well below the gap to any shady donor met in
#: practice.
DEFAULT_TOL = 1e-7


def _centred(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Validate the design and return the centred donor columns ``B - A``."""
    B = np.asarray(B, dtype=float)
    A = np.asarray(A, dtype=float)
    if B.ndim != 2:
        raise ValueError(f"B must be a 2-D (m, J) matrix; got shape {B.shape}.")
    if B.shape[1] == 0:
        raise ValueError("B has no columns; the screen needs at least one donor.")
    if A.ndim != 1:
        raise ValueError(f"A must be a 1-D vector of length m; got shape {A.shape}.")
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            f"A has {A.shape[0]} entries but B has {B.shape[0]} rows; they must match."
        )
    if not (np.isfinite(B).all() and np.isfinite(A).all()):
        raise ValueError("B and A must be finite; got a NaN or an infinity.")
    return B - A[:, None]


def _program(Xt: np.ndarray):
    """Build the shared part of the donor LP.

    The variables are ``[alpha, lambda_1, ..., lambda_J]``; the rows say
    ``alpha x_j = Xt lambda`` and ``sum lambda = 1``. Only the first column depends
    on ``j``, so the rest is built once and reused across donors.
    """
    m, J = Xt.shape
    A_eq = np.zeros((m + 1, J + 1))
    A_eq[:m, 1:] = -Xt
    A_eq[m, 1:] = 1.0
    b_eq = np.zeros(m + 1)
    b_eq[m] = 1.0
    c = np.zeros(J + 1)
    c[0] = 1.0
    return A_eq, b_eq, c


def _alpha(Xt: np.ndarray, j: int, A_eq: np.ndarray, b_eq: np.ndarray,
           c: np.ndarray) -> float:
    """Solve for ``alpha*(j)``.

    ``alpha = 1, lambda = e_j`` is always feasible and ``alpha >= 0`` bounds the
    objective, so a failure here is numerical. The fallback is 1, which reports the
    donor as sunny: that keeps the column, where a wrong "shady" would drop one
    that should carry weight.
    """
    A_eq[:-1, 0] = Xt[:, j]
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0.0, None), method="highs")
    if not res.success:
        warnings.warn(
            f"The sunny/shady linear program for donor {j} did not solve "
            f"({res.message.strip()}); reporting the donor as sunny, which keeps it.",
            RuntimeWarning,
            stacklevel=3,
        )
        return 1.0
    return float(res.fun)


def sunny_alphas(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Return ``alpha*(j)`` for every donor, one linear program each.

    This is the reference: it consults no certificate, so it is what the cheap
    tests are measured against.
    """
    Xt = _centred(B, A)
    A_eq, b_eq, c = _program(Xt)
    return np.array([_alpha(Xt, j, A_eq, b_eq, c) for j in range(Xt.shape[1])])


def certified_sunny(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Flag the donors that a single Gram matrix proves sunny.

    Row ``i`` of ``G = Xt' Xt`` carries the bound from direction ``c = x_i``, namely
    ``min_k G[i, k] / G[i, j]`` for every ``j`` whose inner product with ``x_i`` is
    positive; donor ``j`` is certified when the best bound over all rows reaches 1.
    The flags are a subset of the sunny donors -- a donor left unflagged has not been
    shown to be shady and still needs its linear program.
    """
    Xt = _centred(B, A)
    G = Xt.T @ Xt
    scale = float(np.diag(G).max())
    if not scale > 0.0:
        return np.zeros(Xt.shape[1], dtype=bool)
    usable = G > 1e-12 * scale
    bounds = np.where(
        usable, G.min(axis=1)[:, None] / np.where(usable, G, 1.0), -np.inf
    )
    return bounds.max(axis=0) >= 1.0


def sunny_donors(B: np.ndarray, A: np.ndarray, *, tol: float = DEFAULT_TOL,
                 certify: bool = True) -> np.ndarray:
    """Classify every donor as sunny (``True``) or shady (``False``).

    With ``certify`` the Gram test above settles what it can and a linear program is
    solved only for the rest; the answer is the same either way, since the
    certificate proves sunniness and never asserts shadiness.
    """
    Xt = _centred(B, A)
    flags = (certified_sunny(B, A) if certify
             else np.zeros(Xt.shape[1], dtype=bool))
    undecided = np.flatnonzero(~flags)
    if undecided.size:
        A_eq, b_eq, c = _program(Xt)
        for j in undecided:
            flags[j] = _alpha(Xt, j, A_eq, b_eq, c) >= 1.0 - tol
    return flags


def sunny_support(B: np.ndarray, A: np.ndarray, *, tol: float = DEFAULT_TOL,
                  certify: bool = True) -> np.ndarray:
    """Return the donor indices a simplex solve can be restricted to.

    When some donor is sunny these are the sunny ones: by Proposition 2 the shady
    columns take zero weight at every optimum, so the restricted program has the
    same value and the same reachable fits. When no donor is sunny an exact fit
    exists, Proposition 2 does not apply, and every index is returned -- the shady
    columns are the ones that reach the fit.
    """
    flags = sunny_donors(B, A, tol=tol, certify=certify)
    if not flags.any():
        return np.arange(flags.size)
    return np.flatnonzero(flags)
