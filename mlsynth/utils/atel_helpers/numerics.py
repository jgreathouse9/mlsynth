"""The weighted least-squares solve the three estimation steps share.

Every stage of ATEL -- the bandwidth search, the loading fit, and the variance's
pre-period refit -- solves the same kernel-weighted normal equations. Both the
factor block and the local-linear block are built from the same factors, so the
design is near-singular by construction and the pseudo-inverse's cutoff decides
what is treated as rank. The tolerance here is the one the reference
implementation uses, ``max(shape) * eps`` relative to the largest singular
value, which is Octave's and MATLAB's ``pinv`` default; numpy's own default of
``1e-15`` is tighter and moves the fit on these designs.
"""

from __future__ import annotations

import numpy as np

__all__ = ["reference_pinv", "weighted_ls"]

_EPS = float(np.finfo(float).eps)


def reference_pinv(A: np.ndarray) -> np.ndarray:
    """Pseudo-inverse at ``max(shape) * eps(norm(A))``, the reference tolerance."""
    A = np.asarray(A, dtype=float)
    return np.linalg.pinv(A, rcond=max(A.shape) * _EPS)


def weighted_ls(
    design: np.ndarray, kernel: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Solve ``(X' K X)^+ X' K y`` for a diagonal kernel ``K``.

    A positive rescaling of ``kernel`` leaves the result unchanged, which is why
    a kernel normalizer that is off by a constant factor cannot move any
    reported number.
    """
    X = np.asarray(design, dtype=float)
    k = np.asarray(kernel, dtype=float)
    y = np.asarray(target, dtype=float)
    return reference_pinv(X.T @ (k[:, None] * X)) @ (X.T @ (k * y))
