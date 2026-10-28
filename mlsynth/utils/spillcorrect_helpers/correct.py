"""The two corrections of a design's contaminated controls, on arrays.

A design's estimate is its treated composite ``sum_j w_j y_j`` less its control
composite ``sum_i v_i y_i``. Controls ``k`` carrying spillovers ``gamma_k(t)``
bias it by exactly ``-sum_k v_k gamma_k(t)``. Both corrections estimate the
``gamma_k`` and leave the composites as they are.

``iterative`` (Melnychuk 2024, Section 2)
    Each contaminated control gets its own synthetic control from the clean
    controls, and that spillover-free synthetic replaces the control's outcomes
    from ``start`` on.
``inclusive`` (Di Stefano and Mellace 2024, Sections 2 and 3)
    The treated composite is the "main treated" unit and the design's control
    composite its synthetic control. Each contaminated control's synthetic
    control draws on the clean controls, the treated composite (weight ``l_k``)
    and the other contaminated controls (weights ``l_ki``). Under their
    Assumptions 1 and 2 the gaps from ``start`` on are

        gap_1 = theta - sum_k v_k gamma_k,
        gap_k = gamma_k - l_k theta - sum_{i != k} l_ki gamma_i,

    the system ``Omega [theta, gamma]' = [gap_1, gap]'`` with ``Omega`` the
    identity less the cross-weight matrix (SPILLSYNTH's ``build_omega`` and
    ``solve_inclusive``). It has a solution only where ``det Omega`` is not
    zero.

Each synthetic control is SPILLSYNTH's per-unit fit
(:func:`~mlsynth.utils.spillsynth_helpers.iscm.weights.build_unit_sc`) on the
periods before ``start``. With ``scale`` it fits each series divided by its mean
over those periods, so it reaches a control larger than every clean one, and
its weights are level weights, which the inclusive system takes unchanged.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ...exceptions import MlsynthEstimationError
from ..spillsynth_helpers.iscm.system import build_omega, solve_inclusive
from ..spillsynth_helpers.iscm.weights import build_unit_sc
from .structures import Correction

#: Below this absolute determinant the inclusive system has no usable solution:
#: the contaminated controls and the treated composite load on each other fully.
SINGULAR = 1e-8


def iterative_correction(Y: np.ndarray, ks: Sequence[int], clean: Sequence[int],
                         start: int, *, scale: bool = True) -> Correction:
    """Rebuild each row in ``ks`` from the rows in ``clean`` and replace it from
    ``start`` on."""
    Y = np.asarray(Y, dtype=float)
    cleaned = Y.copy()
    donors = np.asarray(list(clean), dtype=int)
    spill, rmspe = [], []
    for k in ks:
        _, cf, gap, pre_rmspe, _ = build_unit_sc(int(k), donors, Y, start, scale=scale)
        cleaned[k, start:] = cf[start:]
        spill.append(gap[start:])
        rmspe.append(pre_rmspe)
    return Correction(cleaned=cleaned,
                      spillover=np.array(spill, dtype=float).reshape(len(ks), Y.shape[1] - start),
                      cleaning_rmspe=np.array(rmspe, dtype=float), start=int(start))


def inclusive_correction(Y: np.ndarray, ks: Sequence[int], clean: Sequence[int],
                         start: int, *, treated_series: np.ndarray, gap: np.ndarray,
                         v: np.ndarray, scale: bool = True) -> Correction:
    """Solve the inclusive system for the spillovers on the rows in ``ks``.

    ``treated_series`` is the treated composite's observed series and ``gap``
    its contrast with the design's control composite, both over every period;
    ``v`` holds the rows' weights in that control composite.

    Raises
    ------
    MlsynthEstimationError
        If ``|det Omega|`` is below :data:`SINGULAR`.
    """
    Y = np.asarray(Y, dtype=float)
    ks = [int(k) for k in ks]
    clean = [int(j) for j in clean]
    v = np.asarray(v, dtype=float)
    aug = np.vstack([Y, np.asarray(treated_series, dtype=float)[None, :]])
    composite = Y.shape[0]
    m = len(ks)
    cross = np.zeros((m + 1, m + 1))
    cross[0, 1:] = v
    gaps, rmspe = [np.asarray(gap, dtype=float)[start:]], []
    for i, k in enumerate(ks):
        others = [j for j in range(m) if j != i]
        pool = np.array(clean + [composite] + [ks[j] for j in others], dtype=int)
        w, _, gap_k, pre_rmspe, _ = build_unit_sc(k, pool, aug, start, scale=scale)
        cross[i + 1, 0] = w[len(clean)]
        cross[i + 1, [j + 1 for j in others]] = w[len(clean) + 1:]
        gaps.append(gap_k[start:])
        rmspe.append(pre_rmspe)
    omega = build_omega(cross)
    det = float(np.linalg.det(omega))
    if abs(det) < SINGULAR:
        raise MlsynthEstimationError(
            f"SPILLCORRECT method='inclusive': the cross-weight system is singular "
            f"(det Omega = {det:.2e}). The contaminated controls and the treated "
            "composite load on each other fully, so the effect and the spillovers "
            "cannot be told apart. Use method='iterative'."
        )
    solved = solve_inclusive(omega, np.vstack(gaps))
    cleaned = Y.copy()
    for i, k in enumerate(ks):
        cleaned[k, start:] = Y[k, start:] - solved[i + 1]
    return Correction(cleaned=cleaned, spillover=solved[1:], cleaning_rmspe=np.array(rmspe),
                      start=int(start), composite_weight=cross[1:, 0].copy(), omega_det=det)
