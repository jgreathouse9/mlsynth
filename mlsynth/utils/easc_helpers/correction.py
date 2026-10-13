"""The cross-donor exposure regression and the EASC correction (Spruk 2026, Section 3.3).

Pure NumPy. For each period ``t`` and a pre-treatment base period ``b``, the
donors' outcome changes ``Y_jt - Y_jb`` are regressed across donors on
``(1, s_j, c_j)``, where ``c_j`` is the donor's own growth within each of a few
equal sub-periods that end before ``b``. The coefficient on ``s_j`` is
``phi_t``, and EASC subtracts ``phi_t EI(w)`` from the synthetic-control gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .diagnostics import exposure_imbalance


@dataclass(frozen=True)
class ExposureRegression:
    """``phi_t``, its delete-one-donor jackknife standard error, and the replicates.

    ``replicates`` is ``J x T``: row ``j`` is ``phi`` with donor ``j`` deleted.
    Phase averages need the replicates, not the per-period standard errors.
    """

    phi: np.ndarray
    se: np.ndarray
    replicates: np.ndarray


def growth_controls(Y0: np.ndarray, pre: np.ndarray, base: int, n_subperiods: int) -> np.ndarray:
    """``c_j``: each donor's change over each of ``n_subperiods`` sub-periods before ``b``.

    The pre-treatment periods strictly before the base period are split into
    ``n_subperiods`` consecutive groups of (nearly) equal length, and the
    control is the change from a group's first period to its last. The groups
    end before ``b`` so the controls share no noise with ``Y_jt - Y_jb``.
    Returns a ``J x n_subperiods`` matrix.
    """
    Y0 = np.asarray(Y0, dtype=float)
    if n_subperiods == 0:
        return np.zeros((Y0.shape[1], 0))
    before = np.flatnonzero(pre)
    before = before[before < base]
    groups = np.array_split(before, n_subperiods)
    return np.column_stack([Y0[g[-1]] - Y0[g[0]] for g in groups])


def regression_design(Y0: np.ndarray, s0: np.ndarray, pre: np.ndarray, base: int,
                      n_subperiods: int) -> np.ndarray:
    """The ``J x (2 + n_subperiods)`` design ``(1, s_j, c_j)`` of equation (8)."""
    s0 = np.asarray(s0, dtype=float)
    C = growth_controls(Y0, pre, base, n_subperiods)
    return np.column_stack([np.ones(s0.size), s0, C])


def _exposure_slope(X: np.ndarray, D: np.ndarray) -> np.ndarray:
    """The coefficient on the second column, for every column of ``D`` (J x T)."""
    return np.linalg.lstsq(X, D, rcond=None)[0][1]


def exposure_effects(Y0: np.ndarray, s0: np.ndarray, pre: np.ndarray, base: int,
                     n_subperiods: int) -> ExposureRegression:
    """``phi_t`` for every period, with its delete-one-donor jackknife (equation 8).

    Under Assumption 3 -- the growth controls span donors' composite latent
    loadings -- ``phi_t`` identifies the out-of-span part of the shock,
    ``Gamma' (r_t - r_b)`` (Proposition 4(a)).
    """
    Y0 = np.asarray(Y0, dtype=float)
    J = Y0.shape[1]
    X = regression_design(Y0, s0, pre, base, n_subperiods)
    D = Y0 - Y0[base][None, :]
    phi = _exposure_slope(X, D.T)
    replicates = np.array([_exposure_slope(np.delete(X, j, 0), np.delete(D, j, 1).T) for j in range(J)])
    se = np.sqrt((J - 1) / J * ((replicates - replicates.mean(0)) ** 2).sum(0))
    return ExposureRegression(phi=phi, se=se, replicates=replicates)


def jackknife_phase_se(replicates: np.ndarray, mask: np.ndarray) -> float:
    """Jackknife standard error of the phase average of a per-period statistic."""
    replicates = np.asarray(replicates, dtype=float)
    J = replicates.shape[0]
    ja = replicates[:, mask].mean(1)
    return float(np.sqrt((J - 1) / J * ((ja - ja.mean()) ** 2).sum()))


def hat_values(Y0: np.ndarray, s0: np.ndarray, pre: np.ndarray, base: int, n_subperiods: int) -> np.ndarray:
    """Leverage of each donor in the exposure regression (Section 6.3's influence check)."""
    X = regression_design(Y0, s0, pre, base, n_subperiods)
    return np.diag(X @ np.linalg.pinv(X.T @ X) @ X.T)


def exposure_is_identified(Y0: np.ndarray, s0: np.ndarray, pre: np.ndarray, base: int,
                           n_subperiods: int) -> bool:
    """Whether ``s_j`` varies beyond what ``(1, c_j)`` already spans across donors."""
    X = regression_design(Y0, s0, pre, base, n_subperiods)
    without = np.delete(X, 1, axis=1)
    return bool(np.linalg.matrix_rank(X) > np.linalg.matrix_rank(without))


def easc_gap(y1: np.ndarray, Y0: np.ndarray, w: np.ndarray, s1: float, s0: np.ndarray,
             phi: np.ndarray) -> Tuple[np.ndarray, float]:
    """The exposure-augmented gap ``y_1t - sum_j w_j Y_jt - phi_t EI(w)`` (equation 9)."""
    ei = exposure_imbalance(s1, s0, w)
    return np.asarray(y1, dtype=float) - np.asarray(Y0, dtype=float) @ w - np.asarray(phi) * ei, ei
