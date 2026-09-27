"""Convex (simplex-constrained) weight solver for PCR-SC.

mlsynth extension on top of Algorithm 2 of Rho et al. (2025): keep the
HSVT denoising step, but replace the OLS inner solve with the classical
Abadie-Diamond-Hainmueller (2010) simplex-constrained program,

.. math::

   \\widehat{f} = \\arg\\min_{f \\in \\mathbb{R}_{\\geq 0}^{J},\\, \\mathbf{1}^\\top f = 1}
                 \\| \\widetilde{M}^- f - x_0^- \\|_2.

This combines the denoising robustness of PCR with the
non-extrapolation and interpretability properties of convex synthetic
control weights.

The program is the library's shared simplex least-squares solve, reached
through :func:`~mlsynth.utils.solvers.active_set.solve_simplex_qp`. That
solver minimises the squared residual where the display above minimises the
norm; squaring is monotone on the non-negative reals, so the two have the same
minimiser. The active set solves the equality-constrained system on each free
set directly, which makes the answer exact under a rescaling of the panel
where a conic solver stops a few 1e-05 short.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ....exceptions import MlsynthEstimationError
from ...solvers.active_set import solve_simplex_qp


def solve_simplex(
    denoised_donor_pre: np.ndarray,
    target_pre: np.ndarray,
    donor_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Return simplex-constrained weights against the denoised donor matrix.

    Solves ``min_w ||x0_pre - M_pre w||_2`` subject to ``w >= 0`` and
    ``sum(w) == 1``.

    Parameters
    ----------
    denoised_donor_pre : np.ndarray
        HSVT-denoised pre-period donor matrix, shape ``(T0, J)``.
    target_pre : np.ndarray
        Treated unit's pre-period outcomes, shape ``(T0,)``.
    donor_names : list of str, optional
        Unused; retained for call-signature compatibility.
    """
    if denoised_donor_pre.ndim != 2:
        raise MlsynthEstimationError("denoised_donor_pre must be 2D (T0, J).")
    if denoised_donor_pre.shape[0] != target_pre.shape[0]:
        raise MlsynthEstimationError(
            f"Pre-period length mismatch: donors have {denoised_donor_pre.shape[0]} "
            f"rows but target has {target_pre.shape[0]}."
        )

    if denoised_donor_pre.shape[1] == 0:
        raise MlsynthEstimationError(
            "No donors: sum(w) == 1 has no solution over an empty pool."
        )

    try:
        return solve_simplex_qp(denoised_donor_pre, target_pre)
    except ValueError as exc:  # pragma: no cover - the guards above precede it
        raise MlsynthEstimationError(f"Simplex weight solve failed: {exc}") from exc
