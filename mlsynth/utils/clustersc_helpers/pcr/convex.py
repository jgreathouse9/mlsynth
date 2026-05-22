"""Convex (simplex-constrained) weight solver for PCR-SC.

mlsynth extension on top of Algorithm 2 of Rho et al. (2025): keep the
HSVT denoising step, but replace the OLS inner solve with the classical
Abadie-Diamond-Hainmueller (2010) simplex-constrained program,

.. math::

   \\widehat{f} = \\arg\\min_{f \\in \\mathbb{R}_{\\geq 0}^{J},\\, \\mathbf{1}^\\top f = 1}
                 \\| \\widetilde{M}^- f - x_0^- \\|_2^2.

This combines the denoising robustness of PCR with the
non-extrapolation and interpretability properties of convex synthetic
control weights.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ....exceptions import MlsynthEstimationError
from ...estutils import Opt


def solve_simplex(
    denoised_donor_pre: np.ndarray,
    target_pre: np.ndarray,
    donor_names: List[str],
) -> np.ndarray:
    """Return simplex-constrained weights against the denoised donor matrix.

    Parameters
    ----------
    denoised_donor_pre : np.ndarray
        HSVT-denoised pre-period donor matrix, shape ``(T0, J)``.
    target_pre : np.ndarray
        Treated unit's pre-period outcomes, shape ``(T0,)``.
    donor_names : list of str
        Donor labels (required by :class:`Opt.SCopt`).
    """
    if denoised_donor_pre.ndim != 2:
        raise MlsynthEstimationError(
            "denoised_donor_pre must be 2D (T0, J)."
        )

    problem = Opt.SCopt(
        num_control_units=denoised_donor_pre.shape[1],
        target_outcomes_pre_treatment=target_pre,
        num_pre_treatment_periods=target_pre.shape[0],
        donor_outcomes_pre_treatment=denoised_donor_pre,
        scm_model_type="SIMPLEX",
        donor_names=list(donor_names),
    )
    primal = problem.solution.primal_vars
    return np.asarray(primal[next(iter(primal))], dtype=float)
