"""SDID weight-computation primitives duplicated for SpSyDiD.

These functions are *intentionally duplicated* from
``mlsynth/utils/sdid_helpers/weights.py`` rather than imported. The
duplication isolates SpSyDiD from future changes to the SDID pipeline
so silent behavioural drift cannot occur. If the upstream SDID
formulas change, this module should be updated deliberately.

Wraps the Arkhangelsky-Athey-Hirshberg-Imbens-Wager (2021) unit-weight
QP, time-weight QP, and the :math:`\\zeta = (N_{\\text{tr}}
T_{\\text{post}})^{1/4} \\cdot \\mathrm{std}(\\Delta Y)` regularisation rule
(matching the authors' ``functions_ssdid.calculate_regularization``).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..solvers.active_set import solve_simplex_qp
from ...exceptions import MlsynthDataError, MlsynthEstimationError


def fit_time_weights(
    donor_outcomes_pre: np.ndarray,
    mean_donor_outcomes_post: np.ndarray,
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """SDID time-weight QP.

    Solve for ``(beta_0, lambda)`` minimising
    :math:`\\| \\beta_0 \\mathbf 1 + \\Lambda^\\top \\mathrm{Y}_{0,\\mathrm{pre}}
    - \\bar y_{0,\\mathrm{post}} \\|_2^2` subject to ``sum(lambda) == 1``
    and ``lambda >= 0``.
    """
    if donor_outcomes_pre.ndim != 2:
        raise MlsynthDataError("donor_outcomes_pre must be 2-D (T0, N_donors).")
    if mean_donor_outcomes_post.ndim != 1:
        raise MlsynthDataError("mean_donor_outcomes_post must be 1-D.")
    T0, J_pre = donor_outcomes_pre.shape
    if T0 == 0 or J_pre == 0:
        raise MlsynthDataError("Empty donor pre-period matrix.")
    if mean_donor_outcomes_post.shape[0] != J_pre:
        raise MlsynthDataError(
            f"Donor-count mismatch: pre has {J_pre}, post mean has "
            f"{mean_donor_outcomes_post.shape[0]}."
        )

    # One row per donor: the weights combine periods, so the design is the
    # transpose. The intercept is unconstrained, so at the optimum it equals the
    # mean residual; substituting that back leaves the same program on centred
    # data, and the intercept is recovered from the weights afterwards.
    design = donor_outcomes_pre.T
    target = np.asarray(mean_donor_outcomes_post, dtype=float)
    col_means = design.mean(axis=0)
    target_mean = float(target.mean())

    try:
        lam, info = solve_simplex_qp(
            design - col_means, target - target_mean, return_info=True
        )
    except ValueError as exc:
        raise MlsynthEstimationError(
            f"Simplex solve failed in SpSyDiD fit_time_weights: {exc}"
        ) from exc
    if not info["converged"]:
        return None, None
    return float(target_mean - col_means @ lam), lam


def compute_regularization(
    donor_outcomes_pre: np.ndarray,
    num_post_periods: int,
    num_treated_units: int = 1,
) -> float:
    """SDID :math:`\\zeta = (N_{\\text{tr}} T_{\\text{post}})^{1/4}
    \\cdot \\mathrm{sd}(\\Delta Y)`.

    The standard deviation is of the first-differenced pre-period donor
    outcomes (Arkhangelsky et al. 2021 Section 3). The tuning count is the
    number of directly-treated-unit post-period observations
    :math:`N_{\\text{tr}} T_{\\text{post}}`, matching the authors' reference
    ``functions_ssdid.calculate_regularization`` (serenini/spatial_SDID), whose
    ``n_treated_post`` counts the treated-and-post rows. ``num_treated_units``
    defaults to ``1``, so a single directly-treated unit reduces to the
    :math:`T_{\\text{post}}^{1/4}` form and leaves such designs unchanged.
    """
    if donor_outcomes_pre.ndim != 2:
        raise MlsynthDataError("donor_outcomes_pre must be 2-D.")
    if num_post_periods < 0:
        raise MlsynthDataError("num_post_periods must be non-negative.")
    if num_treated_units < 1:
        raise MlsynthDataError("num_treated_units must be a positive integer.")
    if donor_outcomes_pre.shape[0] < 2 or donor_outcomes_pre.shape[1] == 0:
        sd_diff = 1.0
    else:
        diffs = np.diff(donor_outcomes_pre, axis=0).flatten()
        sd_diff = float(np.std(diffs, ddof=1)) if diffs.size > 1 else 1.0
        if not np.isfinite(sd_diff) or sd_diff <= 0:
            sd_diff = 1.0
    return float(((num_treated_units * num_post_periods) ** 0.25) * sd_diff)


def fit_unit_weights(
    donor_outcomes_pre: np.ndarray,
    mean_treated_outcome_pre: np.ndarray,
    zeta: float,
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """SDID unit-weight QP.

    Solve for ``(omega_0, omega)`` minimising
    :math:`\\| \\omega_0 \\mathbf 1 + \\mathrm Y_{0,\\mathrm{pre}} \\omega
    - \\bar y_{1,\\mathrm{pre}} \\|_2^2
    + T_0 \\zeta^2 \\|\\omega\\|_2^2`
    subject to ``sum(omega) == 1`` and ``omega >= 0``.
    """
    if donor_outcomes_pre.ndim != 2:
        raise MlsynthDataError("donor_outcomes_pre must be 2-D.")
    if mean_treated_outcome_pre.ndim != 1:
        raise MlsynthDataError("mean_treated_outcome_pre must be 1-D.")
    if zeta < 0:
        raise MlsynthDataError("zeta must be non-negative.")
    T0, J = donor_outcomes_pre.shape
    if T0 == 0:
        raise MlsynthDataError("Empty pre-period matrix.")
    if J == 0:
        raise MlsynthDataError("Need at least one donor for the unit-weight QP.")
    if mean_treated_outcome_pre.shape[0] != T0:
        raise MlsynthDataError(
            f"Pre-period length mismatch: {T0} vs {mean_treated_outcome_pre.shape[0]}."
        )

    # Two reshapings. The intercept profiles out by centring, as in
    # fit_time_weights above. The ridge T0 * zeta^2 * ||omega||^2 becomes J
    # design rows of sqrt(T0) * zeta * I carrying no target, which is the same
    # penalty written as a residual (Zou and Hastie 2005, Lemma 1).
    target = np.asarray(mean_treated_outcome_pre, dtype=float)
    col_means = donor_outcomes_pre.mean(axis=0)
    target_mean = float(target.mean())
    ridge_scale = np.sqrt(T0) * float(zeta)

    design = np.vstack([donor_outcomes_pre - col_means, ridge_scale * np.eye(J)])
    rhs = np.concatenate([target - target_mean, np.zeros(J)])

    try:
        omega, info = solve_simplex_qp(design, rhs, return_info=True)
    except ValueError as exc:
        raise MlsynthEstimationError(
            f"Simplex solve failed in SpSyDiD fit_unit_weights: {exc}"
        ) from exc
    if not info["converged"]:
        return None, None
    return float(target_mean - col_means @ omega), omega
