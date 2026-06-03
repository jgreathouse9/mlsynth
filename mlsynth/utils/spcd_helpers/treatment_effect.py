"""Treatment-effect assembly for SPCD (final step of Algorithms 1 & 2).

Implements the synthetic-paths construction and treatment-effect
formula at the bottom of Algorithm 1 (page 7) and the bottom of
Algorithm 2 (page 13)::

    Treat Unit i if  gamma(i) = -sgn(sum_j gamma(j))            (minority rule)

    tau_hat = sum_t (
                  sum_{gamma(i)=sgn(sum_j gamma(j))}  w(i) Y_{i, T+t}
                - sum_{gamma(i)=-sgn(sum_j gamma(j))} w(i) Y_{i, T+t}
              )

The signed-weight convention used elsewhere in this module collapses
the above difference-of-sums into a single dot product
``Y_post @ contrast_weights`` once the signs of ``contrast_weights``
match the minority-flipped sign vector.

Reference
---------
Lu, Y., Li, J., Ying, L., & Blanchet, J. (2022).
"Synthetic Principal Component Design: Fast Covariate Balancing with
Synthetic Controls." arXiv:2211.15241v1.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def apply_minority_flip(
    y_star: np.ndarray, raw_weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the minority-group convention from the bottom of Algorithm 1.

    Per the paper (page 7, bottom of Algorithm 1):

        "Treat Unit i if  gamma(i) = -sgn(sum_j gamma(j))
         (To ensure the size of the treated group is smaller than the
         control group.)"

    If the spectral iteration converges to a sign vector whose positive
    entries outnumber its negative entries, we flip the sign so that
    ``+1`` corresponds to the minority (treated) group. The raw weights
    are flipped in lock-step so that ``raw_weights`` continues to be the
    signed Eq. (9) (or Eq. 6) weights of the chosen treated and control
    groups.

    Parameters
    ----------
    y_star : np.ndarray
        Length-N sign vector ``y* in {-1, +1}^N`` from the iteration.
    raw_weights : np.ndarray
        Length-N signed weights from Eq. (9) or Eq. (6).

    Returns
    -------
    assignment_pm1 : np.ndarray
        Possibly sign-flipped version of ``y_star``, with ``+1`` marking
        the minority (treated) group.
    raw_weights_flipped : np.ndarray
        Possibly sign-flipped ``raw_weights`` consistent with the new
        assignment.
    """

    s = np.sign(np.sum(y_star))
    if s > 0:
        return -y_star.astype(float), -raw_weights.astype(float)
    return y_star.astype(float), raw_weights.astype(float)


def build_weight_groups(
    assignment_pm1: np.ndarray, raw_weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split signed weights into treated/control/contrast components.

    The signed weight vector returned by Eq. (9) or Eq. (6) already
    encodes group membership: positive entries belong to the treated
    group, negative entries to the control group. This helper turns
    that into the four arrays used downstream by the plotter and by the
    treatment-effect formula.

    Parameters
    ----------
    assignment_pm1 : np.ndarray
        Length-N sign vector with ``+1`` for treated units, ``-1`` for
        control units.
    raw_weights : np.ndarray
        Length-N signed weights.

    Returns
    -------
    selected_mask : np.ndarray
        0/1 indicator of treated units (parity with the SYNDES design
        container's ``assignment`` field).
    treated_weights : np.ndarray
        Length-N weights restricted to the treated group; zero
        elsewhere. Normalized to sum to 1 over treated units.
    control_weights : np.ndarray
        Length-N weights restricted to the control group; zero
        elsewhere. Normalized to sum to 1 over control units.
    contrast_weights : np.ndarray
        Length-N signed contrast weights ``treated_weights -
        control_weights``. The synthetic gap is
        ``Y_full @ contrast_weights``.
    """

    selected_mask = (assignment_pm1 > 0).astype(int)

    treated_mask = (assignment_pm1 > 0)
    control_mask = (assignment_pm1 < 0)

    treated_raw = np.where(treated_mask, raw_weights, 0.0)
    control_raw = np.where(control_mask, -raw_weights, 0.0)

    treated_sum = treated_raw.sum()
    control_sum = control_raw.sum()
    treated_weights = treated_raw / treated_sum if treated_sum != 0 else treated_raw
    control_weights = control_raw / control_sum if control_sum != 0 else control_raw

    contrast_weights = treated_weights - control_weights
    return selected_mask, treated_weights, control_weights, contrast_weights


def compute_att_and_fit(
    Y_pre: np.ndarray,
    Y_post: Optional[np.ndarray],
    treated_weights: np.ndarray,
    control_weights: np.ndarray,
) -> Tuple[float, float, Optional[float]]:
    """Compute SPCD's ATT and pre/post fit RMSEs.

    Implements the ``tau_hat`` formula at the bottom of Algorithm 1
    (page 7) of the paper, averaged over the post-treatment horizon::

        ATT = (1/S) sum_{t=T+1..T+S} (
                sum_{gamma(i)=+1} w(i) Y_{i,t}
              - sum_{gamma(i)=-1} w(i) Y_{i,t}
              )
            = mean(synthetic_gap[post])

    When no post-treatment matrix is provided, the SPCD design step is
    pure covariate-balancing and ATT is reported as 0.0.

    ``rmse_pre`` measures the pre-period balance achieved by the design
    (the residual of Eq. (1)/(2) on the chosen sign vector); smaller is
    better. ``rmse_post`` measures the post-period dispersion of the
    synthetic gap, useful for sanity-checking convergence behavior.

    Parameters
    ----------
    Y_pre : np.ndarray
        Pre-treatment matrix of shape ``(T_pre, N)``.
    Y_post : np.ndarray or None
        Post-treatment matrix of shape ``(T_post, N)``, or ``None``.
    treated_weights : np.ndarray
        Length-N treated-group weights summing to 1.
    control_weights : np.ndarray
        Length-N control-group weights summing to 1.

    Returns
    -------
    att : float
        Mean of the post-period synthetic gap, or ``0.0`` when ``Y_post``
        is ``None``.
    rmse_pre : float
        Root-mean-square synthetic gap over the pre-treatment period.
    rmse_post : float or None
        Root-mean-square synthetic gap over the post-treatment period,
        or ``None`` when ``Y_post`` is ``None``.
    """

    pre_gap = Y_pre @ (treated_weights - control_weights)
    rmse_pre = float(np.sqrt(np.mean(pre_gap ** 2)))

    if Y_post is None:
        return 0.0, rmse_pre, None

    post_gap = Y_post @ (treated_weights - control_weights)
    att = float(np.mean(post_gap))
    rmse_post = float(np.sqrt(np.mean(post_gap ** 2)))
    return att, rmse_pre, rmse_post

def build_synthetic_paths(
    Y_pre: np.ndarray,
    Y_post: Optional[np.ndarray],
    treated_weights: np.ndarray,
    control_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Form synthetic treated, synthetic control, and gap trajectories.

    Parameters
    ----------
    Y_pre : np.ndarray
        Pre-treatment matrix of shape ``(T_pre, N)``.
    Y_post : np.ndarray or None
        Post-treatment matrix of shape ``(T_post, N)``, or ``None``.
    treated_weights : np.ndarray
        Length-N treated-group weights summing to 1.
    control_weights : np.ndarray
        Length-N control-group weights summing to 1.

    Returns
    -------
    synthetic_treated : np.ndarray
        Synthetic treated trajectory over ``T_pre + T_post`` periods.
    synthetic_control : np.ndarray
        Synthetic control trajectory over ``T_pre + T_post`` periods.
    synthetic_gap : np.ndarray
        Difference ``synthetic_treated - synthetic_control``.
    """

    if Y_post is None:
        Y_full = Y_pre
    else:
        Y_full = np.vstack([Y_pre, Y_post])

    synthetic_treated = Y_full @ treated_weights
    synthetic_control = Y_full @ control_weights
    synthetic_gap = synthetic_treated - synthetic_control
    return synthetic_treated, synthetic_control, synthetic_gap
