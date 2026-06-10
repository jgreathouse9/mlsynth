"""Algorithm 1: Sequential SDiD outer / inner loop.

Iterates over ``k = 0, 1, ..., K`` (outer) and treated cohorts
``a = a_min, ..., a_max`` (inner). At each ``(k, a)`` step the routine
solves the two regularized QPs, computes the weighted double-difference
``tau_{a,k}``, and overwrites ``Y_{a, a+k}`` with its estimated
counterfactual ``Y_{a, a+k} - tau_{a,k}``. Later steps therefore see a
panel where every previously-estimated treated cell has been replaced by
its imputed counterfactual — this is the sequential cascade that gives
the estimator its name.
"""

from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np

from .structures import SeqSDIDCohortEffect
from .weights import solve_time_qp, solve_unit_qp

# A treated cohort needs at least this many donor cohorts (later-adopting plus
# never-treated) for the unit-weight QP to balance even a rank-1 interactive
# fixed effect: matching a one-dimensional loading plus the intercept spans
# ``(1, lambda)``, which requires two affinely-independent donors. With a single
# donor the sum-to-one constraint forces ``omega = [1]`` and the cohort effect
# collapses to an unbalanced DiD against that lone donor -- biased whenever the
# factor loadings differ. We surface that as a diagnostic rather than silently
# averaging the biased cohort into the pooled event study.
_MIN_DONORS_FOR_BALANCE = 2


def run_sequential_sdid(
    Y_agg: np.ndarray,
    pi: np.ndarray,
    cohort_periods: np.ndarray,
    treated_cohort_indices: np.ndarray,
    a_min: int,
    a_max: int,
    K: int,
    eta: float,
    in_place_imputation: bool = True,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], SeqSDIDCohortEffect]]:
    """Run Algorithm 1 of Arkhangelsky & Samkov (2025).

    Parameters
    ----------
    Y_agg : np.ndarray
        Cohort-level outcome matrix, shape ``(T, A)``.
    pi : np.ndarray
        Cohort shares, length ``A``, summing to 1 across the whole sample.
    cohort_periods : np.ndarray
        1-based time index of each cohort's adoption period, length ``A``.
        Never-treated cohorts use ``np.iinfo(np.int64).max``.
    treated_cohort_indices : np.ndarray
        Column indices into ``Y_agg`` (and ``pi``) identifying the
        finitely-adopting cohorts.
    a_min, a_max : int
        Earliest / latest cohort adoption-period index to estimate.
    K : int
        Maximum horizon ``k`` to estimate.
    eta : float
        Regularization parameter (>= 0). At ``eta -> infinity`` the unit
        weights collapse to ``omega_j proportional to pi_j`` and the time
        weights to ``1 / (a + k - 1)`` (Remark 2.2 of the paper).
    in_place_imputation : bool
        Whether to update ``Y_agg[a, a+k]`` in place with the estimated
        counterfactual. The paper's Algorithm 1 does this; we expose the
        flag for diagnostics. Default: True. ``Y_agg`` is copied before
        modification, so the input is never mutated.

    Returns
    -------
    Y_imputed : np.ndarray
        The (possibly imputed) ``Y_agg`` matrix.
    cohort_effects : Dict[Tuple[int, int], SeqSDIDCohortEffect]
        Per-(cohort_period, k) point estimates plus the fitted weights.
    """

    T, A = Y_agg.shape
    Y = Y_agg.copy()

    # Treated-cohort columns sorted by adoption period (ascending).
    treated_period_order = np.argsort(cohort_periods[treated_cohort_indices])
    treated_indices_sorted = treated_cohort_indices[treated_period_order]
    treated_periods_sorted = cohort_periods[treated_indices_sorted]
    # Donors at cohort a are all cohort columns whose adoption period is
    # strictly greater than a, including the never-treated column if any.
    A_to_idx = {int(period): col for col, period in zip(treated_indices_sorted, treated_periods_sorted)}

    cohort_effects: Dict[Tuple[int, int], SeqSDIDCohortEffect] = {}
    # donor count per estimated cohort (constant across k; donors are the
    # cohorts adopting strictly after a, plus never-treated).
    donor_counts: Dict[int, int] = {}

    for k in range(K + 1):
        for a in range(a_min, a_max + 1):
            # Only estimate cohorts that actually exist in the data.
            if a not in A_to_idx:
                continue

            a_col = A_to_idx[a]
            event_period = a + k - 1  # 0-based index of period a + k

            if event_period >= T:
                # Beyond the panel; nothing to estimate.
                continue

            # Pre-event window: l < a + k  (1-based), i.e. 0-based indices 0..event_period - 1.
            pre_end = event_period  # exclusive upper bound
            if pre_end < 1:
                continue

            # Donors j > a: cohorts with adoption period > a, plus the
            # never-treated cohort.
            donor_cols = np.asarray(
                [col for col, period in zip(range(A), cohort_periods)
                 if int(period) > a],
                dtype=int,
            )
            if donor_cols.size == 0:
                continue
            donor_counts.setdefault(a, int(donor_cols.size))

            Y_pre_donors = Y[:pre_end, donor_cols]                # (pre_end, J)
            y_pre_treated = Y[:pre_end, a_col]                    # (pre_end,)
            pi_donors = pi[donor_cols]
            y_event_donors = Y[event_period, donor_cols]          # (J,)

            omega, _ = solve_unit_qp(Y_pre_donors, y_pre_treated, pi_donors, eta)
            lam, _ = solve_time_qp(Y_pre_donors, y_event_donors, eta)

            # Weighted double-difference (Algorithm 1, line 6).
            post_gap = float(Y[event_period, a_col] - omega @ y_event_donors)
            pre_gap = y_pre_treated - Y_pre_donors @ omega        # (pre_end,)
            tau = post_gap - float(lam @ pre_gap)

            cohort_effects[(a, k)] = SeqSDIDCohortEffect(
                cohort_period=int(a),
                k=int(k),
                tau=float(tau),
                omega=omega.copy(),
                lambda_w=lam.copy(),
            )

            if in_place_imputation:
                # Replace the observed treated outcome with its estimated
                # counterfactual (Algorithm 1, line 7).
                Y[event_period, a_col] = Y[event_period, a_col] - tau

    _warn_donor_starved_cohorts(donor_counts)

    return Y, cohort_effects


def _warn_donor_starved_cohorts(donor_counts: Dict[int, int]) -> None:
    """Warn if any estimated cohort lacks enough donors to balance the factor.

    Donor-starved late cohorts cannot balance their factor loadings, so their
    effects reduce to an unbalanced DiD and bias the pooled event study (the
    bias also cascades backward through the sequential imputation). We report
    which cohorts are affected and the largest ``a_max`` that keeps every
    estimated cohort balanced -- the minimal fix -- without silently dropping
    them, mirroring the library's report-don't-relax stance.
    """
    if not donor_counts:
        return
    starved = sorted(a for a, n in donor_counts.items()
                     if n < _MIN_DONORS_FOR_BALANCE)
    if not starved:
        return
    balanced = [a for a, n in donor_counts.items()
                if n >= _MIN_DONORS_FOR_BALANCE]
    detail = ", ".join(f"{a} ({donor_counts[a]} donor"
                       f"{'s' if donor_counts[a] != 1 else ''})"
                       for a in starved)
    if balanced:
        fix = (f"Lower a_max to {max(balanced)} (the latest cohort with at "
               f"least {_MIN_DONORS_FOR_BALANCE} donors)")
    else:
        fix = ("Add later-adopting or never-treated donor cohorts, or estimate "
               "fewer horizons")
    warnings.warn(
        "Sequential SDiD: treated cohort(s) "
        f"{detail} have fewer than {_MIN_DONORS_FOR_BALANCE} donor cohorts "
        "(later-adopting plus never-treated). Their effects reduce to an "
        "unbalanced DiD and can bias the pooled event study under interactive "
        f"fixed effects. {fix}.",
        UserWarning,
        stacklevel=2,
    )


def pooled_event_study(
    cohort_effects: Dict[Tuple[int, int], SeqSDIDCohortEffect],
    pi: np.ndarray,
    cohort_periods: np.ndarray,
    a_min: int,
    a_max: int,
    K: int,
) -> np.ndarray:
    """Aggregate per-cohort effects into horizon-k pooled estimates.

    Implements ``tau_hat_k^SSDiD(mu) = sum_a mu_a * tau_hat_{a, k}`` with
    ``mu_a = pi_a / sum_{a' in [a_min, a_max]} pi_a'``, the cohort-share
    weighting recommended in the paper (Eq. 2.5).

    Parameters
    ----------
    cohort_effects : Dict[(int, int), SeqSDIDCohortEffect]
        Output of :func:`run_sequential_sdid`.
    pi : np.ndarray
        Cohort shares.
    cohort_periods : np.ndarray
        1-based time indices.
    a_min, a_max : int
        Range of treated cohorts that participated in estimation.
    K : int
        Maximum horizon.

    Returns
    -------
    np.ndarray
        Length-``K + 1`` array of pooled effects.
    """

    pooled = np.zeros(K + 1)
    period_to_pi = {int(p): pi_a for p, pi_a in zip(cohort_periods, pi)}
    total = sum(period_to_pi[a] for a in range(a_min, a_max + 1)
                if a in period_to_pi)
    if total <= 0:
        return np.full(K + 1, np.nan)

    for k in range(K + 1):
        s = 0.0
        for a in range(a_min, a_max + 1):
            key = (a, k)
            if key in cohort_effects:
                s += period_to_pi[a] * cohort_effects[key].tau
        pooled[k] = s / total
    return pooled
