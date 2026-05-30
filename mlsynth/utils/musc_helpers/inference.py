"""Design-based inference for MUSC.

Two inference outputs from Bottmer, Imbens, Spiess & Warnick (2024):

1. **Proposition 1 unbiased variance estimator** (equation 3.3).
   Under random unit assignment, the conditional variance of any
   Generalised-Synthetic-Control estimator has a closed-form unbiased
   estimator that depends only on the weight matrix and the realised
   outcomes at the treated period. The formula has four terms; we
   implement them as a single nested loop over candidate treated
   units, matching the paper's ``var_gsc_intercept.m`` reference.

2. **Randomization-based confidence interval** (Section 3.5).
   For each non-treated unit ``i``, refit MUSC pretending ``i`` is
   treated (leave-one-out), giving placebo ATTs that, under random
   assignment, are draws from the null distribution of the test
   statistic. Inverting the resulting permutation test gives an exact
   ``(1 − alpha)`` confidence interval for the treated unit's ATT.

We also expose a Normal-approximation CI keyed off the unbiased
variance, which Table 6 of the paper shows can mildly under- or
over-cover relative to the randomization-based interval.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .estimation import att_for_unit, solve_musc_qp


# ---------------------------------------------------------------------------
# Proposition 1: unbiased variance
# ---------------------------------------------------------------------------

def unbiased_variance(M_t: np.ndarray, y_t: np.ndarray) -> float:
    """Unbiased estimate of ``Var[τ̂]`` under random unit assignment.

    Direct port of Bottmer et al. (2024) equation 3.3, originally
    ``var_gsc_intercept.m`` in the authors' archive. The estimator is
    the *mean across candidate treated units* of a four-term per-unit
    expression:

    .. math::

       \\hat V \\;=\\; \\frac{1}{N}\\sum_i \\Big[
           \\frac{1}{N-3}\\sum_{k \\neq i}
               \\big(\\sum_{j \\neq i, j \\neq k}
                     M_{k,j}(Y_k - Y_j)\\big)^2
           - \\frac{1}{(N-2)(N-3)}\\sum_{k \\neq i}\\sum_{j \\neq i, j \\neq k}
               M_{k,j}^2 (Y_k - Y_j)^2
           + \\frac{2}{N-2}\\sum_{k \\neq i} \\alpha_k
               \\sum_{j \\neq i, j \\neq k} M_{k,j}(Y_j - Y_k)
           + \\frac{1}{N}\\sum_{k} \\alpha_k^2
       \\Big],

    where ``α_k = M_t[k, 0]`` is the intercept column and
    ``M_{k, j} = M_t[k, j+1]`` is the weight block.

    Parameters
    ----------
    M_t : np.ndarray
        ``(N, N+1)`` weight matrix at the treated period. For time-
        invariant constraint sets (as in MUSC) this is the same ``M``
        used for prediction.
    y_t : np.ndarray
        Outcomes at the treated period, length ``N``.

    Returns
    -------
    float
        Unbiased estimate of ``Var[τ̂]``. May be negative in finite
        samples due to Monte Carlo noise; callers should fall back to
        ``nan`` for the standard error in that case.
    """
    M_t = np.asarray(M_t, dtype=float)
    y_t = np.asarray(y_t, dtype=float)
    N = y_t.shape[0]
    if M_t.shape != (N, N + 1):
        raise ValueError(
            f"M_t must be (N, N+1) = ({N}, {N + 1}); got {M_t.shape}."
        )
    if N < 4:
        raise ValueError("unbiased_variance requires N >= 4.")

    alpha = M_t[:, 0]
    W = M_t[:, 1:]
    var_terms = np.zeros(N)

    for i in range(N):
        idx = np.r_[0:i, i + 1 : N]                             # noqa: E203
        Wi = W[np.ix_(idx, idx)]                                # (N-1, N-1)
        yi = y_t[idx]
        a_i = alpha[idx]
        diffs = yi[:, None] - yi[None, :]                       # Y_k - Y_j

        # Term 1: (1/(N-3)) Σ_k (Σ_j M_{k,j} (Y_k - Y_j))^2
        inner_sum = (Wi * diffs).sum(axis=1)                    # (N-1,)
        t1 = (inner_sum ** 2).sum() / (N - 3)

        # Term 2: -(1/((N-2)(N-3))) Σ_k Σ_j M_{k,j}^2 (Y_k - Y_j)^2
        t2 = -(Wi ** 2 * diffs ** 2).sum() / ((N - 2) * (N - 3))

        # Term 3: (2/(N-2)) Σ_k α_k * (Σ_j M_{k,j} (Y_j - Y_k))
        t3 = (2.0 / (N - 2)) * (a_i * -(Wi * diffs).sum(axis=1)).sum()

        # Term 4: (1/N) Σ_k α_k^2
        t4 = (a_i ** 2).sum() / N

        var_terms[i] = t1 + t2 + t3 + t4

    return float(var_terms.mean())


def normal_ci_from_variance(
    att: float, variance: float, alpha: float = 0.05,
) -> Tuple[float, float]:
    """Normal-approximation ``(1 - alpha)`` CI for the ATT.

    Falls back to ``(nan, nan)`` when ``variance < 0`` (which Prop 1
    permits in finite samples).
    """
    if not np.isfinite(variance) or variance < 0:
        return (float("nan"), float("nan"))
    from scipy.stats import norm
    z = norm.ppf(1.0 - alpha / 2.0)
    se = float(np.sqrt(variance))
    return (float(att - z * se), float(att + z * se))


# ---------------------------------------------------------------------------
# Section 3.5: randomization-based CI by placebo permutation
# ---------------------------------------------------------------------------

def randomization_ci(
    Y: np.ndarray,
    *,
    treated_idx: int,
    T0: int,
    column_balance: bool,
    att_observed: float,
    alpha: float = 0.05,
    solver: Optional[str] = None,
) -> Tuple[Tuple[float, float], np.ndarray]:
    """Exact randomization CI for the treated unit's ATT.

    Implements Bottmer et al. (2024) Section 3.5: for each non-treated
    unit ``j``, refit the estimator pretending ``j`` is treated
    (leave-one-out), giving a placebo ATT ``τ̂_j``. Under the random-
    unit-assignment design these are draws from the null distribution.
    The resulting ``(1 − alpha)`` CI inverts a two-sided permutation
    test on the order statistics of the placebo ATTs centred on the
    observed ATT.

    Parameters
    ----------
    Y : np.ndarray
        Full ``(T, N)`` outcome panel in time-major layout (rows =
        periods).
    treated_idx, T0 : int
        Index of the treated unit and the pre-period length.
    column_balance : bool
        Whether to use the MUSC unbiasedness restriction during
        refits. Should match the variant of interest.
    att_observed : float
        The realised ATT of the treated unit.
    alpha : float, default 0.05
        Two-sided significance level.
    solver : str, optional
        cvxpy solver name; forwarded to the QP.

    Returns
    -------
    (ci, placebo_atts) : ((float, float), np.ndarray)
        ``ci`` is the randomization-based ``(1 − alpha)`` interval.
        ``placebo_atts`` is the length-``(N − 1)`` array of leave-one-
        out placebo ATTs, sorted ascending. ``ci`` is ``(nan, nan)``
        when too few placebos solved successfully.
    """
    Y = np.asarray(Y, dtype=float)
    T, N = Y.shape
    if T0 < 1 or T0 >= T:
        return ((float("nan"), float("nan")), np.array([]))

    Y_pre_T_pre_by_N = Y[:T0, :]                                # (T_pre, N)
    placebo_atts = []
    for j in range(N):
        if j == treated_idx:
            continue
        try:
            M_j, _ = solve_musc_qp(
                Y_pre_T_pre_by_N,
                column_balance=column_balance,
                solver=solver,
                verbose=False,
            )
        except Exception:                                       # noqa: BLE001
            continue
        _, _, att_j, _ = att_for_unit(M_j, Y, j, T0)
        if np.isfinite(att_j):
            placebo_atts.append(att_j)

    placebo_atts = np.sort(np.asarray(placebo_atts, dtype=float))
    if placebo_atts.size < 4:
        return ((float("nan"), float("nan")), placebo_atts)

    # CI by inversion: the (1 - alpha)-quantile interval on the
    # centred placebo distribution. We pick the order statistics
    # closest to the alpha/2 and 1 - alpha/2 fractional indices.
    lo_idx = int(np.floor((alpha / 2.0) * placebo_atts.size))
    hi_idx = int(np.ceil((1.0 - alpha / 2.0) * placebo_atts.size)) - 1
    lo_idx = max(lo_idx, 0)
    hi_idx = min(hi_idx, placebo_atts.size - 1)
    # The interval flips: observed ATT minus the upper quantile gives
    # the lower bound, and vice versa (Section 3.5 derivation).
    lo = float(att_observed - placebo_atts[hi_idx])
    hi = float(att_observed - placebo_atts[lo_idx])
    if lo > hi:                                                 # numerical guard
        lo, hi = hi, lo
    return ((lo, hi), placebo_atts)
