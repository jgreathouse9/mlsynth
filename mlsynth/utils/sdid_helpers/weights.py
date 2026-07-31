"""Unit-weight, time-weight, and regularization solvers for SDID.

Both SDID weight programs are simplex-constrained least squares with a free
intercept (and, for the unit weights, an L2 ridge):

* unit weights ``omega`` minimise
  ``||a + Y0_pre @ omega - y_treated_pre||^2 + T0 * zeta^2 * ||omega||^2``
  subject to ``sum(omega) = 1, omega >= 0`` (Arkhangelsky et al. 2021 eq. 4 /
  Clarke et al. 2024 eq. 4);
* time weights ``lambda`` minimise
  ``||a + lambda @ Y0_pre - mean_post||^2`` subject to ``sum(lambda) = 1,
  lambda >= 0`` (eq. 6).

Rather than canonicalise these through cvxpy on every call -- expensive in the
placebo / jackknife loops, which re-solve them hundreds of times -- we solve
them natively with the library's active-set simplex QP
(:func:`mlsynth.utils.bilevel.active_set.solve_simplex_qp`). Two standard
reductions make the active-set primitive applicable without changing the
optimum:

1. the free intercept ``a`` is *profiled out* by centering the design and
   target over the observation axis (for fixed weights, the optimal intercept
   is the mean residual, which is exactly what centering enforces); the
   intercept is recovered afterwards as ``a* = mean(target) - colmean(X) @ w``;
2. the unit-weight ridge ``lambda_r = T0 * zeta^2`` is folded in by stacking a
   ``sqrt(lambda_r) * I`` block beneath the centered design with a zero target,
   so ``||X_aug w - b_aug||^2 == ||X_c w - b_c||^2 + lambda_r ||w||^2``.

The unit program is strictly convex by its positive ridge, so its optimum is
unique. The time program is strictly convex only when the donors outnumber the
pre-periods; with ``N0 <= T0`` the centered design is rank deficient and the
argmin need not be. ``synthdid`` guards this with an infinitesimal tie-breaker
(``zeta.lambda = 1e-6 * noise.level``); mlsynth does not, and on the panels
tested it makes no difference -- adding one moves the ``N0 == T0 == 15`` fixture
in ``tests/test_sdid_covariates.py`` by 5e-9 -- because the simplex constraint
pins the solution on its own. Worth knowing rather than assuming, since the
earlier version of this note claimed the precondition always held.

In both cases the active-set solution coincides with CLARABEL's to solver
tolerance -- the Prop 99 ATT is preserved bit-for-bit within the pinned
benchmark tolerance. Parity is asserted in
``tests/test_sdid_weights_native.py``.

These are exact solves. ``synthdid`` instead runs projected gradient to a
stopping rule (``min.decrease = 1e-5 * noise.level``), so the two agree only to
that rule's residual: usually ~1e-7, but 5e-3 on a ridge-dominated panel where
the objective near the optimum is nearly flat. See ``TestSynthdidsEarlyStop`` in
``tests/test_sdid_covariates.py`` for a worked instance.
"""

import numpy as np
from typing import Optional, Tuple

from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError
from mlsynth.utils.bilevel.active_set import solve_simplex_qp


def _solve_intercept_simplex(
    design: np.ndarray, target: np.ndarray, ridge: float = 0.0
) -> Tuple[float, np.ndarray]:
    """Simplex least squares with a free intercept and optional L2 ridge.

    Solves ``min_{a, w} ||a + design @ w - target||^2 + ridge * ||w||^2`` over
    ``sum(w) = 1, w >= 0`` and returns ``(a, w)``. The intercept is profiled out
    by centering over the observation (row) axis; the ridge is folded in by
    augmenting the centered design with a ``sqrt(ridge) * I`` block.

    Parameters
    ----------
    design : np.ndarray, shape (m, J)
        Design matrix (observations x weights).
    target : np.ndarray, shape (m,)
        Target vector.
    ridge : float, optional
        Non-negative L2 penalty coefficient on ``w`` (default ``0``).

    Returns
    -------
    (intercept, weights) : tuple of (float, np.ndarray)
        The optimal free intercept and the ``(J,)`` simplex weights.
    """
    col_mean = design.mean(axis=0)
    target_mean = float(target.mean())
    design_c = design - col_mean[None, :]
    target_c = target - target_mean

    if ridge > 0.0:
        J = design_c.shape[1]
        design_c = np.vstack([design_c, np.sqrt(ridge) * np.eye(J)])
        target_c = np.concatenate([target_c, np.zeros(J)])

    weights = solve_simplex_qp(design_c, target_c)
    intercept = target_mean - float(col_mean @ weights)
    return intercept, weights


def fit_time_weights(
    donor_outcomes_pre_treatment: np.ndarray, mean_donor_outcomes_post_treatment: np.ndarray
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """
    Fit time weights for SDID.

    Parameters
    ----------
    donor_outcomes_pre_treatment : np.ndarray
        Donor outcomes in pre-treatment period, shape (T0, N_donors).
    mean_donor_outcomes_post_treatment : np.ndarray
        Mean outcome of each donor unit in post-treatment period, shape (N_donors,).

    Returns
    -------
    Tuple[Optional[float], Optional[np.ndarray]]
        - intercept : Optional[float]
            The estimated intercept term (beta_0 in some notations).
            Returns `None` if the optimization fails or does not converge.
        - time_weights : Optional[np.ndarray]
            The estimated time weights (lambda_t in some notations).
            Shape (num_pre_treatment_periods,). These weights sum to 1 and are non-negative.
            Returns `None` if the optimization fails or does not converge.

    Notes
    -----
    This function solves an optimization problem to find time weights and an
    intercept that best reconstruct the average post-treatment donor outcomes
    using a weighted average of pre-treatment donor outcomes.
    The objective is to minimize the sum of squared differences between
    `mean_donor_outcomes_post_treatment` and
    `intercept + time_weights @ donor_outcomes_pre_treatment`, subject to `sum(time_weights) = 1`
    and `time_weights >= 0`.

    Examples
    --------
    >>> T0_ex, N_donors_ex = 5, 3
    >>> Y0_pre_donors_ex = np.random.rand(T0_ex, N_donors_ex)
    >>> Y0_post_donors_mean_ex = np.random.rand(N_donors_ex)
    >>> intercept_val, time_w_val = fit_time_weights(Y0_pre_donors_ex, Y0_post_donors_mean_ex)
    >>> if time_w_val is not None:
    ...     print(f"Time weights shape: {time_w_val.shape}")
    ...     print(f"Sum of time weights: {np.sum(time_w_val):.2f}")
    Time weights shape: (5,)
    Sum of time weights: 1.00
    """
    # Input Validation
    # Ensure donor_outcomes_pre_treatment is a 2D NumPy array.
    if not isinstance(donor_outcomes_pre_treatment, np.ndarray):
        raise MlsynthDataError("donor_outcomes_pre_treatment must be a NumPy array.")
    if donor_outcomes_pre_treatment.ndim != 2:
        raise MlsynthDataError("donor_outcomes_pre_treatment must be a 2D array (T0, N_donors).")
    # Ensure mean_donor_outcomes_post_treatment is a 1D NumPy array.
    if not isinstance(mean_donor_outcomes_post_treatment, np.ndarray):
        raise MlsynthDataError("mean_donor_outcomes_post_treatment must be a NumPy array.")
    if mean_donor_outcomes_post_treatment.ndim != 1:
        raise MlsynthDataError("mean_donor_outcomes_post_treatment must be a 1D array (N_donors,).")
    
    # Get dimensions: number of pre-treatment periods and number of donors.
    num_pre_treatment_periods, num_donors_pre = donor_outcomes_pre_treatment.shape
    num_donors_post = mean_donor_outcomes_post_treatment.shape[0]

    # Validate dimensions.
    if num_pre_treatment_periods == 0: # Must have pre-treatment periods to fit weights.
        raise MlsynthDataError("donor_outcomes_pre_treatment cannot have zero pre-treatment periods (num_pre_treatment_periods must be > 0).")
    if num_donors_pre == 0: # Must have donors.
        # This case might be implicitly handled by CVXPY if num_donors_post is also 0,
        # but explicit check is better. If num_donors_post > 0, it's a mismatch.
        raise MlsynthDataError("donor_outcomes_pre_treatment cannot have zero donors if mean_donor_outcomes_post_treatment has donors.")

    # Number of donors must be consistent between pre-treatment and post-treatment data.
    if num_donors_pre != num_donors_post:
        raise MlsynthDataError(
            f"Shape mismatch: donor_outcomes_pre_treatment has {num_donors_pre} donors, "
            f"but mean_donor_outcomes_post_treatment has {num_donors_post} donors."
        )

    # Native simplex least squares with a free intercept. The time weights
    # reconstruct the mean post-treatment donor outcomes (one per donor) as a
    # weighted average over pre-treatment periods, so the *observations* are the
    # donors and the *weights* are the time periods: the design is
    # ``donor_outcomes_pre_treatment.T`` (N_donors x T0) and the target is
    # ``mean_donor_outcomes_post_treatment`` (N_donors,). No ridge (eq. 6 uses
    # only an infinitesimal tie-breaker, which the overdetermined Prop 99 system
    # does not need).
    intercept, time_weights = _solve_intercept_simplex(
        donor_outcomes_pre_treatment.T, mean_donor_outcomes_post_treatment
    )
    return float(intercept), time_weights


def compute_regularization(
    donor_outcomes_pre_treatment: np.ndarray,
    num_post_treatment_periods: int,
    num_treated_units: int = 1,
) -> float:
    """
    Compute regularization parameter zeta for unit weights.

    Parameters
    ----------
    donor_outcomes_pre_treatment : np.ndarray
        Donor outcomes in pre-treatment period, shape (T0, N_donors).
    num_post_treatment_periods : int
        Number of post-treatment periods (``T_post``).
    num_treated_units : int, optional
        Number of treated units in the cohort (``N_tr``), by default ``1``.
        Arkhangelsky et al. (2021) fold the treated count into the unit-weight
        ridge; the ``synthdid`` R package uses
        ``eta.omega = ((N - N0) * (T - T0))^(1/4) = (N_tr * T_post)^(1/4)``.
        A single treated unit (the default) leaves ``zeta`` at the
        ``(T_post)^(1/4)`` form, so single-treated designs are unchanged.

    Returns
    -------
    float
        The calculated regularization parameter zeta. If `donor_outcomes_pre_treatment` has
        fewer than 2 time periods, a fallback value (currently 1.0, though this
        might indicate insufficient data for robust estimation) is used for
        `std_dev_of_first_differenced_donor_outcomes`, which then influences zeta.

    Notes
    -----
    The regularization parameter `zeta` is calculated as:
    `zeta = ((num_treated_units * num_post_treatment_periods) ** 0.25) * std_dev_of_first_differenced_donor_outcomes`
    where `std_dev_of_first_differenced_donor_outcomes` is the standard deviation of the first-differenced
    outcomes of donor units in the pre-treatment period. This matches the
    ``synthdid`` unit-weight tuning parameter ``zeta.omega``.

    Examples
    --------
    >>> T0_ex, N_donors_ex = 10, 5
    >>> Y0_pre_donors_ex = np.random.rand(T0_ex, N_donors_ex) * 100
    >>> T_post_ex = 5
    >>> zeta = compute_regularization(Y0_pre_donors_ex, T_post_ex)
    >>> print(f"Zeta: {zeta:.2f}")
    Zeta: ...

    >>> # Example with insufficient pre-treatment periods for diff
    >>> Y0_short_pre_donors_ex = np.random.rand(1, N_donors_ex)
    >>> zeta_short = compute_regularization(Y0_short_pre_donors_ex, T_post_ex)
    >>> # Based on fallback std_dev_of_first_differenced_donor_outcomes = 1.0
    >>> # Expected: (5**0.25) * 1.0 = 1.495...
    >>> print(f"Zeta for short pre-period: {zeta_short:.2f}")
    Zeta for short pre-period: 1.50
    """
    # Input Validation
    if not isinstance(donor_outcomes_pre_treatment, np.ndarray):
        raise MlsynthDataError("donor_outcomes_pre_treatment must be a NumPy array.")
    if donor_outcomes_pre_treatment.ndim != 2:
        # Allow 0 donors for flexibility, std calculation will handle it or fallback.
        # However, if shape[0] (periods) is 0, diff will fail.
        raise MlsynthDataError("donor_outcomes_pre_treatment must be a 2D array (T0, N_donors).")
    if not isinstance(num_post_treatment_periods, int) or num_post_treatment_periods < 0:
        raise MlsynthConfigError("num_post_treatment_periods must be a non-negative integer.")
    if not isinstance(num_treated_units, int) or num_treated_units < 1:
        raise MlsynthConfigError("num_treated_units must be a positive integer.")

    # Calculate the standard deviation of the first-differenced donor outcomes in the pre-treatment period.
    # This term captures the volatility of donor outcomes.
    if donor_outcomes_pre_treatment.shape[0] < 2 : # Need at least 2 pre-treatment periods to calculate differences.
        # Fallback value if insufficient pre-treatment periods for differencing.
        # This implies high uncertainty or reliance on the num_post_treatment_periods term.
        std_dev_of_first_differenced_donor_outcomes = 1.0 
    elif donor_outcomes_pre_treatment.shape[1] == 0: # No donors.
        std_dev_of_first_differenced_donor_outcomes = 1.0 # Fallback if no donors to calculate differences from.
    else:
        # Calculate first differences of donor outcomes along the time axis (axis=0).
        diffs = np.diff(donor_outcomes_pre_treatment, axis=0)
        if diffs.size == 0: # pragma: no cover - unreachable: T0<2 and N_donors==0 are both caught above.
             std_dev_of_first_differenced_donor_outcomes = 1.0 # Fallback if differences result in an empty array.
        else:
             # Calculate standard deviation of these differences. ddof=1 for sample standard deviation.
             std_dev_of_first_differenced_donor_outcomes = np.std(diffs.flatten(), ddof=1)
             if np.isnan(std_dev_of_first_differenced_donor_outcomes): # If all diffs were NaN, leading to NaN std.
                 std_dev_of_first_differenced_donor_outcomes = 1.0 # Fallback if std dev is NaN.

    # Calculate zeta: ((N_tr * T_post) ^ 0.25) * std_dev_of_first_differenced_donor_outcomes.
    # The (N_tr * T_post) ^ 0.25 term is synthdid's eta.omega: it scales the
    # ridge by both the post-treatment horizon and the number of treated units,
    # so a block with several treated units is regularized more strongly than a
    # single-treated design on the same panel.
    regularization_parameter_zeta: float = (
        (num_treated_units * num_post_treatment_periods) ** 0.25
    ) * std_dev_of_first_differenced_donor_outcomes
    return regularization_parameter_zeta


def unit_weights(
    donor_outcomes_pre_treatment: np.ndarray,
    mean_treated_outcome_pre_treatment: np.ndarray,
    regularization_parameter_zeta: float
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """
    Fit unit (donor) weights for SDID.

    Parameters
    ----------
    donor_outcomes_pre_treatment : np.ndarray
        Donor outcomes in pre-treatment period, shape (T0, N_donors).
    mean_treated_outcome_pre_treatment : np.ndarray
        Mean outcome of treated units in pre-treatment period, shape (T0,).
    regularization_parameter_zeta : float
        Regularization parameter.

    Returns
    -------
    Tuple[Optional[float], Optional[np.ndarray]]
        - intercept : Optional[float]
            The estimated intercept term (beta_0 in some notations).
            Returns `None` if the optimization fails or does not converge.
        - unit_weights : Optional[np.ndarray]
            The estimated donor weights (omega_j in some notations).
            Shape (N_donors,). These weights sum to 1 and are non-negative.
            Returns `None` if the optimization fails or does not converge.

    Notes
    -----
    This function solves an optimization problem to find donor weights and an
    intercept that best reconstruct the pre-treatment trajectory of the
    (mean) treated unit using a weighted average of donor unit outcomes.
    The objective is to minimize the sum of squared differences between
    `mean_treated_outcome_pre_treatment` and `intercept + donor_outcomes_pre_treatment @ unit_weights`,
    plus an L2 penalty on the `unit_weights` scaled by `regularization_parameter_zeta`.
    Constraints are `sum(unit_weights) = 1` and `unit_weights >= 0`.

    Examples
    --------
    >>> T0_ex, N_donors_ex = 10, 5
    >>> Y0_pre_donors_ex = np.random.rand(T0_ex, N_donors_ex)
    >>> y_pre_mean_treated_ex = np.random.rand(T0_ex)
    >>> zeta_ex = 0.1
    >>> intercept_val, unit_w_val = unit_weights(
    ...     Y0_pre_donors_ex, y_pre_mean_treated_ex, zeta_ex
    ... )
    >>> if unit_w_val is not None:
    ...     print(f"Unit weights shape: {unit_w_val.shape}")
    ...     print(f"Sum of unit weights: {np.sum(unit_w_val):.2f}")
    Unit weights shape: (5,)
    Sum of unit weights: 1.00
    """
    # Input Validation
    if not isinstance(donor_outcomes_pre_treatment, np.ndarray):
        raise MlsynthDataError("donor_outcomes_pre_treatment must be a NumPy array.")
    if donor_outcomes_pre_treatment.ndim != 2: # Must be 2D: (Time, Donors)
        raise MlsynthDataError("donor_outcomes_pre_treatment must be a 2D array (T0, N_donors).")
    if not isinstance(mean_treated_outcome_pre_treatment, np.ndarray):
        raise MlsynthDataError("mean_treated_outcome_pre_treatment must be a NumPy array.")
    if mean_treated_outcome_pre_treatment.ndim != 1: # Must be 1D: (Time,)
        raise MlsynthDataError("mean_treated_outcome_pre_treatment must be a 1D array (T0,).")
    if not isinstance(regularization_parameter_zeta, (float, int)) or regularization_parameter_zeta < 0:
        raise MlsynthConfigError("regularization_parameter_zeta must be a non-negative float or int.")

    # Get dimensions: number of pre-treatment periods and number of donors.
    num_pre_treatment_periods, num_donors = donor_outcomes_pre_treatment.shape
    
    # Validate dimensions.
    if num_pre_treatment_periods == 0: # Must have pre-treatment periods.
        raise MlsynthDataError("donor_outcomes_pre_treatment cannot have zero pre-treatment periods (num_pre_treatment_periods must be > 0).")
    if num_donors == 0: # Must have donors.
        raise MlsynthDataError("donor_outcomes_pre_treatment cannot have zero donors (num_donors must be > 0).")
    # Number of pre-treatment periods must match between donor outcomes and mean treated outcome.
    if mean_treated_outcome_pre_treatment.shape[0] != num_pre_treatment_periods:
        raise MlsynthDataError(
            f"Shape mismatch: donor_outcomes_pre_treatment has {num_pre_treatment_periods} pre-periods, "
            f"but mean_treated_outcome_pre_treatment has {mean_treated_outcome_pre_treatment.shape[0]}."
        )

    # Native simplex least squares with a free intercept and the SDID ridge.
    # The unit weights reconstruct the mean pre-treatment treated trajectory
    # from the donors, so the *observations* are the pre-periods and the
    # *weights* are the donors: the design is ``donor_outcomes_pre_treatment``
    # (T0 x N_donors) and the target is ``mean_treated_outcome_pre_treatment``
    # (T0,). The ridge coefficient T0 * zeta^2 (eq. 4) makes the program
    # strictly convex.
    penalty_coefficient = num_pre_treatment_periods * (float(regularization_parameter_zeta) ** 2)
    intercept, omega = _solve_intercept_simplex(
        donor_outcomes_pre_treatment,
        mean_treated_outcome_pre_treatment,
        ridge=penalty_coefficient,
    )
    return float(intercept), omega


def _standardise_rows(A: np.ndarray, b: np.ndarray):
    """Scale each predictor row to unit standard deviation across donors.

    Abadie's ``V`` weights predictors that are on arbitrary and unrelated
    scales -- a GDP level and an employment share, say -- so without a common
    scale the search starts somewhere meaningless and the ridge penalises the
    rows unevenly. Rows with no cross-donor variation carry no information for
    matching and are given scale 1 so they neither blow up nor vanish.
    """
    scale = A.std(axis=1, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return A / scale[:, None], b / scale


def match_unit_weights(
    donor_outcomes_pre_treatment: np.ndarray,
    treated_outcome_pre_treatment: np.ndarray,
    donor_covariates: np.ndarray,
    treated_covariates: np.ndarray,
    regularization_parameter_zeta: float,
    max_iter: int = 200,
) -> Tuple[float, np.ndarray]:
    """Unit weights matching on covariates as well as pre-treatment outcomes.

    Implements de Brabander, Juodis & Miyazato Szini (2025) eqs. (11)-(12): the
    SDID/DSC unit-weight program with covariates stacked into the matching
    problem and a diagonal ``V`` over the stacked rows chosen by nested
    optimisation.

    Inner problem, for a given ``v``::

        min_w  (z1 - Z0 w)' diag(v) (z1 - Z0 w) + T0 zeta^2 ||w||^2
        s.t.   sum(w) = 1,  w >= 0

    where ``z1`` stacks the treated unit's *demeaned* pre-treatment outcomes on
    top of its covariate summaries, and ``Z0`` the same for the donors. Outer
    problem: choose ``v`` to minimise the pre-treatment fit on the raw outcomes,
    ``||y_pre - Y0_pre w(v)||^2``.

    Two details are load-bearing and both follow the paper.

    Only the outcome rows are demeaned. Abadie (2021b)'s recommendation, which
    the paper adopts: the covariates sit on their own scales and demeaning them
    alongside the outcome mixes those scales together. Demeaning the outcome
    rows is what makes this the *demeaned* SC program that SDID's unit weights
    are built on, so it cannot simply be dropped either.

    The intercept is not profiled out separately. Demeaning the outcome rows
    already absorbs it; centring the stacked design again -- which is what the
    outcome-only path does via ``_solve_intercept_simplex`` -- would also centre
    the covariate rows and change the estimand.

    Returns ``(intercept, weights)``, the intercept recovered afterwards as the
    level difference the demeaning removed, so the return shape matches
    :func:`unit_weights`.
    """
    Y0 = np.asarray(donor_outcomes_pre_treatment, dtype=float)
    y1 = np.asarray(treated_outcome_pre_treatment, dtype=float)
    Z0 = np.atleast_2d(np.asarray(donor_covariates, dtype=float))
    z1 = np.atleast_1d(np.asarray(treated_covariates, dtype=float))

    if Y0.ndim != 2:
        raise MlsynthDataError("donor_outcomes_pre_treatment must be 2D (T0, J).")
    T0, J = Y0.shape
    if y1.shape[0] != T0:
        raise MlsynthDataError(
            f"treated outcome has {y1.shape[0]} pre-periods against {T0} for "
            "the donors.")
    if Z0.shape[1] != J:
        raise MlsynthDataError(
            f"donor covariates have {Z0.shape[1]} columns against {J} donors.")
    if z1.shape[0] != Z0.shape[0]:
        raise MlsynthDataError(
            f"treated covariates have {z1.shape[0]} entries against "
            f"{Z0.shape[0]} covariate rows for the donors.")
    if not (np.all(np.isfinite(Y0)) and np.all(np.isfinite(y1))
            and np.all(np.isfinite(Z0)) and np.all(np.isfinite(z1))):
        raise MlsynthDataError(
            "covariate matching needs finite outcomes and covariates.")

    # Demean the OUTCOME rows only -- each series by its own pre-treatment mean.
    Y0c = Y0 - Y0.mean(axis=0, keepdims=True)
    y1c = y1 - y1.mean()

    A = np.vstack([Y0c, Z0])
    b = np.concatenate([y1c, z1])
    A, b = _standardise_rows(A, b)
    n_rows = A.shape[0]
    ridge = T0 * (float(regularization_parameter_zeta) ** 2)

    def solve_inner(v: np.ndarray) -> np.ndarray:
        sw = np.sqrt(np.maximum(v, 0.0))[:, None]
        design, target = A * sw, b * sw.ravel()
        if ridge > 0.0:
            design = np.vstack([design, np.sqrt(ridge) * np.eye(J)])
            target = np.concatenate([target, np.zeros(J)])
        return solve_simplex_qp(design, target)

    def outer_loss(w: np.ndarray) -> float:
        # Raw pre-treatment outcomes, level-matched -- the intercept is free in
        # this program, so the outer fit is judged on the demeaned residual.
        resid = y1c - Y0c @ w
        return float(resid @ resid)

    # Equal weights over the standardised rows is the natural starting point and
    # is already a sensible estimator; the search refines it.
    v = np.full(n_rows, 1.0 / n_rows)
    best_w = solve_inner(v)
    best = outer_loss(best_w)

    # Coordinate-wise multiplicative search on the simplex of row weights.
    # Deterministic, cheap, and monotone by construction -- a step is kept only
    # when it lowers the outer loss -- which matters because this runs inside
    # placebo and bootstrap loops where a stochastic optimiser would make the
    # reported estimate depend on the resampling seed.
    step = 2.0
    for _ in range(max_iter):
        improved = False
        for i in range(n_rows):
            for factor in (step, 1.0 / step):
                cand = v.copy()
                cand[i] *= factor
                cand /= cand.sum()
                w = solve_inner(cand)
                loss = outer_loss(w)
                if loss < best - 1e-14:
                    v, best, best_w, improved = cand, loss, w, True
                    break
        if not improved:
            step = np.sqrt(step)
            if step < 1.01:
                break

    intercept = float(np.mean(y1) - np.mean(Y0 @ best_w))
    return intercept, best_w
