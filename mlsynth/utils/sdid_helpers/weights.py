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

import numbers
import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple

from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError
from mlsynth.utils.bilevel.active_set import solve_simplex_qp
from mlsynth.utils.bilevel.minnorm import ridged_gram_reduction_is_safe


def _solve_intercept_simplex(
    design: np.ndarray, target: np.ndarray, ridge: float = 0.0,
    warm_start: Optional[np.ndarray] = None,
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
    warm_start : np.ndarray, shape (J,), optional
        Feasible weights to seed the active set with, e.g. the solution of a
        neighbouring problem. A hint only: ``solve_simplex_qp`` ignores one that
        is infeasible or the wrong length, so it cannot change the optimum.

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

    weights = solve_simplex_qp(design_c, target_c, warm_start=warm_start)
    intercept = target_mean - float(col_mean @ weights)
    return intercept, weights


def solve_intercept_simplex_many(
    problems: Sequence[Tuple[np.ndarray, np.ndarray, float]]
) -> List[Tuple[float, np.ndarray]]:
    """:func:`_solve_intercept_simplex` for a family of problems at once.

    Placebo inference refits the same program once per draw -- 500 times by
    default -- and the draws differ only in which donors are in the design, what
    the target is, and how large the ridge is. None of that needs a fresh
    factorisation. Centring is per column, so it survives subsetting; the ridge
    augmentation carries no target rows, so with the weights summing to one it
    enters the Gram as ``+ ridge I``; and the whole family's Grams are therefore
    a broadcast off quantities formed once. The batched active set then
    certifies a shape-group in a handful of linear solves.

    Parameters
    ----------
    problems : sequence of (design, target, ridge)
        One entry per solve, as :func:`_solve_intercept_simplex` takes them.
        Entries may differ in shape; they are grouped before solving.

    Returns
    -------
    list of (intercept, weights)
        In the order given, identical to solving them one at a time.

    Notes
    -----
    A group is batched only where
    :func:`~mlsynth.utils.bilevel.minnorm.ridged_gram_reduction_is_safe` passes
    on its centred design, and solved one at a time otherwise. Forming the Gram
    squares the design's condition number, and on a rank-deficient design the
    optimum is a face whose points the two solvers pick differently -- the same
    fit, other weights. The guard is on the design, not on the caller.

    That guard was the fit's largest single cost before it was asked this way:
    1000 problems per fit at ``B=500``, each answered with a full singular
    spectrum, all 1000 answering yes. The unit-weight program carries a ridge,
    and a ridge bounds the augmented Gram's smallest eigenvalue from below for
    free, so its half of those spectra is now never computed. The time-weight
    program carries none and still pays.

    Which of SDID's two programs clears that guard depends on the panel's shape.
    The unit-weight design is ``T0`` by ``N0`` and carries the ridge, so it
    batches. The time-weight design is ``N0`` by ``T0``, so it batches only when
    ``N0 > T0``: Prop 99 (38 donors, 19 pre-years) does, and a daily geo panel
    (40 markets, 75 pre-days) does not. Where it does not, every draw takes the
    fallback -- and that is also the pivot-heavy program, carrying ``T0``
    variables against the unit program's ``N0``.

    The fallback therefore chains its warm start: consecutive placebo draws
    differ only in which columns were reassigned to treatment, so the previous
    draw's solution seeds the next one's active set. The chain is keyed by
    centred-design shape, and ``solve_simplex_qp`` ignores an infeasible or
    wrongly sized seed, so it can only change how many pivots the solve takes.
    On a 40-market daily panel this cuts ``vce="placebo"`` at ``B=500`` by about
    6x with the ATT and its standard error unchanged to full precision.
    """
    from ..bilevel.minnorm import simplex_gram, solve_simplex_minnorm_batch

    out: List[Optional[Tuple[float, np.ndarray]]] = [None] * len(problems)
    groups: Dict[Tuple[int, int], List[int]] = {}
    prepared: List[Optional[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]]] = []
    # Last fallback solution per centred-design shape, to seed the next one.
    # Keyed by shape so a warm start is never offered to a differently sized
    # program; consecutive placebo draws share a shape, which is what the chain
    # exploits.
    last_fallback: Dict[Tuple[int, int], np.ndarray] = {}

    for i, (design, target, ridge) in enumerate(problems):
        design = np.asarray(design, dtype=float)
        target = np.asarray(target, dtype=float).ravel()
        ridge = float(ridge)
        col_mean = design.mean(axis=0)
        design_c = design - col_mean[None, :]
        J = design_c.shape[1]
        # The safety test is on the design the solve actually sees, ridge block
        # included: a ridge large enough to condition the problem is what makes
        # an otherwise rank-deficient design safe, and one too small to do that
        # is precisely the case the guard exists to catch. Asked through
        # ``ridged_gram_reduction_is_safe``, which describes that block instead
        # of building and factorising it wherever the ridge alone settles the
        # question -- the same decision, and for the unit-weight family no
        # spectrum at all.
        if J == 1 or not ridged_gram_reduction_is_safe(design_c, ridge):
            shape = design_c.shape
            solved = _solve_intercept_simplex(
                design, target, ridge, warm_start=last_fallback.get(shape)
            )
            last_fallback[shape] = solved[1]
            out[i] = solved
            prepared.append(None)
            continue
        prepared.append((design_c, target - float(target.mean()), ridge,
                         col_mean, float(target.mean())))
        groups.setdefault(design_c.shape, []).append(i)

    for shape, idx in groups.items():
        J = shape[1]
        G = np.stack([simplex_gram(prepared[i][0], prepared[i][1]) for i in idx])
        G += np.array([prepared[i][2] for i in idx])[:, None, None] * np.eye(J)[None]
        W = solve_simplex_minnorm_batch(G)
        for slot, i in enumerate(idx):
            w = W[slot]
            out[i] = (prepared[i][4] - float(prepared[i][3] @ w), w)

    return [o for o in out]  # type: ignore[misc]


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
    # numbers.Real, not (float, int): np.float64 subclasses Python float but
    # np.float32 and np.int64 do not, so the narrow check rejected perfectly
    # good values whenever the panel's dtype happened to be 32-bit -- which is
    # what pandas gives you for a Stata .dta. See issue #320.
    if (not isinstance(regularization_parameter_zeta, numbers.Real)
            or isinstance(regularization_parameter_zeta, bool)
            or not np.isfinite(float(regularization_parameter_zeta))
            or regularization_parameter_zeta < 0):
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


def _standardise_rows(A: np.ndarray, b: np.ndarray, return_scale: bool = False):
    """Scale each predictor row to unit standard deviation across donors.

    Abadie's ``V`` weights predictors that are on arbitrary and unrelated
    scales -- a GDP level and an employment share, say -- so without a common
    scale the search starts somewhere meaningless and the ridge penalises the
    rows unevenly. Rows with no cross-donor variation carry no information for
    matching and are given scale 1 so they neither blow up nor vanish.
    """
    scale = A.std(axis=1, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    if return_scale:
        return A / scale[:, None], b / scale, scale
    return A / scale[:, None], b / scale


def match_unit_weights(
    donor_outcomes_pre_treatment: np.ndarray,
    treated_outcome_pre_treatment: np.ndarray,
    donor_covariates: np.ndarray,
    treated_covariates: np.ndarray,
    regularization_parameter_zeta: float,
    pre_periods=None,
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
    # Demeaning uses the full pre-period regardless of how many rows then enter
    # the matching, because it is what makes this the demeaned-SC program; the
    # scheme selects which rows are *matched on*, not what they are centred by.
    Y0c = Y0 - Y0.mean(axis=0, keepdims=True)
    y1c = y1 - y1.mean()

    # How many pre-treatment outcome rows enter the matching. With all of them,
    # Kaul et al. (2022) show the nested V search zeroes the covariate rows and
    # the covariates cannot matter -- so this choice decides whether the
    # covariates do anything at all. See SDIDConfig.match_pre_periods.
    scheme = "last" if pre_periods is None else pre_periods
    if scheme == "all":
        k = T0
    elif scheme == "half":
        k = max(1, T0 // 2)
    elif scheme == "last":
        k = 1
    else:
        k = int(scheme)
        if k < 1:
            raise MlsynthConfigError(
                f"match_pre_periods must be at least 1; got {k}.")
        if k > T0:
            raise MlsynthConfigError(
                f"match_pre_periods={k} exceeds the {T0} pre-treatment "
                "period(s) available.")
    Y0m, y1m = Y0c[T0 - k:], y1c[T0 - k:]

    A = np.vstack([Y0m, Z0])
    b = np.concatenate([y1m, z1])
    A, b, row_scale = _standardise_rows(A, b, return_scale=True)
    n_rows = A.shape[0]

    # No ridge here, and that is the reference's choice rather than an
    # oversight. de Brabander et al. take omega from an UNPENALISED
    # Synth::synth and pass zeta.omega = 0 to synthdid; they report the
    # penalised variant separately (their "Including penalty" results).
    #
    # It is also the only coherent option once the rows are standardised.
    # synthdid's zeta is calibrated on the volatility of first-differenced raw
    # outcomes -- 224.47 on the #309 fixture -- while the standardised matching
    # rows have unit variance by construction and the cross-donor scale they
    # were divided by is 4.2. The two live on different scales, so T0 * zeta^2
    # is not a penalty on this design at any rescaling: carried over unchanged
    # it is 755,818 against an O(1) design and pins w to the uniform simplex
    # point; rescaled by the row variance it is still 42,673 and moves w by
    # 3e-5 no matter what V does. Either way the covariates cannot bind, which
    # is exactly the symptom that sent us here.
    #
    # regularization_parameter_zeta is accepted so the signature matches
    # unit_weights and a caller can be explicit, but it is deliberately unused.
    def solve_inner(v: np.ndarray) -> np.ndarray:
        sw = np.sqrt(np.maximum(v, 0.0))[:, None]
        # No ridge block: see the note above -- this program is unpenalised, so
        # there is nothing to fold in. Keeping a dead `if ridge > 0` branch here
        # would suggest the penalty is merely switched off rather than absent by
        # design.
        return solve_simplex_qp(A * sw, b * sw.ravel())

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
