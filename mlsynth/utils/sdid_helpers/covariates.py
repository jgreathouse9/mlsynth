"""Time-varying covariates for SDID: the Kranz (2022) two-step adjustment.

Synthetic DiD absorbs unit and time effects by construction, so it has no
natural slot for other controls. Kranz's ``xsynthdid`` adds them by adjusting the
outcome before the estimator ever runs:

1. fit ``y ~ X | unit + time`` on the rows with no treatment;
2. keep the covariate coefficients only;
3. subtract ``X @ beta`` from the outcome across the *whole* panel;
4. run ordinary SDID on the adjusted outcome.

Three properties of that recipe are load-bearing, and each is the opposite of a
plausible-looking alternative:

* The regression is fit on untreated rows and applied to *all* rows. Restricting
  the projection to the estimation rows would leave the treated observations
  unadjusted, which is exactly the ones the estimate depends on.
* Only the covariate coefficients are removed. The unit and time effects stay in
  the outcome, because SDID handles those itself -- subtracting them here would
  be a different estimator, not a cleaner one.
* ``add_mean`` defaults to ``False``, matching ``adjust.outcome.for.x``. Turning
  it on re-centres the adjusted outcome on the original mean; because SDID is
  invariant to a constant shift it cannot change the estimate, only the scale
  the counterfactual is reported on.

Reference: `skranz/xsynthdid <https://github.com/skranz/xsynthdid>`_,
``R/adjust_y.R``. Cross-validated in
``mlsynth/tests/test_sdid_covariates.py`` against a live run, at the seam (the
fitted coefficient and the adjusted outcome, element-wise) as well as the
endpoint.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError, MlsynthEstimationError

#: Convergence tolerance and iteration cap for the alternating-projections
#: within-transformation. A balanced panel converges in one pass; the cap exists
#: for the unbalanced case, where the projections converge geometrically.
_DEMEAN_TOL = 1e-12
_DEMEAN_MAX_ITER = 200


def _design_matrix(frame: pd.DataFrame, covariates: Sequence[str]) -> np.ndarray:
    """Numeric design matrix, expanding categoricals to dummies (no intercept).

    Mirrors ``model.matrix(~ x1 + x2)[, -1]``: a categorical covariate becomes
    ``k - 1`` indicator columns, so a factor control behaves the same way it
    does in the reference.
    """
    sub = frame.loc[:, list(covariates)]
    expanded = pd.get_dummies(sub, drop_first=True, dtype=float)
    return np.asarray(expanded.to_numpy(), dtype=float)


def _absorb(values: np.ndarray, unit_codes: np.ndarray, time_codes: np.ndarray,
            n_units: int, n_times: int) -> np.ndarray:
    """Two-way within transformation by alternating projections.

    Double-demeaning in one pass is exact only on a balanced panel. The rows
    this is applied to are the *untreated* ones -- every control unit for every
    period, plus the treated units before treatment -- which is unbalanced by
    construction whenever there is a treated unit. So the projections are
    iterated to convergence rather than applied once, which is what ``fixest``
    does and what makes the coefficient match it.
    """
    out = np.array(values, dtype=float, copy=True)
    for _ in range(_DEMEAN_MAX_ITER):
        before = out.copy()
        for codes, n in ((unit_codes, n_units), (time_codes, n_times)):
            counts = np.bincount(codes, minlength=n).astype(float)
            counts[counts == 0.0] = 1.0
            if out.ndim == 1:
                out -= (np.bincount(codes, weights=out, minlength=n) / counts)[codes]
            else:
                for j in range(out.shape[1]):
                    means = np.bincount(codes, weights=out[:, j], minlength=n) / counts
                    out[:, j] -= means[codes]
        if np.max(np.abs(out - before)) < _DEMEAN_TOL:
            break
    return out


def twfe_covariate_beta(
    frame: pd.DataFrame,
    outcome: str,
    covariates: Sequence[str],
    unit: str,
    time: str,
) -> np.ndarray:
    """Covariate coefficients from ``outcome ~ covariates | unit + time``.

    The fixed effects are absorbed rather than estimated, so this returns only
    the ``len(covariates)`` slopes -- which is all the adjustment uses.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows to fit on. Callers pass the untreated subset.
    outcome, unit, time : str
        Column names.
    covariates : sequence of str
        Covariate column names. Categorical columns are expanded to dummies.

    Returns
    -------
    numpy.ndarray
        Shape ``(k,)`` coefficients, in the order the design matrix expands to.

    Raises
    ------
    MlsynthDataError
        A named column is absent, or the rows carry non-finite values.
    MlsynthEstimationError
        There are too few rows, or the absorbed design is rank deficient.
    """
    missing = [c for c in list(covariates) + [outcome, unit, time]
               if c not in frame.columns]
    if missing:
        raise MlsynthDataError(
            f"column(s) {missing} not in the panel; available: "
            f"{list(frame.columns)}")
    if frame.empty:
        raise MlsynthEstimationError(
            "no rows to estimate the covariate effect on.")

    X = _design_matrix(frame, covariates)
    y = np.asarray(frame[outcome].to_numpy(), dtype=float)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise MlsynthDataError(
            "covariate adjustment needs a panel free of NaN/inf in the outcome "
            "and covariate columns.")
    if X.shape[0] <= X.shape[1]:
        raise MlsynthEstimationError(
            f"covariate regression has {X.shape[0]} row(s) for {X.shape[1]} "
            "covariate(s); not enough to identify the slopes.")

    u_codes, u_uniq = pd.factorize(frame[unit], sort=True)
    t_codes, t_uniq = pd.factorize(frame[time], sort=True)
    Xw = _absorb(X, u_codes, t_codes, len(u_uniq), len(t_uniq))
    yw = _absorb(y, u_codes, t_codes, len(u_uniq), len(t_uniq))

    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    if not np.all(np.isfinite(beta)):  # pragma: no cover - defensive
        raise MlsynthEstimationError(
            "the absorbed covariate design is rank deficient; drop collinear "
            "covariates or ones that vary only across units or only over time.")
    return np.asarray(beta, dtype=float)


def adjust_outcome_for_covariates(
    panel: pd.DataFrame,
    unit: str,
    time: str,
    outcome: str,
    treat: str,
    covariates: Sequence[str],
    rows: Optional[Any] = None,
    add_mean: bool = False,
) -> np.ndarray:
    """Kranz's adjusted outcome, ready to hand to SDID.

    Parameters
    ----------
    panel : pandas.DataFrame
        Balanced long panel.
    unit, time, outcome, treat : str
        Column names.
    covariates : sequence of str
        Time-varying controls.
    rows : optional
        Boolean mask, or the name of a boolean column, selecting the rows the
        covariate effect is estimated on. Defaults to every row with
        ``treat == 0`` -- which includes control units *during* the treatment
        period, matching ``adjust.outcome.for.x``.
    add_mean : bool, default False
        Add back the mean covariate effect, so the adjusted outcome keeps the
        original mean. Off by default, as in the reference. SDID is invariant to
        a constant shift, so this changes the reported level and not the effect.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_rows,)`` adjusted outcome, aligned to ``panel``'s row order.

    Raises
    ------
    MlsynthDataError
        A named column is absent, or the data carry non-finite values.
    MlsynthEstimationError
        No untreated rows to estimate on.
    """
    for name in (unit, time, outcome, treat):
        if name not in panel.columns:
            raise MlsynthDataError(
                f"column {name!r} not in the panel; available: "
                f"{list(panel.columns)}")
    covariates = list(covariates)
    missing = [c for c in covariates if c not in panel.columns]
    if missing:
        raise MlsynthDataError(
            f"covariate column(s) {missing} not in the panel; available: "
            f"{list(panel.columns)}")

    if rows is None:
        mask = np.asarray(panel[treat].to_numpy(), dtype=float) == 0.0
    elif isinstance(rows, str):
        if rows not in panel.columns:
            raise MlsynthDataError(f"rows column {rows!r} not in the panel.")
        mask = np.asarray(panel[rows].to_numpy()).astype(bool)
    else:
        mask = np.asarray(rows).astype(bool)
        if mask.shape != (len(panel),):
            raise MlsynthDataError(
                f"rows mask has shape {mask.shape}; expected ({len(panel)},).")

    if not mask.any():
        raise MlsynthEstimationError(
            "the covariate effect is estimated on untreated rows and none "
            "remain; every row is marked treated.")

    beta = twfe_covariate_beta(panel.loc[mask], outcome, covariates, unit, time)

    # Fit on the untreated rows, project over ALL of them. Building the design
    # from the full panel (not the masked subset) also keeps dummy columns
    # aligned when a categorical level appears only in the treated rows.
    X_full = _design_matrix(panel, covariates)
    if X_full.shape[1] != beta.shape[0]:  # pragma: no cover - defensive
        raise MlsynthEstimationError(
            f"covariate design has {X_full.shape[1]} column(s) on the full "
            f"panel but {beta.shape[0]} on the estimation rows; a categorical "
            "level is present in one and not the other.")
    x_effect = X_full @ beta

    y = np.asarray(panel[outcome].to_numpy(), dtype=float)
    adjusted = y - x_effect
    if add_mean:
        adjusted = adjusted + float(np.mean(x_effect))
    return adjusted


# ---------------------------------------------------------------------------
# The 'optimized' method: Arkhangelsky et al. (2021) footnote 4.
#
# Two covariate methods, and the difference is not solver detail. Kranz's
# 'projected' (above) estimates beta once, by OLS on rows with no treatment in
# force, and hands SDID a residualised outcome -- the weight programs never see
# a covariate. The 'optimized' method estimates beta *jointly* with the unit and
# time weights, minimising a single objective in (beta, omega, lambda), so beta
# and the weights determine each other.
#
# Stata's ``sdid`` takes ``optimized`` as its *default*, which is why a paper
# that simply writes ``covariates(x)`` needs this and not 'adjust'.
#
# Why this ports the reference solver instead of solving exactly
# --------------------------------------------------------------
# Everywhere else in this module mlsynth solves the weight programs exactly and
# accepts a ~0.005 disagreement with ``synthdid``'s Frank-Wolfe iteration (see
# ``weights.py``). That trade does not carry over to beta, and it is worth being
# precise about why, because the naive port is wrong in an interesting way.
#
# The two weight programs are strictly convex, so "the optimum" is a well-posed
# target and solving it exactly is unambiguously better. The beta block is not.
# Holding the weights at their optima, the objective is very nearly flat in
# beta: on the quota panel below, cohort 1 runs from 21.57 at ``beta = 0`` to a
# minimum of 20.92 near ``beta = 1.05``, and the reference never gets there.
# ``synthdid`` steps beta by ``1/t`` along the gradient, so its total step budget
# grows like ``log(max.iter)``; the iteration hits its cap rather than its
# convergence test, and beta is still drifting when it stops:
#
#     max_iter      10     100    1000   10000  100000
#     beta        0.094   0.182   0.266   0.344   0.414
#
# The stopping rule is therefore load-bearing, not incidental -- it acts as
# shrinkage toward zero on a direction the data barely identify. Solving the
# beta block exactly is a *different estimator*: it gives beta = 8.4 on one
# cohort of this panel and moves the ATT to 8.011, against the 8.051 every
# published ``optimized`` result was produced with. So beta is fitted by the
# reference's own iteration, at its own default cap, and the weights that go
# into the reported estimate are then solved exactly as everywhere else.
# ``sdid_covariate_objective`` is exposed so this is testable rather than
# asserted: ``tests/test_sdid_optimized_covariates.py`` pins both that the
# fitted beta lowers the objective and that it stops short of minimising it.
#
# Reference: ``synthdid::sc.weight.fw.covariates`` (``R/solver.R``); Clarke,
# Pailanir, Athey & Imbens (2024), Stata Journal, Table 1 column 2.
# ---------------------------------------------------------------------------

#: ``synthdid``'s defaults, which fix where the beta iteration stops and so are
#: part of the estimator's definition rather than tuning knobs.
_FW_MAX_ITER = 10_000
_FW_MIN_DECREASE = 1e-5
_ZETA_LAMBDA = 1e-6


def _fw_step(A: np.ndarray, x: np.ndarray, b: np.ndarray,
             eta: float) -> np.ndarray:
    """One Frank-Wolfe step for ``||A x - b||^2 + eta ||x||^2`` on the simplex.

    ``solver.R: fw.step``. The same primitive appears in
    ``propsc_helpers/pipeline.py``, ported there from the same source; it is
    repeated rather than shared because the two estimators are separate
    packages and neither should reach into the other's internals.
    """
    Ax = A @ x
    half_grad = (Ax - b) @ A + eta * x
    i = int(np.argmin(half_grad))
    d_x = -x.copy()
    d_x[i] = 1.0 - x[i]
    if np.all(d_x == 0.0):  # pragma: no cover - already at the optimal vertex
        return x
    d_err = A[:, i] - Ax
    denom = float(np.sum(d_err ** 2) + eta * np.sum(d_x ** 2))
    if denom <= 0.0:  # pragma: no cover - degenerate direction
        return x
    step = -float(half_grad @ d_x) / denom
    return x + min(1.0, max(0.0, step)) * d_x


def _collapse(donors: np.ndarray, treated: np.ndarray,
              pre_periods: int) -> np.ndarray:
    """``(N0+1) x (T0+1)`` collapsed block: donors then the treated mean.

    ``utils.R: collapsed.form``. Post-treatment periods collapse to their mean
    and the treated units to theirs, which is the form both weight programs are
    defined on.
    """
    t0 = int(pre_periods)
    top = np.column_stack([donors[:t0].T, donors[t0:].mean(axis=0)])
    bot = np.append(treated[:t0], treated[t0:].mean())
    return np.vstack([top, bot])


def _validate_block(donor_outcomes, treated_outcome, donor_covariates,
                    treated_covariates, pre_periods):
    """Shape and finiteness checks shared by the objective and the solver."""
    Y0 = np.asarray(donor_outcomes, dtype=float)
    y1 = np.asarray(treated_outcome, dtype=float)
    X0 = np.asarray(donor_covariates, dtype=float)
    x1 = np.asarray(treated_covariates, dtype=float)
    if Y0.ndim != 2 or y1.ndim != 1 or Y0.shape[0] != y1.shape[0]:
        raise MlsynthDataError(
            f"the 'optimized' covariate block needs donor outcomes (T, N0) and "
            f"a treated path (T,); got {Y0.shape} and {y1.shape}.")
    if X0.ndim != 3 or x1.ndim != 2:
        raise MlsynthDataError(
            f"the 'optimized' covariate block needs donor covariates "
            f"(K, T, N0) and treated covariates (K, T); got {X0.shape} and "
            f"{x1.shape}.")
    if X0.shape[1:] != Y0.shape or x1.shape != (X0.shape[0], y1.shape[0]):
        raise MlsynthDataError(
            f"covariate blocks do not line up with the outcome block: outcomes "
            f"{Y0.shape}, donor covariates {X0.shape}, treated covariates "
            f"{x1.shape}.")

    t0 = int(pre_periods)
    n_post = Y0.shape[0] - t0
    if t0 < 2 or n_post < 1:
        raise MlsynthDataError(
            "covariates={'optimized': ...} needs at least two pre-treatment "
            f"and one post-treatment period; got {t0} and {n_post}.")
    if Y0.shape[1] < 1:
        raise MlsynthDataError(
            "covariates={'optimized': ...} needs at least one donor unit.")
    for name, arr in (("outcome", Y0), ("treated outcome", y1),
                      ("covariate", X0), ("treated covariate", x1)):
        if not np.all(np.isfinite(arr)):
            raise MlsynthDataError(
                f"the 'optimized' covariate fit needs a {name} block free of "
                "NaN/inf; check the covariate columns for missing values.")
    return Y0, y1, X0, x1, t0, n_post


def _noise_level(donor_outcomes: np.ndarray, pre_periods: int) -> float:
    """Pooled sd of the donors' pre-period first differences (``ddof=1``).

    ``synthdid`` scales both the ridge and the stopping rule by this, and takes
    it from the *unadjusted* outcome -- so it is a constant of the problem and
    not re-estimated as beta moves.
    """
    diffs = np.diff(np.asarray(donor_outcomes, dtype=float)[:int(pre_periods)],
                    axis=0)
    if diffs.size < 2:  # pragma: no cover - guarded by the pre-period check
        return 0.0
    return float(diffs.std(ddof=1))


def sdid_covariate_objective(
    beta,
    donor_outcomes: np.ndarray,
    treated_outcome: np.ndarray,
    donor_covariates: np.ndarray,
    treated_covariates: np.ndarray,
    pre_periods: int,
    n_treated: int = 1,
) -> float:
    """The value the 'optimized' covariate method descends on, at fixed ``beta``.

    The weights are profiled out at their exact optima, so this is a function of
    ``beta`` alone -- the *reduced* objective. Both blocks carry a free
    intercept, which is what makes the value interpretable: without it ``beta``
    can buy an apparent improvement simply by shifting the level of a covariate,
    and the argmin runs off to wherever that shift is largest.

    Exposed because 'optimized' is defined by a minimisation and the reference
    solver stops well short of the minimum (see the module comment above). With
    the objective in hand a test can state both facts instead of taking the
    fitted coefficient on trust.

    Parameters
    ----------
    beta : array-like
        Shape ``(K,)`` coefficients, one per covariate.
    donor_outcomes : numpy.ndarray
        Shape ``(T, N0)`` donor outcome paths.
    treated_outcome : numpy.ndarray
        Shape ``(T,)`` treated path, averaged over the cohort's treated units.
    donor_covariates, treated_covariates : numpy.ndarray
        Shapes ``(K, T, N0)`` and ``(K, T)``, aligned to the outcome blocks.
    pre_periods : int
        Number of pre-treatment periods ``T0``.
    n_treated : int, default 1
        Treated units in the cohort; sets the ridge scale.

    Returns
    -------
    float
        ``||err_omega||^2 / T0 + ||err_lambda||^2 / N0 + zeta^2 ||omega||^2``.
    """
    from .weights import fit_time_weights, unit_weights

    Y0, y1, X0, x1, t0, n_post = _validate_block(
        donor_outcomes, treated_outcome, donor_covariates,
        treated_covariates, pre_periods)
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    if beta.shape != (X0.shape[0],):
        raise MlsynthDataError(
            f"beta has shape {beta.shape}; expected ({X0.shape[0]},), one "
            "coefficient per covariate.")

    donors = Y0 - np.tensordot(beta, X0, axes=(0, 0))
    treated = y1 - beta @ x1
    zeta = ((int(n_treated) * n_post) ** 0.25) * _noise_level(Y0, t0)

    donor_pre = donors[:t0]
    post_mean = donors[t0:].mean(axis=0)
    a_omega, omega = unit_weights(donor_pre, treated[:t0], zeta)
    a_lambda, lam = fit_time_weights(donor_pre, post_mean)

    err_omega = treated[:t0] - a_omega - donor_pre @ omega
    err_lambda = post_mean - a_lambda - donor_pre.T @ lam
    n0 = donors.shape[1]
    return float(err_omega @ err_omega / t0 + err_lambda @ err_lambda / n0
                 + (zeta ** 2) * (omega @ omega))


def optimized_covariate_beta(
    donor_outcomes: np.ndarray,
    treated_outcome: np.ndarray,
    donor_covariates: np.ndarray,
    treated_covariates: np.ndarray,
    pre_periods: int,
    n_treated: int = 1,
    max_iter: int = _FW_MAX_ITER,
) -> np.ndarray:
    """Covariate coefficients for one SDID block, by the reference iteration.

    Alternates a Frank-Wolfe step on each weight vector with a ``1/t`` gradient
    step on ``beta``, stopping when the objective stops falling by
    ``min_decrease^2`` or at ``max_iter`` -- ``synthdid``'s
    ``sc.weight.fw.covariates``. In practice the cap binds; see the module
    comment for why that is the estimator rather than a defect to fix.

    Parameters
    ----------
    donor_outcomes, treated_outcome, donor_covariates, treated_covariates,
    pre_periods, n_treated
        As in :func:`sdid_covariate_objective`.
    max_iter : int, default 10000
        ``synthdid``'s cap. Changing it changes the estimate.

    Returns
    -------
    numpy.ndarray
        Shape ``(K,)`` coefficients.

    Raises
    ------
    MlsynthDataError
        The blocks do not line up, carry non-finite values, or leave too few
        pre/post periods or donors to fit on.
    """
    Y0, y1, X0, x1, t0, n_post = _validate_block(
        donor_outcomes, treated_outcome, donor_covariates,
        treated_covariates, pre_periods)

    noise = _noise_level(Y0, t0)
    zeta_omega = ((int(n_treated) * n_post) ** 0.25) * noise
    zeta_lambda = _ZETA_LAMBDA * noise
    min_decrease = _FW_MIN_DECREASE * noise

    Yc = _collapse(Y0, y1, t0)
    Xc = np.stack([_collapse(X0[k], x1[k], t0) for k in range(X0.shape[0])])
    n0 = Yc.shape[0] - 1

    lam = np.full(t0, 1.0 / t0)
    omega = np.full(n0, 1.0 / n0)
    beta = np.zeros(X0.shape[0])

    def update(block, lam, omega):
        # Each block is centered before its weight fit, which is exactly how
        # the reference carries the free intercept (``lambda.intercept`` /
        # ``omega.intercept``): centering both sides is the FWL form of it.
        Yl = block[:n0] - block[:n0].mean(axis=0, keepdims=True)
        lam = _fw_step(Yl[:, :t0], lam, Yl[:, t0], n0 * zeta_lambda ** 2)
        err_l = Yl @ np.append(lam, -1.0)
        Yo = block[:, :t0].T
        Yo = Yo - Yo.mean(axis=0, keepdims=True)
        omega = _fw_step(Yo[:, :n0], omega, Yo[:, n0], t0 * zeta_omega ** 2)
        err_o = Yo @ np.append(omega, -1.0)
        value = float(zeta_omega ** 2 * (omega @ omega)
                      + zeta_lambda ** 2 * (lam @ lam)
                      + err_l @ err_l / n0 + err_o @ err_o / t0)
        return value, lam, omega, err_l, err_o

    value, lam, omega, err_l, err_o = update(Yc, lam, omega)
    for t in range(1, int(max_iter) + 1):
        # err_l and err_o are already centered, so contracting the *uncentered*
        # covariate block against them is the same inner product -- which is
        # what the reference does.
        grad = np.empty(X0.shape[0])
        for k in range(X0.shape[0]):
            g_l = err_l @ (Xc[k][:n0] @ np.append(lam, -1.0)) / n0
            g_o = err_o @ (Xc[k][:, :t0].T @ np.append(omega, -1.0)) / t0
            grad[k] = -(g_l + g_o)
        if not np.all(np.isfinite(grad)):  # pragma: no cover - defensive
            break
        beta = beta - grad / t
        block = Yc - np.tensordot(beta, Xc, axes=(0, 0))
        previous = value
        value, lam, omega, err_l, err_o = update(block, lam, omega)
        if t >= 2 and previous - value <= min_decrease ** 2:
            break
    return beta


def optimized_block_adjustment(
    donor_outcomes: np.ndarray,
    treated_outcome: np.ndarray,
    donor_covariates: np.ndarray,
    treated_covariates: np.ndarray,
    pre_periods: int,
    n_treated: int = 1,
):
    """Residualise one block's outcomes by its jointly-fitted coefficients.

    Returns ``(donor_outcomes_adjusted, treated_outcome_adjusted, beta)``. The
    fit is per block because in a staggered design the donor pool, the horizon
    and the ridge all vary by adoption cohort, so a single panel-wide
    coefficient would not be the quantity any cohort's objective defines.
    """
    beta = optimized_covariate_beta(
        donor_outcomes, treated_outcome, donor_covariates,
        treated_covariates, pre_periods, n_treated=n_treated)
    Y0 = np.asarray(donor_outcomes, dtype=float)
    y1 = np.asarray(treated_outcome, dtype=float)
    X0 = np.asarray(donor_covariates, dtype=float)
    x1 = np.asarray(treated_covariates, dtype=float)
    return (Y0 - np.tensordot(beta, X0, axes=(0, 0)),
            y1 - np.tensordot(beta, x1, axes=(0, 0)),
            beta)
