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
