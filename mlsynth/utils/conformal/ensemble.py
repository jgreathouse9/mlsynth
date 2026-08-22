"""Out-of-sample errors from a bootstrap ensemble, at one fit per member.

:mod:`.scores` buys out-of-sample errors by withholding periods: an origin slides
across the pre-period, the estimator refits behind it, and the next stretch is
scored. That costs one refit per origin and yields errors only on the stretches
after the first origin.

This module buys them a second way, following Xu and Xie's EnbPI. Fit ``B``
members, each on a bootstrap sample of the training periods, and score period
``t`` using only the members whose sample excluded it. Since a bootstrap sample
of ``n`` from ``n`` omits about ``1/e`` of the periods, every member scores about
a third of them, and a period goes unscored only when every member happened to
sample it -- probability about ``0.63 ** B``, so a few dozen members leave none.
Every training period then carries an error that the members scoring it never
saw, at ``B`` fits total instead of one per origin.

The model is the caller's. ``weight_fn`` is handed the resampled period indices
and returns donor weights, so a simplex control, a ridge-augmented one and a
lasso all reach the same error series through the same function.

Two properties decide whether the series is usable, and neither is this module's
to assert. The first is what the ensemble was allowed to see: leave-one-out makes
a training period out of sample for the weights, and for nothing that was decided
before the ensemble ran. A design that chose its donor pool or its treated set on
those periods leaves errors there that are out of sample in one respect and not in
the other, and measurably tighter than errors on periods the design never saw at
all. The second is the aggregation: the members are averaged into one prediction,
so the error series describes the ensemble's counterfactual and not any single
member's. A band built from these errors belongs around
:attr:`EnsembleErrors.counterfactual`, not around a separately fitted point
estimate.

References
----------
Xu, C., & Xie, Y. (2021). Conformal prediction interval for dynamic time-series.
Proceedings of the 38th International Conference on Machine Learning, PMLR 139,
11559-11569.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError

#: How members are combined into one prediction. The median is the default: a
#: bootstrap member fitted on a resample that happened to omit an influential
#: period can be far from the rest, and one such member moves a mean.
AGGREGATIONS = {"mean": np.mean, "median": np.median}

#: Members, when the caller does not say. Leaves about ``0.63 ** 200`` of the
#: training periods unscored, which is zero at any panel length.
DEFAULT_MEMBERS = 200


@dataclass(frozen=True)
class EnsembleErrors:
    """Leave-one-out errors on the training periods, and what follows them.

    Parameters
    ----------
    errors : np.ndarray
        Shape ``(n_train,)``. ``target[t]`` minus the aggregate of the members
        that did not sample ``t``. ``nan`` at any period every member sampled,
        which is a statement that no honest error exists there, not a zero.
    counterfactual : np.ndarray
        Shape ``(n_periods - n_train,)``. The aggregate over *every* member on
        the periods after training. No member saw them, so none is excluded.
    n_members : int
        Members fitted.
    n_dropped : int
        Training periods left ``nan``.
    """

    errors: np.ndarray
    counterfactual: np.ndarray
    n_members: int
    n_dropped: int


def _check_members(n_members) -> int:
    if isinstance(n_members, bool) or not isinstance(n_members, (int, np.integer)):
        raise MlsynthConfigError(
            f"n_members must be a positive int; got {n_members!r}.")
    if n_members < 1:
        raise MlsynthConfigError(
            f"n_members must be at least 1; got {n_members}.")
    return int(n_members)


def _check_series(target, donors):
    target = np.asarray(target, dtype=float)
    donors = np.asarray(donors, dtype=float)
    if target.ndim != 1:
        raise MlsynthDataError(
            f"target must be one-dimensional; got shape {target.shape}.")
    if donors.ndim != 2:
        raise MlsynthDataError(
            f"donors must be two-dimensional (n_donors, n_periods); "
            f"got shape {donors.shape}.")
    if donors.shape[1] != target.size:
        raise MlsynthDataError(
            f"target and donors must cover the same periods; target has "
            f"{target.size} periods and donors has {donors.shape[1]}.")
    if not np.isfinite(target).all():
        raise MlsynthDataError("target must be finite; it contains nan or inf.")
    if not np.isfinite(donors).all():
        raise MlsynthDataError("donors must be finite; they contain nan or inf.")
    return target, donors


def _check_train(n_train, n_periods) -> int:
    if isinstance(n_train, bool) or not isinstance(n_train, (int, np.integer)):
        raise MlsynthConfigError(
            f"n_train must be a positive int; got {n_train!r}.")
    if not 1 <= n_train <= n_periods:
        raise MlsynthConfigError(
            f"n_train must be between 1 and the {n_periods} periods available; "
            f"got {n_train}.")
    return int(n_train)


def _check_weights(w, n_donors) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    if w.ndim != 1 or w.size != n_donors:
        raise MlsynthDataError(
            f"weight_fn must return one weight per donor, shape ({n_donors},); "
            f"got shape {w.shape}.")
    if not np.isfinite(w).all():
        raise MlsynthDataError(
            "weight_fn returned weights that are not finite; the fit did not "
            "converge on this resample.")
    return w


def bootstrap_loo_errors(
    target,
    donors,
    n_train,
    weight_fn: Callable[[np.ndarray], np.ndarray],
    *,
    n_members: int = DEFAULT_MEMBERS,
    rng: Optional[np.random.Generator] = None,
    aggregate: str = "median",
) -> EnsembleErrors:
    """Leave-one-out errors from a bootstrap ensemble of donor-weight fits.

    Parameters
    ----------
    target : array-like
        The treated series, shape ``(n_periods,)``.
    donors : array-like
        The donor block, shape ``(n_donors, n_periods)``, on the same periods.
    n_train : int
        How many leading periods the members are fitted on. Periods after these
        are scored by no member and become
        :attr:`EnsembleErrors.counterfactual`.
    weight_fn : callable
        ``weight_fn(idx) -> w``, the caller's fit on the resampled period indices
        ``idx`` (which repeat), returning weights of shape ``(n_donors,)``.
    n_members : int
        Members to fit. Fewer than about a dozen leaves periods unscored.
    rng : numpy.random.Generator, optional
        Source of the resamples. Defaults to a fresh default generator.
    aggregate : {"median", "mean"}
        How members are combined.

    Returns
    -------
    EnsembleErrors

    Raises
    ------
    MlsynthConfigError
        On a bad ``n_train``, ``n_members``, ``aggregate`` or ``weight_fn``.
    MlsynthDataError
        On mismatched or non-finite inputs, or weights of the wrong shape.
    """
    target, donors = _check_series(target, donors)
    n_train = _check_train(n_train, target.size)
    n_members = _check_members(n_members)
    if aggregate not in AGGREGATIONS:
        raise MlsynthConfigError(
            f"aggregate must be one of {sorted(AGGREGATIONS)}; got "
            f"{aggregate!r}.")
    if not callable(weight_fn):
        raise MlsynthConfigError(
            f"weight_fn must be callable, taking period indices and returning "
            f"donor weights; got {type(weight_fn).__name__}.")

    rng = np.random.default_rng() if rng is None else rng
    combine = AGGREGATIONS[aggregate]
    n_donors = donors.shape[0]

    member_fits = np.empty((n_members, n_donors))
    sampled = np.zeros((n_members, n_train), dtype=bool)
    for b in range(n_members):
        idx = rng.integers(0, n_train, size=n_train)
        sampled[b, np.unique(idx)] = True
        member_fits[b] = _check_weights(weight_fn(idx), n_donors)

    # every member's prediction everywhere; the mask decides which ones count
    predictions = member_fits @ donors
    errors = np.full(n_train, np.nan)
    for t in range(n_train):
        excluded = ~sampled[:, t]
        if excluded.any():
            errors[t] = target[t] - combine(predictions[excluded, t])

    return EnsembleErrors(
        errors=errors,
        counterfactual=combine(predictions[:, n_train:], axis=0),
        n_members=n_members,
        n_dropped=int(np.isnan(errors).sum()),
    )
