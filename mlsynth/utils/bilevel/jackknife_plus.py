"""jackknife+ intervals for ridge-augmented SCM (augsynth ``inf_type="jackknife+"``).

Ben-Michael, Feller & Rothstein's ``augsynth`` offers a jackknife-plus interval
that resamples over *pre-treatment periods*, not over units. Each pre-period is
held out in turn, the ASCM is refitted without it, and the interval is built from
how badly the refit predicts the period it could not see. Under the usual
jackknife-plus logic, a period the fit struggles to predict is evidence that the
post-treatment prediction is also uncertain.

Worth separating from two neighbours that sound like it:

* it is not the Barber, Candes, Ramdas & Tibshirani jackknife-plus over
  observations, which leaves out data *rows*;
* it is not the delete-one-*unit* jackknife in
  :mod:`mlsynth.utils.ppscm_helpers.inference`, which serves the partially
  pooled estimator of the authors' 2022 paper.

Two details are load-bearing and both are easy to get wrong.

The penalty is pinned. ``augsynth.R`` stores the fitted ridge penalty into
``extra_args`` before any refit (``extra_args$lambda <- augsynth$lambda``), so
every held-out refit reuses the *original* fit's lambda and none of them re-runs
cross-validation. Re-selecting per drop would give one penalty per period and a
different interval -- and would cost an extra CV sweep per refit.

The interval is reflected, not copied. The quantiles are taken on the
counterfactual scale and then subtracted from the observed treated path, which
*reverses* them: the ATT lower bound comes from the counterfactual upper bound.
Both scales are returned so a caller can see this rather than infer it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ...exceptions import MlsynthEstimationError
from .ridge_augment import ridge_augment_weights


def _quantile(x: np.ndarray, p: float) -> float:
    """R's ``stats::quantile`` default (type 7).

    numpy's ``linear`` method is type 7, so this is a thin wrapper -- but an
    explicit one, because picking a different type shifts every bound by a
    fraction of an order statistic: small, systematic, and undetectable without
    a reference to compare against.
    """
    return float(np.quantile(np.asarray(x, dtype=float), p, method="linear"))


@dataclass(frozen=True)
class JackknifePlusResult:
    """Outcome of :func:`jackknife_plus`.

    Attributes
    ----------
    lower, upper : np.ndarray, shape (T1 + 1,)
        ATT-scale bounds, one per post-treatment period followed by the bound
        for the post-period average. Pre-treatment periods have no bound under
        this procedure; callers that need a full-length path should pad with
        ``nan``, as augsynth's own output does.
    counterfactual_lower, counterfactual_upper : np.ndarray, shape (T1 + 1,)
        The same bounds before reflection onto the ATT scale.
    held_out_errors : np.ndarray, shape (T0,)
        Per-drop error at the held-out period: observed treated outcome minus
        the refit's prediction for it.
    per_drop_estimates : np.ndarray, shape (T0, T1 + 1)
        Each refit's counterfactual over the post periods, with its mean
        appended.
    lambda_ : float
        The ridge penalty, selected once on the full pre-period and reused by
        every refit.
    alpha : float
    conservative : bool
    """

    lower: np.ndarray
    upper: np.ndarray
    counterfactual_lower: np.ndarray
    counterfactual_upper: np.ndarray
    held_out_errors: np.ndarray
    per_drop_estimates: np.ndarray
    lambda_: float
    alpha: float
    conservative: bool


def jackknife_plus(
    y_pre: np.ndarray,
    Y0_pre: np.ndarray,
    y_post: np.ndarray,
    Y0_post: np.ndarray,
    *,
    alpha: float = 0.05,
    conservative: bool = False,
    Z0: Optional[np.ndarray] = None,
    z1: Optional[np.ndarray] = None,
    residualize: bool = False,
    lambda_: Optional[float] = None,
    **ridge_kwargs: Any,
) -> JackknifePlusResult:
    """augsynth's ``time_jackknife_plus`` for a ridge-augmented SCM.

    Parameters
    ----------
    y_pre : np.ndarray, shape (T0,)
        Treated pre-treatment outcomes.
    Y0_pre : np.ndarray, shape (T0, J)
        Donor pre-treatment outcomes.
    y_post : np.ndarray, shape (T1,)
        Treated post-treatment outcomes.
    Y0_post : np.ndarray, shape (J, T1)
        Donor post-treatment outcomes.
    alpha : float, default 0.05
        Two-sided level; the interval is ``1 - alpha``.
    conservative : bool, default False
        Use augsynth's conservative branch, which brackets the per-drop
        predictions by their min and max and widens by a quantile of the
        absolute held-out errors, rather than taking quantiles of
        ``est +/- |err|``. augsynth's own default is ``False``.
    Z0, z1, residualize
        Auxiliary covariates, passed through to
        :func:`~mlsynth.utils.bilevel.ridge_augment.ridge_augment_weights`.
        Covariates are unit-level pre-period summaries and so are unaffected by
        holding a period out -- augsynth passes them through untouched too.
    lambda_ : float, optional
        Ridge penalty. When ``None`` it is selected once by CV on the full
        pre-period and then reused by every refit, which is what augsynth does.

    Raises
    ------
    MlsynthEstimationError
        On a degenerate panel or an out-of-range level.
    """
    y_pre = np.asarray(y_pre, dtype=float).ravel()
    Y0_pre = np.asarray(Y0_pre, dtype=float)
    y_post = np.asarray(y_post, dtype=float).ravel()
    Y0_post = np.asarray(Y0_post, dtype=float)

    if not 0.0 < float(alpha) < 1.0:
        raise MlsynthEstimationError(
            f"jackknife+: alpha must lie strictly between 0 and 1; got {alpha}."
        )
    n_pre = y_pre.size
    if n_pre < 3:
        raise MlsynthEstimationError(
            "jackknife+ holds out one pre-treatment period at a time and "
            f"refits on the rest, so it needs at least 3; got {n_pre}."
        )
    if Y0_pre.ndim != 2 or Y0_pre.shape[0] != n_pre:
        raise MlsynthEstimationError(
            f"jackknife+: Y0_pre must be (T0, J) with T0 = {n_pre}; got "
            f"{Y0_pre.shape}."
        )
    n_donors = Y0_pre.shape[1]
    if Y0_post.ndim != 2 or Y0_post.shape[0] != n_donors:
        raise MlsynthEstimationError(
            f"jackknife+: Y0_post must be (J, T1) with the same {n_donors} "
            f"donor(s) as Y0_pre; got {Y0_post.shape}."
        )
    if y_post.size == 0 or Y0_post.shape[1] == 0:
        raise MlsynthEstimationError(
            "jackknife+ needs at least one post-treatment period to bound."
        )
    if y_post.size != Y0_post.shape[1]:
        raise MlsynthEstimationError(
            f"jackknife+: y_post has {y_post.size} period(s) but Y0_post has "
            f"{Y0_post.shape[1]}."
        )

    fit_kwargs: Dict[str, Any] = dict(
        Z0=Z0, z1=z1, residualize=residualize, **ridge_kwargs)

    # Select the penalty once, on the full pre-period, and pin it. This mirrors
    # augsynth stashing `lambda` into `extra_args` before it refits.
    if lambda_ is None:
        lambda_ = float(ridge_augment_weights(y_pre, Y0_pre, **fit_kwargs).lambda_)

    n_post = y_post.size
    per_drop = np.empty((n_pre, n_post + 1), dtype=float)
    held_out = np.empty(n_pre, dtype=float)

    for j in range(n_pre):
        keep = np.arange(n_pre) != j
        w = ridge_augment_weights(
            y_pre[keep], Y0_pre[keep], lambda_=lambda_, **fit_kwargs).W
        # The refit never saw period j, so its prediction there is out-of-sample.
        held_out[j] = y_pre[j] - float(Y0_pre[j] @ w)
        est = Y0_post.T @ w
        per_drop[j, :n_post] = est
        per_drop[j, n_post] = est.mean()

    abs_err = np.abs(held_out)[:, None]
    if conservative:
        q = _quantile(np.abs(held_out), 1.0 - alpha)
        cf_lower = per_drop.min(axis=0) - q
        cf_upper = per_drop.max(axis=0) + q
    else:
        lo = per_drop - abs_err
        hi = per_drop + abs_err
        cf_lower = np.array([_quantile(lo[:, k], alpha / 2.0)
                             for k in range(n_post + 1)])
        cf_upper = np.array([_quantile(hi[:, k], 1.0 - alpha / 2.0)
                             for k in range(n_post + 1)])

    # Reflect onto the ATT scale. Subtracting from the observed path reverses
    # the bounds, so the ATT lower bound is built from the counterfactual upper.
    y1 = np.append(y_post, y_post.mean())
    return JackknifePlusResult(
        lower=y1 - cf_upper,
        upper=y1 - cf_lower,
        counterfactual_lower=cf_lower,
        counterfactual_upper=cf_upper,
        held_out_errors=held_out,
        per_drop_estimates=per_drop,
        lambda_=float(lambda_),
        alpha=float(alpha),
        conservative=bool(conservative),
    )
